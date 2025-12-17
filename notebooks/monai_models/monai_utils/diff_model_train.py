# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import monai
import torch
import torch.distributed as dist
from monai.data import DataLoader, partition_dataset
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance, ResizeMaskToImage
import sys

def safe_allreduce(tensor, op=dist.ReduceOp.SUM, name="tensor"):
    """
    Ensure `tensor` is a CUDA tensor with the same numel on all ranks,
    and perform dist.all_reduce. Raises an informative error before calling NCCL.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Convert python lists / numpy to torch tensor on correct device
    if not isinstance(tensor, torch.Tensor):
        # try to create tensor on current device (should be GPU when using NCCL)
        device = torch.device("cuda", torch.cuda.current_device())
        tensor = torch.tensor(tensor, device=device)

    # move to CUDA (NCCL requires CUDA tensors)
    if not tensor.is_cuda:
        try:
            tensor = tensor.to(torch.device("cuda", torch.cuda.current_device()))
        except Exception as e:
            raise RuntimeError(f"Rank {rank}: failed to move {name} to CUDA: {e}")

    # make contiguous
    tensor = tensor.contiguous()

    # Broadcast local metadata (numel, dtype) via a small CPU tensor to assert equality
    local_meta = torch.tensor([tensor.numel(), int(tensor.is_cuda)], dtype=tensor.dtype, device="cpu")
    meta_list = [torch.zeros_like(local_meta) for _ in range(world_size)]
    # gather: use all_gather to collect metas from all ranks
    dist.all_gather(meta_list, local_meta)
    metas = [m.tolist() for m in meta_list]

    for r, meta in enumerate(metas):
        numel, dtype_num, is_cuda_flag = meta
        if numel != tensor.numel():
            raise RuntimeError(
                f"Mismatch before all_reduce: rank {rank} has numel={tensor.numel()}, "
                f"but rank {r} reports numel={numel}."
            )
        if dtype_num != tensor.dtype.num:
            raise RuntimeError(
                f"Mismatch before all_reduce: rank {rank} dtype={tensor.dtype} != rank {r} dtype_num={dtype_num}"
            )
        if is_cuda_flag == 0:
            raise RuntimeError(f"Rank {r} reports tensor is not CUDA; NCCL require CUDA tensors.")

    # optional barrier so all ranks reach here (helpful for debugging)
    dist.barrier()

    # finally do the all_reduce on the verified tensor
    dist.all_reduce(tensor, op=op)

    return tensor


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]

    #filenames_train = [{
    #"image": "ds000001/sub-02/anat/sub-02_T1w.nii.gz",
    #}]

    #print(f"Training only with {filenames_train}")
    return [
        _item["image"].replace(".nii.gz", "_emb.nii.gz") for _item in filenames_train
    ]


def prepare_data(
    train_files: list,
    device: torch.device,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    include_body_region: bool = False,
) -> DataLoader:

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    train_transforms_list = [
        monai.transforms.LoadImaged(keys=["image", "mask"]),
        monai.transforms.EnsureChannelFirstd(keys=["image", "mask"]),
        monai.transforms.Orientationd(keys=["image", "mask"], axcodes="RAS"),
        monai.transforms.EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
        # monai.transforms.ScaleIntensityRanged(keys=["mask"], a_min=0, a_max=1, b_min=-1, b_max=1, clip=True),
        # Resize mask to match image spatial size
	ResizeMaskToImage(keys=("image", "mask")),
        # Concatenate mask to image
        monai.transforms.Lambdad(
            keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")
        ),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
    ]

    if include_body_region:
        train_transforms_list += [
            monai.transforms.Lambdad(
                keys="top_region_index",
                func=lambda x: _load_data_from_file(x, "top_region_index"),
            ),
            monai.transforms.Lambdad(
                keys="bottom_region_index",
                func=lambda x: _load_data_from_file(x, "bottom_region_index"),
            ),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(
                keys="bottom_region_index", func=lambda x: x * 1e2
            ),
        ]

    train_transforms = Compose(train_transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    return DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model if wrapped in DistributedDataParallel, else return itself.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def load_unet(
    args: argparse.Namespace, device: torch.device, logger: logging.Logger
) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        unet = DistributedDataParallel(
            unet, device_ids=[device], find_unused_parameters=True
        )

    if args.existing_ckpt_filepath is None:
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(
            f"{args.existing_ckpt_filepath}", map_location=device, weights_only=False
        )
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def calculate_scale_factor(
    train_loader: DataLoader, device: torch.device, logger: logging.Logger
) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt: torch.nn.L1Loss,
    scaler: GradScaler,
    scale_factor: torch.Tensor,
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    amp: bool = True,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    raw_unet = unwrap_model(unet)
    include_body_region = raw_unet.include_top_region_index_input
    include_modality = raw_unet.num_class_embeds is not None

    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    unet.train()
    pbar = tqdm(train_loader, leave=False, total=len(train_loader))
    for train_data in pbar:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data["image"].to(device)
        images = images * scale_factor
        mask = train_data["mask"].to(device)  # deface mask
        if include_body_region:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
        # We trained with only CT in this version
        if include_modality:
            modality_tensor = torch.ones((len(images),), dtype=torch.long).to(device)
        spacing_tensor = train_data["spacing"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=amp):
            noise = torch.randn_like(images)

            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(images)
            else:
                timesteps = torch.randint(
                    0, num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

            noisy_latent = noise_scheduler.add_noise(
                original_samples=images, noise=noise, timesteps=timesteps
            )
            # Create a dictionary to store the inputs
            unet_inputs = {
                "x": torch.cat([noisy_latent, mask], dim=1),
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
            }
            # Add extra arguments if include_body_region is True
            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )
            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )
            model_output = unet(**unet_inputs)

            if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                # predict noise
                model_gt = noise
            elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                # predict sample
                model_gt = images
            elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                # predict velocity
                model_gt = images - noise
            else:
                raise ValueError(
                    "noise scheduler prediction type has to be chosen from ",
                    f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                )

            loss = loss_pt(model_output.float(), model_gt.float())

        if loss.isnan():
            logger.error(f"Loss is nan in epoch {epoch}")
            continue
        if amp:
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            pbar.set_description(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19],
                    epoch + 1,
                    _iter,
                    len(train_loader),
                    loss.item(),
                    current_lr,
                )
            )

    if dist.is_initialized():
        safe_allreduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: torch.Tensor,
    ckpt_folder: str,
    args: argparse.Namespace,
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unwrap_model(unet).state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor,
            "unet_state_dict": unet_state_dict,
        },
        f"{ckpt_folder}/{args.model_filename}",
    )


def diff_model_train(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_gpus: int,
    amp: bool = True,
) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        amp (bool): Use automatic mixed precision training.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")

    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.json_data_list}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(
            f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}."
        )

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    raw_unet = unwrap_model(unet)
    include_body_region = raw_unet.include_top_region_index_input

    filenames_train = load_filenames(args.json_data_list)
    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")

    train_files = []
    for _i in range(len(filenames_train)):
        str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
        if not os.path.exists(str_img):
            continue

        str_info = os.path.join(args.embedding_base_dir, filenames_train[_i]) + ".json"
        

        # The original JSON data list has deface_mask paths (relative to data root)
        with open(str_info, "r") as f:
            json_data = json.load(f)
        deface_mask_path = json_data["mask_path"]

        train_files_i = {
            "image": str_img,
            "mask": deface_mask_path,
            "spacing": str_info,
        }
        if include_body_region:
            train_files_i["top_region_index"] = str_info
            train_files_i["bottom_region_index"] = str_info

        train_files.append(train_files_i)

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[local_rank]

    train_loader = prepare_data(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"],
        include_body_region=include_body_region,
    )

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (
        args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)
    ) / args.diffusion_unet_train["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")

    for epoch in range(args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")

            save_checkpoint(
                epoch,
                unet,
                loss_torch_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args,
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision training",
    )

    args = parser.parse_args()
    diff_model_train(
        args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp
    )
