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
from pathlib import Path

import monai
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.transforms import Compose
from monai.utils import set_determinism
from math import prod


from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance
from tqdm.auto import tqdm

# Set the random seed for reproducibility
set_determinism(seed=0)


def create_transforms(dim: tuple = None) -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    if dim:
        return Compose(
            [
                monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
                monai.transforms.ScaleIntensityRanged(
                    keys="image", a_min=0, a_max=1258, b_min=0, b_max=1, clip=True
                ),
                monai.transforms.Resized(
                    keys="image", spatial_size=dim, mode="trilinear"
                ),
            ]
        )
    else:
        return Compose(
            [
                monai.transforms.LoadImaged(keys="image"),
                monai.transforms.EnsureChannelFirstd(keys="image"),
                monai.transforms.Orientationd(keys="image", axcodes="RAS"),
            ]
        )


def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(
        base_number
    )
    return int(new_number)


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
    filenames_raw = json_data["training"]
    return [_item["image"] for _item in filenames_raw]


def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    preloaded: dict,  # <- received from caller
    new_transforms: Compose,
    logger: logging.Logger,
) -> None:
    out_filename_base = filepath.replace(".gz", "").replace(".nii", "")
    out_filename_base = os.path.join(args.embedding_base_dir, out_filename_base)
    out_filename = out_filename_base + "_emb.nii.gz"
    if os.path.isfile(out_filename):
        return

    # reuse already loaded/oriented image & meta
    nda = preloaded["image"]
    dim = [int(nda.meta["dim"][i]) for i in range(1, 4)]
    spacing = [float(nda.meta["pixdim"][i]) for i in range(1, 4)]
    logger.info(f"old dim: {dim}, old spacing: {spacing}")

    # IMPORTANT: don't re-LoadImaged; transform from dict with same key
    new_data = new_transforms({"image": nda})
    nda_image = new_data["image"]

    new_affine = nda_image.meta["affine"].numpy()
    nda_image = nda_image.numpy().squeeze()
    logger.info(f"new dim: {nda_image.shape}, new affine: {new_affine}")

    try:
        out_path = Path(out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"out_filename: {out_filename}")

        # Ensure no graph is built and mixed precision is scoped
        with torch.inference_mode(), torch.amp.autocast("cuda"):
            pt_nda = (
                torch.from_numpy(nda_image)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            logger.info(f"z: {z.size()}, {z.dtype}")

            out_nda = z.squeeze().to("cpu").numpy().transpose(1, 2, 3, 0)

        # free GPU caches for varying input sizes (prevents allocator growth)
        out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)
        nib.save(out_img, out_filename)
        del pt_nda, z
        if device.type == "cuda":
            torch.cuda.empty_cache()
        del out_img, out_nda, nda_image, new_data
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")
        try:
            del pt_nda, z
            if device.type == "cuda":
                torch.cuda.empty_cache()
            del out_img, out_nda, nda_image, new_data
        except:
            pass


@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(
            args.trained_autoencoder_path, weights_only=True
        )
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    filenames_raw = load_filenames(args.json_data_list)
    # logger.info(f"filenames_raw: {filenames_raw}")

    plain_transforms = create_transforms(dim=None)

    for _iter in tqdm(range(len(filenames_raw)), desc="Embedding images..."):
        if _iter % world_size != local_rank:
            continue
        try:
            filepath = filenames_raw[_iter]
            test_data = {"image": os.path.join(args.data_base_dir, filepath)}
            td = plain_transforms(test_data)  # load once
            img = td["image"]
            dims = [int(img.meta["dim"][i]) for i in range(1, 4)]
            new_dim = tuple(round_number(d) for d in dims)
            if prod(new_dim) > 2e7:
                new_dim = tuple(d//2 for d in new_dim)
            new_transforms = create_transforms(new_dim)
            process_file(filepath, args, autoencoder, device, td, new_transforms, logger)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {e}\nContinuing...")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Model Training Data Creation"
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed training",
    )

    args = parser.parse_args()
    diff_model_create_training_data(
        args.env_config, args.model_config, args.model_def, args.num_gpus
    )
