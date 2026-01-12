import lightning_trainable
import fff
from fff.subject_model import SubjectModel
from fff.data import load_dataset
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from safetensors.torch import load_file
import sys
sys.path.append("/home/hd/hd_hd/hd_gu452/FFF_fiberlearning/scripts/")
from fff.ndtm import NDTMConfig, NDTMTimestepCompatability, DiffusionScheduleConfig, StableDiffusionInterface, DiffusionSchedule, DiffusionModel, NDTM, get_gamma_t_fct
sys.path.append("/home/hd/hd_hd/hd_gu452/oc-guidance/")
from utils.functions import get_timesteps
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
from datetime import datetime
import random
import h5py
from diffusers import UNet2DModel
import argparse

class StableDiffusionInterface(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = UNet2DModel(
            sample_size=28,           # the target image size. For MNIST it's 28
            in_channels=3,            # colored images
            out_channels=3,           # predict noise with 3 channels
            layers_per_block=2,
            block_out_channels=(64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        ).to(device)
        base_model_ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(base_model_ckpt["model_state_dict"])
        self.model.eval()

    def forward(self, x, t, y=None):
        # Assuming y is not used in this case
        noise_pred = self.model(x, t).sample
        return noise_pred

batch_size = 500
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_subject_model(name):
    log_folder = "/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/color_logs"
    try:
        checkpoint = lightning_trainable.utils.find_checkpoint(root=f"{log_folder}/{name}", version=0, epoch="best")
    except:
        checkpoint = lightning_trainable.utils.find_checkpoint(root=f"{log_folder}/{name}", version=0, epoch="last")
    ckpt = torch.load(checkpoint, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    hparams["cond_dim"] = 0
    hparams["data_set"]["root"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist'
    hparams["data_set"]["subject_model_path"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist/subject_model/checkpoints/299_fixed.ckpt'
    hparams["lossless_ae"] = {"model_spec": hparams["lossless_ae"]}
    hparams["load_lossless_ae_path"] = '/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/color_logs/Lossless_VAE/checkpoints/last_fixed.ckpt'
    model = fff.fiber_model.FiberModel(hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model.subject_model

class ColorMNISTSubjectModel(nn.Module):
    def __init__(self, model_name='250_nf'):
        super().__init__()
        self.model = load_subject_model(model_name)
        self.model.eval()  # Set to eval mode

    def forward(self, x):
        x = denormalize(x)
        return self.model.encode(x.permute(0, 1, 3, 2), torch.empty(torch.Size([x.shape[0], 0])))

    def decode(self, y):
        raise NotImplementedError("Subject Model does not support decoding.")

def normalize(x):
    return x*2 - 1

def denormalize(x):
    return (x + 1)/2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1.0,
                    help="terminal loss strength")
    args = parser.parse_args()
    timestep_config = NDTMTimestepCompatability()
    
    diffusion_schedule_config = DiffusionScheduleConfig()
    NDTM_config = NDTMConfig(N=6, 
                             gamma_t = get_gamma_t_fct([(0, 0, 1000, 500), (0, args.gamma, 500, 300), (args.gamma, args.gamma, 300, 0)], max_timesteps=1000),
                             u_lr=0.002, 
                             w_terminal=1.0,
                             eta=0.25,
                             u_lr_scheduler="linear",
                             w_score_scheme="zero",
                             w_control_scheme=3.e-4,
                             clip_images=True,
                             clip_range=[-1, 1],
                             compute_target_per_timestep=False,
                             ancestral_sampling=False,
                             variance_type="large")

    
    with h5py.File('/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-colorMNIST/color_mnist_eval/cc_mnist/data.h5', 'r') as f:
        val_data = torch.from_numpy(f['test_images'][:])
        val_data = normalize(val_data)

    # Diffusion Model
    generative_model_chkpt_path = "/home/hd/hd_hd/hd_gu452/FFF_fiberlearning/notebooks/color_mnist_outputs/checkpoints/ckpt_epoch_correlated_dist_250.pt"
    base_model = StableDiffusionInterface(generative_model_chkpt_path)
    diffusion_schedule = DiffusionSchedule(diffusion_schedule_config)
    generative_model = DiffusionModel(base_model, diffusion_schedule, class_cond_diffusion_model=False)

    # Subject Models
    subject_model = ColorMNISTSubjectModel().to(device)

    # Dataloader
    test_set = torch.utils.data.TensorDataset(val_data) # test_image.permute(0, 1, 3, 2))
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    #Guidance
    ndtm = NDTM(
        generative_model=generative_model,
        subject_model=subject_model,
        hparams=NDTM_config
    )

    # Start to sample invariances
    invariances = []
    originals = []    
    invariances_embeddings = []
    original_embeddings = []
    
    start_time = datetime.now().strftime('%H_%M_%S__%d_%m_%Y')
    random_suffix = start_time + "_" + str(random.getrandbits(16))
    filename = f"sampled_colormnist_invariances_gamma={args.gamma}_{random_suffix}.pt"
    
    if os.path.exists(filename):
        raise(RuntimeError("Incredible..."))
    
    for n_batch, batch in enumerate(dataloader):
        x = batch[0].to(device)
        with torch.no_grad():
            test_image_embedding = subject_model(x)
        originals.append(x)
        original_embeddings.append(test_image_embedding)
        
        ts = get_timesteps(NDTMTimestepCompatability())
        imgs_noised, imgs_approximated = ndtm.sample(x, None, ts, y_0 = test_image_embedding.to(device))
        invariances.append(imgs_noised[0])
        with torch.no_grad():
            invariances_embeddings.append(subject_model(imgs_noised[0].to(device)))

        torch.save({
            "invariances": torch.cat(invariances, dim=0),
            "originals": torch.cat(originals, dim=0),
            "invariances_embeddings": torch.cat(invariances_embeddings, dim=0),
            "original_embeddings": torch.cat(original_embeddings, dim=0),
        }, filename)





