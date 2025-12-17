#!/usr/bin/env python
# coding: utf-8

# # Training

# In[1]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path
from fff.data import load_dataset
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        # Register parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """Apply EMA weights to model."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original (non-EMA) weights."""
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name])
        self.original = {}


# In[3]:


def train_ct_ddpm(
    dataset_config,
    image_size=384,
    batch_size=16,
    num_epochs=5,
    learning_rate=2e-4,
    save_dir="./diffusion-chexpert",
    num_train_timesteps=1000,
):
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # Load dataset
    train_ds, _, _ = load_dataset(**dataset_config)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    # Define UNet
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,             # two residual blocks per level
        block_out_channels=(64, 128, 128, 256),  # channel widths
        down_block_types=(
            "DownBlock2D",      # 384 -> 192
            "DownBlock2D",      # 192 -> 96
            "DownBlock2D",      # 96 -> 48
            "AttnDownBlock2D",  # 48 -> 24 (adds attention at bottleneck)
        ),
        up_block_types=(
            "AttnUpBlock2D",    # 24 -> 48
            "UpBlock2D",        # 48 -> 96
            "UpBlock2D",        # 96 -> 192
            "UpBlock2D",        # 192 -> 384
        ),
        norm_num_groups=32,
    )

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler(clip_sample=False, thresholding=False) 

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )
    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model.train()
    model
    ema = EMA(model, decay=0.999)

    loss_history = []
    for epoch in range(num_epochs):
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True
        )
        for batch in pbar:
            # Get image: [B, 3, H, W]
            clean_images = (
                batch[0].reshape(batch_size, 1, image_size, image_size).to(device)
            )

            # Sample noise
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=clean_images.device,
            ).long()

            # Add noise to image
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


            # Predict the noise using UNet
            noise_pred = model(noisy_images, timesteps).sample

            # Loss
            loss = nn.MSELoss()(noise_pred, noise)
            loss_history.append(loss.item())
            accelerator.backward(loss)
            # gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            ema.update(model)
            optimizer.zero_grad()

            pbar.set_postfix(loss=np.mean(loss_history[-100:]))

        # Save model after each epoch
        if accelerator.is_main_process:
            ema.apply_shadow(model)
            # Now sample or save model here
            model.save_pretrained(Path(save_dir) / f"epoch_{epoch+1}")
            ema.restore(model)

    print("✅ Training complete.")


# In[4]:


dataset_config = {
    "name": "chexpert",
    "root": "/home/hd/hd_hd/hd_gu452/workspaces/gpfs/hd_gu452-chexpert",
    "patchsize": None,
    "resize_to": 384,
    "to_grayscale": True,
}
train_ct_ddpm(dataset_config)
