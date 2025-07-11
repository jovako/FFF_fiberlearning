# train_ct_ddpm.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
from accelerate import Accelerator
from pathlib import Path
from fff.data import load_dataset


# ema.py
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


def train_ct_ddpm(
    dataset_config,
    image_size=224,
    batch_size=8,
    num_epochs=20,
    learning_rate=1e-4,
    save_dir="./saved_models",
    num_train_timesteps=1000,
):
    accelerator = Accelerator()
    device = accelerator.device

    # Load dataset
    train_ds, _, _ = load_dataset(**dataset_config)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Define UNet
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=("DownBlock2D",) * 4,
        up_block_types=("UpBlock2D",) * 4,
    )

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps, beta_schedule="linear"
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0
    )
    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    model.train()
    ema = EMA(model, decay=0.999)

    for epoch in range(num_epochs):
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True
        )
        for batch in pbar:
            # Get image: [B, H, W] or [B, 1, H, W]
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
            accelerator.backward(loss)
            # gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr_scheduler.step()
            ema.update(model)
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

        # Save model after each epoch
        if accelerator.is_main_process:
            ema.apply_shadow(model)
            # Now sample or save model here
            model.save_pretrained(Path(save_dir) / f"epoch_{epoch+1}")
            ema.restore(model)

    print("✅ Training complete.")


if __name__ == "__main__":
    data_set_config = {
        "name": "ldct",
        "root": "/home/hd/hd_hd/hd_gu452/ldct_data",
        "condition": "lowdose",
        "data": "highdose",
        "patchsize": 512,
        "resize_to": 224,
        "augment": True,
        "data_norm": "meanstd",
    }
    train_ct_ddpm(data_set_config)
