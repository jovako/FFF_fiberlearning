import torch
from torch import nn
from diffusers import UNet2DModel
from fff.base import ModelHParams
from typing import Tuple
from fff.model.utils import guess_image_shape
from collections import OrderedDict
from math import prod


def make_unet(
    sample_size: int | Tuple[int, int],
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    block_out_channels: Tuple[int, ...],
    down_block_types: Tuple[str, ...],
    up_block_types: Tuple[str, ...],
) -> UNet2DModel:

    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )


class UNetHParams(ModelHParams):
    layers_per_block: int = 3
    block_out_channels: tuple = (64, 128, 128, 256)
    down_block_types: tuple = ("DownBlock2D",) * 4
    up_block_types: tuple = ("UpBlock2D",) * 4
    image_shape: None | list[int] = None


class UNet(nn.Module):
    hparams: UNetHParams

    def __init__(self, hparams: dict | UNetHParams):
        if not isinstance(hparams, UNetHParams):
            hparams = UNetHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        if self.hparams.image_shape is not None:
            self.input_shape = torch.Size(self.hparams.image_shape)
        else:
            self.input_shape = guess_image_shape(self.hparams.data_dim)
        self.model = self.build_model()

    def cat_x_c(self, x, c):
        # Reshape as image, and concatenate conditioning as channel dimensions
        has_batch_dimension = len(x.shape) > 1
        if not has_batch_dimension:
            x = x[None, :]
            c = c[None, :]
        batch_size = x.shape[0]

        x_img = x.reshape(batch_size, *self.input_shape)
        if len(c.shape) == 2:
            c_img = c[:, :, None, None] * torch.ones(
                batch_size,
                self.hparams.cond_dim,
                *self.input_shape[1:],
                device=c.device,
            )
        else:
            if c.shape[2:] != self.input_shape[1:]:
                # Resize conditioning to match image shape
                c_img = nn.functional.interpolate(
                    c, size=self.input_shape[1:], mode="bilinear", align_corners=False
                )
            else:
                c_img = c
        if c_img.shape[1] != self.hparams.cond_dim:
            raise ValueError(
                f"Conditioning shape {c_img.shape} does not match expected shape {self.hparams.cond_dim}"
            )
        out = torch.cat([x_img, c_img], -3).reshape(batch_size, -1)
        if not has_batch_dimension:
            out = out[0]
        return out

    def build_model(self):
        cond_dim = self.hparams.cond_dim
        latent_channels = int(self.hparams.latent_dim / prod(self.input_shape[1:]))

        encoder = make_unet(
            sample_size=self.input_shape[1:],
            in_channels=self.input_shape[0] + cond_dim,
            out_channels=latent_channels,
            layers_per_block=self.hparams.layers_per_block,
            block_out_channels=self.hparams.block_out_channels,
            down_block_types=self.hparams.down_block_types,
            up_block_types=self.hparams.up_block_types,
        )

        decoder = make_unet(
            sample_size=self.input_shape[1:],
            in_channels=latent_channels + cond_dim,
            out_channels=self.input_shape[0],
            layers_per_block=self.hparams.layers_per_block,
            block_out_channels=self.hparams.block_out_channels,
            down_block_types=self.hparams.down_block_types,
            up_block_types=self.hparams.up_block_types,
        )

        return nn.Sequential(OrderedDict(encoder=encoder, decoder=decoder))

    def encode(self, x, c, t):
        return self.model.encoder(self.cat_x_c(x, c), t)

    def decode(self, u, c, t):
        return self.model.decoder(torch.cat([u, c.flatten(1)], -1), t)

    def forward(self, x, c, t):
        z = self.encode(x, c, t)
        return self.decode(z, c, t)

    def sample(self, u, c, t):
        return self.decode(u, c, t)
