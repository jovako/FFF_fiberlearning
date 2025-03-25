import math
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
import torch.nn.functional as F
from fff.model.utils import guess_image_shape
from fff.base import ModelHParams

from ldctinv.vae.network import ResnetEncoder, ClassUp
from ldctinv.vae.blocks import (
    ActNorm,
    GBlock,
    SelfAttention,
    SpectralNorm,
)


class LDCTInvHParams(ModelHParams):
    encoder_norm: str = "bn"
    decoder_norm: str = "an"
    encoder_type: str = "resnet50"
    encoder_pretrained: bool = False
    ch_factor: int = 96
    image_shape: list[int] | None = None
    latent_image: bool = False


class LDCTInvModel(nn.Module):
    def __init__(self, hparams: dict | LDCTInvHParams):
        super().__init__()
        if not isinstance(hparams, LDCTInvHParams):
            hparams = LDCTInvHParams(**hparams)
        self.hparams = hparams

        # Translate FFF hparams to ldctinv hparams
        cond_ch = hparams.cond_dim
        if hparams.image_shape is not None:
            input_shape = torch.Size(hparams.image_shape)
        else:
            input_shape = guess_image_shape(hparams.data_dim)
        in_ch = input_shape[0]
        input_size = input_shape[1]
        z_dim = hparams.latent_dim

        self.encoder = ResnetEncoder(
            in_ch + cond_ch,
            z_dim // 2,
            input_size,
            hparams.encoder_norm,
            hparams.encoder_type,
            hparams.encoder_pretrained,
            hparams.latent_image,
        )
        self.decoder = BigGANDecoderAnySize(
            in_ch,
            hparams.ch_factor,
            cond_ch,
            z_dim,
            input_size,
            hparams.decoder_norm,
            hparams.latent_image,
        )

    def cat_x_c(self, x, c):
        # Reshape as image, and concatenate conditioning as channel dimensions
        has_batch_dimension = len(x.shape) > 1
        if not has_batch_dimension:
            x = x[None, :]
            c = c[None, :]
        batch_size = x.shape[0]
        if self.hparams.image_shape is not None:
            input_shape = torch.Size(self.hparams.image_shape)
        else:
            input_shape = guess_image_shape(self.hparams.data_dim)
        x_img = x.reshape(batch_size, *input_shape)
        assert c.shape[1] == self.hparams.cond_dim, (
            f"Condition channels do not match {c.shape[1]} != {self.hparams.cond_dim}. \n"
            + "Hint: The cond_dim parameter is the number of channels if the condition is an image."
        )
        if c.ndim == 2:
            c = c[:, :, None, None] * torch.ones(
                batch_size, self.hparams.cond_dim, *input_shape[1:], device=c.device
            )

        out = torch.cat([x_img, c], -3)
        if not has_batch_dimension:
            out = out[0]
        return out

    def encode(self, x, c):
        return self.encoder(self.cat_x_c(x, c))

    def decode(self, u, c):
        if self.hparams.image_shape is not None:
            resolution = torch.Size(self.hparams.image_shape)[1:]
        else:
            resolution = guess_image_shape(self.hparams.data_dim)[1:]
        if c.ndim == 2:
            c = c[:, :, None, None] * torch.ones(
                c.shape[0],
                self.hparams.cond_dim,
                *resolution,
                device=c.device,
            )
        return self.decoder(u, im_cond=c).flatten(1)


class BigGANDecoderAnySize(nn.Module):
    """Wraps a BigGAN into our autoencoding framework"""

    def __init__(
        self, im_ch, chn, cond_ch, z_dim, input_size, decoder_norm, latimage=False
    ):
        super().__init__()
        self.latimage = latimage
        self.z_dim = z_dim
        use_actnorm = True if decoder_norm == "an" else False
        class_embedding_dim = 1000
        self.extra_z_dims = list()

        self.map_to_class_embedding = ClassUp(
            z_dim,
            depth=2,
            hidden_dim=2 * class_embedding_dim,
            use_sigmoid=False,
            out_dim=class_embedding_dim,
        )
        self.decoder = Generator(
            code_dim=120,
            z_dim=z_dim,
            im_ch=im_ch,
            n_class=class_embedding_dim,
            chn=chn,
            use_actnorm=use_actnorm,
            cond_ch=cond_ch,
            output_size=input_size,
            latimage=self.latimage,
        )

    def forward(self, x, labels=None, im_cond=None):
        emb = self.map_to_class_embedding(x[:, : self.z_dim, ...])
        x = self.decoder(x, emb, im_cond=im_cond)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        code_dim,
        z_dim,
        im_ch,
        n_class,
        chn,
        use_actnorm,
        cond_ch,
        output_size=128,
        latimage=False,
    ):
        super().__init__()

        self.latimage = latimage
        self.z_dim = z_dim
        self.output_size = output_size
        if not latimage:
            self.num_layers = int(math.log2(output_size)) - 2
        else:
            self.num_layers = 5  # number of downsampling layers in ResNet50
        self.init_side_length = self.output_size // (2 ** (self.num_layers - latimage))
        if latimage and z_dim % (self.init_side_length) ** 2 != 0:
            raise ValueError(
                f"z_dim must be divisible by latent resolution**2 {(self.init_side_length**2)} when latimage=True"
            )

        self.sa_id = self.num_layers - 2
        self.num_split = self.num_layers + 1
        self.linear = nn.Linear(n_class, 2 * (self.init_side_length**2), bias=False)

        if self.latimage:
            split_size = 1
            self.first_view = (
                z_dim // (self.init_side_length**2) - (self.num_split - 1) * split_size
            )
            first_split = self.first_view
        else:
            split_size = 20
            self.first_view = (output_size // 8) ** 2 * chn
            first_split = z_dim - (self.num_split - 1) * split_size
            self.G_linear = SpectralNorm(nn.Linear(first_split, self.first_view))

        self.split_at = [first_split] + [split_size for _ in range(self.num_split - 1)]

        G_block_z_dim = 3 if self.latimage else code_dim + 28
        first_chn = (
            first_split
            if self.latimage
            else self.first_view // (self.output_size // (2**self.num_layers)) ** 2
        )

        channels = [first_chn] + [
            chn * (2**i) for i in range(self.num_layers - 1, -1, -1)
        ]
        print(channels)
        self.GBlock = nn.ModuleList(
            [
                GBlock(
                    channels[i] + cond_ch,
                    channels[i + 1],
                    n_class=n_class,
                    z_dim=G_block_z_dim,
                    upsample=(not self.latimage) or i > 0,
                    latimage=self.latimage,
                )
                for i in range(self.num_layers)
            ]
        )

        self.attention = SelfAttention(channels[self.sa_id])
        self.ScaledCrossReplicaBN = (
            BatchNorm2d(channels[-1], eps=1e-4)
            if not use_actnorm
            else ActNorm(channels[-1])
        )
        self.colorize = nn.Conv2d(
            channels[-1] + cond_ch, im_ch, kernel_size=3, padding=1
        )

    def forward(self, input, class_id, im_cond=None):

        if self.latimage:
            input = input.view(
                -1,
                self.z_dim // (self.init_side_length**2),
                self.init_side_length,
                self.init_side_length,
            )

        codes = torch.split(input, self.split_at, 1)
        class_emb = self.linear(class_id)

        if not self.latimage:
            out = self.G_linear(codes[0])
            out = out.view(
                -1,
                self.init_side_length,
                self.init_side_length,
                self.first_view // (self.init_side_length**2),
            ).permute(0, 3, 1, 2)
        else:
            class_emb = class_emb.view(
                -1, 2, self.init_side_length, self.init_side_length
            )
            out = codes[0]

        for i, (code, GBlock) in enumerate(zip(codes[1:], self.GBlock)):
            if i == self.sa_id:
                out = self.attention(out)
            condition = torch.cat([code, class_emb], 1)
            if i > 1 and self.latimage:
                condition = F.interpolate(condition, scale_factor=2 ** (i - 1))
            if im_cond is not None:
                scale_factor = 1 / 2 ** (self.num_layers - i)
                if self.latimage and i == 0:
                    scale_factor = 1 / 2 ** (self.num_layers - i - 1)
                im_cond_rescaled = F.interpolate(im_cond, scale_factor=scale_factor)
                out = torch.cat([out, im_cond_rescaled], 1)
            out = GBlock(out, condition)

        out = self.ScaledCrossReplicaBN(out)
        if im_cond is not None:
            out = torch.cat([out, im_cond], 1)
        out = self.colorize(out)
        return out

    def encode(self, *args, **kwargs):
        raise RuntimeError("BigGAN architecture does not have an encoder")

    def decode(self, z, cls):
        z = z.float()
        cls_one_hot = torch.nn.functional.one_hot(cls, num_classes=1000).float()
        return self.forward(z, cls_one_hot)
