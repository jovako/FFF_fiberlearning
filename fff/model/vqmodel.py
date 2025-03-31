import torch
from torch import nn
from fff.base import ModelHParams
from fff.model.utils import guess_image_shape, is_local_path
from os import path
import os
from omegaconf import OmegaConf
import requests
from math import prod

try:
    from taming.models.vqgan import VQModel as VQModelTaming
    from taming.modules.vqvae.quantize import VectorQuantizer2
except:
    print("Importing vq models failed")


CONFIG_URLS = {
    "ImageNet_16384": "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
    "ImageNet_1024": "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1",
}

WEIGHT_URLS = {
    "ImageNet_16384": "https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
    "ImageNet_1024": "https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1",
}


class VQModelHParams(ModelHParams):
    architecture_type: str = "ImageNet_16384"
    image_shape: list[int] | None = None
    pretrained: bool = True
    condition_batchnorm: bool = False
    latent_shape: list[int] | None = None


class VQModel(nn.Module):
    def __init__(self, hparams: dict | VQModelHParams):
        super().__init__()
        if not isinstance(hparams, VQModelHParams):
            hparams = VQModelHParams(**hparams)
        self.hparams = hparams
        model = self.get_model(hparams.architecture_type, hparams.pretrained)
        self.model = self.adjust_dimensions(model)
        if self.hparams.condition_batchnorm:
            raise NotImplementedError(
                "Condition batchnorm is not yet implemented for VQ models."
            )

    def adjust_dimensions(self, model):
        del model.loss

        if self.hparams.image_shape is not None:
            input_shape = torch.Size(self.hparams.image_shape)
        else:
            input_shape = guess_image_shape(self.hparams.data_dim)

        # Adjust input and output convolutions to match input shape
        model.encoder.conv_in = adjust_and_copy_conv(
            input_shape[0] + self.hparams.cond_dim,
            model.encoder.conv_in.out_channels,
            model.encoder.conv_in,
            sum_in_channels=input_shape[0] == 1,
        )

        model.decoder.conv_in = adjust_and_copy_conv(
            model.decoder.conv_in.in_channels + self.hparams.cond_dim,
            model.decoder.conv_in.out_channels,
            model.decoder.conv_in,
        )

        model.post_quant_conv = adjust_and_copy_conv(
            model.post_quant_conv.in_channels + self.hparams.cond_dim,
            model.post_quant_conv.out_channels + self.hparams.cond_dim,
            model.post_quant_conv,
        )

        model.decoder.conv_out = adjust_and_copy_conv(
            model.decoder.conv_out.in_channels,
            input_shape[0],
            model.decoder.conv_out,
            sum_out_channels=input_shape[0] == 1,
        )

        self.input_shape = input_shape
        test_input = torch.randn(1, *input_shape)
        test_cond = torch.randn(1, self.hparams.cond_dim)
        z, _, _ = model.encode(self.cat_x_c(test_input, test_cond, side="data"))
        self.latent_shape = z.shape[1:]
        test_output = model.decode(self.cat_x_c(z.flatten(1), test_cond, side="latent"))
        assert test_output.shape == test_input.shape, (
            f"Model output shape {test_output.shape} does not match input shape {test_input.shape}."
            + "Hint: Check the input and output shapes of the model."
        )

        if prod(z.shape[1:]) != self.hparams.latent_dim:
            print("Adjusting model output to match latent dimension.")
            latent_res = z.shape[2]
            assert self.hparams.latent_dim % latent_res**2 == 0, (
                f"Latent dimension {self.hparams.latent_dim} does not divide latent resolution {latent_res}."
                + "Hint: Check the latent dimension and resolution of the model."
            )
            target_latent_channels = self.hparams.latent_dim // latent_res**2
            self.latent_shape = [target_latent_channels, latent_res, latent_res]

            model.encoder.conv_out = adjust_and_copy_conv(
                model.encoder.conv_out.in_channels,
                target_latent_channels,
                model.encoder.conv_out,
            )

            model.decoder.conv_in = adjust_and_copy_conv(
                target_latent_channels + self.hparams.cond_dim,
                model.decoder.conv_in.out_channels,
                model.decoder.conv_in,
            )

            model.quant_conv = adjust_and_copy_conv(
                target_latent_channels,
                target_latent_channels,
                model.quant_conv,
            )

            model.post_quant_conv = adjust_and_copy_conv(
                target_latent_channels + self.hparams.cond_dim,
                target_latent_channels + self.hparams.cond_dim,
                model.post_quant_conv,
            )

            model.quantize = VectorQuantizer2(
                model.quantize.n_e,
                target_latent_channels,
                beta=model.quantize.beta,
                remap=model.quantize.remap,
                unknown_index=(
                    "random"
                    if not hasattr(model.quantize, "unkown_index")
                    else model.quantize.unknown_index
                ),
                sane_index_shape=model.quantize.sane_index_shape,
                legacy=model.quantize.legacy,
            )

            min_channels = min(
                model.quantize.embedding.weight.data.shape[1], target_latent_channels
            )
            new_embedding = nn.Embedding(
                model.quantize.embedding.num_embeddings,
                target_latent_channels,
                padding_idx=model.quantize.embedding.padding_idx,
            )
            new_embedding.weight.data[:, :min_channels] = (
                model.quantize.embedding.weight.data[:, :min_channels]
            )
            model.quantize.embedding = new_embedding

        return model

    def cat_x_c(self, x, c, side="data"):
        # Reshape as image, and concatenate conditioning as channel dimensions
        has_batch_dimension = len(x.shape) > 1
        if not has_batch_dimension:
            x = x[None, :]
            c = c[None, :]
        batch_size = x.shape[0]
        shape = self.input_shape if side == "data" else self.latent_shape
        x_img = x.reshape(batch_size, *shape)
        assert c.shape[1] == self.hparams.cond_dim, (
            f"Condition channels do not match {c.shape[1]} != {self.hparams.cond_dim}. \n"
            + "Hint: The cond_dim parameter is the number of channels if the condition is an image."
        )
        if c.ndim == 2:
            c = c[:, :, None, None] * torch.ones(
                batch_size, self.hparams.cond_dim, *shape[1:], device=c.device
            )

        out = torch.cat([x_img, c], -3)
        if not has_batch_dimension:
            out = out[0]
        return out

    def encode(self, x, c, return_codebook_loss=False):
        out, codebook_loss, _ = self.model.encode(self.cat_x_c(x, c, side="data"))
        if return_codebook_loss:
            return out.flatten(1), codebook_loss
        return out.flatten(1)

    def decode(self, u, c):
        return self.model.decode(self.cat_x_c(u, c, side="latent")).flatten(1)

    @staticmethod
    def get_model(architecture_type: str, pretrained: bool = True):
        if is_local_path(architecture_type):
            config_path = path.join(architecture_type, "config.yaml")
            assert path.exists(
                config_path
            ), f"Model config file {architecture_type} not found."
            if pretrained:
                weight_path = path.join(architecture_type, "weights.ckpt")
                assert path.exists(
                    weight_path
                ), f"Model weight file {architecture_type} not found."
        else:
            config_path = path.join("vqmodels", architecture_type, "config.yaml")
            weight_path = path.join("vqmodels", architecture_type, "weights.ckpt")
            if not path.exists(config_path) or (
                pretrained and not path.exists(weight_path)
            ):
                VQModel.download_model(architecture_type, pretrained)

        config = OmegaConf.load(config_path)

        if hasattr(config, "model"):
            config = config.model
        if hasattr(config, "params"):
            config = config.params

        model = VQModelTaming(**config)
        if pretrained:
            model.load_state_dict(
                torch.load(weight_path, map_location="cpu")["state_dict"], strict=False
            )
        return model

    @staticmethod
    def download_model(architecture_type: str, pretrained: bool = True):
        print(f"Downloading {architecture_type} model...")
        os.makedirs(path.join("vqmodels", architecture_type), exist_ok=True)

        assert (
            architecture_type in CONFIG_URLS.keys()
        ), f"Architecture type {architecture_type} config url not found."
        config_url = CONFIG_URLS[architecture_type]
        response = requests.get(config_url, stream=True)
        with open(
            path.join("vqmodels", architecture_type, "config.yaml"), "wb"
        ) as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        if pretrained:
            assert (
                architecture_type in WEIGHT_URLS.keys()
            ), f"Architecture type {architecture_type} weight url not found."
            weight_url = WEIGHT_URLS[architecture_type]
            response = requests.get(weight_url, stream=True)
            with open(
                path.join("vqmodels", architecture_type, "weights.ckpt"), "wb"
            ) as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)


def adjust_and_copy_conv(
    new_in_ch: int,
    new_out_ch: int,
    old_conv: nn.Conv2d,
    sum_in_channels: bool = False,
    sum_out_channels: bool = False,
):
    new_conv = nn.Conv2d(
        new_in_ch,
        new_out_ch,
        old_conv.kernel_size,
        old_conv.stride,
        old_conv.padding,
    )

    min_channels_out = min(new_out_ch, old_conv.out_channels)
    min_channels_in = min(new_in_ch, old_conv.in_channels)
    new_conv.weight.data[:min_channels_out, :min_channels_in] = old_conv.weight.data[
        :min_channels_out, :min_channels_in
    ]

    if sum_in_channels:
        new_conv.weight.data[:min_channels_out, 0] = old_conv.weight.data[
            :min_channels_out
        ].sum(1)
    if sum_out_channels:
        new_conv.weight.data[0, :min_channels_in] = old_conv.weight.data[
            :, :min_channels_in
        ].sum(0)
    return new_conv
