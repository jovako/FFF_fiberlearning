from torch.nn import Module
import torch.nn as nn
import torch
import warnings

from fff.base import ModelHParams, build_model
from ldctinv.pretrained import load_pretrained
from ldctinv import utils
from fff.model.utils import guess_image_shape, wrap_batch_norm2d
from fff.model import Identity


class LosslessAEHParams(ModelHParams):
    model_spec: list = []
    cond_dim: int | list = 0
    path: str | None = None
    vae: bool = True
    data_dim: int
    train: bool = False
    use_ldct_networks: bool = False
    cond_embedding_network: list = []
    use_cond_decoder: bool = False


class LosslessAE(Module):

    hparams: LosslessAEHParams

    def __init__(self, hparams: LosslessAEHParams | dict):
        if (
            "path" in hparams
            and hparams["path"] is not None
            and not hparams["use_ldct_networks"]
        ):
            print("Loading lossless_ae checkpoint from: ", hparams["path"])
            checkpoint = torch.load(hparams["path"])
            hparams["model_spec"] = checkpoint["hyper_parameters"]["lossless_ae"]

        if not isinstance(hparams, LosslessAEHParams):
            hparams = LosslessAEHParams(**hparams)
        if hparams.use_ldct_networks and not hparams.path:
            raise ValueError(
                "Can only load pretrained models from ldctinv, so path must be provided."
            )
        super().__init__()

        self.hparams = hparams
        self.data_dim = self.hparams.data_dim
        if self.hparams.vae and hparams["path"] is None:
            lat_dim = self.hparams.model_spec[-1]["latent_dim"]
            self.hparams.model_spec[-1]["latent_dim"] = lat_dim * 2

        if self.hparams.use_ldct_networks:
            input_shape = guess_image_shape(self.data_dim)
            cond_shape = guess_image_shape(self.hparams.cond_dim)
            self.unflatten = nn.Unflatten(
                -1, (input_shape[0] + cond_shape[0], *input_shape[1:])
            )
            self.flatten = nn.Flatten()
            self.unflatten_c = nn.Unflatten(-1, cond_shape)
            try:
                vae, vae_args = utils.setup_trained_model(
                    hparams["path"],
                    network_name="BigAE",
                    state_dict="generator",
                    in_ch=2,
                    out_ch=1,
                    cond_ch=1,
                    return_args=True,
                )
                self.models = nn.Sequential(
                    vae.eval(),
                )
            except:
                self.models = nn.Sequential(
                    load_pretrained(hparams["path"], eval=not self.hparams.train)[0][
                        "vae"
                    ],
                )
        else:
            self.models = build_model(
                self.hparams.model_spec, self.data_dim, self.hparams.cond_dim
            )
            if self.hparams.path:
                lossless_ae_weights = {
                    k[19:]: v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("lossless_ae.models.")
                }
                self.models.load_state_dict(lossless_ae_weights)

        if not self.hparams.train:
            self.models.eval()

        if self.hparams.cond_embedding_network:
            if self.hparams.use_ldct_networks:
                warnings.warn(
                    "cond_embedding_network is not tested with use_ldct_networks"
                )
            # Build a network to embed the conditioning
            self.cond_embedder = build_model(
                self.hparams.cond_embedding_network, self.hparams.cond_dim, 0
            )
            if not self.hparams.train:
                self.cond_embedder.eval()
        else:
            self.cond_embedder = Identity()

    @property
    def latent_dim(self):
        if not self.hparams.use_ldct_networks:
            latent_dim = self.models[-1].hparams.latent_dim
            if self.hparams.vae:
                return latent_dim // 2
            else:
                return latent_dim
        else:
            return self.models[-1].encoder.z_dim

    def decode(self, z, c):
        if self.hparams.cond_dim == 0:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        if self.hparams.use_ldct_networks:
            c = self.unflatten_c(c)
        if self.hparams.cond_embedding_network:
            if self.hparams.use_cond_decoder:
                c = self.cond_embedder.decode(c)
            else:
                c = self.cond_embedder.encode(c)
        if self.hparams.vae and not self.hparams.use_ldct_networks:
            z = torch.nn.functional.pad(z, (0, z.shape[1]))
        for model in self.models[::-1]:
            z = model.decode(z, c)
        if self.hparams.use_ldct_networks:
            z = self.flatten(z)
        return z

    def encode(self, x, c, mu_var=False):
        if self.hparams.cond_dim == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        if self.hparams.cond_embedding_network:
            if self.hparams.use_cond_decoder:
                c = self.cond_embedder.decode(c)
            else:
                c = self.cond_embedder.encode(c)
        if self.hparams.use_ldct_networks:
            x = self.cat_x_c(x, c)
            x = self.unflatten(x)
            for model in self.models:
                x = model.encode(x).sample().squeeze()
        else:
            for model in self.models:
                x = model.encode(x, c)
        mu, logvar = None, None
        if self.hparams.vae and not self.hparams.use_ldct_networks:
            # VAE latent sampling
            mu = x[:, : x.shape[1] // 2].reshape(-1, x.shape[1] // 2)
            logvar = x[:, x.shape[1] // 2 :].reshape(-1, x.shape[1] // 2)
            epsilon = torch.randn_like(logvar).to(mu.device)
            x = mu + torch.exp(0.5 * logvar) * epsilon
        if mu_var:
            x = x, mu, logvar
        return x

    def cat_x_c(self, x, c):
        # Reshape as image, and concatenate conditioning as channel dimensions
        return torch.cat([x, c], 1)
