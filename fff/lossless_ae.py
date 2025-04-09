from torch.nn import Module
import torch.nn as nn
import torch
import warnings
from math import prod

from fff.base import ModelHParams, build_model
from ldctinv.pretrained import load_pretrained
from ldctinv import utils
from fff.model.utils import guess_image_shape, wrap_batch_norm2d
from fff.model import Identity
import copy

class LosslessAEHParams(ModelHParams):
    model_spec: list = []
    cond_dim: int | list = 0
    path: str | None = None
    vae: bool = True
    data_dim: int
    train: bool = False
    use_pretrained_ldct_networks: bool = False
    cond_embedding_network: list = []
    cond_embedding_shape: int | list | None = None
    use_condition_decoder: bool = False


class LosslessAE(Module):

    hparams: LosslessAEHParams

    def __init__(self, hparams: LosslessAEHParams | dict):
        if hparams.get("path") and not hparams.get("use_pretrained_ldct_networks"):
            checkpoint = torch.load(hparams["path"], weights_only=False)
            print("Overwriting lossless ae model spec with pretrained model")
            hparams["model_spec"] = checkpoint["hyper_parameters"]["lossless_ae"][
                "model_spec"
            ]
            hparams["cond_embedding_network"] = checkpoint["hyper_parameters"][
                "lossless_ae"
            ].get("cond_embedding_network")
            hparams["cond_embedding_shape"] = checkpoint["hyper_parameters"][
                "lossless_ae"
            ].get("cond_embedding_shape")
            hparams["use_condition_decoder"] = checkpoint["hyper_parameters"][
                "lossless_ae"
            ].get("use_condition_decoder")
            hparams["vae"] = checkpoint["hyper_parameters"]["lossless_ae"].get("vae")

        if not isinstance(hparams, LosslessAEHParams):
            hparams = LosslessAEHParams(**hparams)
        if hparams.use_pretrained_ldct_networks and not hparams.path:
            raise ValueError(
                "use_pretrained_ldct_networks requires a path to a pretrained model"
            )
        super().__init__()

        self.hparams = hparams
        self.data_dim = self.hparams.data_dim

        if self.hparams.cond_embedding_shape is None:
            assert (
                self.hparams.cond_embedding_network == []
            ), "cond_embedding_shape must be specified if cond_embedding_network is specified"
            self.hparams.cond_embedding_shape = [self.hparams.cond_dim]
        else:
            if isinstance(self.hparams.cond_embedding_shape, int):
                self.hparams.cond_embedding_shape = [self.hparams.cond_embedding_shape]
            assert not (
                self.hparams.cond_embedding_network == []
            ), "cond_embedding_network must be specified if cond_embedding_shape is specified"

        if self.hparams.vae and hparams["path"] is None:
        model_spec = copy.deepcopy(self.hparams.model_spec)
        if self.hparams.vae:
            lat_dim = self.hparams.model_spec[-1]["latent_dim"]
            model_spec[-1]["latent_dim"] = lat_dim * 2

        if self.hparams.use_pretrained_ldct_networks:
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
                model_spec,
                self.data_dim,
                self.hparams.cond_embedding_shape[0],
            )

            if self.hparams.cond_embedding_network:
                if self.hparams.use_pretrained_ldct_networks:
                    warnings.warn(
                        "cond_embedding_network is not tested with use_pretrained_ldct_networks"
                    )
                # Build a network to embed the conditioning
                if self.hparams.use_condition_decoder:
                    self.condition_embedder = build_model(
                        self.hparams.cond_embedding_network,
                        prod(self.hparams.cond_embedding_shape),
                        0,
                    )
                    for model in self.condition_embedder:
                        del model.model.encoder
                else:
                    self.condition_embedder = build_model(
                        self.hparams.cond_embedding_network,
                        self.hparams.cond_dim,
                        0,
                    )
                    for model in self.condition_embedder:
                        del model.model.decoder
                if not self.hparams.train:
                    self.condition_embedder.eval()
            else:
                self.condition_embedder = Identity(self.hparams)

            if self.hparams.path:
                try:
                    print("Loading lossless_ae checkpoint from: ", hparams["path"])
                    lossless_ae_weights = {
                        k[len("lossless_ae.") :]: v
                        for k, v in checkpoint["state_dict"].items()
                        if k.startswith("lossless_ae.")
                    }
                    self.load_state_dict(lossless_ae_weights)
                except:
                    print("Loading lossless_ae checkpoint from: ", hparams["path"])
                    lossless_ae_weights = {
                        k[len("models.") :]: v
                        for k, v in checkpoint["state_dict"].items()
                        if k.startswith("models.")
                    }
                    self.models.load_state_dict(lossless_ae_weights)

        if not self.hparams.train:
            self.models.eval()

    @property
    def latent_dim(self):
        if not self.hparams.use_pretrained_ldct_networks:
            latent_dim = self.models[-1].hparams.latent_dim
            if self.hparams.vae:
                return latent_dim // 2
            else:
                return latent_dim
        else:
            return self.models[-1].encoder.z_dim

    def embed_condition(self, c):
        if self.hparams.cond_embedding_network:
            if self.hparams.use_condition_decoder:
                for model in self.condition_embedder[::-1]:
                    c = model.decode(
                        c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                    )
            else:
                for model in self.condition_embedder:
                    c = model.encode(
                        c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                    )
        return c.reshape(c.shape[0], *self.hparams.cond_embedding_shape)

    def decode(self, z, c, **kwargs):
        if self.hparams.cond_dim == 0:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        if self.hparams.use_pretrained_ldct_networks:
            c = self.unflatten_c(c)
        c = self.embed_condition(c)
        if self.hparams.vae:
            z = torch.nn.functional.pad(z, (0, z.shape[1]))
        for model in self.models[::-1]:
            z = model.decode(z, c, **kwargs)
        if self.hparams.use_pretrained_ldct_networks:
            z = self.flatten(z)
        return z

    def encode(self, x, c, return_only_x=False, deterministic=False **kwargs):
        if self.hparams.cond_dim == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        c = self.embed_condition(c)
        if self.hparams.use_pretrained_ldct_networks:
            x = self.cat_x_c(x, c)
            x = self.unflatten(x)
            for model in self.models:
                x = model.encode(x).sample().squeeze()
        else:
            for model in self.models:
                x = model.encode(x, c, **kwargs)
                other = []
                if isinstance(x, tuple):
                    x, other = x[0], x[1:]
        mu, logvar = None, None
        if self.hparams.vae and not self.hparams.use_pretrained_ldct_networks:
            # VAE latent sampling
            mu = x[:, : x.shape[1] // 2].reshape(-1, x.shape[1] // 2)
            logvar = x[:, x.shape[1] // 2 :].reshape(-1, x.shape[1] // 2)
            if deterministic:
                x = mu
            else:
                epsilon = torch.randn_like(logvar).to(mu.device)
                x = mu + torch.exp(0.5 * logvar) * epsilon
        if return_only_x:
            return x
        return x, mu, logvar, *other

    def cat_x_c(self, x, c):
        return torch.cat([x, c], 1)
