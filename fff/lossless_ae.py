from torch.nn import Module
import torch.nn as nn
import torch

from fff.base import ModelHParams, build_model
from ldctinv.pretrained import load_pretrained
from ldctinv import utils
from fff.model.utils import guess_image_shape, wrap_batch_norm2d
import copy

class LosslessAEHParams(ModelHParams):
    model_spec: list = []
    cond_dim: int = 0
    path: str | None = None
    vae: bool = True
    data_dim: int
    train: bool = False

class LosslessAE(Module):

    hparams: LosslessAEHParams
    def __init__(self, hparams: LosslessAEHParams | dict):
        self.ldct = False
        load_orig = False
        if hparams["path"] in ["cnn10", "redcnn", "wganvgg", "dugan"]:
            load_orig=True
            self.ldct = True 
        if "path" in hparams and hparams["path"] is not None and not self.ldct:
            print("Loading lossless_ae checkpoint from: ", hparams["path"])
            checkpoint = torch.load(hparams["path"], weights_only=False)
            try:
                hparams["model_spec"] = checkpoint["hyper_parameters"]["lossless_ae"]
            except:
                hparams["model_spec"] = checkpoint["hyper_parameters"]["models"]

        if not isinstance(hparams, LosslessAEHParams):
            hparams = LosslessAEHParams(**hparams)
        super().__init__()
        self.hparams = hparams
        self.data_dim = self.hparams.data_dim
        model_spec = copy.deepcopy(self.hparams.model_spec)
        if self.hparams.vae:
            lat_dim = self.hparams.model_spec[-1]["latent_dim"]
            model_spec[-1]["latent_dim"] = lat_dim * 2

        if self.ldct:
            input_shape = guess_image_shape(self.data_dim)
            cond_shape = guess_image_shape(self.hparams.cond_dim)
            print("input_shape", cond_shape)
            self.unflatten = nn.Unflatten(-1, (input_shape[0] + cond_shape[0], *input_shape[1:]))
            self.flatten = nn.Flatten()
            self.unflatten_c = nn.Unflatten(-1, cond_shape)
            if not load_orig:
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
            else:
                self.models = nn.Sequential(
                    load_pretrained(hparams["path"], eval=not self.hparams.train)[0]["vae"],
                )
            print(not self.hparams.train)
        else:
            self.models = build_model(
                model_spec, self.data_dim, self.hparams.cond_dim
            )
            if self.hparams.path:
                try:
                    lossless_ae_weights = {k[19:]: v for k, v in checkpoint["state_dict"].items()
                                      if k.startswith("lossless_ae.models.")}
                    self.models.load_state_dict(lossless_ae_weights)
                except:
                    lossless_ae_weights = {k[7:]: v for k, v in checkpoint["state_dict"].items()
                                      if k.startswith("models.")}
                    self.models.load_state_dict(lossless_ae_weights)
        if not self.hparams.train:
            self.models.eval()

    @property
    def latent_dim(self):
        if not self.ldct:
            latent_dim = self.models[-1].hparams.latent_dim
            if self.hparams.vae:
                return latent_dim//2
            else:
                return latent_dim
        else:
            return self.models[-1].encoder.z_dim

    def decode(self, z, c):
        if self.hparams.cond_dim == 0:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        if self.ldct:
            c = self.unflatten_c(c)
            #if len(z.shape)!=4:
            #    z = z.unsqueeze(-1).unsqueeze(-1)
        if self.hparams.vae and not self.ldct:
            z = torch.nn.functional.pad(z, (0, z.shape[1]))
        for model in self.models[::-1]:
            z = model.decode(z, c)
        if self.ldct:
            z = self.flatten(z)
        return z

    def encode(self, x, c, mu_var=False, deterministic=False):
        if self.hparams.cond_dim == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        if self.ldct:
            x = self.cat_x_c(x,c)
            x = self.unflatten(x)
            for model in self.models:
                x = model.encode(x).sample().squeeze()
        else:
            for model in self.models:
                x = model.encode(x,c)
        mu, logvar = None, None
        if self.hparams.vae and not self.ldct:
            # VAE latent sampling
            mu = x[:,:x.shape[1]//2].reshape(-1,x.shape[1]//2)
            logvar = x[:,x.shape[1]//2:].reshape(-1,x.shape[1]//2)
            if deterministic:
                x = mu
            else:
                epsilon = torch.randn_like(logvar).to(mu.device)
                x = mu + torch.exp(0.5 * logvar) * epsilon
        if mu_var:
            x = x, mu, logvar
        return x

    def cat_x_c(self, x, c):
        # Reshape as image, and concatenate conditioning as channel dimensions
        return torch.cat([x,c],1)
        """
        has_batch_dimension = len(x.shape) > 1
        if not has_batch_dimension:
            x = x[None, :]
            c = c[None, :]
        batch_size = x.shape[0]
        input_shape = guess_image_shape(self.hparams.data_dim)
        x_img = x.reshape(batch_size, *input_shape)
        c_img = c[:, :, None, None] * torch.ones(batch_size, self.hparams.cond_dim, *input_shape[1:], device=c.device)
        out = torch.cat([x_img, c_img], -3).reshape(batch_size, -1)
        if not has_batch_dimension:
            out = out[0]
        return out
        """
