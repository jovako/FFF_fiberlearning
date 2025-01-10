from torch.nn import Module
import torch

from fff.base import ModelHParams, build_model

class LosslessAEHParams(ModelHParams):
    model_spec: list
    cond_dim: int = 0
    path: bool | str = False
    vae: bool = True
    data_dim: int

class LosslessAE(Module):

    hparams: LosslessAEHParams
    def __init__(self, hparams: LosslessAEHParams | dict):
        if not isinstance(hparams, LosslessAEHParams):
            hparams = LosslessAEHParams(**hparams)
        super().__init__()
        self.hparams = hparams
        self.data_dim = self.hparams.data_dim
        self.models = build_model(
            self.hparams.model_spec, self.data_dim, self.hparams.cond_dim
        )
        if self.hparams.path:
            print("load lossless_ae checkpoint")
            checkpoint = torch.load(self.hparams.path)
            lossless_ae_weights = {k[19:]: v for k, v in checkpoint["state_dict"].items()
                              if k.startswith("lossless_ae.models.")}
            self.models.load_state_dict(lossless_ae_weights)

    @property
    def latent_dim(self):
        return self.models[-1].hparams.latent_dim

    def decode(self, z, c):
        if self.hparams.cond_dim == 0:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        for model in self.models[::-1]:
            z = model.decode(z, c)
        return z

    def encode(self, x, c, mu_var=False):
        if self.hparams.cond_dim == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        for model in self.models:
            x = model.encode(x, c)
        mu, logvar = None, None
        if self.hparams.vae:
            # VAE latent sampling
            mu, logvar = x
            epsilon = torch.randn_like(logvar).to(mu.device)
            x = mu + torch.exp(0.5 * logvar) * epsilon
        if mu_var:
            x = x, mu, logvar
        return x
