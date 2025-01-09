from torch.nn import Module
import torch

from fff.base_model import ModelHParams

class LosslessAEHParams(ModelHParams):
    model_specs: list
    cond_dim: int = 0
    path: bool | str = False
    vae = bool = True

class LosslessAE(Module):

    hparams: LosslessAEHParams
    def __init__(self, hparams: LosslessAEHParams | dict):
        super().__init__()
        self.hparams = hparams
        self.datat_dim = self.hparams.data_dim
        self.models = build_model(
            self.hparams.model_specs, self.data_dim, self.hparams.cond_dim
        )
        if self.hparams.path:
            print("load lossless_ae checkpoint")
            checkpoint = torch.load(self.hparams.path)
            lossless_ae_weights = {k[19:]: v for k, v in checkpoint["state_dict"].items()
                              if k.startswith("lossless_ae.models.")}
            self.models.load_state_dict(lossless_ae_weights)

    def decode(self, z, c):
        for model in self.models:
            z = model.decode(z, c)
        return z

    def encode(self, x, c, mu_var=False):
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
