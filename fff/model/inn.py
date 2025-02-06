import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class INNHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = False


class INN(nn.Module):
    """
    This uses a INN to map from data to latent space and back.
    In the case that latent_dim < data_dim, the latent space is a subspace of the data space.
    For reverting, the latent space is padded with zeros.
    """
    hparams: INNHParams

    def __init__(self, hparams: dict | INNHParams):
        if not isinstance(hparams, INNHParams):
            hparams = INNHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

    def encode(self, x, c):
        return self.model(x, [c], jac=True, rev=False)

    #def encode(self, u, c=None):
    #    return u

    def decode(self, u, c):
        return self.model(u, [c], jac=False, rev=True)[0]

    def sample(self, u, c):
        return self.decode(u, c)

    #def decode(self, z, c=None):
    #    return z

    def build_model(self) -> nn.Module:
        dim = self.hparams.latent_dim
        print(dim)
        cond_dim = self.hparams.cond_dim
        return make_inn(self.hparams.inn_spec, dim, cond_dim=cond_dim, zero_init=self.hparams.zero_init)
