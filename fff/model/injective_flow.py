import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class InjectiveFlowHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = True


class InjectiveFlow(nn.Module):
    """
    This uses a INN to map from data to latent space and back.
    In the case that latent_dim < data_dim, the latent space is a subspace of the data space.
    For reverting, the latent space is padded with zeros.
    """
    hparams: InjectiveFlowHParams

    def __init__(self, hparams: dict | InjectiveFlowHParams):
        if not isinstance(hparams, InjectiveFlowHParams):
            hparams = InjectiveFlowHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

    def latent_encode(self, x, c):
        return self.model(x, [c], jac=True, rev=False)

    def encode(self, u, c=None):
        return u

    def latent_decode(self, u, c):
        return self.model(u, [c], jac=False, rev=True)[0]

    def decode(self, z, c=None):
        return z

    def build_model(self) -> nn.Module:
        dim = self.hparams.latent_dim
        return make_inn(self.hparams.inn_spec, dim, zero_init=self.hparams.zero_init)
