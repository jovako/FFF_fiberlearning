import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class MultilevelFlowHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = True


class MultilevelFlow(nn.Module):
    """
    This uses a INN to map from data to latent space and back.
    In the case that latent_dim < data_dim, the latent space is a subspace of the data space.
    For reverting, the latent space is padded with zeros.
    """
    hparams: MultilevelFlowHParams

    def __init__(self, hparams: dict | MultilevelFlowHParams):
        if not isinstance(hparams, MultilevelFlowHParams):
            hparams = MultilevelFlowHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.wavelet_inn = self.build_inn(hparams.latent_dim, cond=None)
        dim_details = hparams.latent_dim - hparams.cond_dim
        self.details_inn = self.build_inn(dim_details, cond_dim=hparams.cond_dim)
        #self.cwavelet_inn = self.build_inn(self.hparams.cond_dim, cond=None)
        self.coarse_layer = nn.Linear(self.hparams.cond_dim, self.hparams.cond_dim)
        self.hparams.latent_dim = self.hparams.latent_dim - self.hparams.cond_dim

    def encode(self, x, c):
        out0 = self.wavelet_inn(x, jac=False, rev=False)[0]
        coarse = self.coarse_layer(out0[:, -self.hparams.cond_dim:])
        _out0_details = out0[:, :-self.hparams.cond_dim].detach()
        details, jac_d = self.details_inn(_out0_details, [coarse.detach()], jac=True, rev=False)
        #coarse, jac_coarse = self.coarse_inn(c_hat, jac=True, rev=False)
        #jac = torch.sum(torch.stack([jac0, jac_d], dim=1), dim=1)
        #z_dense = torch.cat([details, coarse], dim=1)
        return  (details, coarse), jac_d

    #def encode(self, u, c=None):
    #    return u

    def decode(self, u, c):
        #details_in = u[:, :-self.hparams.cond_dim]
        out_d = self.details_inn(u, [c], jac=False, rev=True)[0]
        in0 = torch.cat([out_d, c], dim=1)
        out = self.wavelet_inn(in0, jac=False, rev=True)[0]
        return out

    #def decode(self, z, c=None):
    #    return z

    def build_inn(self, dim, cond_dim=0, cond=0) -> nn.Module:
        return make_inn(self.hparams.inn_spec, dim, cond_dim=cond_dim,
                        cond=cond, zero_init=self.hparams.zero_init)

    def build_details(self) -> nn.Module:
        dim = self.hparams.latent_dim - self.hparams.cond_dim
        coarse_dim = self.hparams.cond_dim
        return make_inn(self.hparams.inn_spec, dim, cond_dim=coarse_dim, zero_init=self.hparams.zero_init)
