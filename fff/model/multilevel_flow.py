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
        self.cwavelet_inn = self.build_inn(self.hparams.cond_dim, cond=None)
        self.coarse_inn = self.build_inn(self.hparams.cond_dim, cond=None)
        #self.hparams.latent_dim = self.hparams.latent_dim - self.hparams.cond_dim
        self.details_dim = self.hparams.latent_dim - self.hparams.cond_dim

    def encode(self, x, c):
        #global wavelet
        out0, jac0 = self.wavelet_inn(x, jac=True, rev=False)
        #coarse wavelet
        c_hat, jac_c = self.cwavelet_inn(out0[:, -self.hparams.cond_dim:], jac=True, rev=False)
        #details split
        _out0_details = out0[:, :-self.hparams.cond_dim]
        details, jac_d = self.details_inn(_out0_details, [c_hat], jac=True, rev=False)
        #coarse output
        coarse, jac_coarse = self.coarse_inn(c_hat, jac=True, rev=False)
        jac = torch.sum(torch.stack([jac0, jac_d, jac_c, jac_coarse], dim=1), dim=1)
        z_dense = torch.cat([details, coarse], dim=1)
        return  (z_dense, c_hat), jac

    #def encode(self, u, c=None):
    #    return u

    def decode(self, u, c):
        details_in = u[:, :self.details_dim]
        out_d = self.details_inn(details_in, [c], jac=False, rev=True)[0]
        out_c = self.cwavelet_inn(c, jac=False, rev=True)[0]
        in0 = torch.cat([out_d, out_c], dim=1)
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
