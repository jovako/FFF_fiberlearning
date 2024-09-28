import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class DenoisingFlowHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = True
    ssf: bool = True


class DenoisingFlow(nn.Module):
    """
    A Denoising Splitflow with fixed architecture:
    If hparams.ssl 2 wavelet levels (0 and coarse wavelet) and two output INNs,
    else 1 wavelet and 1 output INN (coarse).
    This uses a INN to map from data to latent space and back.
    In the case that latent_dim < data_dim, the latent space is a subspace of the data space.
    For reverting, the latent space is padded with zeros.
    """
    hparams: DenoisingFlowHParams

    def __init__(self, hparams: dict | DenoisingFlowHParams):
        if not isinstance(hparams, DenoisingFlowHParams):
            hparams = DenoisingFlowHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.wavelet_inn = self.build_inn(hparams.latent_dim, cond_dim=hparams.cond_dim)
        self.coarse_dim = self.hparams.latent_dim - self.hparams.cond_dim
        self.coarse_inn = self.build_inn(self.coarse_dim, cond_dim=2*hparams.cond_dim)

    def encode(self, x, c):
        #global wavelet
        out0, jac0 = self.wavelet_inn(x, [c], jac=True, rev=False)

        _out0_coarse = out0[:, :-self.hparams.cond_dim]
        _out0_details = out0[:, -self.hparams.cond_dim:]
        details = _out0_details
        #coarse split
        cond = torch.cat([details, c], dim=1)
        coarse, jac_c = self.coarse_inn(_out0_coarse, [cond], jac=True, rev=False)
        jac = torch.sum(torch.stack([jac0, jac_c], dim=1), dim=1)
        z_dense = torch.cat([coarse, details], dim=1)
        return  (z_dense, details), jac

    #def encode(self, u, c=None):
    #    return u

    def decode(self, u, c):
        _coarse_in = u[:, :self.coarse_dim]
        out_d = u[:, self.coarse_dim:]
        cond = torch.cat([out_d, c], dim=1)
        out_c = self.coarse_inn(_coarse_in, [cond], jac=False, rev=True)[0]
        in0 = torch.cat([out_c, out_d], dim=1)
        out = self.wavelet_inn(in0, [c], jac=False, rev=True)[0]
        return out

    #def decode(self, z, c=None):
    #    return z

    def build_inn(self, dim, cond_dim=0, cond=0) -> nn.Module:
        return make_inn(self.hparams.inn_spec, dim, cond_dim=cond_dim,
                        cond=cond, zero_init=self.hparams.zero_init)

    def build_coarse(self) -> nn.Module:
        dim = self.hparams.latent_dim - self.hparams.cond_dim
        details_dim = self.hparams.cond_dim
        return make_inn(self.hparams.inn_spec, dim, cond_dim=details_dim, zero_init=self.hparams.zero_init)
