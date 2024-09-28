import torch.nn
from torch import nn

from fff.base import ModelHParams
from .utils import batch_wrap, make_inn


class MultilevelFlowHParams(ModelHParams):
    inn_spec: list
    zero_init: bool = True
    phase: bool | int = False


class MultilevelFlow(nn.Module):
    """
    A Multilevel Splitflow with fixed architecture:
    If hparams.ssl 2 wavelet levels (0 and coarse wavelet) and two output INNs,
    else 1 wavelet and 1 output INN (details).
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
        self.cwavelet_inn = self.build_inn(self.hparams.cond_dim, cond=None)
        if hparams.phase == False:
            self.coarse_inn = self.build_inn(self.hparams.cond_dim, cond=None)
        self.details_dim = self.hparams.latent_dim - self.hparams.cond_dim
        self.details_inn = self.build_inn(self.details_dim, cond_dim=hparams.cond_dim)

    def encode(self, x, c):
        if self.hparams.phase in [False, 1]:
            out0, jac0 = self.wavelet_inn(x, jac=True, rev=False)
            _out0_details = out0[:, :-self.hparams.cond_dim]
            _out0_coarse = out0[:, -self.hparams.cond_dim:]
            c_hat, jac_c = self.cwavelet_inn(_out0_coarse, jac=True, rev=False)
            if self.hparams.phase == False:
                details, jac_details = self.details_inn(_out0_details, [c_hat], jac=True, rev=False)
                coarse, jac_coarse = self.coarse_inn(c_hat, jac=True, rev=False)
                jacs = [jac0, jac_details, jac_c, jac_coarse]
            else:
                jacs = [jac0, jac_c]
                details = _out0_details
                coarse = c_hat
        elif self.hparams.phase == 2:
            with torch.no_grad():
                out0, jac0 = self.wavelet_inn(x, jac=True, rev=False)
            _out0_details = out0[:, :-self.hparams.cond_dim]
            details, jac_details = self.details_inn(_out0_details, [c], jac=True, rev=False)
            coarse = c
            jacs = [jac_details]
        else:
            raise NotImplementedError("phase parameter has to be either False, 1 or 2")
        jac = torch.sum(torch.stack(jacs, dim=1), dim=1)
        z_dense = torch.cat([details, coarse], dim=1)
        return  (z_dense, c_hat), jac

    #def encode(self, u, c=None):
    #    return u

    def decode(self, u, c):
        _details_in = u[:, :self.details_dim]
        out_c = self.cwavelet_inn(c, jac=False, rev=True)[0]
        out_d = self.details_inn(_details_in, [c], jac=False, rev=True)[0]
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
