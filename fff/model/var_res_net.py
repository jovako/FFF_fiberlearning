from collections import OrderedDict

import torch.nn
from torch import nn

from .auto_encoder import SkipConnection
from fff.base import ModelHParams
from .utils import wrap_batch_norm1d, make_dense


class ResNetHParams(ModelHParams):
    classification: bool = False
    layers_spec: list
    activation: str = "silu"
    id_init: bool = False
    batch_norm: str | bool = False
    dropout: float | None = None

    final_batch_norm: bool | str = False
    final_linear: None | bool = None

    def __init__(self, **hparams):
        if "latent_spec" in hparams:
            assert len(hparams["latent_spec"]) == 0
            del hparams["latent_spec"]
        super().__init__(**hparams)


def make_res_net(data_dim, layers_widths, activation, id_init: bool,
                 batch_norm: str | bool, dropout: float = None):
    sequential = nn.Sequential()
    for widths in layers_widths:
        sequential.append(SkipConnection(
            make_dense([data_dim, *widths, data_dim], activation,
                       batch_norm=batch_norm, dropout=dropout),
            id_init=id_init
        ))
    return sequential


class VarResNet(nn.Module):
    hparams: ResNetHParams

    def __init__(self, hparams: dict | ResNetHParams):
        if not isinstance(hparams, ResNetHParams):
            hparams = ResNetHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.model = self.build_model()

    def encode(self, x, c):
        out = self.model.encoder(torch.cat([x, c], -1))
        mu = self.model.mean_layer(out)
        logvar = self.model.logvar_layer(out)
        return mu, logvar

    def decode(self, z, c):
        return self.model.decoder(torch.cat([z, c], -1))[..., :self.hparams.data_dim]

    def build_model(self) -> nn.Module:
        data_dim = self.hparams.data_dim
        cond_dim = self.hparams.cond_dim
        print("cond_dim: ", cond_dim)
        latent_dim = self.hparams.latent_dim
        print("latent_dim: ", latent_dim)

        # ResNet in data space + projection to latent space
        activation = self.hparams.activation
        encoder = make_res_net(
            data_dim + cond_dim, self.hparams.layers_spec, activation,
            id_init=self.hparams.id_init,
            batch_norm=self.hparams.batch_norm, dropout=self.hparams.dropout
        )
        mean_layer = nn.Linear(data_dim + cond_dim, self.hparams.latent_dim)
        logvar_layer = nn.Linear(data_dim + cond_dim, self.hparams.latent_dim)

        decoder = nn.Sequential()
        if data_dim + cond_dim != latent_dim:
            decoder.append(nn.Linear(
                latent_dim + cond_dim, data_dim + cond_dim
            ))
        decoder.extend(make_res_net(
            data_dim + cond_dim, self.hparams.layers_spec[::-1], activation,
            id_init=self.hparams.id_init,
            batch_norm=self.hparams.batch_norm, dropout=self.hparams.dropout
        ))

        return torch.nn.Sequential(OrderedDict(
            encoder=encoder,
            mean_layer=mean_layer,
            logvar_layer=logvar_layer,
            decoder=decoder
        ))