import torch
from torch import Tensor, nn

from .auto_encoder import SkipConnection
from fff.base import ModelHParams
from .res_net import ResNetHParams, make_res_net
#from hydrantic.model import Model, ModelHparams  # https://github.com/hummerichsander/hydrantic
from flow_matching.solver import ODESolver  # pip install flow-matching
from .utils import expand_like, make_dense, InterpolantMixin
from typing import Literal

class FlowMatchingHParams(ResNetHParams):
    #Interpolation = Literal["linear", "trigonometric"]
    interpolation: str = "linear"

    sigma: float = 0.1  # interpolation noise amplitude

    def __init__(self, **hparams):
        super().__init__(**hparams)

class FlowMatching(nn.Module, InterpolantMixin):
    hparams: FlowMatchingHParams

    def __init__(self, hparams: dict | FlowMatchingHParams):
        if not isinstance(hparams, FlowMatchingHParams):
            hparams = FlowMatchingHParams(**hparams)
        super().__init__()
        self.hparams = hparams
        InterpolantMixin.__init__(self)

        self.net = self.build_model()

    def encode(self, x, *args):
        return x

    def decode(self, x: Tensor, t: Tensor) -> Tensor:
        t = expand_like(t, x[..., :1])
        x = torch.cat([x, t], dim=1)
        return self.net(x)

    def concat_noise(self, x0: Tensor) -> Tensor:
        noise = torch.randn(x0.shape[0], self.hparams.data_dim - self.hparams.cond_dim, device=x0.device)
        return torch.cat([x0, noise], dim=-1)

    @torch.no_grad()
    def compute_path_samples(self, x0: Tensor) -> Tensor:
        if x0.shape[-1] != self.hparams.data_dim:
            x0 = self.concat_noise(x0)

        solver = ODESolver(self.decode)
        sol = solver.sample(
            time_grid=torch.linspace(0, 1, 101), x_init=x0, method="midpoint", step_size=1e-1, return_intermediates=True
        )
        return sol

    def sample(self, z, condition):
        sol = self.compute_path_samples(condition)
        return sol[-1]

    def build_model(self) -> nn.Module:
        input_dim = self.hparams.data_dim + 1
        output_dim = self.hparams.data_dim
        activation = self.hparams.activation
        dropout = self.hparams.dropout

        net = make_res_net(
            input_dim, self.hparams.layers_spec, activation,
            id_init=self.hparams.id_init,
            batch_norm=self.hparams.batch_norm, dropout=self.hparams.dropout
        )
        net.append(nn.Linear(input_dim, output_dim))
        return net
