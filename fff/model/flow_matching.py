import torch
from torch import Tensor, nn

from .res_net import ResNetHParams, make_res_net

# from hydrantic.model import Model, ModelHparams  # https://github.com/hummerichsander/hydrantic
from flow_matching.solver import ODESolver  # pip install flow-matching
from flow_matching.path import (
    AffineProbPath,
    PathSample,
)
from flow_matching.path.scheduler import (
    CondOTScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
    LinearVPScheduler,
    CosineScheduler,
)
from .utils import expand_like
from typing import Literal, Callable, Type
from abc import ABC
from torch.nn import functional as F
from functools import partial
from fff.utils.utils import sum_except_batch

SCHEDULER_CLASSES = {
    "linear": CondOTScheduler,
    "polynomial": PolynomialConvexScheduler,
    "volume_preserving": VPScheduler,
    "linear_volume_preserving": LinearVPScheduler,
    "trigonometric": CosineScheduler,
}


class FlowMatchingHParams(ResNetHParams):
    interpolation_schedule: str = "linear"
    scheduler_kwargs: dict = {}
    conditional: bool = False
    default_sampling_step_size: float = 1e-2
    default_sampling_method: str = "midpoint"
    loss_norm: str = "l2"
    sigma: float = 0.1  # interpolation noise amplitude


class FlowMatching(nn.Module):
    def __init__(self, hparams):
        if not isinstance(hparams, FlowMatchingHParams):
            hparams = FlowMatchingHParams(**hparams)
        super().__init__()
        self.hparams = hparams

        self.net = self.build_model()
        self.conditional = self.hparams.conditional
        self.path = self._get_path()

    def _get_path(self) -> Type[AffineProbPath]:
        scheduler_cls = SCHEDULER_CLASSES.get(self.hparams.interpolation_schedule)
        if scheduler_cls is None:
            raise ValueError(
                f"Interpolation schedule '{self.hparams.interpolation_schedule}' not found. "
                f"Available schedules: {list(SCHEDULER_CLASSES.keys())}"
            )
        return AffineProbPath(scheduler_cls(**self.hparams.scheduler_kwargs))

    def build_model(self) -> nn.Module:
        time_dim = 1  # no time embedding for now
        input_dim = self.hparams.data_dim + time_dim
        if self.hparams.conditional:
            input_dim += self.hparams.cond_dim
        output_dim = self.hparams.data_dim

        net = make_res_net(
            input_dim,
            self.hparams.layers_spec,
            self.hparams.activation,
            id_init=self.hparams.id_init,
            batch_norm=self.hparams.batch_norm,
            dropout=self.hparams.dropout,
        )
        net.append(nn.Linear(input_dim, output_dim))
        return net

    def get_vector_field(self, x: Tensor, t: Tensor) -> Tensor:
        t = expand_like(t, x[..., :1])
        x = torch.cat([x, t], dim=-1)
        vf = self.net(x)
        return vf

    def get_path_sample(self, t: Tensor, x0: Tensor, x1: Tensor) -> PathSample:
        return self.path.sample(t=t, x_0=x0, x_1=x1)

    def _norm(self, a: Tensor, b: Tensor) -> Tensor:
        if self.hparams.loss_norm == "l2":
            return torch.norm(a - b, p=2, dim=-1)
        elif self.hparams.loss_norm == "l1":
            return torch.norm(a - b, p=1, dim=-1)
        else:
            raise ValueError(f"Unknown norm {self.hparams.loss_norm}")

    def compute_fm_loss(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        # Compute the path sample
        path_sample = self.get_path_sample(t, x0, x1)
        # Compute the vector field
        vf = self.get_vector_field(path_sample.x_t, t)
        # Compute the loss
        return self._norm(vf, path_sample.dx_t)

    def encode(
        self, x: Tensor, c: Tensor, step_size: int = -1, method: str = "default"
    ) -> Tensor:
        if step_size == -1:
            step_size = self.hparams.default_sampling_step_size
        if method == "default":
            method = self.hparams.default_sampling_method

        if self.conditional:
            x = torch.cat([x, c], -1)
        solver = ODESolver(self.get_vector_field)
        sol = solver.sample(
            x_init=x,
            method=method,
            time_grid=torch.tensor([1.0, 0.0]),  # reverse time
            step_size=step_size,
            return_intermediates=True,
        )
        return sol[-1]

    def decode(
        self, z: Tensor, c: Tensor, step_size: int = -1, method: str = "default"
    ) -> Tensor:
        if step_size == -1:
            step_size = self.hparams.default_sampling_step_size
        if method == "default":
            method = self.hparams.default_sampling_method

        if self.conditional:
            z = torch.cat([z, c], -1)
        solver = ODESolver(self.get_vector_field)
        sol = solver.sample(
            x_init=z,
            method=method,
            step_size=step_size,
            return_intermediates=True,
        )
        return sol[-1]

    def sample(self, z: Tensor, c: Tensor) -> Tensor:
        return self.decode(z, c)
