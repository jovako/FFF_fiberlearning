import torch
from torch import Tensor, nn

from .res_net import ResNetHParams, ResNet
from .unet import UNetHParams, UNet

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
from functools import partial
from fff.base import ModelHParams
from .utils import guess_image_shape


SCHEDULER_CLASSES = {
    "linear": CondOTScheduler,
    "polynomial": PolynomialConvexScheduler,
    "volume_preserving": VPScheduler,
    "linear_volume_preserving": LinearVPScheduler,
    "trigonometric": CosineScheduler,
}

INPUT_DIM_BY_ARCHITECTURE = {
    "resnet": 2,
    "unet": 4,
}


class FlowMatchingHParams(ModelHParams):
    interpolation_schedule: str = "linear"
    scheduler_kwargs: dict = {}
    conditional: bool = False
    default_sampling_step_size: float = 1e-2
    default_sampling_method: str = "midpoint"
    loss_norm: str = "l2"
    regress_to_condition: bool = False
    network_hparams: dict = {}
    architecture: str = "resnet"


class FlowMatching(nn.Module):
    def __init__(self, hparams):
        if not isinstance(hparams, FlowMatchingHParams):
            hparams = FlowMatchingHParams(**hparams)
        super().__init__()
        self.hparams = hparams

        self.net = self.build_model()
        self.conditional = self.hparams.conditional
        assert not (
            self.hparams.conditional and self.hparams.regress_to_condition
        ), "Cannot regress to condition and use it as extra input at the same time"
        self.path = self._get_path()

    def _get_path(self) -> AffineProbPath:
        scheduler_cls = SCHEDULER_CLASSES.get(self.hparams.interpolation_schedule)
        if scheduler_cls is None:
            raise ValueError(
                f"Interpolation schedule '{self.hparams.interpolation_schedule}' not found. "
                f"Available schedules: {list(SCHEDULER_CLASSES.keys())}"
            )
        return AffineProbPath(scheduler_cls(**self.hparams.scheduler_kwargs))

    def build_model(self) -> nn.Module:
        time_dim = 1  # no time embedding for now
        output_dim = self.hparams.data_dim
        if self.hparams.architecture == "resnet":
            network_hparams = ResNetHParams(
                data_dim=self.hparams.data_dim,
                cond_dim=self.hparams.cond_dim + time_dim,
                latent_dim=output_dim,
                **self.hparams.network_hparams,
            )
            return ResNet(network_hparams).model.encoder
        elif self.hparams.architecture == "unet":
            network_hparams = UNetHParams(
                data_dim=self.hparams.data_dim,
                cond_dim=self.hparams.cond_dim,
                latent_dim=output_dim,
                **self.hparams.network_hparams,
            )
            return UNet(network_hparams).model.encoder
        else:
            raise ValueError(f"Unknown architecture {self.hparams.architecture}")

    def get_vector_field_image(self, x: Tensor, t: Tensor) -> Tensor:
        assert (
            x.ndim == 4
        ), "Input must be a 4D tensor (batch_size, channels, height, width)"
        return self.net(x, t).sample

    def get_vector_field_flat(self, x: Tensor, t: Tensor) -> Tensor:
        assert x.ndim == 2, "Input must be a 2D tensor (batch_size, features)"
        t = expand_like(t, x[:, :1])
        x = torch.cat([x, t], dim=1)
        vf = self.net(x)
        return vf

    def get_vector_field(self, x: Tensor, t: Tensor) -> Tensor:
        if INPUT_DIM_BY_ARCHITECTURE[self.hparams.architecture] == 4:
            x = x.reshape(x.shape[0], -1, *guess_image_shape(self.hparams.data_dim)[1:])
            return self.get_vector_field_image(x, t).flatten(1)
        elif INPUT_DIM_BY_ARCHITECTURE[self.hparams.architecture] == 2:
            return self.get_vector_field_flat(x, t)
        else:
            raise ValueError(
                f"Unknown input dimension for architecture {self.hparams.architecture}"
            )

    def get_vector_field_conditional(
        self, x: Tensor, t: Tensor, c: Tensor, reverse=False
    ) -> Tensor:
        if self.conditional:
            if INPUT_DIM_BY_ARCHITECTURE[self.hparams.architecture] == 4:
                x = x.reshape(x.shape[0], *guess_image_shape(self.hparams.data_dim))
                try:
                    c = c.reshape(
                        c.shape[0],
                        self.hparams.cond_dim,
                        *guess_image_shape(self.hparams.data_dim)[1:],
                    )
                except Exception as e:
                    c = c[..., None, None] * torch.ones(
                        x.shape[0],
                        self.hparams.cond_dim,
                        *x.shape[2:],
                        device=x.device,
                    )
            x = torch.cat([x, c], dim=1)
        return (
            self.get_vector_field(x, t) if not reverse else -self.get_vector_field(x, t)
        )

    def get_path_sample(self, t: Tensor, x0: Tensor, x1: Tensor) -> PathSample:
        return self.path.sample(t=t, x_0=x0, x_1=x1)

    def _norm(self, a: Tensor, b: Tensor) -> Tensor:
        if self.hparams.loss_norm == "l2":
            return torch.norm(a - b, p=2, dim=-1)
        elif self.hparams.loss_norm == "l1":
            return torch.norm(a - b, p=1, dim=-1)
        else:
            raise ValueError(f"Unknown norm {self.hparams.loss_norm}")

    def compute_fm_loss(self, t: Tensor, x0: Tensor, x1: Tensor, c: Tensor) -> Tensor:
        # Compute the path sample
        path_sample = self.get_path_sample(t, x0, x1)
        # Compute the vector field
        vf = self.get_vector_field_conditional(path_sample.x_t, path_sample.t, c)
        # Compute the loss
        return self._norm(vf, path_sample.dx_t)

    def encode(
        self,
        x: Tensor,
        c: Tensor,
        step_size: int = -1,
        method: str = "default",
        enable_grad=False,
    ) -> Tensor:
        if step_size == -1:
            step_size = self.hparams.default_sampling_step_size
        if method == "default":
            method = self.hparams.default_sampling_method

        vector_field = partial(self.get_vector_field_conditional, c=c)
        solver = ODESolver(vector_field)
        sol = solver.sample(
            x_init=x,
            method=method,
            time_grid=torch.tensor([1.0, 0.0]),  # reverse time
            step_size=step_size,
            return_intermediates=True,
            enable_grad=enable_grad,
        )
        return sol[-1]

    def decode(
        self,
        z: Tensor,
        c: Tensor,
        step_size: int = -1,
        method: str = "default",
        enable_grad=False,
    ) -> Tensor:
        if step_size == -1:
            step_size = self.hparams.default_sampling_step_size
        if method == "default":
            method = self.hparams.default_sampling_method

        vector_field = partial(self.get_vector_field_conditional, c=c)
        solver = ODESolver(vector_field)
        sol = solver.sample(
            x_init=z,
            method=method,
            step_size=step_size,
            return_intermediates=True,
            enable_grad=enable_grad,
        )
        return sol[-1]

    def sample(self, z: Tensor, c: Tensor, **kwargs) -> Tensor:
        return self.decode(z, c, **kwargs)

    def sample_with_guidance(
        self,
        z: Tensor,
        c: Tensor,
        null_condition: Tensor,
        guidance_scale: float = 1.0,
        **kwargs,
    ):
        if guidance_scale == 0:
            return self.sample(z, null_condition, **kwargs)
        elif guidance_scale == 1:
            return self.sample(z, c, **kwargs)
        else:

            def guided_vector_field(x: Tensor, t: Tensor) -> Tensor:
                vf_c = self.get_vector_field_conditional(x, t, c)
                vf_null = self.get_vector_field_conditional(x, t, null_condition)
                return vf_null + guidance_scale * (vf_c - vf_null)

            solver = ODESolver(guided_vector_field)
            sol = solver.sample(
                x_init=z,
                method=self.hparams.default_sampling_method,
                step_size=self.hparams.default_sampling_step_size,
                return_intermediates=True,
                enable_grad=kwargs.get("enable_grad", False),
            )
            return sol[-1]
