import functools
import time
from inspect import isclass
from math import sqrt
from abc import ABC
from typing import Literal, Callable, Type

import FrEIA
import torch
from lightning import Callback
from torch import nn, Tensor
import torch.nn.functional as F


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
    
torch.nn.Sin = Sin
torch.nn.Swish = Swish

def expand_like(input: Tensor, reference: Tensor) -> Tensor:
    """Expands the input tensor to match the shape of the reference tensor.

    :param input: tensor to be expanded.
    :param reference: tensor to match the shape."""

    expanded = input.clone()
    while expanded.dim() < reference.dim():
        expanded = expanded.unsqueeze(-1)
    return expanded.expand_as(reference)

def get_module(name):
    """ Get a nn.Module in a case-insensitive way """
    modules = torch.nn.__dict__
    modules = {
        key.lower(): value for key, value in modules.items()
        if isclass(value) and issubclass(value, torch.nn.Module)
    }

    return modules[name.lower()]


def make_dense(widths: list[int], activation: str, dropout: float = None, batch_norm: str | bool = False):
    """ Make a Dense Network from given layer widths and activation function """
    if len(widths) < 2:
        raise ValueError("Need at least Input and Output Layer.")
    #print(widths)
    Activation = get_module(activation)

    network = nn.Sequential()

    # input is x, time, condition
    for i in range(0, len(widths) - 2):
        if i > 0 and dropout is not None:
            network.add_module(f"Dropout_{i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden_Layer_{i}", hidden_layer)
        if batch_norm is not False:
            network.add_module(f"Batch_Norm_{i}", wrap_batch_norm1d(batch_norm, widths[i + 1]))
        network.add_module(f"Hidden_Activation_{i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output_Layer", output_layer)

    return network


def guess_image_shape(dim):
    if dim == 3 * 38804:
        return 3, 178, 218
    if dim % 3 == 0:
        n_channels = 3
    else:
        n_channels = 1
    size = round(sqrt(dim // n_channels))
    if size ** 2 * n_channels != dim:
        raise ValueError(f"Input is not square: "
                         f"{size} ** 2 != {dim // n_channels}")
    return n_channels, size, size


def subnet_factory(inner_widths, activation, zero_init=True):
    def make_subnet(dim_in, dim_out):
        network = make_dense([
            dim_in,
            *inner_widths,
            dim_out
        ], activation)

        if zero_init:
            network[-1].weight.data.zero_()
            network[-1].bias.data.zero_()

        return network

    return make_subnet


def make_inn(inn_spec, *data_dim, cond_dim=0, zero_init=True, cond=0):
    inn = FrEIA.framework.SequenceINN(*data_dim)
    for inn_layer in inn_spec:
        module_name, module_args, subnet_widths = inn_layer
        if (module_name == "RationalQuadraticSpline" and data_dim[0] == 1):
            print(data_dim, cond_dim)
            module_name = "ElementwiseRationalQuadraticSpline"
        module_class = getattr(FrEIA.modules, module_name)
        extra_module_args = dict()
        if "subnet_constructor" not in module_args and module_name not in ["PermuteRandom", "ActNorm", "InvAutoActTwoSided"]:
            extra_module_args["subnet_constructor"] = subnet_factory(
                subnet_widths, "leakyrelu", zero_init=zero_init
            )
            extra_module_args["cond"] = cond
            extra_module_args["cond_shape"] = (cond_dim,)
        inn.append(module_class, **module_args, **extra_module_args)
    return inn


def batch_wrap(fn):
    """
    Add a batch dimension to each tensor argument.

    :param fn:
    :return:
    """

    def deep_unsqueeze(arg):
        if torch.is_tensor(arg):
            return arg[None, ...]
        elif isinstance(arg, dict):
            return {key: deep_unsqueeze(value) for key, value in arg.items()}
        elif isinstance(arg, (list, tuple)):
            return [deep_unsqueeze(value) for value in arg]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        args = deep_unsqueeze(args)
        return fn(*args, **kwargs)[0]

    return wrapper


class RunningBatchNorm(torch.nn.Module):
    """
    Wrap BatchNorm to normalize only using running mean and std,
    instead of per-batch normalization.
    """

    def __init__(self, train_batch_norm, eval_batch_norm):
        super().__init__()
        self.train_batch_norm = train_batch_norm
        self.eval_batch_norm = eval_batch_norm

    def forward(self, x):
        # Apply BatchNorms, updating their running mean/std
        if self.training:
            try:
                self.train_batch_norm(x)
                if self.train_batch_norm is not self.eval_batch_norm:
                    self.eval_batch_norm(x)
            except ValueError:
                pass

        # Use result from aggregated mean and std for actual computation
        if self.training:
            self.train_batch_norm.eval()
            out = self.train_batch_norm(x)
            self.train_batch_norm.train()
        else:
            try:
                out = self.eval_batch_norm(x)
            except ValueError:
                # In vmap calls, fake a batch dimension
                out = self.eval_batch_norm(x[None])[0]
        return out


class VmapBatchNorm(torch.nn.Module):
    def __init__(self, batch_norm):
        super().__init__()
        self.batch_norm = batch_norm

    def forward(self, x):
        try:
            return self.batch_norm(x)
        except ValueError:
            if self.training:
                raise
            return self.batch_norm(x[None])[0]


def _make_batch_norm(kind):
    @functools.wraps(kind)
    def wrapper(batch_norm_spec: str | bool, *args, **kwargs):
        assert batch_norm_spec is not False
        batch_norm = kind(*args, **kwargs)
        if batch_norm_spec == "no-batch-grads":
            # This mode behaves like traditional BatchNorm, but it ignores
            # gradients between batch entries
            train_batch_norm = kind(*args, momentum=1.0, **kwargs)
            batch_norm = RunningBatchNorm(train_batch_norm, batch_norm)
        elif batch_norm_spec == "running-only":
            # This mode keeps track of running stats
            batch_norm = RunningBatchNorm(batch_norm, batch_norm)
        elif batch_norm_spec == "vmap":
            # This mode uses unpacks vmap tensors
            batch_norm = VmapBatchNorm(batch_norm)
        elif batch_norm_spec is not True:
            raise ValueError(f"{batch_norm_spec=}")
        return batch_norm

    return wrapper


wrap_batch_norm1d = _make_batch_norm(nn.BatchNorm1d)
wrap_batch_norm2d = _make_batch_norm(nn.BatchNorm2d)
wrap_batch_norm3d = _make_batch_norm(nn.BatchNorm3d)


class TrainWallClock(Callback):
    def __init__(self):
        self.batch_start = None
        self.state = {"steps": 0, "time": 0}

    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.batch_start = time.monotonic()

    def on_train_batch_end(self, *args, **kwargs):
        self.state["steps"] += 1
        self.state["time"] += time.monotonic() - self.batch_start
        self.batch_start = None

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()

class CrossAttention(nn.Module):
    def __init__(self, input_dim, condition_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(condition_dim, input_dim)
        self.value = nn.Linear(condition_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)
        self.init = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, condition):
        batch_size = x.size(0)
        # Linear projections
        queries = self.query(x)
        keys = self.key(condition)
        values = self.value(condition)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, -1, self.num_heads, queries.size(-1) // self.num_heads).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, keys.size(-1) // self.num_heads).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, values.size(-1) // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (keys.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.num_heads * (attn_output.size(-1)))
        attn_output = attn_output * self.init

        output = self.out(attn_output)
        return output

class InterpolantMixin(ABC):
    # flow matching util
    class hparams:
        Interpolation = Literal["linear", "trigonometric"]
        sigma: float
        interpolation: Interpolation

    _alpha: Callable[[Tensor], Tensor]
    _alpha_dot: Callable[[Tensor], Tensor]
    _beta: Callable[[Tensor], Tensor]
    _beta_dot: Callable[[Tensor], Tensor]

    def __init__(self, *args, **kwargs):
        self._configure_interpolation()

    def loss_fm(self, bt: Tensor, bt_hat: Tensor) -> Tensor:
        return sum_except_batch(torch.pow(bt_hat, 2)) - 2 * sum_except_batch(bt * bt_hat)

    def loss_sm(self, etat: Tensor, etat_hat: Tensor) -> Tensor:
        return sum_except_batch(torch.pow(etat_hat, 2)) - 2 * sum_except_batch(etat * etat_hat)

    def _configure_interpolation(self) -> None:
        match self.hparams.interpolation:
            case "linear":
                print("Using a linear interpolation")
                self._alpha = lambda t: 1 - t
                self._alpha_dot = lambda t: -torch.ones_like(t)
                self._beta = lambda t: t
                self._beta_dot = lambda t: torch.ones_like(t)
            case "trigonometric":
                print("Using a trigonometric interpolation")
                self._alpha = lambda t: torch.cos(torch.pi / 2 * t)
                self._alpha_dot = lambda t: -torch.pi / 2 * torch.sin(torch.pi / 2 * t)
                self._beta = lambda t: torch.sin(torch.pi / 2 * t)
                self._beta_dot = lambda t: torch.pi / 2 * torch.cos(torch.pi / 2 * t)
            case _:
                raise ValueError(f"Interpolation '{self.hparams.interpolation}' not supported")

    def _gamma(self, t: Tensor) -> Tensor:
        return (t * (1 - t)) * self.hparams.sigma

    def _gamma_dot(self, t: Tensor) -> Tensor:
        return (1 - 2 * t) * self.hparams.sigma

    def compute_path_sample(self, t: Tensor, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = torch.randn_like(x0)
        xt = self.compute_xt(t, x0, x1, z)
        bt = self.compute_bt(t, x0, x1, z)
        return xt, bt, z

    def compute_xt(self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor) -> Tensor:
        alpha_t = expand_like(self._alpha(t), x0)
        beta_t = expand_like(self._beta(t), x1)
        gamma_t = expand_like(self._gamma(t), z)
        return alpha_t * x0 + beta_t * x1 + gamma_t * z

    def compute_bt(self, t: Tensor, x0: Tensor, x1: Tensor, z: Tensor) -> Tensor:
        alpha_t_dot = expand_like(self._alpha_dot(t), x0)
        beta_t_dot = expand_like(self._beta_dot(t), x1)
        gamma_t_dot = expand_like(self._gamma_dot(t), z)
        return alpha_t_dot * x0 + beta_t_dot * x1 + gamma_t_dot * z
