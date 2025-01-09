from collections import OrderedDict

import torch.nn
from torch import nn

from .auto_encoder import SkipConnection
from fff.base import ModelHParams
from .utils import wrap_batch_norm1d, make_dense, CrossAttention


class DiffHParams(ModelHParams):
    layers_spec: list
    activation: str = "gelu"
    id_init: bool = False
    batch_norm: str | bool = False
    dropout: float | None = None
    num_heads: int = 4
    hidden_dim: int = 32
    time_dim: int = 8
    betas: tuple
    num_timesteps: int = 1000
    eta: float = 0.1

    def __init__(self, **hparams):
        if "latent_spec" in hparams:
            assert len(hparams["latent_spec"]) == 0
            del hparams["latent_spec"]
        super().__init__(**hparams)

def res_layer(data_dim, widths, activation, id_init: bool,
                 batch_norm: str | bool, dropout: float = None):
    return SkipConnection(
        make_dense([data_dim, *widths, data_dim], activation,
                   batch_norm=batch_norm, dropout=dropout),
        id_init=id_init
    )


class DiffusionModel(nn.Module):
    hparams: DiffHParams

    def __init__(self, hparams: dict | DiffHParams):
        if not isinstance(hparams, DiffHParams):
            hparams = DiffHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.build_model()

        # Precompute constants for sampling
        self.betas = self.hparams.betas[0]
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), self.alpha_cumprod[:-1]])
        self.sqrt_1malpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
    def forward(self, x_in, time_step, condition, guidance_scale=1.0, conditional=True):
        # Embed the time step
        time_embedding = self.time_embedding(time_step)
        #condition = torch.zeros_like(x_in)
        
        # Initial input
        x = torch.cat((x_in, time_embedding), dim=-1)
        x = self.fc_in(x)
        
        if conditional:
            # Conditional forward pass
            conditional_x = x
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    conditional_x = conditional_x + layer(conditional_x, condition)
                else:
                    conditional_x = layer(conditional_x)
            conditional_output = self.fc_out(conditional_x)
            
            # Unconditional forward pass (condition replaced with zeros)
            unconditional_x = x
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    unconditional_x = unconditional_x + layer(unconditional_x, torch.zeros_like(condition))
                else:
                    unconditional_x = layer(unconditional_x)
            unconditional_output = self.fc_out(unconditional_x)
            
            # Classifier-free guidance
            output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
            #output = conditional_output
        else:
            # Unconditional forward pass
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    x = x + layer(x, torch.zeros_like(condition))
                else:
                    x = layer(x)
            output = self.fc_out(x)
        #print(torch.sum(x-x_in))
        return output

    def build_model(self):
        input_dim = self.hparams.data_dim
        hidden_dim = self.hparams.hidden_dim
        condition_dim = self.hparams.cond_dim
        #condition_dim = 2
        time_dim = self.hparams.time_dim
        num_heads = self.hparams.num_heads
        activation = self.hparams.activation
        
        # Time embedding layer
        self.time_embedding = nn.Embedding(1000, time_dim)  # Assuming 1000 different time steps
        
        self.fc_in = nn.Linear(input_dim + time_dim, hidden_dim)
        
        # Hidden layers with cross-attention
        self.layers = nn.ModuleList()
        
        for widths in self.hparams.layers_spec:
            self.layers.append(res_layer(hidden_dim, widths, activation,
                id_init=self.hparams.id_init,
                batch_norm=self.hparams.batch_norm, dropout=self.hparams.dropout)
            )
            self.layers.append(CrossAttention(hidden_dim, condition_dim, num_heads))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def sample(self, x, condition, guidance_scale=1.0):
        device = x.device
        num_steps = self.num_timesteps

        for i in reversed(range(num_steps)):
            t = torch.full(shape, i, device=device, dtype=torch.long)
            alpha_t = self.alpha_cumprod[i]
            alpha_t_prev = self.alpha_cumprod_prev[i]
            beta_t = self.betas[i]

            # Predict noise
            eps_pred = self.forward(x, t, condition, guidance_scale)

            # Compute the mean for the reverse process
            pred_x0 = (
                x - self.sqrt_1malpha_cumprod[i] * eps_pred
            ) / alpha_t.sqrt()

            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)
            sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * beta_t)

            dir_xt = (1. - alpha_t_prev - sigma_t**2).sqrt() * eps_pred

            x = alpha_t_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise

        return x
