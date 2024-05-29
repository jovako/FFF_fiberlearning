from torch.distributions import Distribution
from torch.nn import Module
import torch


class ModuleDistribution(Module):
    """
    A distribution that is also a module.

    This allows to use parameters of the distribution as parameters of the module.
    The parameters are stored in the module and the distribution is instantiated
    with these parameters. In order to fulfill constraints, you can derive sensible
    values from the parameters in instantiate().
    """
    def __init__(self):
        super().__init__()

    def instantiate(self) -> Distribution:
        raise NotImplementedError()

    def log_prob(self, x):
        return self.instantiate().log_prob(x)

    def rsample(self, sample_shape):
        return self.instantiate().rsample(sample_shape)

    def sample(self, sample_shape):
        return self.instantiate().sample(sample_shape)


class TransformedDistribution():
    def __init__(self, Transform, Distribution, mask_dims):
        self.Trans = Transform
        self.Dist = Distribution
        self.mask_dims = mask_dims

    def sample(self, shape=torch.Size(), c=None):
        samples = self.Dist.sample(shape)
        latent_mask = torch.ones(samples.shape, device=samples.device)
        if self.mask_dims > 0:
            latent_mask[:, -self.mask_dims:] = 0
        samples_coarse = samples * latent_mask
        transformed_samples = self.Trans.decode(samples_coarse, c)
        return transformed_samples

    def log_prob(self, z, c=None):
        z_dense, jac = self.Trans.encode(z, c)
        if isinstance(z_dense, tuple):
            z_details, z_coarse = z_dense
            log_prob = self.Dist.log_prob(z_details)
        else:
            log_prob = self.Dist.log_prob(z_dense)

        return log_prob, jac, z_dense
        

class FIFTransformedDistribution(TransformedDistribution):
    def log_prob(self, z_dense, c=None):
        return self.Dist.log_prob(z_dense)


class DiffTransformedDistribution():
    def __init__(self, model, Distribution, betas, timesteps=1000, eta=0.0):
        self.Diff = model
        self.Dist = Distribution
        self.betas = betas
        self.num_timesteps = timesteps
        self.eta = eta

        # Precompute constants for sampling
        self.alphas = 1.0 - betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), self.alpha_cumprod[:-1]])
        
        self.sqrt_1malpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)

    def sample(self, shape, condition, guidance_scale=2.0):
        x = self.Dist.sample(shape)
        device = x.device
        num_steps = self.num_timesteps

        for i in reversed(range(num_steps)):
            t = torch.full(shape, i, device=device, dtype=torch.long)
            alpha_t = self.alpha_cumprod[i]
            alpha_t_prev = self.alpha_cumprod_prev[i]
            beta_t = self.betas[i]

            # Predict noise
            eps_pred = self.Diff(x, t, condition, guidance_scale)

            # Compute the mean for the reverse process
            pred_x0 = (
                x - self.sqrt_1malpha_cumprod[i] * eps_pred
            ) / alpha_t.sqrt()

            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)
            sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * beta_t)

            dir_xt = (1. - alpha_t_prev - sigma_t**2).sqrt() * eps_pred

            x = alpha_t_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise

        return x
