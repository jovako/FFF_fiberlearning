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
    def __init__(self, model, Distribution, timesteps, eta=0.0):
        self.Diff = model
        self.Dist = Distribution
        self.timesteps = timesteps
        self.eta = eta

    def sample(self, shape=torch.Size(), condition=None, guidance_scale=4.0):
        x = self.Dist.sample(shape)
        num_steps = len(self.timesteps)
        device = x.device

        for i in range(num_steps):
            t = self.timesteps[i]
            t_next = self.timesteps[i + 1] if i + 1 < num_steps else 0

            # Forward pass through the model
            eps = self.Diff(x, torch.tensor([t] * x.size(0), device=device).long(), condition, guidance_scale)

            # Compute alpha values
            alpha_t = (1 - (t / num_steps)) ** 2
            alpha_t_next = (1 - (t_next / num_steps)) ** 2

            # Compute the noise scale
            sigma_t = self.eta * ((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next)) ** 0.5

            # Update the sample
            x = (
                (x - (1 - alpha_t) / (1 - alpha_t) * eps)
                * (alpha_t_next / alpha_t) ** 0.5
                + sigma_t * torch.randn_like(x)
            )

        return x
