from collections import namedtuple, defaultdict
from copy import deepcopy
from importlib import import_module
from math import prod, log10
import ldctinv.pretrained

import torch
import lightning_trainable
from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.hparams import HParams
from lightning_trainable.trainable.trainable import auto_pin_memory, SkipBatch
from torch.distributions import Independent, Normal
from torch.nn import Sequential, CrossEntropyLoss
from torch.utils.data import DataLoader, IterableDataset

from fff.base_model import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult
import fff.data
from fff.distributions.learnable import *
from fff.distributions.multivariate_student_t import MultivariateStudentT
from fff.loss import volume_change_surrogate
from fff.model.utils import TrainWallClock
from fff.utils.jacobian import compute_jacobian
from fff.utils.diffusion import make_betas

class FiberModelHParams(FreeFormBaseHParams):
    lossless_ae: list
    density_model: dict = {}
    load_lossless_ae_path: bool | str = False
    load_density_model_path: bool | str = False
    load_subject_model: bool = False
    train_lossless_ae: bool = True
    train_density_model: bool = True
    vae: bool = False
    betas_max: float = 0.2
    beta_schedule: str = "linear"

    eval_all: bool = True
    fiber_loss_every: int = 1
    cnew_every: int = 1 #deprecated and not used anymore

    warm_up_fiber: int | list = 0
    warm_up_epochs: int | list = 0


class FiberModel(FreeFormBase):
    """
    This class abstracts the functionalities of a model which learns
    the fibers of a "subject model".
    """
    hparams: FiberModelHParams

    def __init__(self, hparams: FreeFormBaseHParams | dict):
        super().__init__(hparams)
        """
        elif self.classification:
        self._data_cond_dim = data_sample[1].shape[0]
        self.cross_entropy = CrossEntropyLoss(reduction='none', label_smoothing=0.2)
        """

        # Ask whether the latent variebles should be passed by another learning model and which model class to use
        if self.hparams.density_model:
            if (self.hparams.density_model.name in [
                    "fff.model.InjectiveFlow", "fff.model.MultilevelFlow",
                    "fff.model.DenoisingFlow"]):
                self.density_model_name = "inn"
            elif self.hparams.density_model.name == "fff.model.DiffusionModel":
                self.density_model_name = "diffusion"
                self.betas = make_betas(1000, self.hparams.betas_max, self.hparams.beta_schedule)
                self.alphas_ = torch.cumprod((1 - self.betas), axis=0)
                print(self.alphas_.shape)
                self.sample_steps = torch.linspace(0, 1, 1000).flip(0)
            else:
                self.density_model_name = "fif"
        else:
            self.density_model_name = False
        # Check whether self.lossless_ae is a VAE
        self.vae = self.hparams.vae
        try:
            if self.hparams.lossless_ae[1]["name"] == "fff.model.VarResNet":
                self.vae = True
        except: 
            self.vae = False

        # Build models
        # First the lossless vae
        """
        CT_nets, _ = ldctinv.pretrained.load_pretrained("cnn10")
        self.lossless_ae = Sequential(CT_nets["vae"])
        """
        self.lossless_ae = build_model(self.hparams.lossless_ae, self.data_dim, 0)
        if self.hparams.load_lossless_ae_path:
            print("load lossless_ae checkpoint")
            checkpoint = torch.load(self.hparams.load_lossless_ae_path)
            lossless_ae_weights = {k[12:]: v for k, v in checkpoint["state_dict"].items()
                              if k.startswith("lossless_ae.")}
            self.lossless_ae.load_state_dict(lossless_ae_weights)

        # Build density_model that operates in the latent space of the lossless_ae
        if self.density_model_name:
            print("Only the density_model will be trained while the model is kept fixed!")
            print("Also the noise is added only to the latent variables!")
            self.density_model = build_model([self.hparams.density_model],
                                                self.lossless_ae[-1].hparams.latent_dim,
                                                self._data_cond_dim)[0]
            if self.hparams.load_density_model_path:
                print("load density_model checkpoint")
                checkpoint = torch.load(self.hparams.load_density_model_path)
                density_model_weights = {k[16:]: v for k, v in checkpoint["state_dict"].items()
                                  if k.startswith("density_model.")}
                self.density_model.load_state_dict(density_model_weights)

        # Add learnable parameter for standard deviation for vae training
        self.lamb = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def _make_latent(self, name, device, **kwargs):
        if name == "normal":
            loc = torch.zeros(self.latent_dim, device=device)
            scale = torch.ones(self.latent_dim, device=device)

            return Independent(
                Normal(loc, scale), 1,
            )
        elif name == "student_t":
            df = self.hparams.latent_distribution["df"] * torch.ones(1, device=device)
            return MultivariateStudentT(df, self.latent_dim)
        #DiffTransformedDistribution(self.density_model, self._make_latent("normal", device), self.betas, 1000, eta=0.1)
        else:
            raise ValueError(f"Unknown latent distribution: {name!r}")

    @property
    def latent_dim(self):
        if self.density_model_name:
            return self.density_model[-1].hparams.latent_dim
        else:
            return self.lossless_ae[-1].hparams.latent_dim

    def encode(self, x, c):
        for model in self.lossless_ae:
            x = model.encode(x, c)
        if self.vae:
            # VAE latent sampling
            mu, logvar = x
            epsilon = torch.randn_like(logvar).to(mu.device)
            z = mu + torch.exp(0.5 * logvar) * epsilon
            x = z, mu, logvar
        return x

    def decode(self, z, c):
        for model in self.lossless_ae[::-1]:
            z = model.decode(z, c)
        return z

    def _encoder_jac(self, x, c, **kwargs):
        if self.density_model_name:
            return compute_jacobian(
                x, self.density_model.encode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs
            )
        else:
            return compute_jacobian(
                x, self.encode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs
            )

    def _decoder_jac(self, z, c, **kwargs):
        if self.density_model_name:
            return compute_jacobian(
                z, self.density_model.decode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs
            )
        else:
            return compute_jacobian(
                z, self.decode, c,
                chunk_size=self.hparams.exact_chunk_size,
                **kwargs
            )

    def forward(self, x, c):
        return self.decode(self.encode(x, c), c)

    def _latent_log_prob(self, z, c):
        try:
            return self.get_latent(z.device).log_prob(z, c)
        except TypeError:
            return self.get_latent(z.device).log_prob(z)

    def sample(self, sample_shape, condition=None):
        """
        Sample via the decoder.
        """
        # sample first via the density_model, if included in the latent distribution
        try:
            z = self.get_latent(self.device).sample(sample_shape, condition)
        except TypeError:
            z = self.get_latent(self.device).sample(sample_shape)
        z = z.reshape(prod(sample_shape), *z.shape[len(sample_shape):])
        batch = [z]
        if condition is not None:
            batch.append(condition)
        if self.density_model_name:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        else:
            c = self.apply_conditions(batch).condition
        x = self.decode(z, c)
        return x.reshape(sample_shape + x.shape[1:])

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"
        
        if self.density_model_name:
            out = volume_change_surrogate(
                x,
                lambda _x: self.density_model.encode(_x, c),
                lambda z: self.density_model.decode(z, c),
                **kwargs
            )
        else:
            encoder_intermediates = []
            decoder_intermediates = []

            def wrapped_encode(x):
                z, intermediates = self.encode(x, c, intermediate=True)
                encoder_intermediates.extend(intermediates)
                return z

            def wrapped_decode(z):
                x, intermediates = self.decode(z, c, intermediate=True)
                decoder_intermediates.extend(intermediates)
                return x

            out = volume_change_surrogate(
                x,
                wrapped_encode,
                wrapped_decode,
                **kwargs
            )
            )
        volume_change = out.surrogate

        out.regularizations.update(self.intermediate_reconstructions(decoder_intermediates, encoder_intermediates))

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    # Computaion of cumulative density function as distance measure for loss
    def _cdf_loss(self, a, b):
        fl =  torch.squeeze(torch.abs(self.teacher_normal.cdf(a) - self.teacher_normal.cdf(b)))
        return fl

    def _reconstruction_loss(self, a, b):
        #return (torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)) ** self.lamb - torch.log(self.lamb)
        if self.vae and not self.density_model_name:
            return (torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)) / self.lamb + torch.log(self.lamb)
        else:
            return torch.sqrt(torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1))
    
    def _sqr_reconstruction_loss(self, a, b):
        return torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)

    def _reduced_rec_loss(self, a, b):
        return torch.sqrt(torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1) / float(a.shape[-1]))

    def _l1_loss(self, a, b):
        return torch.sum(torch.abs(a - b).reshape(a.shape[0], -1), -1)


    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Computes the metrics for the given batch.

        Rationale:
        - In training, we only compute the terms that are actually used in the loss function.
        - During validation, all possible terms and metrics are computed.

        :param batch:
        :param batch_idx:
        :return:
        """
        val_all_metrics = not self.training and self.hparams.eval_all
        conditioned = self.apply_conditions(batch)
        loss_weights = conditioned.loss_weights
        x = conditioned.x_noisy
        #x = torch.clamp(x, min=1e-4,max=1.-1e-5)
        c = conditioned.condition
        x0 = conditioned.x0
        #x0 = torch.clamp(x, min=1e-4,max=1.-1e-5)
        deq_vol_change = conditioned.dequantization_jac

        loss_values = {}
        metrics = {}

        def check_keys(*keys):
            return any(
                (loss_key in loss_weights)
                and
                (
                    torch.any(loss_weights[loss_key] > 0)
                    if torch.is_tensor(loss_weights[loss_key]) else
                    loss_weights[loss_key] > 0
                )
                for loss_key in keys
            )

        # For classification use c as targets
        # or when the conditions are meant only for the density_model
        c_full = c.clone()
        if self.classification or self.density_model_name:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)

        # Empty until computed
        x1 = z = z1 = None
        # Negative log-likelihood
        # exact
        if (not self.density_model_name or self.density_model_name == "fif") and (self.hparams.eval_all and (
                not self.training or (self.hparams.exact_train_nll_every is not None and
                batch_idx % self.hparams.exact_train_nll_every == 0))):
            key = "nll_exact" if self.training else "nll"
            # todo unreadable
            if self.training or (self.hparams.skip_val_nll is not True and (self.hparams.skip_val_nll is False or (
                    isinstance(self.hparams.skip_val_nll, int)
                    and batch_idx < self.hparams.skip_val_nll
            ))):
                with torch.no_grad():
                    if self.density_model_name:
                        z = self.encode(x, c)
                        if self.vae:
                            z, _, __ = z
                        log_prob_result = self.exact_log_prob(x=z, c=c_full, jacobian_target="both")
                        z_dense = log_prob_result.z
                        z1 = log_prob_result.x1
                    else:
                        log_prob_result = self.exact_log_prob(x=x, c=c, jacobian_target="both")
                        z = z_dense = log_prob_result.z
                        x1 = log_prob_result.x1
                loss_values[key] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)
            else:
                loss_weights["nll"] = 0
        
        # surrogate
        if (not self.density_model_name or self.density_model_name == "fif") and self.training and check_keys("nll"):
            warm_up = self.hparams.warm_up_epochs
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            warm_up = map(lambda x: x * self.hparams.max_epochs // 100, warm_up)
            nll_start, warm_up_end = warm_up
            if nll_start == 0:
                nll_warmup = 1
            else:
                nll_warmup = soft_heaviside(
                    self.current_epoch + batch_idx / len(
                        self.trainer.train_dataloader
                        if self.training else
                        self.trainer.val_dataloaders
                    ),
                    nll_start, warm_up_end
                )
            loss_weights["nll"] *= nll_warmup
            if check_keys("nll"):
                if self.density_model_name:
                    z = self.encode(x, c)
                    if self.vae:
                        z, _, __ = z
                    z = z + torch.randn_like(z) * self.hparams.noise
                    log_prob_result = self.surrogate_log_prob(x=z.detach(), c=c_full)
                    z_dense = log_prob_result.z
                    z1 = log_prob_result.x1
                else:
                    log_prob_result = self.surrogate_log_prob(x=x, c=c)
                    x1 = log_prob_result.x1
                    z = z_dense = log_prob_result.z
                loss_values["nll"] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)


        # In case they were skipped above
        if z is None:
            z = self.encode(x, c)
            if self.vae:
                z, mu, logvar = z
            if self.density_model_name == "diffusion":
                t = torch.randint(0, 1000, (z.size(0),), device=z.device).long()
                z_diff, epsilon = self.diffuse(z, t, self.alphas_.to(z.device))
            elif self.density_model_name:
                # Add noise on latent variables
                z = z + torch.randn_like(z) * self.hparams.noise
            z_dense = z
        if x1 is None:
            x1 = self.decode(z, c)
        if self.classification:
            x1 = self.decode(z.detach(),c)



        # Diffusion model
        if check_keys("diff_mse") and self.density_model_name == "diffusion":
            epsilon_pred = self.density_model(z_diff.detach(), t, c_full)
            loss_values["diff_mse"] = self._reconstruction_loss(epsilon_pred, epsilon.detach())

        # KL-Divergence for VAE
        if check_keys("kl") and self.vae:
            loss_values["kl"] = -0.5 * torch.sum((1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar)), -1)

        # NLL loss for INN-architectures
        if ((val_all_metrics or check_keys("nll") or check_keys("coarse_supervised"))
                and self.density_model_name == "inn"):
            z_detach = z.detach()
            if check_keys("coarse_supervised"):
                c_full_n = c_full + torch.randn_like(c_full) * self.hparams.noise
            else: 
                c_full_n = c_full
            if (not check_keys("nll") and check_keys("coarse_supervised")):
                z_dense, _ = self.density_model.encode(z_detach, c_full_n)
            else:
                log_prob, log_det, z_dense = self._latent_log_prob(z_detach, c_full_n)
                loss_values["nll"] = -(log_prob + log_det)
            if isinstance(z_dense, tuple):
                z_dense, z_coarse = z_dense
                if check_keys("coarse_supervised"):
                    loss_values["coarse_supervised"] = self._reconstruction_loss(c_full, z_coarse)
            if check_keys("latent_reconstruction") or not self.training:
                if self.hparams.mask_dims==0:
                    z1 = self.density_model.decode(z_dense, c_full) 
                else:
                    latent_mask = torch.ones(x.shape[0], self.latent_dim, device=x.device)
                    latent_mask[:, -self.hparams.mask_dims:] = 0
                    z_masked_dense = z_dense * latent_mask
                    z1 = self.density_model.decode(z_masked_dense, c_full) 

        
        if ((self.density_model_name and not self.density_model_name=="diffusion") and 
                (val_all_metrics or check_keys("latent_reconstruction"))):
            if z1 is None:
                z_dense = self.density_model.encode(z.detach(), c_full)
                z1 = self.density_model.decode(z_dense, c_full)
            loss_values["latent_reconstruction"] = self._reconstruction_loss(z.detach(), z1)

        # Wasserstein distance of marginal to Gaussian
        with torch.no_grad():
            z_marginal = z_dense.reshape(-1)
            z_gauss = torch.randn_like(z_marginal)

            z_marginal_sorted = z_marginal.sort().values
            z_gauss_sorted = z_gauss.sort().values

            metrics["z 1D-Wasserstein-1"] = (z_marginal_sorted - z_gauss_sorted).abs().mean()
            metrics["z std"] = torch.std(z_marginal)

        if val_all_metrics or check_keys("z std"):
            z_details = z_dense[:, :-1]
            std = torch.mean(torch.abs(torch.std(z_details, dim=0) - 1))
            loss_values["z std"] = torch.ones_like(x[:,0]) * std

        # Classification
        if check_keys("classification"):
            loss_values["classification"] = self.cross_entropy(z,c_full.float())
            if not self.training:
                oneminusacc = 0.5 * torch.sum(
                    torch.abs(c_full - torch.nn.functional.one_hot(torch.argmax(z,dim=1), num_classes=10)), 
                    dim=1)
                loss_values["accuracy"] = 1 - oneminusacc

        # Reconstruction
        if val_all_metrics or check_keys("reconstruction", "noisy_reconstruction", "sqr_reconstruction"):
            loss_values["reconstruction"] = self._reconstruction_loss(x0, x1)
            loss_values["noisy_reconstruction"] = self._reconstruction_loss(x, x1)
            loss_values["sqr_reconstruction"] = self._sqr_reconstruction_loss(x, x1)
            #loss_values["reconstruction"] = self._l1_loss(x0, x1)

        if val_all_metrics or check_keys("masked_reconstruction"):
            latent_mask = torch.zeros(z.shape[0], self.latent_dim, device=z.device)
            latent_mask[:, 0] = 1
            if (self.density_model_name and not self.density_model_name=="diffusion"):
                z_masked_dense = z_dense * latent_mask
                z_masked = self.density_model.decode(z_masked_dense, c_full) 
            else:
                z_masked = z * latent_mask
            x_masked = self.decode(z_masked, c)
            loss_values["masked_reconstruction"] = self._reconstruction_loss(x, x_masked)

        # Cyclic consistency of latent code -- gradient only to encoder
        if val_all_metrics or check_keys("z_reconstruction_encoder"):
            # Not reusing x1 from above, as it does not detach z
            if self.density_model_name in ["fif"]:
                z1_detached = z1.detach()
                z1_dense = self.density_model.encode(z1_detached, c_full)
                loss_values["z_reconstruction_encoder"] = self._reconstruction_loss(z_dense, z1_dense)
            else:
                x1_detached = x1.detach()
                z1 = self.encode(x1_detached, c)
                if self.vae:
                    z1 = z1[0]
                loss_values["z_reconstruction_encoder"] = self._reconstruction_loss(z, z1)

        # Cyclic consistency of latent code sampled from Gaussian and fiber loss
        if ((val_all_metrics or
                check_keys("fiber_loss", "z_sample_reconstruction")) and 
                self.current_epoch % self.hparams.fiber_loss_every == 0):
            warm_up = self.hparams.warm_up_fiber
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            warm_up = map(lambda x: x * self.hparams.max_epochs // 100, warm_up)
            fl_start, warm_up_end = warm_up
            if fl_start == 0:
                fl_warmup = 1
            else:
                fl_warmup = soft_heaviside(
                    self.current_epoch + batch_idx / len(
                        self.trainer.train_dataloader
                        if self.training else
                        self.trainer.val_dataloaders
                    ),
                    fl_start, warm_up_end
                )
            loss_weights["z_sample_reconstruction"] *= fl_warmup
            loss_weights["fiber_loss"] *= fl_warmup
            if val_all_metrics or check_keys(
                    "fiber_loss", "z_sample_reconstruction"):
                try:
                    z_random = self.get_latent(z.device).sample((z.shape[0],), c_full)
                except TypeError:
                    z_random = self.get_latent(z.device).sample((z.shape[0],))
                if isinstance(z_random, tuple):
                    z_random, c_random = z_random
                else:
                    c_random = c
                x_random = self.decode(z_random, c_random)
                # Try whether the model learns fibers and therefore has a subject model
                try:
                    # There might be no subject model
                    cT = torch.empty(x_random.shape[0],0).to(x_random.device)
                    c1 = ((self.subject_model.encode(x_random, cT) - self.data_shift)
                          / self.data_scale)
                    loss_values["fiber_loss"] = self._reduced_rec_loss(
                        c_full, c1)
                except:
                    loss_values["fiber_loss"] = (
                        float("nan") * torch.ones(z_random.shape[0]))
                try:
                    # Sanity checks might fail for random data
                    z1_random = self.encode(x_random, c_random)
                    if self.vae:
                        z1_random, _, __ = z1_random
                    loss_values["z_sample_reconstruction"] = self._reconstruction_loss(
                        z_random, z1_random)
                except:
                    loss_values["z_sample_reconstruction"] = (
                        float("nan") * torch.ones(z_random.shape[0]))

        """
        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("x_sample_reconstruction"):
            # As we only care about the reconstruction, can ignore noise scale
            x_random = self.get_latent(z.device).sample((z.shape[0],))
            if isinstance(x_random, tuple):
                x_random, c_random = x_random
            else:
                c_random = c
            try:
                # Sanity checks might fail for random data
                x1_random = self.decode(self.encode(x_random, c_random), c_random)
                loss_values["x_sample_reconstruction"] = self._reconstruction_loss(x_random, x1_random)
            except:
                loss_values["x_sample_reconstruction"] = float("nan") * torch.ones(x_random.shape[0])

        # Reconstruction of Gauss with double std -- for invertibility
        if not self.training or check_keys("shuffled_reconstruction"):
            # Make noise scale independent of applied noise, reconstruction should still be fine
            x_shuffled = x[torch.randperm(x.shape[0])]
            z_shuffled = self.encode(x_shuffled, c)
            x_shuffled1 = self.decode(z_shuffled, c)
            loss_values["shuffled_reconstruction"] = self._reconstruction_loss(x_shuffled, x_shuffled1)
        """

        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in loss_weights.items()
            if check_keys(key) and (self.training or key in loss_values)
        )

        #metrics["lambda"] = self.lamb

        # Metrics are averaged, non-weighted loss_values
        invalid_losses = []
        for key, weight in loss_values.items():
            # One value per key
            if loss_values[key].shape != (x.shape[0],):
                invalid_losses.append(key)
            else:
                metrics[key] = loss_values[key].mean(-1)
        if len(invalid_losses) > 0:
            raise ValueError(f"Invalid loss shapes for {invalid_losses}")

        # Store loss weights
        if self.training:
            for key, weight in loss_weights.items():
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                self.log(f"weights/{key}", weight.float().mean())

        # Check finite loss
        if not torch.isfinite(metrics["loss"]) and self.training:
            self.trainer.save_checkpoint("erroneous.ckpt")
            print(f"Encountered nan loss from: {metrics}!")
            raise SkipBatch

        return metrics


    def dequantize(self, batch):
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        noise = self.hparams.noise
        if isinstance(noise, list):
            min_noise, max_noise = noise
            if not self.training:
                max_noise = min_noise
            noise_scale = rand_log_uniform(
                max_noise, min_noise,
                shape=base_cond_shape, device=device, dtype=dtype
            )
            x = x0 + torch.randn_like(x0) * (10 ** noise_scale)
            noise_conds = [noise_scale]
        else:
            if noise > 0 and not self.density_model_name:
                x = x0 + torch.randn_like(x0) * noise
            else:
                x = x0
            noise_conds = []
        return noise_conds, x, torch.zeros(x0.shape[0], device=device, dtype=dtype)

    def diffuse(self, x, t, alphas_):
        noise = torch.randn_like(x)
        alpha_t = alphas_[t].unsqueeze(1)
        alpha_t = alpha_t.repeat([1, x.shape[1]])
        noisy_x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise
        #noisy_x = alpha_t.sqrt() * x + noise
        return noisy_x, noise

    def configure_optimizers(self):
        params = []
        if self.hparams.train_lossless_ae:
            params.extend(list(self.lossless_ae.parameters()))
            if self.vae:
                params.append(self.lamb)
            if self.density_model_name:
                print("WARNING: lossless_ae is not meant to be trained when a density_model is included")
        else:
            print("WARNING: lossless_ae get not trained")
        if self.density_model_name:
            if self.hparams.train_density model:
                params.extend(list(self.density_model.parameters()))
            else:
                print("WARNING: density model gets not trained")
        kwargs = dict()

        match self.hparams.optimizer:
            case str() as name:
                optimizer = lightning_trainable.utils.get_optimizer(name)(
                    params, **kwargs
                )
            case dict() as kwargs:
                name = kwargs.pop("name")
                optimizer = lightning_trainable.utils.get_optimizer(name)(params, **kwargs)
                self.hparams.optimizer["name"] = name
            case type(torch.optim.Optimizer) as Optimizer:
                optimizer = Optimizer(params, **kwargs)
            case torch.optim.Optimizer() as optimizer:
                pass
            case None:
                return None
            case other:
                raise NotImplementedError(f"Unrecognized Optimizer: {other}")

        lr_scheduler = lightning_trainable.trainable.lr_schedulers.configure(
                self, optimizer
                )

        if lr_scheduler is None:
            return optimizer
        
        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )


def build_model(models, data_dim: int, cond_dim: int):
    if not isinstance(models[0], dict):
        return Sequential(*models)
    models = deepcopy(models)
    model = Sequential()
    for model_spec in models:
        module_name, class_name = model_spec.pop("name").rsplit(".", 1)
        model_spec["data_dim"] = data_dim
        model_spec["cond_dim"] = cond_dim
        if model_spec.get("latent_dim", "data") == "data":
            print(module_name, data_dim)
            model_spec["latent_dim"] = data_dim
        model.append(
            getattr(import_module(module_name), class_name)(model_spec)
        )
        data_dim = model_spec["latent_dim"]
    return model


def soft_heaviside(pos, start, stop):
    return max(0., min(
        1.,
        (pos - start)
        /
        (stop - start)
    ))


def rand_log_uniform(vmin, vmax, shape, device, dtype):
    vmin, vmax = map(log10, [vmin, vmax])
    return torch.rand(
        shape, device=device, dtype=dtype
    ) * (vmin - vmax) + vmax


def wasserstein2_distance_gaussian_approximation(x1, x2):
    # Returns the squared 2-Wasserstein distance between the Gaussian approximation of two datasets x1 and x2
    # 1. Calculate mean and covariance of x1 and x2
    # 2. Use fact that tr( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ) = sum(eigvals( ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) ))
    # = sum(eigvals( cov1 cov2 )^(1/2))
    # 3. Return ||m1 - m2||^2 + tr( cov1 + cov2 - 2 ( cov1^(1/2) cov2 cov1^(1/2) )^(1/2) )
    m1 = x1.mean(0)
    m2 = x2.mean(0)
    cov1 = (x1 - m1[None]).T @ (x1 - m1[None]) / x1.shape[0]
    cov2 = (x2 - m2[None]).T @ (x2 - m2[None]) / x2.shape[0]
    cov_product = cov1 @ cov2
    eigenvalues_prod = torch.relu(torch.linalg.eigvals(cov_product).real)
    m_part = torch.sum((m1 - m2) ** 2)
    cov_part = torch.trace(cov1) + torch.trace(cov2) - 2 * torch.sum(torch.sqrt(eigenvalues_prod))
    return m_part + cov_part
