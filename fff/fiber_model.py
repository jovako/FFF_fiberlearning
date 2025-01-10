from copy import deepcopy
from importlib import import_module
from math import prod, log10
# import ldctinv.pretrained
from warnings import warn

import torch
import lightning_trainable
from lightning_trainable.trainable.trainable import auto_pin_memory, SkipBatch
from torch.distributions import Independent, Normal
from torch.nn import Sequential, CrossEntropyLoss

from fff.base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult, build_model
from fff.base import wasserstein2_distance_gaussian_approximation, rand_log_uniform, soft_heaviside
from fff.base import LogProbResult
from fff.lossless_ae import LosslessAE
from fff.subject_model import SubjectModel
from fff.loss import volume_change_surrogate
from fff.utils.jacobian import compute_jacobian
from fff.utils.diffusion import make_betas

class FiberModelHParams(FreeFormBaseHParams):
    lossless_ae: list
    density_model: list
    load_lossless_ae_path: bool | str = False
    load_density_model_path: bool | str = False
    load_subject_model: bool = False
    train_lossless_ae: bool = True
    ae_conditional: bool = False
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
        
        # Add learnable parameter for standard deviation for vae training
        self.lamb = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def init_models(self):
        # Ask whether the latent variebles should be passed by another learning model and which model class to use
        if (self.hparams.density_model[-1]["name"] == "fff.model.Identity"):
            self.density_model_name = None
        elif (self.hparams.density_model[-1]["name"] in [
                "fff.model.InjectiveFlow", "fff.model.MultilevelFlow",
                "fff.model.DenoisingFlow"]):
            self.density_model_name = "inn"
        elif self.hparams.density_model[-1]["name"] == "fff.model.DiffusionModel":
            self.density_model_name = "diffusion"
            self.betas = make_betas(1000, self.hparams.betas_max, self.hparams.beta_schedule)
            self.hparams.density_model[-1]["betas"] = (self.betas,)
            self.alphas_ = torch.cumprod((1 - self.betas), axis=0)
            print(self.alphas_.shape)
            self.sample_steps = torch.linspace(0, 1, 1000).flip(0)
        else:
            self.density_model_name = "fif"
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
        ae_hparams = {}
        ae_hparams["model_spec"] = self.hparams.lossless_ae
        ae_hparams["data_dim"] = self.data_dim
        if self.hparams.ae_conditional:
            ae_hparams["cond_dim"] = self._data_cond_dim
        ae_hparams["vae"] = self.vae
        ae_hparams["path"] = self.hparams.load_lossless_ae_path
        self.lossless_ae = LosslessAE(ae_hparams)

        # Build density_model that operates in the latent space of the lossless_ae
        self.density_model = build_model(self.hparams.density_model,
                                            self.lossless_ae.latent_dim,
                                            self._data_cond_dim)
        if self.hparams.load_density_model_path:
            print("load density_model checkpoint")
            checkpoint = torch.load(self.hparams.load_density_model_path)
            density_model_weights = {k[16:]: v for k, v in checkpoint["state_dict"].items()
                              if k.startswith("density_model.")}
            self.density_model.load_state_dict(density_model_weights)

        if self.hparams.load_subject_model:
            print("loading subject_model")
            data_dir = self.hparams["data_set"]["path"]
            self.subject_model = SubjectModel(f"data/{data_dir}/subject_model/checkpoints/last.ckpt")

    @property
    def latent_dim(self):
        if self.density_model_name:
            return self.density_model[-1].hparams.latent_dim
        else:
            return self.lossless_ae.latent_dim

    def encode_lossless(self, x, c, mu_var=True):
        return self.lossless_ae.encode(x, c, mu_var=mu_var)

    def encode_density(self, z, c, jac=False):
        jacs = []
        for net in self.density_model:
            z = net.encode(z, c)
            if isinstance(z, tuple):
                z, jac_i = z
                jacs.append(jac_i)
        if jac:
            z = z, torch.sum(torch.stack(jacs, dim=1), dim=1)
        return z

    def encode(self, x, c):
        z_dense = self.encode_density(self.lossless_ae.encode(x, c)[0], c)
        return z_dense

    def decode_lossless(self, z, c):
        return self.lossless_ae.decode(z, c)

    def decode_density(self, z_dense, c):
        for net in self.density_model:
            z_dense = net.decode(z_dense, c)
        return z_dense

    def decode(self, z_dense, c):
        x = self.decode_lossless(self.decode_density(z_dense, c), c)
        return x

    def _encoder_jac(self, x, c, **kwargs):
        return compute_jacobian(
            x, self.encode_density, c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs
        )

    def _decoder_jac(self, z, c, **kwargs):
        return compute_jacobian(
            z, self.decode_density, c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs
        )

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        jtj = torch.einsum("bik,bjk->bij", jac_enc, jac_enc)
        log_det = jtj.slogdet()[1] / 2
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        # Forward gradient is faster because latent dimension is smaller than data dimension
        x1, jac_dec = self._decoder_jac(z, c, grad_type="forward", **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        jjt = torch.einsum("bki,bkj->bij", jac_dec, jac_dec)
        log_det = jjt.slogdet()[1] / 2
        return VolumeChangeResult(x1, log_det, {})

    def _latent_log_prob(self, z, c):
        try:
            return self.get_latent(z.device).log_prob(z, c)
        except TypeError:
            return self.get_latent(z.device).log_prob(z)

    def sample(self, sample_shape, condition=None):
        """
        Sample via the density_model and lossless_ae decoder.
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
        c = self.apply_conditions(batch).condition
        for net in self.density_model:
            z = net.sample(z, c)
        x = self.decode_lossless(z, c)
        return x.reshape(sample_shape + x.shape[1:])

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"
        
        out = volume_change_surrogate(
            x,
            lambda _x: self.encode_density(_x, c),
            lambda z: self.decode_density(z, c),
            **kwargs
        )

        volume_change = out.surrogate

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    def _cdf_loss(self, a, b):
        # Computaion of cumulative density function as distance measure for loss
        fl =  torch.squeeze(torch.abs(self.teacher_normal.cdf(a) - self.teacher_normal.cdf(b)))
        return fl

    def _reconstruction_loss(self, a, b):
        return torch.sqrt(torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1))

    def _lamb_reconstruction_loss(self, a, b):
        return (torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)) / self.lamb + torch.log(self.lamb)
    
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
        c = conditioned.condition
        x0 = conditioned.x0
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

        # Empty until computed
        x1 = z = z1 = z_dense = None
        # Negative log-likelihood
        # exact
        if self.density_model_name == "fif" and (self.hparams.eval_all and (
                not self.training or (self.hparams.exact_train_nll_every is not None and
                batch_idx % self.hparams.exact_train_nll_every == 0))):
            key = "nll_exact" if self.training else "nll"
            # todo unreadable
            if self.training or (self.hparams.skip_val_nll is not True and (self.hparams.skip_val_nll is False or (
                    isinstance(self.hparams.skip_val_nll, int)
                    and batch_idx < self.hparams.skip_val_nll
            ))):
                with torch.no_grad():
                    z, mu, logvar = self.encode_lossless(x, c, mu_var=True)
                    log_prob_result = self.exact_log_prob(x=z.detach(), c=c, jacobian_target="both")
                    z_dense = log_prob_result.z
                    z1 = log_prob_result.x1
                loss_values[key] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)
            else:
                loss_weights["nll"] = 0
        
        # surrogate
        if (self.density_model_name == "fif") and self.training and check_keys("nll"):
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
                z, mu, logvar = self.encode_lossless(x, c, mu_var=True)
                z = z + torch.randn_like(z) * self.hparams.noise
                log_prob_result = self.surrogate_log_prob(x=z.detach(), c=c)
                z_dense = log_prob_result.z
                z1 = log_prob_result.x1
                loss_values["nll"] = -log_prob_result.log_prob - deq_vol_change
                loss_values.update(log_prob_result.regularizations)


        # In case they were skipped above
        if z is None:
            z, mu, logvar = self.encode_lossless(x, c, mu_var=True)
            if self.density_model_name == "diffusion":
                t = torch.randint(0, 1000, (z.size(0),), device=z.device).long()
                z_diff, epsilon = self.diffuse(z, t, self.alphas_.to(z.device))
            elif self.density_model_name:
                # Add noise on latent variables
                z = z + torch.randn_like(z) * self.hparams.noise
        if x1 is None:
            x1 = self.decode_lossless(z, c)

        # KL-Divergence for VAE
        if check_keys("kl"):
            loss_values["kl"] = -0.5 * torch.sum((1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar)), -1)

        # Diffusion model
        if check_keys("diff_mse") and self.density_model_name == "diffusion":
            epsilon_pred = self.density_model(z_diff.detach(), t, c)
            loss_values["diff_mse"] = self._reconstruction_loss(epsilon_pred, epsilon.detach())

        # NLL loss for INN-architectures
        if ((val_all_metrics or check_keys("nll") or check_keys("coarse_supervised"))
                and self.density_model_name == "inn"):
            if check_keys("coarse_supervised"):
                c_n = c + torch.randn_like(c_full) * self.hparams.noise
            else: 
                c_n = c
            z_dense, log_det = self.encode_density(z.detach(), c_n, jac=True)
            if isinstance(z_dense, tuple):
                z_dense, z_coarse = z_dense
                if check_keys("coarse_supervised"):
                    loss_values["coarse_supervised"] = self._reconstruction_loss(c, z_coarse)
            log_prob = self._latent_log_prob(z_dense, c_n)
            loss_values["nll"] = -(log_prob + log_det)
                
        if z_dense is None:
            z_dense = self.encode_density(z.detach(), c)

        if (not self.density_model_name=="diffusion" and 
                (val_all_metrics or check_keys("latent_reconstruction"))):
            if z1 is None:
                #if self.hparams.mask_dims==0:
                z1 = self.decode_density(z_dense, c) 
                """
                else:
                    latent_mask = torch.ones(x.shape[0], self.latent_dim, device=x.device)
                    latent_mask[:, -self.hparams.mask_dims:] = 0
                    z_masked_dense = z_dense * latent_mask
                    z1 = self.decode_density(z_masked_dense, c) 
                """
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

        """
        # Classification
        if check_keys("classification"):
            loss_values["classification"] = self.cross_entropy(z,c.float())
            if not self.training:
                oneminusacc = 0.5 * torch.sum(
                    torch.abs(c - torch.nn.functional.one_hot(torch.argmax(z,dim=1), num_classes=10)), 
                    dim=1)
                loss_values["accuracy"] = 1 - oneminusacc
        """

        # Reconstruction
        if val_all_metrics or check_keys(
                "reconstruction", "noisy_reconstruction", 
                "sqr_reconstruction", "lamb_reconstruction"):
            loss_values["reconstruction"] = self._reconstruction_loss(x0, x1)
            loss_values["noisy_reconstruction"] = self._reconstruction_loss(x, x1)
            loss_values["sqr_reconstruction"] = self._sqr_reconstruction_loss(x, x1)
            loss_values["lamb_reconstruction"] = self._lamb_reconstruction_loss(x, x1)
            #loss_values["reconstruction"] = self._l1_loss(x0, x1)

        if val_all_metrics or check_keys("1d-masked_reconstruction"):
            latent_mask = torch.zeros(z.shape[0], self.latent_dim, device=z.device)
            latent_mask[:, 0] = 1
            if not self.density_model_name=="diffusion":
                z_masked_dense = z_dense * latent_mask
                x_zmask = self.decode(z_masked_dense, c) 
                loss_values["1d-masked_reconstruction"] = self._reconstruction_loss(x, x_zmask)
            else:
                loss_values["1d-masked_reconstruction"] = float("nan") * torch.ones(z.shape[0])

        # Cyclic consistency of latent code -- gradient only to encoder
        if val_all_metrics or check_keys("z_reconstruction_encoder"):
            # Not reusing x1 from above, as it does not detach z
            if self.density_model_name in ["fif"]:
                z1_detached = z1.detach()
                z1_dense = self.encode_density(z1_detached, c)
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
                    z_random = self.get_latent(z.device).sample((z.shape[0],), c)
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
                    c1 = self.subject_model.encode(x_random, cT)
                    loss_values["fiber_loss"] = self._reduced_rec_loss(c, c1)
                except Exception as e:
                    warn("Error in computing fiber loss, setting to nan. Error: " + str(e))
                    loss_values["fiber_loss"] = (
                        float("nan") * torch.ones(z_random.shape[0]))
                try:
                    # Sanity checks might fail for random data
                    z1_random = self.encode(x_random, c_random)
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
                print("WARNING: lossless_ae gets trained jointly with a density_model")
        else:
            print("WARNING: lossless_ae gets not trained")
        if self.density_model_name:
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
