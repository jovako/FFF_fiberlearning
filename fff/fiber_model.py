from copy import deepcopy
from collections import namedtuple, defaultdict
from importlib import import_module
from math import prod, log10

# import ldctinv.pretrained
from warnings import warn

import torch
import torch.nn as nn
import torchvision.models as torchmodels
import lightning_trainable
from lightning_trainable.trainable.trainable import auto_pin_memory, SkipBatch
from torch.distributions import Independent, Normal
from torch.nn import Sequential, CrossEntropyLoss

from fff.base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult, build_model
from fff.base import (
    wasserstein2_distance_gaussian_approximation,
    rand_log_uniform,
    soft_heaviside,
)
from fff.lossless_ae import LosslessAE, LosslessAEHParams
from fff.base import (
    wasserstein2_distance_gaussian_approximation,
    rand_log_uniform,
    soft_heaviside,
)
from fff.base import LogProbResult
from fff.subject_model import SubjectModel
from fff.loss import volume_change_surrogate
from fff.utils.jacobian import compute_jacobian
from fff.utils.diffusion import make_betas
from fff.utils.utils import sum_except_batch
from fff.data import get_model_path
from fff.evaluate.plot_fiber_model import *


ConditionedBatch = namedtuple(
    "ConditionedBatch",
    ["x0", "x_noisy", "loss_weights", "condition", "dequantization_jac", "jac_sm"],
)


class FiberModelHParams(FreeFormBaseHParams):
    val_every_n_epoch: int = 1
    cond_dim: int | None = None
    compute_c_on_fly: bool = False
    density_model: list
    lossless_ae: dict | LosslessAEHParams | None = None
    load_lossless_ae_path: str | None = None
    load_density_model_path: str | None = None
    load_subject_model: bool = False
    sm_input_transform: str | None = None
    sm_empty_condition: bool = False
    train_lossless_ae: bool = True
    ae_conditional: bool = False
    ae_deterministic_encode: bool | None = None
    vae: bool = False
    reconstruct_dims: int = 1
    diffusion_betas_max: float = 0.2
    diffusion_beta_schedule: str = "linear"
    add_noise_for_sm: bool = False
    cfg: dict = {}

    eval_all: bool = True
    fiber_loss_every: int = 1
    cnew_every: int = 1  # deprecated and not used anymore

    warm_up_fiber: int | list = 0
    warm_up_epochs: int | list = 0

    condition_noise: float = 0.0

    def __post__init__(self):
        # delete models list
        if "models" in self:
            del self["models"]


class FiberModel(FreeFormBase):
    """
    This class abstracts the functionalities of a model which learns
    the fibers of a "subject model".
    """

    hparams: FiberModelHParams

    def __init__(self, hparams: FiberModelHParams | dict):
        if not isinstance(hparams, FiberModelHParams):
            hparams = FiberModelHParams(**hparams)
        super().__init__(hparams)

        # Add learnable parameter for standard deviation for vae training
        self.lamb = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        if "perceptual_loss" in defaultdict(float, self.hparams.loss_weights):
            vgg = torchmodels.vgg16(weights=torchmodels.VGG16_Weights.IMAGENET1K_V1)
            vgg.eval()
            self.vgg_features = vgg.features
            for param in self.vgg_features.parameters():
                param.requires_grad = False

    def init_models(self):
        # Ask whether the latent variebles should be passed by another learning model and which model class to use
        if all(
            [
                model_hparams["name"] == "fff.model.Identity"
                for model_hparams in self.hparams.density_model
            ]
        ):
            self.density_model_type = None
        elif any(
            [
                model_hparams["name"]
                in [
                    "fff.model.DenoisingFlow",
                    "fff.model.MultilevelFlow",
                    "fff.model.INN",
                ]
                for model_hparams in self.hparams.density_model
            ]
        ):
            self.density_model_type = "inn"
            assert all(
                [
                    model_hparams["name"]
                    in [
                        "fff.model.DenoisingFlow",
                        "fff.model.MultilevelFlow",
                        "fff.model.INN",
                    ]
                    for model_hparams in self.hparams.density_model
                ]
            ), "Coupling Flows cannot be mixed with other models for now."

        elif any(
            [
                model_hparams["name"] == "fff.model.DiffusionModel"
                for model_hparams in self.hparams.density_model
            ]
        ):
            assert (
                len(self.hparams.density_model) == 1
            ), "Diffusion model must be the only model in the density model"
            self.density_model_type = "diffusion"
            self.betas = make_betas(
                1000,
                self.hparams.diffusion_betas_max,
                self.hparams.diffusion_beta_schedule,
            )
            self.hparams.density_model[-1]["betas"] = (self.betas,)
            self.alphas_ = torch.cumprod((1 - self.betas), axis=0)
            self.sample_steps = torch.linspace(0, 1, 1000).flip(0)
            if self.hparams.cfg:
                raise (NotImplementedError("Diffusion model cfg not implemented"))
        elif any(
            [
                model_hparams["name"] == "fff.model.FlowMatching"
                for model_hparams in self.hparams.density_model
            ]
        ):
            assert (
                len(self.hparams.density_model) == 1
            ), "Flow matching model must be the only model in the density model"
            self.density_model_type = "flow_matching"
            if (
                not self.hparams.density_model[0].get("conditional", False)
                and self.hparams.cfg
            ):
                raise (ValueError("Flow matching model must be conditional to use cfg"))
        else:
            self.density_model_type = "fff"

        # Check whether self.lossless_ae is a VAE
        self.vae = self.hparams.vae

        if self.hparams.load_subject_model:
            print("loading subject_model")
            sm_dir = get_model_path(**self.hparams["data_set"])
            self.subject_model = SubjectModel(
                sm_dir,
                self.hparams.data_set.subject_model_type,
                fixed_transform=self.hparams.sm_input_transform,
                empty_condition=self.hparams.sm_empty_condition,
            )
            self.subject_model.eval()
            for param in self.subject_model.parameters():
                param.require_grad = False
        else:
            self.subject_model = None

        # Build condition embedder
        self.condition_embedder = build_model(
            self.hparams.condition_embedder, self.ae_cond_dim, 0
        )
        if self.condition_embedder is not None:
            assert not any(
                [
                    loss == "coarse_supervised"
                    for loss, _ in self.hparams.loss_weights.items()
                ]
            ), "coarse_supervised loss is not applicable for a model with condition embedder."
            for model in self.condition_embedder:
                del model.model.decoder

        # Build models
        # First the lossless vae
        ae_hparams = {}
        if self.hparams.load_lossless_ae_path is None:
            if self.hparams.lossless_ae is None:
                raise ValueError("No lossless_ae specified!")
            ae_hparams = self.hparams.lossless_ae
        elif self.hparams.lossless_ae is not None:
            warn("Overwriting model_spec from config with loaded model!")
        ae_hparams["data_dim"] = self.data_dim
        if self.hparams.ae_conditional:
            ae_hparams["cond_dim"] = self.ae_cond_dim
        ae_hparams["vae"] = self.vae
        if ae_hparams.get("path") is not None:
            raise (
                RuntimeError(
                    "Specificy pretrained models via the load_lossless_ae_path flag, not the path key in lossless_ae hparams"
                )
            )
        ae_hparams["path"] = self.hparams.load_lossless_ae_path
        ae_hparams["train"] = self.hparams.train_lossless_ae
        self.lossless_ae = LosslessAE(ae_hparams)

        # Build density_model that operates in the latent space of the lossless_ae
        self.density_model = build_model(
            self.hparams.density_model,
            self.lossless_ae.latent_dim,
            self.cond_dim,
        )
        if self.hparams.load_density_model_path:
            print("load density_model checkpoint")
            checkpoint = torch.load(self.hparams.load_density_model_path)
            density_model_weights = {
                k[14:]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("density_model.")
            }
            self.density_model.load_state_dict(density_model_weights)

        if not self.hparams.train_lossless_ae:
            for param in self.lossless_ae.parameters():
                param.requires_grad = False

    @property
    def latent_dim(self):
        if self.density_model_type:
            return self.density_model[-1].hparams.latent_dim
        else:
            return self.lossless_ae.latent_dim

    @property
    def cond_dim(self):
        if self.condition_embedder is not None:
            return self.condition_embedder[-1].hparams.latent_dim
        else:
            return self.ae_cond_dim

    @property
    def ae_cond_dim(self):
        if self.hparams.compute_c_on_fly:
            assert self.subject_model != None, "No subject model loaded!"
            return self.hparams.cond_dim
        else:
            return self._data_cond_dim

    def is_conditional(self):
        return self.cond_dim != 0

    def encode_lossless(self, x, c, return_only_x=True, return_codebook_loss=False):
        deterministic = self.hparams.ae_deterministic_encode
        if deterministic is None:
            if not self.training:
                deterministic = True
            else:
                deterministic = False
        if return_codebook_loss:
            return self.lossless_ae.encode(
                x,
                c,
                return_only_x=return_only_x,
                return_codebook_loss=return_codebook_loss,
                deterministic=deterministic,
            )
        else:
            return self.lossless_ae.encode(
                x,
                c,
                return_only_x=return_only_x,
                deterministic=deterministic,
            )

    def encode_density(self, z, c, jac=False):
        if self.condition_embedder is not None:
            for model in self.condition_embedder:
                c = model.encode(
                    c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                )
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
        z_dense = self.encode_density(self.encode_lossless(x, c, return_only_x=True), c)
        return z_dense

    def decode_lossless(self, z, c):
        return self.lossless_ae.decode(z, c)

    def decode_density(self, z_dense, c):
        # c = self.unflatten_ce(c).unsqueeze(1)
        if self.density_model_type in ["diffusion", "flow_matching"]:
            t, c = c
        if self.condition_embedder is not None:
            for model in self.condition_embedder:
                # c = model(c)
                c = model.encode(
                    c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                )
        if self.density_model_type in ["diffusion", "flow_matching"]:
            c = t, c
        for net in self.density_model:
            z_dense = net.decode(z_dense, c)
        return z_dense

    def decode(self, z_dense, c):
        x = self.decode_lossless(self.decode_density(z_dense, c), c)
        return x

    def sample_density(self, z_dense, c, **kwargs):
        # Diffusion sampler we need seperate sampling function
        # c = self.unflatten_ce(c).unsqueeze(1)
        if self.condition_embedder is not None:
            for model in self.condition_embedder:
                # c = model(c)
                c = model.encode(
                    c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                )
        if self.hparams.cfg:
            for net in self.density_model:
                z_dense = net.sample_with_guidance(
                    z_dense, c, self.get_null_condition(c), **kwargs
                )
        else:
            for net in self.density_model:
                z_dense = net.sample(z_dense, c, **kwargs)
        return z_dense

    def _encoder_jac(self, x, c, **kwargs):
        return compute_jacobian(
            x,
            self.encode_density,
            c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs,
        )

    def _decoder_jac(self, z, c, **kwargs):
        return compute_jacobian(
            z,
            self.decode_density,
            c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs,
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
            z_dense = self.get_latent(self.device).sample(sample_shape, condition)
        except TypeError:
            z_dense = self.get_latent(self.device).sample(sample_shape)
        z_dense = z_dense.reshape(
            prod(sample_shape), *z_dense.shape[len(sample_shape) :]
        )
        batch = [z_dense]
        if condition is not None:
            c = condition
        else:
            c = torch.empty(z_dense.shape[0], 0).to(z_dense.device)
        z = self.sample_density(z_dense, c)
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
            **kwargs,
        )

        volume_change = out.surrogate

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    def _cdf_loss(self, a, b):
        # Computaion of cumulative density function as distance measure for loss
        fl = torch.squeeze(
            torch.abs(self.teacher_normal.cdf(a) - self.teacher_normal.cdf(b))
        )
        return fl

    def _reconstruction_loss(self, a, b):
        return torch.sqrt(torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1))

    def _lamb_reconstruction_loss(self, a, b):
        return (
            torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)
        ) / self.lamb + torch.log(self.lamb)

    def _sqr_reconstruction_loss(self, a, b):
        return torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)

    def _reduced_rec_loss(self, a, b):
        return torch.sqrt(torch.mean((a - b).reshape(a.shape[0], -1) ** 2, -1))

    def _l1_loss(self, a, b):
        return torch.mean(torch.abs(a - b).reshape(a.shape[0], -1), -1)

    def _jacreduced_l2(self, a, b, jac, epsilon=0.01):
        return (
            torch.sqrt(
                torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1) / float(a.shape[-1])
            )
            / jac
            / epsilon
        )

    def _fm_loss(self, bt, bt_hat):
        return (
            sum_except_batch(torch.pow(bt_hat, 2))
            - 2 * sum_except_batch(bt * bt_hat)
            + sum_except_batch(torch.pow(bt, 2))
        )

    def get_null_condition(self, cond_batch):
        """
        Returns a null condition for the given batch.
        """
        if self.hparams.cfg.get("null_condition", "learned") == "learned":
            if not hasattr(self, "null_condition"):
                self.null_condition = torch.nn.Parameter(
                    torch.randn(1, *cond_batch.shape[1:]), requires_grad=True
                )
            return self.null_condition.expand(*cond_batch.shape).to(cond_batch.device)
        elif self.hparams.cfg["null_condition"] == "zero":
            return torch.zeros_like(cond_batch)
        else:
            raise ValueError(
                "Unknown null condition type: "
                + self.hparams.cfg["null_condition"]
                + ". Use 'learned' or 'zero'."
            )

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
        jac_sm = conditioned.jac_sm
        deq_vol_change = conditioned.dequantization_jac

        loss_values = {}
        metrics = {}

        def check_keys(*keys):
            return any(
                (loss_key in loss_weights)
                and (
                    torch.any(loss_weights[loss_key] > 0)
                    if torch.is_tensor(loss_weights[loss_key])
                    else loss_weights[loss_key] > 0
                )
                for loss_key in keys
            )

        if check_keys("codebook_loss"):
            (
                z,
                mu,
                logvar,
                codebook_loss,
            ) = self.encode_lossless(
                x, c, return_only_x=False, return_codebook_loss=True
            )
            loss_values["codebook_loss"] = codebook_loss

        else:
            z, mu, logvar = self.encode_lossless(x, c, return_only_x=False)
        # Reconstruction of lossless latent variables z
        x1 = self.decode_lossless(z, c)
        # z = z + torch.randn_like(z) * 0.01

        # Losses for lossless ae:
        # Reconstruction
        if not self.training or check_keys(
            "ae_reconstruction",
            "ae_noisy_reconstruction",
            "ae_sqr_reconstruction",
            "ae_lamb_reconstruction",
            "ae_l1_reconstruction",
            "ae_noisy_l1_reconstruction",
        ):
            loss_values["ae_reconstruction"] = self._reconstruction_loss(x0, x1)
            loss_values["ae_noisy_reconstruction"] = self._reconstruction_loss(x, x1)
            loss_values["ae_sqr_reconstruction"] = self._sqr_reconstruction_loss(x, x1)
            loss_values["ae_lamb_reconstruction"] = self._lamb_reconstruction_loss(
                x, x1
            )
            loss_values["ae_l1_reconstruction"] = self._l1_loss(x0, x1)
            loss_values["ae_noisy_l1_reconstruction"] = self._l1_loss(x, x1)

        if (
            not self.training or check_keys("ae_rec_fiber_loss")
        ) and self.subject_model is not None:
            c_orig = self.subject_model.encode(x0)
            c1 = self.subject_model.encode(x1)
            loss_values["ae_rec_fiber_loss"] = self._reduced_rec_loss(c_orig, c1)

        # KL-Divergence for VAE
        if check_keys("ae_elbo"):
            loss_values["ae_elbo"] = -0.5 * torch.sum(
                (1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar)), -1
            )

        # Cyclic consistency of latent code -- gradient only to encoder
        if not self.training or check_keys("ae_cycle_loss"):
            x1_detached = x1.detach()
            z_cycle = self.encode_lossless(x1_detached, c, return_only_x=True)
            loss_values["ae_cycle_loss"] = self._reconstruction_loss(z, z_cycle)

        if check_keys("ae_perceptual_loss"):
            perceptual_loss = 0
            # reshape into image and duplicate channels if necessary
            from fff.model.utils import guess_image_shape

            vgg_input = x1.reshape(-1, *guess_image_shape(x1.shape[1]))
            if vgg_input.shape[1] == 1:
                vgg_input = vgg_input.repeat(1, 3, 1, 1)
            vgg_target = x.reshape(-1, *guess_image_shape(x.shape[1]))
            if vgg_target.shape[1] == 1:
                vgg_target = vgg_target.repeat(1, 3, 1, 1)
            for i, m in self.vgg_features._modules.items():
                vgg_input = m(vgg_input)
                vgg_target = m(vgg_target)
                if i in ["3", "8", "15", "22"]:
                    perceptual_loss += self._l1_loss(vgg_input, vgg_target) / prod(
                        vgg_input.shape[1:]
                    )
            loss_values["ae_perceptual_loss"] = perceptual_loss

        # Losses for density model:
        # Empty until computed
        z1 = z_dense = None

        # NLL losses
        if val_all_metrics or check_keys("nll") or check_keys("coarse_supervised"):
            # NLL loss for INN-architectures
            if self.density_model_type == "inn":
                if check_keys("coarse_supervised"):
                    c_n = c + torch.randn_like(c) * self.hparams.condition_noise
                else:
                    c_n = c
                z_combined, log_det = self.encode_density(z.detach(), c_n, jac=True)
                if isinstance(z_combined, tuple):
                    z_dense, z_coarse = z_combined
                    if check_keys("coarse_supervised"):
                        loss_values["coarse_supervised"] = self._reconstruction_loss(
                            c, z_coarse
                        )
                else:
                    z_dense = z_combined
                log_prob = self._latent_log_prob(z_dense, c_n)
                loss_values["nll"] = -(log_prob + log_det)

            # Freeform loss
            elif self.density_model_type == "fff" and (
                self.hparams.eval_all
                and (
                    not self.training
                    or (
                        self.hparams.exact_train_nll_every is not None
                        and batch_idx % self.hparams.exact_train_nll_every == 0
                    )
                )
            ):
                # exact
                key = "nll_exact" if self.training else "nll"
                # todo unreadable
                if self.training or (
                    self.hparams.skip_val_nll is not True
                    and (
                        self.hparams.skip_val_nll is False
                        or (
                            isinstance(self.hparams.skip_val_nll, int)
                            and batch_idx < self.hparams.skip_val_nll
                        )
                    )
                ):
                    with torch.no_grad():
                        log_prob_result = self.exact_log_prob(
                            x=z.detach(), c=c, jacobian_target="both"
                        )
                        z_dense = log_prob_result.z
                        z1 = log_prob_result.x1
                    loss_values[key] = -log_prob_result.log_prob - deq_vol_change
                    loss_values.update(log_prob_result.regularizations)
                else:
                    loss_weights["nll"] = 0

            # surrogate
            elif (
                (self.density_model_type == "fff")
                and self.training
                and check_keys("nll")
            ):
                warm_up = self.hparams.warm_up_epochs
                if isinstance(warm_up, int):
                    warm_up = warm_up, warm_up + 1
                warm_up = map(lambda x: x * self.hparams.max_epochs // 100, warm_up)
                nll_start, warm_up_end = warm_up
                if nll_start == 0:
                    nll_warmup = 1
                else:
                    nll_warmup = soft_heaviside(
                        self.current_epoch
                        + batch_idx
                        / len(
                            self.trainer.train_dataloader
                            if self.training
                            else self.trainer.val_dataloaders
                        ),
                        nll_start,
                        warm_up_end,
                    )
                loss_weights["nll"] *= nll_warmup
                if check_keys("nll"):
                    log_prob_result = self.surrogate_log_prob(x=z.detach(), c=c)
                    z_dense = log_prob_result.z
                    z1 = log_prob_result.x1
                    loss_values["nll"] = -log_prob_result.log_prob - deq_vol_change
                    loss_values.update(log_prob_result.regularizations)

        # Diffusion model mean squared error
        if check_keys("diff_mse"):
            if not self.density_model_type == "diffusion":
                raise ValueError("diff_mse is only available for diffusion models")
            t = torch.randint(0, 1000, (z.size(0),), device=z.device).long()
            z_diff, epsilon = self.diffuse(z.detach(), t, self.alphas_.to(z.device))
            epsilon_pred = self.decode_density(z_diff.detach(), (t, c))
            loss_values["diff_mse"] = self._reconstruction_loss(
                epsilon_pred, epsilon.detach()
            )

        if check_keys("fm_loss"):
            if not self.density_model_type == "flow_matching":
                raise ValueError("fm_loss is only available for flow matching models")
            t = torch.rand(z.shape[0], device=z.device)
            z_fm = self.get_latent(z.device).sample((z.shape[0],))
            # Sander's method
            if not self.density_model[0].conditional:
                z_fm[:, : self.cond_dim] = c
            elif self.hparams.cfg:
                p_uncond = self.hparams.cfg.get("p_unconditional", 0.1)
                if p_uncond > 0:
                    # Randomly set the condition to zero
                    mask = torch.rand(c.shape[0], device=c.device) < p_uncond
                    c[mask] = self.get_null_condition(c[mask])
            loss_values["fm_loss"] = self.density_model[0].compute_fm_loss(
                t, z_fm, z, c
            )

        need_z_dense = val_all_metrics or check_keys(
            "latent_reconstruction",
            "latent_l1_reconstruction",
            "masked_reconstruction",
            "cycle_loss",
            "perceptual_loss",
        )

        if z_dense is None and need_z_dense:
            z_dense = self.encode_density(z.detach(), c)

        # Reconstruction of latent z
        if val_all_metrics or check_keys(
            "latent_reconstruction", "latent_l1_reconstruction"
        ):
            if self.density_model_type in ["diffusion", "flow_matching"]:
                raise ValueError(
                    "latent_reconstruction is not available for diffusion models"
                )
            if z1 is None:
                z1 = self.decode_density(z_dense, c)
            loss_values["latent_reconstruction"] = self._reconstruction_loss(
                z.detach(), z1
            )
            loss_values["latent_l1_reconstruction"] = self._l1_loss(z.detach(), z1)

        # Wasserstein distance of marginal to Gaussian
        if val_all_metrics:
            with torch.no_grad():
                z_marginal = z_dense.reshape(-1)
                z_gauss = torch.randn_like(z_marginal)

                z_marginal_sorted = z_marginal.sort().values
                z_gauss_sorted = z_gauss.sort().values

                metrics["z 1D-Wasserstein-1"] = (
                    (z_marginal_sorted - z_gauss_sorted).abs().mean()
                )
                metrics["z std"] = torch.std(z_marginal)
                if check_keys("ae_lamb_reconstruction"):
                    metrics["lambda"] = self.lamb

        if val_all_metrics or check_keys("masked_reconstruction"):
            if self.density_model_type in ["diffusion", "flow_matching"]:
                raise ValueError(
                    "masked_reconstruction is not available for diffusion models"
                )
            latent_mask = torch.zeros(z.shape[0], self.latent_dim, device=z.device)
            latent_mask[:, : self.hparams.reconstruct_dims] = 1
            z_masked_dense = z_dense * latent_mask
            x_zmask = self.decode(z_masked_dense, c)
            loss_values["masked_reconstruction"] = self._reconstruction_loss(x, x_zmask)

        # Cyclic consistency of latent code -- gradient only to encoder
        if val_all_metrics or check_keys("cycle_loss"):
            if z1 is None:
                z1 = self.decode_density(z_dense, c)
            z1_detached = z1.detach()
            z_dense1 = self.encode_density(z1_detached, c)
            if isinstance(z_dense1, tuple):
                z_dense1, _ = z_dense1
            loss_values["cycle_loss"] = self._reconstruction_loss(z_dense, z_dense1)

        if check_keys("perceptual_loss"):
            perceptual_loss = 0
            # reshape into image and duplicate channels if necessary
            from fff.model.utils import guess_image_shape

            if z1 is None:
                z1 = self.decode_density(z_dense, c)
            x_full_recon = self.decode_lossless(z1, c)
            vgg_input = x_full_recon.reshape(
                -1, *guess_image_shape(x_full_recon.shape[1])
            )
            if vgg_input.shape[1] == 1:
                vgg_input = vgg_input.repeat(1, 3, 1, 1)
            vgg_target = x.reshape(-1, *guess_image_shape(x.shape[1]))
            if vgg_target.shape[1] == 1:
                vgg_target = vgg_target.repeat(1, 3, 1, 1)
            for i, m in self.vgg_features._modules.items():
                vgg_input = m(vgg_input)
                vgg_target = m(vgg_target)
                if i in ["3", "8", "15", "22"]:
                    perceptual_loss += self._l1_loss(vgg_input, vgg_target) / prod(
                        vgg_input.shape[1:]
                    )
            loss_values["perceptual_loss"] = perceptual_loss

        # Cyclic consistency of latent code sampled from Gaussian and fiber loss
        if check_keys("fiber_loss", "jac_fiber_loss", "z_sample_reconstruction") or (
            not self.training  # TODO: This should be val_all_metrics, but then FM/Diffusion models are not getting validated
            and (
                self.current_epoch % self.hparams.fiber_loss_every == 0
                or self.current_epoch == self.hparams.max_epochs - 1
            )
        ):
            warm_up = self.hparams.warm_up_fiber
            if isinstance(warm_up, int):
                warm_up = warm_up, warm_up + 1
            warm_up = map(lambda x: x * self.hparams.max_epochs // 100, warm_up)
            fl_start, warm_up_end = warm_up
            if fl_start == 0:
                fl_warmup = 1
            else:
                fl_warmup = soft_heaviside(
                    self.current_epoch
                    + batch_idx
                    / len(
                        self.trainer.train_dataloader
                        if self.training
                        else self.trainer.val_dataloaders
                    ),
                    fl_start,
                    warm_up_end,
                )
            loss_weights["fiber_loss"] *= fl_warmup
            try:
                z_dense_random = self.get_latent(z.device).sample((z.shape[0],), c)
            except TypeError:
                z_dense_random = self.get_latent(z.device).sample((z.shape[0],))
            if isinstance(z_dense_random, tuple):
                z_dense_random, c_random = z_dense_random
            else:
                c_random = c
            z_random = self.sample_density(z_dense_random, c_random)
            x_random = self.decode_lossless(z_random, c_random)
            x_random_sm = x_random
            if self.hparams.add_noise_for_sm:
                if self.hparams["data_set"].get("data") == "highdose":
                    # Add noise to the sampled highdose samples
                    x_random_sm = x_random + batch[1] - x
                else:
                    raise (
                        ValueError(
                            "Adding noise from condition only works for highdose images as data"
                        )
                    )

            # Try whether the model learns fibers and therefore has a subject model
            try:
                # There might be no subject model
                c1 = self.subject_model.encode(x_random_sm)
                c_sm = torch.empty(x_random.shape[0], 0).to(x_random.device)
                if jac_sm is not None:
                    loss_values["jac_fiber_loss"] = self._jacreduced_l2(
                        c_random, c1, jac_sm, epsilon=0.01
                    )
                loss_values["fiber_loss"] = self._reduced_rec_loss(c_random, c1)
            except Exception as e:
                warn("Error in computing fiber loss, setting to nan. Error: " + str(e))
                loss_values["fiber_loss"] = float("nan") * torch.ones(z_random.shape[0])
            try:
                # Sanity checks might fail for random data
                z1_random = self.encode(x_random, c_random)
                if isinstance(z1_random, tuple):
                    z1_random, _ = z1_random
                loss_values["z_sample_reconstruction"] = self._reconstruction_loss(
                    z_dense_random, z1_random
                )
            except Exception as e:
                warn(
                    "Error in computing z_sample_reconstruction, setting to nan. Error: "
                    + str(e)
                )
                loss_values["z_sample_reconstruction"] = float("nan") * torch.ones(
                    z_random.shape[0]
                )

        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in loss_weights.items()
            if check_keys(key) and (self.training or key in loss_values)
        )

        # Metrics are averaged, non-weighted loss_values
        invalid_losses = []
        for key, weight in loss_values.items():
            # One value per key
            if loss_values[key].shape == (x.shape[0],):
                metrics[key] = loss_values[key].mean(-1)
            elif loss_values[key].ndim == 0:
                metrics[key] = loss_values[key]
            else:
                invalid_losses.append(key)
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

    def on_train_epoch_end(self) -> None:
        """
        if (((self.current_epoch%5==0 and self.current_epoch%self.hparams.fiber_loss_every==0) or
            self.current_epoch==self.hparams.max_epochs-1) and
            self.subject_model is not None):
            with torch.no_grad():
                val_data = self.trainer.val_dataloaders
                batch = next(iter(val_data))
                for i in range(len(batch)):
                    batch[i] = batch[i].to(self.device)
                conditioned = self.apply_conditions(batch)
                x = conditioned.x_noisy
                c = conditioned.condition
                x_samples = self.sample(torch.Size([x.shape[0]]), c)
                c_samples = self.subject_model.encode(x_samples)
                # x_samples_sm = self.subject_model.decode(c_samples)
                c_orig = self.subject_model.encode(x)
                # x_orig_sm = self.subject_model.decode(c_orig)
                writer = self.logger.experiment
                x_plot = [
                    torch.clip(x, min=0, max=1),
                    torch.clip(x_samples, min=0, max=1),
                ]
                titles = ["x_orig", "x_sampled"]
                fig = plot_mnist(x_plot, titles)
                writer.add_figure(f"Fiber samples", fig, self.current_epoch)
        """
        """
                x_plot = [x_orig_sm, x_samples_sm, torch.abs(x_orig_sm-x_samples_sm)]
                titles = ["SM(x_orig)", "SM(x_sampled)", "Residual"]
                fig = plot_mnist(x_plot, titles)
                writer.add_figure(f"Verify samples", fig, self.current_epoch)
        """

    def diffuse(self, x, t, alphas_):
        noise = torch.randn_like(x)
        alpha_t = alphas_[t].unsqueeze(1)
        alpha_t = alpha_t.repeat([1, x.shape[1]])
        noisy_x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * noise
        # noisy_x = alpha_t.sqrt() * x + noise
        return noisy_x, noise

    def apply_conditions(self, batch) -> ConditionedBatch:
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        conds = []
        x_sm = x0
        if self.hparams.add_noise_for_sm:
            if self.hparams["data_set"].get("data") == "highdose":
                x_sm = batch[1]
            else:
                raise (
                    ValueError(
                        "Adding noise from condition only works for highdose images as data"
                    )
                )

        # Dataset condition
        if self.is_conditional() and len(batch) < 2:
            if self.hparams.compute_c_on_fly:
                conds.append(self.subject_model.encode(x_sm).detach())
            else:
                raise ValueError(
                    "You must pass a batch including conditions for each dataset condition"
                )
        if len(batch) > 1:
            if self.hparams.compute_c_on_fly:
                dataset_cond = self.subject_model.encode(x_sm).detach()
            else:
                dataset_cond = batch[1]
            conds.append(dataset_cond)
        if len(batch) > 2:
            jac_sm = batch[2]
        else:
            jac_sm = None

        # SoftFlow
        noise_conds, x, dequantization_jac = self.dequantize(batch)
        conds.extend(noise_conds)

        # Loss weight aware
        loss_weights = defaultdict(float, self.hparams.loss_weights)
        for loss_key, loss_weight in self.hparams.loss_weights.items():
            if isinstance(loss_weight, list):
                min_weight, max_weight = loss_weight
                if not self.training:
                    # Per default, select the first value in the list
                    max_weight = min_weight
                weight_scale = rand_log_uniform(
                    min_weight,
                    max_weight,
                    shape=base_cond_shape,
                    device=device,
                    dtype=dtype,
                )
                loss_weights[loss_key] = (10**weight_scale).squeeze(1)
                conds.append(weight_scale)

        if len(conds) == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        elif len(conds) == 1:
            # This is a hack to pass through the info dict from QM9
            (c,) = conds
        else:
            c = torch.cat(conds, -1)
        return ConditionedBatch(x0, x, loss_weights, c, dequantization_jac, jac_sm)

    def configure_optimizers(self):
        params = []
        if self.hparams.train_lossless_ae:
            params.extend(list(self.lossless_ae.parameters()))
            if self.vae:
                params.append(self.lamb)
            if self.density_model_type:
                print("WARNING: lossless_ae gets trained jointly with a density_model")
        else:
            print("WARNING: lossless_ae gets not trained")
        if self.density_model_type:
            params.extend(list(self.density_model.parameters()))
        else:
            print("WARNING: density model gets not trained")
        if self.condition_embedder is not None:
            params.extend(list(self.condition_embedder.parameters()))
        else:
            print("WARNING: no condition embedder for optimizer")

        kwargs = dict()

        match self.hparams.optimizer:
            case str() as name:
                optimizer = lightning_trainable.utils.get_optimizer(name)(
                    params, **kwargs
                )
            case dict() as kwargs:
                name = kwargs.pop("name")
                optimizer = lightning_trainable.utils.get_optimizer(name)(
                    params, **kwargs
                )
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

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.hparams.val_every_n_epoch == 0:
            metrics = self.compute_metrics(batch, batch_idx)
            for key, value in metrics.items():
                self.log(f"validation/{key}",
                         value,
                         prog_bar=key == self.hparams.loss,
                         sync_dist=self.hparams.strategy.startswith("ddp"))
