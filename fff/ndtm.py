import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import sys
sys.path.append("/home/hd/hd_hd/hd_gu452/oc-guidance/")
from utils.functions import sigmoid
# load and show test_image.png
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from math import prod
from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel


device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffusionScheduleConfig:
    beta_schedule: str = 'linear'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    given_betas: torch.Tensor = None  # Optional, if provided, will override the schedule

class DiffusionSchedule:
    def __init__(self, hparams):
        # Instantiate the diffusion process
        if hparams.given_betas is None:
            if hparams.beta_schedule == "quad":
                betas = (
                    np.linspace(
                        hparams.beta_start**0.5,
                        hparams.beta_end**0.5,
                        hparams.num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif hparams.beta_schedule == "linear":
                betas = np.linspace(
                    hparams.beta_start, hparams.beta_end, hparams.num_diffusion_timesteps, dtype=np.float64
                )
            elif hparams.beta_schedule == "const":
                betas = hparams.beta_end * np.ones(hparams.num_diffusion_timesteps, dtype=np.float64)
            elif hparams.beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    hparams.num_diffusion_timesteps, 1, hparams.num_diffusion_timesteps, dtype=np.float64
                )
            elif hparams.beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, hparams.num_diffusion_timesteps)
                betas = sigmoid(betas) * (hparams.beta_end - hparams.beta_start) + hparams.beta_start
            else:
                raise NotImplementedError(hparams.beta_schedule)
            assert betas.shape == (hparams.num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = hparams.given_betas
        self.betas = torch.cat([torch.zeros(1).to(betas.device), betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).cuda().float()
        self.hparams = hparams

    def alpha(self, t):
        return self.alphas[t+1]
    
    def beta(self, t):
        return self.betas[t+1]

    def predict_x_from_eps(self, xt, et, t):
        alpha_t = self.alpha(t).view(-1, 1, 1, 1)
        return (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

class DiffusionModel:
    def __init__(self, model: nn.Module, diffusion_schedule: object, class_cond_diffusion_model=False):
        self.model = model
        self.diffusion_schedule = diffusion_schedule
        self.class_cond_diffusion_model = class_cond_diffusion_model

    def __call__(self, xt, y, t, predict_variance=False):
        y = y if self.class_cond_diffusion_model else None
        out = self.model(xt, t, y)
        et = out[:, :3]
        if not predict_variance:
            return et
        else:
            logvar = out[:, 3:]
            return et, logvar

class Combine_fn(ABC):
    def __init__(self, gamma_t=None):
        self.gamma_t = gamma_t

    @abstractmethod
    def forward(self, xt, ut, t=None, **kwargs):
        pass

    def __call__(self, xt, ut, t=None, **kwargs):
        return self.forward(xt, ut, t=t, **kwargs)


class Additive(Combine_fn):
    def forward(self, xt, ut, t=None, **kwargs):
        gamma_t = self.gamma_t(t) if callable(self.gamma_t) else self.gamma_t
        return xt + (gamma_t * ut if gamma_t is not None else ut)

@dataclass
class NDTMConfig:
    eta: float = 0.1
    N: int = 2  # Number of optimization steps
    gamma_t: float = 4.0  # u_t weight
    u_lr: float = 0.01  # learning rate for u_t
    combine_fn: str = "additive"  # Function to combine scores
    w_score_scheme: str = "ddim"  # Weighting scheme for score
    w_control_scheme: str = "ddim"  # Weighting scheme for score
    u_lr_scheduler: str = "linear"  # Learning rate scheduler for u_t
    init_control: str = "zero"  # Initialization scheme for u_t
    init_xT: str = "random" # Initialization scheme for x_T
    w_terminal: float = 50.0
    ancestral_sampling: bool = False  # If True, use ancestral sampling
    clip_images: bool = True  # If True, clip images
    clip_range: list = field(default_factory=lambda: [-1, 1])  # Range to clip images
    variance_type: str = "small"

class NDTM:
    def __init__(self, generative_model, subject_model, hparams):
        self.generative_model = generative_model
        self.diffusion = generative_model.diffusion_schedule
        self.subject_model = subject_model
        self.hparams = hparams
        self.F = self._get_combine_fn()

    def _get_combine_fn(self):
        if self.hparams.combine_fn == "additive":
            return Additive(gamma_t=self.hparams.gamma_t)

    def _get_score_weight(self, scheme, t, s, **kwargs):
        alpha_t = self.diffusion.alpha(t)
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device) * 1.e-4
        elif scheme == "ddpm":
            return (beta_t**2) / (alpha_t_im * (1 - alpha_t))
        elif scheme == "ddim":
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.hparams.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()
            c2_ = ((alpha_s / alpha_t) * (1 - alpha_t)).sqrt()
            return (c2 - c2_) ** 2

        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def _get_control_weight(self, scheme, t, s):
        alpha_t = self.diffusion.alpha(t)
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device) * 1.e-4
        elif scheme == "ddpm":
            return 1 / alpha_t_im
        elif scheme == "ddim":
            return alpha_t / alpha_s
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def get_learning_rate(self, base_lr, current_step, total_steps):
        assert self.hparams.u_lr_scheduler in ["linear", "const"], \
            f"Unknown learning rate scheduler: {self.hparams.u_lr_scheduler}"
        
        if self.hparams.u_lr_scheduler == "linear":
            return base_lr * (1.0 - current_step / total_steps)
        else:  # const
            return base_lr

    def sample(self, x, y, ts, **kwargs):
        x_orig = x.clone()
        x = self.initialize(x, y, ts, **kwargs)
        y_0 = kwargs["y_0"]
        bs = x.size(0)
        xt = x
        ss = [-1] + list(ts[:-1])
        xt_s = [xt.cpu()]
        x0_s = []
        uts = []
        scale = self.diffusion.hparams.num_diffusion_timesteps/len(ts)
        beta_start = scale * self.diffusion.hparams.beta_start
        beta_end = scale * self.diffusion.hparams.beta_end
        assert self.diffusion.hparams.beta_schedule == "linear", "Only linear schedule supported for ancestral sampling"
        given_betas = torch.linspace(
            beta_start, beta_end, len(ts), dtype=torch.float64, device=device
        )

        u_t = torch.zeros_like(xt)
        pbar = tqdm(enumerate(zip(reversed(ts), reversed(ss))), total=len(ts), leave=False)
        for i, (ti, si) in pbar:

            t = torch.ones(bs).to(x.device).long() * ti
            s = torch.ones(bs).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            if self.hparams.variance_type == "small":
                c1 = (
                    (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
                ).sqrt() * self.hparams.eta
                c2 = ((1 - alpha_s) - c1**2).sqrt()
            elif self.hparams.variance_type == "large":
                c1 = (1 - alpha_t / alpha_s).sqrt() * self.hparams.eta
                c2 = ((1 - alpha_s) - ((1 - alpha_s) / (1 - alpha_t))*c1**2).sqrt()

            # Initialize control and the optimizer
            u_t = self.initialize_ut(u_t, i)
            ut_clone = u_t.clone().detach()
            ut_clone.requires_grad = True
            current_lr = self.get_learning_rate(self.hparams.u_lr, i, len(ts))
            optimizer = torch.optim.Adam([ut_clone], lr=current_lr)

            # Loss weightings
            w_terminal = self.hparams.w_terminal
            w_score = self._get_score_weight(self.hparams.w_score_scheme, t, s, **kwargs)
            w_control = self._get_control_weight(self.hparams.w_control_scheme, t, s)
            time_rev_ind = len(ts) - i - 1
            beta_eff = given_betas[time_rev_ind]
            alphas = (1 - given_betas)
            alphas_cumprod = alphas.cumprod(dim=0).cuda().float()
            alphas_cumprod_prev = torch.cat((torch.ones(1, device=device), alphas_cumprod[:-1]), dim=0)
            
            ####################################################
            ############## Control Optimization ################
            ####################################################
            et = self.generative_model(xt, y, t).detach()
            for _ in range(self.hparams.N):
                if callable(self.hparams.gamma_t):
                    gamma_t = self.hparams.gamma_t(t)
                else:
                    gamma_t = self.hparams.gamma_t
                if gamma_t == 0:
                    break
                # Guided state vector
                cxt = self.F(xt, ut_clone, t=t, **kwargs)

                # Unguided and guided noise estimates
                et_control = self.generative_model(cxt, y, t)

                # Tweedie's estimate from the guided state vector
                if self.hparams.ancestral_sampling:
                    x0_pred = cxt/alphas_cumprod[time_rev_ind].sqrt() - (1/alphas_cumprod[time_rev_ind] - 1).sqrt()*et_control
                else:
                    x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                if self.hparams.clip_images:
                    x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                score_diff = ((et - et_control) ** 2).reshape(bs, -1).sum(dim=1)
                c_score = w_score * score_diff

                # Control loss
                control_loss = (
                    ((self.F(xt, ut_clone, t=t, **kwargs) - xt) ** 2).reshape(bs, -1).sum(dim=1)
                )

                c_control = w_control * control_loss * (gamma_t**2)

                # Terminal Cost
                # c_terminal = ((y_0 - self.subject_model(x0_pred)) ** 2).reshape(bs, -1).sum(dim=1)
                c_terminal = torch.norm((y_0 - self.subject_model(x0_pred)), p=2, dim=-1).reshape(bs, -1).sum(dim=1)
                c_terminal = w_terminal * c_terminal

                # Aggregate Cost and optimize
                c_t = c_score + c_control + c_terminal

                # print(
                #     f"Diffusion step: {ti} Terminal Loss: {c_terminal.mean().item()} "
                #     f"Control loss: {c_control.mean().item()} Score loss: {c_score.mean().item()}"
                # )
                

                optimizer.zero_grad()
                c_t.sum().backward()
                optimizer.step()
            if self.hparams.N > 0 and gamma_t != 0:
                pbar.set_description(
                    f"Diffusion step: {ti} Terminal Loss: {c_terminal.mean().item()} "
                    f"Control loss: {c_control.mean().item()} Score loss: {c_score.mean().item()}"
            )
            ###########################################
            ############## DDIM update ################
            ###########################################
            with torch.no_grad():

                u_t = ut_clone.detach()
                cxt = self.F(xt, u_t, t=t, **kwargs)
                
                if self.hparams.ancestral_sampling:
                    et_control, log_var = self.generative_model(cxt, y, t, predict_variance=True)
                    x0_pred = cxt/alphas_cumprod[time_rev_ind].sqrt() - (1/alphas_cumprod[time_rev_ind] - 1).sqrt()*et_control
                    # x0_pred = self.diffusion.predict_x_from_eps(xt, et_control, t)
                    if self.hparams.clip_images:
                        x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                    posterior_mean_coef1 = (
                        beta_eff * alphas_cumprod_prev[time_rev_ind].sqrt() / (1.0 - alphas_cumprod[time_rev_ind])
                    )
                    posterior_mean_coef2 = (
                        (1.0 - alphas_cumprod_prev[time_rev_ind])
                        * (1 - beta_eff).sqrt()
                        / (1.0 - alphas_cumprod[time_rev_ind])
                    )
                    xt = (
                        posterior_mean_coef1 * x0_pred
                        + posterior_mean_coef2 * cxt
                    )
                    
                    if log_var.shape == et_control.shape and ti > 0:
                        if time_rev_ind > 0:
                            min_log = torch.log(
                                beta_eff * (1.0 - alphas_cumprod_prev[time_rev_ind]) / (1.0 - alphas_cumprod[time_rev_ind])
                            )
                        else:
                            min_log = torch.log(
                                beta_eff * (1.0 - alphas_cumprod_prev[time_rev_ind]) / (1.0 - alphas_cumprod[time_rev_ind])
                            )

                        max_log = torch.log(beta_eff)
                        # The log_var is [-1, 1] for [min_var, max_var].
                        frac = (log_var + 1) / 2
                        model_log_variance = frac * max_log + (1 - frac) * min_log
                        noise = torch.randn_like(xt) * (0.5*model_log_variance).exp()
                        xt = xt + noise
                else:
                    
                    et_control = self.generative_model(cxt, y, t)
                    x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                    if self.hparams.clip_images:
                        x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                    xt = (
                        alpha_s.sqrt() * x0_pred
                        + c1 * torch.randn_like(xt)
                        + c2 * et_control
                    )
                uts.append(u_t.cpu())

            xt_s.append(xt.cpu())
            x0_s.append(x0_pred.cpu())

        return list(reversed(xt_s)), list(reversed(x0_s))

    def initialize_ut(self, ut, i):
        init_control = self.hparams.init_control

        if init_control == "zero":  # constant zero
            return torch.zeros_like(ut)
        elif init_control == "random":  # constant random
            return torch.randn_like(ut)
    
        elif "causal" in init_control:
            if "zero" in init_control and i == 0:  # causal_zero
                return torch.zeros_like(ut)
            elif "random" in init_control and i == 0:  # causal_random
                return torch.randn_like(ut)

            else:
                return ut

    def initialize(self, x, y, ts, **kwargs):
        """
        random: Initialization with x_T ~ N(0, 1)
        guided: Initialization with x_T ~ DDPM(H^(y_0)) - Only for Linear IP
        """
        init_scheme = self.hparams.init_xT
        
        if init_scheme == "sdedit":
            n = x.size(0)
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            return x * alpha_t.sqrt() + torch.randn_like(x) * (1 - alpha_t).sqrt()
        elif init_scheme == "guided":
            raise(NotImplementedError("Guided initialization not implemented (could be useful if subject model decoder is available)"))
            y_0 = kwargs["y_0"]
            n = x.size(0)
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
        else:
            return torch.randn_like(x)

class SampleRefinement:
    """
    Class for refining samples using gradient descent or autoencoder latent space optimization.
    """
    def __init__(self, subject_model, autoencoder=None):
        self.subject_model = subject_model
        self.autoencoder = autoencoder
        subject_model.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()

    @torch.enable_grad()
    def refine_with_gradient_descent(self, samples, originals, steps=25, lr=0.01):
        samples = samples.clone().requires_grad_(True).to(device)
        optimizer = torch.optim.Adam([samples], lr=lr)
        original_embeddings = self.subject_model(originals.to(device)).detach()
        criterion = torch.nn.MSELoss(reduce="mean")
        loss_start = 0.0
        loss_end = 0.0
        
        for i in trange(steps):
            optimizer.zero_grad()
            loss = criterion(self.subject_model(samples), original_embeddings)
            if i == 0:
                loss_start = torch.sqrt(loss).item()
            if i == steps - 1:
                loss_end = torch.sqrt(loss).item()
            loss.backward()
            optimizer.step()
    
        print(f"Loss before refinement: {loss_start}, after refinement: {loss_end}")
        return samples.detach()
    
    @torch.enable_grad()
    def refine_in_ae_latent_space(self, samples, originals, steps=25, lr=0.01):
        assert self.autoencoder is not None, "Autoencoder must be provided for latent space refinement."
        with torch.no_grad():
            z_samples = self.autoencoder.encode(samples).detach()
        z_samples = z_samples.clone().requires_grad_(True).to(device)
        optimizer = torch.optim.Adam([z_samples], lr=lr)
        original_embeddings = self.subject_model(originals.to(device)).detach()
        criterion = torch.nn.MSELoss(reduce="mean")
        loss_start = 0.0
        loss_end = 0.0
        
        for i in trange(steps):
            optimizer.zero_grad()
            decoded = self.autoencoder.decode(z_samples)
            loss = criterion(self.subject_model(decoded), original_embeddings)
            if i == 0:
                loss_start = torch.sqrt(loss).item()
            if i == steps - 1:
                loss_end = torch.sqrt(loss).item()
            loss.backward()
            optimizer.step()
    
        print(f"Loss before refinement: {loss_start}, after refinement: {loss_end}")
        return self.autoencoder.decode(z_samples).detach().reshape(samples.shape)

class NoiseRobustnessChecker:
    def __init__(self, subject_model, autoencoder=None):
        self.subject_model = subject_model
        self.autoencoder = autoencoder
        self.subject_model.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()

    @torch.no_grad()
    def check_noise_robustness(self, sample, original, noise_levels=torch.logspace(-5, -1, 5), num_noise_per_level=1): 
        """
        Check the noise robustness of a sample against an original image.
        :param sample: The generated sample image.
        :param original: The original image to compare against.
        :param subject_model: The subject model to use for evaluation.
        :param noise_range: Range of noise levels to test.
        :param num_samples: Number of noise samples to generate.
        :return: Distances and noise levels.
        """
        assert sample.shape == original.shape, "Sample and original must have the same shape."
        sample = sample.to(device)
        original = original.to(device)
        original_embedding = self.subject_model(original).detach()
        sample_embedding = self.subject_model(sample).detach()
        initial_distance = torch.norm(original_embedding - sample_embedding, p=2).item()
        
        distances = [initial_distance]
        for noise_level in noise_levels:
            noise = torch.randn_like(sample) * noise_level
            distances_level = []
            for _ in range(num_noise_per_level):
                noisy_sample = sample + noise
                noisy_embedding = self.subject_model(noisy_sample).detach()
                distance = torch.norm(original_embedding - noisy_embedding, p=2).item()
                distances_level.append(distance)
            distances.append(np.mean(distances_level))     
        return distances, torch.cat((torch.zeros(1), noise_levels.cpu()), dim=0).numpy()

    @torch.no_grad()
    def check_noise_robustness_in_ae_latent_space(self, sample, original, noise_levels=torch.logspace(-5, -1, 5), num_noise_per_level=1): 
        """
        Check the noise robustness of a sample against an original image in the autoencoder latent space.
        :param sample: The generated sample image.
        :param original: The original image to compare against.
        :param subject_model: The subject model to use for evaluation.
        :param noise_range: Range of noise levels to test.
        :param num_samples: Number of noise samples to generate.
        :return: Distances and noise levels.
        """
        assert sample.shape == original.shape, "Sample and original must have the same shape."
        sample = sample.to(device)
        original = original.to(device)
        original_embedding = self.subject_model(original).detach()
        sample_embedding = self.subject_model(self.autoencoder.decode(self.autoencoder.encode(sample))).detach()
        initial_distance = torch.norm(original_embedding - sample_embedding, p=2).item()

        z_sample = self.autoencoder.encode(sample).detach()
        
        initial_distance = torch.norm(original_embedding - sample_embedding, p=2).item()
        
        distances = [initial_distance]
        for noise_level in noise_levels:
            noise = torch.randn_like(z_sample) * noise_level
            distances_level = []
            for _ in range(num_noise_per_level):
                noisy_z_sample = z_sample + noise
                noisy_sample = self.autoencoder.decode(noisy_z_sample)
                noisy_embedding = self.subject_model(noisy_sample).detach()
                distance = torch.norm(original_embedding - noisy_embedding, p=2).item()
                distances_level.append(distance)
            distances.append(np.mean(distances_level))     
        return distances, torch.cat((torch.zeros(1), noise_levels.cpu()), dim=0).numpy()

class NearestNeighborSearch:
    def __init__(self, train_ds, val_ds, test_ds, subject_model, batch_size=32):
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.subject_model = subject_model

    @torch.no_grad()
    def find_nearest_neighbor(self, original, use_datasets="all", identity_thresh=1e-5):
        if isinstance(use_datasets, str):
            use_train = use_datasets in ["train", "all"]
            use_val = use_datasets in ["val", "all"]
            use_test = use_datasets in ["test", "all"]
        elif isinstance(use_datasets, (list, tuple)):
            use_train = "train" in use_datasets
            use_val = "val" in use_datasets
            use_test = "test" in use_datasets

        best_dist = torch.inf
        best_sample = None
        if use_train:
            print("Searching train dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.train_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        if use_val:
            print("Searching val dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.val_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        if use_test:
            print("Searching test dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.test_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        return best_dist, best_sample

    @torch.no_grad()
    def search_in_dataset(self, original, loader, best_dist, best_sample, identity_thresh=1e-5):
        original_embedding = self.subject_model(original).detach()
        for batch in tqdm(loader):
            sample = batch[0].to(device).reshape(-1, *original.shape[1:])
            sample_embedding = self.subject_model(sample).detach()
            dist = torch.norm(original_embedding - sample_embedding, p=2, dim=1)
            dist, sample = dist[dist > identity_thresh], sample[dist > identity_thresh]
            min_in_batch = torch.argmin(dist)
            dist, sample = dist[min_in_batch], sample[min_in_batch]
            if dist < best_dist:
                best_dist = dist
                best_sample = sample
        return best_dist, best_sample

@dataclass
class TimestepConfig:
    t_start: int = 0
    t_end: int = 1000
    num_steps: int = 100
    seed: int = 0
    stride: str = "ddpm_uniform"
    root: str = "/path/to/experiment/root"
    name: str = "samples"
    ckpt_root: str = "/path/to/pretrained/checkpoints"
    samples_root: str = "/path/to/save/samples"
    overwrite: bool = True
    use_wandb: bool = False
    save_ori: bool = True
    save_deg: bool = True
    smoke_test: int = 1

@dataclass
class NDTMTimestepCompatability:
    exp: TimestepConfig = field(default_factory=TimestepConfig)

class StableDiffusionInterface(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = UNet2DModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def forward(self, x, t, y=None):
        # Assuming y is not used in this case
        noise_pred = self.model(x, t).sample
        return noise_pred