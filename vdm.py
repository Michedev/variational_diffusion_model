"""
Diffusion model for continuous likelihood estimation.

Input:  historical trajectory (≥10 steps) + map_obs + scene context
Output: future trajectory (10 steps)

Data layout (from collect_expert_coordinates.py):
  coordinates : [num_agents, T, 2]   float16
  map_obs     : [map_points, feats]  float16  (static per episode)
  agent_ids   : [num_agents]         int16
  collisions  : [num_agents]         bool
"""

import math
import h5py
import numpy as np
from denoiser import ImageTransformerDenoiser
from learned_schedule import MonotonicScheduleNet
from p_ode import ProbabilityFlowODE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from torchdiffeq import odeint, odeint_adjoint
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import tensorguard as tg

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] int or float diffusion timestep
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)   # [B, dim]


def sample_k_hot(logits: torch.Tensor, k: int, deterministic: bool = False) -> torch.Tensor:
    """
    Sample k-hot vector from logits using Gumbel-Top-k sampling.
    
    This function samples a k-hot encoding where exactly k elements are set to 1 and
    the rest are 0. During training, it uses the Gumbel-Top-k trick with a straight-
    through estimator to allow gradients to flow. During evaluation, it can perform
    deterministic top-k selection.
    
    The straight-through estimator allows backpropagation through the discrete sampling
    by using soft probabilities in the forward pass but hard one-hot vectors in the
    backward pass.
    
    Args:
        logits: Input logits of shape (batch_size, num_classes). Unnormalized scores
                for each class.
        k: Number of hot elements to select. Must be <= num_classes.
        deterministic: If True, uses deterministic top-k selection without Gumbel noise.
                      Useful for evaluation/inference.
    
    Returns:
        k-hot vector of shape (batch_size, num_classes) with exactly k elements set to 1.
        During training, gradients flow through the soft probabilities.
    """
    if deterministic:
        top_k = torch.topk(logits, k, dim=1).indices
        z = torch.zeros_like(logits)
        z.scatter_(1, top_k, 1.0)
        return z

    # Gumbel-Top-k sampling with straight-through estimator
    uniform = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(uniform + 1e-12) + 1e-12)
    perturbed_logits = logits + gumbel

    top_k = torch.topk(perturbed_logits, k, dim=1).indices
    z_hard = torch.zeros_like(logits).scatter_(1, top_k, 1.0)

    z_soft = torch.sigmoid(perturbed_logits)
    return (z_hard - z_soft).detach() + z_soft


# ==========================================
# 1. Fourier Features (Input Space)
# ==========================================

class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=128, scale=1.0, include_original=True):
        """
        Args:
            input_dim (int): The dimensionality of the input feature (e.g., coordinates, channels).
            mapping_size (int): The number of random frequencies to generate. 
                                The output dimension will be mapping_size * 2.
            scale (float): The standard deviation of the Gaussian distribution. 
                           Controls the frequency spectrum (higher scale = higher frequencies).
            include_original (bool): Whether to concatenate the original input data.
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.include_original = include_original
        
        # Generate the random matrix B sampled from N(0, scale^2)
        # Shape: (input_dim, mapping_size)
        B = torch.randn(input_dim, mapping_size) * scale
        
        # Register B as a buffer. 
        # This ensures it is moved to the GPU with .to(device) and saved in the state_dict,
        # but it will NOT be updated by the optimizer (requires_grad=False by default).
        self.register_buffer('B', B)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input data of shape (..., input_dim)
        Returns:
            torch.Tensor: Encoded features of shape (..., mapping_size * 2 + [input_dim])
        """
        # Project the input features using the random matrix
        # x shape: (..., input_dim) @ B shape: (input_dim, mapping_size) -> (..., mapping_size)
        projected = x @ self.B
        
        # Multiply by 2*pi
        projected = 2.0 * math.pi * projected
        
        # Apply sin and cos
        sin_features = torch.sin(projected)
        cos_features = torch.cos(projected)
        
        # Concatenate features
        features = [sin_features, cos_features]
        if self.include_original:
            features.insert(0, x)
            
        # Final shape: (..., mapping_size * 2 + input_dim)
        return torch.cat(features, dim=-1)


class VDM(pl.LightningModule):
    """
    Variational Diffusion Model (VDM) for continuous likelihood estimation.
    """
    def __init__(
        self,
        # Image dims
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        # Denoiser dims
        t_emb_dim: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        # VDM Schedule params
        schedule_type: str = 'learned',
        gamma_min: float = -13.3,
        gamma_max: float = 5.0,
        # Optimization
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        # ODE solver params
        ode_method: str = 'dopri5',
        ode_atol: float = 1e-5,
        ode_rtol: float = 1e-5,
        ode_timesteps: int = 10,
        ode_adjoint: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.schedule_type = schedule_type
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        self.t_emb = SinusoidalPosEmb(t_emb_dim)
        
        self.denoiser = ImageTransformerDenoiser(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            t_emb_dim=t_emb_dim,
            num_layers=num_blocks
        )
        
        if self.schedule_type == 'learned':
            self.noise_scheduler = MonotonicScheduleNet(hidden_dim=32)

        # ODE hyperparameters
        self.ode_method = ode_method
        self.ode_atol = ode_atol
        self.ode_rtol = ode_rtol
        self.ode_adjoint = ode_adjoint

        self.norm_eps = 1e-6

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Log-SNR schedule."""
        if self.schedule_type == 'learned':
            return self.noise_scheduler(t)
        elif self.schedule_type == 'linear':
            out = self.gamma_min + t * (self.gamma_max - self.gamma_min)
            return out.view(-1, 1, 1)
        elif self.schedule_type == 'cosine':
            s = 0.008
            phi = ((t + s) / (1 + s) * math.pi / 2).clamp(0, math.pi / 2)
            alpha_sq = torch.cos(phi) ** 2
            alpha_sq = torch.clamp(alpha_sq, min=1e-10, max=0.9999999)
            out = torch.log(1 - alpha_sq) - torch.log(alpha_sq)
            return out.view(-1, 1, 1)
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

    def alpha_sigma(self, t: torch.Tensor):
        """Compute alpha_t and sigma_t from gamma(t)."""
        g = self.gamma(t)
        alpha = torch.sqrt(torch.sigmoid(-g))
        sigma = torch.sqrt(torch.sigmoid(g))
        return alpha, sigma
        
    def forward(self, noisy_x, t):
        """
        Forward pass through denoiser.
        """
        t_emb = self.t_emb(t)
        return self.denoiser(noisy_x, t_emb)

    def training_step(self, batch, batch_idx):
        x0 = batch['x'].float()
        B = x0.shape[0]

        # Continuous time sampling t ~ U(0, 1)
        t = torch.rand((B,), device=self.device)
        
        loss = self._forward_diffusion(x0, t)

        # latent loss
        t_0 = torch.zeros_like(t)
        gamma_0, gamma_1 = self.gamma(t_0), self.gamma(torch.ones_like(t))
        var_0, var_1 = gamma_0.sigmoid(), gamma_1.sigmoid()

        # prior loss
        mean1_sqr = (1. - var_1) * x0
        loss_kl_prior = 0.5 * torch.sum((mean1_sqr + var_1 - torch.log(var_1) - 1.).flatten(1), dim=1)
        loss_kl_prior = loss_kl_prior.mean()

        # recon loss
        z_0 = torch.sqrt(1. - var_0) * x0 + torch.sqrt(var_0) * torch.randn_like(x0)
        v_hat_0 = self(z_0, t_0)
        x_recon = torch.sqrt(1. - var_0) * z_0 - torch.sqrt(var_0) * v_hat_0

        snr_0 = torch.exp(-gamma_0)
        loss_recon = 0.5 * snr_0 * torch.pow((x0 - x_recon).flatten(1) , 2)
        loss_recon = loss_recon.mean()

        loss = loss.mean() + 0.1 * loss_kl_prior + 0.1 * loss_recon

        if self.global_step % 100 == 0:
            self.log('train/loss', loss, prog_bar=True,)
        return loss

    def _forward_diffusion(self, x0, t):
        alpha_t, sigma_t = self.alpha_sigma(t)
        
        # Ensure dimensions match for broadcasting on images
        while len(alpha_t.shape) < len(x0.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        noise = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * noise
        
        pred_noise = self(x_t, t)
        
        loss = F.mse_loss(pred_noise, noise, reduction='none')
        return loss

    def validation_step(self, batch, batch_idx):
        t_start_denoise = 0.5

        x0 = batch['x'].float()
        B = x0.shape[0]

        t_start_tensor = torch.full((B,), t_start_denoise, device=self.device)
        alpha_t, sigma_t = self.alpha_sigma(t_start_tensor)
        
        while len(alpha_t.shape) < len(x0.shape):
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        z_start = alpha_t * x0 + sigma_t * torch.randn_like(x0)

        ode_func = ProbabilityFlowODE(
            self, B, x0.shape[1:]
        )
        state_init = (z_start, torch.zeros(B, 1, device=self.device))
        
        t_steps = torch.linspace(t_start_denoise, 0.0, steps=self.hparams.ode_timesteps, device=self.device)
        step_sz = (t_start_denoise - 0.0) / (self.hparams.ode_timesteps - 1)

        _odeint = odeint_adjoint if self.ode_adjoint else odeint
        with torch.no_grad():
            state_traj = _odeint(
                ode_func, state_init, t_steps,
                method=self.ode_method,
                options={'step_size': step_sz}
            )

        z_traj, logp_traj = state_traj
        z_0_pred, delta_logp = z_traj[-1], logp_traj[-1]

        # Per-sample reconstruction error [B]
        recon_per_sample = F.mse_loss(z_0_pred, x0, reduction='none').flatten(1).mean(dim=1)

        # Per-sample log-likelihood [B]
        log_p_z_start = -0.5 * torch.sum(z_start.flatten(1)**2 + math.log(2 * math.pi), dim=1, keepdim=True)
        log_likelihood_per_sample = (log_p_z_start + delta_logp).squeeze(1)  # [B]

        self.log('val_recon_loss', recon_per_sample.mean())
        self.log('val_log_likelihood', log_likelihood_per_sample.mean())


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return (epoch + 1) / max(self.warmup_epochs, 1)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(self.total_epochs - self.warmup_epochs, 1),
            eta_min=self.lr * 1e-2,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}