"""Denoising Diffusion Probabilistic Model (DDPM) implementation.

Formulation (Ho et al. 2020, Nichol & Dhariwal 2021):
    - Forward Process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    - Objective: E_{t, x_0, eps} [ || eps - eps_theta(x_t, t, condition) || ]
    - Reverse Transition: p_theta(x_{t-1} | x_t, condition)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nightvision.models.conditioning import ConditioningManager


def make_beta_schedule(
    schedule: str = "cosine",
    timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Compute beta variance schedule for diffusion forward process.

    Args:
        schedule: 'linear' or 'cosine' (Nichol & Dhariwal).
        timesteps: Total diffusion steps T.
        beta_start: Minimum beta for linear schedule.
        beta_end: Maximum beta for linear schedule.

    Returns:
        1D Tensor of shape (timesteps,) containing beta_t.
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    elif schedule == "cosine":
        # Cosine schedule from Improved DDPM
        steps = timesteps + 1
        s = 0.008
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, min=1e-5, max=0.999)

    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    """Encapsulates the discrete-time Gaussian diffusion process and training logic."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ) -> None:
        """Initialize GaussianDiffusion.

        Args:
            timesteps: Total timesteps T.
            beta_schedule: 'linear' or 'cosine'.
            beta_start: Start beta for linear schedule.
            beta_end: End beta for linear schedule.
            prediction_type: 'epsilon' (predicts noise) or 'x0'.
        """
        super().__init__()
        self.timesteps = timesteps
        self.prediction_type = prediction_type

        # Compute and register diffusion constants as persistent buffers
        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Buffers for forward diffusion q(x_t | x_0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Buffers for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20))
        )

        # Coefficients for posterior mean: mu_t = coef1 * x_0 + coef2 * x_t
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        """Extract coefficients at index t and broadcast to shape of x."""
        b = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the forward diffusion distribution q(x_t | x_0).

        Equation: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x_start: Target ground truth image x_0 of shape (B, C, H, W).
            t: Discrete timesteps of shape (B,).
            noise: Optional pre-sampled Gaussian noise epsilon.

        Returns:
            Noisy tensor x_t of shape (B, C, H, W).
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
        clip: bool = True,
    ) -> torch.Tensor:
        """Derive predicted x_0 from noisy sample x_t and predicted noise epsilon."""
        coef_x = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        coef_eps = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        pred_x0 = coef_x * x_t - coef_eps * eps
        if clip:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        return pred_x0

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the true posterior distribution q(x_{t-1} | x_t, x_0)."""
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_start + coef2 * x_t
        var = self._extract(self.posterior_variance, t, x_t.shape)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
        conditioning_manager: ConditioningManager | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Single reverse diffusion transition step p_theta(x_{t-1} | x_t)."""
        b = x_t.shape[0]

        # Prepare U-Net condition
        if conditioning_manager is not None:
            unet_in = conditioning_manager.prepare_unet_input(x_t, dark_image, illumination_map)
        else:
            unet_in = torch.cat([x_t, dark_image], dim=1)

        # Predict noise
        pred_noise = model(unet_in, t)

        # Derive predicted x_0
        pred_x0 = self.predict_x0_from_eps(x_t, t, pred_noise, clip=clip_denoised)

        # Compute posterior mean and variance
        mean, _, log_var = self.q_posterior_mean_variance(pred_x0, x_t, t)

        # Inject noise unless t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(b, 1, 1, 1)
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
        conditioning_manager: ConditioningManager | None = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Full DDPM reverse generation process across all T timesteps."""
        device = dark_image.device
        b, _, h, w = dark_image.shape

        # Start from pure standard Gaussian noise x_T ~ N(0, I)
        img = torch.randn((b, 3, h, w), device=device)

        for step in reversed(range(self.timesteps)):
            t = torch.full((b,), step, device=device, dtype=torch.long)
            img = self.p_sample(
                model=model,
                x_t=img,
                t=t,
                dark_image=dark_image,
                illumination_map=illumination_map,
                conditioning_manager=conditioning_manager,
                clip_denoised=clip_denoised,
            )

        return img
