"""Denoising Diffusion Implicit Models (DDIM) sampling module.

Implements accelerated deterministic or stochastic sampling (Song et al. 2020):
    - Sub-sequence of S timesteps chosen from total T timesteps (e.g. 25, 50, 100 steps)
    - Interpolation parameter eta (eta = 0.0: deterministic ODE; eta = 1.0: DDPM)
    - GPU synchronization for accurate wall-clock latency measurement
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from nightvision.models.conditioning import ConditioningManager
from nightvision.models.ddpm import GaussianDiffusion
from nightvision.utils.device import synchronize_device


class DDIMSampler:
    """Accelerated DDIM sampler compatible with conditional DDPM."""

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        ddim_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize DDIMSampler.

        Args:
            diffusion: GaussianDiffusion instance containing precomputed alphas.
            ddim_steps: Number of sampling steps S (e.g. 25, 50, 100).
            eta: Stochasticity weight eta in [0, 1]. eta=0 is deterministic.
            clip_denoised: Whether to clamp predicted x_0 in [-1, 1].
        """
        self.diffusion = diffusion
        self.total_timesteps = diffusion.timesteps
        self.ddim_steps = min(ddim_steps, self.total_timesteps)
        self.eta = float(eta)
        self.clip_denoised = clip_denoised

        # Build sub-sequence of timesteps tau_1, ..., tau_S
        step_ratio = self.total_timesteps // self.ddim_steps
        self.timesteps_subseq = np.asarray(list(range(0, self.total_timesteps, step_ratio)))[
            : self.ddim_steps
        ]

    def set_ddim_steps(self, ddim_steps: int) -> None:
        """Update sampling step count dynamically."""
        self.ddim_steps = min(ddim_steps, self.total_timesteps)
        step_ratio = max(1, self.total_timesteps // self.ddim_steps)
        self.timesteps_subseq = np.asarray(list(range(0, self.total_timesteps, step_ratio)))[
            : self.ddim_steps
        ]

    @torch.no_grad()
    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        step_idx: int,
        t_curr: int,
        t_prev: int,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
        conditioning_manager: ConditioningManager | None = None,
    ) -> torch.Tensor:
        """Execute a single DDIM reverse step from t_curr down to t_prev.

        Args:
            model: Conditional U-Net.
            x_t: Latent state at timestep t_curr.
            step_idx: Index within DDIM sequence.
            t_curr: Current integer timestep index.
            t_prev: Previous integer timestep index (-1 for final step).
            dark_image: Conditioning low-light image (B, 3, H, W).
            illumination_map: Optional Retinex map (B, 1, H, W).
            conditioning_manager: ConditioningManager for Model A, B, or C.

        Returns:
            x_{t_prev}: Latent state at previous timestep.
        """
        b = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((b,), t_curr, device=device, dtype=torch.long)

        # Prepare conditioned U-Net input
        if conditioning_manager is not None:
            unet_in = conditioning_manager.prepare_unet_input(x_t, dark_image, illumination_map)
        else:
            unet_in = torch.cat([x_t, dark_image], dim=1)

        # Predict noise epsilon_theta
        eps_theta = model(unet_in, t_tensor)

        # Alphas at current and previous steps
        alpha_bar_curr = self.diffusion.alphas_cumprod[t_curr]
        alpha_bar_prev = (
            self.diffusion.alphas_cumprod[t_prev]
            if t_prev >= 0
            else torch.tensor(1.0, device=device)
        )

        # 1. Derive predicted clean image x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_curr) * eps_theta) / torch.sqrt(alpha_bar_curr)
        if self.clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 2. Compute variance sigma_t
        if self.eta > 0.0 and t_prev >= 0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_bar_prev)
                / (1.0 - alpha_bar_curr)
                * (1.0 - alpha_bar_curr / alpha_bar_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=device)

        # 3. Direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps_theta

        # 4. Stochastic noise (if eta > 0)
        noise = torch.randn_like(x_t) if sigma > 0.0 else torch.zeros_like(x_t)

        # 5. Compute x_{t_prev}
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
        conditioning_manager: ConditioningManager | None = None,
        measure_timing: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, float]:
        """Run accelerated DDIM sampling loop from noise to restored image.

        Args:
            model: Conditional U-Net.
            dark_image: Low-light condition tensor (B, 3, H, W).
            illumination_map: Optional illumination map tensor (B, 1, H, W).
            conditioning_manager: ConditioningManager for Model A, B, or C.
            measure_timing: If True, returns (restored_tensor, inference_seconds).

        Returns:
            Restored image tensor in [-1, 1], or tuple (restored_tensor, elapsed_sec).
        """
        device = dark_image.device
        b, _, h, w = dark_image.shape

        if measure_timing:
            synchronize_device(device)
            start_time = time.perf_counter()

        # Start from pure standard Gaussian noise x_T ~ N(0, I)
        img = torch.randn((b, 3, h, w), device=device)

        # Traverse reverse sub-sequence
        reversed_subseq = list(reversed(self.timesteps_subseq))
        for i, t_curr in enumerate(reversed_subseq):
            t_prev = reversed_subseq[i + 1] if i + 1 < len(reversed_subseq) else -1
            img = self.ddim_step(
                model=model,
                x_t=img,
                step_idx=i,
                t_curr=t_curr,
                t_prev=t_prev,
                dark_image=dark_image,
                illumination_map=illumination_map,
                conditioning_manager=conditioning_manager,
            )

        if measure_timing:
            synchronize_device(device)
            elapsed_sec = time.perf_counter() - start_time
            return img, elapsed_sec

        return img
