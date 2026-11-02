"""Diffusion loss functions for training conditional DDPM models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DiffusionLoss(nn.Module):
    """Calculates regression loss between predicted noise and sampled Gaussian noise."""

    def __init__(self, loss_type: str = "l1") -> None:
        """Initialize DiffusionLoss.

        Args:
            loss_type: 'l1' (MAE), 'l2' (MSE), or 'smooth_l1'.
        """
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type not in ("l1", "l2", "smooth_l1"):
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'l1', 'l2', or 'smooth_l1'.")

    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """Compute diffusion noise prediction loss.

        Args:
            noise_pred: Predicted noise epsilon_theta (B, 3, H, W).
            noise_target: Sampled true noise epsilon (B, 3, H, W).

        Returns:
            Scalar loss tensor.
        """
        if self.loss_type == "l1":
            return F.l1_loss(noise_pred, noise_target)
        elif self.loss_type == "l2":
            return F.mse_loss(noise_pred, noise_target)
        elif self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(noise_pred, noise_target)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
