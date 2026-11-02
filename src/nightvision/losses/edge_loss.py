"""Edge preservation loss module for Model C.

Extracts gradient contours using 2D Sobel operators and penalizes discrepancies
in high-frequency structural edges between the reconstructed image and the reference.

Mathematical Formulation:
    G_x = [[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]
    G_y = [[-1, -2, -1], [ 0,  0,  0], [+1, +2, +1]]
    |grad(I)| = sqrt( (conv2d(I, G_x))^2 + (conv2d(I, G_y))^2 + eps )
    L_edge(I_hat, I_ref) = (1 / N) * || |grad(I_hat)| - |grad(I_ref)| ||_1

Total Objective for Model C:
    L_total = L_diffusion + lambda_edge * L_edge
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SobelFilter(nn.Module):
    """Computes spatial gradient magnitude using separable 2D Sobel filters."""

    def __init__(self, in_channels: int = 3, eps: float = 1e-6) -> None:
        """Initialize SobelFilter.

        Args:
            in_channels: Number of input image channels (typically 3 for RGB).
            eps: Epsilon to ensure numerical stability during square root.
        """
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps

        # Sobel horizontal and vertical kernels
        gx = (
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
            )
            / 8.0
        )
        gy = (
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32
            )
            / 8.0
        )

        # Reshape to depthwise conv kernels (C, 1, 3, 3)
        kernel_x = gx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        kernel_y = gy.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute edge gradients.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tuple (gradient_magnitude, grad_x, grad_y), where gradient_magnitude has shape (B, C, H, W).
        """
        # Replicate padding on borders
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, self.kernel_x, groups=self.in_channels)
        gy = F.conv2d(x_pad, self.kernel_y, groups=self.in_channels)

        mag = torch.sqrt(gx**2 + gy**2 + self.eps)
        return mag, gx, gy


class EdgeLoss(nn.Module):
    """Edge preservation loss with configurable lambda_edge weight."""

    def __init__(
        self,
        lambda_edge: float = 0.05,
        target_mode: str = "target",
        loss_norm: str = "l1",
        in_channels: int = 3,
    ) -> None:
        """Initialize EdgeLoss.

        Args:
            lambda_edge: Weight hyperparameter lambda for edge loss term.
            target_mode: 'target' (compare with clean reference x_0),
                         'input' (compare with dark image I_s),
                         or 'both' (average comparison against target and input).
            loss_norm: 'l1' (MAE) or 'l2' (MSE).
            in_channels: Input image channel count.
        """
        super().__init__()
        self.lambda_edge = float(lambda_edge)
        self.target_mode = target_mode.lower()
        self.loss_norm = loss_norm.lower()
        self.sobel = SobelFilter(in_channels=in_channels)

    def forward(
        self,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        input_dark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute weighted edge loss.

        Args:
            pred_x0: Reconstructed target image estimate of shape (B, 3, H, W).
            target_x0: Clean ground truth reference image of shape (B, 3, H, W).
            input_dark: Optional input dark image of shape (B, 3, H, W).

        Returns:
            Weighted edge loss tensor: lambda_edge * L_edge.
        """
        if self.lambda_edge <= 0.0:
            return torch.tensor(0.0, device=pred_x0.device, dtype=pred_x0.dtype)

        pred_mag, _, _ = self.sobel(pred_x0)

        if self.target_mode == "target" or input_dark is None:
            ref_mag, _, _ = self.sobel(target_x0)
            if self.loss_norm == "l1":
                edge_loss = F.l1_loss(pred_mag, ref_mag)
            else:
                edge_loss = F.mse_loss(pred_mag, ref_mag)

        elif self.target_mode == "input":
            ref_mag, _, _ = self.sobel(input_dark)
            if self.loss_norm == "l1":
                edge_loss = F.l1_loss(pred_mag, ref_mag)
            else:
                edge_loss = F.mse_loss(pred_mag, ref_mag)

        elif self.target_mode == "both":
            ref_mag_target, _, _ = self.sobel(target_x0)
            ref_mag_dark, _, _ = self.sobel(input_dark)
            loss_t = (
                F.l1_loss(pred_mag, ref_mag_target)
                if self.loss_norm == "l1"
                else F.mse_loss(pred_mag, ref_mag_target)
            )
            loss_d = (
                F.l1_loss(pred_mag, ref_mag_dark)
                if self.loss_norm == "l1"
                else F.mse_loss(pred_mag, ref_mag_dark)
            )
            edge_loss = 0.5 * (loss_t + loss_d)
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        return self.lambda_edge * edge_loss


class CombinedDiffusionLoss(nn.Module):
    """Combines DDPM noise prediction loss with optional edge preservation loss."""

    def __init__(
        self,
        diffusion_loss_type: str = "l1",
        lambda_edge: float = 0.0,
        edge_target: str = "target",
        edge_norm: str = "l1",
    ) -> None:
        """Initialize CombinedDiffusionLoss.

        Args:
            diffusion_loss_type: 'l1', 'l2', or 'smooth_l1'.
            lambda_edge: Weight for edge loss (0.0 for Models A & B, > 0.0 for Model C).
            edge_target: 'target', 'input', or 'both'.
            edge_norm: 'l1' or 'l2'.
        """
        super().__init__()
        from nightvision.losses.diffusion_loss import DiffusionLoss

        self.diff_loss = DiffusionLoss(loss_type=diffusion_loss_type)
        self.edge_loss = EdgeLoss(
            lambda_edge=lambda_edge,
            target_mode=edge_target,
            loss_norm=edge_norm,
        )

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        pred_x0: torch.Tensor | None = None,
        target_x0: torch.Tensor | None = None,
        input_dark: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss and return breakdown dictionary."""
        l_diff = self.diff_loss(noise_pred, noise_target)
        breakdown = {"l_diff": l_diff.item()}

        if self.edge_loss.lambda_edge > 0.0 and pred_x0 is not None and target_x0 is not None:
            l_edge = self.edge_loss(pred_x0, target_x0, input_dark)
            l_total = l_diff + l_edge
            breakdown["l_edge"] = l_edge.item()
            breakdown["l_total"] = l_total.item()
        else:
            l_total = l_diff
            breakdown["l_edge"] = 0.0
            breakdown["l_total"] = l_total.item()

        return l_total, breakdown
