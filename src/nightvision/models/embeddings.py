"""Time and positional embeddings for the diffusion process.

Implements sinusoidal step embeddings (Vaswani et al., Ho et al.)
projected through a two-layer MLP with SiLU activation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time-step embedding mapping integer timesteps to vectors."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize SinusoidalTimeEmbedding.

        Args:
            embed_dim: Output embedding dimension (must be even).
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be divisible by 2, got {embed_dim}")
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embeddings.

        Args:
            timesteps: 1D Tensor of shape (B,) containing integer time steps.

        Returns:
            Tensor of shape (B, embed_dim).
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2
        # Frequencies: exp(-log(10000) * i / (half_dim - 1))
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(-emb_scale * torch.arange(half_dim, dtype=torch.float32, device=device))

        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, embed_dim)
        return embedding


class TimeEmbeddingMLP(nn.Module):
    """Projects sinusoidal embeddings through a 2-layer MLP."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize TimeEmbeddingMLP.

        Args:
            in_features: Base sinusoidal feature dimension.
            out_features: Projected dimension used across U-Net ResBlocks.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
