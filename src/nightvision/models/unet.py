"""Conditional U-Net architecture for 128x128 diffusion generation.

Designed according to DDPM best practices (Ho et al., Nichol & Dhariwal):
    - Residual convolution blocks with GroupNorm and SiLU activations
    - Sinusoidal timestep embeddings integrated into every residual block
    - Self-attention at low-resolution bottlenecks (16x16)
    - Skip connections between encoder and decoder levels
    - Configurable in_channels (6 for Model A, 7 for Models B & C)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nightvision.models.embeddings import SinusoidalTimeEmbedding, TimeEmbeddingMLP


def _get_num_groups(channels: int, max_groups: int = 32) -> int:
    """Find the largest divisor of channels that is <= max_groups."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SelfAttention(nn.Module):
    """Spatial Self-Attention block for diffusion feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        # Ensure num_heads divides channels
        self.num_heads = num_heads if channels % num_heads == 0 else 1
        self.head_dim = channels // self.num_heads

        groups = _get_num_groups(channels, 32)
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = 1.0 / (self.head_dim**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        norm_x = self.norm(x)

        qkv = self.qkv(norm_x)  # (B, 3*C, H, W)
        qkv = qkv.reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each is (B, heads, head_dim, HW)

        # Transpose for matrix multiplication: (B, heads, HW, head_dim) @ (B, heads, head_dim, HW)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        attn_scores = torch.matmul(q, k) * self.scale  # (B, heads, HW, HW)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, v.permute(0, 1, 3, 2))  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)  # (B, C, H, W)
        out = self.proj(out)
        return out + residual


class ResnetBlock(nn.Module):
    """Residual convolutional block with timestep embedding projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.1,
        norm_groups: int = 32,
    ) -> None:
        """Initialize ResnetBlock.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            time_embed_dim: Dimension of time embedding MLP projection.
            dropout: Dropout probability.
            norm_groups: Number of groups for GroupNorm.
        """
        super().__init__()
        groups_in = _get_num_groups(in_channels, norm_groups)
        groups_out = _get_num_groups(out_channels, norm_groups)

        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Project time embedding and add across spatial dimensions
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Strided 2D convolution for 2x downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Nearest neighbor 2x upsampling followed by 3x3 convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """Complete Conditional U-Net for 128x128 Denoising Diffusion.

    Takes concatenated (noisy target + condition) tensor and timestep integer,
    predicts the Gaussian noise epsilon of shape (B, out_channels, 128, 128).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        norm_groups: int = 32,
    ) -> None:
        """Initialize ConditionalUNet.

        Args:
            in_channels: 6 for Model A, 7 for Models B & C.
            out_channels: 3 (RGB noise prediction).
            base_channels: Starting convolution channel width.
            channel_mults: Multipliers per resolution level (e.g. 1, 2, 4, 8).
            num_res_blocks: Residual blocks per level.
            attention_resolutions: Spatial sizes where SelfAttention is applied (e.g. 16).
            dropout: Dropout probability.
            time_embed_dim: Dimension of time embedding representation.
            norm_groups: Group count for GroupNorm layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim

        # Time embedding pipeline
        sinusoidal_dim = time_embed_dim // 2
        self.time_embed_generator = SinusoidalTimeEmbedding(sinusoidal_dim)
        self.time_mlp = TimeEmbeddingMLP(sinusoidal_dim, time_embed_dim)

        # Initial projection convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder (Downsampling path)
        self.down_blocks = nn.ModuleList()
        current_ch = base_channels
        feat_channels = [current_ch]
        current_res = 128

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResnetBlock(
                    in_channels=current_ch,
                    out_channels=out_ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                attn = (
                    SelfAttention(out_ch) if current_res in attention_resolutions else nn.Identity()
                )
                self.down_blocks.append(nn.ModuleDict({"res": block, "attn": attn}))
                current_ch = out_ch
                feat_channels.append(current_ch)

            # Downsample if not the lowest resolution
            if level != len(channel_mults) - 1:
                self.down_blocks.append(nn.ModuleDict({"downsample": Downsample(current_ch)}))
                feat_channels.append(current_ch)
                current_res = current_res // 2

        # Bottleneck (Middle block)
        self.mid_block1 = ResnetBlock(
            current_ch, current_ch, time_embed_dim, dropout=dropout, norm_groups=norm_groups
        )
        self.mid_attn = SelfAttention(current_ch)
        self.mid_block2 = ResnetBlock(
            current_ch, current_ch, time_embed_dim, dropout=dropout, norm_groups=norm_groups
        )

        # Decoder (Upsampling path)
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = feat_channels.pop()
                block = ResnetBlock(
                    in_channels=current_ch + skip_ch,
                    out_channels=out_ch,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    norm_groups=norm_groups,
                )
                attn = (
                    SelfAttention(out_ch) if current_res in attention_resolutions else nn.Identity()
                )
                self.up_blocks.append(nn.ModuleDict({"res": block, "attn": attn}))
                current_ch = out_ch

            if level != 0:
                self.up_blocks.append(nn.ModuleDict({"upsample": Upsample(current_ch)}))
                current_res = current_res * 2

        # Final output projection
        groups_final = _get_num_groups(current_ch, norm_groups)
        self.final_norm = nn.GroupNorm(groups_final, current_ch)
        self.final_conv = nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass of U-Net.

        Args:
            x: Concatenated input tensor of shape (B, in_channels, H, W).
            timesteps: 1D tensor of shape (B,) with diffusion timesteps.

        Returns:
            Predicted noise tensor of shape (B, out_channels, H, W).
        """
        # Embed timesteps
        sinusoidal_emb = self.time_embed_generator(timesteps)
        time_emb = self.time_mlp(sinusoidal_emb)  # (B, time_embed_dim)

        # Initial conv
        h = self.init_conv(x)
        skips = [h]

        # Encoder path
        for module_dict in self.down_blocks:
            if "downsample" in module_dict:
                h = module_dict["downsample"](h)
                skips.append(h)
            else:
                h = module_dict["res"](h, time_emb)
                h = module_dict["attn"](h)
                skips.append(h)

        # Bottleneck path
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        # Decoder path
        for module_dict in self.up_blocks:
            if "upsample" in module_dict:
                h = module_dict["upsample"](h)
            else:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = module_dict["res"](h, time_emb)
                h = module_dict["attn"](h)

        # Final projection to RGB noise
        h = F.silu(self.final_norm(h))
        out = self.final_conv(h)
        return out
