"""Preprocessing, tensor normalization, and paired data augmentation module.

Diffusion models standardly work with images normalized to [-1, 1].
All spatial data augmentations (e.g. random horizontal/vertical flips, 90-degree rotations)
must be applied synchronously across paired inputs (low, reference, illumination map).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def normalize_to_neg1_1(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize image tensor from [0, 1] to [-1, 1].

    Args:
        tensor: Tensor with values in [0, 1].

    Returns:
        Tensor with values in [-1, 1].
    """
    return tensor * 2.0 - 1.0


def denormalize_to_0_1(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor from [-1, 1] to [0, 1].

    Args:
        tensor: Tensor with values in [-1, 1].

    Returns:
        Tensor with values clamped in [0, 1].
    """
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def load_image_tensor(
    path: str | Path,
    target_size: tuple[int, int] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Load an image from disk as a float32 RGB tensor in [-1, 1] or [0, 1].

    Args:
        path: Path to image file.
        target_size: Optional (H, W) to resize.
        normalize: If True, maps to [-1, 1]. If False, maps to [0, 1].

    Returns:
        Tensor of shape (3, H, W).
    """
    path = Path(path)
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        if target_size is not None:
            img_rgb = img_rgb.resize((target_size[1], target_size[0]), Image.BICUBIC)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    if normalize:
        tensor = normalize_to_neg1_1(tensor)
    return tensor


class PairedAugmenter:
    """Synchronous paired data augmentations for paired low/normal/illumination images."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.2,
        rot90_prob: float = 0.2,
    ) -> None:
        """Initialize PairedAugmenter.

        Args:
            hflip_prob: Probability of horizontal flip.
            vflip_prob: Probability of vertical flip.
            rot90_prob: Probability of 90-degree rotation.
        """
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob

    def __call__(
        self,
        low: torch.Tensor,
        normal: torch.Tensor,
        illumination: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply synchronous spatial augmentations.

        Args:
            low: Tensor (C, H, W).
            normal: Tensor (C, H, W).
            illumination: Optional tensor (1, H, W).

        Returns:
            Tuple of transformed tensors.
        """
        # Horizontal flip
        if random.random() < self.hflip_prob:
            low = TF.hflip(low)
            normal = TF.hflip(normal)
            if illumination is not None:
                illumination = TF.hflip(illumination)

        # Vertical flip
        if random.random() < self.vflip_prob:
            low = TF.vflip(low)
            normal = TF.vflip(normal)
            if illumination is not None:
                illumination = TF.vflip(illumination)

        # Random 90-degree rotation
        if random.random() < self.rot90_prob:
            k = random.choice([1, 2, 3])
            low = torch.rot90(low, k=k, dims=[-2, -1])
            normal = torch.rot90(normal, k=k, dims=[-2, -1])
            if illumination is not None:
                illumination = torch.rot90(illumination, k=k, dims=[-2, -1])

        return low, normal, illumination
