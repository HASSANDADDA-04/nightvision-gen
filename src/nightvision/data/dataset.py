"""PyTorch Dataset classes for LOL-v2 and ExDark.

Provides:
    - LOLv2PatchDataset: Loads 128x128 patches with paired augmentations for DDPM training.
    - LOLv2Dataset: Loads full-resolution image pairs for evaluation.
    - ExDarkDataset: Loads ExDark low-light images and annotations for object detection.
    - SyntheticPairDataset: Synthetic dataset for fast CPU testing and unit tests without downloading raw data.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from nightvision.data.preprocessing import PairedAugmenter, normalize_to_neg1_1
from nightvision.data.split_manager import ImagePair
from nightvision.illumination.retinex import RetinexEstimator

logger = logging.getLogger(__name__)


class LOLv2PatchDataset(Dataset):
    """PyTorch Dataset yielding 128x128 patches for conditional DDPM training."""

    def __init__(
        self,
        pairs: list[ImagePair],
        patch_size: int = 128,
        patches_per_image: int = 4,
        is_train: bool = True,
        compute_retinex: bool = True,
        augment: bool = True,
        retinex_sigma: float = 3.0,
    ) -> None:
        """Initialize LOLv2PatchDataset.

        Args:
            pairs: List of ImagePair instances.
            patch_size: Square patch dimensions (128x128).
            patches_per_image: Number of random patches sampled per image pair per epoch.
            is_train: Whether this is the training set (enables random cropping).
            compute_retinex: Whether to compute/yield illumination maps.
            augment: Whether to apply synchronous paired spatial augmentations.
            retinex_sigma: Smoothing sigma for Retinex.
        """
        self.pairs = pairs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.is_train = is_train
        self.compute_retinex = compute_retinex
        self.augmenter = PairedAugmenter() if (augment and is_train) else None
        self.retinex_estimator = RetinexEstimator(sigma=retinex_sigma) if compute_retinex else None

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair_idx = idx // self.patches_per_image
        pair = self.pairs[pair_idx]

        # Load images as float tensors in [0, 1]
        with Image.open(pair.low_path) as img_low, Image.open(pair.normal_path) as img_norm:
            low_np = np.array(img_low.convert("RGB"), dtype=np.float32) / 255.0
            norm_np = np.array(img_norm.convert("RGB"), dtype=np.float32) / 255.0

        h, w, _ = low_np.shape
        p = self.patch_size

        # Crop 128x128
        if h < p or w < p:
            # Pad
            pad_h = max(0, p - h)
            pad_w = max(0, p - w)
            low_np = np.pad(low_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            norm_np = np.pad(norm_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = low_np.shape

        if self.is_train:
            top = np.random.randint(0, h - p + 1)
            left = np.random.randint(0, w - p + 1)
        else:
            top = (h - p) // 2
            left = (w - p) // 2

        low_patch = low_np[top : top + p, left : left + p, :]
        norm_patch = norm_np[top : top + p, left : left + p, :]

        # Compute Retinex illumination map
        if self.compute_retinex and self.retinex_estimator is not None:
            illum_patch = self.retinex_estimator.estimate_from_numpy(
                low_patch
            )  # (P, P, 1) in [0, 1]
            illum_t = torch.from_numpy(illum_patch).permute(2, 0, 1).float()  # (1, P, P)
        else:
            illum_t = torch.zeros((1, p, p), dtype=torch.float32)

        # Convert RGB to tensors: (P, P, 3) -> (3, P, P)
        low_t = torch.from_numpy(low_patch).permute(2, 0, 1).float()
        norm_t = torch.from_numpy(norm_patch).permute(2, 0, 1).float()

        # Synchronous data augmentation
        if self.augmenter is not None:
            low_t, norm_t, illum_t = self.augmenter(low_t, norm_t, illum_t)

        # Normalize RGB to [-1, 1] for diffusion process
        # Illumination map is kept in [0, 1] or [-1, 1] (we map to [-1, 1] for conditioning uniformity)
        low_t_norm = normalize_to_neg1_1(low_t)
        norm_t_norm = normalize_to_neg1_1(norm_t)
        illum_t_norm = normalize_to_neg1_1(illum_t)

        return {
            "low": low_t_norm,
            "normal": norm_t_norm,
            "illumination": illum_t_norm,
            "low_orig": low_t,
            "normal_orig": norm_t,
            "pair_id": pair.pair_id,
        }


class LOLv2FullImageDataset(Dataset):
    """PyTorch Dataset yielding full-resolution image pairs for evaluation."""

    def __init__(
        self,
        pairs: list[ImagePair],
        compute_retinex: bool = True,
        retinex_sigma: float = 3.0,
    ) -> None:
        """Initialize LOLv2FullImageDataset.

        Args:
            pairs: List of ImagePair instances.
            compute_retinex: Whether to compute Retinex illumination maps.
            retinex_sigma: Smoothing sigma for Retinex.
        """
        self.pairs = pairs
        self.compute_retinex = compute_retinex
        self.retinex_estimator = RetinexEstimator(sigma=retinex_sigma) if compute_retinex else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[idx]
        with Image.open(pair.low_path) as img_low, Image.open(pair.normal_path) as img_norm:
            low_np = np.array(img_low.convert("RGB"), dtype=np.float32) / 255.0
            norm_np = np.array(img_norm.convert("RGB"), dtype=np.float32) / 255.0

        low_t = torch.from_numpy(low_np).permute(2, 0, 1).float()
        norm_t = torch.from_numpy(norm_np).permute(2, 0, 1).float()

        if self.compute_retinex and self.retinex_estimator is not None:
            illum_np = self.retinex_estimator.estimate_from_numpy(low_np)
            illum_t = torch.from_numpy(illum_np).permute(2, 0, 1).float()
        else:
            illum_t = torch.zeros((1, low_t.shape[1], low_t.shape[2]), dtype=torch.float32)

        return {
            "low": normalize_to_neg1_1(low_t),
            "normal": normalize_to_neg1_1(norm_t),
            "illumination": normalize_to_neg1_1(illum_t),
            "low_orig": low_t,
            "normal_orig": norm_t,
            "pair_id": pair.pair_id,
        }


class SyntheticPairDataset(Dataset):
    """Lightweight synthetic paired dataset for unit tests and CPU pipeline verification.

    Generates smooth gradients and geometric shapes simulating dark/bright image pairs.
    """

    def __init__(
        self,
        length: int = 16,
        image_size: int = 128,
        compute_retinex: bool = True,
    ) -> None:
        """Initialize SyntheticPairDataset.

        Args:
            length: Number of synthetic samples in dataset.
            image_size: Square dimensions (default: 128x128).
            compute_retinex: Whether to generate Retinex illumination maps.
        """
        self.length = length
        self.image_size = image_size
        self.compute_retinex = compute_retinex
        self.retinex_estimator = RetinexEstimator(sigma=2.0) if compute_retinex else None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        p = self.image_size
        # Deterministic generation based on index
        x = np.linspace(0, 1, p, dtype=np.float32)
        y = np.linspace(0, 1, p, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Well-lit synthetic image
        r = 0.5 + 0.4 * np.sin(2 * np.pi * xx + idx * 0.1)
        g = 0.5 + 0.4 * np.cos(2 * np.pi * yy + idx * 0.1)
        b = 0.5 + 0.3 * np.sin(2 * np.pi * (xx + yy))
        normal_np = np.stack([r, g, b], axis=-1)
        normal_np = np.clip(normal_np, 0.0, 1.0)

        # Darkened image simulating low-light capture (non-linear attenuation + low noise)
        low_np = normal_np**2.5 * 0.25 + 0.02 * np.random.RandomState(idx).randn(p, p, 3).astype(
            np.float32
        )
        low_np = np.clip(low_np, 0.0, 1.0)

        if self.compute_retinex and self.retinex_estimator is not None:
            illum_np = self.retinex_estimator.estimate_from_numpy(low_np)
            illum_t = torch.from_numpy(illum_np).permute(2, 0, 1).float()
        else:
            illum_t = torch.zeros((1, p, p), dtype=torch.float32)

        low_t = torch.from_numpy(low_np).permute(2, 0, 1).float()
        norm_t = torch.from_numpy(normal_np).permute(2, 0, 1).float()

        return {
            "low": normalize_to_neg1_1(low_t),
            "normal": normalize_to_neg1_1(norm_t),
            "illumination": normalize_to_neg1_1(illum_t),
            "low_orig": low_t,
            "normal_orig": norm_t,
            "pair_id": f"synthetic_{idx:03d}",
        }
