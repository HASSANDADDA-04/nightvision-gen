"""Patch extraction and full-image reconstruction module.

Handles:
    - 128x128 patch extraction for training (with configurable stride and filtering)
    - Regular grid patch extraction for full-image evaluation
    - Seamless reconstruction via 2D overlapping window weighting (Hann / Cosine / Gaussian)
      preventing visible seams and boundary artifacts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def get_blending_window(patch_size: int = 128, mode: str = "hann") -> np.ndarray:
    """Create a 2D weighting window of shape (patch_size, patch_size) with values in (0, 1].

    Args:
        patch_size: Dimension of the square patch.
        mode: 'hann', 'cosine', or 'linear'.

    Returns:
        2D numpy array of shape (patch_size, patch_size).
    """
    if mode == "hann":
        w1d = 0.2 + 0.8 * np.hanning(patch_size)
        window = np.outer(w1d, w1d)
    elif mode == "cosine":
        idx = np.linspace(-np.pi / 2, np.pi / 2, patch_size)
        w1d = 0.2 + 0.8 * np.cos(idx)
        window = np.outer(w1d, w1d)
    elif mode == "linear":
        w1d = 1.0 - 0.8 * np.abs(np.linspace(-1.0, 1.0, patch_size))
        window = np.outer(w1d, w1d)
    else:
        window = np.ones((patch_size, patch_size), dtype=np.float32)

    return window.astype(np.float32)


class PatchExtractor:
    """Extracts patches and stitches them back seamlessly."""

    def __init__(
        self,
        patch_size: int = 128,
        stride_train: int = 64,
        stride_eval: int = 96,
        blending_mode: str = "hann",
    ) -> None:
        """Initialize PatchExtractor.

        Args:
            patch_size: Square patch dimensions (128x128).
            stride_train: Stride for extraction during training dataset generation.
            stride_eval: Stride for tile extraction during evaluation inference.
            blending_mode: Blending window function ('hann', 'cosine', or 'linear').
        """
        self.patch_size = patch_size
        self.stride_train = stride_train
        self.stride_eval = stride_eval
        self.blending_mode = blending_mode
        self._window_np = get_blending_window(patch_size, blending_mode)

    def extract_paired_patches_tensor(
        self,
        low_tensor: torch.Tensor,
        normal_tensor: torch.Tensor,
        illumination_tensor: torch.Tensor | None = None,
        stride: int | None = None,
        min_variance: float = 0.0005,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
        """Extract matched patches from low-light, normal, and optional illumination tensors.

        Args:
            low_tensor: Low-light image tensor (C, H, W).
            normal_tensor: Reference image tensor (C, H, W).
            illumination_tensor: Optional illumination map (1, H, W).
            stride: Stride in pixels. Defaults to stride_train.
            min_variance: Variance threshold to discard solid black or uninformative patches.

        Returns:
            List of tuples (low_patch, normal_patch, illumination_patch).
        """
        if stride is None:
            stride = self.stride_train

        _, h, w = low_tensor.shape
        p = self.patch_size

        # Pad image if smaller than patch size
        pad_h = max(0, p - h)
        pad_w = max(0, p - w)
        if pad_h > 0 or pad_w > 0:
            low_tensor = F.pad(low_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            normal_tensor = F.pad(normal_tensor, (0, pad_w, 0, pad_h), mode="reflect")
            if illumination_tensor is not None:
                illumination_tensor = F.pad(
                    illumination_tensor, (0, pad_w, 0, pad_h), mode="reflect"
                )
            _, h, w = low_tensor.shape

        # Grid positions
        y_steps = list(range(0, h - p + 1, stride))
        if (h - p) not in y_steps:
            y_steps.append(h - p)

        x_steps = list(range(0, w - p + 1, stride))
        if (w - p) not in x_steps:
            x_steps.append(w - p)

        patches = []
        for y in y_steps:
            for x in x_steps:
                low_p = low_tensor[:, y : y + p, x : x + p]
                norm_p = normal_tensor[:, y : y + p, x : x + p]
                illum_p = (
                    illumination_tensor[:, y : y + p, x : x + p]
                    if illumination_tensor is not None
                    else None
                )

                # Filter out pure black / uninformative patches
                if low_p.var().item() >= min_variance or norm_p.var().item() >= min_variance:
                    patches.append((low_p, norm_p, illum_p))

        return patches

    def decompose_image_for_eval(
        self,
        image_tensor: torch.Tensor,
        stride: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int]], tuple[int, int]]:
        """Decompose an arbitrary size image tensor into a batch of 128x128 patches for evaluation.

        Args:
            image_tensor: Tensor of shape (C, H, W) or (1, C, H, W).
            stride: Overlap stride. Defaults to stride_eval.

        Returns:
            Tuple containing:
                - Patches tensor of shape (N, C, patch_size, patch_size)
                - List of (y, x) top-left coordinates for each patch
                - Original (H, W) tuple
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        if stride is None:
            stride = self.stride_eval

        _c, h, w = image_tensor.shape
        p = self.patch_size

        # Pad if needed
        pad_h = max(0, p - h)
        pad_w = max(0, p - w)
        if pad_h > 0 or pad_w > 0:
            image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")

        _cur_c, cur_h, cur_w = image_tensor.shape

        y_steps = list(range(0, cur_h - p + 1, stride))
        if (cur_h - p) not in y_steps:
            y_steps.append(cur_h - p)

        x_steps = list(range(0, cur_w - p + 1, stride))
        if (cur_w - p) not in x_steps:
            x_steps.append(cur_w - p)

        patch_list = []
        coords = []
        for y in y_steps:
            for x in x_steps:
                patch = image_tensor[:, y : y + p, x : x + p]
                patch_list.append(patch)
                coords.append((y, x))

        patches = torch.stack(patch_list, dim=0)  # (N, C, P, P)
        return patches, coords, (h, w)

    def reconstruct_image(
        self,
        patches: torch.Tensor,
        coords: list[tuple[int, int]],
        orig_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Seamlessly stitch reconstructed patches into the full image using smooth blending.

        Args:
            patches: Tensor of shape (N, C, patch_size, patch_size).
            coords: List of (y, x) coordinates for each patch.
            orig_shape: Original image (H, W).

        Returns:
            Reconstructed full image tensor of shape (C, H, W).
        """
        orig_h, orig_w = orig_shape
        p = self.patch_size
        _n, c, _, _ = patches.shape
        device = patches.device
        dtype = patches.dtype

        # Determine canvas size
        max_y = max(y for y, _ in coords) + p
        max_x = max(x for _, x in coords) + p
        canvas_h = max(orig_h, max_y)
        canvas_w = max(orig_w, max_x)

        # 2D window tensor
        window = (
            torch.from_numpy(self._window_np).to(device=device, dtype=dtype).unsqueeze(0)
        )  # (1, P, P)

        canvas = torch.zeros((c, canvas_h, canvas_w), device=device, dtype=dtype)
        weight_sum = torch.zeros((1, canvas_h, canvas_w), device=device, dtype=dtype)

        for i, (y, x) in enumerate(coords):
            patch = patches[i]  # (C, P, P)
            weighted_patch = patch * window
            canvas[:, y : y + p, x : x + p] += weighted_patch
            weight_sum[:, y : y + p, x : x + p] += window

        # Normalize by weight sum (avoid division by zero)
        weight_sum = torch.clamp(weight_sum, min=1e-5)
        canvas = canvas / weight_sum

        # Crop back to exact original dimensions
        restored = canvas[:, :orig_h, :orig_w]
        return restored
