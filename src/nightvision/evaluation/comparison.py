"""Classical low-light image enhancement baselines.

Implements standard benchmark methods operating without learned weights:
    1. Gamma Correction: Non-linear brightness curve mapping.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization):
       Applies CLAHE on the luminance channel (LAB color space) to prevent color cast,
       with configurable clip limits and tile grid sizes.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None


def apply_gamma_correction(
    image: np.ndarray | torch.Tensor | Image.Image,
    gamma: float = 2.2,
) -> np.ndarray | torch.Tensor:
    """Apply gamma correction for low-light brightening.

    For low-light images with values in [0, 1], brightening is achieved by:
        I_out = I ** (1.0 / gamma) for gamma > 1.0

    Args:
        image: NumPy array in [0, 1] or uint8, PyTorch tensor, or PIL Image.
        gamma: Gamma parameter (typically 1.5, 2.2, or 2.8).

    Returns:
        Gamma corrected image in the same format.
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma}")

    inv_gamma = 1.0 / float(gamma)

    if isinstance(image, torch.Tensor):
        is_neg1_1 = image.min() < 0.0
        t = (image + 1.0) / 2.0 if is_neg1_1 else image
        t = torch.clamp(t, 0.0, 1.0)
        # Brightening
        out = torch.pow(t, inv_gamma)
        return (out * 2.0 - 1.0) if is_neg1_1 else out

    elif isinstance(image, np.ndarray):
        is_uint8 = image.dtype == np.uint8
        arr = image.astype(np.float32) / 255.0 if is_uint8 else image.astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        out = np.power(arr, inv_gamma)
        if is_uint8:
            return np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return out

    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        out = np.power(arr, inv_gamma)
        out_uint8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out_uint8)

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def apply_clahe(
    image: np.ndarray | torch.Tensor | Image.Image,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray | torch.Tensor:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Converts image to LAB color space and applies CLAHE strictly on the L (luminance) channel,
    preventing color shifts and chromatic noise amplification.

    Args:
        image: Image as NumPy array, PyTorch tensor, or PIL Image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for histogram equalization (rows, cols).

    Returns:
        CLAHE-enhanced image in same format.
    """
    if isinstance(image, torch.Tensor):
        device = image.device
        dtype = image.dtype
        is_neg1_1 = image.min() < 0.0
        t = (image + 1.0) / 2.0 if is_neg1_1 else image
        t = torch.clamp(t, 0.0, 1.0)

        # Convert tensor to numpy uint8 for OpenCV processing
        if t.dim() == 4:
            # Batched tensor
            results = []
            for b in range(t.shape[0]):
                single_np = (t[b].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                enhanced_np = _clahe_numpy(single_np, clip_limit, tile_grid_size)
                enhanced_t = torch.from_numpy(enhanced_np.astype(np.float32) / 255.0).permute(
                    2, 0, 1
                )
                results.append(enhanced_t)
            out = torch.stack(results, dim=0).to(device=device, dtype=dtype)
        else:
            single_np = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            enhanced_np = _clahe_numpy(single_np, clip_limit, tile_grid_size)
            out = (
                torch.from_numpy(enhanced_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .to(device=device, dtype=dtype)
            )

        return (out * 2.0 - 1.0) if is_neg1_1 else out

    elif isinstance(image, np.ndarray):
        is_uint8 = image.dtype == np.uint8
        arr_uint8 = image if is_uint8 else np.clip(image * 255.0, 0, 255).astype(np.uint8)
        enhanced = _clahe_numpy(arr_uint8, clip_limit, tile_grid_size)
        return enhanced if is_uint8 else (enhanced.astype(np.float32) / 255.0)

    elif isinstance(image, Image.Image):
        arr_uint8 = np.array(image.convert("RGB"))
        enhanced = _clahe_numpy(arr_uint8, clip_limit, tile_grid_size)
        return Image.fromarray(enhanced)

    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def _clahe_numpy(
    bgr_or_rgb_uint8: np.ndarray,
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> np.ndarray:
    """Internal helper to apply OpenCV CLAHE on the L-channel of an RGB uint8 array."""
    if cv2 is None:
        # Fallback pure-python/numpy histogram equalization if cv2 is not present
        return _clahe_fallback(bgr_or_rgb_uint8, clip_limit)

    lab = cv2.cvtColor(bgr_or_rgb_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return rgb_enhanced


def _clahe_fallback(rgb_uint8: np.ndarray, clip_limit: float) -> np.ndarray:
    """Fallback basic contrast stretching when OpenCV is compiling/installing."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    # Global adaptive contrast stretch
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 > p2:
        arr = np.clip((arr - p2) / (p98 - p2), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)
