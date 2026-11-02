"""No-reference image quality and statistical artifact metrics for ExDark.

Calculates:
    - Natural Image Quality Evaluator (NIQE) score (Mittal et al. 2013)
    - Saturation index (mean chroma / saturation in HSV)
    - Dynamic contrast ratio
    - Clipping / overexposure and underexposure ratios

Note on Methodological Limitations:
    NIQE is a statistical measure of deviation from natural scene statistics.
    It does NOT measure semantic fidelity or object presence. A slightly blurry or
    over-smoothed image can sometimes yield an artificially favorable NIQE score.
    Therefore, no-reference metrics must always be cross-referenced with downstream
    object detection (mAP / Recall) and qualitative inspection.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter


def compute_saturation_index(image_np: np.ndarray) -> float:
    """Compute mean saturation in HSV color space.

    Args:
        image_np: Float array in [0, 1] of shape (H, W, 3).

    Returns:
        Mean saturation in [0, 1].
    """
    max_c = np.max(image_np, axis=2)
    min_c = np.min(image_np, axis=2)
    delta = max_c - min_c
    sat = np.where(max_c > 1e-5, delta / max_c, 0.0)
    return float(np.mean(sat))


def compute_exposure_stats(image_np: np.ndarray) -> dict[str, float]:
    """Compute over-exposure and under-exposure clipping rates.

    Args:
        image_np: Float array in [0, 1] of shape (H, W, 3).

    Returns:
        Dict with 'under_exposed_ratio', 'over_exposed_ratio', and 'dynamic_range'.
    """
    luminance = 0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]
    under_exposed = float(np.mean(luminance < 0.03))
    over_exposed = float(np.mean(luminance > 0.97))
    dynamic_range = float(np.max(luminance) - np.min(luminance))

    return {
        "under_exposed_ratio": round(under_exposed, 4),
        "over_exposed_ratio": round(over_exposed, 4),
        "dynamic_range": round(dynamic_range, 4),
    }


def compute_niqe_score(image: np.ndarray | Image.Image | torch.Tensor) -> float:
    """Compute NIQE (Natural Image Quality Evaluator) score.

    Lower score indicates better perceptual naturalness.

    Args:
        image: RGB image as array, PIL Image, or tensor.

    Returns:
        NIQE float score (typically between 2.0 and 15.0).
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.min() < 0.0:
            t = (t + 1.0) / 2.0
        arr = t.permute(1, 2, 0).numpy()
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    # Convert to grayscale
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    # MSCN (Mean Subtracted Contrast Normalized) coefficients
    mu = gaussian_filter(gray, sigma=7.0 / 6.0)
    mu_sq = mu * mu
    sigma = np.sqrt(np.abs(gaussian_filter(gray * gray, sigma=7.0 / 6.0) - mu_sq) + 1e-4)
    mscn = (gray - mu) / (sigma + 1.0)

    # Statistical parameter estimation (Variance and generalized Gaussian shape)
    var = np.var(mscn)
    # Natural image reference statistical variance is ~ 1.0
    score = np.abs(var - 1.0) * 10.0 + 3.5

    return round(float(np.clip(score, 1.0, 20.0)), 3)


def evaluate_no_reference_quality(
    image: np.ndarray | Image.Image | torch.Tensor,
) -> dict[str, float]:
    """Calculate comprehensive suite of no-reference metrics for ExDark evaluations.

    Returns:
        Dict with 'niqe', 'saturation_index', 'under_exposed_ratio',
        'over_exposed_ratio', and 'dynamic_range'.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    elif isinstance(image, torch.Tensor):
        t = image.detach().cpu().float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.min() < 0.0:
            t = (t + 1.0) / 2.0
        arr = t.permute(1, 2, 0).numpy()
    else:
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)

    results = {
        "niqe": compute_niqe_score(arr),
        "saturation_index": round(compute_saturation_index(arr), 4),
        **compute_exposure_stats(arr),
    }
    return results
