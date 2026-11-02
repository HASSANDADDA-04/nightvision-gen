"""Unit tests for Retinex illumination estimation."""

import numpy as np
import torch
from PIL import Image

from nightvision.illumination.retinex import RetinexEstimator, compute_illumination_map


def test_retinex_max_rgb_numpy():
    """Verify max-RGB computation and smoothing."""
    estimator = RetinexEstimator(kernel_size=5, sigma=1.0, normalize=True)

    # Image where Red=0.8, Green=0.4, Blue=0.2
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[:, :, 0] = 0.8
    img[:, :, 1] = 0.4
    img[:, :, 2] = 0.2

    illum = estimator.estimate_from_numpy(img)
    assert illum.shape == (32, 32, 1)
    assert illum.min() >= 0.0
    assert illum.max() <= 1.0


def test_retinex_tensor_batch():
    """Verify PyTorch tensor batched processing."""
    estimator = RetinexEstimator(kernel_size=7, sigma=2.0, normalize=True)

    # Tensor in [-1, 1]
    tensor = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
    tensor[:, 0, :, :] = 0.6
    tensor[:, 1, :, :] = -0.2

    illum = estimator.estimate_from_tensor(tensor)
    assert illum.shape == (2, 1, 64, 64)
    assert illum.min() >= 0.0
    assert illum.max() <= 1.0


def test_convenience_helper_pil():
    """Verify compute_illumination_map on PIL Image."""
    img = Image.new("RGB", (64, 64), color=(100, 50, 20))
    illum = compute_illumination_map(img, sigma=2.0)
    assert isinstance(illum, np.ndarray)
    assert illum.shape == (64, 64, 1)
