"""Retinex illumination map estimation module.

Implements illumination estimation following Retinex theory:
    I(x, y) = R(x, y) * L(x, y)
Where:
    - I(x, y) is the observed dark image
    - R(x, y) is the scene reflectance (intrinsic surface properties)
    - L(x, y) is the illumination map describing lighting distribution

Following the project specification:
    L_raw(x, y) = max(R(x, y), G(x, y), B(x, y))
Followed by configurable spatial smoothing (Gaussian filter) to retain low-frequency
lighting variations without high-frequency texture bleeding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter


class RetinexEstimator:
    """Estimates the illumination map from low-light RGB images using max-RGB Retinex."""

    def __init__(
        self,
        kernel_size: int = 15,
        sigma: float = 3.0,
        normalize: bool = True,
    ) -> None:
        """Initialize RetinexEstimator.

        Args:
            kernel_size: Size of Gaussian smoothing kernel (must be odd).
            sigma: Standard deviation for Gaussian kernel.
            normalize: If True, ensures values are in [0, 1].
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = float(sigma)
        self.normalize = normalize

    def estimate_from_numpy(self, image: np.ndarray) -> np.ndarray:
        """Estimate illumination map from a NumPy image array.

        Args:
            image: Float array in [0, 1] or uint8 in [0, 255], shape (H, W, 3).

        Returns:
            Illumination map as float32 array in [0, 1], shape (H, W, 1).
        """
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.astype(np.float32)
            if img_float.max() > 1.0:
                img_float = img_float / 255.0

        # L_raw(x, y) = max(R(x, y), G(x, y), B(x, y))
        l_raw = np.max(img_float, axis=2)

        # Apply Gaussian smoothing to isolate the low-frequency illumination component
        if self.sigma > 0:
            l_smooth = gaussian_filter(l_raw, sigma=self.sigma, mode="reflect")
        else:
            l_smooth = l_raw

        if self.normalize:
            l_min, l_max = l_smooth.min(), l_smooth.max()
            if l_max - l_min > 1e-6:
                l_smooth = (l_smooth - l_min) / (l_max - l_min)
            else:
                l_smooth = np.clip(l_smooth, 0.0, 1.0)

        return np.expand_dims(l_smooth.astype(np.float32), axis=2)

    def estimate_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Estimate illumination map from a PyTorch tensor.

        Args:
            tensor: Tensor of shape (B, 3, H, W) or (3, H, W).
                    Values can be in [-1, 1] or [0, 1].

        Returns:
            Illumination map tensor of shape (B, 1, H, W) or (1, H, W) in [0, 1].
        """
        is_batched = tensor.dim() == 4
        if not is_batched:
            tensor = tensor.unsqueeze(0)

        # Detect and convert from [-1, 1] to [0, 1] if needed
        if tensor.min() < 0.0:
            tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0.0, 1.0)

        # L_raw = max over RGB channels (dim=1)
        l_raw, _ = torch.max(tensor, dim=1, keepdim=True)  # (B, 1, H, W)

        # Gaussian smoothing via separable depthwise 2D conv
        if self.sigma > 0:
            device = tensor.device
            dtype = tensor.dtype
            kernel = self._get_gaussian_kernel_2d(
                self.kernel_size, self.sigma, device=device, dtype=dtype
            )
            pad = self.kernel_size // 2
            # Replicate padding on boundaries to prevent dark edges
            l_padded = F.pad(l_raw, (pad, pad, pad, pad), mode="reflect")
            l_smooth = F.conv2d(l_padded, kernel, groups=1)
        else:
            l_smooth = l_raw

        if self.normalize:
            # Batch-wise min-max normalization
            b, _, _h, _w = l_smooth.shape
            min_v = l_smooth.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
            max_v = l_smooth.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
            diff = max_v - min_v
            diff = torch.where(diff > 1e-6, diff, torch.ones_like(diff))
            l_smooth = torch.clamp((l_smooth - min_v) / diff, 0.0, 1.0)

        return l_smooth if is_batched else l_smooth.squeeze(0)

    @staticmethod
    def _get_gaussian_kernel_2d(
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a 2D Gaussian kernel tensor for F.conv2d."""
        coords = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        g_1d = g_1d / g_1d.sum()
        g_2d = torch.outer(g_1d, g_1d)
        return g_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    def process_file(
        self,
        input_image_path: str | Path,
        output_map_path: str | Path | None = None,
    ) -> Image.Image:
        """Compute illumination map for an image file on disk.

        Args:
            input_image_path: Path to RGB input image.
            output_map_path: Optional output path to save the resulting map.

        Returns:
            PIL Image of the illumination map (mode 'L').
        """
        input_path = Path(input_image_path)
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")
            arr = np.array(rgb_img)

        illum_map = self.estimate_from_numpy(arr)  # (H, W, 1)
        map_uint8 = (illum_map.squeeze(2) * 255.0).astype(np.uint8)
        map_pil = Image.fromarray(map_uint8, mode="L")

        if output_map_path is not None:
            out_p = Path(output_map_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            map_pil.save(out_p)

        return map_pil


def compute_illumination_map(
    image: np.ndarray | torch.Tensor | Image.Image,
    sigma: float = 3.0,
    kernel_size: int = 15,
) -> np.ndarray | torch.Tensor:
    """Convenience helper to estimate Retinex illumination map.

    Args:
        image: NumPy array, PyTorch tensor, or PIL Image.
        sigma: Gaussian smoothing sigma.
        kernel_size: Gaussian kernel size.

    Returns:
        Illumination map in corresponding format.
    """
    estimator = RetinexEstimator(kernel_size=kernel_size, sigma=sigma, normalize=True)

    if isinstance(image, torch.Tensor):
        return estimator.estimate_from_tensor(image)
    elif isinstance(image, np.ndarray):
        return estimator.estimate_from_numpy(image)
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
        return estimator.estimate_from_numpy(arr)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
