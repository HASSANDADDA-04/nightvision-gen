"""Full-reference image quality assessment metrics for LOL-v2.

Implements standard benchmark metrics:
    - PSNR (Peak Signal-to-Noise Ratio in dB)
    - SSIM (Structural Similarity Index Measure in [0, 1])
    - LPIPS (Learned Perceptual Image Patch Similarity, lower is better)
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

try:
    import lpips

    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False


class ImageMetricsEvaluator:
    """Computes PSNR, SSIM, and LPIPS between enhanced images and ground truth references."""

    def __init__(self, device: str = "cpu", lpips_net: str = "alex") -> None:
        """Initialize ImageMetricsEvaluator.

        Args:
            device: 'cpu' or 'cuda'.
            lpips_net: 'alex' or 'vgg'.
        """
        self.device = torch.device(device)
        self.lpips_fn = None
        if _HAS_LPIPS:
            try:
                self.lpips_fn = lpips.LPIPS(net=lpips_net).to(self.device)
                self.lpips_fn.eval()
            except Exception:
                self.lpips_fn = None

    def compute_psnr(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> float:
        """Compute Peak Signal-to-Noise Ratio (dB) on images in [0, 1].

        Args:
            pred: Enhanced image (H, W, 3) or (3, H, W).
            target: Ground truth reference.

        Returns:
            PSNR value in decibels (dB).
        """
        pred_np = self._to_numpy_0_1(pred)
        target_np = self._to_numpy_0_1(target)
        return float(sk_psnr(target_np, pred_np, data_range=1.0))

    def compute_ssim(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> float:
        """Compute Structural Similarity Index (SSIM) on images in [0, 1].

        Args:
            pred: Enhanced image (H, W, 3) or (3, H, W).
            target: Ground truth reference.

        Returns:
            SSIM value in [0, 1].
        """
        pred_np = self._to_numpy_0_1(pred)
        target_np = self._to_numpy_0_1(target)
        # channel_axis=2 for (H, W, 3)
        return float(sk_ssim(target_np, pred_np, channel_axis=2, data_range=1.0))

    def compute_lpips(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> float | None:
        """Compute Learned Perceptual Image Patch Similarity (LPIPS).

        LPIPS expects tensors in [-1, 1]. Lower values indicate higher perceptual fidelity.

        Args:
            pred: Enhanced image.
            target: Ground truth reference.

        Returns:
            LPIPS score (float) or None if LPIPS model weights cannot be loaded.
        """
        if self.lpips_fn is None:
            return None

        pred_t = self._to_tensor_neg1_1(pred).to(self.device)
        target_t = self._to_tensor_neg1_1(target).to(self.device)

        if pred_t.dim() == 3:
            pred_t = pred_t.unsqueeze(0)
        if target_t.dim() == 3:
            target_t = target_t.unsqueeze(0)

        with torch.no_grad():
            score = self.lpips_fn(pred_t, target_t)
        return float(score.item())

    def evaluate_pair(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> dict[str, float]:
        """Compute all available full-reference metrics for an image pair.

        Returns:
            Dictionary with keys 'psnr', 'ssim', and optionally 'lpips'.
        """
        res = {
            "psnr": round(self.compute_psnr(pred, target), 3),
            "ssim": round(self.compute_ssim(pred, target), 4),
        }
        lpips_score = self.compute_lpips(pred, target)
        if lpips_score is not None:
            res["lpips"] = round(lpips_score, 4)
        return res

    @staticmethod
    def _to_numpy_0_1(img: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert any tensor/array format to (H, W, 3) float32 in [0, 1]."""
        if isinstance(img, torch.Tensor):
            t = img.detach().cpu().float()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.shape[0] == 3:
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            arr = img.astype(np.float32)

        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.min() < 0.0:
            arr = (arr + 1.0) / 2.0
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _to_tensor_neg1_1(img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert any tensor/array format to (3, H, W) float32 tensor in [-1, 1]."""
        if isinstance(img, np.ndarray):
            arr = img.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.shape[-1] == 3:
                t = torch.from_numpy(arr).permute(2, 0, 1)
            else:
                t = torch.from_numpy(arr)
            return t * 2.0 - 1.0
        else:
            t = img.detach().clone().float()
            if t.dim() == 4:
                t = t.squeeze(0)
            if t.min() >= 0.0:
                t = t * 2.0 - 1.0
            return torch.clamp(t, -1.0, 1.0)
