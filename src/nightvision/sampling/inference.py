"""Inference service for restoring low-light images using trained NightVision Gen models.

Supports:
    - Any image resolution via tile decomposition and seamless windowed stitching
    - Automatic Retinex illumination calculation for Models B and C
    - Configurable DDIM sampling steps (25, 50, 100)
    - Precise latency measurement
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nightvision.data.patch_extractor import PatchExtractor
from nightvision.data.preprocessing import denormalize_to_0_1, normalize_to_neg1_1
from nightvision.illumination.retinex import RetinexEstimator
from nightvision.models.conditioning import ConditioningManager
from nightvision.models.ddpm import GaussianDiffusion
from nightvision.models.unet import ConditionalUNet
from nightvision.sampling.ddim import DDIMSampler
from nightvision.training.checkpoint import CheckpointManager
from nightvision.utils.device import get_device, synchronize_device

logger = logging.getLogger(__name__)


class InferenceService:
    """High-level restoration service encapsulating model loading and patch-based inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_variant: str = "A",
        device: str = "auto",
        ddim_steps: int = 50,
        eta: float = 0.0,
        patch_size: int = 128,
        patch_stride: int = 96,
    ) -> None:
        """Initialize InferenceService.

        Args:
            checkpoint_path: Path to model checkpoint file (.pt).
            model_variant: 'A', 'B', or 'C'.
            device: 'cuda', 'cpu', or 'auto'.
            ddim_steps: Number of DDIM reverse steps.
            eta: DDIM stochasticity parameter.
            patch_size: Size of patch tiles (128).
            patch_stride: Overlap stride for patch tiling (e.g. 96 for smooth blending).
        """
        self.device = get_device(device)
        self.variant = model_variant.upper()
        self.conditioning_mgr = ConditioningManager(variant=self.variant)
        self.retinex_estimator = RetinexEstimator(sigma=3.0) if self.variant in ("B", "C") else None
        self.patch_extractor = PatchExtractor(
            patch_size=patch_size,
            stride_eval=patch_stride,
            blending_mode="hann",
        )

        # Build architecture
        in_channels = self.conditioning_mgr.expected_in_channels
        self.model = ConditionalUNet(
            in_channels=in_channels,
            out_channels=3,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(16,),
            dropout=0.0,
        ).to(self.device)

        self.diffusion = GaussianDiffusion(timesteps=1000, beta_schedule="cosine").to(self.device)
        self.sampler = DDIMSampler(self.diffusion, ddim_steps=ddim_steps, eta=eta)

        # Load weights
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            CheckpointManager.load(checkpoint_path, model=self.model, device=self.device)
            logger.info("Loaded weights from %s for Model %s", checkpoint_path, self.variant)
        else:
            logger.warning(
                "Checkpoint %s not found. Model initialized with random weights (useful for smoke tests).",
                checkpoint_path,
            )

        self.model.eval()

    def set_ddim_steps(self, steps: int) -> None:
        """Update DDIM sampling step count."""
        self.sampler.set_ddim_steps(steps)

    @torch.no_grad()
    def enhance_image(
        self,
        image: Image.Image | np.ndarray | Path | str,
        batch_size: int = 8,
    ) -> tuple[Image.Image, Image.Image | None, float]:
        """Enhance a full-resolution low-light image.

        Args:
            image: PIL Image, NumPy RGB array, or file path.
            batch_size: Number of 128x128 patches to process in parallel on GPU.

        Returns:
            Tuple of:
                - Restored enhanced PIL Image
                - Retinex illumination map PIL Image (or None for Model A)
                - Inference elapsed wall-clock seconds
        """
        # 1. Standardize to PIL Image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(image, mode="RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        orig_w, orig_h = pil_img.size
        arr_rgb = np.array(pil_img, dtype=np.float32) / 255.0

        # 2. Compute Retinex map if needed
        illum_pil = None
        illum_tensor = None
        if self.variant in ("B", "C") and self.retinex_estimator is not None:
            illum_np = self.retinex_estimator.estimate_from_numpy(arr_rgb)  # (H, W, 1)
            illum_uint8 = (illum_np.squeeze(2) * 255.0).astype(np.uint8)
            illum_pil = Image.fromarray(illum_uint8, mode="L")
            illum_tensor = torch.from_numpy(illum_np).permute(2, 0, 1).to(self.device)  # (1, H, W)
            illum_tensor = normalize_to_neg1_1(illum_tensor)

        # 3. Low-light image tensor in [-1, 1]
        low_tensor = torch.from_numpy(arr_rgb).permute(2, 0, 1).to(self.device)  # (3, H, W)
        low_tensor = normalize_to_neg1_1(low_tensor)

        synchronize_device(self.device)
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == "cuda" else None

        if start_time:
            start_time.record()
        else:
            import time

            cpu_start = time.perf_counter()

        # 4. Check if image is exactly 128x128
        if orig_h == 128 and orig_w == 128:
            low_b = low_tensor.unsqueeze(0)
            illum_b = illum_tensor.unsqueeze(0) if illum_tensor is not None else None
            restored_b = self.sampler.sample(
                model=self.model,
                dark_image=low_b,
                illumination_map=illum_b,
                conditioning_manager=self.conditioning_mgr,
            )
            restored_tensor = restored_b.squeeze(0)
        else:
            # Full image patch-based inference
            patches_low, coords, orig_shape = self.patch_extractor.decompose_image_for_eval(
                low_tensor
            )
            num_patches = patches_low.shape[0]

            if illum_tensor is not None:
                patches_illum, _, _ = self.patch_extractor.decompose_image_for_eval(illum_tensor)
            else:
                patches_illum = None

            restored_patches_list = []
            for i in range(0, num_patches, batch_size):
                b_low = patches_low[i : i + batch_size].to(self.device)
                b_illum = (
                    patches_illum[i : i + batch_size].to(self.device)
                    if patches_illum is not None
                    else None
                )

                b_restored = self.sampler.sample(
                    model=self.model,
                    dark_image=b_low,
                    illumination_map=b_illum,
                    conditioning_manager=self.conditioning_mgr,
                )
                restored_patches_list.append(b_restored.cpu())

            all_restored_patches = torch.cat(restored_patches_list, dim=0)
            restored_tensor = self.patch_extractor.reconstruct_image(
                all_restored_patches, coords=coords, orig_shape=orig_shape
            )

        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_sec = start_time.elapsed_time(end_time) / 1000.0
        else:
            import time

            elapsed_sec = time.perf_counter() - cpu_start

        # 5. Denormalize to [0, 255] PIL Image
        restored_0_1 = denormalize_to_0_1(restored_tensor)
        restored_np = (restored_0_1.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        restored_pil = Image.fromarray(restored_np, mode="RGB")

        return restored_pil, illum_pil, elapsed_sec
