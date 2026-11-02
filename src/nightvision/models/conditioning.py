"""Conditioning strategies and input fusion for Models A, B, and C.

Strategy: Early Fusion via Channel Concatenation:
    - Model A: concat(x_t [3ch], dark_image [3ch]) -> 6 channels total
    - Model B: concat(x_t [3ch], dark_image [3ch], retinex_map [1ch]) -> 7 channels total
    - Model C: concat(x_t [3ch], dark_image [3ch], retinex_map [1ch]) -> 7 channels total
"""

from __future__ import annotations

from typing import ClassVar

import torch


class ConditioningManager:
    """Manages conditioning tensor construction for diffusion models A, B, and C."""

    VALID_VARIANTS: ClassVar[set[str]] = {"A", "B", "C"}

    def __init__(self, variant: str = "A") -> None:
        """Initialize ConditioningManager.

        Args:
            variant: One of 'A', 'B', or 'C'.
        """
        variant = variant.upper()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Must be one of {self.VALID_VARIANTS}")
        self.variant = variant

    @property
    def expected_in_channels(self) -> int:
        """Return expected total input channels to U-Net."""
        return 6 if self.variant == "A" else 7

    def prepare_unet_input(
        self,
        x_noisy: torch.Tensor,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concatenate noisy image with condition tensor(s) along channel dimension.

        Args:
            x_noisy: Noisy target tensor x_t of shape (B, 3, H, W).
            dark_image: Low-light condition image of shape (B, 3, H, W).
            illumination_map: Retinex illumination map of shape (B, 1, H, W).
                              Required for variants 'B' and 'C'.

        Returns:
            Concatenated input tensor of shape (B, in_channels, H, W).
        """
        if x_noisy.shape[1] != 3:
            raise ValueError(f"x_noisy expected 3 channels, got shape {x_noisy.shape}")
        if dark_image.shape[1] != 3:
            raise ValueError(f"dark_image expected 3 channels, got shape {dark_image.shape}")

        if self.variant == "A":
            # 3 (noisy) + 3 (dark) = 6 channels
            unet_in = torch.cat([x_noisy, dark_image], dim=1)
        elif self.variant in ("B", "C"):
            if illumination_map is None:
                raise ValueError(
                    f"Model variant '{self.variant}' requires an illumination_map (shape B, 1, H, W)."
                )
            if illumination_map.shape[1] != 1:
                raise ValueError(
                    f"illumination_map expected 1 channel, got shape {illumination_map.shape}"
                )
            # 3 (noisy) + 3 (dark) + 1 (retinex) = 7 channels
            unet_in = torch.cat([x_noisy, dark_image, illumination_map], dim=1)
        else:
            raise ValueError(f"Unsupported variant: {self.variant}")

        return unet_in

    def extract_condition_only(
        self,
        dark_image: torch.Tensor,
        illumination_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract conditioning tensor without noisy x_t (useful for conditioning caches)."""
        if self.variant == "A":
            return dark_image
        else:
            if illumination_map is None:
                raise ValueError(f"Variant '{self.variant}' requires illumination_map.")
            return torch.cat([dark_image, illumination_map], dim=1)
