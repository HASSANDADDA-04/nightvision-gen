"""Visualization and image export utilities for NightVision Gen.

Provides functions to denormalize tensors, create comparison grids
(Dark / Illumination / Restored / Ground Truth), and save high-resolution figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """Convert a PyTorch tensor (C, H, W) or (1, C, H, W) to a NumPy image (H, W, C) in [0, 255].

    Args:
        tensor: Tensor with values in [-1, 1] if denormalize=True, or [0, 1].
        denormalize: If True, maps [-1, 1] to [0, 1].

    Returns:
        uint8 numpy array of shape (H, W, C) or (H, W) for single channel.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    tensor = tensor.detach().cpu().float()

    if denormalize:
        tensor = (tensor + 1.0) / 2.0

    tensor = torch.clamp(tensor, 0.0, 1.0)
    arr = tensor.numpy()

    if arr.shape[0] == 1:
        # Grayscale / 1-channel (e.g. Retinex illumination map)
        arr = arr.squeeze(0)
        img = (arr * 255.0).astype(np.uint8)
    else:
        # RGB (3, H, W) -> (H, W, 3)
        arr = np.transpose(arr, (1, 2, 0))
        img = (arr * 255.0).astype(np.uint8)

    return img


def save_image_tensor(
    tensor: torch.Tensor,
    output_path: str | Path,
    denormalize: bool = True,
) -> None:
    """Save a single image tensor to disk.

    Args:
        tensor: Tensor (C, H, W).
        output_path: Output file path (.png, .jpg).
        denormalize: If True, assumes input values in [-1, 1].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_image(tensor, denormalize=denormalize)
    img = Image.fromarray(arr)
    img.save(output_path)


def make_comparison_grid(
    images_dict: dict[str, torch.Tensor],
    output_path: str | Path | None = None,
    denormalize: bool = True,
    padding: int = 4,
) -> Image.Image:
    """Create a horizontal side-by-side comparison from a dictionary of named tensors.

    Args:
        images_dict: Dict of name -> tensor (1, C, H, W) or (C, H, W).
                     Common keys: 'Dark', 'Illumination', 'Model_A', 'Model_B', 'Model_C', 'Ground_Truth'.
        output_path: Optional path to save resulting comparison image.
        denormalize: Whether to map from [-1, 1] to [0, 1].
        padding: Padding between tiles.

    Returns:
        PIL Image of the comparison.
    """
    pil_images = []
    for name, tensor in images_dict.items():
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        # Convert single-channel to 3-channel RGB for unified display
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        arr = tensor_to_image(tensor, denormalize=denormalize)
        pil_images.append((name, Image.fromarray(arr)))

    # Compute canvas dimensions
    total_w = sum(img.width for _, img in pil_images) + padding * (len(pil_images) - 1)
    max_h = max(img.height for _, img in pil_images)

    grid = Image.new("RGB", (total_w, max_h), color=(30, 30, 30))
    offset_x = 0
    for _, img in pil_images:
        grid.paste(img, (offset_x, 0))
        offset_x += img.width + padding

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(output_path)

    return grid
