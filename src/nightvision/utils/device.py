"""Device and hardware acceleration utilities for NightVision Gen.

Handles automatic detection of CUDA, MPS, or CPU, memory statistics,
and GPU synchronization for accurate inference timing.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def get_device(device_preference: str = "auto") -> torch.device:
    """Select the appropriate computing device based on preference and availability.

    Args:
        device_preference: 'auto', 'cuda', 'cuda:0', 'mps', or 'cpu'.

    Returns:
        torch.device instance.
    """
    pref = (device_preference or "auto").lower()

    if pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif pref.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(pref)
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            logger.warning("MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def synchronize_device(device: torch.device) -> None:
    """Synchronize device for accurate wall-clock timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def get_device_info(device: torch.device) -> dict[str, Any]:
    """Retrieve diagnostic information for the selected device."""
    info: dict[str, Any] = {"type": device.type, "index": device.index}

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        info["name"] = torch.cuda.get_device_name(idx)
        info["capability"] = torch.cuda.get_device_capability(idx)
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(idx).total_memory / (1024**3), 2
        )
        info["allocated_memory_mb"] = round(torch.cuda.memory_allocated(idx) / (1024**2), 2)
    else:
        info["name"] = "CPU"
        info["total_memory_gb"] = None

    return info
