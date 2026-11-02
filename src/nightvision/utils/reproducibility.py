"""Reproducibility utilities for NightVision Gen.

Ensures deterministic seeds across Python, NumPy, PyTorch CPU and CUDA.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seed across all libraries for deterministic execution.

    Args:
        seed: Random seed integer.
        deterministic: If True, configures CuDNN for deterministic algorithms.

    Returns:
        The applied seed integer.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    return seed
