"""Checkpoint manager for saving, loading, and resuming model training sessions.

Preserves full state:
    - Model weights (model_state_dict)
    - EMA weights (ema_state_dict)
    - Optimizer state (optimizer_state_dict)
    - Learning rate scheduler state (scheduler_state_dict)
    - Epoch index and global step count
    - Best validation metrics
    - Training configuration dict
    - Random seed for reproducibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handles persistent serialization and deserialization of training state."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_to_keep: int = 5,
    ) -> None:
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Path to directory where checkpoints are stored.
            max_to_keep: Maximum number of periodic epoch checkpoints to preserve.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.saved_checkpoints: list[Path] = []

    def save(
        self,
        epoch: int,
        global_step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        val_metric: float | None = None,
        is_best: bool = False,
        ema_state: dict[str, torch.Tensor] | None = None,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Path:
        """Save a complete training checkpoint.

        Args:
            epoch: Current completed epoch index.
            global_step: Total optimization steps completed.
            model: PyTorch model module.
            optimizer: PyTorch optimizer.
            scheduler: Optional learning rate scheduler.
            val_metric: Validation metric (e.g. val_loss or PSNR).
            is_best: True if this checkpoint achieves the best validation metric so far.
            ema_state: Optional Exponential Moving Average state dictionary.
            config: Full configuration dictionary for audit and reproducibility.
            seed: Random seed used in run.

        Returns:
            Path of saved checkpoint file.
        """
        # Unwrap DDP or DataParallel if wrapped
        raw_model = model.module if hasattr(model, "module") else model

        payload: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "val_metric": val_metric,
            "ema_state_dict": ema_state,
            "config": config,
            "seed": seed,
        }

        # 1. Periodic epoch checkpoint
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(payload, ckpt_path)
        self.saved_checkpoints.append(ckpt_path)

        # 2. Always update latest checkpoint symlink/copy for immediate resumption
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(payload, latest_path)

        # 3. Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(payload, best_path)
            logger.info("Saved new best checkpoint to %s (metric: %s)", best_path, val_metric)

        # Prune old periodic checkpoints if exceeding max_to_keep
        while len(self.saved_checkpoints) > self.max_to_keep:
            oldest = self.saved_checkpoints.pop(0)
            if oldest.exists() and oldest != latest_path and oldest.name != "checkpoint_best.pt":
                try:
                    oldest.unlink()
                except OSError:
                    pass

        return ckpt_path

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load checkpoint weights and training state.

        Args:
            checkpoint_path: Path to .pt checkpoint file.
            model: PyTorch model module to populate with weights.
            optimizer: Optional optimizer to restore state.
            scheduler: Optional scheduler to restore state.
            device: Target device for loaded tensors.

        Returns:
            Checkpoint metadata dictionary (epoch, step, metric, config, etc.).
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path.resolve()}")

        checkpoint = torch.load(path, map_location=device)

        # Load model weights
        raw_model = model.module if hasattr(model, "module") else model
        state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
        if state_dict is not None and len(state_dict) > 0:
            missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded model weights from %s (missing: %d, unexpected: %d)",
                path,
                len(missing),
                len(unexpected),
            )

        # Load optimizer state if provided
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Restored optimizer state from %s", path)

        # Load scheduler state if provided
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Restored scheduler state from %s", path)

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "val_metric": checkpoint.get("val_metric", None),
            "config": checkpoint.get("config", {}),
            "seed": checkpoint.get("seed", 42),
            "has_ema": checkpoint.get("ema_state_dict") is not None,
        }
