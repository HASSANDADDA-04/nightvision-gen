"""Experiment logging utilities for NightVision Gen.

Tracks metrics across epochs, writes to console, disk log files,
and exports structured metrics to CSV / JSON for reporting.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Manages experiment metrics tracking and export."""

    def __init__(self, log_dir: str | Path, experiment_name: str = "run") -> None:
        """Initialize ExperimentLogger.

        Args:
            log_dir: Directory where logs and metrics are stored.
            experiment_name: Name identifier for this training run.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: list[dict[str, Any]] = []

        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.json_path = self.log_dir / f"{experiment_name}_history.json"
        self._csv_writer = None
        self._csv_file = None

    def log_step(self, step_metrics: dict[str, Any]) -> None:
        """Log a single training step."""
        # Optional: fine-grained step logging if needed

    def log_epoch(self, epoch_metrics: dict[str, Any]) -> None:
        """Record epoch-level metrics and update CSV and JSON on disk.

        Args:
            epoch_metrics: Dict containing metrics like 'epoch', 'train_loss', 'val_loss', 'lr'.
        """
        self.history.append(epoch_metrics)

        # Collect union of all keys across all logged epochs to guarantee uniform column count
        all_fieldnames: list[str] = []
        for entry in self.history:
            for k in entry.keys():
                if k not in all_fieldnames:
                    all_fieldnames.append(k)

        # Write uniform CSV with all known headers
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.history)

        # Update JSON history
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def print_epoch_summary(self, epoch_metrics: dict[str, Any]) -> None:
        """Format and print clean terminal status message."""
        epoch = epoch_metrics.get("epoch", 0)
        t_loss = epoch_metrics.get("train_loss", 0.0)
        v_loss = epoch_metrics.get("val_loss", None)
        lr = epoch_metrics.get("lr", 0.0)

        msg = f"[Epoch {epoch:03d}] Train Loss: {t_loss:.4f} | LR: {lr:.2e}"
        if v_loss is not None:
            msg += f" | Val Loss: {v_loss:.4f}"
        if "val_psnr" in epoch_metrics:
            msg += f" | Val PSNR: {epoch_metrics['val_psnr']:.2f} dB"

        logger.info(msg)
        print(msg)
