"""Resume interrupted training from a checkpoint file.

Usage:
    python scripts/resume_training.py --checkpoint artifacts/checkpoints/model_A/checkpoint_latest.pt --config configs/train_A.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from nightvision.data.dataset import LOLv2PatchDataset, SyntheticPairDataset
from nightvision.data.split_manager import LOLv2SplitManager
from nightvision.training.trainer import Trainer
from nightvision.utils.config import load_config
from nightvision.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("resume_training")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume training from checkpoint.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint .pt file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML.")
    parser.add_argument("--additional_epochs", type=int, default=0, help="Optional extra epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("device", "auto"))

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    variant = str(cfg.get("experiment", {}).get("model_variant", "A")).upper()
    compute_retinex = variant in ("B", "C")
    batch_size = int(cfg.get("training", {}).get("batch_size", 16))

    manifest_path = Path("data/processed/lol_v2_splits.json")
    if manifest_path.exists():
        splits = LOLv2SplitManager.load_split_manifest(manifest_path)
        train_ds = LOLv2PatchDataset(splits["train"], compute_retinex=compute_retinex)
        val_ds = LOLv2PatchDataset(splits["val"], compute_retinex=compute_retinex)
    else:
        logger.info("Using synthetic fallback for resuming.")
        train_ds = SyntheticPairDataset(16, compute_retinex=compute_retinex)
        val_ds = SyntheticPairDataset(4, compute_retinex=compute_retinex)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader, device=device)
    trainer.resume_from_checkpoint(ckpt_path)

    if args.additional_epochs > 0:
        cfg["training"]["epochs"] = trainer.start_epoch + args.additional_epochs

    trainer.train()


if __name__ == "__main__":
    main()
