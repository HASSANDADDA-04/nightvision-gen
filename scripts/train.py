"""Main training script for NightVision Gen models (A, B, and C).

Usage:
    python scripts/train.py --config configs/train_A.yaml
    python scripts/train.py --config configs/train_B.yaml
    python scripts/train.py --config configs/train_C.yaml
    python scripts/train.py --config configs/train_A.yaml --synthetic --epochs 2  # Smoke test on CPU
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from nightvision.data.dataset import LOLv2PatchDataset, SyntheticPairDataset
from nightvision.data.split_manager import LOLv2SplitManager
from nightvision.training.trainer import Trainer
from nightvision.utils.config import load_config
from nightvision.utils.device import get_device, get_device_info
from nightvision.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train conditional DDPM model.")
    parser.add_argument("--config", type=str, required=True, help="Path to train YAML config.")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic dataset for CPU test."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI Overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["learning_rate"] = args.lr
    if args.device != "auto":
        cfg["device"] = args.device

    device = get_device(cfg.get("device", "auto"))
    logger.info("Selected hardware: %s", get_device_info(device))

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    variant = str(cfg.get("experiment", {}).get("model_variant", "A")).upper()
    compute_retinex = variant in ("B", "C")
    batch_size = int(cfg.get("training", {}).get("batch_size", 16))
    num_workers = int(cfg.get("training", {}).get("num_workers", 2))

    # Disable multiprocessing workers on Windows if CPU
    if device.type == "cpu" or torch.cuda.is_available() is False:
        num_workers = 0

    if args.synthetic:
        logger.info("Using SyntheticPairDataset for rapid local CPU testing.")
        train_ds = SyntheticPairDataset(length=16, image_size=128, compute_retinex=compute_retinex)
        val_ds = SyntheticPairDataset(length=4, image_size=128, compute_retinex=compute_retinex)
    else:
        manifest_path = Path("data/processed/lol_v2_splits.json")
        if manifest_path.exists():
            splits = LOLv2SplitManager.load_split_manifest(manifest_path)
        else:
            lol_root = cfg.get("datasets", {}).get("lol_v2", {}).get("root_dir", "data/raw/LOLv2")
            lol_mgr = LOLv2SplitManager(root_dir=lol_root, seed=seed)
            splits = lol_mgr.create_splits()

        if len(splits["train"]) == 0:
            logger.warning("No LOL-v2 train images found. Falling back to SyntheticPairDataset.")
            train_ds = SyntheticPairDataset(
                length=16, image_size=128, compute_retinex=compute_retinex
            )
            val_ds = SyntheticPairDataset(length=4, image_size=128, compute_retinex=compute_retinex)
        else:
            train_ds = LOLv2PatchDataset(
                pairs=splits["train"],
                patch_size=128,
                patches_per_image=4,
                is_train=True,
                compute_retinex=compute_retinex,
            )
            val_ds = LOLv2PatchDataset(
                pairs=splits["val"],
                patch_size=128,
                patches_per_image=2,
                is_train=False,
                compute_retinex=compute_retinex,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    trainer = Trainer(config=cfg, train_loader=train_loader, val_loader=val_loader, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
