"""Data preparation script for LOL-v2 and ExDark.

Performs:
    1. Validation of dataset paths and structure
    2. Identification and pairing of dark/normal images
    3. Train/Val/Test splitting (reserving 100 test pairs, no leakage)
    4. Stratified selection of ExDark images (~1000 across 12 classes)
    5. Saving of JSON manifests for reproducibility

Usage:
    python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nightvision.data.split_manager import ExDarkManager, LOLv2SplitManager
from nightvision.utils.config import load_config
from nightvision.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset splits and manifests.")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Path to data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("datasets", {}).get("lol_v2", {}).get("seed", 42)
    set_seed(seed)

    # 1. LOL-v2 Splits
    lol_cfg = cfg.get("datasets", {}).get("lol_v2", {})
    lol_root = Path(lol_cfg.get("root_dir", "data/raw/LOLv2"))
    val_ratio = float(lol_cfg.get("val_split_ratio", 0.15))

    logger.info("Initializing LOL-v2 Split Manager for root: %s", lol_root)
    lol_manager = LOLv2SplitManager(root_dir=lol_root, val_ratio=val_ratio, seed=seed)

    splits = lol_manager.create_splits()
    output_dir = Path(cfg.get("processed", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "lol_v2_splits.json"
    lol_manager.save_split_manifest(splits, manifest_path)
    logger.info(
        "LOL-v2 Manifest generated: %d train, %d val, %d test pairs.",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    # 2. ExDark Stratified Selection
    exdark_cfg = cfg.get("datasets", {}).get("exdark", {})
    exdark_root = Path(exdark_cfg.get("root_dir", "data/raw/ExDark"))
    exdark_manager = ExDarkManager(root_dir=exdark_root, seed=seed)

    sample_size = int(exdark_cfg.get("stratified_sample_size", 1000))
    exdark_samples = exdark_manager.get_stratified_subset(total_samples=sample_size)

    exdark_manifest = output_dir / "exdark_stratified_subset.json"
    if exdark_samples:
        import json

        with open(exdark_manifest, "w", encoding="utf-8") as f:
            from dataclasses import asdict

            json.dump([asdict(s) for s in exdark_samples], f, indent=2)
        logger.info("ExDark Manifest generated with %d samples.", len(exdark_samples))
    else:
        logger.info(
            "ExDark raw folder not yet populated (or empty). Run on Colab after dataset mounting."
        )


if __name__ == "__main__":
    main()
