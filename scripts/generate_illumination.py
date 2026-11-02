"""Precompute and cache Retinex illumination maps for images.

Usage:
    python scripts/generate_illumination.py --input_dir data/raw/LOLv2/Real_captured/Train/Low --output_dir data/processed/LOLv2/illumination_maps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from nightvision.illumination.retinex import RetinexEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_illumination")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save Retinex illumination maps.")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing dark images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save illumination maps."
    )
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian smoothing sigma.")
    parser.add_argument("--kernel_size", type=int, default=15, help="Kernel size.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = RetinexEstimator(kernel_size=args.kernel_size, sigma=args.sigma, normalize=True)

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    img_files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

    logger.info("Found %d images in %s. Generating illumination maps...", len(img_files), input_dir)
    for f in tqdm(img_files, desc="Retinex"):
        out_f = output_dir / f"{f.stem}.png"
        estimator.process_file(input_image_path=f, output_map_path=out_f)

    logger.info("Completed. Illumination maps saved to %s", output_dir)


if __name__ == "__main__":
    main()
