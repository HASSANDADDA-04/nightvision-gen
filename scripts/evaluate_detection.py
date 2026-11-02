"""Downstream object detection evaluation script on ExDark.

Usage:
    python scripts/evaluate_detection.py --config configs/evaluation.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from nightvision.evaluation.comparison import apply_clahe, apply_gamma_correction
from nightvision.evaluation.detection import (
    FrozenCOCODetector,
    evaluate_detection_metrics,
    parse_exdark_annotation_file,
)
from nightvision.sampling.inference import InferenceService
from nightvision.utils.config import load_config
from nightvision.utils.device import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_detection")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate object detection on ExDark.")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)

    detector = FrozenCOCODetector(
        confidence_threshold=float(cfg.get("detection", {}).get("confidence_threshold", 0.5)),
        iou_threshold=float(cfg.get("detection", {}).get("iou_threshold", 0.5)),
        device=str(device),
    )

    manifest_path = Path("data/processed/exdark_stratified_subset.json")
    if not manifest_path.exists():
        logger.info("ExDark manifest not found. ExDark evaluation requires full dataset on Colab.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[: args.max_samples]

    models = {
        "Raw_Dark": None,
        "Gamma_2.2": None,
        "CLAHE": None,
        "Model_A": "artifacts/checkpoints/model_A/checkpoint_best.pt",
        "Model_B": "artifacts/checkpoints/model_B/checkpoint_best.pt",
        "Model_C": "artifacts/checkpoints/model_C/checkpoint_best.pt",
    }

    results = []
    ground_truths = [parse_exdark_annotation_file(s["annotation_path"]) for s in samples]

    for name, ckpt in models.items():
        service = None
        if ckpt and Path(ckpt).exists():
            variant = name[-1]
            service = InferenceService(
                ckpt, model_variant=variant, device=str(device), ddim_steps=25
            )

        preds = []
        for s in tqdm(samples, desc=f"Detecting on {name}"):
            with Image.open(s["image_path"]) as img:
                rgb = img.convert("RGB")

            if name == "Raw_Dark":
                enhanced = rgb
            elif name == "Gamma_2.2":
                enhanced = apply_gamma_correction(rgb, gamma=2.2)
            elif name == "CLAHE":
                enhanced = apply_clahe(rgb, clip_limit=3.0)
            elif service is not None:
                enhanced, _, _ = service.enhance_image(rgb)
            else:
                enhanced = rgb

            pred = detector.detect_image(enhanced)
            preds.append(pred)

        metrics = evaluate_detection_metrics(preds, ground_truths, iou_thresh=0.5)
        results.append({"Method": name, **metrics})

    output_csv = Path(
        cfg.get("output", {}).get("detection_csv", "artifacts/metrics/detection_results.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    logger.info("Saved detection evaluation to %s", output_csv)


if __name__ == "__main__":
    main()
