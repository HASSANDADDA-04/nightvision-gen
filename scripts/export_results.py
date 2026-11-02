"""Aggregates all quantitative evaluation results and exports report.

Usage:
    python scripts/export_results.py --output artifacts/metrics/final_summary.md
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_results")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export combined project evaluation results.")
    parser.add_argument("--output", type=str, default="artifacts/metrics/final_summary.md")
    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    quality_csv = Path("artifacts/metrics/quality_results.csv")
    detection_csv = Path("artifacts/metrics/detection_results.csv")
    timing_csv = Path("artifacts/metrics/inference_timing.csv")

    lines = [
        "# NightVision Gen — Synthèse Globale des Évaluations\n",
        "## 1. Qualité de Restauration sur LOL-v2 (Paires Réelles)\n",
    ]

    if quality_csv.exists():
        with open(quality_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lines.append("| Méthode | PSNR (dB) | SSIM | Échantillons |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for row in reader:
                lines.append(
                    f"| {row.get('Method')} | {row.get('Mean_PSNR_dB')} | {row.get('Mean_SSIM')} | {row.get('Num_Images')} |"
                )
    else:
        lines.append(
            "*(Évaluation LOL-v2 non exécutée ou fichiers en attente d'entraînement GPU)*\n"
        )

    lines.append("\n## 2. Détection d'Objets sur ExDark (Détecteur COCO Gelé)\n")
    if detection_csv.exists():
        with open(detection_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lines.append("| Méthode | mAP@0.5 | Rappel@0.5 | Objets GT |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for row in reader:
                lines.append(
                    f"| {row.get('Method')} | {row.get('mAP_50')} | {row.get('recall_50')} | {row.get('total_gt_objects')} |"
                )
    else:
        lines.append("*(Évaluation ExDark en attente d'exécution)*\n")

    lines.append("\n## 3. Latence d'Inférence DDIM\n")
    if timing_csv.exists():
        with open(timing_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            lines.append("| Variante | Pas DDIM | Temps Moyen (s) | Débit (FPS) |")
            lines.append("| :--- | :---: | :---: | :---: |")
            for row in reader:
                lines.append(
                    f"| {row.get('variant')} | {row.get('ddim_steps')} | {row.get('mean_time_sec')} | {row.get('fps')} |"
                )
    else:
        lines.append("*(Mesures de latence en attente d'exécution)*\n")

    with open(out_p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Summary report exported to %s", out_p)


if __name__ == "__main__":
    main()
