"""Image enhancement and generation CLI script.

Usage:
    python scripts/generate_images.py --image path/to/dark.png --model_variant A --checkpoint artifacts/checkpoints/model_A/checkpoint_best.pt --output artifacts/generated_images/restored.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from nightvision.sampling.inference import InferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a low-light image with NightVision Gen.")
    parser.add_argument("--image", type=str, required=True, help="Path to input dark image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--model_variant", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM sampling steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta parameter.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="artifacts/generated_images/restored.png")
    args = parser.parse_args()

    service = InferenceService(
        checkpoint_path=args.checkpoint,
        model_variant=args.model_variant,
        device=args.device,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
    )

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path.resolve()}")

    logger.info(
        "Restoring image: %s using Model %s (DDIM %d steps)",
        img_path.name,
        args.model_variant,
        args.ddim_steps,
    )
    restored, illum, latency = service.enhance_image(img_path)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    restored.save(out_p)
    logger.info("Restored image saved to %s in %.3f seconds.", out_p, latency)

    if illum is not None:
        illum_p = out_p.parent / f"{out_p.stem}_retinex_illum.png"
        illum.save(illum_p)
        logger.info("Illumination map saved to %s", illum_p)


if __name__ == "__main__":
    main()
