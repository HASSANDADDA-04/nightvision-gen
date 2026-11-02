"""Inference latency and throughput benchmarking module.

Benchmarks DDIM generation speed across step budgets (e.g. 25, 50, 100)
with GPU synchronization, warm-up iterations, and statistics aggregation.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nightvision.models.conditioning import ConditioningManager
from nightvision.sampling.ddim import DDIMSampler
from nightvision.utils.device import synchronize_device

logger = logging.getLogger(__name__)


def benchmark_sampling_latency(
    model: torch.nn.Module,
    sampler: DDIMSampler,
    conditioning_manager: ConditioningManager,
    steps_list: list[int] | None = None,
    image_size: int = 128,
    batch_size: int = 1,
    num_runs: int = 5,
    warmup_runs: int = 2,
    device: torch.device | None = None,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Benchmark inference time across different DDIM step settings.

    Args:
        model: Conditional U-Net.
        sampler: DDIMSampler instance.
        conditioning_manager: ConditioningManager.
        steps_list: List of DDIM step counts to evaluate (e.g. [25, 50, 100]).
        image_size: Resolution (128).
        batch_size: Batch size (1 for per-image latency).
        num_runs: Measured runs per configuration.
        warmup_runs: Warm-up runs before measurement.
        device: Device to execute benchmark on.
        output_csv: Optional path to save CSV benchmark results.

    Returns:
        List of dicts with timing statistics for each step count.
    """
    if steps_list is None:
        steps_list = [25, 50, 100]

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    dummy_dark = torch.randn((batch_size, 3, image_size, image_size), device=device)
    dummy_illum = (
        torch.randn((batch_size, 1, image_size, image_size), device=device)
        if conditioning_manager.variant in ("B", "C")
        else None
    )

    results: list[dict[str, Any]] = []

    for steps in steps_list:
        sampler.set_ddim_steps(steps)

        # Warm-up runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = sampler.sample(
                    model=model,
                    dark_image=dummy_dark,
                    illumination_map=dummy_illum,
                    conditioning_manager=conditioning_manager,
                )
        synchronize_device(device)

        # Timed runs
        timings = []
        for _ in range(num_runs):
            synchronize_device(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = sampler.sample(
                    model=model,
                    dark_image=dummy_dark,
                    illumination_map=dummy_illum,
                    conditioning_manager=conditioning_manager,
                )
            synchronize_device(device)
            t1 = time.perf_counter()
            timings.append(t1 - t0)

        mean_t = float(np.mean(timings))
        std_t = float(np.std(timings))
        fps = float(batch_size / mean_t)

        entry = {
            "variant": conditioning_manager.variant,
            "ddim_steps": steps,
            "device": str(device),
            "mean_time_sec": round(mean_t, 4),
            "std_time_sec": round(std_t, 4),
            "fps": round(fps, 2),
            "batch_size": batch_size,
        }
        results.append(entry)
        logger.info(
            "DDIM %d steps: %.3f ± %.3f s/image (%.2f FPS)",
            steps,
            mean_t,
            std_t,
            fps,
        )

    if output_csv is not None:
        out_p = Path(output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info("Saved timing benchmark to %s", out_p)

    return results
