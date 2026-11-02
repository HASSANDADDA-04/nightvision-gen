"""Inference service wrapper tailored for Streamlit caching and session management."""

from __future__ import annotations

from nightvision.sampling.inference import InferenceService


def get_inference_service(
    checkpoint_path: str,
    model_variant: str = "A",
    ddim_steps: int = 50,
    device: str = "auto",
) -> InferenceService:
    """Instantiate InferenceService."""
    return InferenceService(
        checkpoint_path=checkpoint_path,
        model_variant=model_variant,
        device=device,
        ddim_steps=ddim_steps,
        eta=0.0,
    )
