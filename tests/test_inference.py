"""Unit tests for DDIM sampling, timing, and inference service."""

import torch

from nightvision.models.conditioning import ConditioningManager
from nightvision.models.ddpm import GaussianDiffusion
from nightvision.models.unet import ConditionalUNet
from nightvision.sampling.ddim import DDIMSampler
from nightvision.sampling.inference import InferenceService


def test_ddim_sampling_steps():
    """Verify DDIM sampling generates correct tensor shape and bounded values."""
    diff = GaussianDiffusion(timesteps=100, beta_schedule="linear")
    sampler = DDIMSampler(diff, ddim_steps=10, eta=0.0)

    # Lightweight test model
    model = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        time_embed_dim=32,
    )

    cond_mgr = ConditioningManager(variant="A")
    dark_image = torch.randn((1, 3, 32, 32))

    restored, latency = sampler.sample(
        model=model,
        dark_image=dark_image,
        conditioning_manager=cond_mgr,
        measure_timing=True,
    )

    assert restored.shape == (1, 3, 32, 32)
    assert not torch.isnan(restored).any()
    assert latency > 0


def test_inference_service_full_pipeline(tmp_path):
    """Verify InferenceService handles image inputs and returns PIL images."""
    from PIL import Image

    # Create dummy dark image
    img = Image.new("RGB", (64, 64), color=(20, 20, 20))
    dummy_ckpt = tmp_path / "model.pt"
    torch.save({"model_state_dict": {}}, dummy_ckpt)

    service = InferenceService(
        checkpoint_path=dummy_ckpt,
        model_variant="B",
        device="cpu",
        ddim_steps=5,
        patch_size=32,
        patch_stride=24,
    )

    restored, illum, latency = service.enhance_image(img)
    assert isinstance(restored, Image.Image)
    assert isinstance(illum, Image.Image)
    assert restored.size == (64, 64)
    assert latency > 0
