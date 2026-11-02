"""Unit tests for U-Net architecture and tensor shapes."""

import torch

from nightvision.models.conditioning import ConditioningManager
from nightvision.models.embeddings import SinusoidalTimeEmbedding
from nightvision.models.unet import ConditionalUNet


def test_sinusoidal_embeddings():
    """Verify time embedding shapes."""
    embed_dim = 128
    emb_module = SinusoidalTimeEmbedding(embed_dim)
    t = torch.tensor([0, 100, 500, 999], dtype=torch.long)
    out = emb_module(t)
    assert out.shape == (4, embed_dim)


def test_conditioning_manager_shapes():
    """Verify ConditioningManager produces 6 channels for A and 7 channels for B/C."""
    mgr_a = ConditioningManager(variant="A")
    assert mgr_a.expected_in_channels == 6

    mgr_b = ConditioningManager(variant="B")
    assert mgr_b.expected_in_channels == 7

    x_noisy = torch.randn((2, 3, 128, 128))
    dark = torch.randn((2, 3, 128, 128))
    illum = torch.randn((2, 1, 128, 128))

    out_a = mgr_a.prepare_unet_input(x_noisy, dark)
    assert out_a.shape == (2, 6, 128, 128)

    out_b = mgr_b.prepare_unet_input(x_noisy, dark, illum)
    assert out_b.shape == (2, 7, 128, 128)


def test_unet_forward_pass_model_a():
    """Verify U-Net forward pass for Model A (6 channels)."""
    # Use light channel configuration for fast CPU test
    unet = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(16,),
        time_embed_dim=64,
    )

    x = torch.randn((2, 6, 128, 128))
    t = torch.tensor([50, 200], dtype=torch.long)
    out = unet(x, t)

    assert out.shape == (2, 3, 128, 128)
    assert not torch.isnan(out).any()


def test_unet_forward_pass_model_b_c():
    """Verify U-Net forward pass for Model B and C (7 channels)."""
    unet = ConditionalUNet(
        in_channels=7,
        out_channels=3,
        base_channels=16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attention_resolutions=(16,),
        time_embed_dim=64,
    )

    x = torch.randn((2, 7, 128, 128))
    t = torch.tensor([10, 800], dtype=torch.long)
    out = unet(x, t)

    assert out.shape == (2, 3, 128, 128)
    assert not torch.isnan(out).any()
