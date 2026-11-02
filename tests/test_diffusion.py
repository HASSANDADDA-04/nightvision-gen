"""Unit tests for Gaussian diffusion process and losses."""

import torch

from nightvision.losses.diffusion_loss import DiffusionLoss
from nightvision.losses.edge_loss import CombinedDiffusionLoss, EdgeLoss
from nightvision.models.ddpm import GaussianDiffusion, make_beta_schedule


def test_beta_schedule():
    """Verify beta schedules produce monotonically non-decreasing cumulative variance."""
    betas_linear = make_beta_schedule("linear", timesteps=1000)
    assert betas_linear.shape == (1000,)
    assert (betas_linear > 0).all() and (betas_linear < 1).all()

    betas_cosine = make_beta_schedule("cosine", timesteps=1000)
    assert betas_cosine.shape == (1000,)
    assert (betas_cosine > 0).all() and (betas_cosine < 1).all()


def test_q_sample_forward():
    """Verify forward noisy sampling preserves shape and expected statistics."""
    diff = GaussianDiffusion(timesteps=1000, beta_schedule="cosine")
    x_start = torch.randn((2, 3, 128, 128))
    t = torch.tensor([0, 999], dtype=torch.long)

    noise = torch.randn_like(x_start)
    x_noisy = diff.q_sample(x_start, t, noise=noise)

    assert x_noisy.shape == x_start.shape
    # At t=0, alpha_bar is very close to 1, so x_noisy is very close to x_start
    assert torch.isclose(x_noisy[0], x_start[0], atol=0.2).all()


def test_predict_x0():
    """Verify inversion equation x_0 derivation from (x_t, eps)."""
    diff = GaussianDiffusion(timesteps=100, beta_schedule="linear")
    x0 = torch.clamp(torch.randn((2, 3, 32, 32)), -1.0, 1.0)
    t = torch.tensor([10, 50], dtype=torch.long)
    eps = torch.randn_like(x0)

    # Make noisy
    xt = diff.q_sample(x0, t, noise=eps)

    # Reconstruct
    pred_x0 = diff.predict_x0_from_eps(xt, t, eps, clip=False)
    assert torch.allclose(pred_x0, x0, atol=1e-4)


def test_losses():
    """Verify diffusion loss and Sobel edge loss."""
    pred_noise = torch.randn((2, 3, 64, 64))
    true_noise = torch.randn((2, 3, 64, 64))

    diff_loss = DiffusionLoss(loss_type="l1")
    l_d = diff_loss(pred_noise, true_noise)
    assert l_d.item() > 0

    edge_loss = EdgeLoss(lambda_edge=0.05, target_mode="target")
    img_a = torch.randn((2, 3, 64, 64))
    img_b = torch.randn((2, 3, 64, 64))
    l_e = edge_loss(img_a, img_b)
    assert l_e.item() > 0

    # Combined loss
    comb = CombinedDiffusionLoss(diffusion_loss_type="l1", lambda_edge=0.05)
    total_loss, breakdown = comb(pred_noise, true_noise, pred_x0=img_a, target_x0=img_b)
    assert "l_diff" in breakdown and "l_edge" in breakdown and "l_total" in breakdown
    assert total_loss.item() > 0
