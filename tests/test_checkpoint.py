"""Unit tests for checkpoint persistence and state restoration."""

import torch
from torch import nn

from nightvision.training.checkpoint import CheckpointManager


def test_checkpoint_save_and_load(tmp_path):
    """Verify that model weights, optimizer state, and epoch counters are faithfully restored."""
    ckpt_mgr = CheckpointManager(checkpoint_dir=tmp_path)

    # Simple model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    # Perform a dummy training step
    x = torch.randn(2, 10)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    saved_path = ckpt_mgr.save(
        epoch=3,
        global_step=150,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        val_metric=0.45,
        is_best=True,
        config={"test_param": 123},
        seed=42,
    )

    assert saved_path.exists()
    assert (tmp_path / "checkpoint_latest.pt").exists()
    assert (tmp_path / "checkpoint_best.pt").exists()

    # Create new model and restore
    new_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=5)

    meta = CheckpointManager.load(
        saved_path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    assert meta["epoch"] == 3
    assert meta["global_step"] == 150
    assert meta["val_metric"] == 0.45

    # Check that weights are identical
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
