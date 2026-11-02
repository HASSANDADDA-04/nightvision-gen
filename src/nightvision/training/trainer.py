"""Core training loop and orchestration module for NightVision Gen.

Supports independent training of Models A, B, and C from scratch with:
    - Mixed precision (FP16) with GradScaler on GPU
    - Gradient clipping
    - Exponential Moving Average (EMA) of network weights
    - Periodic validation loss and sample image generation
    - Stateful checkpoint saving and automatic resumption
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nightvision.losses.edge_loss import CombinedDiffusionLoss
from nightvision.models.conditioning import ConditioningManager
from nightvision.models.ddpm import GaussianDiffusion
from nightvision.models.unet import ConditionalUNet
from nightvision.training.checkpoint import CheckpointManager
from nightvision.training.logger import ExperimentLogger
from nightvision.utils.config import ConfigDict
from nightvision.utils.device import get_device
from nightvision.utils.reproducibility import set_seed
from nightvision.utils.visualization import make_comparison_grid

logger = logging.getLogger(__name__)


class ModelEMA:
    """Maintains Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone() for k, v in state_dict.items()}


class Trainer:
    """Trainer orchestrating the complete lifecycle of conditional DDPM training."""

    def __init__(
        self,
        config: ConfigDict | dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            config: Full experiment configuration.
            train_loader: DataLoader yielding training batches.
            val_loader: Optional DataLoader yielding validation batches.
            device: Optional torch.device.
        """
        self.cfg = ConfigDict(config) if not isinstance(config, ConfigDict) else config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 1. Device selection
        device_str = self.cfg.get("device", "auto")
        self.device = get_device(device_str) if device is None else device
        logger.info("Training running on device: %s", self.device)

        # 2. Seed
        seed = self.cfg.get("experiment", {}).get("seed", 42)
        set_seed(seed)

        # 3. Model variant & Conditioning
        self.variant = str(self.cfg.get("experiment", {}).get("model_variant", "A")).upper()
        self.conditioning_mgr = ConditioningManager(variant=self.variant)

        # 4. Initialize Conditional U-Net from scratch (random initialization)
        in_channels = self.conditioning_mgr.expected_in_channels
        model_cfg = self.cfg.get("model", {})
        self.model = ConditionalUNet(
            in_channels=in_channels,
            out_channels=model_cfg.get("out_channels", 3),
            base_channels=model_cfg.get("base_channels", 64),
            channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4, 8])),
            num_res_blocks=model_cfg.get("num_res_blocks", 2),
            attention_resolutions=tuple(model_cfg.get("attention_resolutions", [16])),
            dropout=model_cfg.get("dropout", 0.1),
            time_embed_dim=model_cfg.get("time_embed_dim", 256),
            norm_groups=model_cfg.get("norm_groups", 32),
        ).to(self.device)

        # 5. Initialize Gaussian Diffusion process
        diff_cfg = self.cfg.get("diffusion", {})
        self.diffusion = GaussianDiffusion(
            timesteps=diff_cfg.get("timesteps", 1000),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            beta_start=diff_cfg.get("beta_start", 0.0001),
            beta_end=diff_cfg.get("beta_end", 0.02),
            prediction_type=diff_cfg.get("prediction_type", "epsilon"),
        ).to(self.device)

        # 6. Loss function
        loss_cfg = self.cfg.get("loss", {})
        lambda_edge = loss_cfg.get("lambda_edge", 0.05 if self.variant == "C" else 0.0)
        self.loss_fn = CombinedDiffusionLoss(
            diffusion_loss_type=loss_cfg.get("diffusion_loss_type", "l1"),
            lambda_edge=lambda_edge,
            edge_target=loss_cfg.get("edge_target", "target"),
            edge_norm=loss_cfg.get("edge_norm", "l1"),
        ).to(self.device)

        # 7. Optimizer & Scheduler
        train_cfg = self.cfg.get("training", {})
        lr = float(train_cfg.get("learning_rate", 1.0e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1.0e-4))
        opt_name = train_cfg.get("optimizer", "AdamW")

        if opt_name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
            )

        epochs = int(train_cfg.get("epochs", 100))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1.0e-6))
        )

        # 8. Mixed precision & Gradient clipping
        mixed_prec = train_cfg.get("mixed_precision", "fp16").lower()
        self.use_amp = mixed_prec == "fp16" and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.grad_clip = float(train_cfg.get("gradient_clip_val", 1.0))

        # 9. EMA
        self.use_ema = train_cfg.get("use_ema", True)
        self.ema = (
            ModelEMA(self.model, decay=float(train_cfg.get("ema_decay", 0.9999)))
            if self.use_ema
            else None
        )

        # 10. Checkpoints & Logging
        paths_cfg = self.cfg.get("paths", {})
        ckpt_dir = paths_cfg.get("checkpoint_dir", f"artifacts/checkpoints/model_{self.variant}")
        log_dir = paths_cfg.get("log_dir", f"artifacts/logs/model_{self.variant}")
        self.ckpt_manager = CheckpointManager(checkpoint_dir=ckpt_dir)
        self.logger = ExperimentLogger(log_dir=log_dir, experiment_name=f"model_{self.variant}")
        self.generated_dir = Path(
            paths_cfg.get("generated_dir", f"artifacts/generated_images/model_{self.variant}")
        )
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 1
        self.global_step = 0
        self.best_val_loss = float("inf")

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training state from an existing checkpoint."""
        meta = self.ckpt_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )
        self.start_epoch = meta["epoch"] + 1
        self.global_step = meta["global_step"]
        if meta.get("val_metric") is not None:
            self.best_val_loss = meta["val_metric"]
        logger.info("Resuming training from epoch %d (step %d)", self.start_epoch, self.global_step)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Execute one complete training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        total_diff_loss = 0.0
        total_edge_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch:03d}", leave=False)
        for batch in pbar:
            low = batch["low"].to(self.device)  # (B, 3, 128, 128)
            normal = batch["normal"].to(self.device)  # (B, 3, 128, 128)
            illum = batch["illumination"].to(self.device) if "illumination" in batch else None

            b = normal.shape[0]

            # Sample random timesteps uniformly: t ~ U(0, T-1)
            t = torch.randint(
                0, self.diffusion.timesteps, (b,), device=self.device, dtype=torch.long
            )

            # Sample standard Gaussian noise epsilon ~ N(0, I)
            noise = torch.randn_like(normal)

            # Forward diffusion: q(x_t | x_0)
            x_noisy = self.diffusion.q_sample(x_start=normal, t=t, noise=noise)

            # Prepare conditioned U-Net input
            unet_in = self.conditioning_mgr.prepare_unet_input(
                x_noisy=x_noisy, dark_image=low, illumination_map=illum
            )

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Predict noise
                noise_pred = self.model(unet_in, t)

                # For Model C: estimate x_0 to compute edge preservation loss
                pred_x0 = None
                if self.loss_fn.edge_loss.lambda_edge > 0.0:
                    pred_x0 = self.diffusion.predict_x0_from_eps(x_noisy, t, noise_pred, clip=True)

                loss, loss_breakdown = self.loss_fn(
                    noise_pred=noise_pred,
                    noise_target=noise,
                    pred_x0=pred_x0,
                    target_x0=normal,
                    input_dark=low,
                )

            # Backward pass
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update EMA weights
            if self.ema is not None:
                self.ema.update(self.model)

            self.global_step += 1
            total_loss += loss.item()
            total_diff_loss += loss_breakdown.get("l_diff", 0.0)
            total_edge_loss += loss_breakdown.get("l_edge", 0.0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": self.global_step})

        avg_loss = total_loss / max(1, num_batches)
        avg_diff = total_diff_loss / max(1, num_batches)
        avg_edge = total_edge_loss / max(1, num_batches)

        return {"train_loss": avg_loss, "train_diff_loss": avg_diff, "train_edge_loss": avg_edge}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict[str, float]:
        """Execute validation pass and compute validation loss."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_val_loss = 0.0
        num_batches = len(self.val_loader)

        # Generate sample visual verification
        sample_batch = None

        for batch in self.val_loader:
            low = batch["low"].to(self.device)
            normal = batch["normal"].to(self.device)
            illum = batch["illumination"].to(self.device) if "illumination" in batch else None
            b = normal.shape[0]

            t = torch.randint(
                0, self.diffusion.timesteps, (b,), device=self.device, dtype=torch.long
            )
            noise = torch.randn_like(normal)
            x_noisy = self.diffusion.q_sample(x_start=normal, t=t, noise=noise)
            unet_in = self.conditioning_mgr.prepare_unet_input(x_noisy, low, illum)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                noise_pred = self.model(unet_in, t)
                pred_x0 = None
                if self.loss_fn.edge_loss.lambda_edge > 0.0:
                    pred_x0 = self.diffusion.predict_x0_from_eps(x_noisy, t, noise_pred, clip=True)

                loss, _ = self.loss_fn(
                    noise_pred=noise_pred,
                    noise_target=noise,
                    pred_x0=pred_x0,
                    target_x0=normal,
                    input_dark=low,
                )
            total_val_loss += loss.item()
            if sample_batch is None:
                sample_batch = batch

        avg_val_loss = total_val_loss / max(1, num_batches)

        # Quick validation sample save (using fast 10-step preview or DDPM reverse)
        if (
            sample_batch is not None
            and epoch % self.cfg.get("training", {}).get("validation_frequency", 5) == 0
        ):
            self._save_sample_preview(epoch, sample_batch)

        return {"val_loss": avg_val_loss}

    def _save_sample_preview(self, epoch: int, batch: dict[str, torch.Tensor]) -> None:
        """Save comparison image grid during validation."""
        try:
            low = batch["low"][:1].to(self.device)
            normal = batch["normal"][:1].to(self.device)
            illum = batch["illumination"][:1].to(self.device) if "illumination" in batch else None

            # Quick generation using DDIM
            from nightvision.sampling.ddim import DDIMSampler

            sampler = DDIMSampler(diffusion=self.diffusion, ddim_steps=20, eta=0.0)
            restored = sampler.sample(
                model=self.model,
                dark_image=low,
                illumination_map=illum,
                conditioning_manager=self.conditioning_mgr,
            )

            grid_dict = {"Dark": low[0], "Restored": restored[0], "Reference": normal[0]}
            if illum is not None:
                grid_dict["Illumination"] = illum[0]

            out_path = self.generated_dir / f"val_epoch_{epoch:04d}.png"
            make_comparison_grid(grid_dict, output_path=out_path)
            logger.info("Saved validation preview grid to %s", out_path)
        except Exception as e:
            logger.warning("Could not generate validation preview: %s", e)

    def train(self) -> None:
        """Run full training lifecycle."""
        train_cfg = self.cfg.get("training", {})
        total_epochs = int(train_cfg.get("epochs", 100))
        ckpt_freq = int(train_cfg.get("checkpoint_frequency", 10))
        val_freq = int(train_cfg.get("validation_frequency", 5))

        logger.info(
            "Starting training Model %s for %d epochs (start_epoch=%d)",
            self.variant,
            total_epochs,
            self.start_epoch,
        )

        for epoch in range(self.start_epoch, total_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Validate
            val_metrics = {}
            if epoch % val_freq == 0 or epoch == total_epochs:
                val_metrics = self.validate(epoch)

            elapsed = time.time() - epoch_start_time

            # Aggregate epoch metrics
            combined_metrics = {
                "epoch": epoch,
                "lr": current_lr,
                "elapsed_sec": round(elapsed, 2),
                **train_metrics,
                **val_metrics,
            }

            # Log
            self.logger.log_epoch(combined_metrics)
            self.logger.print_epoch_summary(combined_metrics)

            # Checkpoint
            val_loss = val_metrics.get("val_loss", train_metrics["train_loss"])
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if epoch % ckpt_freq == 0 or epoch == total_epochs or is_best:
                ema_state = self.ema.state_dict() if self.ema is not None else None
                self.ckpt_manager.save(
                    epoch=epoch,
                    global_step=self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    val_metric=val_loss,
                    is_best=is_best,
                    ema_state=ema_state,
                    config=self.cfg.to_dict(),
                )

        logger.info(
            "Training complete for Model %s. Best Val Loss: %.4f", self.variant, self.best_val_loss
        )
