"""
Masked Image Modeling (MIM) pretraining trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import torch
from loguru import logger
from torch.cuda.amp import autocast
from tqdm import tqdm

from vision_spectra.training.base import BaseTrainer
from vision_spectra.utils.visualization import save_mim_examples

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from vision_spectra.models.mim import MIMModel
    from vision_spectra.settings import ExperimentConfig


class MIMTrainer(BaseTrainer):
    """Trainer for Masked Image Modeling pretraining."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: MIMModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_channels: int = 3,
    ) -> None:
        super().__init__(
            config,
            model,
            train_loader,
            val_loader,
            num_channels=num_channels,
        )

        self.mask_ratio = config.model.mask_ratio

    def _save_prediction_examples(self, num_examples: int = 8) -> None:
        """
        Save MIM reconstruction examples as artifacts to MLflow.

        Args:
            num_examples: Number of examples to save
        """
        try:
            logger.debug("Saving MIM examples...")

            # Save MIM visualizations (original, masked, reconstructed)
            saved_paths = save_mim_examples(
                model=self.model,
                dataloader=self.val_loader,
                save_dir=self.artifacts_dir,
                num_examples=num_examples,
                num_channels=self.num_channels,
                device=self.device,
            )

            # Log artifacts to MLflow
            for path in saved_paths:
                mlflow.log_artifact(str(path), artifact_path="mim_examples")

            logger.debug(f"Saved {len(saved_paths)} MIM example images")

        except Exception as e:
            logger.warning(f"Failed to save MIM examples: {e}")

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        max_batches = len(self.train_loader)
        if self.config.training.smoke_test:
            max_batches = min(5, max_batches)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [MIM Train]",
            total=max_batches,
            leave=False,
        )

        for batch_idx, (images, _) in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)

            # Warmup LR
            self._warmup_lr(self.current_epoch, batch_idx, len(self.train_loader))

            self.optimizer.zero_grad()

            # Forward pass
            if self.use_amp:
                with autocast():
                    loss, pred, mask = self.model(images, mask_ratio=self.mask_ratio)

                self.scaler.scale(loss).backward()

                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, pred, mask = self.model(images, mask_ratio=self.mask_ratio)
                loss.backward()

                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )

                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": loss.item()})

        return {"loss": total_loss / max(num_batches, 1)}

    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        max_batches = len(self.val_loader)
        if self.config.training.smoke_test:
            max_batches = min(3, max_batches)

        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break

                images = images.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        loss, pred, mask = self.model(images, mask_ratio=self.mask_ratio)
                else:
                    loss, pred, mask = self.model(images, mask_ratio=self.mask_ratio)

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}
