"""
Classification trainer.

Trains a Vision Transformer for image classification with:
- Configurable loss functions
- Mixed precision training
- Spectral metrics logging
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchmetrics import Accuracy, F1Score
from tqdm import tqdm

from vision_spectra.training.base import BaseTrainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from vision_spectra.settings import ExperimentConfig


class ClassificationTrainer(BaseTrainer):
    """Trainer for supervised image classification."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__(config, model, train_loader, val_loader)

        self.criterion = criterion.to(self.device)
        self.num_classes = num_classes

        # Metrics
        task = "multiclass"
        self.train_accuracy = Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.val_accuracy = Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.val_f1 = F1Score(task=task, num_classes=num_classes, average="macro").to(self.device)

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_accuracy.reset()

        total_loss = 0.0
        num_batches = 0

        # Determine iteration limit for smoke test
        max_batches = len(self.train_loader)
        if self.config.training.smoke_test:
            max_batches = min(5, max_batches)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [Train]",
            total=max_batches,
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Warmup LR
            self._warmup_lr(self.current_epoch, batch_idx, len(self.train_loader))

            self.optimizer.zero_grad()

            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

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
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()

                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )

                self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            self.train_accuracy.update(preds, targets)

            pbar.set_postfix({"loss": loss.item()})

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": self.train_accuracy.compute().item(),
        }

    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.val_accuracy.reset()
        self.val_f1.reset()

        total_loss = 0.0
        num_batches = 0

        max_batches = len(self.val_loader)
        if self.config.training.smoke_test:
            max_batches = min(3, max_batches)

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.use_amp:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                num_batches += 1

                preds = logits.argmax(dim=-1)
                self.val_accuracy.update(preds, targets)
                self.val_f1.update(preds, targets)

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": self.val_accuracy.compute().item(),
            "f1_macro": self.val_f1.compute().item(),
        }

    def _is_best(self, val_metric: float) -> bool:
        """For classification, higher accuracy is better."""
        # We track val loss, so lower is still better
        return val_metric < self.best_val_metric
