"""
Multitask trainer for joint classification and MIM.
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

    from vision_spectra.models.multitask import MultitaskViT
    from vision_spectra.settings import ExperimentConfig


class MultitaskTrainer(BaseTrainer):
    """
    Trainer for joint classification and MIM.

    Loss = cls_weight * cls_loss + mim_weight * mim_loss
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: MultitaskViT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cls_criterion: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__(config, model, train_loader, val_loader)

        self.cls_criterion = cls_criterion.to(self.device)
        self.num_classes = num_classes

        # MTL weights
        self.cls_weight = config.loss.mtl_cls_weight
        self.mim_weight = config.loss.mtl_mim_weight
        self.mask_ratio = config.model.mask_ratio

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
        total_cls_loss = 0.0
        total_mim_loss = 0.0
        num_batches = 0

        max_batches = len(self.train_loader)
        if self.config.training.smoke_test:
            max_batches = min(5, max_batches)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch} [MTL Train]",
            total=max_batches,
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self._warmup_lr(self.current_epoch, batch_idx, len(self.train_loader))

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits, mim_loss, pred, mask = self.model.forward_multitask(
                        images, mask_ratio=self.mask_ratio
                    )
                    cls_loss = self.cls_criterion(logits, targets)
                    loss = self.cls_weight * cls_loss + self.mim_weight * mim_loss

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
                logits, mim_loss, pred, mask = self.model.forward_multitask(
                    images, mask_ratio=self.mask_ratio
                )
                cls_loss = self.cls_criterion(logits, targets)
                loss = self.cls_weight * cls_loss + self.mim_weight * mim_loss
                loss.backward()

                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )

                self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mim_loss += mim_loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            self.train_accuracy.update(preds, targets)

            pbar.set_postfix({"loss": loss.item(), "cls": cls_loss.item(), "mim": mim_loss.item()})

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "cls_loss": total_cls_loss / n,
            "mim_loss": total_mim_loss / n,
            "accuracy": self.train_accuracy.compute().item(),
        }

    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.val_accuracy.reset()
        self.val_f1.reset()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_mim_loss = 0.0
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
                        logits, mim_loss, pred, mask = self.model.forward_multitask(
                            images, mask_ratio=self.mask_ratio
                        )
                        cls_loss = self.cls_criterion(logits, targets)
                else:
                    logits, mim_loss, pred, mask = self.model.forward_multitask(
                        images, mask_ratio=self.mask_ratio
                    )
                    cls_loss = self.cls_criterion(logits, targets)

                loss = self.cls_weight * cls_loss + self.mim_weight * mim_loss

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_mim_loss += mim_loss.item()
                num_batches += 1

                preds = logits.argmax(dim=-1)
                self.val_accuracy.update(preds, targets)
                self.val_f1.update(preds, targets)

        n = max(num_batches, 1)
        return {
            "loss": total_loss / n,
            "cls_loss": total_cls_loss / n,
            "mim_loss": total_mim_loss / n,
            "accuracy": self.val_accuracy.compute().item(),
            "f1_macro": self.val_f1.compute().item(),
        }
