"""
Finetuning trainer for pretrained models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from loguru import logger

from vision_spectra.training.classification import ClassificationTrainer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from vision_spectra.settings import ExperimentConfig


class FinetuneTrainer(ClassificationTrainer):
    """
    Trainer for finetuning a pretrained model.

    Extends ClassificationTrainer with:
    - Loading pretrained weights
    - Optional layer freezing
    - Lower learning rate for pretrained layers
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        num_classes: int,
        pretrained_path: Path | None = None,
        freeze_encoder: bool = False,
        encoder_lr_scale: float = 0.1,
    ) -> None:
        # Load pretrained weights before parent init
        if pretrained_path is not None:
            self._load_pretrained(model, pretrained_path)

        super().__init__(config, model, train_loader, val_loader, criterion, num_classes)

        self.freeze_encoder = freeze_encoder
        self.encoder_lr_scale = encoder_lr_scale

        # Apply freezing if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Recreate optimizer with layer-wise LR if not freezing
        if not freeze_encoder and encoder_lr_scale != 1.0:
            self.optimizer = self._create_layerwise_optimizer()

    def _load_pretrained(self, model: nn.Module, path: Path) -> None:
        """Load pretrained weights from checkpoint."""
        logger.info(f"Loading pretrained weights from {path}")

        checkpoint = torch.load(path, map_location="cpu")

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        # Filter out classification head if present
        encoder_state = {}
        for k, v in state_dict.items():
            # Skip classification head weights
            if "head" in k or "classifier" in k or "fc" in k:
                continue
            # Handle MIM model: extract encoder weights
            if k.startswith("encoder."):
                encoder_state[k] = v
            else:
                encoder_state[k] = v

        # Load with strict=False to allow missing/extra keys
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)

        if missing:
            logger.warning(f"Missing keys in pretrained: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys in pretrained: {unexpected[:5]}...")

        logger.info("Pretrained weights loaded successfully")

    def _freeze_encoder(self) -> None:
        """Freeze encoder weights, only train classification head."""
        logger.info("Freezing encoder weights")

        for name, param in self.model.named_parameters():
            # Only train classification head
            if "head" in name or "classifier" in name or "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,}")

    def _create_layerwise_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different LR for encoder and head."""
        encoder_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "head" in name or "classifier" in name or "fc" in name:
                head_params.append(param)
            else:
                encoder_params.append(param)

        opt_config = self.config.optimizer
        base_lr = opt_config.learning_rate

        param_groups = [
            {"params": encoder_params, "lr": base_lr * self.encoder_lr_scale},
            {"params": head_params, "lr": base_lr},
        ]

        logger.info(
            f"Layer-wise LR: encoder={base_lr * self.encoder_lr_scale:.2e}, " f"head={base_lr:.2e}"
        )

        if opt_config.name.value == "adamw":
            return torch.optim.AdamW(
                param_groups,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay,
            )
        else:
            return torch.optim.Adam(
                param_groups,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay,
            )
