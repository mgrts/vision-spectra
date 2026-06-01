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


def _is_head_param(name: str) -> bool:
    """Return True if a parameter name refers to the classification head.

    The ViTClassifier head is the timm ``head`` linear (``encoder.head.*``). We
    deliberately anchor on the ``head`` module name instead of a bare ``fc``
    substring, because every transformer block's MLP is named ``mlp.fc1``/
    ``mlp.fc2`` and must NOT be treated as the head.
    """
    return name.endswith(("head.weight", "head.bias")) or ".head." in name or "classifier" in name


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
        num_channels: int = 3,
        class_names: list[str] | None = None,
        pretrained_path: Path | None = None,
        freeze_encoder: bool = False,
        encoder_lr_scale: float = 0.1,
    ) -> None:
        # Load pretrained weights before parent init
        if pretrained_path is not None:
            self._load_pretrained(model, pretrained_path)

        super().__init__(
            config,
            model,
            train_loader,
            val_loader,
            criterion,
            num_classes,
            num_channels=num_channels,
            class_names=class_names,
        )

        self.freeze_encoder = freeze_encoder
        self.encoder_lr_scale = encoder_lr_scale

        # Apply freezing if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Recreate optimizer with layer-wise LR if not freezing
        if not freeze_encoder and encoder_lr_scale != 1.0:
            self.optimizer = self._create_layerwise_optimizer()

    def _load_pretrained(self, model: nn.Module, path: Path) -> None:
        """Load pretrained encoder weights from a checkpoint.

        MIM checkpoints are produced by ``MIMModel.state_dict()``. Because
        ``MIMModel`` stores a ``ViTClassifier`` as ``self.encoder`` and the
        ``ViTClassifier`` stores the timm backbone as ``self.encoder``, the
        checkpoint keys are double-prefixed (``encoder.encoder.blocks.0...``).
        The finetune target is a bare ``ViTClassifier`` whose params are
        ``encoder.blocks.0...``, so the extra ``encoder.`` prefix is stripped
        here. Only the classification head is skipped; transformer MLP layers
        (``blocks.N.mlp.fc1/fc2``) are deliberately retained.
        """
        logger.info(f"Loading pretrained weights from {path}")

        # weights_only=False: checkpoints embed the full experiment config
        # (Path/enum objects) and are produced by this trusted codebase.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        state_dict = checkpoint.get("model_state_dict", checkpoint)

        encoder_state = {}
        for k, v in state_dict.items():
            # Strip the MIM double-prefix: "encoder.encoder.X" -> "encoder.X"
            if k.startswith("encoder.encoder."):
                new_k = k[len("encoder.") :]
            elif k.startswith("encoder."):
                new_k = k
            elif k.startswith(("decoder.", "decoder_pos_embed")):
                continue  # MIM decoder, not part of the classifier
            else:
                new_k = "encoder." + k

            # Skip ONLY the classification head, not block-internal mlp.fc1/fc2
            if _is_head_param(new_k):
                continue
            encoder_state[new_k] = v

        # Load with strict=False to allow the (intentionally) missing head keys
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        matched = sum(1 for k in model.state_dict() if k in encoder_state)

        if matched == 0:
            raise RuntimeError(
                f"No pretrained encoder weights matched the target model from {path}; "
                "the checkpoint key layout may have changed."
            )

        logger.info(
            f"Loaded {matched}/{len(model.state_dict())} params "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    def _freeze_encoder(self) -> None:
        """Freeze encoder weights, only train classification head."""
        logger.info("Freezing encoder weights")

        for name, param in self.model.named_parameters():
            # Only train classification head (encoder MLP fc1/fc2 stay frozen)
            if _is_head_param(name):
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

            if _is_head_param(name):
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
