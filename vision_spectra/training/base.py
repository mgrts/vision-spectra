"""
Base trainer with shared functionality.

Provides:
- Optimizer and scheduler creation
- Mixed precision training
- Checkpointing
- Spectral metrics logging
- MLflow integration
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from vision_spectra.metrics.extraction import extract_all_weights
from vision_spectra.metrics.spectral import (
    SpectralTracker,
    aggregate_spectral_metrics,
    get_spectral_metrics,
)
from vision_spectra.utils.visualization import save_prediction_examples

if TYPE_CHECKING:
    from vision_spectra.settings import ExperimentConfig


class BaseTrainer(ABC):
    """
    Abstract base trainer with common functionality.

    Subclasses must implement:
    - train_epoch()
    - validate()
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int = 10,
        num_channels: int = 3,
        class_names: list[str] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.class_names = class_names

        # Setup device
        self.device = config.get_device()
        self.model = self.model.to(self.device)

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Mixed precision
        self.use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.best_val_metric = float("inf")  # Assume lower is better
        self.patience_counter = 0

        # Output directory
        self.run_dir = config.get_run_dir()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Spectral distribution tracker
        self.spectral_tracker: SpectralTracker | None = None
        if config.spectral.enabled and config.spectral.track_distributions:
            self.spectral_tracker = SpectralTracker(
                layer_patterns=config.spectral.layers,
                include_qkv=config.spectral.extract_qkv,
                include_mlp=config.spectral.extract_mlp,
                include_patch_embed=config.spectral.extract_patch_embed,
                max_singular_values=config.spectral.max_singular_values,
            )

        # Smoke test mode
        if config.training.smoke_test:
            logger.warning("Smoke test mode: limiting iterations")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.config.optimizer
        params = self.model.parameters()

        if opt_config.name.value == "adam":
            return Adam(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name.value == "adamw":
            return AdamW(
                params,
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                weight_decay=opt_config.weight_decay,
            )
        elif opt_config.name.value == "sgd":
            return SGD(
                params,
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config.name}")

    def _create_scheduler(self):
        """Create learning rate scheduler from config."""
        opt_config = self.config.optimizer
        total_epochs = self.config.training.epochs

        if opt_config.scheduler.value == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - opt_config.warmup_epochs,
                eta_min=opt_config.min_lr,
            )
        elif opt_config.scheduler.value == "step":
            return StepLR(
                self.optimizer,
                step_size=max(1, total_epochs // 3),
                gamma=0.1,
            )
        elif opt_config.scheduler.value == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        else:
            return None

    def _warmup_lr(self, epoch: int, step: int, total_steps: int) -> None:
        """Apply learning rate warmup."""
        warmup_epochs = self.config.optimizer.warmup_epochs

        if epoch < warmup_epochs:
            warmup_steps = warmup_epochs * total_steps
            current_step = epoch * total_steps + step
            lr = self.config.optimizer.learning_rate * current_step / warmup_steps

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    @abstractmethod
    def train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def validate(self) -> dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        pass

    def train(self) -> dict[str, Any]:
        """
        Full training loop.

        Returns:
            Dictionary with final metrics and best checkpoint path
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

        # Setup MLflow
        mlflow.set_tracking_uri(str(self.config.output_dir.parent / "mlruns"))
        mlflow.set_experiment(self.config.name)

        with mlflow.start_run():
            # Log config
            mlflow.log_params(self.config.to_flat_dict())

            best_checkpoint = None

            for epoch in range(self.config.training.epochs):
                self.current_epoch = epoch

                # Train
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Log metrics
                self._log_metrics(train_metrics, prefix="train")
                self._log_metrics(val_metrics, prefix="val")

                # Log spectral metrics
                if self._should_log_spectral(epoch):
                    spectral_metrics = self._compute_spectral_metrics()
                    self._log_metrics(spectral_metrics, prefix="spectral")

                    # Track spectral distributions
                    if self.spectral_tracker is not None:
                        self.model.eval()
                        snapshot = self.spectral_tracker.record_epoch(self.model, epoch)
                        logger.debug(
                            f"Recorded spectral snapshot for epoch {epoch} "
                            f"({len(snapshot.distributions)} layers)"
                        )

                # Update scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics.get("loss", 0))
                    elif epoch >= self.config.optimizer.warmup_epochs:
                        self.scheduler.step()

                # Log learning rate
                mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"], step=epoch)

                # Checkpointing
                val_metric = val_metrics.get("loss", val_metrics.get("accuracy", 0))
                is_best = self._is_best(val_metric)

                if is_best:
                    self.best_val_metric = val_metric
                    best_checkpoint = self._save_checkpoint("best.pt")
                    self.patience_counter = 0

                    # Save prediction examples for best model
                    self._save_prediction_examples()
                else:
                    self.patience_counter += 1

                if epoch % self.config.training.save_every_n_epochs == 0:
                    self._save_checkpoint(f"epoch_{epoch:04d}.pt")

                # Early stopping
                if (
                    self.config.training.early_stopping
                    and self.patience_counter >= self.config.training.patience
                ):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                # Logging
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics.get('loss', 0):.4f}, "
                    f"val_loss={val_metrics.get('loss', 0):.4f}, "
                    f"val_acc={val_metrics.get('accuracy', 0):.4f}"
                )

                # Smoke test: exit early
                if self.config.training.smoke_test and epoch >= 1:
                    logger.info("Smoke test complete")
                    break

            # Log best checkpoint
            if best_checkpoint:
                mlflow.log_artifact(str(best_checkpoint))

            # Save spectral distribution history and plots
            if (
                self.spectral_tracker is not None
                and self.config.spectral.save_distribution_history
                and self.spectral_tracker.history
            ):
                # Save JSON history
                spectral_history_path = self.artifacts_dir / "spectral_history.json"
                self.spectral_tracker.save(spectral_history_path)
                mlflow.log_artifact(str(spectral_history_path))
                logger.info(f"Saved spectral distribution history: {spectral_history_path}")

                # Save spectral distribution plots
                from vision_spectra.utils.visualization import (
                    save_spectral_distribution_plots,
                )

                try:
                    plot_paths = save_spectral_distribution_plots(
                        self.spectral_tracker,
                        self.artifacts_dir,
                        prefix="",
                    )
                    for plot_path in plot_paths:
                        mlflow.log_artifact(str(plot_path))
                    logger.info(f"Saved {len(plot_paths)} spectral distribution plots")
                except Exception as e:
                    logger.warning(f"Failed to save spectral plots: {e}")

            return {
                "best_val_metric": self.best_val_metric,
                "best_checkpoint": best_checkpoint,
                "final_epoch": self.current_epoch,
                "spectral_tracker": self.spectral_tracker,
            }

    def _should_log_spectral(self, epoch: int) -> bool:
        """Check if spectral metrics should be logged this epoch."""
        if not self.config.spectral.enabled:
            return False
        return epoch % self.config.spectral.log_every_n_epochs == 0

    def _compute_spectral_metrics(self) -> dict[str, float]:
        """Compute spectral metrics for configured layers."""
        self.model.eval()

        weights = extract_all_weights(
            self.model,
            layer_patterns=self.config.spectral.layers,
            include_qkv=self.config.spectral.extract_qkv,
            include_mlp=self.config.spectral.extract_mlp,
            include_patch_embed=self.config.spectral.extract_patch_embed,
        )

        if not weights:
            return {}

        # Compute metrics for each weight matrix
        all_metrics = []
        metrics_by_type: dict[str, list[dict]] = {}

        for w in weights:
            m = get_spectral_metrics(w.weight)
            all_metrics.append(m)

            if w.matrix_type not in metrics_by_type:
                metrics_by_type[w.matrix_type] = []
            metrics_by_type[w.matrix_type].append(m)

        # Aggregate
        result = aggregate_spectral_metrics(all_metrics)

        # Also aggregate by type
        for matrix_type, type_metrics in metrics_by_type.items():
            type_agg = aggregate_spectral_metrics(type_metrics)
            for k, v in type_agg.items():
                result[f"{matrix_type}_{k}"] = v

        return result

    def _is_best(self, val_metric: float) -> bool:
        """Check if current validation metric is best."""
        # Lower is better for loss
        return val_metric < self.best_val_metric

    def _save_prediction_examples(self, num_examples: int = 16) -> None:
        """
        Save prediction examples as artifacts to MLflow.

        Args:
            num_examples: Number of examples to save
        """
        try:
            logger.debug("Saving prediction examples...")

            # Save prediction visualizations
            saved_paths = save_prediction_examples(
                model=self.model,
                dataloader=self.val_loader,
                save_dir=self.artifacts_dir,
                num_examples=num_examples,
                num_channels=self.num_channels,
                class_names=self.class_names,
                device=self.device,
            )

            # Log artifacts to MLflow
            for path in saved_paths:
                mlflow.log_artifact(str(path), artifact_path="predictions")

            logger.debug(f"Saved {len(saved_paths)} prediction example images")

        except Exception as e:
            logger.warning(f"Failed to save prediction examples: {e}")

    def _log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Log metrics to MLflow."""
        for k, v in metrics.items():
            if np.isfinite(v):
                key = f"{prefix}/{k}" if prefix else k
                mlflow.log_metric(key, v, step=self.current_epoch)

    def _save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename

        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_metric": self.best_val_metric,
            "config": self.config.model_dump(),
        }

        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, path)
        logger.debug(f"Saved checkpoint: {path}")

        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_epoch = state["epoch"]
        self.best_val_metric = state.get("best_val_metric", float("inf"))

        if self.scheduler is not None and "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")

    def cleanup(self) -> None:
        """
        Clean up resources to free memory.

        This should be called after training is complete, especially when
        running multiple experiments in sequence. It moves the model to CPU,
        clears CUDA cache, and removes references to large objects.
        """
        import gc

        # Move model to CPU to free GPU memory
        if hasattr(self, "model") and self.model is not None:
            self.model.cpu()
            del self.model
            self.model = None

        # Clear optimizer state (can hold GPU tensors)
        if hasattr(self, "optimizer") and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None

        # Clear scheduler
        if hasattr(self, "scheduler") and self.scheduler is not None:
            del self.scheduler
            self.scheduler = None

        # Clear GradScaler
        if hasattr(self, "scaler") and self.scaler is not None:
            del self.scaler
            self.scaler = None

        # Clear spectral tracker (can hold numpy arrays)
        if hasattr(self, "spectral_tracker") and self.spectral_tracker is not None:
            self.spectral_tracker.history.clear()
            del self.spectral_tracker
            self.spectral_tracker = None

        # Clear data loader references
        if hasattr(self, "train_loader"):
            del self.train_loader
            self.train_loader = None
        if hasattr(self, "val_loader"):
            del self.val_loader
            self.val_loader = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.debug("Trainer resources cleaned up")
