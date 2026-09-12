"""
Centralized configuration using Pydantic settings.

Supports configuration via:
1. CLI arguments (highest priority)
2. YAML config files
3. Default values
"""

from __future__ import annotations

import contextlib
import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Path Configuration
# =============================================================================


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
CONFIGS_DIR = PROJECT_ROOT / "configs"


# =============================================================================
# Enums
# =============================================================================


class DatasetName(str, Enum):
    """Supported datasets."""

    PATHMNIST = "pathmnist"
    PNEUMONIAMNIST = "pneumoniamnist"
    BLOODMNIST = "bloodmnist"
    DERMAMNIST = "dermamnist"
    OCTMNIST = "octmnist"
    ORGANAMNIST = "organamnist"
    SYNTHETIC = "synthetic"


class LossName(str, Enum):
    """Supported classification losses."""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"
    CLASS_BALANCED = "class_balanced"
    ASYMMETRIC = "asymmetric"


class MIMLossName(str, Enum):
    """Supported MIM reconstruction losses."""

    MSE = "mse"
    L1 = "l1"
    SMOOTH_L1 = "smooth_l1"
    HUBER = "huber"
    CAUCHY = "cauchy"
    TUKEY = "tukey"


class OptimizerName(str, Enum):
    """Supported optimizers."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerName(str, Enum):
    """Supported LR schedulers."""

    COSINE = "cosine"
    STEP = "step"
    PLATEAU = "plateau"
    NONE = "none"


# =============================================================================
# Configuration Models
# =============================================================================


class _StrictModel(BaseModel):
    """Base for all config models.

    - ``extra="forbid"``: unknown / misplaced YAML keys raise a ValidationError
      instead of being silently dropped (which would otherwise corrupt
      experiments by silently using defaults).
    - ``validate_assignment=True``: field constraints (e.g. ``0 < mask_ratio <
      1``) are re-checked on attribute assignment, so CLI/runtime overrides
      fail fast with a clear error rather than reaching the model degenerate.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DatasetConfig(_StrictModel):
    """Dataset configuration."""

    name: DatasetName = Field(default=DatasetName.PATHMNIST, description="Dataset name")
    image_size: int = Field(default=28, gt=0, description="Image size (square)")
    batch_size: int = Field(default=64, gt=0, description="Batch size")
    num_workers: int = Field(default=4, ge=0, description="DataLoader workers")
    pin_memory: bool = Field(default=True, description="Pin memory for GPU")
    sample_ratio: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Fraction of dataset to use (0-1]. Use <1 for faster experiments.",
    )
    train_subsample: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Keep only this many TRAIN images (stratified by class); val/test stay full. "
            "Used by the step-matched controls (e.g. PathMNIST-1k). Applied after sample_ratio."
        ),
    )

    # Synthetic dataset specific
    num_classes: int = Field(default=5, gt=1, description="Number of classes (synthetic)")
    num_samples_train: int = Field(default=5000, gt=0, description="Train samples (synthetic)")
    num_samples_val: int = Field(default=1000, gt=0, description="Val samples (synthetic)")
    num_samples_test: int = Field(default=1000, gt=0, description="Test samples (synthetic)")


class ModelConfig(_StrictModel):
    """Model configuration."""

    name: str = Field(
        default="vit_small_patch14_dinov2.lvd142m", description="Model name from timm"
    )
    patch_size: int | None = Field(
        default=None,
        gt=0,
        description="Override the timm model's patch size. With 28px inputs, "
        "patch_size=4 gives a real 7x7=49-patch grid; None uses the model default.",
    )
    pretrained: bool = Field(default=False, description="Use pretrained weights")
    drop_rate: float = Field(default=0.0, ge=0, le=1, description="Dropout rate")
    attn_drop_rate: float = Field(default=0.0, ge=0, le=1, description="Attention dropout")
    drop_path_rate: float = Field(default=0.1, ge=0, le=1, description="Drop path rate")

    # MIM specific
    mask_ratio: float = Field(default=0.75, gt=0, lt=1, description="MIM mask ratio")
    decoder_embed_dim: int = Field(default=128, gt=0, description="MIM decoder dim")
    decoder_depth: int = Field(default=2, gt=0, description="MIM decoder layers")
    decoder_num_heads: int = Field(default=4, gt=0, description="MIM decoder heads")


class LossConfig(_StrictModel):
    """Loss function configuration."""

    # Classification loss
    classification: LossName = Field(default=LossName.CROSS_ENTROPY)
    label_smoothing: float = Field(default=0.1, ge=0, le=1, description="Label smoothing eps")
    focal_gamma: float = Field(default=2.0, ge=0, description="Focal loss gamma")
    focal_alpha: float | None = Field(default=None, description="Focal loss alpha (class weights)")
    class_balanced_beta: float = Field(default=0.9999, ge=0, lt=1, description="CB loss beta")

    # MIM loss
    mim: MIMLossName = Field(default=MIMLossName.MSE)
    mim_norm_pix: bool = Field(default=True, description="Normalize pixel targets for MIM")

    # MTL weights
    mtl_cls_weight: float = Field(default=1.0, ge=0, description="Classification weight in MTL")
    mtl_mim_weight: float = Field(default=0.5, ge=0, description="MIM weight in MTL")


class OptimizerConfig(_StrictModel):
    """Optimizer configuration."""

    name: OptimizerName = Field(default=OptimizerName.ADAMW)
    learning_rate: float = Field(default=1e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.05, ge=0, description="Weight decay")
    momentum: float = Field(default=0.9, ge=0, le=1, description="Momentum (SGD)")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Adam betas")

    # Scheduler
    scheduler: SchedulerName = Field(default=SchedulerName.COSINE)
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")
    min_lr: float = Field(default=1e-6, ge=0, description="Minimum LR")


class TrainingConfig(_StrictModel):
    """Training configuration."""

    epochs: int = Field(default=100, gt=0, description="Number of epochs")
    gradient_clip: float = Field(default=1.0, ge=0, description="Gradient clipping norm")
    mixed_precision: bool = Field(default=True, description="Use AMP")

    # Checkpointing
    save_every_n_epochs: int = Field(default=10, gt=0, description="Checkpoint frequency")
    keep_top_k: int = Field(default=3, gt=0, description="Keep top K checkpoints")

    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=15, gt=0, description="Early stopping patience")

    # Checkpoint-selection / early-stopping monitor. For a fair loss-function
    # comparison, monitor a loss-agnostic metric (accuracy/auroc) so different
    # losses' incomparable loss scales don't select checkpoints at different
    # points on different trajectories.
    monitor: Literal["loss", "accuracy", "auroc", "f1_macro"] = Field(
        default="loss", description="Validation metric used to select the best model"
    )

    # Smoke test mode
    smoke_test: bool = Field(default=False, description="Quick test mode")


class SpectralConfig(_StrictModel):
    """Spectral metrics configuration."""

    enabled: bool = Field(default=True, description="Enable spectral logging")
    log_every_n_epochs: int = Field(default=5, gt=0, description="Logging frequency")
    log_every_n_steps: int | None = Field(default=None, description="Step-based logging")
    log_first_epochs: bool = Field(
        default=True,
        description="Log spectral metrics for first 5 epochs (0-4) to capture initial dynamics",
    )

    # Distribution tracking
    track_distributions: bool = Field(
        default=False, description="Track full singular value distributions (memory intensive)"
    )
    max_singular_values: int = Field(
        default=50, gt=0, description="Max singular values to store per layer"
    )
    save_distribution_history: bool = Field(
        default=False, description="Save distribution history to JSON (memory intensive)"
    )

    # Which layers to analyze (fewer layers = less memory)
    layers: list[str] = Field(
        default=["blocks.0", "blocks.5"],
        description="Layer patterns to analyze",
    )

    # Which matrices to extract
    extract_qkv: bool = Field(default=True, description="Extract Q/K/V weights")
    extract_mlp: bool = Field(default=False, description="Extract MLP weights")
    extract_patch_embed: bool = Field(default=True, description="Extract patch embedding")


class ExperimentConfig(_StrictModel):
    """Complete experiment configuration."""

    # Experiment metadata
    name: str = Field(default="experiment", description="Experiment name")
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(default="auto")

    # Sub-configurations
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)

    # Paths
    output_dir: Path = Field(default=MLRUNS_DIR, description="Output directory for mlflow")
    data_dir: Path = Field(default=DATA_DIR, description="Data directory")

    def get_device(self) -> torch.device:
        """Resolve the configured device string to a torch.device."""
        return resolve_device(self.device)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path) -> None:
        """Save to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten config for logging."""
        return _flatten_dict(self.model_dump(mode="json"))

    def get_run_dir(self) -> Path:
        """Get run-specific output directory."""
        run_dir = self.output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dictionary."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list | tuple):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


# =============================================================================
# Device / Reproducibility
# =============================================================================


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string ('auto'/'cpu'/'cuda'/'mps') to a torch.device.

    The single canonical device resolver for the whole project (auto ->
    cuda -> mps -> cpu).
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Enable deterministic operations (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # torch.use_deterministic_algorithms is a GLOBAL setting that also governs
    # CPU/MPS op determinism, so it must NOT be gated on CUDA availability
    # (otherwise deterministic=True is a no-op on the Mac/CPU dev path).
    if deterministic:
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
