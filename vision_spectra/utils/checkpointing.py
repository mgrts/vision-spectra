"""
Checkpointing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        path: Output path
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        epoch: Current epoch
        metrics: Optional metrics dictionary
        config: Optional config dictionary
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        state["metrics"] = metrics

    if config is not None:
        state["config"] = config

    torch.save(state, path)
    logger.debug(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to map tensors to
        strict: Strict loading for model state dict

    Returns:
        Dictionary with epoch, metrics, and config if present
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model
    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    # Load optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint.get('epoch', 'unknown')})")

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config", {}),
    }


def get_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """
    Find the best checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to best checkpoint, or None if not found
    """
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return best_path

    # Fall back to latest epoch checkpoint
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if checkpoints:
        return checkpoints[-1]

    return None
