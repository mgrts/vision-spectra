"""
Loss function registry.

Provides factory functions for creating loss instances from configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch.nn as nn

from vision_spectra.losses.classification import (
    AsymmetricLoss,
    ClassBalancedLoss,
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
)
from vision_spectra.losses.reconstruction import (
    L1Loss,
    MSELoss,
    SmoothL1Loss,
)

if TYPE_CHECKING:
    from vision_spectra.settings import LossConfig


# Classification loss registry
LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "cross_entropy": CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "class_balanced": ClassBalancedLoss,
    "asymmetric": AsymmetricLoss,
}

# MIM loss registry
MIM_LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "mse": MSELoss,
    "l1": L1Loss,
    "smooth_l1": SmoothL1Loss,
}


def get_loss(
    config: LossConfig,
    samples_per_class: np.ndarray | None = None,
) -> nn.Module:
    """
    Create a classification loss from config.

    Args:
        config: Loss configuration
        samples_per_class: Class counts for class-balanced loss

    Returns:
        Loss module

    Raises:
        ValueError: If loss name is not registered
    """
    loss_name = config.classification.value

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(LOSS_REGISTRY.keys())}")

    # Build kwargs based on loss type
    kwargs: dict[str, Any] = {}

    if loss_name == "focal":
        kwargs["gamma"] = config.focal_gamma
        if config.focal_alpha is not None:
            kwargs["alpha"] = config.focal_alpha

    elif loss_name == "label_smoothing":
        kwargs["epsilon"] = config.label_smoothing

    elif loss_name == "class_balanced":
        if samples_per_class is None:
            raise ValueError("class_balanced loss requires samples_per_class")
        kwargs["samples_per_class"] = samples_per_class
        kwargs["beta"] = config.class_balanced_beta

    elif loss_name == "asymmetric":
        pass  # Uses default parameters

    return LOSS_REGISTRY[loss_name](**kwargs)


def get_mim_loss(config: LossConfig) -> nn.Module:
    """
    Create a MIM reconstruction loss from config.

    Args:
        config: Loss configuration

    Returns:
        Loss module
    """
    loss_name = config.mim.value

    if loss_name not in MIM_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown MIM loss: {loss_name}. Available: {list(MIM_LOSS_REGISTRY.keys())}"
        )

    return MIM_LOSS_REGISTRY[loss_name]()


def register_loss(name: str, loss_class: type[nn.Module]) -> None:
    """
    Register a new classification loss.

    Args:
        name: Loss name for registry
        loss_class: Loss class (must be nn.Module subclass)
    """
    if not issubclass(loss_class, nn.Module):
        raise TypeError(f"Loss class must be nn.Module subclass, got {type(loss_class)}")
    LOSS_REGISTRY[name] = loss_class


def register_mim_loss(name: str, loss_class: type[nn.Module]) -> None:
    """
    Register a new MIM reconstruction loss.

    Args:
        name: Loss name for registry
        loss_class: Loss class (must be nn.Module subclass)
    """
    if not issubclass(loss_class, nn.Module):
        raise TypeError(f"Loss class must be nn.Module subclass, got {type(loss_class)}")
    MIM_LOSS_REGISTRY[name] = loss_class
