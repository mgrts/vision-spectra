"""Loss functions for vision spectra experiments."""

from vision_spectra.losses.classification import (
    AsymmetricLoss,
    ClassBalancedLoss,
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
)
from vision_spectra.losses.reconstruction import (
    CauchyLoss,
    HuberLoss,
    L1Loss,
    MSELoss,
    SGTLoss,
    SmoothL1Loss,
    TukeyLoss,
)
from vision_spectra.losses.registry import LOSS_REGISTRY, get_loss, get_mim_loss

__all__ = [
    "LOSS_REGISTRY",
    "get_loss",
    "get_mim_loss",
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "ClassBalancedLoss",
    "AsymmetricLoss",
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "CauchyLoss",
    "SGTLoss",
    "HuberLoss",
    "TukeyLoss",
]
