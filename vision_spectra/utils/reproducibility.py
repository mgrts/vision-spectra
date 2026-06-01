"""
Reproducibility utilities.

``set_seed`` and ``get_device`` are the single canonical implementations defined
in ``vision_spectra.settings`` and re-exported here for convenience, so there is
only one copy of each to maintain.
"""

from __future__ import annotations

import torch

from vision_spectra.settings import resolve_device as get_device
from vision_spectra.settings import set_seed

__all__ = ["set_seed", "get_device", "count_parameters"]


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
