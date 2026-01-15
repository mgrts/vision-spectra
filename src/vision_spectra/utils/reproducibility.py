"""
Reproducibility utilities.
"""

from __future__ import annotations

import contextlib
import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed
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
            # Enable deterministic algorithms
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device: str = "auto") -> torch.device:
    """
    Get PyTorch device.

    Args:
        device: Device specification ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        torch.device instance
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device)


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
