"""
Reconstruction loss functions for MIM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error loss for reconstruction."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Scalar loss
        """
        loss = (pred - target) ** 2

        if mask is not None:
            # Average per-element, then apply mask
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss for reconstruction."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute L1 loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Scalar loss
        """
        loss = torch.abs(pred - target)

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 (Huber) loss for reconstruction.

    Combines L1 and L2: quadratic for small errors, linear for large.

    Args:
        beta: Threshold for switching between L1 and L2 (default: 1.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute Smooth L1 loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Scalar loss
        """
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
