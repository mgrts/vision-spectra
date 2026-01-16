"""
Classification loss functions.

Implements various losses for handling:
- Standard classification (cross-entropy)
- Imbalanced classes (focal, class-balanced)
- Regularization (label smoothing)
- Multi-label scenarios (asymmetric)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with optional class weights."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction=self.reduction,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference:
        Lin, T.Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights tensor or None for uniform
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)

        focal_term = (1 - p_t) ** self.gamma
        loss = focal_term * ce_loss

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, float | int):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.

    Smoothed targets: y_smooth = (1 - eps) * y_onehot + eps / C

    Args:
        epsilon: Smoothing factor (default: 0.1)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0 <= epsilon < 1:
            raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.

        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        num_classes = logits.size(-1)

        # Convert to one-hot
        targets_onehot = F.one_hot(targets, num_classes).float()

        # Apply smoothing
        targets_smooth = (1 - self.epsilon) * targets_onehot + self.epsilon / num_classes

        # Compute cross-entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    Reference:
        Cui, Y., et al. (2019). Class-Balanced Loss Based on Effective
        Number of Samples. CVPR.

    Weight_c = (1 - beta) / (1 - beta^n_c)

    Args:
        samples_per_class: Number of samples per class
        beta: Hyperparameter (default: 0.9999)
        loss_type: 'cross_entropy' or 'focal'
        gamma: Focal loss gamma (if using focal)
    """

    def __init__(
        self,
        samples_per_class: np.ndarray | list[int],
        beta: float = 0.9999,
        loss_type: str = "cross_entropy",
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        samples_per_class = np.array(samples_per_class)

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, samples_per_class)

        # Compute weights
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(weights)  # Normalize

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.loss_type = loss_type
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced loss.

        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices

        Returns:
            Scalar loss
        """
        # Get per-sample weights
        sample_weights = self.weights[targets]

        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(logits, targets, reduction="none")
            p_t = torch.exp(-ce_loss)
            focal_term = (1 - p_t) ** self.gamma
            loss = focal_term * ce_loss
        else:
            loss = F.cross_entropy(logits, targets, reduction="none")

        # Apply class-balanced weights
        loss = sample_weights * loss

        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Reference:
        Ridnik, T., et al. (2021). Asymmetric Loss For Multi-Label Classification.

    Designed for scenarios with many negative labels, reduces the contribution
    of easy negatives.

    Args:
        gamma_neg: Focusing parameter for negatives (default: 4)
        gamma_pos: Focusing parameter for positives (default: 1)
        clip: Probability clipping threshold (default: 0.05)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Note: For single-label classification, converts to multi-label format.

        Args:
            logits: [B, C] unnormalized logits
            targets: [B] class indices or [B, C] multi-hot

        Returns:
            Scalar loss
        """
        # Convert to multi-label format if needed
        if targets.dim() == 1:
            num_classes = logits.size(-1)
            targets = F.one_hot(targets, num_classes).float()

        # Probabilities
        probs = torch.sigmoid(logits)
        probs_neg = 1 - probs

        # Asymmetric clipping for negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1)

        # Compute losses
        loss_pos = targets * torch.log(probs.clamp(min=1e-8))
        loss_neg = (1 - targets) * torch.log(probs_neg.clamp(min=1e-8))

        # Apply focusing
        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1 - probs) ** self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs**self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        return loss.mean()
