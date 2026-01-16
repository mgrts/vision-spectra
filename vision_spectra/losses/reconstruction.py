"""
Reconstruction loss functions for MIM.
"""

from __future__ import annotations

import math
from typing import Literal

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


class CauchyLoss(nn.Module):
    """
    Cauchy (Lorentzian) loss function for robust reconstruction.

    This loss is more robust to outliers than MSE, as it has heavier tails.
    The loss is: log(1 + (pred - target)^2 / gamma^2)

    This is derived from the negative log-likelihood of the Cauchy distribution.

    Args:
        gamma: Scale parameter controlling outlier sensitivity (default: 1.0)
               Larger gamma = more tolerance to outliers
        reduction: Reduction mode ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        gamma: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")

        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Cauchy loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Loss value (reduced according to self.reduction)
        """
        diff = pred - target
        # Use log1p for numerical stability: log(1 + x) where x = diff^2 / gamma^2
        loss = torch.log1p((diff / self.gamma) ** 2)

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, reduction='{self.reduction}'"


def _log_beta(a: float, b: float) -> float:
    """Compute log(Beta(a, b)) = log(Gamma(a)) + log(Gamma(b)) - log(Gamma(a+b))."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class SGTLoss(nn.Module):
    """
    Skewed Generalized T-distribution negative log-likelihood loss.

    This loss function is based on the SGT distribution, which generalizes
    the Student's t-distribution with additional skewness and shape parameters.
    It is particularly useful for regression tasks with heavy-tailed or
    asymmetric residual distributions.

    The SGT distribution includes many common distributions as special cases:
    - p=2, q→∞, λ=0: Normal distribution
    - p=2, λ=0: Student's t-distribution
    - p=1, q→∞, λ=0: Laplace distribution

    Args:
        p: Shape parameter controlling tail heaviness (default: 2.0)
           p=2 gives quadratic behavior near zero (like MSE)
           p=1 gives linear behavior (like MAE)
        q: Shape parameter; larger values -> lighter tails (default: 2.0)
           q→∞ approaches generalized normal distribution
        lam: Skewness parameter in (-1, 1); 0 = symmetric (default: 0.0)
        sigma: Scale parameter; must be > 0 (default: 1.0)
        reduction: Reduction mode ('mean', 'sum', or 'none')

    References:
        Hansen, C., McDonald, J. B., & Newey, W. K. (2010).
        "Instrumental variables estimation with flexible distributions."
        Journal of Business & Economic Statistics.
    """

    def __init__(
        self,
        p: float = 2.0,
        q: float = 2.0,
        lam: float = 0.0,
        sigma: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        # Validate parameters
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if q <= 0:
            raise ValueError(f"q must be positive, got {q}")
        if not (-1 < lam < 1):
            raise ValueError(f"lam must be in (-1, 1), got {lam}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        self.p = p
        self.q = q
        self.lam = lam
        self.sigma = sigma
        self.reduction = reduction

        # Precompute constants (these don't depend on data)
        # Beta function ratios for the SGT normalization
        self._log_B1 = _log_beta(1.0 / p, q)
        self._log_B2 = _log_beta(2.0 / p, q - 1.0 / p) if q > 1.0 / p else 0.0
        self._log_B3 = _log_beta(3.0 / p, q - 2.0 / p) if q > 2.0 / p else 0.0

        B1 = math.exp(self._log_B1)
        B2 = math.exp(self._log_B2) if q > 1.0 / p else 1.0
        B3 = math.exp(self._log_B3) if q > 2.0 / p else 1.0

        # Compute v (scaling factor) - avoid numerical issues
        v_denom_sq = (1 + 3 * lam**2) * (B3 / B1) - 4 * lam**2 * (B2 / B1) ** 2
        v_denom = math.sqrt(max(v_denom_sq, 1e-10))
        self._v = (q ** (-1.0 / p)) / v_denom

        # Mode adjustment for skewness
        self._m = 2 * lam * self._v * sigma * (q ** (1.0 / p)) * B2 / B1

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the SGT loss (negative log-likelihood).

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Scalar loss value
        """
        p = self.p
        q = self.q
        lam = self.lam
        sigma = self.sigma
        v = self._v
        m = self._m

        # Compute centered and scaled residuals
        diff = target - pred + m
        z = torch.abs(diff) / (sigma * v + 1e-10)

        # Compute skew adjustment: (1 + λ * sign(diff))^p
        sign_diff = torch.sign(diff)
        skew_term = (1 + lam * sign_diff) ** p

        # Core SGT loss: (q + 1/p) * log(1 + |z|^p / (q * skew_term))
        ratio = (z**p) / (q * skew_term + 1e-10)
        loss = (q + 1.0 / p) * torch.log1p(ratio)

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def extra_repr(self) -> str:
        return f"p={self.p}, q={self.q}, lam={self.lam}, sigma={self.sigma}"


class HuberLoss(nn.Module):
    """
    Huber loss for robust reconstruction.

    Combines the best of L1 and L2 losses:
    - Quadratic for small errors (|error| < delta): 0.5 * error^2
    - Linear for large errors (|error| >= delta): delta * (|error| - 0.5 * delta)

    This is equivalent to SmoothL1Loss but with a configurable delta parameter.
    The loss is differentiable everywhere and robust to outliers.

    Args:
        delta: Threshold for switching between quadratic and linear (default: 1.0)
               Smaller delta = more robust to outliers but less efficient for small errors
        reduction: Reduction mode ('mean', 'sum', or 'none')
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        self.delta = delta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Huber loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Loss value
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Quadratic for small errors, linear for large errors
        quadratic = 0.5 * diff**2
        linear = self.delta * (abs_diff - 0.5 * self.delta)

        loss = torch.where(abs_diff <= self.delta, quadratic, linear)

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def extra_repr(self) -> str:
        return f"delta={self.delta}, reduction='{self.reduction}'"


class TukeyLoss(nn.Module):
    """
    Tukey's biweight (bisquare) loss for robust reconstruction.

    This loss function completely ignores outliers beyond a threshold c,
    making it extremely robust to gross errors.

    For |error| <= c:
        loss = (c^2 / 6) * (1 - (1 - (error/c)^2)^3)
    For |error| > c:
        loss = c^2 / 6  (constant, gradient = 0)

    The key property is that the influence function (gradient) becomes zero
    for large residuals, completely rejecting outliers.

    Args:
        c: Tuning constant controlling outlier rejection threshold (default: 4.685)
            The default value of 4.685 gives 95% asymptotic efficiency
            for normal distributions while rejecting outliers.
            Smaller c = more aggressive outlier rejection
        reduction: Reduction mode ('mean', 'sum', or 'none')

    References:
        Beaton, A. E., & Tukey, J. W. (1974). "The fitting of power series,
        meaning polynomials, illustrated on band-spectroscopic data."
        Technometrics, 16(2), 147-185.
    """

    def __init__(
        self,
        c: float = 4.685,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        if c <= 0:
            raise ValueError(f"c must be positive, got {c}")

        self.c = c
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the Tukey biweight loss.

        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = compute loss, 0 = ignore)

        Returns:
            Loss value
        """
        diff = pred - target
        abs_diff = torch.abs(diff)

        # Normalized residuals
        u = diff / self.c

        # Maximum loss value (for outliers)
        max_loss = self.c**2 / 6.0

        # Tukey biweight: (c^2/6) * (1 - (1 - u^2)^3) for |u| <= 1
        # This equals max_loss for |u| > 1
        inner = 1 - u**2
        tukey_loss = max_loss * (1 - inner**3)

        # For |diff| > c, use constant max_loss (no gradient)
        loss = torch.where(abs_diff <= self.c, tukey_loss, torch.full_like(tukey_loss, max_loss))

        if mask is not None:
            if loss.dim() > mask.dim():
                loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def extra_repr(self) -> str:
        return f"c={self.c}, reduction='{self.reduction}'"
