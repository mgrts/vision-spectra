"""
Gradient alignment metrics for spectral analysis.

This module implements the gradient alignment analysis discussed in research meetings:
- Compare actual training gradient direction with rank-minimizing gradient
- Measure cosine similarity between gradients
- Track alignment throughout training

Key Hypothesis:
    Networks learn by implicitly minimizing the rank of weight matrices.
    Higher alignment with rank-minimizing direction → stronger implicit regularization.

References:
    - Arora et al. (2019). "Implicit Regularization in Deep Matrix Factorization."
    - Gunasekar et al. (2017). "Implicit Regularization in Matrix Factorization."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn
from scipy.linalg import svd


@dataclass
class GradientAlignmentResult:
    """Result of gradient alignment analysis for a single weight matrix.

    Attributes:
        layer_name: Name of the layer/weight matrix
        cosine_similarity: Cosine similarity between training and rank-reducing gradients
        training_grad_norm: L2 norm of the training gradient
        rank_reducing_grad_norm: L2 norm of the rank-reducing gradient
        angle_degrees: Angle between gradients in degrees
        is_aligned: Whether gradients are positively aligned (cos > 0)
    """

    layer_name: str
    cosine_similarity: float
    training_grad_norm: float
    rank_reducing_grad_norm: float
    angle_degrees: float
    is_aligned: bool


def compute_rank_reducing_gradient(weight: np.ndarray, rank_target: int = 1) -> np.ndarray:
    """
    Compute the gradient direction that reduces matrix rank.

    The rank-reducing direction is derived from the SVD. To reduce rank,
    we want to push singular values toward zero, especially the smaller ones.

    For a matrix W = U @ diag(σ) @ V^T, the nuclear norm gradient gives
    the rank-reducing direction: ∂||W||_* / ∂W = U @ V^T

    Args:
        weight: 2D weight matrix
        rank_target: Target rank (not directly used, kept for future extensions)

    Returns:
        Gradient matrix of same shape as weight, pointing toward lower rank
    """
    try:
        U, s, Vt = svd(weight.astype(np.float64), full_matrices=False)
        return U @ Vt
    except Exception:
        return np.zeros_like(weight)


def compute_gradient_alignment(
    training_grad: np.ndarray,
    weight: np.ndarray,
) -> GradientAlignmentResult:
    """
    Compute alignment between training gradient and rank-reducing gradient.

    Args:
        training_grad: Gradient from training loss (same shape as weight)
        weight: Current weight matrix

    Returns:
        GradientAlignmentResult with alignment metrics
    """
    rank_grad = compute_rank_reducing_gradient(weight)

    train_flat = training_grad.flatten().astype(np.float64)
    rank_flat = rank_grad.flatten().astype(np.float64)

    train_norm = np.linalg.norm(train_flat)
    rank_norm = np.linalg.norm(rank_flat)

    if train_norm < 1e-10 or rank_norm < 1e-10:
        return GradientAlignmentResult(
            layer_name="",
            cosine_similarity=0.0,
            training_grad_norm=float(train_norm),
            rank_reducing_grad_norm=float(rank_norm),
            angle_degrees=90.0,
            is_aligned=False,
        )

    cos_sim = float(np.dot(train_flat, rank_flat) / (train_norm * rank_norm))
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cos_sim)))

    return GradientAlignmentResult(
        layer_name="",
        cosine_similarity=cos_sim,
        training_grad_norm=float(train_norm),
        rank_reducing_grad_norm=float(rank_norm),
        angle_degrees=angle,
        is_aligned=cos_sim > 0,
    )


def analyze_model_gradient_alignment(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
) -> list[GradientAlignmentResult]:
    """
    Analyze gradient alignment for all weight matrices in a model.

    This should be called after loss.backward() but before optimizer.step()
    to capture the current training gradients.

    Args:
        model: PyTorch model with gradients computed (.grad attributes populated)
        layer_patterns: Optional patterns to filter layers

    Returns:
        List of GradientAlignmentResult for each analyzed weight matrix
    """
    results = []

    for name, param in model.named_parameters():
        if layer_patterns and not any(pat in name for pat in layer_patterns):
            continue

        if param.dim() != 2:
            continue

        if param.grad is None:
            continue

        weight = param.detach().cpu().numpy()
        grad = param.grad.detach().cpu().numpy()

        result = compute_gradient_alignment(grad, weight)
        result.layer_name = name
        results.append(result)

    return results


def aggregate_gradient_alignment(
    results: list[GradientAlignmentResult],
) -> dict[str, float]:
    """
    Aggregate gradient alignment results across layers.

    Args:
        results: List of GradientAlignmentResult

    Returns:
        Dictionary with aggregated metrics
    """
    if not results:
        return {
            "cos_sim_mean": np.nan,
            "cos_sim_std": np.nan,
            "cos_sim_min": np.nan,
            "cos_sim_max": np.nan,
            "fraction_aligned": np.nan,
            "angle_mean": np.nan,
        }

    cos_sims = [r.cosine_similarity for r in results]
    angles = [r.angle_degrees for r in results]
    aligned_count = sum(1 for r in results if r.is_aligned)

    return {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
        "cos_sim_min": float(np.min(cos_sims)),
        "cos_sim_max": float(np.max(cos_sims)),
        "fraction_aligned": float(aligned_count / len(results)),
        "angle_mean": float(np.mean(angles)),
    }


class GradientAlignmentTracker:
    """
    Tracks gradient alignment throughout training.

    Attributes:
        history: List of (step, aggregated_metrics, per_layer_results)
        layer_patterns: Patterns for filtering layers
    """

    def __init__(self, layer_patterns: list[str] | None = None):
        """Initialize the tracker."""
        self.layer_patterns = layer_patterns
        self.history: list[tuple[int, dict[str, float], list[GradientAlignmentResult]]] = []

    def record(
        self,
        model: nn.Module,
        step: int,
    ) -> dict[str, float]:
        """Record gradient alignment at current step."""
        results = analyze_model_gradient_alignment(model, self.layer_patterns)
        aggregated = aggregate_gradient_alignment(results)
        self.history.append((step, aggregated, results))
        return aggregated

    def get_metric_history(self, metric_name: str) -> tuple[list[int], list[float]]:
        """Get history of a specific metric."""
        steps = []
        values = []
        for step, metrics, _ in self.history:
            if metric_name in metrics and np.isfinite(metrics[metric_name]):
                steps.append(step)
                values.append(metrics[metric_name])
        return steps, values

    def get_layer_history(self, layer_name: str) -> tuple[list[int], list[float]]:
        """Get cosine similarity history for a specific layer."""
        steps = []
        values = []
        for step, _, results in self.history:
            for r in results:
                if r.layer_name == layer_name:
                    steps.append(step)
                    values.append(r.cosine_similarity)
                    break
        return steps, values
