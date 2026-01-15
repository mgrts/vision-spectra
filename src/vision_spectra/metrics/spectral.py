"""
Spectral metrics for weight matrix analysis.

Computes various spectral properties that characterize weight matrix structure:
- Spectral entropy: Information-theoretic measure of singular value spread
- Stable rank: Effective rank of the matrix
- Alpha exponent: Power-law decay rate of singular values

All computations are done on CPU for numerical stability.

References:
    Martin, C. H., & Mahoney, M. W. (2021).
    "Implicit Self-Regularization in Deep Neural Networks." JMLR.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy as scipy_entropy


def spectral_entropy(weight_matrix: np.ndarray) -> float:
    """
    Compute spectral entropy of a weight matrix.

    Spectral entropy measures how uniformly distributed the singular values are.
    Higher entropy indicates more uniform distribution ("democratic" spectrum),
    lower entropy indicates concentration in few singular values.

    Args:
        weight_matrix: 2D weight matrix [m, n]

    Returns:
        Spectral entropy in nats. Returns nan if computation fails.

    Note:
        Entropy is computed on normalized squared singular values (variance explained).
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    # Filter valid singular values
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return np.nan

    # Compute probability distribution from squared SVs
    p = (s**2).astype(np.float64)
    total = p.sum()

    if total <= 0 or not np.isfinite(total):
        return np.nan

    p = p / total

    return float(scipy_entropy(p))


def stable_rank(weight_matrix: np.ndarray) -> float:
    """
    Compute the stable rank of a weight matrix.

    Stable rank = ||W||_F^2 / ||W||_2^2 = sum(s^2) / max(s)^2

    Unlike matrix rank, stable rank is continuous and measures
    the "effective" number of significant singular values.

    Args:
        weight_matrix: 2D weight matrix [m, n]

    Returns:
        Stable rank (always >= 1 for non-zero matrices). Returns nan if fails.
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    s = s[np.isfinite(s) & (s >= 0)]
    if s.size == 0:
        return np.nan

    s_max = s.max()
    if s_max <= 0 or not np.isfinite(s_max):
        return np.nan

    numerator = float(np.sum(s**2))
    denominator = float(s_max**2)

    return numerator / denominator


def alpha_exponent(
    weight_matrix: np.ndarray,
    fit_range: tuple[int, int] | None = None,
) -> float:
    """
    Estimate power-law exponent (alpha) from singular value decay.

    Fits a power law σ_i ∝ i^(-α) in log-log space.
    Larger alpha indicates faster decay (lower effective rank).

    Args:
        weight_matrix: 2D weight matrix
        fit_range: Optional (start, end) indices for fitting

    Returns:
        Estimated alpha exponent. Returns nan if estimation fails.
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    s = s[np.isfinite(s) & (s > 0)]
    s = np.sort(s)[::-1]  # Descending order
    m = s.size

    if m == 0:
        return np.nan

    # Choose fitting window
    if fit_range is None:
        if m < 8:
            return np.nan
        start = max(1, int(0.10 * m))
        end = max(start + 6, int(0.60 * m))
        end = min(end, m)
        if end - start < 2:
            return np.nan
    else:
        start, end = fit_range
        if end > m or end - start < 2:
            return np.nan

    # Fit in log-log space
    ranks = np.arange(1, m + 1, dtype=np.float64)
    log_x = np.log(ranks[start:end])
    log_y = np.log(s[start:end])

    try:
        slope, _ = np.polyfit(log_x, log_y, 1)
        return float(-slope)
    except Exception:
        return np.nan


def power_law_alpha_hill(
    weight_matrix: np.ndarray,
    k: int | None = None,
) -> float:
    """
    Estimate power-law exponent using Hill estimator.

    The Hill estimator is a maximum likelihood estimator for the tail
    index of a power-law distribution.

    Args:
        weight_matrix: 2D weight matrix
        k: Number of order statistics to use (default: ~10% of data)

    Returns:
        Estimated alpha. Returns nan if estimation fails.
    """
    if weight_matrix.ndim != 2:
        return np.nan

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return np.nan

    # Use squared singular values (eigenvalues of W^T W)
    lambdas = (s**2).astype(np.float64)
    lambdas = lambdas[np.isfinite(lambdas) & (lambdas > 0)]
    n = lambdas.size

    if n < 8:
        return np.nan

    if k is None:
        k = max(5, int(0.10 * n))
        k = min(k, max(5, n - 1))

    # Get k largest values
    tail = np.sort(lambdas)[::-1][:k]
    xmin = tail[-1]

    if xmin <= 0 or np.any(tail <= 0):
        return np.nan

    logs = np.log(tail / xmin)
    H = logs.mean()

    if H <= 0 or not np.isfinite(H):
        return np.nan

    return float(1.0 + 1.0 / H)


def get_spectral_metrics(weight_matrix: np.ndarray) -> dict[str, float]:
    """
    Compute all spectral metrics for a weight matrix.

    Args:
        weight_matrix: 2D weight matrix

    Returns:
        Dictionary containing:
        - spectral_entropy: Shannon entropy of normalized singular values
        - stable_rank: Effective rank of the matrix
        - alpha_exponent: Power-law decay rate
        - pl_alpha_hill: Power-law alpha via Hill estimator
    """
    # Ensure we're on CPU and convert to float64 for stability
    if hasattr(weight_matrix, "cpu"):
        weight_matrix = weight_matrix.cpu().numpy()
    weight_matrix = np.asarray(weight_matrix, dtype=np.float64)

    return {
        "spectral_entropy": spectral_entropy(weight_matrix),
        "stable_rank": stable_rank(weight_matrix),
        "alpha_exponent": alpha_exponent(weight_matrix),
        "pl_alpha_hill": power_law_alpha_hill(weight_matrix),
    }


def aggregate_spectral_metrics(
    metrics_list: list[dict[str, float]],
) -> dict[str, float]:
    """
    Aggregate spectral metrics across multiple layers.

    Args:
        metrics_list: List of metric dictionaries from get_spectral_metrics

    Returns:
        Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}

    result = {}

    for key in metrics_list[0]:
        values = [m[key] for m in metrics_list if np.isfinite(m.get(key, np.nan))]

        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
        else:
            result[f"{key}_mean"] = np.nan
            result[f"{key}_std"] = np.nan

    return result
