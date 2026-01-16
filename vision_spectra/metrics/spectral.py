"""
Spectral metrics for weight matrix analysis.

This module computes spectral properties of weight matrices that characterize
their structure and complexity. These metrics are useful for understanding
model regularization, generalization, and training dynamics.

Metrics included:
    - Spectral entropy: Measures how uniformly distributed singular values are.
      Higher entropy indicates a "flatter" spectrum with many active dimensions.
    - Stable rank: The effective dimensionality of the matrix, defined as
      ||W||_F^2 / ||W||_2^2. Unlike matrix rank, stable rank is continuous.
    - Alpha exponent: Power-law decay rate of singular values. Fitted as
      σ_i ∝ i^(-α) in log-log space. Higher α means faster decay.
    - Power-law alpha (Hill): Tail index estimated using the Hill estimator,
      which provides a maximum likelihood estimate of the power-law exponent
      for the eigenvalue distribution tail.

All computations use float64 precision and are done on CPU for stability.

Scientific Background:
    Neural network weight matrices often exhibit power-law singular value
    distributions. The exponent of this distribution correlates with model
    capacity and generalization. Well-trained models typically show α values
    between 2 and 6, with higher values indicating more implicit regularization.

References:
    [1] Martin, C. H., & Mahoney, M. W. (2021). "Implicit Self-Regularization
        in Deep Neural Networks: Evidence from Random Matrix Theory and
        Implications for Learning." Journal of Machine Learning Research.
    [2] Martin, C. H., et al. (2021). "Predicting trends in the quality of
        state-of-the-art neural networks without access to training or
        testing data." Nature Communications.
    [3] Roy, R., & Bhattacharyya, C. (2018). "Understanding Neural Networks
        Through Spectral Analysis." ICLR Workshop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy as scipy_entropy


def spectral_entropy(weight_matrix: np.ndarray) -> float:
    """
    Compute spectral entropy of a weight matrix.

    Spectral entropy measures the uniformity of the singular value distribution.
    It is computed as the Shannon entropy of the normalized squared singular
    values (which represent the variance explained by each singular component).

    Mathematical Definition:
        Given singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ, define:
            p_i = σᵢ² / Σⱼ σⱼ²  (probability distribution)
            H = -Σᵢ p_i · log(p_i)  (Shannon entropy in nats)

    Interpretation:
        - Maximum entropy (log n) occurs when all singular values are equal
          (identity-like matrix, uniform spectrum)
        - Minimum entropy (0) occurs for rank-1 matrices (single non-zero σ)
        - Higher entropy → more distributed representation capacity
        - Lower entropy → information concentrated in few dimensions

    Args:
        weight_matrix: 2D numpy array of shape [m, n]

    Returns:
        Spectral entropy in nats (natural logarithm base).
        Returns np.nan if computation fails (non-2D input, SVD failure, etc.)

    Example:
        >>> import numpy as np
        >>> identity = np.eye(10)
        >>> spectral_entropy(identity)  # Returns ~2.30 (log(10))
        >>> rank_one = np.outer(np.ones(10), np.ones(10))
        >>> spectral_entropy(rank_one)  # Returns ~0.0

    References:
        - Shannon, C. E. (1948). "A Mathematical Theory of Communication"
        - Used in neural network analysis to characterize weight spectra
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

    Stable rank provides a continuous measure of the "effective dimensionality"
    of a matrix. Unlike the integer-valued matrix rank, stable rank smoothly
    captures how many singular values are significant.

    Mathematical Definition:
        stable_rank(W) = ||W||_F² / ||W||_2²
                       = Σᵢ σᵢ² / max(σᵢ)²
                       = Σᵢ σᵢ² / σ₁²

        where ||·||_F is the Frobenius norm and ||·||_2 is the spectral norm.

    Properties:
        - Always satisfies: 1 ≤ stable_rank(W) ≤ rank(W)
        - stable_rank = 1: All singular values are zero except one (rank-1 matrix)
        - stable_rank = n: All singular values are equal (e.g., identity matrix)
        - Robust to small perturbations (unlike matrix rank)

    Interpretation for Neural Networks:
        - Lower stable rank → weight matrix has lower effective dimensionality
        - Higher stable rank → weight matrix uses more of its capacity
        - Can indicate compression or redundancy in learned representations

    Args:
        weight_matrix: 2D numpy array of shape [m, n]

    Returns:
        Stable rank value (float >= 1.0 for non-zero matrices).
        Returns np.nan if computation fails.

    Example:
        >>> import numpy as np
        >>> stable_rank(np.eye(10))  # Returns 10.0
        >>> stable_rank(np.ones((10, 10)))  # Returns 1.0 (rank-1)

    References:
        - Rudelson, M., & Vershynin, R. (2007). "Sampling from large matrices"
        - Used as a regularization-aware complexity measure in deep learning
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
    Estimate power-law exponent (alpha) from singular value decay via log-log regression.

    This metric measures how quickly singular values decay as a function of their
    rank index. It fits a power law σᵢ ∝ i^(-α) in log-log space using ordinary
    least squares regression.

    Mathematical Definition:
        Given sorted singular values σ₁ ≥ σ₂ ≥ ... ≥ σₙ, fit:
            log(σᵢ) = -α·log(i) + c

        The exponent α is the negative slope of this log-log fit.

    Fitting Details:
        - By default, fits the "bulk" of the spectrum (indices 10% to 60%)
        - Avoids the largest singular values (often outliers)
        - Avoids the smallest singular values (often noise/numerical)
        - Requires at least 8 singular values to produce a reliable fit

    Interpretation:
        - α ≈ 0: Flat spectrum, all singular values roughly equal (high entropy)
        - α > 0: Decaying spectrum, larger singular values dominate
        - α ≈ 1: Slow decay, many significant components
        - α ≈ 2-4: Moderate decay, typical for trained neural networks
        - α > 4: Rapid decay, low effective rank, strong implicit regularization
        - Well-trained models typically have α between 2 and 6

    Args:
        weight_matrix: 2D numpy array of shape [m, n]
        fit_range: Optional tuple (start, end) specifying which indices to use
            for fitting (0-indexed, inclusive start, exclusive end). If None,
            automatically selects the range [10%, 60%] of singular value indices.

    Returns:
        Estimated alpha exponent (float).
        Returns np.nan if estimation fails (non-2D input, too few singular values,
        SVD failure, or regression failure).

    Example:
        >>> import numpy as np
        >>> # Create matrix with known power-law decay
        >>> n = 100
        >>> U = np.linalg.qr(np.random.randn(n, n))[0]
        >>> V = np.linalg.qr(np.random.randn(n, n))[0]
        >>> s = np.arange(1, n+1, dtype=float) ** (-2.0)  # α = 2.0
        >>> W = U @ np.diag(s) @ V.T
        >>> alpha_exponent(W)  # Returns ~2.0

    Note:
        This is a rank-based fitting approach. For a complementary distributional
        estimate, see power_law_alpha_hill() which uses the Hill estimator.

    References:
        - Martin & Mahoney (2021). "Implicit Self-Regularization in DNNs." JMLR.
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
    Estimate power-law tail exponent using the Hill estimator on eigenvalues.

    The Hill estimator is a maximum likelihood estimator for the tail index
    of a Pareto (power-law) distribution. It estimates α where the tail
    probability follows P(X > x) ∝ x^(-α).

    Mathematical Definition:
        Given the k largest eigenvalues λ₍₁₎ ≥ λ₍₂₎ ≥ ... ≥ λ₍ₖ₎:

            H = (1/k) · Σᵢ₌₁ᵏ log(λ₍ᵢ₎ / λ₍ₖ₎)

            α_Hill = 1 + 1/H

        This is the MLE for the Pareto tail index.

    Implementation Details:
        - Uses squared singular values (eigenvalues of W^T W)
        - By default, uses the top 10% of eigenvalues (at least 5)
        - More robust than log-log regression for estimating tail behavior
        - Requires at least 8 eigenvalues for reliable estimation

    Interpretation:
        - Higher α: Heavier tail, faster drop-off in the eigenvalue density
        - Lower α: Lighter tail, eigenvalues more uniformly distributed
        - α ∈ (2, 6): Common range for well-trained neural networks
        - Note: This measures the DISTRIBUTIONAL tail exponent, which differs
          from the rank-based decay exponent (see alpha_exponent).

    Important Note:
        The Hill estimator assumes eigenvalues follow a Pareto-like distribution.
        For matrices where eigenvalues follow a deterministic rank-based power law
        (λᵢ ∝ i^(-β)), the Hill estimator does NOT directly estimate β.
        Use alpha_exponent() for rank-based decay estimation.

    Args:
        weight_matrix: 2D numpy array of shape [m, n]
        k: Number of top eigenvalues to use for estimation.
            If None, uses ~10% of eigenvalues (minimum 5).
            Larger k gives lower variance but potentially higher bias.

    Returns:
        Estimated tail exponent α (float > 1).
        Returns np.nan if estimation fails.

    Example:
        >>> import numpy as np
        >>> W = np.random.randn(100, 100)
        >>> alpha = power_law_alpha_hill(W)  # Typically returns ~2-4

    References:
        - Hill, B. M. (1975). "A Simple General Approach to Inference About
          the Tail of a Distribution." Annals of Statistics.
        - Martin & Mahoney (2021). Use this for ESD analysis in neural networks.
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

    This is the main entry point for computing a comprehensive set of spectral
    metrics that characterize the structure and complexity of a weight matrix.

    Args:
        weight_matrix: 2D weight matrix (numpy array or PyTorch tensor).
            Will be converted to float64 numpy array for computation.

    Returns:
        Dictionary containing:
        - spectral_entropy: Shannon entropy of normalized squared singular values.
          Higher values indicate more uniform spectrum (high effective dimension).
        - stable_rank: Effective rank ||W||_F²/||W||_2². Values from 1 to min(m,n).
          Higher values indicate more significant singular values.
        - alpha_exponent: Power-law decay rate from log-log regression.
          Higher values indicate faster singular value decay.
        - pl_alpha_hill: Tail exponent from Hill estimator on eigenvalues.
          Estimates the distributional power-law exponent.

    Example:
        >>> import numpy as np
        >>> W = np.random.randn(64, 64)
        >>> metrics = get_spectral_metrics(W)
        >>> print(metrics['stable_rank'])  # Effective dimensionality

    Note:
        All metrics are computed on CPU with float64 precision for numerical
        stability. Invalid or ill-conditioned matrices may result in np.nan
        for some metrics.
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
    Aggregate spectral metrics across multiple weight matrices (e.g., layers).

    Computes mean and standard deviation for each metric across all input
    dictionaries. Useful for summarizing spectral properties across all
    layers of a neural network.

    Args:
        metrics_list: List of dictionaries from get_spectral_metrics(),
            one per weight matrix/layer.

    Returns:
        Dictionary with aggregated statistics:
        - {metric}_mean: Mean value across all layers (ignoring NaN)
        - {metric}_std: Standard deviation across layers (ignoring NaN)

        For example: spectral_entropy_mean, spectral_entropy_std,
        stable_rank_mean, stable_rank_std, etc.

    Example:
        >>> metrics_layer1 = get_spectral_metrics(W1)
        >>> metrics_layer2 = get_spectral_metrics(W2)
        >>> agg = aggregate_spectral_metrics([metrics_layer1, metrics_layer2])
        >>> print(agg['stable_rank_mean'])  # Average stable rank
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


# =============================================================================
# Spectral Distribution Tracking
# =============================================================================


@dataclass
class SpectralDistribution:
    """
    Container for spectral distribution data from a weight matrix.

    Stores singular values and derived quantities for tracking how the
    spectral distribution evolves during training.

    Attributes:
        name: Identifier for the weight matrix (e.g., "blocks.0.attn.q")
        matrix_type: Type of weight matrix (e.g., "q", "k", "v", "mlp")
        singular_values: Array of singular values in descending order
        eigenvalues: Squared singular values (eigenvalues of W^T W)
        normalized_sv: Singular values normalized by the largest
        cumulative_variance: Cumulative variance explained (for PCA-like analysis)
        metrics: Dictionary of scalar spectral metrics
    """

    name: str
    matrix_type: str
    singular_values: np.ndarray
    eigenvalues: np.ndarray
    normalized_sv: np.ndarray
    cumulative_variance: np.ndarray
    metrics: dict[str, float]


def get_spectral_distribution(
    weight_matrix: np.ndarray,
    name: str = "",
    matrix_type: str = "unknown",
) -> SpectralDistribution | None:
    """
    Extract full spectral distribution from a weight matrix.

    This function computes and stores the complete singular value distribution,
    which can be used for visualization and tracking over training epochs.

    Args:
        weight_matrix: 2D numpy array or PyTorch tensor
        name: Identifier for this weight matrix
        matrix_type: Type of the matrix (e.g., "q", "k", "v", "mlp")

    Returns:
        SpectralDistribution object containing:
        - Raw singular values (descending order)
        - Eigenvalues (squared singular values)
        - Normalized singular values (divided by max)
        - Cumulative variance explained
        - Scalar spectral metrics

        Returns None if SVD fails or input is invalid.

    Example:
        >>> W = np.random.randn(64, 64)
        >>> dist = get_spectral_distribution(W, name="layer1", matrix_type="mlp")
        >>> print(dist.singular_values[:5])  # Top 5 singular values
        >>> print(dist.cumulative_variance[:10])  # Variance explained by top 10
    """
    # Convert to numpy if needed
    if hasattr(weight_matrix, "cpu"):
        weight_matrix = weight_matrix.cpu().numpy()
    weight_matrix = np.asarray(weight_matrix, dtype=np.float64)

    if weight_matrix.ndim != 2:
        return None

    try:
        s = svd(weight_matrix, compute_uv=False)
    except Exception:
        return None

    # Filter and sort
    s = s[np.isfinite(s) & (s >= 0)]
    if s.size == 0:
        return None

    s = np.sort(s)[::-1]  # Descending order

    # Compute derived quantities
    eigenvalues = s**2
    s_max = s[0] if s[0] > 0 else 1.0
    normalized_sv = s / s_max

    # Cumulative variance explained
    total_variance = eigenvalues.sum()
    if total_variance > 0:
        cumulative_variance = np.cumsum(eigenvalues) / total_variance
    else:
        cumulative_variance = np.zeros_like(eigenvalues)

    # Compute scalar metrics
    metrics = get_spectral_metrics(weight_matrix)

    return SpectralDistribution(
        name=name,
        matrix_type=matrix_type,
        singular_values=s,
        eigenvalues=eigenvalues,
        normalized_sv=normalized_sv,
        cumulative_variance=cumulative_variance,
        metrics=metrics,
    )


@dataclass
class EpochSpectralSnapshot:
    """
    Snapshot of spectral distributions for all tracked layers at one epoch.

    Attributes:
        epoch: Training epoch number
        distributions: List of SpectralDistribution for each tracked layer
        aggregated_metrics: Aggregated metrics across all layers
        timestamp: When this snapshot was created
    """

    epoch: int
    distributions: list[SpectralDistribution]
    aggregated_metrics: dict[str, float]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime

            self.timestamp = datetime.now().isoformat()


class SpectralTracker:
    """
    Tracks spectral distributions across training epochs.

    This class maintains a history of spectral distributions for selected
    weight matrices, enabling analysis of how the spectrum evolves during
    training.

    Attributes:
        history: List of EpochSpectralSnapshot objects
        layer_patterns: Patterns for selecting which layers to track
        include_qkv: Whether to track Q/K/V attention weights
        include_mlp: Whether to track MLP weights
        include_patch_embed: Whether to track patch embedding

    Example:
        >>> tracker = SpectralTracker(layer_patterns=["blocks.0", "blocks.2"])
        >>> for epoch in range(num_epochs):
        ...     train_one_epoch()
        ...     tracker.record_epoch(model, epoch)
        >>> # Analyze evolution
        >>> history = tracker.get_metric_history("stable_rank_mean")
    """

    def __init__(
        self,
        layer_patterns: list[str] | None = None,
        include_qkv: bool = True,
        include_mlp: bool = False,
        include_patch_embed: bool = True,
        max_singular_values: int = 100,
    ):
        """
        Initialize the spectral tracker.

        Args:
            layer_patterns: List of layer name patterns to track
            include_qkv: Include Q/K/V attention weights
            include_mlp: Include MLP weights
            include_patch_embed: Include patch embedding weights
            max_singular_values: Maximum number of singular values to store
                per layer (to limit memory usage)
        """
        self.layer_patterns = layer_patterns or []
        self.include_qkv = include_qkv
        self.include_mlp = include_mlp
        self.include_patch_embed = include_patch_embed
        self.max_singular_values = max_singular_values
        self.history: list[EpochSpectralSnapshot] = []

    def record_epoch(
        self,
        model: Any,
        epoch: int,
    ) -> EpochSpectralSnapshot:
        """
        Record spectral distributions for the current epoch.

        Args:
            model: PyTorch model to analyze
            epoch: Current epoch number

        Returns:
            EpochSpectralSnapshot containing all distributions for this epoch
        """
        from vision_spectra.metrics.extraction import extract_all_weights

        # Extract weights
        weights = extract_all_weights(
            model,
            layer_patterns=self.layer_patterns,
            include_qkv=self.include_qkv,
            include_mlp=self.include_mlp,
            include_patch_embed=self.include_patch_embed,
        )

        # Compute distributions
        distributions = []
        for w in weights:
            dist = get_spectral_distribution(
                w.weight,
                name=w.name,
                matrix_type=w.matrix_type,
            )
            if dist is not None:
                # Truncate singular values to save memory
                if len(dist.singular_values) > self.max_singular_values:
                    dist = SpectralDistribution(
                        name=dist.name,
                        matrix_type=dist.matrix_type,
                        singular_values=dist.singular_values[: self.max_singular_values],
                        eigenvalues=dist.eigenvalues[: self.max_singular_values],
                        normalized_sv=dist.normalized_sv[: self.max_singular_values],
                        cumulative_variance=dist.cumulative_variance[: self.max_singular_values],
                        metrics=dist.metrics,
                    )
                distributions.append(dist)

        # Aggregate metrics
        all_metrics = [d.metrics for d in distributions]
        aggregated = aggregate_spectral_metrics(all_metrics) if all_metrics else {}

        snapshot = EpochSpectralSnapshot(
            epoch=epoch,
            distributions=distributions,
            aggregated_metrics=aggregated,
        )

        self.history.append(snapshot)
        return snapshot

    def get_metric_history(self, metric_name: str) -> tuple[list[int], list[float]]:
        """
        Get the history of a specific metric across all epochs.

        Args:
            metric_name: Name of the metric (e.g., "stable_rank_mean")

        Returns:
            Tuple of (epochs, values) lists
        """
        epochs = []
        values = []

        for snapshot in self.history:
            if metric_name in snapshot.aggregated_metrics:
                value = snapshot.aggregated_metrics[metric_name]
                if np.isfinite(value):
                    epochs.append(snapshot.epoch)
                    values.append(value)

        return epochs, values

    def get_layer_sv_history(self, layer_name: str) -> tuple[list[int], list[np.ndarray]]:
        """
        Get singular value history for a specific layer.

        Args:
            layer_name: Name of the layer to track

        Returns:
            Tuple of (epochs, singular_values_list)
        """
        epochs = []
        sv_list = []

        for snapshot in self.history:
            for dist in snapshot.distributions:
                if dist.name == layer_name:
                    epochs.append(snapshot.epoch)
                    sv_list.append(dist.singular_values)
                    break

        return epochs, sv_list

    def get_all_layer_names(self) -> list[str]:
        """Get names of all tracked layers."""
        if not self.history:
            return []
        return [d.name for d in self.history[0].distributions]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert tracker state to a dictionary for serialization.

        Returns:
            Dictionary containing all tracking history
        """
        return {
            "layer_patterns": self.layer_patterns,
            "include_qkv": self.include_qkv,
            "include_mlp": self.include_mlp,
            "include_patch_embed": self.include_patch_embed,
            "max_singular_values": self.max_singular_values,
            "history": [
                {
                    "epoch": s.epoch,
                    "timestamp": s.timestamp,
                    "aggregated_metrics": s.aggregated_metrics,
                    "distributions": [
                        {
                            "name": d.name,
                            "matrix_type": d.matrix_type,
                            "singular_values": d.singular_values.tolist(),
                            "metrics": d.metrics,
                        }
                        for d in s.distributions
                    ],
                }
                for s in self.history
            ],
        }

    def save(self, path: Path) -> None:
        """Save tracker state to a JSON file."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> SpectralTracker:
        """Load tracker state from a JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)

        tracker = cls(
            layer_patterns=data.get("layer_patterns", []),
            include_qkv=data.get("include_qkv", True),
            include_mlp=data.get("include_mlp", False),
            include_patch_embed=data.get("include_patch_embed", True),
            max_singular_values=data.get("max_singular_values", 100),
        )

        # Reconstruct history
        for h in data.get("history", []):
            distributions = []
            for d in h.get("distributions", []):
                sv = np.array(d["singular_values"])
                distributions.append(
                    SpectralDistribution(
                        name=d["name"],
                        matrix_type=d["matrix_type"],
                        singular_values=sv,
                        eigenvalues=sv**2,
                        normalized_sv=sv / sv[0] if sv[0] > 0 else sv,
                        cumulative_variance=np.cumsum(sv**2) / sv.sum() ** 2
                        if sv.sum() > 0
                        else np.zeros_like(sv),
                        metrics=d.get("metrics", {}),
                    )
                )

            snapshot = EpochSpectralSnapshot(
                epoch=h["epoch"],
                distributions=distributions,
                aggregated_metrics=h.get("aggregated_metrics", {}),
                timestamp=h.get("timestamp", ""),
            )
            tracker.history.append(snapshot)

        return tracker
