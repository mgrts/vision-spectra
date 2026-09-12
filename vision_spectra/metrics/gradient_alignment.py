"""
Gradient alignment metrics for spectral analysis.

This module implements the gradient-alignment analysis discussed in the research
meetings: compare the actual training gradient with the direction that would reduce
the rank of each weight matrix, and track that alignment throughout training.

Two probes are computed from ONE thin SVD ``W = U diag(σ) Vᵀ`` (``r = min(m, n)``):

1. **Legacy full-basis cosine** ``cos(∇L, U Vᵀ)`` (``cosine_similarity``). ``U Vᵀ`` is the
   nuclear-norm subgradient, i.e. the direction that shrinks *every* singular value at
   once. Because it sums all ``r`` singular directions its norm is ``√r``, so even a
   gradient perfectly concentrated on one singular direction yields ``|cos| ≤ 1/√r``
   (≈ 0.07 at r = 192). In the June-2026 study this probe was ≈ 0 for every cell — a
   null largely by construction. It is kept under the MLflow key ``alignment/cos_sim_mean``,
   BUT its basis changed on 2026-09-12: it is now averaged over the split Q/K/V blocks (6
   matrices per block, three SVDs) instead of the fused ``attn.qkv`` (4 matrices per block),
   so values are not numerically comparable with stores written before that date.

2. **Subspace-resolved probe** (the informative one). Rotate the gradient into the
   singular basis, ``G̃ = Uᵀ ∇L V``. Its diagonal gives the first-order motion of each
   singular value under the SGD step ``ΔW = −η ∇L``: ``Δσᵢ = −η G̃ᵢᵢ``. Split the
   spectrum into a *head* (top-``k`` singular directions) and a *tail* (the rest) and
   report

   - ``cos_head = cos(−∇L, U_k V_kᵀ)``  > 0 ⇔ the update GROWS the top-k singular values;
   - ``cos_tail = cos(∇L, U_t V_tᵀ)``   > 0 ⇔ the update SHRINKS the tail (rank-reducing);
   - ``head_energy_enrichment`` = ‖U_kᵀ ∇L V_k‖² / ‖∇L‖² divided by its isotropic
     baseline ``k² / (m·n)`` (1 ⇔ the gradient has no preference for the head subspace);
   - ``tail_energy_enrichment`` analogously with baseline ``(r−k)² / (m·n)``;
   - ``frac_tail_shrinking`` = fraction of tail singular values the step shrinks.

   Both cosines have full ``[−1, 1]`` range for a low-rank gradient, so "training is
   locally sharpening the spectrum" (``cos_head > 0`` and ``cos_tail > 0``) is detectable.

The fused timm ``attn.qkv`` weight is split into its Q, K and V blocks so the per-matrix
results line up with the spectral extraction (``blocks.i.attn.qkv.{q,k,v}``).

Key Hypothesis:
    Networks learn by implicitly minimizing the rank of weight matrices. Higher
    alignment with the rank-minimizing direction → stronger implicit regularization.

References:
    - Arora et al. (2019). "Implicit Regularization in Deep Matrix Factorization."
    - Gunasekar et al. (2017). "Implicit Regularization in Matrix Factorization."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch.nn as nn
from scipy.linalg import svd

DEFAULT_HEAD_FRACTION = 0.1  # head = top 10 % of singular directions (≈ the Hill k; Hill
# uses max(5, int(0.1·n)) whereas head_k_for_rank uses round(0.1·r) clamped to [1, r−1])

# Per-matrix-type aggregation targets (same vocabulary as metrics/extraction.py).
MATRIX_TYPES = ("q", "k", "v", "proj", "fc1", "fc2")

_SUBSPACE_KEYS = (
    "cos_head",
    "cos_tail",
    "head_energy",
    "tail_energy",
    "head_energy_enrichment",
    "tail_energy_enrichment",
    "frac_tail_shrinking",
)


@dataclass
class GradientAlignmentResult:
    """Result of gradient alignment analysis for a single weight matrix.

    Attributes:
        layer_name: Name of the layer/weight matrix
        cosine_similarity: Legacy cos(∇L, U Vᵀ) over ALL singular directions
        training_grad_norm: L2 norm of the training gradient
        rank_reducing_grad_norm: L2 norm of the rank-reducing gradient (= √r)
        angle_degrees: Angle between gradient and U Vᵀ in degrees
        is_aligned: Whether cosine_similarity > 0
        matrix_type: one of MATRIX_TYPES (or "unknown")
        k: number of head singular directions used by the subspace probe
        rank: number of singular directions r = min(m, n)
        cos_head: cos(−∇L, U_k V_kᵀ); > 0 ⇔ update grows the top-k singular values
        cos_tail: cos(∇L, U_t V_tᵀ); > 0 ⇔ update shrinks the tail (rank-reducing)
        head_energy / tail_energy: fraction of ‖∇L‖² inside the head / tail subspace
        head_energy_enrichment / tail_energy_enrichment: the above over the isotropic
            baseline (k²/(mn) and (r−k)²/(mn)); 1 ⇔ no preference
        frac_tail_shrinking: fraction of tail singular values the SGD step shrinks
    """

    layer_name: str
    cosine_similarity: float
    training_grad_norm: float
    rank_reducing_grad_norm: float
    angle_degrees: float
    is_aligned: bool
    matrix_type: str = "unknown"
    k: int = 0
    rank: int = 0
    cos_head: float = np.nan
    cos_tail: float = np.nan
    head_energy: float = np.nan
    tail_energy: float = np.nan
    head_energy_enrichment: float = np.nan
    tail_energy_enrichment: float = np.nan
    frac_tail_shrinking: float = np.nan
    extras: dict[str, float] = field(default_factory=dict)


def compute_rank_reducing_gradient(weight: np.ndarray, rank_target: int = 1) -> np.ndarray:
    """
    Compute the gradient direction that reduces matrix rank.

    For a matrix W = U @ diag(σ) @ V^T, the nuclear-norm gradient gives the
    (full-basis) rank-reducing direction: ∂||W||_* / ∂W = U @ V^T.

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


def head_k_for_rank(
    rank: int, head_fraction: float = DEFAULT_HEAD_FRACTION, min_k: int = 1
) -> int:
    """Number of head singular directions: round(head_fraction·r), clamped to [min_k, r−1]."""
    if rank < 2:
        return 0
    k = int(round(head_fraction * rank))
    return max(min_k, min(k, rank - 1))


def subspace_alignment(
    training_grad: np.ndarray,
    weight: np.ndarray,
    head_fraction: float = DEFAULT_HEAD_FRACTION,
    min_k: int = 1,
) -> dict[str, float]:
    """Subspace-resolved gradient/spectrum alignment for one weight matrix.

    Returns a dict with the keys documented in the module docstring plus ``cos_full``
    (the legacy cos(∇L, U Vᵀ)), ``grad_norm``, ``k`` and ``rank``. All values are NaN
    when the gradient is non-finite / ~0 or the matrix has fewer than 2 singular values.
    """
    nan = {key: np.nan for key in (*_SUBSPACE_KEYS, "cos_full", "grad_norm")}
    nan.update({"k": 0, "rank": 0})
    if training_grad.ndim != 2 or weight.ndim != 2 or training_grad.shape != weight.shape:
        return nan

    G = training_grad.astype(np.float64)
    gnorm = float(np.linalg.norm(G))
    if not np.isfinite(gnorm) or gnorm < 1e-10:
        return nan

    try:
        U, s, Vt = svd(weight.astype(np.float64), full_matrices=False)
    except Exception:
        return nan

    m, n = weight.shape
    r = int(s.size)
    k = head_k_for_rank(r, head_fraction, min_k)
    if k == 0:
        return nan

    # Gradient in the singular basis of W; its diagonal is the first-order motion of
    # the singular values under ΔW = −η∇L:  Δσᵢ = −η·G̃ᵢᵢ.
    G_tilde = U.T @ G @ Vt.T  # r × r
    d = np.diag(G_tilde)
    head, tail = d[:k], d[k:]
    n_tail = r - k

    cos_full = float(d.sum() / (gnorm * np.sqrt(r)))
    cos_head = float(-head.sum() / (gnorm * np.sqrt(k)))
    cos_tail = float(tail.sum() / (gnorm * np.sqrt(n_tail)))

    g2 = gnorm**2
    head_energy = float(np.sum(G_tilde[:k, :k] ** 2) / g2)
    tail_energy = float(np.sum(G_tilde[k:, k:] ** 2) / g2)
    base_head = (k * k) / (m * n)
    base_tail = (n_tail * n_tail) / (m * n)

    return {
        "cos_full": float(np.clip(cos_full, -1.0, 1.0)),
        "cos_head": float(np.clip(cos_head, -1.0, 1.0)),
        "cos_tail": float(np.clip(cos_tail, -1.0, 1.0)),
        "head_energy": head_energy,
        "tail_energy": tail_energy,
        "head_energy_enrichment": head_energy / base_head,
        "tail_energy_enrichment": tail_energy / base_tail,
        "frac_tail_shrinking": float(np.mean(tail > 0)),
        "grad_norm": gnorm,
        "k": k,
        "rank": r,
    }


def compute_gradient_alignment(
    training_grad: np.ndarray,
    weight: np.ndarray,
    head_fraction: float = DEFAULT_HEAD_FRACTION,
) -> GradientAlignmentResult:
    """
    Compute alignment between the training gradient and the rank-reducing directions.

    Args:
        training_grad: Gradient from training loss (same shape as weight)
        weight: Current weight matrix
        head_fraction: fraction of singular directions treated as the head (top-k)

    Returns:
        GradientAlignmentResult with the legacy full-basis cosine AND the
        subspace-resolved probe (both from one SVD).
    """
    sub = subspace_alignment(training_grad, weight, head_fraction=head_fraction)
    train_norm = float(np.linalg.norm(training_grad.astype(np.float64)))
    rank_norm = float(np.sqrt(sub["rank"])) if sub["rank"] else 0.0

    # Non-finite / near-zero gradients (AMP overflow, divergence, dead layer) are "no
    # measurement": NaN so the layer is excluded from aggregation rather than
    # masquerading as a real 0.0/90deg result.
    if not np.isfinite(sub["cos_full"]):
        return GradientAlignmentResult(
            layer_name="",
            cosine_similarity=np.nan,
            training_grad_norm=train_norm,
            rank_reducing_grad_norm=rank_norm,
            angle_degrees=np.nan,
            is_aligned=False,
        )

    cos_sim = sub["cos_full"]
    return GradientAlignmentResult(
        layer_name="",
        cosine_similarity=cos_sim,
        training_grad_norm=train_norm,
        rank_reducing_grad_norm=rank_norm,
        angle_degrees=float(np.degrees(np.arccos(cos_sim))),
        is_aligned=cos_sim > 0,
        k=int(sub["k"]),
        rank=int(sub["rank"]),
        cos_head=sub["cos_head"],
        cos_tail=sub["cos_tail"],
        head_energy=sub["head_energy"],
        tail_energy=sub["tail_energy"],
        head_energy_enrichment=sub["head_energy_enrichment"],
        tail_energy_enrichment=sub["tail_energy_enrichment"],
        frac_tail_shrinking=sub["frac_tail_shrinking"],
    )


def matrix_type_from_name(name: str) -> str:
    """Map an extraction-style matrix name to its type (q/k/v/proj/fc1/fc2/unknown)."""
    for suffix, mtype in (
        (".qkv.q", "q"),
        (".qkv.k", "k"),
        (".qkv.v", "v"),
        (".attn.proj", "proj"),
        (".mlp.fc1", "fc1"),
        (".mlp.fc2", "fc2"),
    ):
        if name.endswith(suffix):
            return mtype
    return "unknown"


def _iter_weight_grads(model: nn.Module, layer_patterns: list[str] | None):
    """Yield (extraction-style name, weight, grad) for every 2-D parameter with a grad.

    The fused timm ``attn.qkv.weight`` (3d × d) is split into its Q/K/V row blocks so the
    per-matrix results match ``metrics/extraction.py`` (``blocks.i.attn.qkv.{q,k,v}``).
    """
    for name, param in model.named_parameters():
        # Boundary-aware match so "blocks.2" does not also select blocks 20-29.
        if layer_patterns and not any(
            re.search(rf"(?:^|\.){re.escape(pat)}(?:\.|$)", name) for pat in layer_patterns
        ):
            continue
        if param.dim() != 2 or param.grad is None:
            continue

        weight = param.detach().cpu().numpy()
        grad = param.grad.detach().cpu().numpy()
        base = name[: -len(".weight")] if name.endswith(".weight") else name

        if base.endswith("attn.qkv") and weight.shape[0] == 3 * weight.shape[1]:
            d = weight.shape[1]
            for i, sub in enumerate(("q", "k", "v")):
                sl = slice(i * d, (i + 1) * d)
                yield f"{base}.{sub}", weight[sl], grad[sl]
        else:
            yield base, weight, grad


def analyze_model_gradient_alignment(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
    head_fraction: float = DEFAULT_HEAD_FRACTION,
) -> list[GradientAlignmentResult]:
    """
    Analyze gradient alignment for all weight matrices in a model.

    This should be called after loss.backward() but before optimizer.step()
    to capture the current training gradients.

    Args:
        model: PyTorch model with gradients computed (.grad attributes populated)
        layer_patterns: Optional patterns to filter layers
        head_fraction: fraction of singular directions treated as the head

    Returns:
        List of GradientAlignmentResult for each analyzed weight matrix
    """
    results = []
    for name, weight, grad in _iter_weight_grads(model, layer_patterns):
        result = compute_gradient_alignment(grad, weight, head_fraction=head_fraction)
        result.layer_name = name
        result.matrix_type = matrix_type_from_name(name)
        results.append(result)
    return results


def _nan_aggregate() -> dict[str, float]:
    out = {
        "cos_sim_mean": np.nan,
        "cos_sim_std": np.nan,
        "cos_sim_min": np.nan,
        "cos_sim_max": np.nan,
        "fraction_aligned": np.nan,
        "angle_mean": np.nan,
    }
    for key in _SUBSPACE_KEYS:
        out[f"{key}_mean"] = np.nan
    return out


def aggregate_gradient_alignment(
    results: list[GradientAlignmentResult],
) -> dict[str, float]:
    """
    Aggregate gradient alignment results across layers.

    Returns the legacy keys (``cos_sim_*``, ``fraction_aligned``, ``angle_mean``), the
    subspace probe means (``cos_head_mean``, ``cos_tail_mean``,
    ``head_energy_enrichment_mean``, ``tail_energy_enrichment_mean``,
    ``frac_tail_shrinking_mean``, ``head_energy_mean``, ``tail_energy_mean``) and, for
    every matrix type present, ``{type}_cos_head``, ``{type}_cos_tail`` and
    ``{type}_head_energy_enrichment`` (means over that type's matrices).
    """
    if not results:
        return _nan_aggregate()

    # Exclude layers with no valid measurement (NaN cosine) so a single
    # bad/zero-gradient layer cannot poison the aggregate statistics.
    valid = [r for r in results if np.isfinite(r.cosine_similarity)]
    if not valid:
        return _nan_aggregate()

    cos_sims = [r.cosine_similarity for r in valid]
    angles = [r.angle_degrees for r in valid if np.isfinite(r.angle_degrees)]
    aligned_count = sum(1 for r in valid if r.is_aligned)

    out: dict[str, float] = {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
        "cos_sim_min": float(np.min(cos_sims)),
        "cos_sim_max": float(np.max(cos_sims)),
        "fraction_aligned": float(aligned_count / len(valid)),
        "angle_mean": float(np.mean(angles)) if angles else np.nan,
    }

    def _mean(rs: list[GradientAlignmentResult], key: str) -> float:
        vals = [getattr(r, key) for r in rs if np.isfinite(getattr(r, key))]
        return float(np.mean(vals)) if vals else np.nan

    for key in _SUBSPACE_KEYS:
        out[f"{key}_mean"] = _mean(valid, key)
    ks = [r.k for r in valid if r.k > 0]
    out["k_mean"] = float(np.mean(ks)) if ks else np.nan

    for mtype in MATRIX_TYPES:
        group = [r for r in valid if r.matrix_type == mtype]
        if not group:
            continue
        out[f"{mtype}_cos_head"] = _mean(group, "cos_head")
        out[f"{mtype}_cos_tail"] = _mean(group, "cos_tail")
        out[f"{mtype}_head_energy_enrichment"] = _mean(group, "head_energy_enrichment")

    return out


class GradientAlignmentTracker:
    """
    Tracks gradient alignment throughout training.

    Attributes:
        history: List of (step, aggregated_metrics, per_layer_results)
        layer_patterns: Patterns for filtering layers
    """

    def __init__(
        self,
        layer_patterns: list[str] | None = None,
        head_fraction: float = DEFAULT_HEAD_FRACTION,
    ):
        """Initialize the tracker."""
        self.layer_patterns = layer_patterns
        self.head_fraction = head_fraction
        self.history: list[tuple[int, dict[str, float], list[GradientAlignmentResult]]] = []

    def record(
        self,
        model: nn.Module,
        step: int,
    ) -> dict[str, float]:
        """Record gradient alignment at current step."""
        results = analyze_model_gradient_alignment(
            model, self.layer_patterns, head_fraction=self.head_fraction
        )
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

    def get_layer_history(
        self, layer_name: str, metric: str = "cosine_similarity"
    ) -> tuple[list[int], list[float]]:
        """Get the history of one per-layer metric (default: legacy cosine)."""
        steps = []
        values = []
        for step, _, results in self.history:
            for r in results:
                if r.layer_name == layer_name:
                    steps.append(step)
                    values.append(getattr(r, metric))
                    break
        return steps, values
