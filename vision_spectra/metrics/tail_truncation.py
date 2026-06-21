"""
Low-rank (Eckart-Young) truncation analysis for spectral experiments.

This module measures how much of a trained model's weight spectrum is actually
needed for its accuracy. It applies Eckart-Young-Mirsky low-rank truncation —
keeping the top-k LARGEST singular values and zeroing the SMALLEST ones — and
records the accuracy change as a function of how many singular values are kept.

Note:
    This is standard low-rank truncation (it removes the smallest singular
    values / the spectral bulk), NOT ablation of the heavy tail (the largest
    singular values) in the random-matrix-theory sense.

Experiment Design:
    1. Load a trained model
    2. For each weight matrix, compute SVD
    3. Zero out the smallest singular values (keep the top-k largest)
    4. Measure the accuracy change as a function of the retention ratio

Interpretation:
    - Large accuracy drop under mild truncation → the smaller singular values
      carry important information; the model uses a large effective rank.
    - Small accuracy drop even under aggressive truncation → the smaller
      singular values are largely redundant; the model is effectively low-rank.

References:
    - Eckart-Young-Mirsky theorem: SVD gives optimal low-rank approximation
    - Martin & Mahoney (2021): weight spectra and implicit regularization
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd


@dataclass
class TruncationResult:
    """Result of a single truncation experiment.

    Attributes:
        layer_name: Name of the truncated layer
        original_rank: Original matrix rank (number of non-zero SVs)
        truncated_rank: Number of retained singular values
        retention_ratio: Fraction of singular values retained
        energy_retained: Fraction of spectral energy (sum of σ²) retained
        original_accuracy: Model accuracy before truncation
        truncated_accuracy: Model accuracy after truncation
        accuracy_drop: Absolute accuracy drop
        relative_drop: Relative accuracy drop (percentage)
    """

    layer_name: str
    original_rank: int
    truncated_rank: int
    retention_ratio: float
    energy_retained: float
    original_accuracy: float
    truncated_accuracy: float
    accuracy_drop: float
    relative_drop: float


def truncate_weight_matrix(
    weight: np.ndarray,
    retention_ratio: float = 0.9,
    min_rank: int = 1,
    mode: str = "bulk",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Truncate a weight matrix by zeroing singular values.

    Two complementary probes (singular values are returned in descending order):

    - ``mode="bulk"`` (Eckart-Young): retain the ``k`` LARGEST singular values and
      zero the smallest. Robustness to bulk truncation ⇒ the matrix is effectively
      low-rank (the smaller SVs are redundant) — the effective-rank / MDL probe.
    - ``mode="head"``: retain the ``k`` SMALLEST singular values and zero the LARGEST
      ``len(s)-k`` (the heavy-tail outliers). A large accuracy drop here ⇒ the few
      dominant singular directions carry the task signal — the heavy-tail-importance
      / signal-vs-noise probe from the research meeting.

    In BOTH modes ``retention_ratio`` is the fraction of singular values retained.

    Args:
        weight: 2D weight matrix
        retention_ratio: Fraction of singular values to retain (0, 1]
        min_rank: Minimum number of singular values to keep
        mode: "bulk" (keep largest) or "head" (keep smallest / drop the tail)

    Returns:
        Tuple of (truncated_weight, info_dict)
        info_dict contains: original_rank, truncated_rank, energy_retained
    """
    U, s, Vt = svd(weight.astype(np.float64), full_matrices=False)

    original_rank = np.sum(s > 1e-10)
    total_energy = np.sum(s**2)

    # Determine how many singular values to keep
    k = max(min_rank, int(np.ceil(len(s) * retention_ratio)))
    k = min(k, len(s))

    # Truncate
    s_truncated = s.copy()
    if mode == "head":
        # Keep the k SMALLEST singular values; zero the largest (len-k) outliers.
        s_truncated[: len(s) - k] = 0.0
    else:
        # Keep the k LARGEST singular values; zero the smallest (standard low-rank).
        s_truncated[k:] = 0.0

    # Reconstruct
    truncated_weight = U @ np.diag(s_truncated) @ Vt

    # Compute energy retained
    energy_retained = np.sum(s_truncated**2) / total_energy if total_energy > 0 else 1.0

    info = {
        "original_rank": int(original_rank),
        "truncated_rank": k,
        "energy_retained": float(energy_retained),
    }

    return truncated_weight.astype(weight.dtype), info


def truncate_by_energy(
    weight: np.ndarray,
    energy_threshold: float = 0.99,
    min_rank: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Truncate a weight matrix by keeping enough SVs to retain a fraction of energy.

    Args:
        weight: 2D weight matrix
        energy_threshold: Minimum fraction of spectral energy to retain
        min_rank: Minimum number of singular values to keep

    Returns:
        Tuple of (truncated_weight, info_dict)
    """
    U, s, Vt = svd(weight.astype(np.float64), full_matrices=False)

    original_rank = np.sum(s > 1e-10)
    total_energy = np.sum(s**2)

    if total_energy <= 0:
        return weight, {"original_rank": 0, "truncated_rank": 0, "energy_retained": 1.0}

    # Find minimum k such that cumulative energy >= threshold
    cumulative_energy = np.cumsum(s**2) / total_energy
    k = np.searchsorted(cumulative_energy, energy_threshold) + 1
    k = max(min_rank, min(k, len(s)))

    # Truncate
    s_truncated = s.copy()
    s_truncated[k:] = 0.0

    # Reconstruct
    truncated_weight = U @ np.diag(s_truncated) @ Vt

    energy_retained = np.sum(s_truncated**2) / total_energy

    info = {
        "original_rank": int(original_rank),
        "truncated_rank": int(k),
        "energy_retained": float(energy_retained),
    }

    return truncated_weight.astype(weight.dtype), info


def truncate_model_layer(
    model: nn.Module,
    layer_name: str,
    retention_ratio: float = 0.9,
    use_energy: bool = False,
    mode: str = "bulk",
) -> dict[str, float]:
    """
    Truncate a specific layer in the model (in-place).

    Args:
        model: PyTorch model
        layer_name: Full name of the parameter to truncate
        retention_ratio: Fraction to retain (SVs if use_energy=False, energy if True)
        use_energy: Whether to use energy-based truncation
        mode: "bulk" (keep largest SVs) or "head" (drop the heavy-tail outliers)

    Returns:
        Info dictionary with truncation statistics
    """
    for name, param in model.named_parameters():
        if name == layer_name and param.dim() == 2:
            weight = param.detach().cpu().numpy()

            if use_energy:
                truncated, info = truncate_by_energy(weight, retention_ratio)
            else:
                truncated, info = truncate_weight_matrix(weight, retention_ratio, mode=mode)

            with torch.no_grad():
                param.copy_(torch.from_numpy(truncated).to(param.device))

            return info

    return {"error": f"Layer {layer_name} not found or not 2D"}


def truncate_all_attention_layers(
    model: nn.Module,
    retention_ratio: float = 0.9,
    use_energy: bool = False,
    mode: str = "bulk",
    include_mlp: bool = True,
) -> dict[str, dict[str, float]]:
    """
    Truncate the per-block weight matrices of the model.

    By default this covers the SAME matrices the spectral and gradient-alignment
    analyses use — ``attn.qkv``, ``attn.proj`` and (when ``include_mlp``) ``mlp.fc1``,
    ``mlp.fc2`` — so the three curves are cross-readable. Matching is boundary-aware
    (``(?:^|\\.)attn(?:\\.|$)``), which excludes ``patch_embed.proj`` and the classifier
    head. Pass ``include_mlp=False`` to restrict to attention only.

    Args:
        model: PyTorch model (ViT)
        retention_ratio: Fraction to retain
        use_energy: Whether to use energy-based truncation
        mode: "bulk" (keep largest SVs) or "head" (drop the heavy-tail outliers)
        include_mlp: also truncate ``mlp.fc1``/``mlp.fc2`` (parity with extraction)

    Returns:
        Dictionary mapping layer names to truncation info
    """
    patterns = ["attn"] + (["mlp"] if include_mlp else [])
    results = {}

    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        if not any(re.search(rf"(?:^|\.){p}(?:\.|$)", name) for p in patterns):
            continue
        info = truncate_model_layer(model, name, retention_ratio, use_energy, mode=mode)
        results[name] = info

    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    """Evaluate model accuracy on a data loader.

    Args:
        model: model to evaluate.
        data_loader: data loader yielding (images, labels).
        device: torch device.
        max_batches: if set, evaluate on at most this many batches (a fixed prefix
            of the loader) to keep the truncation sweep cheap when it is run for
            every seed/scenario. The loader must be deterministic (shuffle=False),
            as it is for test/val splits here.
    """
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        # squeeze(-1) (not squeeze()) so a singleton batch does not collapse to 0-dim.
        if labels.dim() > 1:
            labels = labels.squeeze(-1)

        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0


def run_truncation_experiment(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    retention_ratios: list[float] | None = None,
    use_energy: bool = False,
    mode: str = "bulk",
    max_batches: int | None = None,
    include_mlp: bool = True,
) -> list[dict[str, Any]]:
    """
    Run truncation experiment with multiple retention ratios.

    For each ratio, truncates all attention layers and measures accuracy.
    Model is restored after each truncation level.

    Args:
        model: Trained PyTorch model
        val_loader: Held-out data loader (test split for an unbiased estimate)
        device: Torch device
        retention_ratios: List of retention ratios to test
        use_energy: Whether to use energy-based truncation (ratio = fraction of SVs
            when False, fraction of spectral energy when True)
        mode: "bulk" (keep the largest SVs; effective-rank probe) or "head" (drop the
            heavy-tail outliers; signal-vs-noise probe)
        max_batches: cap on eval batches per truncation level (keeps the sweep cheap
            when run per seed/scenario); the loader must be deterministic.

    Returns:
        List of result dictionaries, one per retention ratio
    """
    if retention_ratios is None:
        retention_ratios = [1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]

    # Save original weights
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluate original accuracy
    original_accuracy = evaluate_model(model, val_loader, device, max_batches=max_batches)

    results = []

    for ratio in retention_ratios:
        # Restore original weights
        model.load_state_dict(original_state)

        if ratio < 1.0:
            # Truncate the per-block matrices (attn + mlp by default)
            truncation_info = truncate_all_attention_layers(
                model, ratio, use_energy, mode=mode, include_mlp=include_mlp
            )
        else:
            truncation_info = {}

        # Evaluate truncated model
        truncated_accuracy = evaluate_model(model, val_loader, device, max_batches=max_batches)

        # Compute aggregate statistics
        total_original_rank = sum(
            info.get("original_rank", 0) for info in truncation_info.values()
        )
        total_truncated_rank = sum(
            info.get("truncated_rank", 0) for info in truncation_info.values()
        )
        avg_energy_retained = (
            np.mean([info.get("energy_retained", 1.0) for info in truncation_info.values()])
            if truncation_info
            else 1.0
        )

        results.append(
            {
                "retention_ratio": ratio,
                "use_energy": use_energy,
                "mode": mode,
                "original_accuracy": original_accuracy,
                "truncated_accuracy": truncated_accuracy,
                "accuracy_drop": original_accuracy - truncated_accuracy,
                "relative_drop": (original_accuracy - truncated_accuracy) / original_accuracy * 100
                if original_accuracy > 0
                else 0.0,
                "total_original_rank": total_original_rank,
                "total_truncated_rank": total_truncated_rank,
                "avg_energy_retained": avg_energy_retained,
                "num_layers_truncated": len(truncation_info),
                "per_layer_info": truncation_info,
            }
        )

    # Restore original weights
    model.load_state_dict(original_state)

    return results


def analyze_truncation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyze truncation experiment results.

    Identifies the "critical threshold" where accuracy starts dropping significantly.

    Args:
        results: List of result dictionaries from run_truncation_experiment

    Returns:
        Analysis dictionary with insights
    """
    # Sort by retention ratio descending
    sorted_results = sorted(results, key=lambda x: x["retention_ratio"], reverse=True)

    # Find critical threshold (first ratio with >5% accuracy drop)
    critical_threshold = None
    for r in sorted_results:
        if r["accuracy_drop"] > 5.0:
            critical_threshold = r["retention_ratio"]
            break

    # Find 1% accuracy drop threshold
    one_percent_threshold = None
    for r in sorted_results:
        if r["accuracy_drop"] > 1.0:
            one_percent_threshold = r["retention_ratio"]
            break

    # Compute sensitivity (accuracy drop per % energy removed)
    sensitivities = []
    for r in sorted_results:
        if r["retention_ratio"] < 1.0:
            energy_removed = 1.0 - r["avg_energy_retained"]
            if energy_removed > 0.01:
                sensitivity = r["accuracy_drop"] / (energy_removed * 100)
                sensitivities.append(sensitivity)

    return {
        "critical_threshold": critical_threshold,
        "one_percent_threshold": one_percent_threshold,
        "avg_sensitivity": np.mean(sensitivities) if sensitivities else 0.0,
        "max_accuracy_drop": max(r["accuracy_drop"] for r in results),
        "interpretation": _interpret_results(critical_threshold, sensitivities),
    }


def _interpret_results(critical_threshold: float | None, sensitivities: list[float]) -> str:
    """Generate a human-readable interpretation of low-rank truncation results."""
    if critical_threshold is None:
        return (
            "Smaller singular values appear redundant: the model is robust to "
            "aggressive low-rank truncation, so it is effectively low-rank and "
            "relies mainly on its top singular components."
        )
    elif critical_threshold > 0.9:
        return (
            "The full spectrum matters: accuracy drops as soon as a few of the "
            "smaller singular values are removed, so the network uses a large "
            "effective rank for its representation."
        )
    elif critical_threshold > 0.5:
        return (
            "Moderate spectral redundancy: some low-rank truncation is tolerated, "
            "suggesting a mix of important and redundant singular values."
        )
    else:
        return (
            "Most singular values are redundant: severe low-rank truncation is "
            "tolerated, and only the top singular values carry important information."
        )


def save_truncation_report(
    results: list[dict[str, Any]],
    analysis: dict[str, Any],
    output_path: Path,
) -> None:
    """Save truncation experiment report to file."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "summary": {
            "critical_threshold": analysis["critical_threshold"],
            "one_percent_threshold": analysis["one_percent_threshold"],
            "avg_sensitivity": analysis["avg_sensitivity"],
            "interpretation": analysis["interpretation"],
        },
        "results": [
            {
                "retention_ratio": r["retention_ratio"],
                "original_accuracy": r["original_accuracy"],
                "truncated_accuracy": r["truncated_accuracy"],
                "accuracy_drop": r["accuracy_drop"],
                "avg_energy_retained": r["avg_energy_retained"],
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
