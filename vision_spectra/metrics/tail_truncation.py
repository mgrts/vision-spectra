"""
Tail truncation analysis for spectral experiments.

This module implements experiments to test the effect of pruning heavy-tailed
singular values on model performance. The hypothesis is that if heavy tails
encode important features, truncating them should hurt accuracy.

Experiment Design:
    1. Load a trained model
    2. For each weight matrix, compute SVD
    3. Zero out singular values below various thresholds
    4. Measure accuracy change

Interpretation:
    - Large accuracy drop → heavy tail carries important information
    - Small accuracy drop → heavy tail is mostly noise
    - This helps understand what heavy-tailed spectra represent

References:
    - Eckart-Young-Mirsky theorem: SVD gives optimal low-rank approximation
    - Martin & Mahoney (2021): Heavy tails as implicit regularization
"""

from __future__ import annotations

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
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Truncate a weight matrix by zeroing small singular values.

    Args:
        weight: 2D weight matrix
        retention_ratio: Fraction of singular values to retain (0, 1]
        min_rank: Minimum number of singular values to keep

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
) -> dict[str, float]:
    """
    Truncate a specific layer in the model (in-place).

    Args:
        model: PyTorch model
        layer_name: Full name of the parameter to truncate
        retention_ratio: Fraction to retain (SVs if use_energy=False, energy if True)
        use_energy: Whether to use energy-based truncation

    Returns:
        Info dictionary with truncation statistics
    """
    for name, param in model.named_parameters():
        if name == layer_name and param.dim() == 2:
            weight = param.detach().cpu().numpy()

            if use_energy:
                truncated, info = truncate_by_energy(weight, retention_ratio)
            else:
                truncated, info = truncate_weight_matrix(weight, retention_ratio)

            with torch.no_grad():
                param.copy_(torch.from_numpy(truncated).to(param.device))

            return info

    return {"error": f"Layer {layer_name} not found or not 2D"}


def truncate_all_attention_layers(
    model: nn.Module,
    retention_ratio: float = 0.9,
    use_energy: bool = False,
) -> dict[str, dict[str, float]]:
    """
    Truncate all attention-related weight matrices in the model.

    Args:
        model: PyTorch model (ViT)
        retention_ratio: Fraction to retain
        use_energy: Whether to use energy-based truncation

    Returns:
        Dictionary mapping layer names to truncation info
    """
    results = {}

    for name, param in model.named_parameters():
        # Target attention weights (QKV, projection)
        if param.dim() == 2 and ("attn" in name or "qkv" in name or "proj" in name):
            info = truncate_model_layer(model, name, retention_ratio, use_energy)
            results[name] = info

    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a data loader."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        if labels.dim() > 1:
            labels = labels.squeeze()

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
    use_energy: bool = True,
) -> list[dict[str, Any]]:
    """
    Run truncation experiment with multiple retention ratios.

    For each ratio, truncates all attention layers and measures accuracy.
    Model is restored after each truncation level.

    Args:
        model: Trained PyTorch model
        val_loader: Validation data loader
        device: Torch device
        retention_ratios: List of retention ratios to test
        use_energy: Whether to use energy-based truncation

    Returns:
        List of result dictionaries, one per retention ratio
    """
    if retention_ratios is None:
        retention_ratios = [1.0, 0.99, 0.95, 0.90, 0.80, 0.70, 0.50, 0.30, 0.10]

    # Save original weights
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Evaluate original accuracy
    original_accuracy = evaluate_model(model, val_loader, device)

    results = []

    for ratio in retention_ratios:
        # Restore original weights
        model.load_state_dict(original_state)

        if ratio < 1.0:
            # Truncate all attention layers
            truncation_info = truncate_all_attention_layers(model, ratio, use_energy)
        else:
            truncation_info = {}

        # Evaluate truncated model
        truncated_accuracy = evaluate_model(model, val_loader, device)

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
    """Generate human-readable interpretation of truncation results."""
    if critical_threshold is None:
        return (
            "Heavy tails appear to be noise: model is robust to significant truncation. "
            "This suggests the heavy-tailed singular values do not encode critical features."
        )
    elif critical_threshold > 0.9:
        return (
            "Heavy tails encode important features: accuracy drops quickly with truncation. "
            "This suggests the network uses the full spectrum for representation."
        )
    elif critical_threshold > 0.5:
        return (
            "Moderate importance of heavy tails: some truncation is tolerated. "
            "This suggests a mix of important and redundant singular values."
        )
    else:
        return (
            "Heavy tails are mostly redundant: severe truncation is tolerated. "
            "Only the top singular values appear to carry important information."
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
