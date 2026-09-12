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


# =============================================================================
# Head-drop probe (absolute counts) — "do the top singular directions carry the signal?"
# =============================================================================
#
# The retention-ratio ``head`` sweep above removes the top (1-r)·n singular values per
# matrix; its first grid point (retention 0.9 ⇒ drop 10 % of the SVs, i.e. 38 directions
# at width 384) already collapsed every model in the June-2026 study to chance. The
# informative regime is the first FEW singular values, so this probe drops an ABSOLUTE
# number ``n_drop`` of the largest singular values per matrix, optionally restricted to a
# matrix group, and (unlike the ratio sweeps) splits the fused timm ``attn.qkv`` weight
# into Q/K/V blocks so "q"/"k"/"v" groups mean the same matrices as the spectral metrics.

HEAD_DROP_GROUPS: dict[str, tuple[str, ...]] = {
    "all": ("q", "k", "v", "proj", "fc1", "fc2"),
    "qkv": ("q", "k", "v"),
    "attn": ("q", "k", "v", "proj"),
    "proj": ("proj",),
    "mlp": ("fc1", "fc2"),
    "fc1": ("fc1",),
    "fc2": ("fc2",),
}


def drop_top_singular_values(
    weight: np.ndarray, n_drop: int, min_keep: int = 1
) -> tuple[np.ndarray, dict[str, float]]:
    """Zero the ``n_drop`` LARGEST singular values of ``weight`` (keep at least ``min_keep``).

    Returns (new_weight, info) with ``info = {original_rank, n_dropped, energy_retained,
    top_sv_before, top_sv_after}``.
    """
    U, s, Vt = svd(weight.astype(np.float64), full_matrices=False)
    total_energy = float(np.sum(s**2))
    n_drop = int(max(0, min(n_drop, len(s) - min_keep)))
    s_new = s.copy()
    s_new[:n_drop] = 0.0
    new_w = (U * s_new) @ Vt
    info = {
        "original_rank": int(np.sum(s > 1e-10)),
        "n_dropped": n_drop,
        "energy_retained": float(np.sum(s_new**2) / total_energy) if total_energy > 0 else 1.0,
        "top_sv_before": float(s[0]) if len(s) else 0.0,
        "top_sv_after": float(s_new[n_drop]) if n_drop < len(s_new) else 0.0,
    }
    return new_w.astype(weight.dtype), info


def _matrix_type_of_param(name: str) -> str | None:
    """Type of a per-block 2-D parameter by name (None for patch_embed / head / other)."""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if re.search(r"(?:^|\.)attn\.qkv$", base):
        return "qkv"  # fused; split by the caller
    if re.search(r"(?:^|\.)attn\.proj$", base):
        return "proj"
    if re.search(r"(?:^|\.)mlp\.fc1$", base):
        return "fc1"
    if re.search(r"(?:^|\.)mlp\.fc2$", base):
        return "fc2"
    return None


def iter_target_matrices(model: nn.Module, group: str = "all"):
    """Yield ``(name, param, row_slice, matrix_type)`` for every matrix in ``group``.

    The fused ``attn.qkv`` (3d × d) yields three entries (rows [0:d], [d:2d], [2d:3d])
    typed q/k/v, so groups address the same objects as ``metrics/extraction.py``.
    """
    wanted = HEAD_DROP_GROUPS[group]
    for name, param in model.named_parameters():
        if param.dim() != 2:
            continue
        mtype = _matrix_type_of_param(name)
        if mtype is None:
            continue
        if mtype == "qkv":
            if param.shape[0] != 3 * param.shape[1]:
                continue
            d = param.shape[1]
            for i, sub in enumerate(("q", "k", "v")):
                if sub in wanted:
                    yield name, param, slice(i * d, (i + 1) * d), sub
        elif mtype in wanted:
            yield name, param, slice(None), mtype


def drop_head_all_layers(
    model: nn.Module, n_drop: int, group: str = "all"
) -> dict[str, dict[str, float]]:
    """Drop the top ``n_drop`` singular values from every matrix in ``group`` (in place)."""
    infos: dict[str, dict[str, float]] = {}
    for name, param, rows, mtype in iter_target_matrices(model, group):
        weight = param.detach()[rows].cpu().numpy()
        new_w, info = drop_top_singular_values(weight, n_drop)
        with torch.no_grad():
            param[rows] = torch.from_numpy(new_w).to(param.device, param.dtype)
        key = name if rows == slice(None) else f"{name}[{mtype}]"
        infos[key] = info
    return infos


def run_head_drop_experiment(
    model: nn.Module,
    data_loader: Any,
    device: torch.device,
    n_drops: tuple[int, ...] = (1, 2, 3, 5, 10),
    groups: dict[str, tuple[int, ...]] | None = None,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    """Accuracy after dropping the top-``n`` singular values per matrix, for several ``n``
    and matrix groups. Weights are restored after every setting.

    Args:
        n_drops: counts evaluated for the ``"all"`` group.
        groups: extra ``{group: counts}`` to evaluate (default: q/k/v, proj and MLP at
            n ∈ {1, 3}); keys must be in :data:`HEAD_DROP_GROUPS`.
        max_batches: cap on eval batches per setting (deterministic loader required).

    Returns one dict per (group, n_drop) with ``original_accuracy``, ``truncated_accuracy``,
    ``accuracy_drop``, ``avg_energy_retained`` and ``num_matrices``.
    """
    if groups is None:
        groups = {"qkv": (1, 3), "proj": (1, 3), "mlp": (1, 3)}
    plan: list[tuple[str, int]] = [("all", n) for n in n_drops]
    plan += [(g, n) for g, ns in groups.items() for n in ns]

    original_state = {k: v.clone() for k, v in model.state_dict().items()}
    original_accuracy = evaluate_model(model, data_loader, device, max_batches=max_batches)
    results: list[dict[str, Any]] = []
    for group, n in plan:
        model.load_state_dict(original_state)
        infos = drop_head_all_layers(model, n, group)
        acc = evaluate_model(model, data_loader, device, max_batches=max_batches)
        results.append(
            {
                "group": group,
                "n_drop": int(n),
                "original_accuracy": original_accuracy,
                "truncated_accuracy": acc,
                "accuracy_drop": original_accuracy - acc,
                "avg_energy_retained": float(
                    np.mean([i["energy_retained"] for i in infos.values()]) if infos else 1.0
                ),
                "num_matrices": len(infos),
            }
        )
    model.load_state_dict(original_state)
    return results
