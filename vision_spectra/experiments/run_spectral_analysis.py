#!/usr/bin/env python3
"""
Spectral Analysis Experiments: Three-Scenario Framework.

This module implements the experimental framework from the research meeting notes:
- Scenario A: Expressive network + Simple data → No heavy tails expected
- Scenario B: Expressive network + Complex data → Heavy tails emerge
- Scenario C: Reduced expressivity + Complex data → Heavy tails suppressed

The experiments systematically test the hypothesis that heavy-tailed weight spectra
emerge only when: (1) network has sufficient capacity, AND (2) data is complex enough
to require internal compression.

Usage:
    # Run all three scenarios
    poetry run python -m vision_spectra.experiments.run_spectral_analysis run-all

    # Run individual scenarios
    poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-a
    poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-b
    poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-c

    # Compare scenarios
    poetry run python -m vision_spectra.experiments.run_spectral_analysis compare
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from vision_spectra.data import get_dataset
from vision_spectra.metrics.extraction import (
    extract_attention_weights,
    extract_mlp_weights,
    extract_qkv_weights,
)
from vision_spectra.metrics.gradient_alignment import (
    DEFAULT_HEAD_FRACTION,
    aggregate_gradient_alignment,
    analyze_model_gradient_alignment,
)
from vision_spectra.metrics.spectral import (
    aggregate_spectral_metrics,
    get_spectral_metrics,
)
from vision_spectra.metrics.tail_truncation import (
    analyze_truncation_results,
    run_head_drop_experiment,
    run_truncation_experiment,
)
from vision_spectra.settings import (
    DATA_DIR,
    MLRUNS_DIR,
    DatasetConfig,
    DatasetName,
    OptimizerConfig,
    OptimizerName,
    SchedulerName,
    set_seed,
)
from vision_spectra.training.base import build_optimizer, build_scheduler, warmup_factor

# =============================================================================
# CLI App
# =============================================================================

app = typer.Typer(
    name="spectral-analysis",
    help="Three-scenario spectral analysis experiments.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()


# =============================================================================
# Enums and Configuration
# =============================================================================


class ScenarioType(str, Enum):
    """Experimental scenario types."""

    A_EXPRESSIVE_SIMPLE = "A"  # Expressive network + Simple data
    B_EXPRESSIVE_COMPLEX = "B"  # Expressive network + Complex data
    C_REDUCED_COMPLEX = "C"  # Reduced expressivity + Complex data
    D_REDUCED_SIMPLE = "D"  # Reduced expressivity + Simple data
    E_TINY_SIMPLE = "E"  # Tiny network + Simple data (minimal capacity)
    F_TINY_COMPLEX = "F"  # Tiny network + Complex data (minimal capacity, high complexity)


class DeviceChoice(str, Enum):
    """Device options for training."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class ScenarioConfig:
    """Configuration for a single experimental scenario.

    ``scenario`` is either a canonical :class:`ScenarioType` (the A-F grid, whose
    MLflow experiment name stays ``spectral_scenario_{A-F}`` for the figures contract)
    or a plain string tag for capacity-sweep cells (e.g. ``"w096_path"`` →
    ``spectral_w096_path``). Use :attr:`name` / :attr:`scenario_label` rather than
    ``scenario.value`` so both forms work.
    """

    scenario: ScenarioType | str
    model_name: str
    embed_dim: int
    depth: int
    dataset_name: str
    num_samples: int | None  # None = use full dataset
    num_classes: int
    epochs: int
    batch_size: int
    learning_rate: float
    seeds: list[int]
    log_epochs: list[int]  # Epochs at which to log spectral metrics
    description: str
    # Optional analyses (meeting priorities); on by default so one study run yields them.
    run_alignment: bool = True  # cosine(∇L, U Vᵀ) gradient-alignment to the rank flow
    run_truncation: bool = True  # post-train Eckart-Young bulk/head truncation sweep
    # --- follow-up study knobs (Sept 2026; see EXPERIMENT_PLAN_V2 §9) ---
    weight_decay: float = 0.05  # AdamW decoupled weight decay (0.0 → wd-ablation cells)
    num_eval_samples: int = 200  # synthetic val/test size (2000 for the long control)
    train_subsample: int | None = None  # MedMNIST: keep N TRAIN images; val/test full
    save_checkpoint: bool = False  # log final weights as MLflow artifact model/final.pt
    log_histograms: bool = False  # per-matrix histogram PNGs (regenerable from SV JSON)
    alignment_head_fraction: float = DEFAULT_HEAD_FRACTION  # head = top 10 % of SVs
    head_drop_counts: tuple[int, ...] = (1, 2, 3, 5, 10)  # head-drop probe (absolute n)
    num_workers: int | None = None  # DataLoader workers; None → auto (4 on CUDA else 0)
    study_set: str = ""  # provenance: which run-study set produced the run ("" = legacy)
    val_every: int = 1  # validate every N epochs (always at logged epochs and the last one)
    warmup_epochs: int = 5  # LR warmup length in epochs (scaled for long-epoch cells)

    @property
    def scenario_label(self) -> str:
        """Short label: the enum value for A-F, else the string tag."""
        return (
            self.scenario.value if isinstance(self.scenario, ScenarioType) else str(self.scenario)
        )

    @property
    def name(self) -> str:
        """Identity used for the MLflow experiment name (``spectral_{name}``)."""
        s = self.scenario
        return f"scenario_{s.value}" if isinstance(s, ScenarioType) else str(s)


@dataclass
class ScenarioResult:
    """Results from a single scenario run."""

    scenario: ScenarioType
    seed: int
    success: bool
    final_accuracy: float
    best_val_accuracy: float
    final_metrics: dict[str, float]  # Final epoch spectral metrics
    metrics_history: dict[int, dict[str, float]]  # epoch -> metrics
    training_time: float
    error_message: str | None = None
    test_accuracy: float = 0.0  # held-out test accuracy (the unbiased headline number)
    alignment_history: dict[int, dict[str, float]] | None = None  # epoch -> alignment agg
    truncation: dict[str, Any] | None = None  # {"bulk": ..., "head": ...} summaries


# =============================================================================
# Scenario Definitions
# =============================================================================

# Default scenario configurations based on meeting notes
SCENARIO_CONFIGS = {
    ScenarioType.A_EXPRESSIVE_SIMPLE: ScenarioConfig(
        scenario=ScenarioType.A_EXPRESSIVE_SIMPLE,
        model_name="vit_tiny_patch16_224",
        embed_dim=192,
        depth=6,
        dataset_name="synthetic",
        num_samples=1000,
        num_classes=3,
        epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29],
        description="Expressive network (ViT-Tiny) on simple synthetic data",
    ),
    ScenarioType.B_EXPRESSIVE_COMPLEX: ScenarioConfig(
        scenario=ScenarioType.B_EXPRESSIVE_COMPLEX,
        model_name="vit_tiny_patch16_224",
        embed_dim=192,
        depth=6,
        dataset_name="pathmnist",
        num_samples=None,  # Use full dataset
        num_classes=9,
        epochs=50,
        batch_size=64,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 49],
        description="Expressive network (ViT-Tiny) on complex PathMNIST data",
    ),
    ScenarioType.C_REDUCED_COMPLEX: ScenarioConfig(
        scenario=ScenarioType.C_REDUCED_COMPLEX,
        model_name="vit_tiny_patch16_224",  # Will be modified with reduced width
        embed_dim=96,  # Reduced from 192
        depth=3,  # Reduced from 6
        dataset_name="pathmnist",
        num_samples=None,
        num_classes=9,
        epochs=50,
        batch_size=64,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 49],
        description="Reduced expressivity network on complex PathMNIST data",
    ),
    ScenarioType.D_REDUCED_SIMPLE: ScenarioConfig(
        scenario=ScenarioType.D_REDUCED_SIMPLE,
        model_name="vit_tiny_patch16_224",  # Will be modified with reduced width
        embed_dim=96,  # Reduced from 192
        depth=3,  # Reduced from 6
        dataset_name="synthetic",
        num_samples=1000,
        num_classes=3,
        epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29],
        description="Reduced expressivity network on simple synthetic data",
    ),
    ScenarioType.E_TINY_SIMPLE: ScenarioConfig(
        scenario=ScenarioType.E_TINY_SIMPLE,
        model_name="vit_tiny_patch16_224",  # Will be modified with minimal width
        embed_dim=32,  # Minimal: 1/6 of standard ViT-Tiny
        depth=1,  # Single layer
        dataset_name="synthetic",
        num_samples=1000,
        num_classes=3,
        epochs=30,
        batch_size=32,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29],
        description="Tiny network (embed=32, depth=1) on simple synthetic data",
    ),
    ScenarioType.F_TINY_COMPLEX: ScenarioConfig(
        scenario=ScenarioType.F_TINY_COMPLEX,
        model_name="vit_tiny_patch16_224",  # Will be modified with minimal width
        embed_dim=32,  # Minimal: 1/6 of standard ViT-Tiny
        depth=1,  # Single layer
        dataset_name="pathmnist",
        num_samples=None,  # Use full dataset
        num_classes=9,
        epochs=50,
        batch_size=64,
        learning_rate=1e-4,
        seeds=[42, 123, 456],
        log_epochs=[0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 49],
        description="Tiny network (embed=32, depth=1) on complex PathMNIST data",
    ),
}


# =============================================================================
# Capacity × Complexity study grid (disentangled width vs depth)
# =============================================================================

# Per EXPERIMENT_PLAN_V2.md: capacity is two SEPARATE axes (the old A-F grid moved
# width and depth together). The headline is a WIDTH sweep at fixed depth=6; depth is
# swept separately at fixed width=192. Each cell is a string-tagged ScenarioConfig that
# logs to its own ``spectral_{tag}`` MLflow experiment.

# (tag suffix, dataset_name, num_classes, is_complex)
_SIMPLE = ("syn", "synthetic", 3, False)
_PATH = ("path", "pathmnist", 9, True)

_LOG_EPOCHS_SIMPLE = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 29]
_LOG_EPOCHS_COMPLEX = [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 49]


def _cell(tag: str, embed: int, depth: int, dataset: tuple) -> ScenarioConfig:
    """Build one study cell. ``dataset`` is a (suffix, name, num_classes, is_complex)."""
    suffix, name, num_classes, is_complex = dataset
    return ScenarioConfig(
        scenario=f"{tag}_{suffix}",
        model_name="vit_tiny_patch16_224",
        embed_dim=embed,
        depth=depth,
        dataset_name=name,
        num_samples=None if is_complex else 1000,
        num_classes=num_classes,
        epochs=50 if is_complex else 30,
        batch_size=64 if is_complex else 32,
        learning_rate=1e-4,
        seeds=[],  # filled in by run-study
        log_epochs=_LOG_EPOCHS_COMPLEX if is_complex else _LOG_EPOCHS_SIMPLE,
        description=f"{tag} (embed={embed}, depth={depth}) on {name}",
    )


def build_study_configs(tier: int = 1) -> list[ScenarioConfig]:
    """Assemble the capacity × complexity study for a given tier (see EXPERIMENT_PLAN_V2).

    Tier 1 — width sweep (depth=6) × {synthetic, PathMNIST}: the paper's spine.
    Tier 2 — + depth sweep (width=192) × {synthetic, PathMNIST}; + Blood/DermaMNIST @ w192.
    Tier 3 — + the original A-F grid for continuity.
    """
    configs: list[ScenarioConfig] = []

    # Tier 1: width sweep at fixed depth=6 (the headline capacity axis).
    widths = [("w384", 384), ("w192", 192), ("w096", 96), ("w048", 48)]
    for tag, embed in widths:
        for dataset in (_SIMPLE, _PATH):
            configs.append(_cell(tag, embed, 6, dataset))

    if tier >= 2:
        # Depth sweep at fixed width=192 (d6 already covered by w192 above).
        depths = [("d2", 2), ("d4", 4), ("d8", 8)]
        for tag, depth in depths:
            for dataset in (_SIMPLE, _PATH):
                configs.append(_cell(tag, 192, depth, dataset))
        # Extra medical datasets at the reference width to widen the complexity axis.
        for suffix, name, num_classes in (("blood", "bloodmnist", 8), ("derma", "dermamnist", 7)):
            configs.append(_cell("w192", 192, 6, (suffix, name, num_classes, True)))

    if tier >= 3:
        # Original A-F grid (keeps the confounded width+depth scenarios for continuity).
        configs.extend(SCENARIO_CONFIGS.values())

    return configs


# =============================================================================
# Follow-up study (Sept 2026) — the controls the June-2026 tier-3 results need
# =============================================================================

# MedMNIST v2 official train-split sizes (used only to document step matching).
MEDMNIST_TRAIN_SIZE = {"pathmnist": 89_996, "bloodmnist": 11_959, "dermamnist": 7_007}

STUDY_SETS = ("tiers", "followup", "followup-wide")


def _variant(base: ScenarioConfig, tag: str, description: str, **overrides: Any) -> ScenarioConfig:
    """Copy a sweep cell under a new experiment tag with field overrides."""
    return replace(base, scenario=tag, description=description, **overrides)


def _scale_log_epochs(
    epochs: list[int], from_steps_per_epoch: int, to_steps_per_epoch: int
) -> list[int]:
    """Map a logging schedule to a cell with a different epoch length so the spectral
    snapshots fall at (approximately) the SAME optimizer-step counts."""
    scaled = sorted({round(e * from_steps_per_epoch / to_steps_per_epoch) for e in epochs})
    return [e for e in scaled if e >= 0]


def build_followup_configs(wide: bool = False) -> list[ScenarioConfig]:
    """Cells for the follow-up run (see RESULTS_TIER3_JUNE2026.md §4):

    1. ``w192_synlong``  — synthetic, 90k images × 50 epochs @ bs 64 ⇒ ≈70k optimizer steps,
       i.e. STEP-MATCHED to ``w192_path`` (the "is complexity just training length?" control).
       ``w192_synlong1k`` — the SAME 1k images as ``w192_syn`` × 2200 epochs @ bs 32 ⇒ ≈70k
       steps: separates steps from data volume.
    2. ``w192_pathshort`` — PathMNIST with a 1k-image stratified TRAIN subsample × 30 epochs
       @ bs 32 ⇒ ≈0.9k steps, step-matched to ``w192_syn`` (val/test stay full).
    3. ``w192_path_wd0`` / ``w384_path_wd0`` — weight-decay ablation (mechanism control).
    4. The tier-1 width sweep (8 cells) re-run with the new probes; the figure readers keep
       the latest run per seed, so re-running does not double-count.
    5. ``wide=True`` adds ``w768_path`` and ``w024_path`` (corridor edges).

    Every cell saves its final checkpoint and logs no histogram PNGs.
    """
    w192_syn = _cell("w192", 192, 6, _SIMPLE)
    w192_path = _cell("w192", 192, 6, _PATH)
    w384_path = _cell("w384", 384, 6, _PATH)

    cells: list[ScenarioConfig] = [
        _variant(
            w192_syn,
            "w192_synlong",
            "w192 on synthetic, 90k images × 50 ep (≈70k steps: step-matched to w192_path)",
            num_samples=90_000,
            epochs=50,
            batch_size=64,
            log_epochs=list(_LOG_EPOCHS_COMPLEX),
            num_eval_samples=2000,
        ),
        _variant(
            w192_path,
            "w192_pathshort",
            "w192 on PathMNIST-1k (train subsample) × 30 ep (≈0.9k steps: matched to w192_syn)",
            train_subsample=1000,
            epochs=30,
            batch_size=32,
            log_epochs=list(_LOG_EPOCHS_SIMPLE),
        ),
        # Same 1k synthetic images as w192_syn, trained for the PathMNIST budget: separates
        # optimizer STEPS from data VOLUME (w192_synlong changes both). Spectral logging is
        # scheduled at the same step counts as the 50-epoch cells (32 steps/epoch here).
        _variant(
            w192_syn,
            "w192_synlong1k",
            "w192 on synthetic-1k × 2200 ep @ bs 32 (≈70k steps: steps-only control)",
            num_samples=1000,
            epochs=2200,
            batch_size=32,
            log_epochs=_scale_log_epochs(
                _LOG_EPOCHS_COMPLEX, from_steps_per_epoch=1407, to_steps_per_epoch=32
            ),
            num_eval_samples=2000,
            val_every=44,  # ≈ one 50-epoch-cell epoch (1407 steps) between validations
            warmup_epochs=220,  # ≈ 5 × 1407 steps, the warmup the 50-epoch cells get
        ),
        _variant(
            w192_path, "w192_path_wd0", "w192 on PathMNIST, weight decay 0", weight_decay=0.0
        ),
        _variant(
            w384_path, "w384_path_wd0", "w384 on PathMNIST, weight decay 0", weight_decay=0.0
        ),
    ]
    cells.extend(build_study_configs(1))
    if wide:
        cells.append(_cell("w768", 768, 6, _PATH))
        cells.append(_cell("w024", 24, 6, _PATH))
    for c in cells:
        c.save_checkpoint = True
        c.log_histograms = False
    return cells


def build_study_set(study_set: str, tier: int = 1) -> list[ScenarioConfig]:
    """Dispatch ``run-study --set``: ``tiers`` (cumulative, via ``--tier``) or a follow-up set."""
    if study_set == "tiers":
        return build_study_configs(tier)
    if study_set == "followup":
        return build_followup_configs(wide=False)
    if study_set == "followup-wide":
        return build_followup_configs(wide=True)
    raise ValueError(f"unknown study set {study_set!r}; choose from {STUDY_SETS}")


def expected_total_steps(config: ScenarioConfig) -> int:
    """Optimizer steps the training loop will take (documentation / tests for step matching)."""
    if config.dataset_name == "synthetic":
        n = config.num_samples or 1000
        steps = -(-n // config.batch_size)  # DataLoader without drop_last → ceil
    else:
        n = config.train_subsample or MEDMNIST_TRAIN_SIZE[config.dataset_name]
        steps = n // config.batch_size  # BaseDataset train loader uses drop_last=True
    return steps * config.epochs


# =============================================================================
# Model Creation with Expressivity Control
# =============================================================================


# Input resolution and patch size for the scenarios. patch_size=4 at 28px yields
# a real 7x7=49-patch grid (50 tokens incl. CLS) and divides 28 exactly (no
# discarded border); patch16 would collapse to a single 1x1 patch token.
INPUT_IMAGE_SIZE = 28
PATCH_SIZE = 4


def create_model_for_scenario(
    scenario_config: ScenarioConfig,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create the ViT for a scenario.

    All six scenarios go through ONE parameterized path so the configured
    (embed_dim, depth) are authoritative: A/B = 6 layers, C/D = 3, E/F = 1.
    (Previously A/B silently used timm's default 12 layers because depth was not
    passed.) patch_size=4 makes this a real ViT (49 patch tokens) rather than a
    degenerate 1-patch model.
    """
    import timm

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=scenario_config.num_classes,
        in_chans=3,
        img_size=INPUT_IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=scenario_config.embed_dim,
        depth=scenario_config.depth,
        num_heads=max(1, scenario_config.embed_dim // 32),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
    )

    # Guard against silent config/architecture drift (the A/B depth=6-but-built-12 bug).
    assert len(model.blocks) == scenario_config.depth, (
        f"depth mismatch for scenario {scenario_config.scenario}: "
        f"configured {scenario_config.depth}, built {len(model.blocks)}"
    )

    return model.to(device)


# =============================================================================
# Spectral Analysis Functions
# =============================================================================


def extract_and_analyze_weights(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    Extract all weight matrices and compute spectral metrics.

    Returns:
        Dictionary with:
        - per_layer_metrics: dict[layer_name, metrics]
        - aggregated_metrics: dict of mean/std across layers
        - singular_values: dict[layer_name, list[float]] - full SV arrays
    """
    model.eval()

    # Extract weights from different components
    qkv_weights = extract_qkv_weights(model)
    attn_weights = extract_attention_weights(model)
    mlp_weights = extract_mlp_weights(model)

    all_weights = qkv_weights + attn_weights + mlp_weights

    per_layer_metrics = {}
    singular_values = {}
    layer_metrics_list = []

    for weight_info in all_weights:
        # Compute spectral metrics
        metrics = get_spectral_metrics(weight_info.weight)
        per_layer_metrics[weight_info.name] = metrics
        layer_metrics_list.append(metrics)

        # Store singular values for distribution analysis
        try:
            from scipy.linalg import svd

            s = svd(weight_info.weight.astype(np.float64), compute_uv=False)
            singular_values[weight_info.name] = s.tolist()
        except Exception:
            singular_values[weight_info.name] = []

    # Aggregate metrics across layers
    aggregated = aggregate_spectral_metrics(layer_metrics_list)

    return {
        "per_layer_metrics": per_layer_metrics,
        "aggregated_metrics": aggregated,
        "singular_values": singular_values,
    }


def log_spectral_artifacts(
    analysis: dict[str, Any],
    epoch: int,
    run_id: str | None = None,
    histograms: bool = False,
) -> None:
    """
    Log spectral analysis artifacts to MLflow.

    Creates:
    - spectral/epoch_{N}/values.json - Full singular value arrays
    - spectral/epoch_{N}/metrics.json - Per-layer metrics
    - spectral/epoch_{N}/histograms/*.png - Histogram plots
    """
    import tempfile

    # Create artifact directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        epoch_dir = Path(tmpdir) / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True)

        # Save singular values as JSON
        values_file = epoch_dir / "singular_values.json"
        with open(values_file, "w") as f:
            json.dump(analysis["singular_values"], f, indent=2)
        mlflow.log_artifact(str(values_file), f"spectral/epoch_{epoch}")

        # Save per-layer metrics as JSON
        metrics_file = epoch_dir / "layer_metrics.json"
        with open(metrics_file, "w") as f:
            # Convert any NaN to null for JSON
            clean_metrics = {}
            for layer, metrics in analysis["per_layer_metrics"].items():
                clean_metrics[layer] = {
                    k: v if np.isfinite(v) else None for k, v in metrics.items()
                }
            json.dump(clean_metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_file), f"spectral/epoch_{epoch}")

        # Histogram PNGs are opt-in: ~430 files/run, CPU-bound, and fully regenerable
        # from singular_values.json. Off by default since the follow-up study.
        if not histograms:
            return

        # Generate and save histogram plots
        histograms_dir = epoch_dir / "histograms"
        histograms_dir.mkdir()

        try:
            for layer_name, svs in analysis["singular_values"].items():
                if not svs:
                    continue

                fig, ax = plt.subplots(figsize=(8, 5))
                log_svs = np.log10(np.array(svs) + 1e-10)
                ax.hist(log_svs, bins=30, edgecolor="black", alpha=0.7)
                ax.set_xlabel("log₁₀(σ)")
                ax.set_ylabel("Count")
                safe_name = layer_name.replace(".", "_").replace("/", "_")
                ax.set_title(f"Singular Values: {layer_name}")
                fig.tight_layout()

                plot_path = histograms_dir / f"{safe_name}.png"
                fig.savefig(plot_path, dpi=100)
                plt.close(fig)

            # Log all histograms
            mlflow.log_artifacts(str(histograms_dir), f"spectral/epoch_{epoch}/histograms")

        except Exception as e:
            logger.warning(f"Could not generate histogram plots: {e}")


# =============================================================================
# Meeting-priority analyses: gradient alignment + tail truncation
# =============================================================================


def _num_workers_for(device: torch.device) -> int:
    """Data-loading workers. macOS/MPS/CPU runs leak file descriptors with workers in
    this figure-heavy pipeline (kept at 0); CUDA boxes — where the cloud study runs and
    data loading is the bottleneck — use real workers.

    Note: the loaders pass no ``worker_init_fn``/``generator``, so CUDA multi-worker runs
    are statistically (mean ± std over seeds) but not bit-for-bit reproducible — matching
    the project's stance on determinism (no strict bit-reproducibility is claimed)."""
    return 4 if device.type == "cuda" else 0


def record_gradient_alignment(
    model: torch.nn.Module,
    probe_batch: tuple[torch.Tensor, torch.Tensor],
    criterion: torch.nn.Module,
    device: torch.device,
    head_fraction: float = DEFAULT_HEAD_FRACTION,
) -> dict[str, float]:
    """Alignment between the data gradient and the rank-reducing directions.

    Logs the legacy full-basis cosine (``cos_sim_mean`` etc.) AND the subspace-resolved
    probe (``cos_head_mean``, ``cos_tail_mean``, ``*_energy_enrichment_mean``, per-type
    ``{q,k,v,proj,fc1,fc2}_cos_*``); see ``metrics/gradient_alignment.py``.


    For each attention/MLP weight W the rank-reducing (nuclear-norm) direction is U Vᵀ.
    The SGD update is −η∇L and the rank-reducing step is −U Vᵀ, so the update locally
    reduces nuclear norm (≈ simplifies the matrix toward lower rank) exactly when
    cos(∇L, U Vᵀ) > 0. Measured in eval mode (drop_path off) on a FIXED probe batch so
    the trajectory is deterministic and comparable across epochs/scenarios. Gradients are
    zeroed afterwards, leaving optimizer state untouched.
    """
    images, labels = probe_batch
    images = images.to(device)
    labels = labels.to(device)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    was_training = model.training
    model.eval()
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        results = analyze_model_gradient_alignment(
            model, layer_patterns=["attn", "mlp"], head_fraction=head_fraction
        )
        aggregated = aggregate_gradient_alignment(results)
    except Exception as align_err:
        # An optional diagnostic must never sink an otherwise-finished training run;
        # return all-NaN (callers skip non-finite values when logging).
        logger.warning(f"Gradient-alignment probe failed: {align_err}")
        aggregated = aggregate_gradient_alignment([])
    finally:
        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()
    return aggregated


def run_truncation_analysis(
    model: torch.nn.Module,
    test_loader: Any,
    device: torch.device,
    max_batches: int = 32,
    head_drop_counts: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> dict[str, Any]:
    """Post-training Eckart-Young truncation sweep in both directions, logged to MLflow.

    Also runs the HEAD-DROP probe (drop the top-``n`` singular values per matrix, absolute
    counts, split Q/K/V) and logs ``truncation/headn_acc`` (step = n) for all matrices
    plus ``truncation/headn_{qkv,proj,mlp}_acc`` for the per-group ablations.

    - ``bulk``: keep the largest SVs (effective-rank / MDL probe).
    - ``head``: drop the largest SVs / heavy-tail outliers (signal-vs-noise probe).

    Logs ``truncation/{mode}_acc`` and ``_acc_drop`` as stepped curves
    (step = round(100·retention)) plus scalar summaries; returns the raw results so the
    caller can persist a JSON artifact. Runs on the held-out TEST loader, capped at
    ``max_batches`` so the per-seed cost stays small.
    """
    ratios = [1.0, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1]
    summary: dict[str, Any] = {}
    for mode in ("bulk", "head"):
        results = run_truncation_experiment(
            model,
            test_loader,
            device,
            retention_ratios=ratios,
            use_energy=False,
            mode=mode,
            max_batches=max_batches,
        )
        analysis = analyze_truncation_results(results)
        for r in results:
            step = int(round(r["retention_ratio"] * 100))
            mlflow.log_metric(f"truncation/{mode}_acc", r["truncated_accuracy"], step=step)
            mlflow.log_metric(f"truncation/{mode}_acc_drop", r["accuracy_drop"], step=step)
        mlflow.log_metric(f"truncation/{mode}_max_acc_drop", float(analysis["max_accuracy_drop"]))
        if analysis["critical_threshold"] is not None:
            mlflow.log_metric(
                f"truncation/{mode}_critical_threshold", float(analysis["critical_threshold"])
            )
        summary[mode] = {"results": results, "analysis": analysis}

    if head_drop_counts:
        head_results = run_head_drop_experiment(
            model, test_loader, device, n_drops=tuple(head_drop_counts), max_batches=max_batches
        )
        for r in head_results:
            prefix = (
                "truncation/headn" if r["group"] == "all" else f"truncation/headn_{r['group']}"
            )
            mlflow.log_metric(f"{prefix}_acc", r["truncated_accuracy"], step=r["n_drop"])
            mlflow.log_metric(f"{prefix}_acc_drop", r["accuracy_drop"], step=r["n_drop"])
        summary["head_drop"] = {"results": head_results}
    return summary


def _log_truncation_artifact(summary: dict[str, Any]) -> None:
    """Persist a compact, JSON-safe truncation report as an MLflow artifact."""
    import tempfile

    jsonable: dict[str, Any] = {}
    for mode, payload in summary.items():
        if mode == "head_drop":
            jsonable[mode] = [
                {
                    k: (
                        int(v)
                        if isinstance(v, int) and not isinstance(v, bool)
                        else float(v)
                        if isinstance(v, float)
                        else v
                    )
                    for k, v in r.items()
                }
                for r in payload["results"]
            ]
            continue
        analysis = payload["analysis"]
        jsonable[mode] = {
            "critical_threshold": analysis.get("critical_threshold"),
            "max_accuracy_drop": float(analysis.get("max_accuracy_drop", 0.0)),
            "interpretation": analysis.get("interpretation", ""),
            "curve": [
                {
                    "retention_ratio": float(r["retention_ratio"]),
                    "truncated_accuracy": float(r["truncated_accuracy"]),
                    "accuracy_drop": float(r["accuracy_drop"]),
                    "avg_energy_retained": float(r["avg_energy_retained"]),
                }
                for r in payload["results"]
            ],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "truncation_report.json"
        with open(path, "w") as f:
            json.dump(jsonable, f, indent=2)
        mlflow.log_artifact(str(path), "truncation")


def _log_final_checkpoint(
    model: torch.nn.Module, config: ScenarioConfig, seed: int, test_accuracy: float
) -> None:
    """Save the final weights (+ everything needed to rebuild the model) as
    ``model/final.pt``. Rebuild directly with ``timm.create_model("vit_tiny_patch16_224",
    pretrained=False, img_size=image_size, patch_size=patch_size, embed_dim=..., depth=...,
    num_heads=..., num_classes=...)`` and ``load_state_dict(payload["state_dict"])``; the
    payload holds only tensors/str/int/float so it loads under ``weights_only=True``."""
    import tempfile

    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scenario": config.name,
        "embed_dim": config.embed_dim,
        "depth": config.depth,
        "num_heads": max(1, config.embed_dim // 32),
        "patch_size": PATCH_SIZE,
        "image_size": INPUT_IMAGE_SIZE,
        "num_classes": config.num_classes,
        "dataset": config.dataset_name,
        "seed": seed,
        "test_accuracy": float(test_accuracy),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "final.pt"
        torch.save(payload, path)
        mlflow.log_artifact(str(path), "model")


# =============================================================================
# Training Loop with Spectral Tracking
# =============================================================================


def run_scenario_experiment(
    config: ScenarioConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> ScenarioResult:
    """
    Run a single scenario experiment with spectral tracking.
    """
    start_time = time.time()
    set_seed(seed)

    # Create experiment name. A-F keep ``spectral_scenario_{A-F}`` (figures contract);
    # sweep cells get their own ``spectral_{tag}`` experiment.
    experiment_name = f"spectral_{config.name}"

    try:
        # Setup MLflow
        mlflow.set_tracking_uri(str(output_dir))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"seed_{seed}"):
            # Log configuration
            mlflow.log_params(
                {
                    "scenario": config.scenario_label,
                    "tag": config.name,
                    "model_name": config.model_name,
                    "embed_dim": config.embed_dim,
                    "depth": config.depth,
                    "dataset": config.dataset_name,
                    "num_samples": config.num_samples or "full",
                    "num_classes": config.num_classes,
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "seed": seed,
                    "weight_decay": config.weight_decay,
                    "train_subsample": config.train_subsample or "full",
                    "num_eval_samples": config.num_eval_samples,
                    "study_set": config.study_set or "legacy",
                    "val_every": config.val_every,
                    "warmup_epochs": config.warmup_epochs,
                }
            )

            # Create model
            logger.info(f"Creating model for scenario {config.scenario_label}")
            model = create_model_for_scenario(config, device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("num_parameters", num_params)

            # Load dataset
            num_workers = (
                config.num_workers if config.num_workers is not None else _num_workers_for(device)
            )
            logger.info(f"Loading dataset: {config.dataset_name}")
            if config.dataset_name == "synthetic":
                from vision_spectra.data.synthetic import create_synthetic_dataset

                train_loader, val_loader, test_loader = create_synthetic_dataset(
                    num_classes=config.num_classes,
                    num_samples_train=config.num_samples or 1000,
                    num_samples_val=config.num_eval_samples,
                    num_samples_test=config.num_eval_samples,
                    batch_size=config.batch_size,
                    seed=seed,
                    num_workers=num_workers,
                )
            else:
                # num_workers is 0 on macOS/MPS (FD leaks in this figure-heavy pipeline)
                # and >0 on CUDA, where the cloud study runs and loading is the bottleneck.
                dataset_config = DatasetConfig(
                    name=DatasetName(config.dataset_name),
                    batch_size=config.batch_size,
                    sample_ratio=1.0 if config.num_samples is None else 0.5,
                    train_subsample=config.train_subsample,
                    num_workers=num_workers,
                )
                dataset_obj = get_dataset(dataset_config, data_dir=DATA_DIR)
                train_loader = dataset_obj.get_train_loader()
                val_loader = dataset_obj.get_val_loader()
                test_loader = dataset_obj.get_test_loader()

            # Optimizer-step budget, so the step-count confound (steps track dataset
            # size) can be read off the run params instead of being re-derived.
            steps_per_epoch = len(train_loader)
            mlflow.log_params(
                {
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": steps_per_epoch * config.epochs,
                }
            )

            # Setup loss, optimizer, and LR schedule using the SAME recipe as the
            # other experiment families (build_optimizer/build_scheduler/
            # warmup_factor) so the headline spectral study is not trained by a
            # divergent loop (it previously had no scheduler, warmup, or grad
            # clipping). A fixed epoch budget with NO early stopping is kept
            # deliberately, so Δα is measured at a common endpoint across scenarios.
            criterion = torch.nn.CrossEntropyLoss()
            opt_config = OptimizerConfig(
                name=OptimizerName.ADAMW,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                scheduler=SchedulerName.COSINE,
                warmup_epochs=config.warmup_epochs,
            )
            optimizer = build_optimizer(model, opt_config)
            scheduler = build_scheduler(optimizer, opt_config, config.epochs)
            base_lrs = [g["lr"] for g in optimizer.param_groups]
            grad_clip = 1.0

            # Track metrics + gradient-alignment history
            metrics_history: dict[int, dict[str, float]] = {}
            alignment_history: dict[int, dict[str, float]] = {}
            # Fixed probe batch for the alignment measurement. Drawn from the val loader
            # (shuffle=False) so it is deterministic and does NOT perturb the training
            # data order / reproducibility; val and train share the class distribution.
            probe_batch = next(iter(val_loader)) if config.run_alignment else None

            # Log initial spectral metrics (epoch 0, before training)
            if 0 in config.log_epochs:
                logger.info("Logging initial spectral metrics (epoch 0)")
                analysis = extract_and_analyze_weights(model, device)
                metrics_history[0] = analysis["aggregated_metrics"]
                log_spectral_artifacts(analysis, epoch=0, histograms=config.log_histograms)

                for key, value in analysis["aggregated_metrics"].items():
                    if np.isfinite(value):
                        mlflow.log_metric(f"spectral/{key}", value, step=0)

                if probe_batch is not None:
                    align = record_gradient_alignment(
                        model,
                        probe_batch,
                        criterion,
                        device,
                        head_fraction=config.alignment_head_fraction,
                    )
                    alignment_history[0] = align
                    for key, value in align.items():
                        if np.isfinite(value):
                            mlflow.log_metric(f"alignment/{key}", value, step=0)

            # Training loop
            best_val_accuracy = 0.0
            final_accuracy = 0.0
            val_accuracy = float("nan")

            for epoch in range(1, config.epochs + 1):
                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for step_idx, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    # squeeze(-1) (not squeeze()) so a singleton [1, 1] batch
                    # collapses to [1], not a 0-dim scalar that breaks the loss.
                    if labels.dim() > 1:
                        labels = labels.squeeze(-1)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()

                    # LR warmup (0-based epoch) + gradient clipping, matching the
                    # shared trainer recipe.
                    factor = warmup_factor(
                        epoch - 1, step_idx, len(train_loader), opt_config.warmup_epochs
                    )
                    if factor is not None:
                        for g, base in zip(optimizer.param_groups, base_lrs, strict=False):
                            g["lr"] = base * factor
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                train_accuracy = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)

                # Validation (every ``val_every`` epochs, at every logged epoch and the last
                # one; long-epoch cells such as synthetic-1k × 2200 ep would otherwise spend
                # most of their time validating).
                do_val = (
                    config.val_every <= 1
                    or epoch % config.val_every == 0
                    or epoch == config.epochs
                    or epoch in config.log_epochs
                )
                if do_val:
                    model.eval()
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for images, labels in val_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            if labels.dim() > 1:
                                labels = labels.squeeze(-1)

                            outputs = model(images)
                            _, predicted = outputs.max(1)
                            val_total += labels.size(0)
                            val_correct += predicted.eq(labels).sum().item()

                    val_accuracy = 100.0 * val_correct / val_total
                    final_accuracy = val_accuracy

                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy

                # Step the LR scheduler once warmup is over (epoch is 1-based).
                if scheduler is not None and (epoch - 1) >= opt_config.warmup_epochs:
                    scheduler.step()

                # Log training metrics
                epoch_metrics = {
                    "train/loss": avg_train_loss,
                    "train/accuracy": train_accuracy,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                if do_val:
                    epoch_metrics["val/accuracy"] = val_accuracy
                mlflow.log_metrics(epoch_metrics, step=epoch)

                # Log spectral metrics at specified epochs
                if epoch in config.log_epochs:
                    logger.info(f"Epoch {epoch}: Logging spectral metrics")
                    analysis = extract_and_analyze_weights(model, device)
                    metrics_history[epoch] = analysis["aggregated_metrics"]
                    log_spectral_artifacts(analysis, epoch=epoch, histograms=config.log_histograms)

                    for key, value in analysis["aggregated_metrics"].items():
                        if np.isfinite(value):
                            mlflow.log_metric(f"spectral/{key}", value, step=epoch)

                    # Gradient alignment to the rank-reducing flow at the same epochs.
                    if probe_batch is not None:
                        align = record_gradient_alignment(
                            model,
                            probe_batch,
                            criterion,
                            device,
                            head_fraction=config.alignment_head_fraction,
                        )
                        alignment_history[epoch] = align
                        for key, value in align.items():
                            if np.isfinite(value):
                                mlflow.log_metric(f"alignment/{key}", value, step=epoch)

                # Progress logging
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch}/{config.epochs}: "
                        f"Train Loss={avg_train_loss:.4f}, "
                        f"Train Acc={train_accuracy:.2f}%, "
                        f"Val Acc={val_accuracy:.2f}%"
                    )

            # Final spectral analysis. Also emit the metrics (not just artifacts)
            # so the LAST logged spectral/* metric reflects the truly-final
            # weights; otherwise downstream Δα would use a one-epoch-stale value.
            final_epoch = config.epochs
            if final_epoch not in metrics_history:
                analysis = extract_and_analyze_weights(model, device)
                metrics_history[final_epoch] = analysis["aggregated_metrics"]
                log_spectral_artifacts(
                    analysis, epoch=final_epoch, histograms=config.log_histograms
                )

                for key, value in analysis["aggregated_metrics"].items():
                    if np.isfinite(value):
                        mlflow.log_metric(f"spectral/{key}", value, step=final_epoch)

            # Final-epoch alignment, so the alignment trajectory shares the endpoint.
            if probe_batch is not None and final_epoch not in alignment_history:
                align = record_gradient_alignment(
                    model,
                    probe_batch,
                    criterion,
                    device,
                    head_fraction=config.alignment_head_fraction,
                )
                alignment_history[final_epoch] = align
                for key, value in align.items():
                    if np.isfinite(value):
                        mlflow.log_metric(f"alignment/{key}", value, step=final_epoch)

            # Held-out TEST accuracy on the final model: the unbiased
            # generalization estimate (val accuracy is used only as a running
            # diagnostic / for model comparison and is optimistically biased).
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    if labels.dim() > 1:
                        labels = labels.squeeze(-1)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            test_accuracy = 100.0 * test_correct / max(test_total, 1)

            # Log final metrics
            mlflow.log_metrics(
                {
                    "final/val_accuracy": best_val_accuracy,
                    "final/train_accuracy": train_accuracy,
                    "final/test_accuracy": test_accuracy,
                }
            )

            # Final weights as an artifact so post-hoc probes (new truncation grids,
            # alignment variants, CCDFs) do not require retraining.
            if config.save_checkpoint:
                try:
                    _log_final_checkpoint(model, config, seed, test_accuracy)
                except Exception as ckpt_err:  # an artifact failure must not FAIL the run
                    logger.warning(f"Checkpoint artifact failed: {ckpt_err}")

            # Post-training tail-truncation sweep (bulk + head) on the held-out test
            # split. Runs BEFORE cleanup / del model; restores weights internally.
            truncation_summary: dict[str, Any] | None = None
            if config.run_truncation:
                try:
                    truncation_summary = run_truncation_analysis(
                        model, test_loader, device, head_drop_counts=config.head_drop_counts
                    )
                    _log_truncation_artifact(truncation_summary)
                except Exception as trunc_err:  # never let truncation sink a finished run
                    logger.warning(f"Truncation analysis failed: {trunc_err}")

            training_time = time.time() - start_time

            # Comprehensive cleanup to prevent resource leaks
            # Clean up DataLoaders first (releases multiprocessing workers)
            cleanup_dataloaders(train_loader, val_loader, test_loader)

            # Clean up matplotlib to release figure file handles
            cleanup_matplotlib()

            # Clean up model and tensors
            del model
            del train_loader
            del val_loader
            if "dataset_obj" in locals():
                del dataset_obj

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            return ScenarioResult(
                scenario=config.scenario,
                seed=seed,
                success=True,
                final_accuracy=final_accuracy,
                best_val_accuracy=best_val_accuracy,
                final_metrics=metrics_history.get(final_epoch, {}),
                metrics_history=metrics_history,
                training_time=training_time,
                test_accuracy=test_accuracy,
                alignment_history=alignment_history,
                truncation=truncation_summary,
            )

    except Exception as e:
        logger.error(f"Scenario {config.scenario_label} seed {seed} failed: {e}")
        import traceback

        traceback.print_exc()

        # Cleanup on error to prevent resource leaks
        cleanup_matplotlib()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return ScenarioResult(
            scenario=config.scenario,
            seed=seed,
            success=False,
            final_accuracy=0.0,
            best_val_accuracy=0.0,
            final_metrics={},
            metrics_history={},
            training_time=time.time() - start_time,
            error_message=str(e),
        )


# =============================================================================
# CLI Commands
# =============================================================================


def resolve_device(device_choice: DeviceChoice) -> torch.device:
    """Resolve a DeviceChoice to a torch.device via the canonical resolver."""
    from vision_spectra.settings import resolve_device as _resolve

    return _resolve(device_choice.value)


@app.command("scenario-a")
def run_scenario_a(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n", help="Number of seeds"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario A: Expressive network + Simple data.

    Expected outcome: NO heavy tails (network doesn't need to compress).
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.A_EXPRESSIVE_SIMPLE]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario A: {config.description}[/bold blue]")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("scenario-b")
def run_scenario_b(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario B: Expressive network + Complex data.

    Expected outcome: Heavy tails EMERGE (network must compress).
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.B_EXPRESSIVE_COMPLEX]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario B: {config.description}[/bold blue]")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("scenario-c")
def run_scenario_c(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario C: Reduced expressivity + Complex data.

    Expected outcome: NO heavy tails (network lacks capacity to form them).
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.C_REDUCED_COMPLEX]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario C: {config.description}[/bold blue]")
    console.print(f"  Reduced embed_dim: {config.embed_dim}, depth: {config.depth}")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("scenario-d")
def run_scenario_d(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario D: Reduced expressivity + Simple data.

    Expected outcome: NO heavy tails (reduced network, simple data - no need to compress).
    This tests whether reduced networks develop heavy tails on simple data.
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.D_REDUCED_SIMPLE]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario D: {config.description}[/bold blue]")
    console.print(f"  Reduced embed_dim: {config.embed_dim}, depth: {config.depth}")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("scenario-e")
def run_scenario_e(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario E: Tiny network + Simple data.

    Expected outcome: Test if minimal capacity network develops heavy tails on simple data.
    Uses embed_dim=32 (1/6 of ViT-Tiny) and depth=1 (single layer).
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.E_TINY_SIMPLE]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario E: {config.description}[/bold blue]")
    console.print(f"  Tiny embed_dim: {config.embed_dim}, depth: {config.depth}")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("scenario-f")
def run_scenario_f(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Run Scenario F: Tiny network + Complex data.

    Expected outcome: Extreme over-compression due to minimal capacity on complex data.
    Uses embed_dim=32 (1/6 of ViT-Tiny) and depth=1 (single layer) on PathMNIST.
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    config = SCENARIO_CONFIGS[ScenarioType.F_TINY_COMPLEX]
    config.seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print(f"\n[bold blue]Scenario F: {config.description}[/bold blue]")
    console.print(f"  Tiny embed_dim: {config.embed_dim}, depth: {config.depth}")
    console.print(f"  Seeds: {config.seeds}")
    console.print(f"  Device: {resolved_device}")
    console.print()

    results = []
    for seed in config.seeds:
        console.print(f"\n[cyan]Running seed {seed}...[/cyan]")
        result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
        results.append(result)

        if result.success:
            console.print(f"  ✓ Completed: Val Acc = {result.best_val_accuracy:.2f}%")
        else:
            console.print(f"  ✗ Failed: {result.error_message}")

    _print_scenario_summary(results)


@app.command("run-all")
def run_all_scenarios(
    num_seeds: int = typer.Option(3, "--num-seeds", "-n"),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """Run all six scenarios sequentially."""
    console.print(
        "\n[bold magenta]═══ Running All Spectral Analysis Scenarios ═══[/bold magenta]\n"
    )

    run_scenario_a(num_seeds=num_seeds, device=device, output_dir=output_dir)
    run_scenario_b(num_seeds=num_seeds, device=device, output_dir=output_dir)
    run_scenario_c(num_seeds=num_seeds, device=device, output_dir=output_dir)
    run_scenario_d(num_seeds=num_seeds, device=device, output_dir=output_dir)
    run_scenario_e(num_seeds=num_seeds, device=device, output_dir=output_dir)
    run_scenario_f(num_seeds=num_seeds, device=device, output_dir=output_dir)

    console.print("\n[bold green]All scenarios completed![/bold green]")
    console.print("View results with: poetry run mlflow ui --backend-store-uri mlruns/")


def _study_job(config: ScenarioConfig, seed: int, device_str: str, output_dir: str) -> dict:
    """One (cell, seed) run in a worker process; returns a picklable summary."""
    result = run_scenario_experiment(config, seed, torch.device(device_str), Path(output_dir))
    return {
        "tag": config.name,
        "seed": seed,
        "success": result.success,
        "test_accuracy": result.test_accuracy,
        "best_val_accuracy": result.best_val_accuracy,
        "final_accuracy": result.final_accuracy,
        "training_time": result.training_time,
        "error_message": result.error_message,
        "final_metrics": result.final_metrics,
    }


def _run_study_parallel(
    configs: list[ScenarioConfig],
    seeds: list[int],
    device: torch.device,
    output_dir: Path,
    workers: int,
) -> None:
    """Run every (cell, seed) job across ``workers`` spawned processes.

    MLflow's file store copes with concurrent writers as long as the *experiments* exist
    before the lanes start (two processes racing on ``set_experiment`` can create
    duplicates), so they are pre-created here. Each lane trains one small ViT; the GPU is
    shared, and the per-lane DataLoader workers are ``config.num_workers`` =
    max(1, min(4, cpu_count // workers)), so lanes share the cores without oversubscribing.
    A hard child crash (e.g. a CUDA abort) breaks the pool; the remaining jobs are then
    reported as failed and can be re-run with ``--cells``/``--seeds``.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mlflow.set_tracking_uri(str(output_dir))
    for config in configs:
        mlflow.set_experiment(f"spectral_{config.name}")

    jobs = [(config, seed) for config in configs for seed in seeds]
    per_cell: dict[str, list[ScenarioResult]] = {c.name: [] for c in configs}
    console.print(f"  [bold]{len(jobs)} jobs across {workers} lanes[/bold]\n")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {
            pool.submit(_study_job, config, seed, str(device), str(output_dir)): (config, seed)
            for config, seed in jobs
        }
        for fut in as_completed(futures):
            config, seed = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:  # a crashed lane must not sink the study
                r = {
                    "tag": config.name,
                    "seed": seed,
                    "success": False,
                    "test_accuracy": 0.0,
                    "best_val_accuracy": 0.0,
                    "final_accuracy": 0.0,
                    "training_time": 0.0,
                    "error_message": str(exc),
                    "final_metrics": {},
                }
            per_cell[config.name].append(
                ScenarioResult(
                    scenario=config.scenario,
                    seed=seed,
                    success=r["success"],
                    final_accuracy=r["final_accuracy"],
                    best_val_accuracy=r["best_val_accuracy"],
                    final_metrics=r["final_metrics"],
                    metrics_history={},
                    training_time=r["training_time"],
                    error_message=r["error_message"],
                    test_accuracy=r["test_accuracy"],
                )
            )
            status = (
                f"✓ test={r['test_accuracy']:.2f}% ({r['training_time']:.0f}s)"
                if r["success"]
                else f"✗ {r['error_message']}"
            )
            console.print(f"  [cyan]{config.name} seed {seed}[/cyan] {status}")

    for config in configs:
        console.print(f"\n[bold blue]{config.name}[/bold blue] — {config.description}")
        _print_scenario_summary(sorted(per_cell[config.name], key=lambda x: x.seed))


@app.command("run-study")
def run_study(
    tier: int = typer.Option(1, "--tier", "-t", help="1=width sweep, 2=+depth/datasets, 3=+A-F"),
    study_set: str = typer.Option(
        "tiers", "--set", help="tiers (uses --tier) | followup | followup-wide"
    ),
    num_seeds: int = typer.Option(10, "--num-seeds", "-n", help="Seeds per cell"),
    seeds_csv: str = typer.Option(
        "", "--seeds", help="Explicit comma-separated seeds (overrides --num-seeds)"
    ),
    cells_csv: str = typer.Option(
        "", "--cells", help="Comma-separated cell tags to run (subset of the set)"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w", help="Parallel (cell, seed) lanes as separate processes"
    ),
    loader_workers: int | None = typer.Option(
        None,
        "--loader-workers",
        help="DataLoader workers PER LANE (default: min(4, cpu_count // lanes)); cap this "
        "when sharing the machine with other jobs",
    ),
    device: DeviceChoice = typer.Option(DeviceChoice.AUTO, "--device"),
    output_dir: Path = typer.Option(None, "--output", "-o"),
    alignment: bool = typer.Option(True, "--alignment/--no-alignment"),
    truncation: bool = typer.Option(True, "--truncation/--no-truncation"),
    save_checkpoints: bool | None = typer.Option(
        None,
        "--save-checkpoints/--no-save-checkpoints",
        help="Log final weights per run (default: on for follow-up sets, off for tiers)",
    ),
) -> None:
    """Turnkey capacity × complexity study (EXPERIMENT_PLAN_V2). One command runs the
    whole set × seeds grid with gradient-alignment + tail-truncation on by default.

    Intended for a CUDA box, e.g.::

        vision-spectra spectral run-study --tier 1 --num-seeds 10 --device cuda
        vision-spectra spectral run-study --set followup --num-seeds 10 --workers 4 --device cuda
        vision-spectra spectral run-study --set followup --cells w192_synlong,w192_pathshort
    """
    resolved_output = output_dir or MLRUNS_DIR
    resolved_device = resolve_device(device)

    configs = build_study_set(study_set, tier)
    if cells_csv.strip():
        wanted = {c.strip() for c in cells_csv.split(",") if c.strip()}
        unknown = wanted - {c.name for c in configs}
        if unknown:
            raise typer.BadParameter(f"unknown cells {sorted(unknown)} for set {study_set!r}")
        configs = [c for c in configs if c.name in wanted]
    seeds = (
        [int(x) for x in seeds_csv.split(",") if x.strip()]
        if seeds_csv.strip()
        else [42 + i * 100 for i in range(num_seeds)]
    )
    if save_checkpoints is None:
        save_checkpoints = study_set != "tiers"
    workers = max(1, workers)
    # Per-lane DataLoader workers: share the machine's cores across lanes, capped at the
    # single-run default (4), never below 1. On a 32-core box with 4 lanes → 4 per lane.
    lane_loader_workers = (
        max(1, min(_num_workers_for(resolved_device), (os.cpu_count() or 4) // workers))
        if workers > 1
        else None
    )
    if loader_workers is not None:
        lane_loader_workers = max(0, loader_workers)

    label = f"Tier {tier}" if study_set == "tiers" else f"set {study_set}"
    console.print(f"\n[bold magenta]═══ Capacity × Complexity Study — {label} ═══[/bold magenta]")
    console.print(
        f"  Cells: {len(configs)}  ·  Seeds/cell: {len(seeds)}  ·  Total runs: "
        f"{len(configs) * len(seeds)}  ·  lanes: {workers}"
    )
    console.print(
        f"  Device: {resolved_device}  ·  alignment={alignment} truncation={truncation} "
        f"checkpoints={save_checkpoints}"
    )
    console.print(f"  Output: {resolved_output}\n")

    for config in configs:
        config.seeds = seeds
        config.run_alignment = alignment
        config.run_truncation = truncation
        config.save_checkpoint = save_checkpoints
        config.study_set = study_set if study_set != "tiers" else f"tier{tier}"
        if lane_loader_workers is not None:
            config.num_workers = lane_loader_workers if resolved_device.type == "cuda" else 0
    per_lane_loaders = (
        lane_loader_workers
        if lane_loader_workers is not None
        else _num_workers_for(resolved_device)
    )
    # Steady state: each lane = 1 trainer + persistent train-loader workers; the val/test
    # loaders spawn the same number of workers transiently while they iterate.
    steady = workers * (1 + per_lane_loaders)
    peak = workers * (1 + 2 * per_lane_loaders)
    console.print(
        f"  Loader workers/lane: {per_lane_loaders}  ·  ≈{steady} processes steady, "
        f"≈{peak} peak, on {os.cpu_count()} cores\n"
    )

    if workers > 1:
        _run_study_parallel(configs, seeds, resolved_device, resolved_output, workers)
    else:
        for cfg_idx, config in enumerate(configs, start=1):
            console.print(
                f"[bold blue][{cfg_idx}/{len(configs)}] {config.name}[/bold blue] "
                f"— {config.description}"
            )
            cell_results = []
            for seed in seeds:
                console.print(f"  [cyan]seed {seed}...[/cyan]")
                result = run_scenario_experiment(config, seed, resolved_device, resolved_output)
                cell_results.append(result)
                if result.success:
                    console.print(
                        f"    ✓ test={result.test_accuracy:.2f}% "
                        f"val={result.best_val_accuracy:.2f}% ({result.training_time:.1f}s)"
                    )
                else:
                    console.print(f"    ✗ failed: {result.error_message}")
            _print_scenario_summary(cell_results)

    console.print("\n[bold green]Study complete.[/bold green]")
    console.print("Generate figures with: poetry run vision-spectra figures all")


@app.command("compare")
def compare_scenarios(
    output_dir: Path = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Compare spectral metrics across scenarios.

    Loads results from MLflow and generates comparison tables/plots.
    """
    resolved_output = output_dir or MLRUNS_DIR
    mlflow.set_tracking_uri(str(resolved_output))

    console.print("\n[bold blue]Comparing Spectral Analysis Scenarios[/bold blue]\n")

    # Create comparison table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Description")
    table.add_column("Expected Tails")
    table.add_column("α_mean", justify="right")
    table.add_column("r_s_mean", justify="right")
    table.add_column("Accuracy", justify="right")

    for scenario_type, config in SCENARIO_CONFIGS.items():
        # Get experiment
        experiment = mlflow.get_experiment_by_name(f"spectral_scenario_{scenario_type.value}")

        if experiment is None:
            table.add_row(
                scenario_type.value,
                config.description[:40] + "...",
                "Yes" if scenario_type == ScenarioType.B_EXPRESSIVE_COMPLEX else "No",
                "—",
                "—",
                "—",
            )
            continue

        # Get runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
        )

        if runs.empty:
            table.add_row(
                scenario_type.value,
                config.description[:40] + "...",
                "Yes" if scenario_type == ScenarioType.B_EXPRESSIVE_COMPLEX else "No",
                "—",
                "—",
                "—",
            )
            continue

        # Compute mean metrics across runs
        alpha_cols = [c for c in runs.columns if "alpha_exponent_mean" in c]
        sr_cols = [c for c in runs.columns if "stable_rank_mean" in c]

        alpha_mean = runs[alpha_cols[-1]].mean() if alpha_cols else float("nan")
        sr_mean = runs[sr_cols[-1]].mean() if sr_cols else float("nan")
        acc_mean = (
            runs["metrics.final/val_accuracy"].mean()
            if "metrics.final/val_accuracy" in runs.columns
            else float("nan")
        )

        table.add_row(
            scenario_type.value,
            config.description[:40] + "...",
            "Yes" if scenario_type == ScenarioType.B_EXPRESSIVE_COMPLEX else "No",
            f"{alpha_mean:.2f}" if np.isfinite(alpha_mean) else "—",
            f"{sr_mean:.2f}" if np.isfinite(sr_mean) else "—",
            f"{acc_mean:.1f}%" if np.isfinite(acc_mean) else "—",
        )

    console.print(table)

    console.print("\n[bold]Interpretation Guide:[/bold]")
    console.print("  • Higher α_mean → heavier tails (more power-law decay)")
    console.print("  • Lower r_s_mean → lower effective rank (more compression)")
    console.print("  • Scenario B should show highest α and lowest r_s")


def _print_scenario_summary(results: list[ScenarioResult]) -> None:
    """Print summary of scenario results."""
    successful = [r for r in results if r.success]

    if not successful:
        console.print("\n[red]All runs failed![/red]")
        return

    console.print("\n[bold]Summary:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Seed")
    table.add_column("Test", justify="right")  # held-out test accuracy (the headline)
    table.add_column("Val", justify="right")  # model-selection diagnostic only
    table.add_column("α_Hill", justify="right")  # heavy-tail ESD index (lower=heavier)
    table.add_column("r_s", justify="right")  # stable rank (lower=more compressed)
    table.add_column("λ_decay", justify="right")  # rank-decay slope (higher=heavier)
    table.add_column("Time", justify="right")

    for r in successful:
        hill = r.final_metrics.get("pl_alpha_hill_mean", float("nan"))
        sr = r.final_metrics.get("stable_rank_mean", float("nan"))
        lam = r.final_metrics.get("alpha_exponent_mean", float("nan"))

        table.add_row(
            str(r.seed),
            f"{r.test_accuracy:.2f}%",
            f"{r.best_val_accuracy:.2f}%",
            f"{hill:.2f}" if np.isfinite(hill) else "—",
            f"{sr:.2f}" if np.isfinite(sr) else "—",
            f"{lam:.2f}" if np.isfinite(lam) else "—",
            f"{r.training_time:.1f}s",
        )

    console.print(table)


def cleanup_dataloaders(*loaders: Any) -> None:
    """
    Properly cleanup DataLoaders to release file descriptors.

    This is important on macOS which has a low default file descriptor limit.
    DataLoaders with num_workers > 0 spawn subprocesses that hold file descriptors.
    """
    import contextlib

    for loader in loaders:
        if loader is None:
            continue
        # Try to cleanup any iterator state
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            with contextlib.suppress(Exception):
                loader._iterator._shutdown_workers()
            loader._iterator = None

    # Force garbage collection to clean up worker processes
    gc.collect()


def cleanup_matplotlib() -> None:
    """Clean up matplotlib to release figure file handles."""
    plt.close("all")


if __name__ == "__main__":
    app()
