"""Figures for the capacity × complexity study (EXPERIMENT_PLAN_V2).

The A-F grid is handled by ``publication_figures``; this module reads the *sweep* cells
(``spectral_w*``, ``spectral_d*``, dataset cells) plus the new ``alignment/*`` and
``truncation/*`` metrics, and produces the study's headline figures:

  * width-sweep      — Δα_Hill and Δr_s vs network width, simple vs complex data
  * alignment        — cos(∇L, U Vᵀ) trajectory across capacity/complexity
  * truncation       — bulk & head accuracy-vs-retention curves

Metric conventions (see METHODOLOGY_REVIEW.md): heavy tail ⇒ α_Hill ↓ (band [2,6]),
r_s ↓; the rank-decay slope λ_decay (``alpha_exponent``) ↑ is reported as secondary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from loguru import logger

from vision_spectra.analysis.publication_figures import (
    OutputFormat,
    _ci95_halfwidth,
    get_mlflow_client,
    get_output_dir,
    save_figure,
    select_latest_run_per_seed,
)


@dataclass
class StudyCell:
    """One capacity×complexity cell aggregated over seeds."""

    experiment: str
    tag: str
    embed_dim: int
    depth: int
    dataset: str
    is_complex: bool
    num_runs: int
    accuracy_mean: float
    # spectral init→final means (+ per-seed deltas for CIs)
    delta_hill_mean: float
    delta_hill_values: list[float]
    hill_final_mean: float
    delta_sr_mean: float
    delta_sr_values: list[float]
    sr_final_mean: float
    delta_lambda_mean: float
    # trajectories aggregated across seeds (step -> mean)
    alignment_curve: dict[int, float] = field(default_factory=dict)
    truncation_curves: dict[str, dict[float, float]] = field(default_factory=dict)
    sr_init_mean: float = float("nan")
    hill_init_mean: float = float("nan")
    # subspace-resolved alignment (follow-up probe): cos_tail > 0 ⇔ step shrinks the tail
    alignment_tail_curve: dict[int, float] = field(default_factory=dict)
    alignment_head_curve: dict[int, float] = field(default_factory=dict)
    # head-drop probe: n dropped → test accuracy
    head_drop_curve: dict[int, float] = field(default_factory=dict)

    @property
    def rel_delta_sr_mean(self) -> float:
        """Δr_s / r_s(init): the width-comparable compression measure."""
        return (
            self.delta_sr_mean / self.sr_init_mean
            if np.isfinite(self.sr_init_mean) and self.sr_init_mean
            else float("nan")
        )

    @property
    def is_width_sweep_cell(self) -> bool:
        """``w{NNN}_syn`` / ``w{NNN}_path`` at depth 6 (excludes blood/derma, wd0, controls)."""
        import re

        return bool(re.fullmatch(r"w\d{3}_(syn|path)", self.tag)) and self.depth == 6


def _init_final_deltas(
    client, run_ids: list[str], key: str
) -> tuple[list[float], list[float], list[float]]:
    """Per-run (init, final, delta) for a stepped metric key."""
    inits, finals, deltas = [], [], []
    for rid in run_ids:
        try:
            hist = client.get_metric_history(rid, key)
        except Exception:
            hist = []
        if not hist:
            continue
        hist = sorted(hist, key=lambda x: x.step)
        if not np.isfinite(hist[0].value) or not np.isfinite(hist[-1].value):
            continue
        inits.append(hist[0].value)
        finals.append(hist[-1].value)
        deltas.append(hist[-1].value - hist[0].value)
    return inits, finals, deltas


def _mean_curve(client, run_ids: list[str], key: str) -> dict[int, float]:
    """Average a stepped metric across runs, keyed by step."""
    by_step: dict[int, list[float]] = defaultdict(list)
    for rid in run_ids:
        try:
            hist = client.get_metric_history(rid, key)
        except Exception:
            hist = []
        for m in hist:
            if np.isfinite(m.value):
                by_step[int(m.step)].append(m.value)
    return {s: float(np.mean(v)) for s, v in sorted(by_step.items()) if v}


def extract_study_cell(experiment_name: str) -> StudyCell | None:
    """Aggregate one ``spectral_{tag}`` experiment across its finished seeds."""
    client = get_mlflow_client()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], filter_string="status = 'FINISHED'"
    )
    if runs.empty:
        return None
    runs = select_latest_run_per_seed(runs)

    run_ids = list(runs["run_id"])

    def _param(col: str, default: str = "") -> str:
        c = f"params.{col}"
        if c in runs.columns and len(runs[c]) and runs[c].iloc[0] is not None:
            return str(runs[c].iloc[0])
        return default

    embed_dim = int(_param("embed_dim", "0") or 0)
    depth = int(_param("depth", "0") or 0)
    dataset = _param("dataset", "unknown")
    is_complex = dataset != "synthetic"

    acc_col = next(
        (
            c
            for c in ("metrics.final/test_accuracy", "metrics.final/val_accuracy")
            if c in runs.columns
        ),
        None,
    )
    accuracy_mean = float(runs[acc_col].mean()) if acc_col else float("nan")

    hill_inits, hill_finals, hill_deltas = _init_final_deltas(
        client, run_ids, "spectral/pl_alpha_hill_mean"
    )
    sr_inits, sr_finals, sr_deltas = _init_final_deltas(
        client, run_ids, "spectral/stable_rank_mean"
    )
    _, _, lam_deltas = _init_final_deltas(client, run_ids, "spectral/alpha_exponent_mean")

    # Accuracy averages over all finished runs; the spectral Δ values average only over runs
    # with a logged spectral history. Warn when they disagree so the accuracy n is not mistaken
    # for the n behind Δα_Hill / Δr_s (parity with extract_scenario_metrics).
    if len(hill_deltas) != len(run_ids):
        logger.warning(
            f"{experiment_name}: {len(run_ids)} finished runs but only {len(hill_deltas)} "
            "have spectral history; Δα_Hill / Δr_s use the smaller set."
        )

    tag = experiment_name.removeprefix("spectral_")
    return StudyCell(
        experiment=experiment_name,
        tag=tag,
        embed_dim=embed_dim,
        depth=depth,
        dataset=dataset,
        is_complex=is_complex,
        num_runs=len(run_ids),
        accuracy_mean=accuracy_mean,
        delta_hill_mean=float(np.mean(hill_deltas)) if hill_deltas else float("nan"),
        delta_hill_values=hill_deltas,
        hill_final_mean=float(np.mean(hill_finals)) if hill_finals else float("nan"),
        delta_sr_mean=float(np.mean(sr_deltas)) if sr_deltas else float("nan"),
        delta_sr_values=sr_deltas,
        sr_final_mean=float(np.mean(sr_finals)) if sr_finals else float("nan"),
        delta_lambda_mean=float(np.mean(lam_deltas)) if lam_deltas else float("nan"),
        alignment_curve=_mean_curve(client, run_ids, "alignment/cos_sim_mean"),
        sr_init_mean=float(np.mean(sr_inits)) if sr_inits else float("nan"),
        hill_init_mean=float(np.mean(hill_inits)) if hill_inits else float("nan"),
        alignment_tail_curve=_mean_curve(client, run_ids, "alignment/cos_tail_mean"),
        alignment_head_curve=_mean_curve(client, run_ids, "alignment/cos_head_mean"),
        head_drop_curve=_mean_curve(client, run_ids, "truncation/headn_acc"),
        truncation_curves={
            "bulk": {
                s / 100.0: v
                for s, v in _mean_curve(client, run_ids, "truncation/bulk_acc").items()
            },
            "head": {
                s / 100.0: v
                for s, v in _mean_curve(client, run_ids, "truncation/head_acc").items()
            },
        },
    )


def discover_study_cells() -> list[StudyCell]:
    """All ``spectral_*`` experiments except the A-F grid (handled elsewhere)."""
    client = get_mlflow_client()
    cells = []
    for exp in client.search_experiments():
        if not exp.name.startswith("spectral_") or exp.name.startswith("spectral_scenario_"):
            continue
        cell = extract_study_cell(exp.name)
        if cell is not None:
            cells.append(cell)
    return cells


# =============================================================================
# Figures
# =============================================================================


def generate_width_sweep(
    cells: list[StudyCell], output_dir: Path, fmt=OutputFormat.BOTH
) -> list[Path]:
    """Headline: Δα_Hill and Δr_s vs width (depth=6), simple vs complex data."""
    # Only the two-arm width sweep (w{NNN}_syn / w{NNN}_path, depth 6). Blood/Derma,
    # the wd0 and step-matched controls share the width prefix but are NOT sweep points.
    width_cells = [c for c in cells if c.is_width_sweep_cell and c.embed_dim > 0]
    if not width_cells:
        logger.warning("No width-sweep cells found; skipping width-sweep figure")
        return []
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    for is_complex, color, label in [
        (False, "#27ae60", "simple (synthetic)"),
        (True, "#c0392b", "complex (PathMNIST)"),
    ]:
        grp = sorted(
            [c for c in width_cells if c.is_complex == is_complex], key=lambda c: c.embed_dim
        )
        if not grp:
            continue
        xs = [c.embed_dim for c in grp]
        # 95 % t-interval over seeds (t_{0.975, n-1}·sd/√n; same as the A-F figures) for
        # Δα_Hill and for the per-seed relative Δr_s (Δr_s / cell-mean init). NaN when n ≤ 1
        # so matplotlib draws no bar rather than a misleading zero-length one.
        hill_ci = [_ci95_halfwidth(list(c.delta_hill_values)) for c in grp]
        rel_sr = [c.rel_delta_sr_mean for c in grp]
        rel_ci = [
            _ci95_halfwidth(list(np.asarray(c.delta_sr_values) / c.sr_init_mean))
            if np.isfinite(c.sr_init_mean) and c.sr_init_mean
            else np.nan
            for c in grp
        ]
        axL.errorbar(
            xs,
            [c.delta_hill_mean for c in grp],
            yerr=hill_ci,
            fmt="o-",
            color=color,
            label=label,
            capsize=3,
        )
        axR.errorbar(xs, rel_sr, yerr=rel_ci, fmt="o-", color=color, label=label, capsize=3)

    for ax in (axL, axR):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("embedding width (log scale)")
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.legend()
    axL.set_ylabel("Δα_Hill (init→final); <0 = heavier tail")
    axL.set_title("Heavy-tail emergence vs capacity")
    axR.set_ylabel("Δr_s / r_s(init); <0 = more compressed")
    axR.set_title("Relative stable-rank reduction vs capacity")
    fig.suptitle("Width sweep (depth=6): when do heavy tails emerge?", fontweight="bold")
    fig.tight_layout()
    return save_figure(fig, "study_width_sweep", output_dir, fmt)


def generate_alignment_trajectory(
    cells: list[StudyCell], output_dir: Path, fmt=OutputFormat.BOTH
) -> list[Path]:
    """cos(∇L, U Vᵀ) over training for the reference-width cells (simple vs complex)."""
    # Prefer the subspace-resolved probe (cos_tail); fall back to the legacy full-basis
    # cosine for stores that predate it (its range is bounded by 1/√rank, see module doc).
    use_tail = any(c.alignment_tail_curve for c in cells)
    ref = [c for c in cells if (c.alignment_tail_curve if use_tail else c.alignment_curve)]
    if not ref:
        logger.warning("No alignment curves found; skipping alignment figure")
        return []
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # the width sweep (both arms) first, then anything else, up to 8 cells
    show = sorted(ref, key=lambda c: (not c.is_width_sweep_cell, c.is_complex, c.embed_dim))[:8]
    for c in show:
        curve = c.alignment_tail_curve if use_tail else c.alignment_curve
        steps = sorted(curve)
        ax.plot(steps, [curve[s] for s in steps], "o-", ms=3, label=c.tag, alpha=0.85)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("epoch")
    if use_tail:
        ax.set_ylabel("cos(∇L, U_tail V_tailᵀ)   (>0 ⇒ step shrinks the tail)")
        ax.set_title(
            "Gradient alignment to the tail-shrinking (rank-reducing) direction", fontweight="bold"
        )
    else:
        ax.set_ylabel("cos(∇L, U Vᵀ)   (>0 ⇒ training reduces rank; |cos| ≤ 1/√rank)")
        ax.set_title("Gradient alignment to the rank-minimization flow", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return save_figure(fig, "study_alignment_trajectory", output_dir, fmt)


def generate_truncation_curves(
    cells: list[StudyCell], output_dir: Path, fmt=OutputFormat.BOTH
) -> list[Path]:
    """Bulk vs head accuracy-retention curves for representative cells."""
    with_tr = [c for c in cells if c.truncation_curves.get("bulk")]
    if not with_tr:
        logger.warning("No truncation curves found; skipping truncation figure")
        return []
    plt.style.use("seaborn-v0_8-whitegrid")
    has_head_drop = any(c.head_drop_curve for c in with_tr)
    ncols = 3 if has_head_drop else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5), sharey=True)
    axB, axH = axes[0], axes[1]
    # every width-sweep cell (both arms), then other cells, capped at 10 lines
    show = sorted(with_tr, key=lambda c: (not c.is_width_sweep_cell, c.is_complex, c.embed_dim))[
        :10
    ]
    for c in show:
        for ax, mode in ((axB, "bulk"), (axH, "head")):
            curve = c.truncation_curves.get(mode, {})
            xs = sorted(curve)
            if xs:
                ax.plot(xs, [curve[x] for x in xs], "o-", ms=3, label=c.tag, alpha=0.85)
        if has_head_drop and c.head_drop_curve:
            ns = sorted(c.head_drop_curve)
            axes[2].plot(
                ns, [c.head_drop_curve[n] for n in ns], "o-", ms=3, label=c.tag, alpha=0.85
            )
    axB.set_title("Bulk truncation (keep largest SVs)\n→ effective-rank probe")
    axH.set_title("Head truncation by ratio (drop top 10 %+)\n→ coarse signal-vs-noise probe")
    for ax in (axB, axH):
        ax.set_xlabel("retention ratio (fraction of SVs kept)")
        ax.legend(fontsize=8, ncol=2)
    if has_head_drop:
        axes[2].set_title("Head-drop (top-n SVs removed per matrix)\n→ fine signal-vs-noise probe")
        axes[2].set_xlabel("n largest singular values dropped")
        axes[2].legend(fontsize=8, ncol=2)
    axB.set_ylabel("test accuracy (%)")
    fig.suptitle(
        "Tail-truncation: do the heavy-tail directions carry the signal?", fontweight="bold"
    )
    fig.tight_layout()
    return save_figure(fig, "study_truncation_curves", output_dir, fmt)


def generate_study_figures(
    fmt: OutputFormat = OutputFormat.BOTH, output_dir: Path | None = None
) -> dict[str, list[Path]]:
    """Discover sweep cells and emit all study figures. Best-effort per figure."""
    cells = discover_study_cells()
    if not cells:
        logger.warning("No study (sweep) cells found in MLflow; run `spectral run-study` first.")
        return {}
    logger.info(f"Found {len(cells)} study cells: {sorted(c.tag for c in cells)}")
    out = output_dir or get_output_dir()
    saved: dict[str, list[Path]] = {}
    for name, gen in [
        ("width_sweep", generate_width_sweep),
        ("alignment", generate_alignment_trajectory),
        ("truncation", generate_truncation_curves),
    ]:
        try:
            saved[name] = gen(cells, out, fmt)
        except Exception as e:
            logger.warning(f"study figure '{name}' failed: {e}")
            saved[name] = []
    return saved
