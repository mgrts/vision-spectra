#!/usr/bin/env python3
"""
Publication Figures Generator.

This module generates publication-quality figures from MLflow experiment results.
All data is fetched dynamically from MLflow - no hardcoded values.

Usage:
    # Generate ALL outputs (figures, tables, summary, stats) - recommended
    poetry run python -m vision_spectra.analysis.publication_figures all

    # Generate only figures
    poetry run python -m vision_spectra.analysis.publication_figures generate

    # Generate specific figure type
    poetry run python -m vision_spectra.analysis.publication_figures generate --figure delta-alpha

    # Generate results table (PNG + LaTeX)
    poetry run python -m vision_spectra.analysis.publication_figures table

    # Export results summary to JSON
    poetry run python -m vision_spectra.analysis.publication_figures summary

    # Generate LaTeX table only
    poetry run python -m vision_spectra.analysis.publication_figures latex-table

    # Run statistical tests
    poetry run python -m vision_spectra.analysis.publication_figures stats
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table
from scipy import stats

from vision_spectra.settings import MLRUNS_DIR

# =============================================================================
# Custom JSON Encoder for numpy types
# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# CLI App
# =============================================================================

app = typer.Typer(
    name="publication-figures",
    help="Generate publication-quality figures from MLflow experiment results.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()


# =============================================================================
# Enums and Configuration
# =============================================================================


class FigureType(str, Enum):
    """Available figure types."""

    DELTA_ALPHA = "delta-alpha"
    ACCURACY_VS_COMPRESSION = "accuracy-compression"
    HEATMAP = "heatmap"
    STABLE_RANK = "stable-rank"
    ALL = "all"


class OutputFormat(str, Enum):
    """Output format for figures."""

    PNG = "png"
    PDF = "pdf"
    SVG = "svg"
    BOTH = "both"


@dataclass
class ScenarioMetrics:
    """Aggregated metrics for a scenario."""

    scenario: str
    name: str
    description: str
    accuracy_mean: float
    accuracy_std: float
    alpha_initial_mean: float
    alpha_final_mean: float
    delta_alpha_mean: float
    delta_alpha_std: float
    delta_alpha_values: list[float]
    stable_rank_initial_mean: float
    stable_rank_final_mean: float
    num_runs: int


SCENARIO_METADATA: dict[str, dict[str, str]] = {
    "A": {"name": "Expressive+Simple", "description": "Large network on simple synthetic data"},
    "B": {"name": "Expressive+Complex", "description": "Large network on complex PathMNIST data"},
    "C": {"name": "Reduced+Complex", "description": "Reduced network on complex data"},
    "D": {"name": "Reduced+Simple", "description": "Reduced network on simple data"},
    "E": {"name": "Tiny+Simple", "description": "Minimal network on simple data"},
    "F": {"name": "Tiny+Complex", "description": "Minimal network on complex data"},
}

SCENARIO_COLORS: dict[str, str] = {
    "A": "#2ecc71",
    "B": "#3498db",
    "C": "#e74c3c",
    "D": "#27ae60",
    "E": "#9b59b6",
    "F": "#c0392b",
}


# =============================================================================
# MLflow Data Extraction
# =============================================================================


def get_mlflow_client() -> mlflow.MlflowClient:
    """Get MLflow client configured with project tracking URI."""
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    return mlflow.MlflowClient()


def extract_scenario_metrics(scenario: str) -> ScenarioMetrics | None:
    """Extract metrics for a scenario from MLflow.

    Uses get_metric_history to properly extract initial (epoch 0) and final
    (last epoch) values for spectral metrics.
    """
    experiment_name = f"spectral_scenario_{scenario}"
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    client = mlflow.MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
    )

    if runs.empty:
        logger.warning(f"No finished runs for experiment '{experiment_name}'")
        return None

    # Get accuracy from final/val_accuracy or val/accuracy
    accuracy_col = None
    for col in ["metrics.final/val_accuracy", "metrics.val/accuracy"]:
        if col in runs.columns:
            accuracy_col = col
            break

    accuracy_mean = runs[accuracy_col].mean() if accuracy_col else np.nan
    accuracy_std = runs[accuracy_col].std() if accuracy_col else np.nan

    # Extract metric histories from each run
    alpha_initial_values = []
    alpha_final_values = []
    sr_initial_values = []
    sr_final_values = []
    delta_alpha_values = []

    for _, row in runs.iterrows():
        run_id = row["run_id"]

        # Get alpha_exponent_mean history
        try:
            alpha_history = client.get_metric_history(run_id, "spectral/alpha_exponent_mean")
            if alpha_history:
                # Sort by step to get proper ordering
                alpha_history = sorted(alpha_history, key=lambda x: x.step)
                alpha_init = alpha_history[0].value
                alpha_final = alpha_history[-1].value
                alpha_initial_values.append(alpha_init)
                alpha_final_values.append(alpha_final)
                delta_alpha_values.append(alpha_final - alpha_init)
                logger.debug(
                    f"Run {run_id}: alpha {alpha_init:.4f} -> {alpha_final:.4f} "
                    f"(delta={alpha_final - alpha_init:.4f}, {len(alpha_history)} entries)"
                )
            else:
                logger.warning(f"Run {run_id}: No alpha history found")
        except Exception as e:
            logger.warning(f"Could not get alpha history for run {run_id}: {e}")

        # Get stable_rank_mean history
        try:
            sr_history = client.get_metric_history(run_id, "spectral/stable_rank_mean")
            if sr_history:
                # Sort by step to get proper ordering
                sr_history = sorted(sr_history, key=lambda x: x.step)
                sr_initial_values.append(sr_history[0].value)
                sr_final_values.append(sr_history[-1].value)
                logger.debug(
                    f"Run {run_id}: SR {sr_history[0].value:.2f} -> {sr_history[-1].value:.2f} "
                    f"({len(sr_history)} entries)"
                )
            else:
                logger.warning(f"Run {run_id}: No stable rank history found")
        except Exception as e:
            logger.warning(f"Could not get stable rank history for run {run_id}: {e}")

    # Compute aggregated values
    alpha_initial_mean = float(np.mean(alpha_initial_values)) if alpha_initial_values else np.nan
    alpha_final_mean = float(np.mean(alpha_final_values)) if alpha_final_values else np.nan
    delta_alpha_mean = float(np.mean(delta_alpha_values)) if delta_alpha_values else np.nan
    delta_alpha_std = float(np.std(delta_alpha_values)) if delta_alpha_values else np.nan
    sr_initial_mean = float(np.mean(sr_initial_values)) if sr_initial_values else np.nan
    sr_final_mean = float(np.mean(sr_final_values)) if sr_final_values else np.nan

    metadata = SCENARIO_METADATA.get(scenario, {"name": scenario, "description": ""})

    return ScenarioMetrics(
        scenario=scenario,
        name=metadata["name"],
        description=metadata["description"],
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        alpha_initial_mean=alpha_initial_mean,
        alpha_final_mean=alpha_final_mean,
        delta_alpha_mean=delta_alpha_mean,
        delta_alpha_std=delta_alpha_std,
        delta_alpha_values=delta_alpha_values,
        stable_rank_initial_mean=sr_initial_mean,
        stable_rank_final_mean=sr_final_mean,
        num_runs=len(runs),
    )


def extract_all_scenarios() -> dict[str, ScenarioMetrics]:
    """Extract metrics for all scenarios from MLflow."""
    results = {}
    for scenario in SCENARIO_METADATA:
        metrics = extract_scenario_metrics(scenario)
        if metrics is not None:
            results[scenario] = metrics
    return results


# =============================================================================
# Figure Generation
# =============================================================================


def get_output_dir() -> Path:
    """Get output directory for figures (references/figures)."""
    output_dir = Path(__file__).parent.parent.parent / "references" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
    dpi: int = 300,
) -> list[Path]:
    """Save figure in specified format(s)."""
    saved = []

    if fmt in (OutputFormat.PNG, OutputFormat.BOTH):
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(path)
        logger.info(f"Saved {path}")

    if fmt in (OutputFormat.PDF, OutputFormat.BOTH):
        path = output_dir / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
        logger.info(f"Saved {path}")

    if fmt == OutputFormat.SVG:
        path = output_dir / f"{name}.svg"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
        logger.info(f"Saved {path}")

    plt.close(fig)
    return saved


def generate_delta_alpha_bar(
    metrics: dict[str, ScenarioMetrics],
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
) -> list[Path]:
    """Generate delta alpha bar chart."""
    plt.style.use("seaborn-v0_8-whitegrid")

    scenarios = sorted(metrics.keys())
    delta_alphas = [metrics[s].delta_alpha_mean for s in scenarios]
    delta_stds = [metrics[s].delta_alpha_std for s in scenarios]
    colors = [SCENARIO_COLORS.get(s, "#333333") for s in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        scenarios,
        delta_alphas,
        yerr=delta_stds,
        color=colors,
        edgecolor="black",
        capsize=5,
    )

    for bar, val, _std in zip(bars, delta_alphas, delta_stds, strict=False):
        if np.isfinite(val):
            ax.annotate(
                f"+{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontweight="bold",
                fontsize=10,
            )

    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Heavy-tail threshold")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Δα (Delta Alpha)", fontsize=12)
    ax.set_title("Spectral Compression Across Experimental Scenarios", fontsize=14)
    ax.legend(loc="upper right")

    ax2 = ax.secondary_xaxis("top")
    ax2.set_xticks(range(len(scenarios)))
    ax2.set_xticklabels([metrics[s].name for s in scenarios], fontsize=8)

    return save_figure(fig, "delta_alpha_bar", output_dir, fmt)


def generate_accuracy_vs_compression(
    metrics: dict[str, ScenarioMetrics],
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
) -> list[Path]:
    """Generate accuracy vs compression scatter plot."""
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 8))

    for scenario, m in metrics.items():
        if np.isfinite(m.delta_alpha_mean) and np.isfinite(m.accuracy_mean):
            ax.scatter(
                m.delta_alpha_mean,
                m.accuracy_mean,
                s=200,
                c=SCENARIO_COLORS.get(scenario, "#333333"),
                edgecolors="black",
                linewidth=2,
                label=f"{scenario}: {m.name}",
            )
            ax.annotate(
                scenario,
                (m.delta_alpha_mean, m.accuracy_mean),
                xytext=(8, 8),
                textcoords="offset points",
                fontweight="bold",
                fontsize=12,
            )

    ax.axvline(x=0.3, color="red", linestyle="--", alpha=0.7, label="Heavy-tail threshold")
    ax.set_xlabel("Δα (Spectral Compression)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Spectral Compression", fontsize=14)
    ax.legend(loc="best", fontsize=9)

    return save_figure(fig, "accuracy_vs_compression", output_dir, fmt)


def generate_heatmap(
    metrics: dict[str, ScenarioMetrics],
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
) -> list[Path]:
    """Generate capacity x complexity heatmap."""
    plt.style.use("seaborn-v0_8-whitegrid")

    grid_layout = [["A", "B"], ["D", "C"], ["E", "F"]]

    data = np.zeros((3, 2))
    labels: list[list[str | None]] = [[None, None], [None, None], [None, None]]

    for i, row in enumerate(grid_layout):
        for j, scenario in enumerate(row):
            if scenario in metrics:
                val = metrics[scenario].delta_alpha_mean
                data[i, j] = val if np.isfinite(val) else 0
                labels[i][j] = f"{scenario}\n+{val:.3f}" if np.isfinite(val) else scenario
            else:
                labels[i][j] = f"{scenario}\n—"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    for i in range(3):
        for j in range(2):
            text_color = "white" if data[i, j] > 0.2 else "black"
            ax.text(
                j,
                i,
                labels[i][j],
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=11,
            )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Simple Data", "Complex Data"], fontsize=11)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["High Capacity", "Medium Capacity", "Low Capacity"], fontsize=11)
    ax.set_title("Capacity × Complexity → Spectral Compression", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δα", fontsize=11)

    return save_figure(fig, "capacity_complexity_heatmap", output_dir, fmt)


def generate_stable_rank(
    metrics: dict[str, ScenarioMetrics],
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
) -> list[Path]:
    """Generate stable rank comparison chart."""
    plt.style.use("seaborn-v0_8-whitegrid")

    scenarios = sorted(metrics.keys())
    x = np.arange(len(scenarios))
    width = 0.35

    sr_initial = [metrics[s].stable_rank_initial_mean for s in scenarios]
    sr_final = [metrics[s].stable_rank_final_mean for s in scenarios]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, sr_initial, width, label="Initial (Epoch 0)", color="#3498db")
    ax.bar(x + width / 2, sr_final, width, label="Final", color="#e74c3c")

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Stable Rank", fontsize=12)
    ax.set_title("Stable Rank Reduction During Training", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()

    for i, (init, final) in enumerate(zip(sr_initial, sr_final, strict=False)):
        if np.isfinite(init) and np.isfinite(final) and init > 0:
            reduction = (init - final) / init * 100
            ax.annotate(
                f"-{reduction:.0f}%",
                xy=(i, max(init, final) + 2),
                ha="center",
                fontsize=9,
                color="#555555",
            )

    return save_figure(fig, "stable_rank_comparison", output_dir, fmt)


# =============================================================================
# Statistical Analysis
# =============================================================================


def perform_statistical_tests(metrics: dict[str, ScenarioMetrics]) -> list[dict]:
    """Perform pairwise statistical tests between scenarios."""
    results = []

    test_pairs = [
        ("A", "B"),
        ("D", "C"),
        ("E", "F"),
        ("B", "C"),
        ("C", "F"),
        ("A", "F"),
    ]

    for s1, s2 in test_pairs:
        if s1 not in metrics or s2 not in metrics:
            continue

        vals1 = metrics[s1].delta_alpha_values
        vals2 = metrics[s2].delta_alpha_values

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(vals1, vals2)
        diff = np.mean(vals2) - np.mean(vals1)
        significant = bool(p_value < 0.05)

        interpretation = "No significant difference"
        if significant and diff > 0:
            interpretation = f"{s2} has significantly higher compression"
        elif significant and diff < 0:
            interpretation = f"{s1} has significantly higher compression"

        results.append(
            {
                "comparison": f"{s1} vs {s2}",
                "mean_diff": float(diff),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": significant,
                "interpretation": interpretation,
            }
        )

    return results


# =============================================================================
# CLI Commands
# =============================================================================


@app.command()
def generate(
    figure: FigureType = typer.Option(
        FigureType.ALL,
        "--figure",
        "-f",
        help="Type of figure to generate",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.BOTH,
        "--format",
        "-o",
        help="Output format for figures",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-d",
        help="Custom output directory (default: references/figures)",
    ),
) -> None:
    """Generate publication figures from MLflow experiment results."""
    console.print("[bold]Extracting metrics from MLflow...[/bold]")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found in MLflow![/red]")
        console.print("Run experiments first with:")
        console.print(
            "  poetry run python -m vision_spectra.experiments.run_spectral_analysis run-all"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Found data for {len(metrics)} scenarios: {', '.join(metrics.keys())}[/green]"
    )

    out_dir = output_dir or get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []

    if figure in (FigureType.ALL, FigureType.DELTA_ALPHA):
        console.print("\n[cyan]Generating delta alpha bar chart...[/cyan]")
        saved_files.extend(generate_delta_alpha_bar(metrics, out_dir, output_format))

    if figure in (FigureType.ALL, FigureType.ACCURACY_VS_COMPRESSION):
        console.print("\n[cyan]Generating accuracy vs compression plot...[/cyan]")
        saved_files.extend(generate_accuracy_vs_compression(metrics, out_dir, output_format))

    if figure in (FigureType.ALL, FigureType.HEATMAP):
        console.print("\n[cyan]Generating capacity-complexity heatmap...[/cyan]")
        saved_files.extend(generate_heatmap(metrics, out_dir, output_format))

    if figure in (FigureType.ALL, FigureType.STABLE_RANK):
        console.print("\n[cyan]Generating stable rank comparison...[/cyan]")
        saved_files.extend(generate_stable_rank(metrics, out_dir, output_format))

    console.print(f"\n[bold green]Generated {len(saved_files)} files in {out_dir}[/bold green]")


@app.command()
def summary(
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file (default: references/figures/results_summary.json)",
    ),
) -> None:
    """Export results summary to JSON."""
    console.print("[bold]Extracting metrics from MLflow...[/bold]")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found![/red]")
        raise typer.Exit(1)

    summary_data: dict = {
        "scenarios": {},
        "statistical_tests": perform_statistical_tests(metrics),
    }

    for scenario, m in metrics.items():
        summary_data["scenarios"][scenario] = {
            "name": m.name,
            "description": m.description,
            "num_runs": m.num_runs,
            "accuracy": {
                "mean": float(m.accuracy_mean) if np.isfinite(m.accuracy_mean) else None,
                "std": float(m.accuracy_std) if np.isfinite(m.accuracy_std) else None,
            },
            "alpha": {
                "initial_mean": (
                    float(m.alpha_initial_mean) if np.isfinite(m.alpha_initial_mean) else None
                ),
                "final_mean": (
                    float(m.alpha_final_mean) if np.isfinite(m.alpha_final_mean) else None
                ),
            },
            "delta_alpha": {
                "mean": (float(m.delta_alpha_mean) if np.isfinite(m.delta_alpha_mean) else None),
                "std": (float(m.delta_alpha_std) if np.isfinite(m.delta_alpha_std) else None),
                "values": [float(v) for v in m.delta_alpha_values],
            },
            "stable_rank": {
                "initial_mean": (
                    float(m.stable_rank_initial_mean)
                    if np.isfinite(m.stable_rank_initial_mean)
                    else None
                ),
                "final_mean": (
                    float(m.stable_rank_final_mean)
                    if np.isfinite(m.stable_rank_final_mean)
                    else None
                ),
            },
        }

    out_path = output_file or (get_output_dir() / "results_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)

    console.print(f"[green]Saved summary to {out_path}[/green]")

    table = Table(
        title="Experiment Results Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("Scenario", style="cyan")
    table.add_column("Name")
    table.add_column("Runs", justify="right")
    table.add_column("Acc", justify="right")
    table.add_column("α_init", justify="right")
    table.add_column("α_final", justify="right")
    table.add_column("Δα", justify="right")
    table.add_column("r_s (init→final)", justify="right")

    for scenario in sorted(metrics.keys()):
        m = metrics[scenario]
        acc_str = f"{m.accuracy_mean:.1f}%" if np.isfinite(m.accuracy_mean) else "—"
        alpha_i_str = f"{m.alpha_initial_mean:.3f}" if np.isfinite(m.alpha_initial_mean) else "—"
        alpha_f_str = f"{m.alpha_final_mean:.3f}" if np.isfinite(m.alpha_final_mean) else "—"
        da_str = f"+{m.delta_alpha_mean:.3f}" if np.isfinite(m.delta_alpha_mean) else "—"
        sr_str = (
            f"{m.stable_rank_initial_mean:.1f}→{m.stable_rank_final_mean:.1f}"
            if np.isfinite(m.stable_rank_initial_mean) and np.isfinite(m.stable_rank_final_mean)
            else "—"
        )

        table.add_row(
            scenario, m.name, str(m.num_runs), acc_str, alpha_i_str, alpha_f_str, da_str, sr_str
        )

    console.print(table)


def generate_table_image(
    metrics: dict[str, ScenarioMetrics],
    output_dir: Path,
    fmt: OutputFormat = OutputFormat.BOTH,
) -> list[Path]:
    """Generate results table as PNG/PDF image."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Prepare table data
    scenarios = sorted(metrics.keys())
    columns = [
        "Scenario",
        "Configuration",
        "Acc (%)",
        "α (init)",
        "α (final)",
        "Δα",
        "r_s (init)",
        "r_s (final)",
    ]
    cell_data = []

    for scenario in scenarios:
        m = metrics[scenario]
        acc = f"{m.accuracy_mean:.1f}" if np.isfinite(m.accuracy_mean) else "—"
        alpha_i = f"{m.alpha_initial_mean:.3f}" if np.isfinite(m.alpha_initial_mean) else "—"
        alpha_f = f"{m.alpha_final_mean:.3f}" if np.isfinite(m.alpha_final_mean) else "—"
        da = f"+{m.delta_alpha_mean:.3f}" if np.isfinite(m.delta_alpha_mean) else "—"
        sr_i = (
            f"{m.stable_rank_initial_mean:.1f}" if np.isfinite(m.stable_rank_initial_mean) else "—"
        )
        sr_f = f"{m.stable_rank_final_mean:.1f}" if np.isfinite(m.stable_rank_final_mean) else "—"
        cell_data.append([scenario, m.name, acc, alpha_i, alpha_f, da, sr_i, sr_f])

    # Create figure with table
    fig, ax = plt.subplots(figsize=(14, 3 + len(scenarios) * 0.5))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=cell_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        colColours=["#4472C4"] * len(columns),
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header cells
    for j in range(len(columns)):
        table[(0, j)].set_text_props(weight="bold", color="white")
        table[(0, j)].set_facecolor("#4472C4")

    # Alternate row colors
    for i in range(1, len(scenarios) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#D6DCE5")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")

    ax.set_title(
        "Experimental Results: Spectral Compression Across Scenarios",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    return save_figure(fig, "results_table", output_dir, fmt)


@app.command("table")
def table_cmd(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-d",
        help="Output directory (default: references/figures)",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.BOTH,
        "--format",
        "-o",
        help="Output format for table image",
    ),
    include_tex: bool = typer.Option(
        True,
        "--tex/--no-tex",
        help="Also generate LaTeX table",
    ),
) -> None:
    """Generate results table as image (PNG/PDF) and optionally LaTeX."""
    console.print("[bold]Extracting metrics from MLflow...[/bold]")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found![/red]")
        raise typer.Exit(1)

    out_dir = output_dir or get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate table image
    console.print("\n[cyan]Generating results table image...[/cyan]")
    saved_files = generate_table_image(metrics, out_dir, output_format)

    # Generate LaTeX if requested
    if include_tex:
        console.print("\n[cyan]Generating LaTeX table...[/cyan]")
        tex = _generate_latex_table_content(metrics)
        tex_path = out_dir / "results_table.tex"
        with open(tex_path, "w") as f:
            f.write(tex)
        saved_files.append(tex_path)
        logger.info(f"Saved {tex_path}")

    console.print(f"\n[bold green]Generated {len(saved_files)} files in {out_dir}[/bold green]")


def _generate_latex_table_content(metrics: dict[str, ScenarioMetrics]) -> str:
    """Generate LaTeX table content."""
    tex = r"""\begin{table}[htbp]
\centering
\caption{Experimental Results: Spectral Compression Across Scenarios}
\label{tab:results}
\begin{tabular}{llccccccc}
\toprule
Scenario & Configuration & Acc (\%) & $\alpha_{init}$ & $\alpha_{final}$ & $\Delta\alpha$ & $r_s^{init}$ & $r_s^{final}$ \\
\midrule
"""

    for scenario in sorted(metrics.keys()):
        m = metrics[scenario]
        acc = f"{m.accuracy_mean:.1f}" if np.isfinite(m.accuracy_mean) else "—"
        alpha_i = f"{m.alpha_initial_mean:.3f}" if np.isfinite(m.alpha_initial_mean) else "—"
        alpha_f = f"{m.alpha_final_mean:.3f}" if np.isfinite(m.alpha_final_mean) else "—"
        da = f"+{m.delta_alpha_mean:.3f}" if np.isfinite(m.delta_alpha_mean) else "—"
        sr_i = (
            f"{m.stable_rank_initial_mean:.1f}" if np.isfinite(m.stable_rank_initial_mean) else "—"
        )
        sr_f = f"{m.stable_rank_final_mean:.1f}" if np.isfinite(m.stable_rank_final_mean) else "—"

        tex += f"{scenario} & {m.name} & {acc} & {alpha_i} & {alpha_f} & {da} & {sr_i} & {sr_f} \\\\\n"

    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return tex


@app.command("latex-table")
def latex_table(
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output TeX file (default: references/figures/results_table.tex)",
    ),
) -> None:
    """Generate LaTeX table from results."""
    console.print("[bold]Extracting metrics from MLflow...[/bold]")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found![/red]")
        raise typer.Exit(1)

    tex = _generate_latex_table_content(metrics)

    out_path = output_file or (get_output_dir() / "results_table.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(tex)

    console.print(f"[green]Saved LaTeX table to {out_path}[/green]")
    console.print("\n[bold]Preview:[/bold]")
    console.print(tex)


@app.command("stats")
def statistical_tests_cmd(
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSON file for statistical tests",
    ),
) -> None:
    """Perform and display statistical tests between scenarios."""
    console.print("[bold]Extracting metrics from MLflow...[/bold]")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found![/red]")
        raise typer.Exit(1)

    results = perform_statistical_tests(metrics)

    if not results:
        console.print(
            "[yellow]Not enough data for statistical tests (need ≥2 runs per scenario)[/yellow]"
        )
        raise typer.Exit(0)

    table = Table(
        title="Statistical Tests (t-test)", show_header=True, header_style="bold magenta"
    )
    table.add_column("Comparison", style="cyan")
    table.add_column("Mean Diff", justify="right")
    table.add_column("t-stat", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Significant", justify="center")
    table.add_column("Interpretation")

    for r in results:
        sig_style = "green" if r["significant"] else "dim"
        table.add_row(
            r["comparison"],
            f"{r['mean_diff']:+.3f}",
            f"{r['t_statistic']:.2f}",
            f"{r['p_value']:.4f}",
            "✓" if r["significant"] else "✗",
            r["interpretation"],
            style=sig_style,
        )

    console.print(table)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        console.print(f"\n[green]Saved to {output_file}[/green]")


@app.command("all")
def generate_all(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-d",
        help="Output directory (default: references/figures)",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.BOTH,
        "--format",
        "-o",
        help="Output format for figures",
    ),
) -> None:
    """Generate all publication outputs: figures, tables, summary, and stats."""
    console.print("[bold blue]═══ Generating All Publication Outputs ═══[/bold blue]\n")

    metrics = extract_all_scenarios()

    if not metrics:
        console.print("[red]No experiment data found in MLflow![/red]")
        console.print("Run experiments first with:")
        console.print(
            "  poetry run python -m vision_spectra.experiments.run_spectral_analysis run-all"
        )
        raise typer.Exit(1)

    console.print(
        f"[green]Found data for {len(metrics)} scenarios: {', '.join(sorted(metrics.keys()))}[/green]\n"
    )

    out_dir = output_dir or get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_saved_files: list[Path] = []

    # 1. Generate all figures
    console.print("[bold]1. Generating Figures[/bold]")
    console.print("   [cyan]• Delta alpha bar chart...[/cyan]")
    all_saved_files.extend(generate_delta_alpha_bar(metrics, out_dir, output_format))

    console.print("   [cyan]• Accuracy vs compression plot...[/cyan]")
    all_saved_files.extend(generate_accuracy_vs_compression(metrics, out_dir, output_format))

    console.print("   [cyan]• Capacity-complexity heatmap...[/cyan]")
    all_saved_files.extend(generate_heatmap(metrics, out_dir, output_format))

    console.print("   [cyan]• Stable rank comparison...[/cyan]")
    all_saved_files.extend(generate_stable_rank(metrics, out_dir, output_format))

    # 2. Generate table (image + LaTeX)
    console.print("\n[bold]2. Generating Results Table[/bold]")
    console.print("   [cyan]• Table image (PNG/PDF)...[/cyan]")
    all_saved_files.extend(generate_table_image(metrics, out_dir, output_format))

    console.print("   [cyan]• LaTeX table...[/cyan]")
    tex = _generate_latex_table_content(metrics)
    tex_path = out_dir / "results_table.tex"
    with open(tex_path, "w") as f:
        f.write(tex)
    all_saved_files.append(tex_path)
    logger.info(f"Saved {tex_path}")

    # 3. Generate summary JSON
    console.print("\n[bold]3. Generating Summary[/bold]")
    console.print("   [cyan]• Results summary JSON...[/cyan]")
    summary_data: dict = {
        "scenarios": {},
        "statistical_tests": perform_statistical_tests(metrics),
    }

    for scenario, m in metrics.items():
        summary_data["scenarios"][scenario] = {
            "name": m.name,
            "description": m.description,
            "num_runs": m.num_runs,
            "accuracy": {
                "mean": float(m.accuracy_mean) if np.isfinite(m.accuracy_mean) else None,
                "std": float(m.accuracy_std) if np.isfinite(m.accuracy_std) else None,
            },
            "alpha": {
                "initial_mean": float(m.alpha_initial_mean)
                if np.isfinite(m.alpha_initial_mean)
                else None,
                "final_mean": float(m.alpha_final_mean)
                if np.isfinite(m.alpha_final_mean)
                else None,
            },
            "delta_alpha": {
                "mean": float(m.delta_alpha_mean) if np.isfinite(m.delta_alpha_mean) else None,
                "std": float(m.delta_alpha_std) if np.isfinite(m.delta_alpha_std) else None,
                "values": [float(v) for v in m.delta_alpha_values],
            },
            "stable_rank": {
                "initial_mean": float(m.stable_rank_initial_mean)
                if np.isfinite(m.stable_rank_initial_mean)
                else None,
                "final_mean": float(m.stable_rank_final_mean)
                if np.isfinite(m.stable_rank_final_mean)
                else None,
            },
        }

    summary_path = out_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    all_saved_files.append(summary_path)
    logger.info(f"Saved {summary_path}")

    # 4. Generate statistical tests JSON
    console.print("   [cyan]• Statistical tests JSON...[/cyan]")
    stats_results = perform_statistical_tests(metrics)
    if stats_results:
        stats_path = out_dir / "statistical_tests.json"
        with open(stats_path, "w") as f:
            json.dump(stats_results, f, indent=2, cls=NumpyEncoder)
        all_saved_files.append(stats_path)
        logger.info(f"Saved {stats_path}")

    # Summary
    console.print("\n[bold blue]═══ Summary ═══[/bold blue]")
    console.print(
        f"[bold green]✓ Generated {len(all_saved_files)} files in {out_dir}[/bold green]\n"
    )

    # List generated files by type
    png_files = [f for f in all_saved_files if f.suffix == ".png"]
    pdf_files = [f for f in all_saved_files if f.suffix == ".pdf"]
    json_files = [f for f in all_saved_files if f.suffix == ".json"]
    tex_files = [f for f in all_saved_files if f.suffix == ".tex"]

    if png_files:
        console.print(f"  PNG files: {len(png_files)}")
    if pdf_files:
        console.print(f"  PDF files: {len(pdf_files)}")
    if json_files:
        console.print(f"  JSON files: {len(json_files)}")
    if tex_files:
        console.print(f"  TeX files: {len(tex_files)}")

    # Display results table
    console.print("\n[bold]Results Overview:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Scenario", style="cyan")
    table.add_column("Name")
    table.add_column("Acc", justify="right")
    table.add_column("α_init", justify="right")
    table.add_column("α_final", justify="right")
    table.add_column("Δα", justify="right")
    table.add_column("r_s (init→final)", justify="right")

    for scenario in sorted(metrics.keys()):
        m = metrics[scenario]
        acc_str = f"{m.accuracy_mean:.1f}%" if np.isfinite(m.accuracy_mean) else "—"
        alpha_i_str = f"{m.alpha_initial_mean:.3f}" if np.isfinite(m.alpha_initial_mean) else "—"
        alpha_f_str = f"{m.alpha_final_mean:.3f}" if np.isfinite(m.alpha_final_mean) else "—"
        da_str = f"+{m.delta_alpha_mean:.3f}" if np.isfinite(m.delta_alpha_mean) else "—"
        sr_str = (
            f"{m.stable_rank_initial_mean:.1f}→{m.stable_rank_final_mean:.1f}"
            if np.isfinite(m.stable_rank_initial_mean) and np.isfinite(m.stable_rank_final_mean)
            else "—"
        )
        table.add_row(scenario, m.name, acc_str, alpha_i_str, alpha_f_str, da_str, sr_str)

    console.print(table)


if __name__ == "__main__":
    app()
