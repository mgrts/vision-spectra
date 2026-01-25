"""
Publication-quality spectral plots for analysis and visualization.

This module provides plotting functions for:
- CCDF (Complementary Cumulative Distribution Function) plots
- Log-log rank plots
- Spectral evolution over epochs
- Layer-wise heatmaps
- Comparison across scenarios
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def setup_plot_style() -> None:
    """Set up matplotlib style for publication-quality figures."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "lines.linewidth": 2,
        }
    )


def plot_ccdf(
    singular_values: np.ndarray,
    title: str = "CCDF of Singular Values",
    ax: Any | None = None,
    log_scale: bool = True,
    color: str = "blue",
    label: str | None = None,
    fit_power_law: bool = True,
) -> Any:
    """
    Plot Complementary Cumulative Distribution Function of singular values.

    Args:
        singular_values: Array of singular values
        title: Plot title
        ax: Matplotlib axes (created if None)
        log_scale: Whether to use log-log scale
        color: Line color
        label: Legend label
        fit_power_law: Whether to overlay power-law fit line

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        setup_plot_style()

    sv_sorted = np.sort(singular_values)[::-1]
    n = len(sv_sorted)
    ccdf = np.arange(1, n + 1) / n

    if log_scale:
        ax.loglog(sv_sorted, ccdf, "o-", color=color, markersize=3, label=label, alpha=0.8)
    else:
        ax.plot(sv_sorted, ccdf, "o-", color=color, markersize=3, label=label, alpha=0.8)

    if fit_power_law and len(sv_sorted) > 10:
        start = int(0.1 * n)
        end = int(0.7 * n)
        if end > start + 5:
            log_x = np.log(sv_sorted[start:end])
            log_y = np.log(ccdf[start:end])
            try:
                slope, intercept = np.polyfit(log_x, log_y, 1)
                x_fit = np.linspace(sv_sorted[end], sv_sorted[start], 50)
                y_fit = np.exp(intercept) * x_fit**slope
                ax.loglog(
                    x_fit, y_fit, "--", color="red", alpha=0.7, label=f"Power law (α≈{-slope:.2f})"
                )
            except Exception:
                pass

    ax.set_xlabel("Singular Value (σ)")
    ax.set_ylabel("P(σ' > σ)")
    ax.set_title(title)
    if label or fit_power_law:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_loglog_rank(
    singular_values: np.ndarray,
    title: str = "Log-Log Rank Plot",
    ax: Any | None = None,
    color: str = "blue",
    label: str | None = None,
    show_fit: bool = True,
) -> Any:
    """
    Plot singular values vs rank in log-log scale.

    Args:
        singular_values: Array of singular values
        title: Plot title
        ax: Matplotlib axes
        color: Line color
        label: Legend label
        show_fit: Whether to show linear fit

    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        setup_plot_style()

    sv_sorted = np.sort(singular_values)[::-1]
    ranks = np.arange(1, len(sv_sorted) + 1)

    ax.loglog(ranks, sv_sorted, "o-", color=color, markersize=3, label=label, alpha=0.8)

    if show_fit and len(sv_sorted) > 10:
        n = len(sv_sorted)
        start = int(0.1 * n)
        end = int(0.6 * n)
        if end > start + 5:
            log_x = np.log(ranks[start:end])
            log_y = np.log(sv_sorted[start:end])
            try:
                slope, intercept = np.polyfit(log_x, log_y, 1)
                x_fit = ranks[start:end]
                y_fit = np.exp(intercept) * x_fit**slope
                ax.loglog(
                    x_fit, y_fit, "--", color="red", alpha=0.7, label=f"Fit (α≈{-slope:.2f})"
                )
            except Exception:
                pass

    ax.set_xlabel("Rank")
    ax.set_ylabel("Singular Value (σ)")
    ax.set_title(title)
    if label or show_fit:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_spectral_evolution(
    epochs: list[int],
    metrics: dict[str, list[float]],
    title: str = "Spectral Metrics Evolution",
    save_path: Path | None = None,
) -> Any:
    """
    Plot evolution of spectral metrics over training epochs.

    Args:
        epochs: List of epoch numbers
        metrics: Dictionary mapping metric names to lists of values
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i, (name, values) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(epochs, values, "o-", color=colors[i % len(colors)], linewidth=2, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_layer_heatmap(
    layer_names: list[str],
    epochs: list[int],
    values: np.ndarray,
    metric_name: str = "stable_rank",
    title: str | None = None,
    save_path: Path | None = None,
    cmap: str = "viridis",
) -> Any:
    """
    Plot heatmap of spectral metric across layers and epochs.

    Args:
        layer_names: List of layer names (y-axis)
        epochs: List of epoch numbers (x-axis)
        values: 2D array of shape (n_layers, n_epochs)
        metric_name: Name of the metric being plotted
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap name

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(12, max(6, len(layer_names) * 0.3)))
    im = ax.imshow(values, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    ax.set_yticks(range(len(layer_names)))
    short_names = [name.replace("encoder.", "").replace(".weight", "") for name in layer_names]
    ax.set_yticklabels(short_names, fontsize=8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Layer")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_name.replace("_", " ").title())

    if title is None:
        title = f"{metric_name.replace('_', ' ').title()} Across Layers and Epochs"
    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_scenario_comparison(
    scenarios: dict[str, dict[str, float]],
    metrics: list[str],
    title: str = "Scenario Comparison",
    save_path: Path | None = None,
) -> Any:
    """
    Plot bar chart comparing scenarios across multiple metrics.

    Args:
        scenarios: Dictionary mapping scenario names to metric dictionaries
        metrics: List of metric names to compare
        title: Plot title
        save_path: Path to save figure

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    scenario_names = list(scenarios.keys())
    colors = plt.cm.Set2.colors

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [scenarios[s].get(metric, 0) for s in scenario_names]
        x = range(len(scenario_names))

        bars = ax.bar(x, values, color=[colors[j % len(colors)] for j in range(len(values))])
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())

        for bar, val in zip(bars, values, strict=False):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_sv_distribution_comparison(
    distributions: dict[str, np.ndarray],
    title: str = "Singular Value Distribution Comparison",
    save_path: Path | None = None,
    plot_type: str = "ccdf",
) -> Any:
    """
    Compare singular value distributions across multiple scenarios/layers.

    Args:
        distributions: Dictionary mapping names to singular value arrays
        title: Plot title
        save_path: Path to save figure
        plot_type: "ccdf", "loglog", or "histogram"

    Returns:
        Matplotlib figure object
    """
    import matplotlib.pyplot as plt

    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, (name, sv) in enumerate(distributions.items()):
        color = colors[i % len(colors)]

        if plot_type == "ccdf":
            plot_ccdf(sv, ax=ax, color=color, label=name, fit_power_law=False)
        elif plot_type == "loglog":
            plot_loglog_rank(sv, ax=ax, color=color, label=name, show_fit=False)
        elif plot_type == "histogram":
            ax.hist(
                np.log10(sv + 1e-10),
                bins=30,
                alpha=0.5,
                color=color,
                label=name,
                edgecolor="black",
            )
            ax.set_xlabel("log₁₀(σ)")
            ax.set_ylabel("Count")

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def generate_spectral_report(
    tracker_data: dict[str, Any],
    output_dir: Path,
    scenario_name: str = "experiment",
) -> None:
    """
    Generate a comprehensive spectral analysis report with multiple plots.

    Args:
        tracker_data: Data from SpectralTracker.to_dict()
        output_dir: Directory to save plots
        scenario_name: Name for the scenario/experiment
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = tracker_data.get("history", [])
    if not history:
        return

    epochs = [h["epoch"] for h in history]

    # Spectral evolution plot
    metrics_to_plot = {}
    for metric in ["stable_rank_mean", "alpha_exponent_mean", "spectral_entropy_mean"]:
        values = [h["aggregated_metrics"].get(metric, np.nan) for h in history]
        if any(np.isfinite(v) for v in values):
            metrics_to_plot[metric] = values

    if metrics_to_plot:
        plot_spectral_evolution(
            epochs,
            metrics_to_plot,
            title=f"{scenario_name}: Spectral Evolution",
            save_path=output_dir / "spectral_evolution.png",
        )
        plt.close()

    # Layer-wise heatmaps
    layer_names = []
    if history[0].get("distributions"):
        layer_names = [d["name"] for d in history[0]["distributions"]]

    if layer_names:
        for metric in ["stable_rank", "alpha_exponent"]:
            values = np.zeros((len(layer_names), len(epochs)))
            for j, h in enumerate(history):
                for i, dist in enumerate(h.get("distributions", [])):
                    if dist.get("metrics"):
                        values[i, j] = dist["metrics"].get(metric, np.nan)

            if np.any(np.isfinite(values)):
                plot_layer_heatmap(
                    layer_names,
                    epochs,
                    values,
                    metric_name=metric,
                    title=f"{scenario_name}: {metric.replace('_', ' ').title()} by Layer",
                    save_path=output_dir / f"layer_heatmap_{metric}.png",
                )
                plt.close()

    # Initial vs Final comparison
    if len(history) >= 2:
        first_dist = history[0].get("distributions", [])
        last_dist = history[-1].get("distributions", [])

        if first_dist and last_dist:
            initial_svs = np.concatenate(
                [np.array(d["singular_values"]) for d in first_dist if d.get("singular_values")]
            )
            final_svs = np.concatenate(
                [np.array(d["singular_values"]) for d in last_dist if d.get("singular_values")]
            )

            if len(initial_svs) > 0 and len(final_svs) > 0:
                plot_sv_distribution_comparison(
                    {
                        f"Epoch {history[0]['epoch']}": initial_svs,
                        f"Epoch {history[-1]['epoch']}": final_svs,
                    },
                    title=f"{scenario_name}: Initial vs Final Spectrum",
                    save_path=output_dir / "initial_vs_final_ccdf.png",
                    plot_type="ccdf",
                )
                plt.close()
