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
import time
from dataclasses import dataclass
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
from vision_spectra.metrics.spectral import (
    aggregate_spectral_metrics,
    get_spectral_metrics,
)
from vision_spectra.settings import (
    DATA_DIR,
    MLRUNS_DIR,
    DatasetConfig,
    DatasetName,
    set_seed,
)

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
    """Configuration for a single experimental scenario."""

    scenario: ScenarioType
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
# Model Creation with Expressivity Control
# =============================================================================


def create_model_for_scenario(
    scenario_config: ScenarioConfig,
    device: torch.device,
) -> torch.nn.Module:
    """
    Create a model with the specified expressivity configuration.

    For Scenario C and D, we create a narrower/shallower model.
    """
    import timm

    if scenario_config.scenario in (
        ScenarioType.C_REDUCED_COMPLEX,
        ScenarioType.D_REDUCED_SIMPLE,
        ScenarioType.E_TINY_SIMPLE,
        ScenarioType.F_TINY_COMPLEX,
    ):
        # Create a custom narrow/shallow ViT
        # Using timm's flexibility to create custom configurations
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=scenario_config.num_classes,
            in_chans=3,
            img_size=28,
            embed_dim=scenario_config.embed_dim,  # Reduced width
            depth=scenario_config.depth,  # Reduced depth
            num_heads=max(1, scenario_config.embed_dim // 32),  # Adjust heads
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )
    else:
        # Standard ViT-Tiny for Scenarios A and B
        model = timm.create_model(
            scenario_config.model_name,
            pretrained=False,
            num_classes=scenario_config.num_classes,
            in_chans=3,
            img_size=28,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
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

    # Create experiment name
    experiment_name = f"spectral_scenario_{config.scenario.value}"

    try:
        # Setup MLflow
        mlflow.set_tracking_uri(str(output_dir))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"seed_{seed}"):
            # Log configuration
            mlflow.log_params(
                {
                    "scenario": config.scenario.value,
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
                }
            )

            # Create model
            logger.info(f"Creating model for scenario {config.scenario.value}")
            model = create_model_for_scenario(config, device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            mlflow.log_param("num_parameters", num_params)

            # Load dataset
            logger.info(f"Loading dataset: {config.dataset_name}")
            if config.dataset_name == "synthetic":
                from vision_spectra.data.synthetic import create_synthetic_dataset

                train_loader, val_loader, test_loader = create_synthetic_dataset(
                    num_classes=config.num_classes,
                    num_samples_train=config.num_samples or 1000,
                    num_samples_val=200,
                    num_samples_test=200,
                    batch_size=config.batch_size,
                    seed=seed,
                )
            else:
                # Use num_workers=0 to avoid file descriptor exhaustion on macOS
                # Spectral analysis creates many figures which can leak FDs
                dataset_config = DatasetConfig(
                    name=DatasetName(config.dataset_name),
                    batch_size=config.batch_size,
                    sample_ratio=1.0 if config.num_samples is None else 0.5,
                    num_workers=0,  # Disable multiprocessing to prevent FD leaks
                )
                dataset_obj = get_dataset(dataset_config, data_dir=DATA_DIR)
                train_loader = dataset_obj.get_train_loader()
                val_loader = dataset_obj.get_val_loader()
                # test_loader available via dataset_obj.get_test_loader() if needed

            # Setup loss and optimizer
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.05,
            )

            # Track metrics history
            metrics_history: dict[int, dict[str, float]] = {}

            # Log initial spectral metrics (epoch 0, before training)
            if 0 in config.log_epochs:
                logger.info("Logging initial spectral metrics (epoch 0)")
                analysis = extract_and_analyze_weights(model, device)
                metrics_history[0] = analysis["aggregated_metrics"]
                log_spectral_artifacts(analysis, epoch=0)

                for key, value in analysis["aggregated_metrics"].items():
                    if np.isfinite(value):
                        mlflow.log_metric(f"spectral/{key}", value, step=0)

            # Training loop
            best_val_accuracy = 0.0
            final_accuracy = 0.0

            for epoch in range(1, config.epochs + 1):
                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    if labels.dim() > 1:
                        labels = labels.squeeze()

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                train_accuracy = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)

                # Validation
                model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        if labels.dim() > 1:
                            labels = labels.squeeze()

                        outputs = model(images)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()

                val_accuracy = 100.0 * val_correct / val_total
                final_accuracy = val_accuracy

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy

                # Log training metrics
                mlflow.log_metrics(
                    {
                        "train/loss": avg_train_loss,
                        "train/accuracy": train_accuracy,
                        "val/accuracy": val_accuracy,
                    },
                    step=epoch,
                )

                # Log spectral metrics at specified epochs
                if epoch in config.log_epochs:
                    logger.info(f"Epoch {epoch}: Logging spectral metrics")
                    analysis = extract_and_analyze_weights(model, device)
                    metrics_history[epoch] = analysis["aggregated_metrics"]
                    log_spectral_artifacts(analysis, epoch=epoch)

                    for key, value in analysis["aggregated_metrics"].items():
                        if np.isfinite(value):
                            mlflow.log_metric(f"spectral/{key}", value, step=epoch)

                # Progress logging
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch}/{config.epochs}: "
                        f"Train Loss={avg_train_loss:.4f}, "
                        f"Train Acc={train_accuracy:.2f}%, "
                        f"Val Acc={val_accuracy:.2f}%"
                    )

            # Final spectral analysis
            final_epoch = config.epochs
            if final_epoch not in metrics_history:
                analysis = extract_and_analyze_weights(model, device)
                metrics_history[final_epoch] = analysis["aggregated_metrics"]
                log_spectral_artifacts(analysis, epoch=final_epoch)

            # Log final metrics
            mlflow.log_metrics(
                {
                    "final/val_accuracy": best_val_accuracy,
                    "final/train_accuracy": train_accuracy,
                }
            )

            training_time = time.time() - start_time

            # Comprehensive cleanup to prevent resource leaks
            # Clean up DataLoaders first (releases multiprocessing workers)
            cleanup_dataloaders(train_loader, val_loader)

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
            )

    except Exception as e:
        logger.error(f"Scenario {config.scenario.value} seed {seed} failed: {e}")
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
    """Resolve device choice to torch.device."""
    if device_choice == DeviceChoice.AUTO:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_choice.value)


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
    table.add_column("Accuracy", justify="right")
    table.add_column("α_mean", justify="right")
    table.add_column("r_s_mean", justify="right")
    table.add_column("Time", justify="right")

    for r in successful:
        alpha = r.final_metrics.get("alpha_exponent_mean", float("nan"))
        sr = r.final_metrics.get("stable_rank_mean", float("nan"))

        table.add_row(
            str(r.seed),
            f"{r.best_val_accuracy:.2f}%",
            f"{alpha:.2f}" if np.isfinite(alpha) else "—",
            f"{sr:.2f}" if np.isfinite(sr) else "—",
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
