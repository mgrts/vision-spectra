#!/usr/bin/env python3
"""
Synthetic data classification experiments for spectral analysis.

This module runs systematic experiments on synthetic geometric shapes data
to test hypotheses about spectral behavior on simple, easily-learnable data.

Hypothesis: Simple synthetic data leads to less heavy-tailed weight spectra
because models can learn the patterns quickly without complex internal
representations, resulting in more uniform singular value distributions.

Key features:
- Multiple loss functions comparison on simple synthetic data
- Configurable data complexity (number of shapes, samples)
- Detailed spectral tracking to observe learning dynamics
- Comparison of spectral evolution between simple vs complex data

Usage:
    poetry run python -m vision_spectra.experiments.run_synthetic_experiments run

    # With custom settings:
    poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
        --num-classes 3 --num-samples 1000 --epochs 30 --num-seeds 3

    # Compare complexity levels:
    poetry run python -m vision_spectra.experiments.run_synthetic_experiments compare-complexity
"""

from __future__ import annotations

import contextlib
import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import mlflow
import torch
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from vision_spectra.data import get_dataset
from vision_spectra.losses import get_loss
from vision_spectra.models import create_vit_classifier
from vision_spectra.settings import (
    DATA_DIR,
    MLRUNS_DIR,
    DatasetConfig,
    DatasetName,
    ExperimentConfig,
    LossConfig,
    LossName,
    ModelConfig,
    OptimizerConfig,
    SpectralConfig,
    TrainingConfig,
    set_seed,
)
from vision_spectra.training import ClassificationTrainer

# =============================================================================
# CLI App
# =============================================================================

app = typer.Typer(
    name="synthetic-experiments",
    help="Run experiments on synthetic geometric shapes data for spectral analysis.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console()


# =============================================================================
# Enums for CLI
# =============================================================================


class DeviceChoice(str, Enum):
    """Device options for training."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class LossChoice(str, Enum):
    """Available loss functions."""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"
    CLASS_BALANCED = "class_balanced"
    ASYMMETRIC = "asymmetric"


class ComplexityLevel(str, Enum):
    """Data complexity levels for comparison experiments."""

    TRIVIAL = "trivial"  # 2 classes, 500 samples
    SIMPLE = "simple"  # 3 classes, 1000 samples
    MEDIUM = "medium"  # 5 classes, 5000 samples


# Default loss functions
DEFAULT_LOSSES: list[str] = [
    "cross_entropy",
    "focal",
    "label_smoothing",
]

# Default seeds
DEFAULT_SEEDS: list[int] = [42, 123, 456]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntheticExperimentConfig:
    """Configuration for synthetic data experiments."""

    # Synthetic data settings
    num_classes: int = 3
    num_samples_train: int = 1000
    num_samples_val: int = 200
    num_samples_test: int = 200
    image_size: int = 28

    # Loss functions to compare
    losses: list[str] = field(default_factory=lambda: DEFAULT_LOSSES)

    # Seeds for multiple runs
    seeds: list[int] = field(default_factory=lambda: DEFAULT_SEEDS)

    # Training settings
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10

    # Device settings
    device: str = "auto"

    # Spectral logging settings - more frequent for synthetic data
    log_every_n_epochs: int = 2  # Log more frequently to observe fast convergence
    log_first_epochs: bool = True
    track_distributions: bool = True
    save_distribution_history: bool = True

    # Model settings - use smaller model for simple data
    model_name: str = "vit_tiny_patch16_224"

    # Output
    output_dir: Path = field(default_factory=lambda: MLRUNS_DIR)
    experiment_name: str = "synthetic_spectral_analysis"


@dataclass
class SyntheticExperimentResult:
    """Result of a single synthetic experiment run."""

    num_classes: int
    num_samples: int
    loss: str
    seed: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_f1: float
    best_val_auroc: float
    final_epoch: int
    training_time_seconds: float
    convergence_epoch: int  # Epoch where val loss stabilized
    checkpoint_path: str | None
    mlflow_run_id: str | None
    spectral_metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None


# =============================================================================
# Experiment Functions
# =============================================================================


def run_single_synthetic_experiment(
    config: SyntheticExperimentConfig,
    loss_name: str,
    seed: int,
) -> SyntheticExperimentResult:
    """
    Run a single synthetic data classification experiment.

    Args:
        config: Experiment configuration
        loss_name: Name of the loss function
        seed: Random seed

    Returns:
        SyntheticExperimentResult with metrics and spectral data
    """
    experiment_id = f"synthetic_{config.num_classes}cls_{loss_name}_seed{seed}"
    logger.info(f"Starting experiment: {experiment_id}")

    start_time = time.time()

    # Variables for cleanup
    trainer = None
    model = None
    train_loader = None
    val_loader = None
    dataset_obj = None

    try:
        # Set seed for reproducibility
        set_seed(seed)

        # Create experiment configuration
        exp_config = ExperimentConfig(
            name=experiment_id,
            seed=seed,
            device=config.device,
            output_dir=config.output_dir,
            data_dir=DATA_DIR,
            dataset=DatasetConfig(
                name=DatasetName.SYNTHETIC,
                batch_size=config.batch_size,
                image_size=config.image_size,
                num_classes=config.num_classes,
                num_samples_train=config.num_samples_train,
                num_samples_val=config.num_samples_val,
                num_samples_test=config.num_samples_test,
            ),
            model=ModelConfig(
                name=config.model_name,
            ),
            loss=LossConfig(
                classification=LossName(loss_name),
                label_smoothing=0.1,
                focal_gamma=2.0,
                class_balanced_beta=0.9999,
            ),
            optimizer=OptimizerConfig(
                learning_rate=config.learning_rate,
                warmup_epochs=3,  # Shorter warmup for simpler data
            ),
            training=TrainingConfig(
                epochs=config.epochs,
                early_stopping=True,
                patience=config.early_stopping_patience,
                save_every_n_epochs=5,
            ),
            spectral=SpectralConfig(
                enabled=True,
                log_every_n_epochs=config.log_every_n_epochs,
                log_first_epochs=config.log_first_epochs,
                track_distributions=config.track_distributions,
                save_distribution_history=config.save_distribution_history,
            ),
        )

        # Load synthetic dataset
        dataset_obj = get_dataset(exp_config.dataset, exp_config.data_dir)
        train_loader = dataset_obj.get_train_loader()
        val_loader = dataset_obj.get_val_loader()
        info = dataset_obj.get_info()

        logger.info(
            f"Synthetic dataset: {info.num_classes} classes (shapes), "
            f"{info.train_size} train samples"
        )
        logger.info(f"Shapes: {info.class_names}")

        # Create model
        model = create_vit_classifier(
            exp_config.model,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
            image_size=info.image_size[0],
        )

        # Create loss function
        class_counts = info.class_counts.get("train") if info.class_counts else None
        criterion = get_loss(exp_config.loss, samples_per_class=class_counts)

        # Create trainer
        trainer = ClassificationTrainer(
            config=exp_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            num_classes=info.num_classes,
            num_channels=info.num_channels,
            class_names=info.class_names,
        )

        # Train and track convergence
        result = trainer.train()

        training_time = time.time() - start_time

        # Get final metrics
        val_metrics = trainer.validate()

        # Estimate convergence epoch (when improvement stopped)
        convergence_epoch = result.get("final_epoch", config.epochs)
        if result.get("stopped_early", False):
            convergence_epoch = max(
                0, result.get("final_epoch", 0) - config.early_stopping_patience
            )

        # Get MLflow run ID
        mlflow_run_id = None
        with contextlib.suppress(Exception):
            mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        experiment_result = SyntheticExperimentResult(
            num_classes=config.num_classes,
            num_samples=config.num_samples_train,
            loss=loss_name,
            seed=seed,
            best_val_loss=result.get("best_val_metric", float("inf")),
            best_val_accuracy=val_metrics.get("accuracy", 0.0),
            best_val_f1=val_metrics.get("f1_macro", 0.0),
            best_val_auroc=val_metrics.get("auroc", 0.0),
            final_epoch=result.get("final_epoch", 0),
            convergence_epoch=convergence_epoch,
            training_time_seconds=training_time,
            checkpoint_path=str(result.get("best_checkpoint"))
            if result.get("best_checkpoint")
            else None,
            mlflow_run_id=mlflow_run_id,
            success=True,
        )

    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        import traceback

        traceback.print_exc()

        experiment_result = SyntheticExperimentResult(
            num_classes=config.num_classes,
            num_samples=config.num_samples_train,
            loss=loss_name,
            seed=seed,
            best_val_loss=float("inf"),
            best_val_accuracy=0.0,
            best_val_f1=0.0,
            best_val_auroc=0.0,
            final_epoch=0,
            convergence_epoch=0,
            training_time_seconds=time.time() - start_time,
            checkpoint_path=None,
            mlflow_run_id=None,
            success=False,
            error_message=str(e),
        )

    finally:
        # Clean up resources
        logger.debug("Cleaning up experiment resources...")

        if trainer is not None:
            with contextlib.suppress(Exception):
                trainer.cleanup()
            del trainer
            trainer = None

        if model is not None:
            with contextlib.suppress(Exception):
                model.cpu()
            del model
            model = None

        if train_loader is not None:
            del train_loader
            train_loader = None
        if val_loader is not None:
            del val_loader
            val_loader = None

        if dataset_obj is not None:
            del dataset_obj
            dataset_obj = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.debug("Experiment cleanup complete")

    return experiment_result


def run_all_synthetic_experiments(
    config: SyntheticExperimentConfig,
) -> list[SyntheticExperimentResult]:
    """
    Run all synthetic experiments for all loss functions and seeds.

    Args:
        config: Experiment configuration

    Returns:
        List of all experiment results
    """
    results: list[SyntheticExperimentResult] = []

    total_experiments = len(config.losses) * len(config.seeds)
    current_experiment = 0

    logger.info(f"Starting {total_experiments} synthetic data experiments")
    logger.info(f"Complexity: {config.num_classes} classes, {config.num_samples_train} samples")
    logger.info(f"Losses: {config.losses}")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(f"Max epochs: {config.epochs}, patience: {config.early_stopping_patience}")
    logger.info(f"Device: {config.device}")

    for loss_name in config.losses:
        for seed in config.seeds:
            current_experiment += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {current_experiment}/{total_experiments}")
            logger.info(f"Loss: {loss_name}, Seed: {seed}")
            logger.info(f"{'='*60}")

            result = run_single_synthetic_experiment(
                config=config,
                loss_name=loss_name,
                seed=seed,
            )
            results.append(result)

            # Log result summary
            if result.success:
                logger.success(
                    f"Completed: AUROC={result.best_val_auroc:.4f}, "
                    f"Acc={result.best_val_accuracy:.4f}, "
                    f"Converged@epoch={result.convergence_epoch}, "
                    f"Time={result.training_time_seconds:.1f}s"
                )
            else:
                logger.error(f"Failed: {result.error_message}")

    return results


def save_synthetic_results(results: list[SyntheticExperimentResult], output_path: Path) -> None:
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = []
    for r in results:
        results_dict.append(
            {
                "num_classes": r.num_classes,
                "num_samples": r.num_samples,
                "loss": r.loss,
                "seed": r.seed,
                "best_val_loss": r.best_val_loss,
                "best_val_accuracy": r.best_val_accuracy,
                "best_val_f1": r.best_val_f1,
                "best_val_auroc": r.best_val_auroc,
                "final_epoch": r.final_epoch,
                "convergence_epoch": r.convergence_epoch,
                "training_time_seconds": r.training_time_seconds,
                "checkpoint_path": r.checkpoint_path,
                "mlflow_run_id": r.mlflow_run_id,
                "success": r.success,
                "error_message": r.error_message,
            }
        )

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def print_synthetic_summary(results: list[SyntheticExperimentResult]) -> None:
    """Print a summary of synthetic experiment results."""
    from collections import defaultdict

    import numpy as np

    console.print("\n")
    console.rule("[bold blue]SYNTHETIC DATA EXPERIMENT SUMMARY[/bold blue]")

    # Group by loss function
    by_loss: dict[str, list[SyntheticExperimentResult]] = defaultdict(list)
    for r in results:
        if r.success:
            by_loss[r.loss].append(r)

    # Create summary table
    table = Table(
        title=f"Results: {results[0].num_classes} classes, {results[0].num_samples} samples",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Loss", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right")
    table.add_column("AUROC", justify="right")
    table.add_column("Convergence Epoch", justify="right")
    table.add_column("Runs", justify="center")

    for loss_name in sorted(by_loss.keys()):
        loss_results = by_loss[loss_name]

        if loss_results:
            accs = [r.best_val_accuracy for r in loss_results]
            aurocs = [r.best_val_auroc for r in loss_results]
            conv_epochs = [r.convergence_epoch for r in loss_results]

            acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
            auroc_mean, auroc_std = float(np.mean(aurocs)), float(np.std(aurocs))
            conv_mean = float(np.mean(conv_epochs))

            table.add_row(
                loss_name,
                f"{acc_mean:.4f} ± {acc_std:.4f}",
                f"{auroc_mean:.4f} ± {auroc_std:.4f}",
                f"{conv_mean:.1f}",
                str(len(loss_results)),
            )

    console.print(table)

    # Key observations for hypothesis testing
    if by_loss:
        all_results = [r for r in results if r.success]
        if all_results:
            avg_conv = float(np.mean([r.convergence_epoch for r in all_results]))
            avg_acc = float(np.mean([r.best_val_accuracy for r in all_results]))

            console.print("\n[bold]Key Observations:[/bold]")
            console.print(f"  • Average convergence epoch: [cyan]{avg_conv:.1f}[/cyan]")
            console.print(f"  • Average accuracy: [cyan]{avg_acc:.4f}[/cyan]")

            if avg_conv < 10 and avg_acc > 0.9:
                console.print(
                    "\n  [green]✓ Fast convergence + high accuracy suggests simple data[/green]"
                )
                console.print(
                    "  [green]  → Expect less heavy-tailed spectra (more uniform SVD)[/green]"
                )
            elif avg_conv >= 10:
                console.print(
                    "\n  [yellow]• Slower convergence may indicate model complexity[/yellow]"
                )

    # Failed experiments
    failed = [r for r in results if not r.success]
    if failed:
        console.print(f"\n[red]Failed: {len(failed)} experiments[/red]")

    console.rule()


# =============================================================================
# CLI Commands
# =============================================================================


@app.command("run")
def run_synthetic(
    num_classes: int = typer.Option(
        3,
        "--num-classes",
        "-c",
        min=2,
        max=5,
        help="Number of shape classes (2-5). Fewer = simpler data.",
    ),
    num_samples: int = typer.Option(
        1000,
        "--num-samples",
        "-n",
        min=100,
        help="Number of training samples. Fewer = simpler learning task.",
    ),
    losses: list[str] | None = typer.Option(
        None,
        "--losses",
        "-l",
        help="Loss functions to compare. Default: cross_entropy, focal, label_smoothing.",
    ),
    seeds: list[int] | None = typer.Option(
        None,
        "--seeds",
        "-s",
        help="Specific seeds for reproducibility.",
    ),
    num_seeds: int = typer.Option(
        3,
        "--num-seeds",
        help="Number of seeds if --seeds not specified.",
    ),
    epochs: int = typer.Option(
        30,
        "--epochs",
        "-e",
        help="Maximum training epochs.",
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        "-p",
        help="Early stopping patience.",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Training batch size.",
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate.",
    ),
    device: DeviceChoice = typer.Option(
        DeviceChoice.AUTO,
        "--device",
        help="Device for training.",
    ),
    log_every_n_epochs: int = typer.Option(
        2,
        "--log-every-n-epochs",
        help="Log spectral metrics every N epochs (more frequent for fast convergence).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to mlruns/.",
    ),
) -> None:
    """
    Run classification experiments on synthetic geometric shapes data.

    This experiments on simple synthetic data to test the hypothesis that
    simpler data leads to less heavy-tailed weight spectra due to easier
    learning and faster convergence.

    Examples:
        # Simple experiment with 3 classes
        poetry run python -m vision_spectra.experiments.run_synthetic_experiments run

        # Very simple: 2 classes, 500 samples
        poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
            --num-classes 2 --num-samples 500 --epochs 20

        # More complex synthetic data
        poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
            --num-classes 5 --num-samples 5000 --epochs 50
    """
    # Resolve settings
    resolved_output_dir = output_dir if output_dir is not None else MLRUNS_DIR
    resolved_losses = list(losses) if losses else DEFAULT_LOSSES

    if seeds:
        resolved_seeds = list(seeds)
    elif num_seeds != 3:
        resolved_seeds = [42 + i * 100 for i in range(num_seeds)]
    else:
        resolved_seeds = DEFAULT_SEEDS

    # Calculate val/test sizes (20% each of train)
    num_val = max(100, num_samples // 5)
    num_test = max(100, num_samples // 5)

    # Create config
    config = SyntheticExperimentConfig(
        num_classes=num_classes,
        num_samples_train=num_samples,
        num_samples_val=num_val,
        num_samples_test=num_test,
        losses=resolved_losses,
        seeds=resolved_seeds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        early_stopping_patience=patience,
        device=device.value,
        log_every_n_epochs=log_every_n_epochs,
        output_dir=resolved_output_dir,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    console.print("\n[bold blue]Synthetic Data Experiments[/bold blue]")
    console.print(
        f"  Complexity: [cyan]{config.num_classes} classes, {num_samples} samples[/cyan]"
    )
    console.print(f"  Losses: [cyan]{', '.join(config.losses)}[/cyan]")
    console.print(f"  Seeds: [cyan]{config.seeds}[/cyan]")
    console.print(f"  Device: [cyan]{config.device}[/cyan]")
    console.print(f"  Max epochs: [cyan]{epochs}[/cyan] (patience: {patience})")
    console.print(f"  Output: [cyan]{config.output_dir}[/cyan]")
    console.print()

    # Run experiments
    logger.info("Starting synthetic data experiments")
    results = run_all_synthetic_experiments(config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = config.output_dir / f"synthetic_results_{num_classes}cls_{timestamp}.json"
    save_synthetic_results(results, results_path)

    # Print summary
    print_synthetic_summary(results)

    # MLflow instructions
    console.print("\n[bold]To analyze spectral evolution:[/bold]")
    console.print(f"  cd {MLRUNS_DIR.parent}")
    console.print("  poetry run mlflow ui")
    console.print("\nLook for spectral artifacts in each run to compare distributions.")


@app.command("compare-complexity")
def compare_complexity(
    device: DeviceChoice = typer.Option(
        DeviceChoice.AUTO,
        "--device",
        help="Device for training.",
    ),
    num_seeds: int = typer.Option(
        2,
        "--num-seeds",
        help="Seeds per complexity level.",
    ),
    epochs: int = typer.Option(
        30,
        "--epochs",
        help="Max epochs per experiment.",
    ),
) -> None:
    """
    Compare spectral behavior across different data complexity levels.

    Runs experiments at three complexity levels:
    - TRIVIAL: 2 classes, 500 samples
    - SIMPLE: 3 classes, 1000 samples
    - MEDIUM: 5 classes, 5000 samples

    This helps validate the hypothesis that simpler data leads to
    less heavy-tailed spectra.
    """
    complexity_configs = {
        ComplexityLevel.TRIVIAL: {"num_classes": 2, "num_samples": 500},
        ComplexityLevel.SIMPLE: {"num_classes": 3, "num_samples": 1000},
        ComplexityLevel.MEDIUM: {"num_classes": 5, "num_samples": 5000},
    }

    all_results: dict[str, list[SyntheticExperimentResult]] = {}
    seeds = [42 + i * 100 for i in range(num_seeds)]

    console.print("\n[bold blue]Comparing Complexity Levels[/bold blue]")
    console.print(f"  Levels: {list(complexity_configs.keys())}")
    console.print(f"  Seeds per level: {num_seeds}")
    console.print(f"  Device: {device.value}")
    console.print()

    for level, params in complexity_configs.items():
        console.print(f"\n[bold]Running {level.value.upper()} experiments...[/bold]")

        num_val = max(100, params["num_samples"] // 5)
        num_test = max(100, params["num_samples"] // 5)

        config = SyntheticExperimentConfig(
            num_classes=params["num_classes"],
            num_samples_train=params["num_samples"],
            num_samples_val=num_val,
            num_samples_test=num_test,
            losses=["cross_entropy"],  # Single loss for comparison
            seeds=seeds,
            epochs=epochs,
            device=device.value,
            experiment_name=f"complexity_{level.value}",
        )

        results = run_all_synthetic_experiments(config)
        all_results[level.value] = results

    # Print comparison summary
    console.print("\n")
    console.rule("[bold blue]COMPLEXITY COMPARISON SUMMARY[/bold blue]")

    import numpy as np

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Complexity", style="cyan")
    table.add_column("Classes")
    table.add_column("Samples")
    table.add_column("Accuracy", justify="right")
    table.add_column("Convergence Epoch", justify="right")

    for level, params in complexity_configs.items():
        results = all_results.get(level.value, [])
        successful = [r for r in results if r.success]

        if successful:
            acc_mean = float(np.mean([r.best_val_accuracy for r in successful]))
            conv_mean = float(np.mean([r.convergence_epoch for r in successful]))
            table.add_row(
                level.value.upper(),
                str(params["num_classes"]),
                str(params["num_samples"]),
                f"{acc_mean:.4f}",
                f"{conv_mean:.1f}",
            )

    console.print(table)

    console.print("\n[bold]Interpretation:[/bold]")
    console.print("  • Faster convergence → simpler internal representations")
    console.print("  • Check MLflow spectral artifacts to compare singular value distributions")
    console.print("  • Expect: TRIVIAL has most uniform SVD, MEDIUM has heaviest tails")

    console.rule()


@app.command("list-shapes")
def list_shapes() -> None:
    """List the geometric shapes used in synthetic data."""
    from vision_spectra.data.synthetic import SHAPES

    console.print("\n[bold]Synthetic Dataset Shapes:[/bold]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Label", style="cyan", justify="center")
    table.add_column("Shape", style="green")
    table.add_column("Description")

    descriptions = {
        "circle": "Filled ellipse with random position/size",
        "square": "Filled rectangle with random position/size",
        "triangle": "Filled 3-point polygon (equilateral)",
        "star": "5-pointed star with inner/outer vertices",
        "cross": "Two overlapping rectangles (plus sign)",
    }

    for i, shape in enumerate(SHAPES):
        table.add_row(str(i), shape.capitalize(), descriptions.get(shape, ""))

    console.print(table)

    console.print("\n[bold]Image Properties:[/bold]")
    console.print("  • Size: 28×28 pixels (resized to 224×224 for ViT)")
    console.print("  • Channels: 3 (RGB)")
    console.print("  • Background: Dark noise (RGB 20-60)")
    console.print("  • Shape color: Bright (RGB 150-255)")
    console.print("  • Position/size: Randomized per sample")


if __name__ == "__main__":
    app()
