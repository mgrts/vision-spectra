#!/usr/bin/env python3
"""
Classification experiments with multiple loss functions and seeds.

This module runs systematic experiments comparing different loss functions
for image classification and their effects on transformer weight spectra.

Key features:
- Multiple loss functions: cross_entropy, focal, label_smoothing, class_balanced, asymmetric
- Multiple seeds for statistical reliability (5 runs per configuration)
- Early stopping with configurable patience
- AUROC as the primary metric (alongside accuracy, F1, and spectral metrics)
- MLflow tracking for all experiments

Usage:
    poetry run vision-spectra experiments run-classification

    # Or with custom options:
    poetry run vision-spectra experiments run-classification --dataset bloodmnist --num-seeds 3

    # Direct module execution:
    poetry run python -m vision_spectra.experiments.run_classification_experiments --help
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import mlflow
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
    name="experiments",
    help="Run systematic experiments for comparing loss functions.",
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


# Default loss functions for experiments
DEFAULT_LOSSES: list[str] = [
    "cross_entropy",
    "focal",
    "label_smoothing",
    "class_balanced",
    "asymmetric",
]

# Default seeds for reproducibility
DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024]


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    dataset: str
    loss: str
    seed: int
    best_val_loss: float
    best_val_accuracy: float
    best_val_f1: float
    best_val_auroc: float
    final_epoch: int
    training_time_seconds: float
    checkpoint_path: str | None
    mlflow_run_id: str | None
    spectral_metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str | None = None


@dataclass
class ExperimentConfig_:
    """Configuration for the experiment batch."""

    # Dataset
    dataset: str = "pathmnist"

    # Loss functions to compare
    losses: list[str] = field(
        default_factory=lambda: [
            "cross_entropy",
            "focal",
            "label_smoothing",
            "class_balanced",
            "asymmetric",
        ]
    )

    # Seeds for multiple runs
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Training settings
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10

    # Dataset sampling (for faster experiments)
    sample_ratio: float = 1.0

    # Device settings
    device: str = "auto"

    # Fast mode - disables spectral tracking to save memory
    fast_mode: bool = False

    # Spectral logging settings
    log_every_n_epochs: int = 5
    log_first_epochs: bool = True  # Log first 5 epochs (0-4) for initial dynamics
    track_distributions: bool = True  # Track full singular value distributions
    save_distribution_history: bool = True  # Save distribution history to JSON

    # Model settings
    model_name: str = "vit_tiny_patch16_224"

    # Output - use mlruns directory for all experiment tracking
    output_dir: Path = field(default_factory=lambda: MLRUNS_DIR)
    experiment_name: str = "classification_loss_comparison"


def run_single_experiment(
    dataset_name: str,
    loss_name: str,
    seed: int,
    config: ExperimentConfig_,
) -> ExperimentResult:
    """
    Run a single classification experiment.

    Args:
        dataset_name: Name of the dataset
        loss_name: Name of the loss function
        seed: Random seed
        config: Experiment configuration

    Returns:
        ExperimentResult with metrics and status
    """
    import gc

    import torch

    experiment_id = f"{dataset_name}_{loss_name}_seed{seed}"
    logger.info(f"Starting experiment: {experiment_id}")

    start_time = time.time()

    # Variables to track for cleanup
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
                name=DatasetName(dataset_name),
                batch_size=config.batch_size,
                sample_ratio=config.sample_ratio,
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
                warmup_epochs=5,
            ),
            training=TrainingConfig(
                epochs=config.epochs,
                early_stopping=True,
                patience=config.early_stopping_patience,
                save_every_n_epochs=10,
            ),
            spectral=SpectralConfig(
                enabled=not config.fast_mode,
                log_every_n_epochs=config.log_every_n_epochs,
                log_first_epochs=config.log_first_epochs,
                track_distributions=config.track_distributions and not config.fast_mode,
                save_distribution_history=config.save_distribution_history
                and not config.fast_mode,
            ),
        )

        # Load dataset
        dataset_obj = get_dataset(exp_config.dataset, exp_config.data_dir)
        train_loader = dataset_obj.get_train_loader()
        val_loader = dataset_obj.get_val_loader()
        info = dataset_obj.get_info()

        logger.info(f"Dataset: {info.num_classes} classes, {info.train_size} train samples")

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

        # Train
        result = trainer.train()

        training_time = time.time() - start_time

        # Get final metrics from trainer
        val_metrics = trainer.validate()

        # Get MLflow run ID
        mlflow_run_id = None
        with contextlib.suppress(Exception):
            mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        experiment_result = ExperimentResult(
            dataset=dataset_name,
            loss=loss_name,
            seed=seed,
            best_val_loss=result.get("best_val_metric", float("inf")),
            best_val_accuracy=val_metrics.get("accuracy", 0.0),
            best_val_f1=val_metrics.get("f1_macro", 0.0),
            best_val_auroc=val_metrics.get("auroc", 0.0),
            final_epoch=result.get("final_epoch", 0),
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

        experiment_result = ExperimentResult(
            dataset=dataset_name,
            loss=loss_name,
            seed=seed,
            best_val_loss=float("inf"),
            best_val_accuracy=0.0,
            best_val_f1=0.0,
            best_val_auroc=0.0,
            final_epoch=0,
            training_time_seconds=time.time() - start_time,
            checkpoint_path=None,
            mlflow_run_id=None,
            success=False,
            error_message=str(e),
        )

    finally:
        # Clean up resources to free memory
        logger.debug("Cleaning up experiment resources...")

        # Clean up trainer (this clears model, optimizer, etc.)
        if trainer is not None:
            try:
                trainer.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Trainer cleanup failed: {cleanup_error}")
            del trainer
            trainer = None

        # Clean up model reference (in case trainer cleanup failed)
        if model is not None:
            with contextlib.suppress(Exception):
                model.cpu()
            del model
            model = None

        # Clean up data loaders
        if train_loader is not None:
            del train_loader
            train_loader = None
        if val_loader is not None:
            del val_loader
            val_loader = None

        # Clean up dataset
        if dataset_obj is not None:
            del dataset_obj
            dataset_obj = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.debug("Experiment cleanup complete")

    return experiment_result


def run_all_experiments(config: ExperimentConfig_) -> list[ExperimentResult]:
    """
    Run all experiments for all loss functions and seeds.

    Args:
        config: Experiment configuration

    Returns:
        List of all experiment results
    """
    results: list[ExperimentResult] = []

    total_experiments = len(config.losses) * len(config.seeds)
    current_experiment = 0

    logger.info(f"Starting {total_experiments} experiments")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Losses: {config.losses}")
    logger.info(f"Seeds: {config.seeds}")
    logger.info(
        f"Max epochs: {config.epochs}, Early stopping patience: {config.early_stopping_patience}"
    )
    if config.sample_ratio < 1.0:
        logger.info(f"Dataset sampling: {config.sample_ratio * 100:.0f}% of data")
    logger.info(f"Device: {config.device}")

    for loss_name in config.losses:
        for seed in config.seeds:
            current_experiment += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {current_experiment}/{total_experiments}")
            logger.info(f"Loss: {loss_name}, Seed: {seed}")
            logger.info(f"{'='*60}")

            result = run_single_experiment(
                dataset_name=config.dataset,
                loss_name=loss_name,
                seed=seed,
                config=config,
            )
            results.append(result)

            # Log result summary
            if result.success:
                logger.success(
                    f"Completed: AUROC={result.best_val_auroc:.4f}, "
                    f"Acc={result.best_val_accuracy:.4f}, "
                    f"F1={result.best_val_f1:.4f}, "
                    f"Epoch={result.final_epoch}, "
                    f"Time={result.training_time_seconds:.1f}s"
                )
            else:
                logger.error(f"Failed: {result.error_message}")

    return results


def save_results(results: list[ExperimentResult], output_path: Path) -> None:
    """Save experiment results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results_dict = []
    for r in results:
        results_dict.append(
            {
                "dataset": r.dataset,
                "loss": r.loss,
                "seed": r.seed,
                "best_val_loss": r.best_val_loss,
                "best_val_accuracy": r.best_val_accuracy,
                "best_val_f1": r.best_val_f1,
                "best_val_auroc": r.best_val_auroc,
                "final_epoch": r.final_epoch,
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


def print_summary(results: list[ExperimentResult]) -> None:
    """Print a summary of all experiment results using Rich tables."""
    from collections import defaultdict

    import numpy as np

    console.print("\n")
    console.rule("[bold blue]EXPERIMENT SUMMARY[/bold blue]")

    # Group by loss function
    by_loss: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        if r.success:
            by_loss[r.loss].append(r)

    # Create summary table
    table = Table(
        title="Results by Loss Function",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Loss Function", style="cyan", no_wrap=True)
    table.add_column("AUROC (mean±std)", justify="right")
    table.add_column("Accuracy (mean±std)", justify="right")
    table.add_column("F1 (mean±std)", justify="right")
    table.add_column("Runs", justify="center")

    for loss_name in sorted(by_loss.keys()):
        loss_results = by_loss[loss_name]

        if loss_results:
            aurocs = [r.best_val_auroc for r in loss_results]
            accs = [r.best_val_accuracy for r in loss_results]
            f1s = [r.best_val_f1 for r in loss_results]

            auroc_mean, auroc_std = float(np.mean(aurocs)), float(np.std(aurocs))
            acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
            f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))

            table.add_row(
                loss_name,
                f"{auroc_mean:.4f} ± {auroc_std:.4f}",
                f"{acc_mean:.4f} ± {acc_std:.4f}",
                f"{f1_mean:.4f} ± {f1_std:.4f}",
                str(len(loss_results)),
            )

    console.print(table)

    # Failed experiments
    failed = [r for r in results if not r.success]
    if failed:
        console.print(f"\n[red]Failed experiments: {len(failed)}[/red]")
        for r in failed:
            console.print(f"  [red]• {r.dataset}_{r.loss}_seed{r.seed}: {r.error_message}[/red]")

    console.rule()


@app.command("run")
def run_classification(
    dataset: str = typer.Option(
        "pathmnist",
        "--dataset",
        "-d",
        help="Dataset name to use for experiments.",
    ),
    losses: list[str] | None = typer.Option(
        None,
        "--losses",
        "-l",
        help="Loss functions to compare. If not specified, all available losses are used.",
    ),
    seeds: list[int] | None = typer.Option(
        None,
        "--seeds",
        "-s",
        help="Specific seeds to use for reproducibility.",
    ),
    num_seeds: int = typer.Option(
        5,
        "--num-seeds",
        "-n",
        help="Number of seeds to generate if --seeds not specified.",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs",
        "-e",
        help="Maximum number of training epochs.",
    ),
    patience: int = typer.Option(
        10,
        "--patience",
        "-p",
        help="Early stopping patience.",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        "-b",
        help="Training batch size.",
    ),
    lr: float = typer.Option(
        1e-4,
        "--lr",
        help="Learning rate.",
    ),
    sample_ratio: float = typer.Option(
        1.0,
        "--sample-ratio",
        "-r",
        min=0.01,
        max=1.0,
        help="Fraction of dataset to use (0.01-1.0). Use <1 for faster experiments.",
    ),
    device: DeviceChoice = typer.Option(
        DeviceChoice.AUTO,
        "--device",
        help="Device to use for training.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results. Defaults to runs/.",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        "-f",
        help="Fast mode: disable spectral tracking to save memory.",
    ),
    log_every_n_epochs: int = typer.Option(
        5,
        "--log-every-n-epochs",
        help="Log spectral metrics every N epochs.",
    ),
    log_first_epochs: bool = typer.Option(
        True,
        "--log-first-epochs/--no-log-first-epochs",
        help="Log spectral metrics for first 5 epochs (0-4) to capture initial dynamics.",
    ),
    track_distributions: bool = typer.Option(
        True,
        "--track-distributions/--no-track-distributions",
        help="Track full singular value distributions as JSON arrays.",
    ),
    save_distribution_history: bool = typer.Option(
        True,
        "--save-distribution-history/--no-save-distribution-history",
        help="Save spectral distribution history to JSON and generate histogram plots.",
    ),
) -> None:
    """
    Run classification experiments with multiple loss functions and seeds.

    This command systematically compares different loss functions for image
    classification and analyzes their effects on transformer weight spectra.

    Examples:
        # Run with default settings
        poetry run vision-spectra experiments run

        # Use 50% of data with 2 seeds on CPU
        poetry run vision-spectra experiments run --sample-ratio 0.5 --num-seeds 2 --device cpu

        # Compare specific losses
        poetry run vision-spectra experiments run --losses cross_entropy focal --num-seeds 3
    """
    # Resolve output directory
    resolved_output_dir = output_dir if output_dir is not None else MLRUNS_DIR

    # Determine which losses to use
    resolved_losses = list(losses) if losses else DEFAULT_LOSSES

    # Determine seeds
    if seeds:
        resolved_seeds = list(seeds)
    elif num_seeds != 5:
        resolved_seeds = [42 + i * 100 for i in range(num_seeds)]
    else:
        resolved_seeds = DEFAULT_SEEDS

    # Create config
    config = ExperimentConfig_(
        dataset=dataset,
        losses=resolved_losses,
        seeds=resolved_seeds,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        early_stopping_patience=patience,
        sample_ratio=sample_ratio,
        device=device.value,
        fast_mode=fast,
        log_every_n_epochs=log_every_n_epochs,
        log_first_epochs=log_first_epochs,
        track_distributions=track_distributions,
        save_distribution_history=save_distribution_history,
        output_dir=resolved_output_dir,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Display configuration
    console.print("\n[bold blue]Classification Experiments[/bold blue]")
    console.print(f"  Dataset: [cyan]{config.dataset}[/cyan]")
    console.print(f"  Losses: [cyan]{', '.join(config.losses)}[/cyan]")
    console.print(f"  Seeds: [cyan]{config.seeds}[/cyan]")
    console.print(f"  Device: [cyan]{config.device}[/cyan]")
    if config.sample_ratio < 1.0:
        console.print(f"  Sample ratio: [yellow]{config.sample_ratio * 100:.0f}%[/yellow]")
    if config.fast_mode:
        console.print("  Mode: [yellow]Fast (spectral tracking disabled)[/yellow]")
    console.print(f"  Output: [cyan]{config.output_dir}[/cyan]")
    console.print()

    # Run experiments
    logger.info("Starting classification experiments")
    results = run_all_experiments(config)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = config.output_dir / f"results_{config.dataset}_{timestamp}.json"
    save_results(results, results_path)

    # Print summary
    print_summary(results)

    # Print MLflow instructions
    console.print("\n[bold]To view detailed results:[/bold]")
    console.print(f"  cd {MLRUNS_DIR.parent}")
    console.print("  poetry run mlflow ui")
    console.print(
        "\nThen open [link=http://localhost:5000]http://localhost:5000[/link] in your browser"
    )


@app.command("list-losses")
def list_losses() -> None:
    """List all available loss functions for experiments."""
    console.print("\n[bold]Available Loss Functions:[/bold]\n")

    loss_descriptions = {
        "cross_entropy": "Standard cross-entropy loss for multi-class classification",
        "focal": "Focal loss that down-weights easy examples (gamma=2.0)",
        "label_smoothing": "Cross-entropy with soft labels (epsilon=0.1)",
        "class_balanced": "Re-weighted loss based on effective number of samples",
        "asymmetric": "Asymmetric loss for handling positive/negative imbalance",
    }

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Loss Name", style="cyan")
    table.add_column("Description")

    for loss_name in DEFAULT_LOSSES:
        table.add_row(loss_name, loss_descriptions.get(loss_name, ""))

    console.print(table)


if __name__ == "__main__":
    app()
