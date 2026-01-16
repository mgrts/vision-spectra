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
    poetry run python -m vision_spectra.experiments.run_classification_experiments

    # Or with custom options:
    poetry run python -m vision_spectra.experiments.run_classification_experiments --dataset bloodmnist --seeds 3
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import mlflow
from loguru import logger

from vision_spectra.data import get_dataset
from vision_spectra.losses import get_loss
from vision_spectra.models import create_vit_classifier
from vision_spectra.settings import (
    DATA_DIR,
    RUNS_DIR,
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

    # Model settings
    model_name: str = "vit_tiny_patch16_224"

    # Output - use standard runs directory
    output_dir: Path = field(default_factory=lambda: RUNS_DIR)
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
            device="auto",
            output_dir=config.output_dir,
            data_dir=DATA_DIR,
            dataset=DatasetConfig(
                name=DatasetName(dataset_name),
                batch_size=config.batch_size,
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
                enabled=True,
                log_every_n_epochs=5,
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
    """Print a summary of all experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Group by loss function
    from collections import defaultdict

    by_loss: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        if r.success:
            by_loss[r.loss].append(r)

    print(
        f"\n{'Loss Function':<20} {'AUROC (mean±std)':<20} {'Accuracy (mean±std)':<22} {'F1 (mean±std)':<20} {'Runs':<6}"
    )
    print("-" * 90)

    for loss_name in sorted(by_loss.keys()):
        loss_results = by_loss[loss_name]

        if loss_results:
            import numpy as np

            aurocs = [r.best_val_auroc for r in loss_results]
            accs = [r.best_val_accuracy for r in loss_results]
            f1s = [r.best_val_f1 for r in loss_results]

            auroc_mean, auroc_std = np.mean(aurocs), np.std(aurocs)
            acc_mean, acc_std = np.mean(accs), np.std(accs)
            f1_mean, f1_std = np.mean(f1s), np.std(f1s)

            print(
                f"{loss_name:<20} "
                f"{auroc_mean:.4f} ± {auroc_std:.4f}   "
                f"{acc_mean:.4f} ± {acc_std:.4f}     "
                f"{f1_mean:.4f} ± {f1_std:.4f}   "
                f"{len(loss_results):<6}"
            )

    # Failed experiments
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\nFailed experiments: {len(failed)}")
        for r in failed:
            print(f"  - {r.dataset}_{r.loss}_seed{r.seed}: {r.error_message}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run classification experiments with multiple loss functions"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="pathmnist", help="Dataset name (default: pathmnist)"
    )
    parser.add_argument(
        "--losses",
        "-l",
        type=str,
        nargs="+",
        default=None,
        help="Loss functions to compare (default: all)",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to use (default: 42, 123, 456, 789, 1024)",
    )
    parser.add_argument(
        "--num-seeds",
        "-n",
        type=int,
        default=5,
        help="Number of seeds to use if --seeds not specified (default: 5)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=50, help="Maximum epochs (default: 50)"
    )
    parser.add_argument(
        "--patience", "-p", type=int, default=10, help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size (default: 64)"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for results"
    )

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig_(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
    )

    if args.losses:
        config.losses = args.losses

    if args.seeds:
        config.seeds = args.seeds
    elif args.num_seeds != 5:
        # Generate seeds
        config.seeds = [42 + i * 100 for i in range(args.num_seeds)]

    if args.output:
        config.output_dir = Path(args.output)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
    print("\nTo view detailed results:")
    print(f"  cd {RUNS_DIR.parent}")
    print("  poetry run mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")


if __name__ == "__main__":
    main()
