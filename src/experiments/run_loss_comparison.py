#!/usr/bin/env python3
"""
Example experiment script for running multiple loss function comparisons.

This script demonstrates how to run systematic experiments comparing
different loss functions and their effects on transformer weight spectra.

Usage:
    poetry run python src/experiments/run_loss_comparison.py
"""

import subprocess
from pathlib import Path

# Experiment configuration
DATASETS = ["pathmnist", "bloodmnist"]
LOSSES = ["cross_entropy", "focal", "label_smoothing", "class_balanced"]
EPOCHS = 50
BATCH_SIZE = 64
SEED = 42


def run_experiment(dataset: str, loss: str) -> None:
    """Run a single experiment."""
    experiment_name = f"{dataset}_{loss}_seed{SEED}"

    cmd = [
        "poetry",
        "run",
        "vision-spectra",
        "train-cls",
        "--dataset",
        dataset,
        "--loss",
        loss,
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--seed",
        str(SEED),
        "--name",
        experiment_name,
    ]

    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*60}\n")

    # Run from project root (two levels up from src/experiments/)
    project_root = Path(__file__).parent.parent.parent
    result = subprocess.run(cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"Experiment {experiment_name} failed with code {result.returncode}")
    else:
        print(f"Experiment {experiment_name} completed successfully")


def main():
    """Run all experiments."""
    print("Starting loss function comparison experiments")
    print(f"Datasets: {DATASETS}")
    print(f"Losses: {LOSSES}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}, Seed: {SEED}")

    for dataset in DATASETS:
        for loss in LOSSES:
            run_experiment(dataset, loss)

    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("View results with: poetry run mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()
