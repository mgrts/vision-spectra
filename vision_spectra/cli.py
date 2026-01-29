"""
Command-line interface for vision spectra experiments.

Provides commands for:
- Training classification models
- MIM pretraining
- Finetuning
- Multitask learning
- Evaluation
- Data management
- Experiments (systematic loss function comparisons)
"""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from vision_spectra import __version__
from vision_spectra.analysis.publication_figures import app as figures_app
from vision_spectra.experiments.run_classification_experiments import (
    app as experiments_app,
)

app = typer.Typer(
    name="vision-spectra",
    help="Vision Transformer spectral analysis experiments",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Add experiments as a sub-command group
app.add_typer(experiments_app, name="experiments")

# Add figures generation as a sub-command group
app.add_typer(figures_app, name="figures")


# =============================================================================
# Common Options
# =============================================================================


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"vision-spectra {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """Vision Spectra: Analyzing loss function effects on transformer weight spectra."""
    pass


# =============================================================================
# Training Commands
# =============================================================================


@app.command("train-cls")
def train_classification(
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    loss: str = typer.Option("cross_entropy", "--loss", "-l", help="Loss function"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    model: str = typer.Option("vit_tiny_patch16_224", "--model", "-m", help="Model name"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device (auto/cpu/cuda/mps)"),
    name: str = typer.Option("cls_experiment", "--name", "-n", help="Experiment name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML file"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Quick test mode"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", help="Output directory"),
) -> None:
    """
    Train a Vision Transformer for image classification.
    """
    from vision_spectra.data import get_dataset
    from vision_spectra.losses import get_loss
    from vision_spectra.models import create_vit_classifier
    from vision_spectra.settings import DatasetName, ExperimentConfig, LossName, set_seed
    from vision_spectra.training import ClassificationTrainer

    # Load or create config
    if config is not None:
        cfg = ExperimentConfig.from_yaml(config)
    else:
        cfg = ExperimentConfig(
            name=name,
            seed=seed,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        cfg.dataset.name = DatasetName(dataset)
        cfg.dataset.batch_size = batch_size
        cfg.loss.classification = LossName(loss)
        cfg.optimizer.learning_rate = lr
        cfg.training.epochs = epochs
        cfg.training.smoke_test = smoke_test
        cfg.model.name = model

    # Set seed
    set_seed(cfg.seed)

    logger.info(f"Starting classification training: {cfg.name}")
    logger.info(f"Dataset: {cfg.dataset.name.value}, Loss: {cfg.loss.classification.value}")

    # Load dataset
    dataset_obj = get_dataset(cfg.dataset, cfg.data_dir)
    train_loader = dataset_obj.get_train_loader()
    val_loader = dataset_obj.get_val_loader()
    info = dataset_obj.get_info()

    logger.info(f"Dataset: {info.num_classes} classes, {info.train_size} train samples")

    # Create model
    model_obj = create_vit_classifier(
        cfg.model,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        image_size=info.image_size[0],
    )

    # Create loss
    class_counts = info.class_counts.get("train") if info.class_counts else None
    criterion = get_loss(cfg.loss, samples_per_class=class_counts)

    # Create trainer
    trainer = ClassificationTrainer(
        config=cfg,
        model=model_obj,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        class_names=info.class_names,
    )

    # Train
    result = trainer.train()

    logger.success(f"Training complete! Best checkpoint: {result['best_checkpoint']}")


@app.command("pretrain-mim")
def pretrain_mim(
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    mask_ratio: float = typer.Option(0.75, "--mask-ratio", help="Masking ratio"),
    model: str = typer.Option("vit_tiny_patch16_224", "--model", "-m", help="Model name"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device"),
    name: str = typer.Option("mim_pretrain", "--name", "-n", help="Experiment name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML file"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Quick test mode"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", help="Output directory"),
) -> None:
    """
    Pretrain a Vision Transformer using Masked Image Modeling.
    """
    from vision_spectra.data import get_dataset
    from vision_spectra.models import MIMModel, create_vit_classifier
    from vision_spectra.settings import DatasetName, ExperimentConfig, set_seed
    from vision_spectra.training import MIMTrainer

    # Load or create config
    if config is not None:
        cfg = ExperimentConfig.from_yaml(config)
    else:
        cfg = ExperimentConfig(
            name=name,
            seed=seed,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        cfg.dataset.name = DatasetName(dataset)
        cfg.dataset.batch_size = batch_size
        cfg.optimizer.learning_rate = lr
        cfg.training.epochs = epochs
        cfg.training.smoke_test = smoke_test
        cfg.model.name = model
        cfg.model.mask_ratio = mask_ratio

    set_seed(cfg.seed)

    logger.info(f"Starting MIM pretraining: {cfg.name}")
    logger.info(f"Mask ratio: {cfg.model.mask_ratio}")

    # Load dataset
    dataset_obj = get_dataset(cfg.dataset, cfg.data_dir)
    train_loader = dataset_obj.get_train_loader()
    val_loader = dataset_obj.get_val_loader()
    info = dataset_obj.get_info()

    # Create encoder
    encoder = create_vit_classifier(
        cfg.model,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        image_size=info.image_size[0],
    )

    # Create MIM model
    model_obj = MIMModel(
        encoder=encoder,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mask_ratio=cfg.model.mask_ratio,
        norm_pix_loss=cfg.loss.mim_norm_pix,
    )

    # Create trainer
    trainer = MIMTrainer(
        config=cfg,
        model=model_obj,
        train_loader=train_loader,
        val_loader=val_loader,
        num_channels=info.num_channels,
    )

    # Train
    result = trainer.train()

    logger.success(f"Pretraining complete! Best checkpoint: {result['best_checkpoint']}")


@app.command("finetune")
def finetune(
    checkpoint: Path = typer.Argument(..., help="Path to pretrained checkpoint"),
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    loss: str = typer.Option("cross_entropy", "--loss", "-l", help="Loss function"),
    epochs: int = typer.Option(30, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    freeze_encoder: bool = typer.Option(False, "--freeze", help="Freeze encoder"),
    encoder_lr_scale: float = typer.Option(
        0.1, "--encoder-lr-scale", help="Encoder LR multiplier"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device"),
    name: str = typer.Option("finetune", "--name", "-n", help="Experiment name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML file"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Quick test mode"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", help="Output directory"),
) -> None:
    """
    Finetune a pretrained model for classification.
    """
    from vision_spectra.data import get_dataset
    from vision_spectra.losses import get_loss
    from vision_spectra.models import create_vit_classifier
    from vision_spectra.settings import DatasetName, ExperimentConfig, LossName, set_seed
    from vision_spectra.training import FinetuneTrainer

    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(1)

    # Load or create config
    if config is not None:
        cfg = ExperimentConfig.from_yaml(config)
    else:
        cfg = ExperimentConfig(
            name=name,
            seed=seed,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        cfg.dataset.name = DatasetName(dataset)
        cfg.dataset.batch_size = batch_size
        cfg.loss.classification = LossName(loss)
        cfg.optimizer.learning_rate = lr
        cfg.training.epochs = epochs
        cfg.training.smoke_test = smoke_test

    set_seed(cfg.seed)

    logger.info(f"Starting finetuning from {checkpoint}")

    # Load dataset
    dataset_obj = get_dataset(cfg.dataset, cfg.data_dir)
    train_loader = dataset_obj.get_train_loader()
    val_loader = dataset_obj.get_val_loader()
    info = dataset_obj.get_info()

    # Create model
    model_obj = create_vit_classifier(
        cfg.model,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        image_size=info.image_size[0],
    )

    # Create loss
    class_counts = info.class_counts.get("train") if info.class_counts else None
    criterion = get_loss(cfg.loss, samples_per_class=class_counts)

    # Create trainer
    trainer = FinetuneTrainer(
        config=cfg,
        model=model_obj,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        class_names=info.class_names,
        pretrained_path=checkpoint,
        freeze_encoder=freeze_encoder,
        encoder_lr_scale=encoder_lr_scale,
    )

    # Train
    result = trainer.train()

    logger.success(f"Finetuning complete! Best checkpoint: {result['best_checkpoint']}")


@app.command("train-mtl")
def train_multitask(
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    loss: str = typer.Option("cross_entropy", "--loss", "-l", help="Classification loss"),
    cls_weight: float = typer.Option(1.0, "--cls-weight", help="Classification loss weight"),
    mim_weight: float = typer.Option(0.5, "--mim-weight", help="MIM loss weight"),
    mask_ratio: float = typer.Option(0.5, "--mask-ratio", help="MIM mask ratio"),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--lr", help="Learning rate"),
    model: str = typer.Option("vit_tiny_patch16_224", "--model", "-m", help="Model name"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    device: str = typer.Option("auto", "--device", help="Device"),
    name: str = typer.Option("mtl_experiment", "--name", "-n", help="Experiment name"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Config YAML file"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Quick test mode"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
    output_dir: Path = typer.Option(Path("runs"), "--output-dir", help="Output directory"),
) -> None:
    """
    Train with joint classification and MIM losses (multitask learning).
    """
    from vision_spectra.data import get_dataset
    from vision_spectra.losses import get_loss
    from vision_spectra.models import MultitaskViT, create_vit_classifier
    from vision_spectra.settings import DatasetName, ExperimentConfig, LossName, set_seed
    from vision_spectra.training import MultitaskTrainer

    # Load or create config
    if config is not None:
        cfg = ExperimentConfig.from_yaml(config)
    else:
        cfg = ExperimentConfig(
            name=name,
            seed=seed,
            device=device,
            data_dir=data_dir,
            output_dir=output_dir,
        )
        cfg.dataset.name = DatasetName(dataset)
        cfg.dataset.batch_size = batch_size
        cfg.loss.classification = LossName(loss)
        cfg.loss.mtl_cls_weight = cls_weight
        cfg.loss.mtl_mim_weight = mim_weight
        cfg.optimizer.learning_rate = lr
        cfg.training.epochs = epochs
        cfg.training.smoke_test = smoke_test
        cfg.model.name = model
        cfg.model.mask_ratio = mask_ratio

    set_seed(cfg.seed)

    logger.info(f"Starting multitask training: {cfg.name}")
    logger.info(f"CLS weight: {cfg.loss.mtl_cls_weight}, MIM weight: {cfg.loss.mtl_mim_weight}")

    # Load dataset
    dataset_obj = get_dataset(cfg.dataset, cfg.data_dir)
    train_loader = dataset_obj.get_train_loader()
    val_loader = dataset_obj.get_val_loader()
    info = dataset_obj.get_info()

    # Create encoder
    encoder = create_vit_classifier(
        cfg.model,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        image_size=info.image_size[0],
    )

    # Create multitask model
    model_obj = MultitaskViT(
        encoder=encoder,
        decoder_embed_dim=cfg.model.decoder_embed_dim,
        decoder_depth=cfg.model.decoder_depth,
        decoder_num_heads=cfg.model.decoder_num_heads,
        mask_ratio=cfg.model.mask_ratio,
        norm_pix_loss=cfg.loss.mim_norm_pix,
    )

    # Create classification loss
    class_counts = info.class_counts.get("train") if info.class_counts else None
    cls_criterion = get_loss(cfg.loss, samples_per_class=class_counts)

    # Create trainer
    trainer = MultitaskTrainer(
        config=cfg,
        model=model_obj,
        train_loader=train_loader,
        val_loader=val_loader,
        cls_criterion=cls_criterion,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        class_names=info.class_names,
    )

    # Train
    result = trainer.train()

    logger.success(f"Training complete! Best checkpoint: {result['best_checkpoint']}")


# =============================================================================
# Evaluation Commands
# =============================================================================


@app.command("eval")
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    split: str = typer.Option("test", "--split", help="Dataset split (val/test)"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size"),
    device: str = typer.Option("auto", "--device", help="Device"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
) -> None:
    """
    Evaluate a trained model.
    """
    import torch
    from torchmetrics import Accuracy, F1Score
    from tqdm import tqdm

    from vision_spectra.data import get_dataset
    from vision_spectra.models import create_vit_classifier
    from vision_spectra.settings import DatasetName, ExperimentConfig, set_seed
    from vision_spectra.utils import get_device as get_dev

    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(1)

    device_obj = get_dev(device)

    logger.info(f"Evaluating {checkpoint} on {dataset} ({split})")

    # Load config from checkpoint
    ckpt = torch.load(checkpoint, map_location=device_obj)

    cfg = ExperimentConfig()
    cfg.dataset.name = DatasetName(dataset)
    cfg.dataset.batch_size = batch_size
    cfg.data_dir = data_dir

    # Try to get model config from checkpoint
    if "config" in ckpt:
        saved_cfg = ckpt["config"]
        if "model" in saved_cfg:
            cfg.model.name = saved_cfg["model"].get("name", cfg.model.name)

    set_seed(cfg.seed)

    # Load dataset
    dataset_obj = get_dataset(cfg.dataset, cfg.data_dir)
    info = dataset_obj.get_info()

    loader = dataset_obj.get_test_loader() if split == "test" else dataset_obj.get_val_loader()

    # Create model
    model = create_vit_classifier(
        cfg.model,
        num_classes=info.num_classes,
        num_channels=info.num_channels,
        image_size=info.image_size[0],
    )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device_obj)
    model.eval()

    # Metrics
    accuracy = Accuracy(task="multiclass", num_classes=info.num_classes).to(device_obj)
    f1 = F1Score(task="multiclass", num_classes=info.num_classes, average="macro").to(device_obj)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device_obj)
            targets = targets.to(device_obj)

            logits = model(images)
            preds = logits.argmax(dim=-1)

            accuracy.update(preds, targets)
            f1.update(preds, targets)

    acc_val = accuracy.compute().item()
    f1_val = f1.compute().item()

    logger.info(f"Results on {split}:")
    logger.info(f"  Accuracy: {acc_val:.4f}")
    logger.info(f"  Macro F1: {f1_val:.4f}")


# =============================================================================
# Data Commands
# =============================================================================


@app.command("download-data")
def download_data(
    dataset: str = typer.Option("pathmnist", "--dataset", "-d", help="Dataset name"),
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Data directory"),
) -> None:
    """
    Download a dataset.
    """
    from vision_spectra.data.medmnist import download_medmnist

    logger.info(f"Downloading {dataset} to {data_dir}")
    download_medmnist(dataset, data_dir)


@app.command("info")
def show_info() -> None:
    """
    Show environment and package information.
    """
    import sys

    import numpy as np
    import torch

    from vision_spectra.settings import PROJECT_ROOT

    typer.echo(f"Vision Spectra v{__version__}")
    typer.echo(f"Python: {sys.version}")
    typer.echo(f"PyTorch: {torch.__version__}")
    typer.echo(f"NumPy: {np.__version__}")
    typer.echo(f"Project root: {PROJECT_ROOT}")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        typer.echo(f"CUDA device: {torch.cuda.get_device_name(0)}")

    if hasattr(torch.backends, "mps"):
        typer.echo(f"MPS available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    app()
