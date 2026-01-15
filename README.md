# Vision Spectra

**Analyzing how loss functions affect transformer weight spectra in vision tasks.**

This project trains Vision Transformers (ViT) for image classification and studies how different loss functions influence the spectral properties of weight matrices (Q/K/V projections, etc.) throughout training.

## Research Motivation

The spectral properties of neural network weight matrices—such as spectral entropy, stable rank, and power-law exponents—provide insights into model expressivity, generalization, and implicit regularization. This project systematically studies how:

1. **Different classification losses** (cross-entropy, focal loss, label smoothing, class-balanced losses) affect weight spectra
2. **Training paradigms** (supervised, self-supervised pretraining, multitask learning) influence spectral evolution
3. **Masked Image Modeling (MIM)** pretraining changes the spectral structure compared to pure supervised learning

## Supported Training Regimes

| Regime | Command | Description |
|--------|---------|-------------|
| Supervised Classification | `vision-spectra train-cls` | Standard ViT classification |
| MIM Pretraining | `vision-spectra pretrain-mim` | Masked image modeling pretraining |
| Finetuning | `vision-spectra finetune` | Finetune a pretrained model |
| Multitask Learning | `vision-spectra train-mtl` | Joint classification + MIM |
| Evaluation | `vision-spectra eval` | Evaluate a trained model |

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.7+
- (Optional) CUDA-compatible GPU

### Installation

```bash
cd vision_spectra

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Setup pre-commit hooks
poetry run pre-commit install

# Verify installation
vision-spectra --help
```

### Download Dataset

```bash
# Download MedMNIST (PathMNIST by default)
vision-spectra download-data --dataset pathmnist
```

### Run Training

```bash
# Supervised classification with cross-entropy
vision-spectra train-cls --dataset pathmnist --loss cross_entropy --epochs 50

# With focal loss
vision-spectra train-cls --dataset pathmnist --loss focal --epochs 50

# MIM pretraining
vision-spectra pretrain-mim --dataset pathmnist --epochs 100

# Finetune pretrained model
vision-spectra finetune --checkpoint runs/mim_pretrain/best.pt --epochs 30

# Multitask learning
vision-spectra train-mtl --dataset pathmnist --cls-weight 1.0 --mim-weight 0.5 --epochs 50

# Quick smoke test (for CI)
vision-spectra train-cls --dataset synthetic --epochs 2 --smoke-test
```

### Configuration

All experiments can be configured via:

1. **CLI arguments** (highest priority)
2. **YAML config files** (via `--config path/to/config.yaml`)
3. **Environment variables** (prefixed with `VISION_SPECTRA_`)

Example config file:

```yaml
# configs/example.yaml
dataset:
  name: pathmnist
  image_size: 28
  batch_size: 64

model:
  name: vit_tiny_patch4_28
  num_classes: 9
  pretrained: false

training:
  epochs: 50
  optimizer: adamw
  learning_rate: 1e-4
  weight_decay: 0.05

loss:
  classification: focal
  focal_gamma: 2.0

spectral:
  log_every_n_epochs: 5
  layers: ["blocks.0", "blocks.2", "blocks.4"]
```

Run with config:

```bash
vision-spectra train-cls --config configs/example.yaml
```

## Linux VM Setup

### Install Poetry

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
poetry --version
```

### Install Project

```bash
# Clone repository
git clone <repo-url>
cd vision_spectra

# Install with Poetry
poetry install

# Run experiments
poetry run vision-spectra train-cls --dataset pathmnist
```

### Docker Setup

```bash
# Install Docker (Ubuntu)
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in for group changes

# Verify
docker --version
```

## Docker Execution

### Build Image

```bash
docker build -t vision-spectra .
```

### Run Training in Docker

```bash
# Basic run
docker run --rm vision-spectra train-cls --dataset synthetic --epochs 5

# With volume mounts for data persistence
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/mlruns:/app/mlruns \
  vision-spectra train-cls --dataset pathmnist --epochs 50

# With GPU support (requires nvidia-docker)
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  vision-spectra train-cls --dataset pathmnist --device cuda

# Interactive shell
docker run --rm -it --entrypoint /bin/bash vision-spectra
```

### Environment Variables

```bash
docker run --rm \
  -e VISION_SPECTRA_SEED=42 \
  -e VISION_SPECTRA_DEVICE=cpu \
  -e MLFLOW_TRACKING_URI=/app/mlruns \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/mlruns:/app/mlruns \
  vision-spectra train-cls --dataset pathmnist
```

## Project Structure

```text
vision_spectra/
├── pyproject.toml          # Poetry configuration
├── poetry.lock             # Locked dependencies
├── README.md               # This file
├── Dockerfile              # Container definition
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .github/                # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yaml         # CI pipeline
│       └── release.yaml    # Release pipeline
├── configs/                # Example configuration files
│   └── example.yaml
├── data/                   # Downloaded datasets (gitignored)
│   └── .gitkeep
├── runs/                   # Training outputs (gitignored)
│   └── .gitkeep
├── mlruns/                 # MLflow tracking (gitignored)
│   └── .gitkeep
├── src/
│   ├── experiments/        # Experiment scripts
│   │   └── run_loss_comparison.py
│   └── vision_spectra/     # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI entrypoints
│   ├── settings.py         # Configuration models
│   ├── data/               # Dataset modules
│   │   ├── __init__.py
│   │   ├── base.py         # Dataset abstractions
│   │   ├── medmnist.py     # MedMNIST datasets
│   │   ├── synthetic.py    # Synthetic dataset generator
│   │   └── transforms.py   # Image transforms
│   ├── models/             # Model definitions
│   │   ├── __init__.py
│   │   ├── vit.py          # ViT classifier
│   │   ├── mim.py          # MIM decoder head
│   │   └── multitask.py    # Unified multitask model
│   ├── losses/             # Loss functions
│   │   ├── __init__.py
│   │   ├── registry.py     # Loss registry
│   │   ├── classification.py
│   │   └── reconstruction.py
│   ├── metrics/            # Spectral metrics
│   │   ├── __init__.py
│   │   ├── spectral.py     # Spectral entropy, alpha, stable rank
│   │   └── extraction.py   # Weight matrix extraction
│   ├── training/           # Training loops
│   │   ├── __init__.py
│   │   ├── base.py         # Base trainer
│   │   ├── classification.py
│   │   ├── mim.py          # MIM pretraining
│   │   ├── finetune.py     # Finetuning
│   │   └── multitask.py    # MTL training
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── reproducibility.py
│       ├── checkpointing.py
│       └── logging.py
└── tests/                  # Test suite
    ├── __init__.py
    ├── test_data.py
    ├── test_losses.py
    ├── test_metrics.py
    └── test_training.py
```

## Adding New Loss Functions

1. Implement the loss in `src/vision_spectra/losses/classification.py`:

```python
class MyCustomLoss(nn.Module):
    def __init__(self, param1: float = 1.0):
        super().__init__()
        self.param1 = param1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Implementation
        return loss
```

1. Register in `src/vision_spectra/losses/registry.py`:

```python
LOSS_REGISTRY["my_custom"] = MyCustomLoss
```

1. Add configuration in `src/vision_spectra/settings.py` if needed.

## Spectral Metrics

The following spectral metrics are computed for transformer weight matrices:

| Metric | Description |
|--------|-------------|
| **Spectral Entropy** | Shannon entropy of normalized singular values; measures concentration |
| **Stable Rank** | Frobenius norm squared divided by spectral norm squared; effective dimensionality |
| **Alpha Exponent** | Power-law decay rate of singular values |
| **Power-Law Alpha (Hill)** | Tail index estimated via Hill estimator |

Metrics are logged to MLflow at configurable intervals and extracted from:

- Q/K/V projection weights in each transformer block
- MLP layers (optional)
- Patch embedding layer

## Continuous Integration

This project uses GitHub Actions for CI/CD:

- **Lint & Format**: Runs Ruff linter and formatter checks
- **Type Check**: Runs mypy for static type checking
- **Tests**: Runs pytest with coverage on Python 3.11 and 3.12
- **Smoke Test**: End-to-end training test
- **Docker Build**: Builds and tests the Docker image
- **Security Scan**: Runs Bandit security scanner

CI runs automatically on:

- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`

### Pre-commit Hooks

Local quality checks run automatically before commits:

```bash
# Install hooks (one-time setup)
poetry run pre-commit install

# Run manually on all files
poetry run pre-commit run --all-files
```

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=vision_spectra --cov-report=html

# Run specific test
poetry run pytest tests/test_losses.py -v

# Quick smoke test
poetry run pytest tests/test_training.py::test_smoke_classification -v
```

## MLflow Tracking

Experiments are tracked with MLflow:

```bash
# Start MLflow UI
poetry run mlflow ui --backend-store-uri mlruns/

# Open http://localhost:5000
```

## License

MIT License - see LICENSE file.
