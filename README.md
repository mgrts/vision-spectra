# Vision Spectra

**Studying how network capacity and data complexity shape the weight spectra of Vision Transformers.**

This project trains small Vision Transformers (ViT) for image classification and studies how the spectral properties of their weight matrices (Q/K/V projections, MLP, etc.) evolve during training across a capacity × data-complexity grid.

> **⚠️ Methodology status — read before using the numbers**
>
> A methodological review corrected several issues in how the experiments were set up and reported. **The headline result numbers committed under `references/figures/` and in the table below were produced under the *previous* configuration and must be regenerated before they are cited.** Key points:
>
> - **Architecture fixed:** experiments previously ran `patch16` on 28px inputs, which collapses to a **single patch token** (a degenerate, non-spatial ViT), and scenarios A/B silently ran at **12 layers instead of the documented 6**. Both are now fixed (`patch_size=4` → a real 7×7=49-patch grid; depth is enforced). Re-run the scenarios to get valid numbers.
> - **"Δα" is a rank-decay slope, *not* the Martin & Mahoney heavy-tail / ESD exponent.** They are different (even inversely related) quantities. The theory-matching Hill estimator is now also computed and reported (`pl_alpha_hill`). Treat the heavy-tail framing as motivation, not as something this slope measures directly.
> - **Accuracy is now a held-out *test* metric.** Previously the reported accuracy was best-of-many *validation* accuracy (optimistically biased; no test split was scored).
> - **Statistics are descriptive at n=3 seeds.** Reported p-values reflect near-deterministic seed variance, not generalizability; error bars are 95% CIs and a multiple-comparison correction is applied. Run ≥10 seeds before making inferential claims.
> - **Δα vs accuracy is observational/confounded:** capacity and data complexity drive *both*, so this is not evidence that compression *causes* accuracy loss.

## Research Motivation

The spectral properties of neural network weight matrices—spectral entropy, stable rank, and power-law-style exponents—are studied as a window into model expressivity, generalization, and implicit regularization. The **measured** study here is:

- **Capacity × data complexity → weight spectra** (the six A–F scenarios below): how network width/depth and data complexity jointly shape the singular-value spectrum during training.

The following are **scaffolded but not yet a controlled, reported study** (the code exists; the experiments have not been run as fair comparisons) and are tracked as future work:

- **Loss-function effect on spectra** (cross-entropy / focal / label-smoothing / class-balanced / asymmetric). The loss comparison currently reports classification quality, not spectra; a fair, hyperparameter-controlled spectral comparison across losses has not been run.
- **Self-supervised (MIM) vs supervised** spectral structure, and multitask learning. No controlled MIM-vs-supervised spectral comparison is currently produced.

## Supported Training Regimes

| Regime | Command | Description |
|--------|---------|-------------|
| Supervised Classification | `vision-spectra train-cls` | Standard ViT classification |
| MIM Pretraining | `vision-spectra pretrain-mim` | Masked image modeling pretraining |
| Finetuning | `vision-spectra finetune` | Finetune a pretrained model |
| Multitask Learning | `vision-spectra train-mtl` | Joint classification + MIM |
| Evaluation | `vision-spectra eval` | Evaluate a trained model |
| **Experiments** | `vision-spectra experiments run` | Systematic loss function comparison |

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

## Running Experiments

This project provides experiment modules for systematic analysis of how loss functions affect transformer weight spectra.

---

### Core Six-Scenario Experiment Framework

The research is structured around six scenarios that vary network capacity (`embed_dim`/`depth`) and data complexity (synthetic shapes vs PathMNIST):

| Scenario | ID | Network | Data | Purpose |
|----------|----|---------|------|---------|
| **A: Expressive + Simple** | `EXP_A` | ViT (192d, 6L, patch4) | Synthetic shapes | Baseline: ample capacity for simple data |
| **B: Expressive + Complex** | `EXP_B` | ViT (192d, 6L, patch4) | PathMNIST | Main: compression regime |
| **C: Reduced + Complex** | `EXP_C` | ViT (96d, 3L, patch4) | PathMNIST | Over-compression: reduced capacity |
| **D: Reduced + Simple** | `EXP_D` | ViT (96d, 3L, patch4) | Synthetic shapes | Control: simple data + reduced capacity |
| **E: Tiny + Simple** | `EXP_E` | ViT (32d, 1L, patch4) | Synthetic shapes | Minimal capacity on simple data |
| **F: Tiny + Complex** | `EXP_F` | ViT (32d, 1L, patch4) | PathMNIST | Extreme over-compression |

> **The Δα / accuracy numbers previously shown here have been removed because they
> were produced under the prior degenerate configuration (1-patch ViT; A/B at 12
> layers) and are no longer valid.** Regenerate them by running the scenarios under
> the corrected config (`vision-spectra spectral run-all`) and then
> `vision-spectra figures all`.

**Expected/hypothesized findings (to be confirmed under the corrected config):**

- The bulk singular-value spectrum is hypothesized to change more (larger Δα rank-slope) as the data-complexity / network-capacity ratio increases.
- Δα is reported as a **rank-decay slope** (and alongside the Hill ESD tail exponent); it is **not** the Martin & Mahoney heavy-tail index, so do not read it through the M&M "α ∈ [2,6]" band.
- Any relationship between Δα and accuracy is **observational and confounded** by capacity/complexity (which drive both) — it is not evidence that compression *causes* accuracy loss. Accuracy is also not directly comparable across the 3-class synthetic and 9-class PathMNIST arms (different chance levels).
- Statistics are **descriptive at n=3 seeds** (run ≥10 seeds for inferential claims).

#### Running the Six Scenarios

```bash
# Run all scenarios at once
poetry run python -m vision_spectra.experiments.run_spectral_analysis run-all \
    --num-seeds 3 \
    --device auto

# Or run individual scenarios:

# Scenario A: Expressive + Simple (synthetic data)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-a \
    --num-seeds 3 \
    --device auto

# Scenario B: Expressive + Complex (real medical data)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-b \
    --num-seeds 3 \
    --device auto

# Scenario C: Reduced Expressivity + Complex (narrow network)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-c \
    --num-seeds 3 \
    --device auto

# Scenario D: Reduced Expressivity + Simple (narrow network + simple data)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-d \
    --num-seeds 3 \
    --device auto

# Scenario E: Tiny Network + Simple (minimal capacity: embed=32, depth=1)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-e \
    --num-seeds 3 \
    --device auto

# Scenario F: Tiny Network + Complex (minimal capacity on complex data)
poetry run python -m vision_spectra.experiments.run_spectral_analysis scenario-f \
    --num-seeds 3 \
    --device auto

# Compare results across all scenarios
poetry run python -m vision_spectra.experiments.run_spectral_analysis compare

# Generate publication-quality figures
poetry run vision-spectra figures all  # or: python -m vision_spectra.analysis.publication_figures all
```

#### Publication Figures

After running experiments, generate publication-quality figures:

```bash
poetry run vision-spectra figures all  # or: python -m vision_spectra.analysis.publication_figures all
```

This creates the following in `publication_figures/`:

- `delta_alpha_bar.png` - Bar chart comparing Δα across scenarios
- `accuracy_vs_compression.png` - Scatter plot of accuracy vs compression
- `heatmap.png` - Capacity × Complexity heatmap
- `stable_rank.png` - Stable rank reduction comparison
- `stats.json` - Statistical test results (t-tests, p-values)
- `table.tex` - LaTeX table for papers

---

### Classification Experiments (Real-World Data)

The classification experiments module compares different loss functions on real-world medical imaging datasets (MedMNIST) across several seeds. Note: this comparison currently reports **classification quality** (accuracy/AUROC/F1), not weight spectra, and the losses are run at fixed default hyperparameters under a shared optimizer/learning-rate — so it is not yet a controlled "loss → spectra" study (see the methodology note at the top). With the default of 3–5 seeds, treat any cross-loss differences as descriptive.

#### Classification Quick Start

```bash
# Run full experiment suite (all losses, 5 seeds each)
poetry run vision-spectra experiments run

# Or via direct module execution
poetry run python -m vision_spectra.experiments.run_classification_experiments run

# View help for all options
poetry run vision-spectra experiments run --help
```

#### Classification Usage Examples

```bash
# Quick test: 10% of data, 2 seeds, CPU (recommended for first run)
poetry run vision-spectra experiments run \
    --dataset pathmnist \
    --sample-ratio 0.1 \
    --num-seeds 2 \
    --device cpu

# Compare specific losses only
poetry run vision-spectra experiments run \
    --losses cross_entropy focal label_smoothing \
    --num-seeds 3

# Fast mode: disable spectral tracking for speed
poetry run vision-spectra experiments run --fast --num-seeds 2

# Full experiment with custom settings
poetry run vision-spectra experiments run \
    --dataset pathmnist \
    --losses cross_entropy focal \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --log-every-n-epochs 3

# List available loss functions
poetry run vision-spectra experiments list-losses
```

#### Classification Experiment Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--dataset` | `-d` | Dataset name (`pathmnist`, `bloodmnist`, etc.) | `pathmnist` |
| `--losses` | `-l` | Loss functions to compare (space-separated) | all available |
| `--seeds` | `-s` | Specific random seeds | `42, 123, 456, 789, 1024` |
| `--num-seeds` | `-n` | Number of seeds (if `--seeds` not specified) | `5` |
| `--epochs` | `-e` | Maximum training epochs | `50` |
| `--patience` | `-p` | Early stopping patience (epochs without improvement) | `10` |
| `--batch-size` | `-b` | Training batch size | `64` |
| `--lr` | | Learning rate | `1e-4` |
| `--sample-ratio` | `-r` | Fraction of dataset to use (0.01–1.0) | `1.0` |
| `--device` | | Device: `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--output` | `-o` | Output directory for MLflow artifacts | `mlruns/` |
| `--fast` | `-f` | Fast mode (disable spectral tracking) | `false` |
| `--log-every-n-epochs` | | Log spectral metrics every N epochs | `5` |
| `--log-first-epochs` | | Log spectral metrics for epochs 0–4 | `true` |
| `--track-distributions` | | Track full singular value arrays as JSON | `true` |
| `--save-distribution-history` | | Save spectral history JSON and histogram plots | `true` |

#### Available Loss Functions

| Loss | Description |
|------|-------------|
| `cross_entropy` | Standard cross-entropy loss |
| `focal` | Focal loss that down-weights easy examples (γ=2.0) |
| `label_smoothing` | Cross-entropy with soft labels (ε=0.1) |
| `class_balanced` | Re-weighted loss based on effective number of samples |
| `asymmetric` | Asymmetric loss for handling class imbalance |

---

### Synthetic Data Experiments

The synthetic experiments module tests spectral hypotheses on simple geometric shapes data. This is useful for validating that simpler data leads to less heavy-tailed weight spectra.

**Hypothesis:** Simple synthetic data leads to less heavy-tailed weight spectra because models can learn patterns quickly without complex internal representations, resulting in more uniform singular value distributions.

#### Synthetic Quick Start

```bash
# Run synthetic experiments with defaults (3 classes, 1000 samples)
poetry run python -m vision_spectra.experiments.run_synthetic_experiments run

# View help for all options
poetry run python -m vision_spectra.experiments.run_synthetic_experiments run --help
```

#### Synthetic Usage Examples

```bash
# Very simple data: 2 classes, 500 samples
poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
    --num-classes 2 \
    --num-samples 500 \
    --epochs 20

# More complex synthetic data
poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
    --num-classes 5 \
    --num-samples 5000 \
    --epochs 50 \
    --num-seeds 5

# Compare complexity levels (trivial/simple/medium)
poetry run python -m vision_spectra.experiments.run_synthetic_experiments compare-complexity

# Run on specific device
poetry run python -m vision_spectra.experiments.run_synthetic_experiments run \
    --device cpu \
    --num-classes 3 \
    --num-samples 1000

# List available shapes
poetry run python -m vision_spectra.experiments.run_synthetic_experiments list-shapes
```

#### Synthetic Experiment Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--num-classes` | `-c` | Number of shape classes (2–5). Fewer = simpler | `3` |
| `--num-samples` | `-n` | Number of training samples | `1000` |
| `--losses` | `-l` | Loss functions to compare | `cross_entropy, focal, label_smoothing` |
| `--seeds` | `-s` | Specific random seeds | auto-generated |
| `--num-seeds` | | Number of seeds | `3` |
| `--epochs` | `-e` | Maximum training epochs | `30` |
| `--patience` | `-p` | Early stopping patience | `10` |
| `--batch-size` | `-b` | Training batch size | `32` |
| `--lr` | | Learning rate | `1e-4` |
| `--device` | | Device: `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--log-every-n-epochs` | | Spectral logging frequency | `2` |
| `--output` | `-o` | Output directory | `mlruns/` |

#### Complexity Comparison

The `compare-complexity` command runs experiments at three complexity levels:

| Level | Classes | Samples | Expected Spectral Behavior |
|-------|---------|---------|---------------------------|
| **TRIVIAL** | 2 | 500 | Most uniform SVD (less heavy-tailed) |
| **SIMPLE** | 3 | 1,000 | Moderate tail weight |
| **MEDIUM** | 5 | 5,000 | Heaviest tails (more complex representations) |

```bash
# Run complexity comparison
poetry run python -m vision_spectra.experiments.run_synthetic_experiments compare-complexity \
    --num-seeds 2 \
    --epochs 30 \
    --device auto
```

---

### Extended Experiment Matrix

For comprehensive analysis, experiments can be run across multiple model variants and datasets:

| Model Variant | Synthetic (Simple) | PathMNIST | BloodMNIST | DermaMNIST |
|---------------|-------------------|-----------|------------|------------|
| ViT-Tiny (embed=192, depth=6) | `A1` | `B1` | `B1b` | `B1c` |
| ViT-Tiny-Narrow (embed=96) | `A2` | `C1` | `C1b` | `C1c` |
| ViT-Tiny-Shallow (depth=3) | `A3` | `C2` | `C2b` | `C2c` |
| ViT-Small (embed=384, depth=6) | `A4` | `B2` | `B2b` | `B2c` |

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Spectral metrics | ✅ Complete | `vision_spectra/metrics/spectral.py` |
| Weight extraction | ✅ Complete | `vision_spectra/metrics/extraction.py` |
| Synthetic data | ✅ Complete | `vision_spectra/data/synthetic.py` |
| Classification experiments | ✅ Complete | Scenario B fully supported |
| Synthetic experiments | ✅ Complete | Scenario A fully supported |
| Narrow network variant | ✅ Complete | `embed_dim`/`depth` params in ViT |
| Spectral analysis pipeline | ✅ Complete | `run_spectral_analysis.py` |
| Gradient alignment metric | ⚠️ Implemented, not wired | `vision_spectra/metrics/gradient_alignment.py` — not invoked by any experiment/CLI yet |
| Tail-truncation experiment | ⚠️ Implemented, not wired | `vision_spectra/metrics/tail_truncation.py` — Eckart-Young low-rank truncation; not invoked by any experiment/CLI yet |
| CCDF/log-log plots | ✅ Complete | `vision_spectra/metrics/plotting.py` |
| Statistical comparison | ✅ Complete | `vision_spectra/metrics/statistical.py` (descriptive; underpowered at n=3) |

#### Shapes in Synthetic Data

| Label | Shape | Description |
|-------|-------|-------------|
| 0 | Circle | Filled ellipse with random position/size |
| 1 | Square | Filled rectangle with random position/size |
| 2 | Triangle | Filled 3-point polygon (equilateral) |
| 3 | Star | 5-pointed star with inner/outer vertices |
| 4 | Cross | Two overlapping rectangles (plus sign) |

**Image properties:**

- Size: 28×28 pixels, fed to the ViT at 28×28 (`img_size=28`). With `patch_size=4` this yields a 7×7 = 49-patch grid (50 tokens incl. CLS). Inputs are **not** upsampled to 224.
- Channels: 3 (RGB)
- Background: Dark noise (RGB 20–60)
- Shape color: Bright (RGB 150–255)

---

### Understanding Experiment Output

All experiments are tracked in MLflow. After running experiments:

```bash
# Start MLflow UI to view results
poetry run mlflow ui --backend-store-uri mlruns/

# Open http://localhost:5000 in your browser
```

#### What's Logged in MLflow

- **Metrics:** Accuracy, AUROC, F1-score, loss per epoch
- **Spectral metrics:** Entropy, stable rank, alpha exponent (per epoch when tracked)
- **Artifacts:**
  - `spectral/epoch_N/values.json` — Full singular value arrays
  - `spectral/epoch_N/histograms/*.png` — Distribution histograms
  - Model checkpoints (best model)

#### Recommended Experiment Workflow

1. **Quick validation** — Use `--sample-ratio 0.1 --num-seeds 2` first
2. **Full experiment** — Run with default settings once validated
3. **Analyze in MLflow** — Compare spectral evolution across loss functions
4. **Synthetic comparison** — Use synthetic experiments to validate hypotheses

---

### Configuration

All experiments can be configured via:

1. **CLI arguments** (highest priority)
2. **YAML config files** (via `--config path/to/config.yaml`)

CLI flags override values loaded from a `--config` file, which override the
built-in defaults. Unknown or misplaced YAML keys raise a validation error
(they are not silently ignored), so each key must live under its correct
section (e.g. `learning_rate` belongs under `optimizer:`, and `num_classes`
is derived from the dataset rather than set on the model).

Example config file:

```yaml
# configs/example.yaml
dataset:
  name: pathmnist
  image_size: 28
  batch_size: 64

model:
  name: vit_tiny_patch16_224
  pretrained: false

optimizer:
  name: adamw
  learning_rate: 1e-4
  weight_decay: 0.05

training:
  epochs: 50

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

### Run Settings

Seed and device are passed as CLI flags (there is no `VISION_SPECTRA_*`
environment-variable configuration layer):

```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/mlruns:/app/mlruns \
  vision-spectra train-cls --dataset pathmnist --seed 42 --device cpu
```

## Project Structure

```text
vision-spectra/
├── pyproject.toml          # Poetry configuration
├── poetry.lock             # Locked dependencies
├── README.md               # This file
├── Dockerfile              # Container definition
├── .dockerignore           # Docker build exclusions
├── .gitignore              # Git ignore patterns
├── .pre-commit-config.yaml # Pre-commit hooks config
├── .github/                # GitHub Actions workflows
│   └── workflows/
│       └── ci.yaml         # CI pipeline (lint, test, type-check)
├── configs/                # Example configuration files
│   └── example.yaml
├── data/                   # Downloaded datasets (gitignored)
├── runs/                   # Training outputs (gitignored)
├── mlruns/                 # MLflow tracking (gitignored)
├── vision_spectra/         # Main package
│   ├── __init__.py
│   ├── cli.py              # CLI entrypoints
│   ├── settings.py         # Configuration models
│   ├── experiments/        # Experiment scripts
│   │   ├── __init__.py
│   │   └── run_classification_experiments.py
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
│       └── visualization.py
└── tests/                  # Test suite
    ├── __init__.py
    ├── test_data.py
    ├── test_losses.py
    ├── test_metrics.py
    └── test_training.py
```

## Adding New Loss Functions

1. Implement the loss in `vision_spectra/losses/classification.py`:

```python
class MyCustomLoss(nn.Module):
    def __init__(self, param1: float = 1.0):
        super().__init__()
        self.param1 = param1

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Implementation
        return loss
```

1. Register in `vision_spectra/losses/registry.py`:

```python
LOSS_REGISTRY["my_custom"] = MyCustomLoss
```

1. Add configuration in `vision_spectra/settings.py` if needed.

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
