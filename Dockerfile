# Vision Spectra
# Analyzing how loss functions affect transformer weight spectra in vision tasks.

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies for production)
RUN poetry install --no-root --only main

# Copy source code
COPY vision_spectra/ ./vision_spectra/
COPY configs/ ./configs/
COPY README.md ./

# Install the package
RUN poetry install --only main

# Create directories for data and outputs
RUN mkdir -p data runs mlruns

# Default command: show help
ENTRYPOINT ["poetry", "run", "vision-spectra"]
CMD ["--help"]

# =============================================================================
# GPU Variant
# =============================================================================
# To build with GPU support, use:
#
# FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base
#
# And add after Python installation:
# RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
