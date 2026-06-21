#!/usr/bin/env bash
#
# Turnkey cloud runner for the capacity × complexity ViT spectral study.
# See references/docs/EXPERIMENT_PLAN_V2.md and references/docs/CLOUD_RUN.md.
#
# Usage (on a fresh Ubuntu + NVIDIA GPU VM, repo already cloned):
#     bash scripts/run_cloud_study.sh [TIER] [NUM_SEEDS] [DEVICE]
# Examples:
#     bash scripts/run_cloud_study.sh 1 10 cuda     # Tier-1 spine, 10 seeds (default)
#     bash scripts/run_cloud_study.sh 3 10 cuda     # full study incl. depth/datasets/A-F
#
# Produces:  mlruns/  +  spectral_study_mlruns_<tier>.tar.gz  (copy this back).
set -euo pipefail

TIER="${1:-1}"
NUM_SEEDS="${2:-10}"
DEVICE="${3:-cuda}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "=== vision-spectra cloud study ==="
echo "  repo:   ${REPO_ROOT}"
echo "  tier:   ${TIER}   seeds: ${NUM_SEEDS}   device: ${DEVICE}"

# --- 1. Environment -----------------------------------------------------------
if ! command -v poetry >/dev/null 2>&1; then
  echo "[setup] installing poetry"
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] poetry install"
poetry install

echo "[setup] torch / CUDA check"
poetry run python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  gpu:", torch.cuda.get_device_name(0))
PY

# --- 2. Data ------------------------------------------------------------------
# Tier 1 needs PathMNIST; Tier >=2 also Blood/DermaMNIST. Synthetic is generated on the fly.
DATASETS=(pathmnist)
if [ "${TIER}" -ge 2 ]; then DATASETS+=(bloodmnist dermamnist); fi
for ds in "${DATASETS[@]}"; do
  echo "[data] download ${ds}"
  poetry run vision-spectra download-data --dataset "${ds}"
done

# --- 3. Run the study ---------------------------------------------------------
echo "[run] spectral run-study --tier ${TIER} --num-seeds ${NUM_SEEDS} --device ${DEVICE}"
time poetry run vision-spectra spectral run-study \
  --tier "${TIER}" \
  --num-seeds "${NUM_SEEDS}" \
  --device "${DEVICE}"

# --- 4. Figures (best-effort; safe to skip if it errors) ----------------------
echo "[figures] generating publication outputs (best-effort)"
poetry run vision-spectra figures all || echo "[figures] skipped/failed — regenerate locally from mlruns"

# --- 5. Package results -------------------------------------------------------
OUT="spectral_study_mlruns_tier${TIER}.tar.gz"
echo "[package] ${OUT}"
tar -czf "${OUT}" mlruns/ $( [ -d references/figures ] && echo references/figures )
echo "=== DONE. Copy back: ${REPO_ROOT}/${OUT} ==="
echo "    e.g.  scp <vm>:${REPO_ROOT}/${OUT} ./   then  tar xzf ${OUT}"
