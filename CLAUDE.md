# CLAUDE.md — vision-spectra

Research codebase studying **how network capacity and data complexity shape the weight
spectra of Vision Transformers**. It trains small ViTs (timm) for image classification on
synthetic geometric shapes and MedMNIST, and tracks spectral properties of the weight
matrices (Q/K/V/MLP) — spectral entropy, stable rank, a rank-decay slope (`alpha_exponent`)
and a Hill ESD tail exponent (`pl_alpha_hill`) — across a capacity × complexity grid
(six scenarios A–F). Experiments are logged to MLflow under `mlruns/` and turned into
figures/tables/stats by `vision_spectra/analysis/publication_figures.py`. Everything is
driven through the `vision-spectra` Typer CLI (`vision_spectra/cli.py`).

> **Results status:** the committed numbers under `references/figures/` were produced under
> a *previous* configuration (a degenerate 1-patch ViT; scenarios A/B silently ran at 12
> layers). Both are fixed (`patch_size=4` → a real 49-patch grid; depth is enforced). The
> committed result numbers are stale and must be regenerated before they are cited — see
> the methodology note at the top of `README.md`.

## Package map

- `vision_spectra/cli.py` — Typer entry point (`vision-spectra`). Commands: `train-cls`,
  `pretrain-mim`, `finetune`, `train-mtl`, `eval`, `download-data`, `info`, plus sub-apps
  `experiments` (loss comparison), `figures` (publication figures), `spectral` (6-scenario
  study + `run-study` capacity×complexity sweep), `synthetic`.
- `vision_spectra/settings.py` — Pydantic config (single source of truth). `_StrictModel`
  base gives every config `extra="forbid"` + `validate_assignment`. Canonical `set_seed`
  and `resolve_device` live here (re-exported by `utils`).
- `vision_spectra/models/` — `vit.py` (`ViTClassifier` timm wrapper + `create_vit_classifier`),
  `mim.py` (`MIMModel`, `MIMDecoder`, MAE-style masking), `multitask.py` (`MultitaskViT`).
- `vision_spectra/losses/` — `classification.py` (CE/focal/label-smoothing/class-balanced/
  asymmetric), `reconstruction.py` (MSE/L1/SmoothL1/Huber/Cauchy/SGT/Tukey), `registry.py`
  (`get_loss`, `LOSS_REGISTRY`, `MIM_LOSS_REGISTRY`).
- `vision_spectra/metrics/` — `spectral.py` (the spectral math + `SpectralTracker`),
  `extraction.py` (weight-matrix extraction from timm models), `statistical.py` (t-tests,
  Cohen's d, CIs), `plotting.py` (CCDF/log-log/heatmap), `gradient_alignment.py` (cosine of
  the data gradient with the rank-reducing `U Vᵀ` flow) and `tail_truncation.py` (Eckart-Young
  bulk/head truncation) — **now wired into the spectral runner** (logged as `alignment/*`,
  `truncation/*`).
- `vision_spectra/training/` — `base.py` (`BaseTrainer` + the shared `build_optimizer` /
  `build_scheduler` / `warmup_factor` recipe), `classification.py`, `mim.py`, `finetune.py`,
  `multitask.py`.
- `vision_spectra/data/` — `base.py` (`get_dataset`, `BaseDataset`), `medmnist.py`,
  `synthetic.py`, `transforms.py`.
- `vision_spectra/experiments/` — `run_spectral_analysis.py` (the headline 6-scenario study +
  the `run-study` capacity×complexity sweep; `SCENARIO_CONFIGS`, `build_study_configs`,
  `build_followup_configs` / `build_study_set` (Sept-2026 controls: `--set followup`),
  `create_model_for_scenario`, `run_scenario_experiment`, `record_gradient_alignment`,
  `run_truncation_analysis`, `_run_study_parallel` (`--workers N` spawned lanes)),
  `run_classification_experiments.py` (loss comparison), `run_synthetic_experiments.py`.
- `vision_spectra/analysis/publication_figures.py` — reads MLflow → A-F figures/tables/stats;
  `study_figures.py` — reads the `spectral_*` sweep cells → width-sweep / alignment /
  truncation figures (`figures study`, folded into `figures all`).
- `vision_spectra/utils/` — `reproducibility.py` (re-exports canonical seed/device +
  `count_parameters`), `checkpointing.py`, `logging.py`, `visualization.py`.
- `tests/` — `test_data.py`, `test_losses.py`, `test_metrics.py`, `test_training.py`,
  `test_spectral_study.py`, `test_version.py` (plain pytest, **111 tests**; get the live count with
  `poetry run pytest --collect-only -q`).

## How to run

```bash
poetry install
vision-spectra --help

vision-spectra train-cls --dataset synthetic --epochs 2 --batch-size 8 --smoke-test
vision-spectra spectral run-all --num-seeds 10        # the 6-scenario study (≥10 seeds)
vision-spectra spectral run-study --set followup --num-seeds 10 --workers 4 --device cuda  # Sept-2026 controls
vision-spectra figures all                            # MLflow -> figures/tables/stats
vision-spectra experiments run --losses cross_entropy focal   # loss comparison
```

No Makefile. Tooling is invoked directly:
`poetry run pytest` · `poetry run ruff check vision_spectra tests` ·
`poetry run ruff format vision_spectra tests` · `poetry run mypy vision_spectra` ·
`poetry run pre-commit run --all-files`.

## CRITICAL invariants (these break silently — pytest stays green)

1. **Real-ViT architecture (`patch_size=4`).** With 28px inputs, `patch16` collapses to a
   single 1×1 patch token (a non-spatial "ViT"). All experiments now pass `patch_size=4`
   (→ a 7×7 = 49-patch grid). `create_model_for_scenario` routes ALL six scenarios through
   ONE parameterized `timm.create_model(..., patch_size=PATCH_SIZE, embed_dim, depth,
   num_heads)` and asserts `len(model.blocks) == config.depth` (A/B previously ran at timm's
   default 12 layers, not the documented 6). `ModelConfig.patch_size` threads through
   `create_vit_classifier`. Do not revert to patch16 at 28px or branch the scenario builder.
2. **Spectral metrics are two DIFFERENT quantities.** `alpha_exponent` is a RANK-DECAY slope
   (OLS of `log σ_i` vs `log rank`, bulk indices 10–60 %), values ~0.2–1.0 here.
   `power_law_alpha_hill` is the ESD/Hill tail index — the Martin & Mahoney "heavy-tail α"
   (the `α∈[2,6]` band applies ONLY to it). The headline `Δα = alpha_exponent_final −
   alpha_exponent_init`; the Hill metric is surfaced alongside (`pl_alpha_hill`). Never
   relabel the rank slope as the M&M α or apply the `[2,6]` band to it. SVD is float64 on CPU;
   `aggregate_spectral_metrics` NaN-filters and uses `ddof=1`; `SpectralTracker.load`
   normalizes cumulative variance by `Σσ²` (not `(Σσ)²`).
3. **MLflow ⇄ figures contract.** Scenario experiments are named `spectral_scenario_{A–F}`.
   `publication_figures.extract_scenario_metrics` reads metric HISTORIES
   `spectral/alpha_exponent_mean`, `spectral/stable_rank_mean`, `spectral/pl_alpha_hill_mean`
   (taking step-0 = init and the last step = final, so `Δ = last − first`), plus run-level
   `final/test_accuracy` (preferred) → `final/val_accuracy` (fallback). Renaming a logged key,
   or not logging the FINAL epoch's spectral metric, silently corrupts Δα. The stats
   (`perform_statistical_tests`) are Holm-Bonferroni corrected and **descriptive at n=3**.
4. **Training contract (`BaseTrainer`).** Every `torch.load` passes `weights_only=False`
   (checkpoints embed the Pydantic config = Path/enum objects; torch ≥ 2.6 default fails).
   AMP runs only when `self.use_amp` (CUDA) via `torch.amp` (scaler is `None` otherwise).
   Warmup is 0-based (callers pass `current_epoch - 1`), scales each param group off its own
   base LR, and hands off to the scheduler at `epoch >= warmup_epochs`; the optimizer /
   scheduler / warmup are defined ONCE in `build_optimizer` / `build_scheduler` /
   `warmup_factor` and reused (incl. by the spectral runner). `train()` restores the best
   checkpoint before returning, selects on a direction-aware `monitor` (loss = lower-better;
   accuracy/auroc/f1 = higher-better), and returns
   `{best_val_metric, best_epoch, best_checkpoint, best_checkpoint_uri, final_epoch,
   stopped_early, mlflow_run_id, spectral_tracker}`. `best_val_*` come from the post-train
   re-validation of the restored model.
5. **Held-out test reporting.** Runners evaluate the TEST split and report `test_*` (the
   unbiased estimate); validation is for model selection only. The spectral runner logs
   `final/test_accuracy`. Do not reintroduce best-of-K validation accuracy as the headline.
6. **Loss conventions.** Classification losses are `forward(logits, targets)`; build via
   `get_loss(loss_config, samples_per_class)`. `FocalLoss` with a SCALAR `alpha` is a global
   scale only (per-class balancing needs a tensor `alpha`). `AsymmetricLoss` is a multi-label
   sigmoid/BCE loss (it one-hots single-label targets) — NOT a clean peer of the softmax
   losses; flag any "loss → spectra" comparison that treats it as one. Reconstruction losses
   take an optional patch mask and divide by `mask.sum().clamp(min=1)`. `MIM_LOSS_REGISTRY`
   includes `huber`/`cauchy`/`tukey`; `MIMLossName` lists them.
7. **MIM / multitask / finetune.** `MIMModel`/`MultitaskViT.__init__` assert
   `image_size % patch_size == 0`; `random_masking` keeps `num_keep = max(1, min(N-1, …))`;
   both define `patchify`/`unpatchify`; the masked encoder reaches `encoder.encoder.*` timm
   internals. `FinetuneTrainer._load_pretrained` strips the MIM double-prefix
   (`encoder.encoder.X` → `encoder.X`) and skips ONLY the classification head via
   `_is_head_param` (head ≠ block `mlp.fc1/fc2`), with a `matched == 0` guard. These paths
   run via the CLI + tests but NOT in the reported 6-scenario study (which is cross-entropy
   only) — a "loss → spectra" or "MIM vs supervised" comparison is scaffolded, not yet run.
8. **Config is strict and single-source.** Every config inherits `_StrictModel`
   (`extra="forbid"` + `validate_assignment`), so a misspelled/misplaced YAML key or an
   out-of-range assignment fails fast. The CLI `--config` path overlays only flags the user
   explicitly passed, via `_passed(ctx, param)` (precedence CLI > YAML > defaults). Edit
   `settings.py`, not inline copies; `resolve_device`/`set_seed` are canonical there.
9. **Reproducibility reality.** `set_seed` seeds `random` + `numpy` + `torch` (+cuda) and
   enables `torch.use_deterministic_algorithms(warn_only=True)` on CPU/MPS too (not CUDA-
   gated). On Apple Silicon runs are not bit-reproducible — don't claim strict determinism.

## Dev workflow

- **ruff** (line-length 99) is the linter AND formatter; `ruff format --check` is enforced
  in CI, so run `ruff format` after edits (the `auto_format` hook does this on every saved
  `.py`). **mypy** is advisory (CI `continue-on-error`); the codebase carries many torch
  `Tensor | Module` false positives — don't chase them unless they reflect a real runtime bug.
- Tests are plain pytest (`Test<Thing>` classes, `parametrize`); they mostly assert
  shape/finiteness and use tiny models — so wrong spectral math, a broken MLflow key, or a
  reverted training contract can pass. A new source module gets a matching `test_<module>.py`.
- Datasets are 28×28 (synthetic generated on the fly; MedMNIST native); with `patch_size=4`
  the ViT tokenizes a 7×7 grid. ImageNet mean/std normalization is applied (a known caveat
  for these from-scratch medical/synthetic models).

## Repo-specific gotchas

- The 6-scenario `run_spectral_analysis.py` reports best-of-K val accuracy historically but
  now also logs `final/test_accuracy`; it trains a FIXED epoch budget (no early stopping) so
  Δα is measured at a common endpoint across scenarios — that's deliberate.
- `gradient_alignment.py` and `tail_truncation.py` are wired into `run_scenario_experiment`
  / `run-study` (logged as `alignment/*` and `truncation/*`); they are still NOT used by the
  loss/MIM experiment families. **Alignment has two probes from one SVD:** the legacy
  `cos_sim_mean` = cos(∇L, U Vᵀ) is bounded by 1/√rank and was ≈ 0 in every June-2026 cell
  (a null by construction — do not present it as a finding); the informative one is the
  subspace probe: `cos_head_mean` (>0 ⇔ the SGD step grows the top-k σ), `cos_tail_mean`
  (>0 ⇔ the step shrinks the tail = rank-reducing), `*_energy_enrichment_mean` (1 = no
  preference) and per-type `{q,k,v,proj,fc1,fc2}_cos_*`; the fused `attn.qkv` is split into
  Q/K/V to match the extraction (so `cos_sim_mean` values from runs after 2026-09-12 are not
  numerically comparable with the June-2026 store). `tail_truncation` has THREE probes: `bulk` (Eckart-Young,
  removes the SMALLEST σ — effective-rank probe), `head` by ratio (removes the LARGEST 10 %+;
  too coarse — collapses every model at its first grid point) and **head-drop** (absolute
  top-`n` σ per matrix, Q/K/V split; `truncation/headn_acc` step = n, per-group
  `headn_{qkv,proj,mlp}_acc`). `bulk` is not heavy-tail ablation; `head`/`headn` are.
- **Re-running a cell appends runs to the same MLflow experiment.** Both figure readers
  (`publication_figures.extract_scenario_metrics`, `study_figures.extract_study_cell`) keep the
  LATEST run per `params.seed` (`select_latest_run_per_seed`), so a rerun replaces, never
  double-counts. Run params now include `steps_per_epoch` / `total_steps` / `weight_decay` /
  `train_subsample`; follow-up runs log `model/final.pt` (state_dict + rebuild fields) and no
  histogram PNGs (`ScenarioConfig.log_histograms`, default off; SV JSON keeps everything).
- **Step-count confound.** PathMNIST ≈ 70k optimizer steps vs synthetic ≈ 1k; Blood/Derma sit
  between and the Hill ordering follows steps exactly, and cosine-to-zero LR forces a plateau at
  every run's end. `--set followup` holds the step-matched controls (`w192_synlong`,
  `w192_pathshort` via `DatasetConfig.train_subsample`, TRAIN-only), the wd-ablation cells and
  the width-sweep rerun. Synthetic images are rendered once into a uint8 cache
  (`SyntheticImageDataset(cache=True)`, identical tensors) — the June run was data-path bound,
  not SVD bound.
- `publication_figures.py` is a single ~1.2k-line module (MLflow extraction + stats +
  plotting + LaTeX + CLI). Splitting it is desirable but deferred; when editing, keep the
  metric-key reads in `extract_scenario_metrics` in sync with what the runners log.
- Cross-scenario accuracy mixes 3-class synthetic with 9-class PathMNIST (different chance
  levels); Δα-vs-accuracy is observational/confounded by capacity — keep the language
  correlational, never causal.
- **λ_decay reverses the width ordering.** Its 10–60 % fit window reaches into the tail of a
  48-dim spectrum, so it says "narrower = steeper" while α_Hill / relative Δr_s say "wider =
  heavier / more collapsed". The A–F `delta_alpha_bar` / `statistical_tests.json` still headline
  Δλ; treat them as secondary until switched to Δα_Hill / rel Δr_s (`EXPERIMENT_PLAN_V2` §9).

## What NOT to commit

`.gitignore` excludes `data/`, `runs/`, `mlruns/*`, `*.pt`, `*.pth`, `references/`,
notebooks. Datasets/model weights/MLflow artifacts must stay out of git — the
`block_large_secret` hook additionally blocks staging anything under `data/ runs/ mlruns/`,
binary/model/dataset extensions (`.pt`/`.pth`/`.npz`/`.npy`/`.pkl`/`.ckpt`/…), secrets, and
files > 10 MB. **Never `git add -f`** these (the hook blocks it). Pre-commit enforces
`check-added-large-files` (maxkb=1000) and `detect-private-key`. The package version lives
under `[tool.poetry]` in `pyproject.toml` (poetry-core backend) and is **patch-bumped on every
`/commit-push`** (`poetry version patch`; `--minor`/`--major`/`--no-bump` override).
`vision_spectra.__version__` reads the installed metadata (`importlib.metadata`), so never
hand-edit a version literal; `tests/test_version.py` asserts pyproject ⇄ `__version__` ⇄
`vision-spectra --version` agree (run `poetry install --only-root` after a bump).

## Claude Code setup in this repo

- **Skills** (`.claude/skills/`): `/code-review` (read-only review of the working tree
  against the invariants above; delegates to the subagents below) and `/commit-push` (gated
  review → tests → pre-commit → **patch version bump on every commit** → Conventional-Commits
  commit with a `Version: OLD -> NEW` line → push to `main`; `--release` adds a `v<version>` tag).
- **Subagents** (`.claude/agents/`): `spectral-metric-reviewer`,
  `experiment-reproducibility-auditor`, `training-contract-reviewer`, `loss-correctness-reviewer`.
- **Hooks** (`.claude/settings.json` → `.claude/hooks/`): auto-format edited `.py` with ruff;
  guard against destructive git (force-push, `reset --hard`, `--no-verify`, deleting `main`,
  AND any Claude/AI commit attribution — this project never lists Claude as an author); block
  staging large/secret/artifact files; run pytest on stop when source changed. Disable any
  hook by editing `.claude/settings.json`.
