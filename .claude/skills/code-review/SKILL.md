---
name: code-review
description: Review pending changes in the vision-spectra repo for correctness and the silent-bug classes this ML research codebase actually hits — spectral-metric semantics (rank-slope vs Hill ESD), the MLflow⇄figures key contract, the BaseTrainer training contract (AMP/warmup/scheduler/best-restore/weights_only), the real-ViT patch4/depth scenario config, loss math & registry sync, MIM/finetune masking & weight-loading, config strictness, and secrets/large-file hygiene. Read-only by default; surfaces findings grouped by severity. Use before every commit, or via /commit-push.
---

# Code review for vision-spectra

Review the changes currently in the working tree (staged + unstaged + untracked) against
the standards that matter for this codebase specifically. Most bugs here are **silent**:
they pass `pytest` (tests assert shape + finiteness on tiny models, and the suite never runs
a full scenario) yet change the spectral math, the prediction the model selects, the logged
metric keys, or the experiment's scientific validity. The job is to catch those.

The review is **read-only by default** — fixes are surfaced as recommendations and only
applied if the user explicitly asks.

## Arguments

`$ARGUMENTS` — optional. Specific files or globs to scope the review (defaults to the entire
diff).

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
```

If there is nothing pending, stop: "Nothing to review."

### Step 2: Read the diff

For each changed file read the actual diff (not just the file list) so the review reasons
about what changed. Note which subsystems are touched — that selects which checks below
apply and which subagent to delegate to.

### Step 3: Delegate deep audits to subagents

When the diff touches a fragile subsystem, dispatch the matching subagent (via the Agent
tool, `subagent_type`) and fold its findings into the report. Run independent subagents in
parallel.

- Touches `metrics/spectral.py`, `metrics/extraction.py`, `metrics/statistical.py`,
  `metrics/plotting.py`, or `analysis/publication_figures.py`
  → **`spectral-metric-reviewer`**.
- Touches `experiments/*.py`, the MLflow logging, `settings.py` scenario/config fields, or
  seed/reproducibility handling → **`experiment-reproducibility-auditor`**.
- Touches `training/*.py` or `models/*.py` (AMP, scheduler/warmup, best-restore, checkpoint,
  MIM masking, finetune weight-loading) → **`training-contract-reviewer`**.
- Touches `losses/*.py` → **`loss-correctness-reviewer`**.

For a small diff that clearly matches none of these, do the checks inline.

### Step 4: CRITICAL — Spectral-metric semantics

Applies to `metrics/spectral.py`, `metrics/extraction.py`, `analysis/publication_figures.py`.

- `alpha_exponent` is a **rank-decay slope** (OLS of `log σ_i` vs `log rank`, bulk 10–60 %),
  values ~0.2–1.0. `power_law_alpha_hill` is the **ESD/Hill tail index** (the Martin & Mahoney
  heavy-tail α, `[2,6]` band). They are different (even inversely related). Reject any diff
  that relabels the rank slope as the M&M α, applies the `[2,6]` band to it, or drops the
  Hill metric from the figures/summary.
- SVD is float64 on CPU; degenerate/empty matrices return NaN; `aggregate_spectral_metrics`
  NaN-filters and uses `ddof=1`. `SpectralTracker.load` normalizes cumulative variance by
  `(sv**2).sum()`, NOT `sv.sum()**2`.
- `Δα = alpha_exponent_final − alpha_exponent_init`; the baseline is at random init
  (Marchenko-Pastur, not a power law) — keep that caveat in any new interpretation text.

### Step 5: CRITICAL — MLflow ⇄ figures key contract

Applies to `experiments/run_spectral_analysis.py` and `analysis/publication_figures.py`.

- Scenario experiments are named `spectral_scenario_{A–F}`. `extract_scenario_metrics` reads
  the metric HISTORIES `spectral/alpha_exponent_mean`, `spectral/stable_rank_mean`,
  `spectral/pl_alpha_hill_mean` (step-0 = init, last step = final), plus run-level
  `final/test_accuracy` (preferred) → `final/val_accuracy`. A renamed key, or not logging the
  FINAL epoch's spectral metric, silently NaNs a column / staleness-shifts Δα. List any key
  that exists on only one side.
- The stats (`perform_statistical_tests`) must stay Holm-corrected and framed as descriptive
  at n=3; error bars are 95 % CIs (not ±1 SD). Reject reintroduced raw-p<0.05 "significant"
  framing or ±1 SD error bars.

### Step 6: CRITICAL — BaseTrainer training contract

Applies to `training/base.py` and the subclass trainers.

- Every `torch.load` passes `weights_only=False` (config-bearing checkpoints; torch ≥ 2.6).
- AMP runs only under `if self.use_amp:` (CUDA) via `torch.amp`; the scaler is `None`
  otherwise — no unguarded `self.scaler.*`.
- Warmup is 0-based (callers pass `current_epoch - 1`), scales each param group off its own
  base LR, and the scheduler steps when `epoch >= warmup_epochs`. The optimizer/scheduler/
  warmup come from `build_optimizer` / `build_scheduler` / `warmup_factor` — not a re-rolled
  copy.
- `train()` restores the best checkpoint before returning; `_is_best`/`best_val_metric` are
  direction-aware via `monitor`; `best_val_*` come from the post-train re-validation; the
  return dict carries `best_checkpoint_uri` / `mlflow_run_id` / `best_epoch` /
  `stopped_early`.

### Step 7: HIGH — Real-ViT architecture & scenario config

Applies to `experiments/run_spectral_analysis.py`, `models/vit.py`, `settings.py`.

- Scenarios build via ONE parameterized `timm.create_model(..., patch_size=PATCH_SIZE,
  embed_dim, depth, num_heads)` and assert `len(model.blocks) == config.depth`. Reject a
  reintroduced A/B special-case that omits `depth` (→ timm's default 12 layers) or a revert
  to `patch16` at 28px (→ a degenerate 1-patch model). `ModelConfig.patch_size` must thread
  through `create_vit_classifier`.
- Held-out TEST evaluation stays (runners report `test_*`; the spectral runner logs
  `final/test_accuracy`). Don't reintroduce best-of-K val accuracy as the headline.

### Step 8: HIGH — Loss math, registry & config sync

Applies to `losses/*.py` and `settings.py`.

- Classification losses are `forward(logits, targets)`. Scalar `FocalLoss.alpha` is a global
  scale only (flag any "class balancing" claim that uses a scalar). `AsymmetricLoss` is a
  multi-label sigmoid loss — flag treating it as a softmax-loss peer in a spectra comparison.
- Reconstruction masked losses divide by `mask.sum().clamp(min=1)`. A loss added to
  `LOSS_REGISTRY` / `MIM_LOSS_REGISTRY` is mirrored in `LossName` / `MIMLossName` and tested.

### Step 9: HIGH — MIM / multitask / finetune contracts

Applies to `models/mim.py`, `models/multitask.py`, `training/finetune.py`.

- `MIMModel`/`MultitaskViT.__init__` assert `image_size % patch_size == 0`; `random_masking`
  keeps `num_keep = max(1, min(N-1, …))`; both define `patchify` AND `unpatchify`.
- `FinetuneTrainer._load_pretrained` strips the `encoder.encoder.` double-prefix, skips ONLY
  the head via `_is_head_param` (NOT block `mlp.fc1/fc2`), and raises on `matched == 0`.

### Step 10: MEDIUM — Config strictness, reproducibility, hygiene & docs

- Config models inherit `_StrictModel` (`extra="forbid"` + `validate_assignment`); the CLI
  `--config` overlay uses `_passed(ctx, param)`. `resolve_device`/`set_seed` are edited in
  `settings.py` only (utils re-exports). `set_seed` enables deterministic algorithms on
  CPU/MPS (not CUDA-gated); don't claim strict determinism on Apple Silicon.
- No secrets / no hardcoded credentials. No staged files under `data/`, `runs/`, `mlruns/`;
  no `*.pt`/`*.pth`/`*.npz`/`*.npy`/`*.pkl`; no `git add -f`; no file > 10 MB.
- New public function/class has a docstring + type hints. If an invariant changed (spectral
  metric semantics, MLflow keys, training contract, scenario config), `CLAUDE.md` and
  `README.md` were updated in the same change.

### Step 11: Run tests, lint, format

Run and report exit status:

```bash
poetry run pytest -q -p no:cacheprovider     # currently 87 tests
poetry run ruff check vision_spectra tests
poetry run ruff format --check vision_spectra tests    # enforced in CI
```

Any failure is a critical finding. (`ruff format` fixes formatting on a re-run; report what
it changed.) Optionally run `poetry run mypy vision_spectra` — but mypy is advisory (CI
`continue-on-error`) and the codebase has many torch `Tensor | Module` false positives, so
only flag NEW mypy errors that reflect a real runtime bug.

### Step 12: Report findings

Group findings by severity:

- **Critical** — wrong spectral math, rank-slope/Hill conflation, broken MLflow⇄figures key
  contract, broken training contract (`weights_only`, AMP guard, best-restore, monitor),
  secret/large-file leak, failing test/ruff/format.
- **High** — reverted real-ViT/patch4 or A/B depth fix, best-of-K val as headline, loss
  math/registry desync, broken MIM/finetune contract, data-protocol regression.
- **Medium** — config-strictness/reproducibility violation, hygiene, docs (CLAUDE.md/README)
  drift.
- **Low** — comment / naming / docstring / type-hint polish.

For each finding: file path, the symbol or line, and a concrete suggestion. Do not make
changes unless the user asks.

If there are zero findings: report "Review passed — N files reviewed, M lines changed,
pytest <result>, ruff <result>, ruff-format <result>."
