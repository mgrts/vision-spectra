---
name: commit-push
description: Run code-review, the pytest suite, ruff lint+format, and pre-commit hooks; update docs if drifted; write a Conventional Commits message (NO Claude/AI attribution); commit and push to main on GitHub (origin mgrts/vision-spectra); optionally bump the [tool.poetry] version and push a release tag. Stops at every gate (failed review, failed tests, failed lint/format, conflicting rebase) and requires explicit confirmation before committing and pushing.
---

# Commit & push for vision-spectra

Analyze pending changes, review them, run the test suite + ruff + pre-commit hooks, update
docs if needed, write a Conventional Commits message, and push to `main`. Optionally bump the
package version and push a release tag.

The default branch is **`main`**; origin is **`git@github.com:mgrts/vision-spectra.git`**
(GitHub, owner `mgrts`). This is a solo research repo, so the default flow pushes directly to
`main` after gates pass and the user confirms.

## Arguments

`$ARGUMENTS` — optional. A free-form commit message (used verbatim as the subject after type
inference) and/or flags: `--no-push` (commit only), `--release` (also bump version and offer
a tag). There is **no issue tracker** — never invent ticket references.

## Important

- **Conventional Commits**: `type(scope): subject`. Types: `feat`, `fix`, `refactor`, `perf`,
  `test`, `docs`, `chore`, `build`, `ci`. Scope is optional but encouraged (e.g. `spectral`,
  `metrics`, `training`, `models`, `losses`, `data`, `experiments`, `cli`, `config`,
  `figures`, `analysis`).
- **NEVER** list Claude among commit authors. Do not add a `Co-Authored-By` trailer, set
  `--author` to Claude/Anthropic, use an `@anthropic.com` address, or add a "Generated with
  Claude" line — to the commit message OR a PR body. This is a hard project rule: the
  `guard_git` PreToolUse hook **blocks** any `git commit` carrying such attribution, so a slip
  is denied rather than committed.
- Do **NOT** use `--force`, `--no-verify`, or any destructive git flag. The repo's guard-git
  hook will block these anyway. If a step fails, stop and ask the user.

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
git branch --show-current
```

If there are no changes, stop: "Nothing to commit." If the current branch is not `main`, note
it and ask the user whether to proceed on this branch or switch.

### Step 2: Run the code-review skill

Invoke the `code-review` skill on the pending diff.

- **Critical / High** findings: stop. Show them and ask whether to proceed anyway, fix
  automatically, or cancel. Do not move on without explicit acknowledgement.
- **Medium / Low** findings: print as a heads-up and continue.

### Step 3: Run tests

```bash
poetry run pytest -q -p no:cacheprovider     # currently 87 tests
```

If tests fail: show failures, try to fix obvious causes from the diff (e.g. an import-path
drift after a rename, a pinned test value that must move with a config change), re-run. If
still failing, stop and ask.

### Step 4: Run ruff + pre-commit hooks

```bash
poetry run ruff check vision_spectra tests
poetry run ruff format --check vision_spectra tests
poetry run pre-commit run --all-files
```

If ruff/format/pre-commit fail: `ruff format` and the whitespace/eof hooks auto-fix on a
re-run — run `poetry run ruff format vision_spectra tests` then re-run the checks once. If
they still fail after one auto-fix pass, stop and ask. Never bypass with `--no-verify`. If
`check-added-large-files` or `detect-private-key` trips, do NOT force it through — surface the
offending file to the user.

### Step 5: Update documentation

Read `README.md` and `CLAUDE.md`; update only sections that drifted from reality:

- **New CLI command / sub-app** → README CLI reference + `CLAUDE.md` package map.
- **New `settings.py` config field or changed default** → README Configuration + CLAUDE.md.
- **New module under `vision_spectra/`** → `CLAUDE.md` package map (+ README structure).
- **Test count changed** → README "N tests" claim and `CLAUDE.md` (count via
  `poetry run pytest --collect-only -q`).
- **A CRITICAL invariant changed** (spectral metric semantics, MLflow keys, training
  contract, scenario/patch config, loss conventions) → update the relevant `CLAUDE.md`
  section.

If nothing drifted, skip this step. Do not rewrite docs that are already correct.

### Step 6: Optional version bump + release (only if `--release` or the user asks)

By default, do NOT bump the version on every commit. If a release is requested:

- Patch-bump `version` under **`[tool.poetry]`** in `pyproject.toml` (e.g. `0.1.0 → 0.1.1`).
- Form the tag `v<version>` — created in Step 10 after the push.

### Step 7: Generate the Conventional Commits message

**Subject** (≤ 72 chars): `type(scope): summary`. Infer the type from the diff:

- new capability (loss, metric, model, CLI command, experiment) → `feat`
- bug fix → `fix`
- behaviour-preserving restructure → `refactor`
- speed/memory → `perf`
- tests only → `test`
- docs / CLAUDE.md only → `docs`
- tooling / deps / version / Claude-Code config → `chore` / `build` / `ci`

If `$ARGUMENTS` supplied a message, use it verbatim as the subject (after the type).

**Body** (after a blank line): one line per significant change. If spectral-metric semantics,
MLflow keys, the training contract, scenario configs, or the loss registry changed, explicitly
note the synchronized test/consumer/doc updates so the contract reads as kept-whole. Add the
version line only if Step 6 bumped it:

```text
Version: 0.1.0 -> 0.1.1
```

**No AI-attribution trailer.**

### Step 8: Show summary and confirm

Print: code-review result, test result, ruff/format/pre-commit result, doc updates (or
"none"), version bump (or "none"), files to be committed (`git status --short`), and the full
commit message. Then ask with `AskUserQuestion`:

```text
question: "Commit and push to origin/main?"
header: "Commit & Push"
options:
  - "Yes" — stage all changes, commit, rebase onto origin/main, push.
  - "No"  — cancel, leave the working tree as-is.
```

Do NOT proceed without an explicit "Yes". If `--no-push` was passed, the option is
"Commit only (no push)".

### Step 9: Commit and push

```bash
git add -A
git commit -m "<subject>

<body>"
git fetch origin main
git rebase origin/main
```

If the rebase conflicts, **abort** (`git rebase --abort`) and tell the user to resolve
manually — do not auto-resolve. Then (unless `--no-push`):

```bash
git push origin main
```

If the push fails (branch protection, auth, network), do NOT retry and do NOT force. Show the
error and suggest pushing a feature branch + opening a PR
(`git switch -c <branch> && git push -u origin <branch> && gh pr create`).

### Step 10: Optional release tag

Only if Step 6 bumped the version. Read `version` from `pyproject.toml`, form `v<version>`.
Check it does not already exist:

```bash
git rev-parse "v<version>" 2>/dev/null
```

If it exists, surface that and skip. Otherwise confirm with `AskUserQuestion`, then:

```bash
git tag -a "v<version>" -m "Release v<version>"
git push origin "v<version>"
```

If the tag push fails, do NOT retry or delete the local tag; report that it exists locally and
can be pushed manually.

### Step 11: Final report

```text
Pushed to origin/main.
Review: passed (or: N findings)   Tests: 87 passed   Ruff/format: passed   Pre-commit: passed
Doc updates: <files or "none">
Version: <bump or "no bump">      Tag: <v.. pushed | skipped>
```

Or, if the push was blocked, show the error and the feature-branch + PR suggestion.
