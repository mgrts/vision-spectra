"""The package version is single-sourced from pyproject.toml.

``/commit-push`` bumps ``[tool.poetry] version`` on every commit; ``vision_spectra.__version__``
must follow it (via the installed metadata) so ``vision-spectra --version`` never drifts.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from typer.testing import CliRunner

import vision_spectra
from vision_spectra.cli import app

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["tool"]["poetry"]["version"]


def test_dunder_version_matches_pyproject() -> None:
    assert vision_spectra.__version__ == _pyproject_version(), (
        "vision_spectra.__version__ lags pyproject.toml — run `poetry install --only-root` "
        "after bumping the version"
    )


def test_cli_version_flag_prints_pyproject_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert _pyproject_version() in result.output
