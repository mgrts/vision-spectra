"""Vision Spectra: Analyzing loss function effects on transformer weight spectra."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth is ``[tool.poetry] version`` in pyproject.toml; the installed
    # distribution metadata mirrors it (``poetry install`` refreshes it after a bump).
    __version__ = version("vision-spectra")
except PackageNotFoundError:  # running from a bare checkout without an install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
