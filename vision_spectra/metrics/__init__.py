"""Spectral metrics for analyzing transformer weight matrices."""

from vision_spectra.metrics.extraction import (
    WeightInfo,
    extract_all_weights,
    extract_attention_weights,
    extract_mlp_weights,
    extract_patch_embed_weights,
    extract_qkv_weights,
)
from vision_spectra.metrics.spectral import (
    EpochSpectralSnapshot,
    SpectralDistribution,
    SpectralTracker,
    aggregate_spectral_metrics,
    alpha_exponent,
    get_spectral_distribution,
    get_spectral_metrics,
    power_law_alpha_hill,
    spectral_entropy,
    stable_rank,
)

__all__ = [
    # Scalar metrics
    "spectral_entropy",
    "stable_rank",
    "alpha_exponent",
    "power_law_alpha_hill",
    "get_spectral_metrics",
    "aggregate_spectral_metrics",
    # Distribution tracking
    "SpectralDistribution",
    "EpochSpectralSnapshot",
    "SpectralTracker",
    "get_spectral_distribution",
    # Weight extraction
    "extract_qkv_weights",
    "extract_attention_weights",
    "extract_mlp_weights",
    "extract_patch_embed_weights",
    "extract_all_weights",
    "WeightInfo",
]
