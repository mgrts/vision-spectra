"""Spectral metrics for analyzing transformer weight matrices."""

from vision_spectra.metrics.extraction import (
    WeightInfo,
    extract_all_weights,
    extract_attention_weights,
    extract_mlp_weights,
    extract_patch_embed_weights,
    extract_qkv_weights,
)
from vision_spectra.metrics.gradient_alignment import (
    GradientAlignmentResult,
    GradientAlignmentTracker,
    aggregate_gradient_alignment,
    analyze_model_gradient_alignment,
    compute_gradient_alignment,
    compute_rank_reducing_gradient,
)
from vision_spectra.metrics.plotting import (
    generate_spectral_report,
    plot_ccdf,
    plot_layer_heatmap,
    plot_loglog_rank,
    plot_scenario_comparison,
    plot_spectral_evolution,
    plot_sv_distribution_comparison,
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
from vision_spectra.metrics.statistical import (
    ComparisonResult,
    cohens_d,
    compare_groups,
    compare_scenarios,
    format_comparison_table,
    summarize_results,
    validate_hypothesis,
)
from vision_spectra.metrics.tail_truncation import (
    TruncationResult,
    analyze_truncation_results,
    run_truncation_experiment,
    save_truncation_report,
    truncate_all_attention_layers,
    truncate_by_energy,
    truncate_model_layer,
    truncate_weight_matrix,
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
    # Gradient alignment
    "GradientAlignmentResult",
    "GradientAlignmentTracker",
    "compute_rank_reducing_gradient",
    "compute_gradient_alignment",
    "analyze_model_gradient_alignment",
    "aggregate_gradient_alignment",
    # Plotting
    "plot_ccdf",
    "plot_loglog_rank",
    "plot_spectral_evolution",
    "plot_layer_heatmap",
    "plot_scenario_comparison",
    "plot_sv_distribution_comparison",
    "generate_spectral_report",
    # Statistical analysis
    "ComparisonResult",
    "cohens_d",
    "compare_groups",
    "compare_scenarios",
    "format_comparison_table",
    "summarize_results",
    "validate_hypothesis",
    # Tail truncation
    "TruncationResult",
    "truncate_weight_matrix",
    "truncate_by_energy",
    "truncate_model_layer",
    "truncate_all_attention_layers",
    "run_truncation_experiment",
    "analyze_truncation_results",
    "save_truncation_report",
]
