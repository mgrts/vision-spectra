"""
Experiment scripts for Vision Spectra.

This module contains scripts for running systematic experiments
comparing different loss functions, training regimes, and their
effects on transformer weight spectra.

Available experiment modules:
    - run_classification_experiments: Compare loss functions on real medical imaging data
    - run_synthetic_experiments: Analyze spectral behavior on simple geometric shapes
    - run_spectral_analysis: Three-scenario framework (capacity × complexity)

Usage:
    # Classification experiments on MedMNIST
    poetry run python -m vision_spectra.experiments.run_classification_experiments run

    # Synthetic data experiments for spectral hypothesis testing
    poetry run python -m vision_spectra.experiments.run_synthetic_experiments run

    # Three-scenario spectral analysis (meeting notes implementation)
    poetry run python -m vision_spectra.experiments.run_spectral_analysis run-all

    # Compare complexity levels (simple vs complex data)
    poetry run python -m vision_spectra.experiments.run_synthetic_experiments compare-complexity
"""
