"""
Tests for spectral metrics.
"""

import numpy as np
import torch


class TestSpectralEntropy:
    """Tests for spectral entropy computation."""

    def test_identity_matrix(self):
        """Test entropy of identity matrix (uniform singular values)."""
        from vision_spectra.metrics import spectral_entropy

        # Identity has all singular values = 1
        identity_matrix = np.eye(10)
        entropy = spectral_entropy(identity_matrix)

        # Uniform distribution should have max entropy = log(n)
        max_entropy = np.log(10)

        assert np.isfinite(entropy)
        assert entropy > 0
        assert np.isclose(entropy, max_entropy, rtol=1e-4)

    def test_rank_one_matrix(self):
        """Test entropy of rank-1 matrix (single singular value)."""
        from vision_spectra.metrics import spectral_entropy

        # Rank-1 matrix: outer product
        u = np.random.randn(10, 1)
        W = u @ u.T

        entropy = spectral_entropy(W)

        # Rank-1 should have entropy close to 0
        assert np.isfinite(entropy)
        assert entropy < 0.5

    def test_random_matrix(self):
        """Test entropy of random matrix."""
        from vision_spectra.metrics import spectral_entropy

        np.random.seed(42)
        W = np.random.randn(64, 64)

        entropy = spectral_entropy(W)

        assert np.isfinite(entropy)
        assert entropy > 0

    def test_rectangular_matrix(self):
        """Test with non-square matrix."""
        from vision_spectra.metrics import spectral_entropy

        W = np.random.randn(32, 128)
        entropy = spectral_entropy(W)

        assert np.isfinite(entropy)
        assert entropy > 0

    def test_invalid_input(self):
        """Test with invalid input."""
        from vision_spectra.metrics import spectral_entropy

        # 1D array should return nan
        v = np.random.randn(10)
        assert np.isnan(spectral_entropy(v))


class TestStableRank:
    """Tests for stable rank computation."""

    def test_identity_matrix(self):
        """Test stable rank of identity matrix."""
        from vision_spectra.metrics import stable_rank

        identity_matrix = np.eye(10)
        sr = stable_rank(identity_matrix)

        # Identity has stable rank = n (all SVs equal)
        assert np.isfinite(sr)
        assert np.isclose(sr, 10.0, rtol=1e-4)

    def test_rank_one_matrix(self):
        """Test stable rank of rank-1 matrix."""
        from vision_spectra.metrics import stable_rank

        u = np.random.randn(10, 1)
        W = u @ u.T

        sr = stable_rank(W)

        # Rank-1 should have stable rank = 1
        assert np.isfinite(sr)
        assert np.isclose(sr, 1.0, rtol=1e-4)

    def test_bounds(self):
        """Test that stable rank is in [1, min(m, n)]."""
        from vision_spectra.metrics import stable_rank

        W = np.random.randn(30, 50)
        sr = stable_rank(W)

        assert sr >= 1.0
        assert sr <= 30  # min(30, 50)


class TestAlphaExponent:
    """Tests for alpha exponent estimation."""

    def test_power_law_decay(self):
        """Test with known power-law singular values."""
        from vision_spectra.metrics import alpha_exponent

        # Create matrix with known power-law singular values
        n = 100
        true_alpha = 2.0

        # SVD: W = U @ S @ Vt
        U = np.linalg.qr(np.random.randn(n, n))[0]
        Vt = np.linalg.qr(np.random.randn(n, n))[0]

        # Singular values following power law: s_i = i^(-alpha)
        ranks = np.arange(1, n + 1)
        s = ranks.astype(float) ** (-true_alpha)

        W = U @ np.diag(s) @ Vt

        alpha = alpha_exponent(W)

        assert np.isfinite(alpha)
        # Should be reasonably close to true alpha
        assert abs(alpha - true_alpha) < 1.0

    def test_uniform_singular_values(self):
        """Test with uniform singular values (no decay)."""
        from vision_spectra.metrics import alpha_exponent

        identity_matrix = np.eye(50)
        alpha = alpha_exponent(identity_matrix)

        # Uniform should have alpha near 0
        assert np.isfinite(alpha)
        assert abs(alpha) < 1.0

    def test_small_matrix(self):
        """Test with small matrix (may return nan)."""
        from vision_spectra.metrics import alpha_exponent

        W = np.random.randn(4, 4)
        result = alpha_exponent(W)

        # Small matrices may not have enough data for fitting
        # Result can be nan or finite
        assert result is not None


class TestHillAlpha:
    """Tests for Hill estimator."""

    def test_basic(self):
        """Test basic Hill estimator."""
        from vision_spectra.metrics import power_law_alpha_hill

        np.random.seed(42)
        W = np.random.randn(100, 100)

        alpha = power_law_alpha_hill(W)

        assert np.isfinite(alpha)
        assert alpha > 0

    def test_small_matrix(self):
        """Test with small matrix."""
        from vision_spectra.metrics import power_law_alpha_hill

        W = np.random.randn(5, 5)
        result = power_law_alpha_hill(W)

        # May return nan for very small matrices
        assert result is not None


class TestGetSpectralMetrics:
    """Tests for combined metric computation."""

    def test_all_metrics_present(self):
        """Test that all metrics are returned."""
        from vision_spectra.metrics import get_spectral_metrics

        W = np.random.randn(64, 64)
        metrics = get_spectral_metrics(W)

        expected_keys = ["spectral_entropy", "stable_rank", "alpha_exponent", "pl_alpha_hill"]

        for key in expected_keys:
            assert key in metrics

    def test_with_torch_tensor(self):
        """Test with PyTorch tensor input."""
        from vision_spectra.metrics import get_spectral_metrics

        W = torch.randn(32, 32)
        metrics = get_spectral_metrics(W)

        assert "spectral_entropy" in metrics
        assert np.isfinite(metrics["spectral_entropy"])

    def test_metrics_are_finite(self):
        """Test that all metrics are finite for normal matrices."""
        from vision_spectra.metrics import get_spectral_metrics

        np.random.seed(42)
        W = np.random.randn(50, 50)
        metrics = get_spectral_metrics(W)

        for key, value in metrics.items():
            assert np.isfinite(value), f"{key} is not finite"


class TestWeightExtraction:
    """Tests for weight extraction from models."""

    def test_extract_qkv_weights(self):
        """Test extracting Q/K/V weights from ViT."""
        from vision_spectra.metrics import extract_qkv_weights
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        weights = extract_qkv_weights(model)

        # Should have Q, K, V for each block
        assert len(weights) > 0

        # Check weight info structure
        for w in weights:
            assert w.matrix_type in ["q", "k", "v"]
            assert w.weight.ndim == 2

    def test_extract_with_pattern(self):
        """Test extracting weights with layer pattern."""
        from vision_spectra.metrics import extract_qkv_weights
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        # Only blocks 0 and 2
        weights = extract_qkv_weights(model, layer_patterns=["blocks.0", "blocks.2"])

        # Should only have weights from specified blocks
        for w in weights:
            assert "blocks.0" in w.name or "blocks.2" in w.name

    def test_extract_patch_embed(self):
        """Test extracting patch embedding weights."""
        from vision_spectra.metrics import extract_patch_embed_weights
        from vision_spectra.models import create_vit_classifier
        from vision_spectra.settings import ModelConfig

        config = ModelConfig(name="vit_tiny_patch16_224")
        model = create_vit_classifier(config, num_classes=10, num_channels=3, image_size=32)

        weights = extract_patch_embed_weights(model)

        assert len(weights) > 0
        assert weights[0].matrix_type == "patch_embed"


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_ill_conditioned_matrix(self):
        """Test with ill-conditioned matrix."""
        from vision_spectra.metrics import get_spectral_metrics

        # Create ill-conditioned matrix
        U = np.random.randn(50, 50)
        s = np.logspace(0, -10, 50)  # Huge condition number
        V = np.random.randn(50, 50)
        W = U @ np.diag(s) @ V

        metrics = get_spectral_metrics(W)

        # Should handle gracefully
        for value in metrics.values():
            assert not np.isnan(value) or np.isfinite(value)

    def test_near_zero_matrix(self):
        """Test with near-zero matrix."""
        from vision_spectra.metrics import get_spectral_metrics

        W = np.random.randn(32, 32) * 1e-10

        metrics = get_spectral_metrics(W)

        # Should not crash, may return nan for some metrics
        assert isinstance(metrics, dict)

    def test_large_values(self):
        """Test with large values."""
        from vision_spectra.metrics import get_spectral_metrics

        W = np.random.randn(32, 32) * 1e6

        metrics = get_spectral_metrics(W)

        # Should handle large values
        assert "spectral_entropy" in metrics
