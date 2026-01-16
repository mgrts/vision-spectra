"""
Tests for loss functions.
"""

import numpy as np
import pytest
import torch


class TestCrossEntropyLoss:
    """Tests for cross-entropy loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from vision_spectra.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()

        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_with_weights(self):
        """Test with class weights."""
        from vision_spectra.losses import CrossEntropyLoss

        weights = torch.ones(10)
        weights[0] = 2.0  # Double weight for class 0

        loss_fn = CrossEntropyLoss(weight=weights)

        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)


class TestFocalLoss:
    """Tests for focal loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from vision_spectra.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)

        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_gamma_zero_equals_ce(self):
        """Test that gamma=0 approximates cross-entropy."""
        from vision_spectra.losses import CrossEntropyLoss, FocalLoss

        torch.manual_seed(42)

        focal = FocalLoss(gamma=0.0)
        ce = CrossEntropyLoss()

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        focal_loss = focal(logits, targets)
        ce_loss = ce(logits, targets)

        # Should be approximately equal
        assert torch.allclose(focal_loss, ce_loss, rtol=1e-4)

    def test_higher_gamma_lower_easy_samples(self):
        """Test that higher gamma down-weights easy samples."""
        from vision_spectra.losses import FocalLoss

        # Create confident predictions
        logits = torch.zeros(8, 5)
        logits[:, 0] = 10.0  # High confidence for class 0
        targets = torch.zeros(8, dtype=torch.long)  # All class 0

        focal_low = FocalLoss(gamma=0.0)
        focal_high = FocalLoss(gamma=4.0)

        loss_low = focal_low(logits, targets)
        loss_high = focal_high(logits, targets)

        # Higher gamma should give lower loss for easy samples
        assert loss_high < loss_low


class TestLabelSmoothingLoss:
    """Tests for label smoothing loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from vision_spectra.losses import LabelSmoothingLoss

        loss_fn = LabelSmoothingLoss(epsilon=0.1)

        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_epsilon_zero_equals_ce(self):
        """Test that epsilon=0 equals cross-entropy."""
        from vision_spectra.losses import CrossEntropyLoss, LabelSmoothingLoss

        torch.manual_seed(42)

        ls = LabelSmoothingLoss(epsilon=0.0)
        ce = CrossEntropyLoss()

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        ls_loss = ls(logits, targets)
        ce_loss = ce(logits, targets)

        assert torch.allclose(ls_loss, ce_loss, rtol=1e-4)

    def test_invalid_epsilon(self):
        """Test invalid epsilon raises error."""
        from vision_spectra.losses import LabelSmoothingLoss

        with pytest.raises(ValueError):
            LabelSmoothingLoss(epsilon=1.5)

        with pytest.raises(ValueError):
            LabelSmoothingLoss(epsilon=-0.1)


class TestClassBalancedLoss:
    """Tests for class-balanced loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from vision_spectra.losses import ClassBalancedLoss

        samples_per_class = [1000, 100, 10]  # Imbalanced
        loss_fn = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.999)

        logits = torch.randn(8, 3)
        targets = torch.randint(0, 3, (8,))

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_weights_minority_boosted(self):
        """Test that minority classes get higher weights."""
        from vision_spectra.losses import ClassBalancedLoss

        samples_per_class = [1000, 10]  # Class 1 is minority
        loss_fn = ClassBalancedLoss(samples_per_class=samples_per_class, beta=0.999)

        # Weight for minority class should be higher
        assert loss_fn.weights[1] > loss_fn.weights[0]


class TestAsymmetricLoss:
    """Tests for asymmetric loss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from vision_spectra.losses import AsymmetricLoss

        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0)

        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))  # Single-label converted to multi-label

        loss = loss_fn(logits, targets)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0


class TestLossRegistry:
    """Tests for loss registry."""

    def test_get_cross_entropy(self):
        """Test getting cross-entropy from registry."""
        from vision_spectra.losses import CrossEntropyLoss, get_loss
        from vision_spectra.settings import LossConfig, LossName

        config = LossConfig(classification=LossName.CROSS_ENTROPY)
        loss = get_loss(config)

        assert isinstance(loss, CrossEntropyLoss)

    def test_get_focal(self):
        """Test getting focal loss from registry."""
        from vision_spectra.losses import FocalLoss, get_loss
        from vision_spectra.settings import LossConfig, LossName

        config = LossConfig(classification=LossName.FOCAL, focal_gamma=3.0)
        loss = get_loss(config)

        assert isinstance(loss, FocalLoss)
        assert loss.gamma == 3.0

    def test_get_label_smoothing(self):
        """Test getting label smoothing loss."""
        from vision_spectra.losses import LabelSmoothingLoss, get_loss
        from vision_spectra.settings import LossConfig, LossName

        config = LossConfig(
            classification=LossName.LABEL_SMOOTHING,
            label_smoothing=0.2,
        )
        loss = get_loss(config)

        assert isinstance(loss, LabelSmoothingLoss)
        assert loss.epsilon == 0.2

    def test_get_class_balanced_requires_counts(self):
        """Test that class-balanced loss requires sample counts."""
        from vision_spectra.losses import get_loss
        from vision_spectra.settings import LossConfig, LossName

        config = LossConfig(classification=LossName.CLASS_BALANCED)

        with pytest.raises(ValueError, match="samples_per_class"):
            get_loss(config)

    def test_get_class_balanced_with_counts(self):
        """Test getting class-balanced loss with counts."""
        from vision_spectra.losses import ClassBalancedLoss, get_loss
        from vision_spectra.settings import LossConfig, LossName

        config = LossConfig(classification=LossName.CLASS_BALANCED)
        counts = np.array([100, 50, 25])

        loss = get_loss(config, samples_per_class=counts)

        assert isinstance(loss, ClassBalancedLoss)


class TestMIMLosses:
    """Tests for MIM reconstruction losses."""

    def test_mse_loss(self):
        """Test MSE loss for MIM."""
        from vision_spectra.losses import MSELoss

        loss_fn = MSELoss()

        pred = torch.randn(8, 49, 48)  # Patches
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_mse_with_mask(self):
        """Test MSE loss with mask."""
        from vision_spectra.losses import MSELoss

        loss_fn = MSELoss()

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)
        mask = torch.zeros(8, 49)
        mask[:, :25] = 1  # Mask first 25 patches

        loss = loss_fn(pred, target, mask=mask)

        assert torch.isfinite(loss)

    def test_l1_loss(self):
        """Test L1 loss for MIM."""
        from vision_spectra.losses import L1Loss

        loss_fn = L1Loss()

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_smooth_l1_loss(self):
        """Test Smooth L1 loss for MIM."""
        from vision_spectra.losses import SmoothL1Loss

        loss_fn = SmoothL1Loss(beta=1.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_cauchy_loss(self):
        """Test Cauchy loss for robust reconstruction."""
        from vision_spectra.losses import CauchyLoss

        loss_fn = CauchyLoss(gamma=1.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_cauchy_loss_with_mask(self):
        """Test Cauchy loss with mask."""
        from vision_spectra.losses import CauchyLoss

        loss_fn = CauchyLoss(gamma=2.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)
        mask = torch.zeros(8, 49)
        mask[:, :25] = 1

        loss = loss_fn(pred, target, mask=mask)

        assert torch.isfinite(loss)

    def test_cauchy_outlier_robustness(self):
        """Test that Cauchy loss is more robust to outliers than MSE."""
        from vision_spectra.losses import CauchyLoss, MSELoss

        cauchy_fn = CauchyLoss(gamma=1.0)
        mse_fn = MSELoss()

        # Normal data
        pred = torch.zeros(100)
        target = torch.randn(100) * 0.1

        cauchy_normal = cauchy_fn(pred, target)
        mse_normal = mse_fn(pred, target)

        # Add outlier
        target_outlier = target.clone()
        target_outlier[0] = 10.0  # Large outlier

        cauchy_outlier = cauchy_fn(pred, target_outlier)
        mse_outlier = mse_fn(pred, target_outlier)

        # MSE should increase much more than Cauchy
        mse_ratio = mse_outlier / mse_normal
        cauchy_ratio = cauchy_outlier / cauchy_normal

        assert cauchy_ratio < mse_ratio  # Cauchy is more robust

    def test_sgt_loss(self):
        """Test SGT loss for heavy-tailed reconstruction."""
        from vision_spectra.losses import SGTLoss

        loss_fn = SGTLoss(p=2.0, q=2.0, lam=0.0, sigma=1.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_sgt_loss_with_mask(self):
        """Test SGT loss with mask."""
        from vision_spectra.losses import SGTLoss

        loss_fn = SGTLoss(p=2.0, q=3.0, lam=0.1, sigma=0.5)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)
        mask = torch.zeros(8, 49)
        mask[:, :25] = 1

        loss = loss_fn(pred, target, mask=mask)

        assert torch.isfinite(loss)

    def test_sgt_skewness(self):
        """Test SGT loss with different skewness values."""
        from vision_spectra.losses import SGTLoss

        # Test symmetric
        loss_sym = SGTLoss(p=2.0, q=2.0, lam=0.0)

        # Test positive skew
        loss_pos = SGTLoss(p=2.0, q=2.0, lam=0.5)

        # Test negative skew
        loss_neg = SGTLoss(p=2.0, q=2.0, lam=-0.5)

        pred = torch.zeros(100)
        target = torch.randn(100)

        # All should produce valid losses
        for loss_fn in [loss_sym, loss_pos, loss_neg]:
            loss = loss_fn(pred, target)
            assert torch.isfinite(loss)
            assert loss >= 0

    def test_sgt_invalid_params(self):
        """Test SGT loss parameter validation."""
        from vision_spectra.losses import SGTLoss

        # Invalid p
        with pytest.raises(ValueError):
            SGTLoss(p=-1.0)

        # Invalid q
        with pytest.raises(ValueError):
            SGTLoss(q=0.0)

        # Invalid lambda
        with pytest.raises(ValueError):
            SGTLoss(lam=1.5)

        # Invalid sigma
        with pytest.raises(ValueError):
            SGTLoss(sigma=-1.0)

    def test_cauchy_invalid_gamma(self):
        """Test Cauchy loss parameter validation."""
        from vision_spectra.losses import CauchyLoss

        with pytest.raises(ValueError):
            CauchyLoss(gamma=-1.0)

        with pytest.raises(ValueError):
            CauchyLoss(gamma=0.0)

    def test_huber_loss(self):
        """Test Huber loss for robust reconstruction."""
        from vision_spectra.losses import HuberLoss

        loss_fn = HuberLoss(delta=1.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_huber_loss_with_mask(self):
        """Test Huber loss with mask."""
        from vision_spectra.losses import HuberLoss

        loss_fn = HuberLoss(delta=0.5)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)
        mask = torch.zeros(8, 49)
        mask[:, :25] = 1

        loss = loss_fn(pred, target, mask=mask)

        assert torch.isfinite(loss)

    def test_huber_quadratic_linear_transition(self):
        """Test that Huber loss transitions from quadratic to linear."""
        from vision_spectra.losses import HuberLoss

        delta = 1.0
        loss_fn = HuberLoss(delta=delta, reduction="none")

        # Small errors (quadratic region)
        pred = torch.zeros(10)
        target_small = torch.full((10,), 0.5)  # |error| = 0.5 < delta
        loss_small = loss_fn(pred, target_small)

        # Expected: 0.5 * 0.5^2 = 0.125
        expected_small = 0.5 * 0.5**2
        assert torch.allclose(loss_small, torch.full_like(loss_small, expected_small))

        # Large errors (linear region)
        target_large = torch.full((10,), 2.0)  # |error| = 2.0 > delta
        loss_large = loss_fn(pred, target_large)

        # Expected: delta * (|error| - 0.5 * delta) = 1.0 * (2.0 - 0.5) = 1.5
        expected_large = delta * (2.0 - 0.5 * delta)
        assert torch.allclose(loss_large, torch.full_like(loss_large, expected_large))

    def test_huber_invalid_delta(self):
        """Test Huber loss parameter validation."""
        from vision_spectra.losses import HuberLoss

        with pytest.raises(ValueError):
            HuberLoss(delta=-1.0)

        with pytest.raises(ValueError):
            HuberLoss(delta=0.0)

    def test_tukey_loss(self):
        """Test Tukey biweight loss for robust reconstruction."""
        from vision_spectra.losses import TukeyLoss

        loss_fn = TukeyLoss(c=4.685)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)

        loss = loss_fn(pred, target)

        assert torch.isfinite(loss)
        assert loss >= 0

    def test_tukey_loss_with_mask(self):
        """Test Tukey loss with mask."""
        from vision_spectra.losses import TukeyLoss

        loss_fn = TukeyLoss(c=2.0)

        pred = torch.randn(8, 49, 48)
        target = torch.randn(8, 49, 48)
        mask = torch.zeros(8, 49)
        mask[:, :25] = 1

        loss = loss_fn(pred, target, mask=mask)

        assert torch.isfinite(loss)

    def test_tukey_outlier_rejection(self):
        """Test that Tukey loss rejects outliers (constant loss beyond threshold)."""
        from vision_spectra.losses import TukeyLoss

        c = 2.0
        loss_fn = TukeyLoss(c=c, reduction="none")

        pred = torch.zeros(10)

        # Errors at the boundary
        target_boundary = torch.full((10,), c)
        loss_boundary = loss_fn(pred, target_boundary)

        # Errors beyond the boundary (should have same loss)
        target_outlier = torch.full((10,), c * 10)  # 10x the threshold
        loss_outlier = loss_fn(pred, target_outlier)

        # Both should be at max loss = c^2 / 6
        max_loss = c**2 / 6.0
        assert torch.allclose(loss_boundary, torch.full_like(loss_boundary, max_loss), rtol=1e-4)
        assert torch.allclose(loss_outlier, torch.full_like(loss_outlier, max_loss), rtol=1e-4)

    def test_tukey_zero_error(self):
        """Test Tukey loss with zero error."""
        from vision_spectra.losses import TukeyLoss

        loss_fn = TukeyLoss(c=4.685, reduction="none")

        pred = torch.randn(10)
        target = pred.clone()  # Zero error

        loss = loss_fn(pred, target)

        # Zero error should give zero loss
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)

    def test_tukey_invalid_c(self):
        """Test Tukey loss parameter validation."""
        from vision_spectra.losses import TukeyLoss

        with pytest.raises(ValueError):
            TukeyLoss(c=-1.0)

        with pytest.raises(ValueError):
            TukeyLoss(c=0.0)

    def test_robust_loss_comparison(self):
        """Compare robustness of different losses to outliers."""
        from vision_spectra.losses import CauchyLoss, HuberLoss, MSELoss, TukeyLoss

        mse_fn = MSELoss()
        huber_fn = HuberLoss(delta=1.0)
        cauchy_fn = CauchyLoss(gamma=1.0)
        tukey_fn = TukeyLoss(c=4.685)

        # Normal data
        torch.manual_seed(42)
        pred = torch.zeros(100)
        target = torch.randn(100) * 0.5

        losses_normal = {
            "mse": mse_fn(pred, target).item(),
            "huber": huber_fn(pred, target).item(),
            "cauchy": cauchy_fn(pred, target).item(),
            "tukey": tukey_fn(pred, target).item(),
        }

        # Add outliers
        target_outlier = target.clone()
        target_outlier[:5] = 100.0  # 5% gross outliers

        losses_outlier = {
            "mse": mse_fn(pred, target_outlier).item(),
            "huber": huber_fn(pred, target_outlier).item(),
            "cauchy": cauchy_fn(pred, target_outlier).item(),
            "tukey": tukey_fn(pred, target_outlier).item(),
        }

        # Compute increase ratios
        ratios = {k: losses_outlier[k] / losses_normal[k] for k in losses_normal}

        # MSE should be most affected, Tukey least affected
        assert ratios["mse"] > ratios["huber"]
        assert ratios["huber"] > ratios["cauchy"]
        # Tukey should be most robust (nearly constant for outliers)
