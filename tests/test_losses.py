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
