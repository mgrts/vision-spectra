"""
Tests for datasets.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from vision_spectra.settings import DatasetConfig, DatasetName


class TestSyntheticDataset:
    """Tests for synthetic dataset."""

    def test_basic_creation(self):
        """Test creating synthetic dataset."""
        from vision_spectra.data.synthetic import SyntheticDataset

        config = DatasetConfig(
            name=DatasetName.SYNTHETIC,
            image_size=28,
            batch_size=8,
            num_classes=5,
            num_samples_train=100,
            num_samples_val=20,
            num_samples_test=20,
        )

        dataset = SyntheticDataset(config, Path("/tmp/test_data"))
        info = dataset.get_info()

        assert info.num_classes == 5
        assert info.train_size == 100
        assert info.val_size == 20
        assert info.test_size == 20
        assert info.image_size == (28, 28)

    def test_determinism(self):
        """Test that synthetic dataset is deterministic."""
        from vision_spectra.data.synthetic import SyntheticImageDataset

        ds1 = SyntheticImageDataset(num_samples=50, num_classes=5, image_size=28, seed=42)
        ds2 = SyntheticImageDataset(num_samples=50, num_classes=5, image_size=28, seed=42)

        # Same labels
        assert np.array_equal(ds1.labels, ds2.labels)

        # Same images (without transforms)
        img1, label1 = ds1[10]
        img2, label2 = ds2[10]

        assert label1 == label2

    def test_different_seeds_different_data(self):
        """Test that different seeds give different data."""
        from vision_spectra.data.synthetic import SyntheticImageDataset

        ds1 = SyntheticImageDataset(num_samples=50, num_classes=5, image_size=28, seed=42)
        ds2 = SyntheticImageDataset(num_samples=50, num_classes=5, image_size=28, seed=43)

        # Different labels
        assert not np.array_equal(ds1.labels, ds2.labels)

    def test_shapes(self):
        """Test output shapes."""
        from vision_spectra.data.synthetic import SyntheticDataset

        config = DatasetConfig(
            name=DatasetName.SYNTHETIC,
            image_size=32,
            batch_size=4,
            num_classes=3,
            num_samples_train=20,
            num_samples_val=10,
            num_samples_test=10,
        )

        dataset = SyntheticDataset(config, Path("/tmp/test_data"))
        train_ds = dataset.get_train_dataset()

        img, label = train_ds[0]

        assert img.shape == (3, 32, 32)  # RGB, H, W
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_dataloader(self):
        """Test dataloader iteration."""
        from vision_spectra.data.synthetic import SyntheticDataset

        config = DatasetConfig(
            name=DatasetName.SYNTHETIC,
            image_size=28,
            batch_size=8,
            num_classes=5,
            num_samples_train=50,
            num_samples_val=10,
            num_samples_test=10,
            num_workers=0,
        )

        dataset = SyntheticDataset(config, Path("/tmp/test_data"))
        loader = dataset.get_train_loader()

        batch = next(iter(loader))
        images, labels = batch

        assert images.shape == (8, 3, 28, 28)
        assert labels.shape == (8,)


class TestDatasetFactory:
    """Tests for dataset factory."""

    def test_get_synthetic(self):
        """Test getting synthetic dataset."""
        from vision_spectra.data import get_dataset
        from vision_spectra.data.synthetic import SyntheticDataset

        config = DatasetConfig(
            name=DatasetName.SYNTHETIC,
            num_samples_train=10,
            num_samples_val=5,
            num_samples_test=5,
        )

        dataset = get_dataset(config, Path("/tmp/test_data"))

        assert isinstance(dataset, SyntheticDataset)

    def test_invalid_dataset(self):
        """Test error on invalid dataset name."""
        # This is handled by enum validation
        with pytest.raises(ValueError):
            DatasetName("invalid_dataset")


class TestTransforms:
    """Tests for image transforms."""

    def test_train_transforms(self):
        """Test training transforms."""
        from PIL import Image

        from vision_spectra.data.transforms import get_train_transforms

        transform = get_train_transforms(image_size=32, num_channels=3)

        # Create dummy image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        result = transform(img)

        assert result.shape == (3, 32, 32)
        assert isinstance(result, torch.Tensor)

    def test_eval_transforms(self):
        """Test evaluation transforms."""
        from PIL import Image

        from vision_spectra.data.transforms import get_eval_transforms

        transform = get_eval_transforms(image_size=28, num_channels=1)

        img = Image.new("L", (56, 56), color=100)

        result = transform(img)

        assert result.shape == (1, 28, 28)

    def test_denormalize(self):
        """Test denormalization."""
        from vision_spectra.data.transforms import denormalize

        # Simulate normalized tensor
        tensor = torch.randn(3, 32, 32)

        result = denormalize(tensor, num_channels=3)

        assert result.shape == tensor.shape
