"""
Synthetic image classification dataset generator.

Generates simple geometric shapes for classification experiments.
Useful for quick testing and debugging without downloading real data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from vision_spectra.data.base import BaseDataset, DatasetInfo
from vision_spectra.data.transforms import get_eval_transforms, get_train_transforms

if TYPE_CHECKING:
    from vision_spectra.settings import DatasetConfig


# Shape types for synthetic dataset
SHAPES = ["circle", "square", "triangle", "star", "cross"]


class SyntheticImageDataset(Dataset):
    """
    Synthetic dataset with geometric shapes.

    Generates images on-the-fly for memory efficiency.
    Uses numpy random state for reproducibility.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        image_size: int = 28,
        num_channels: int = 3,
        seed: int = 42,
        transform=None,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = min(num_classes, len(SHAPES))
        self.image_size = image_size
        self.num_channels = num_channels
        self.seed = seed
        self.transform = transform

        # Pre-generate labels and seeds for each sample (for reproducibility)
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, self.num_classes, size=num_samples)
        self.sample_seeds = rng.randint(0, 2**31, size=num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        label = int(self.labels[idx])
        sample_seed = int(self.sample_seeds[idx])

        # Generate image deterministically based on sample seed
        image = self._generate_image(label, sample_seed)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _generate_image(self, label: int, seed: int) -> Image.Image:
        """Generate a single image with the specified shape."""
        rng = np.random.RandomState(seed)

        # Create background with slight noise
        if self.num_channels == 1:
            bg_color = rng.randint(20, 60)
            mode = "L"
        else:
            bg_color = tuple(rng.randint(20, 60, size=3))
            mode = "RGB"

        image = Image.new(mode, (self.image_size, self.image_size), bg_color)
        draw = ImageDraw.Draw(image)

        # Random shape color
        if self.num_channels == 1:
            shape_color = rng.randint(180, 255)
        else:
            shape_color = tuple(rng.randint(150, 255, size=3))

        # Random position and size
        margin = self.image_size // 6
        size = rng.randint(self.image_size // 3, self.image_size - 2 * margin)
        x = rng.randint(margin, self.image_size - size - margin)
        y = rng.randint(margin, self.image_size - size - margin)

        shape_name = SHAPES[label]

        if shape_name == "circle":
            draw.ellipse([x, y, x + size, y + size], fill=shape_color)

        elif shape_name == "square":
            draw.rectangle([x, y, x + size, y + size], fill=shape_color)

        elif shape_name == "triangle":
            cx, cy = x + size // 2, y + size // 2
            r = size // 2
            points = [
                (cx, cy - r),  # top
                (cx - r, cy + r),  # bottom left
                (cx + r, cy + r),  # bottom right
            ]
            draw.polygon(points, fill=shape_color)

        elif shape_name == "star":
            cx, cy = x + size // 2, y + size // 2
            r_outer = size // 2
            r_inner = size // 4
            points = []
            for i in range(5):
                angle_outer = np.pi / 2 + i * 2 * np.pi / 5
                angle_inner = np.pi / 2 + (i + 0.5) * 2 * np.pi / 5
                points.append(
                    (cx + r_outer * np.cos(angle_outer), cy - r_outer * np.sin(angle_outer))
                )
                points.append(
                    (cx + r_inner * np.cos(angle_inner), cy - r_inner * np.sin(angle_inner))
                )
            draw.polygon(points, fill=shape_color)

        elif shape_name == "cross":
            w = size // 3
            # Horizontal bar
            draw.rectangle(
                [x, y + size // 2 - w // 2, x + size, y + size // 2 + w // 2], fill=shape_color
            )
            # Vertical bar
            draw.rectangle(
                [x + size // 2 - w // 2, y, x + size // 2 + w // 2, y + size], fill=shape_color
            )

        return image


class SyntheticDataset(BaseDataset):
    """Synthetic geometric shapes dataset."""

    def __init__(self, config: DatasetConfig, data_dir: Path) -> None:
        super().__init__(config, data_dir)

        self.num_classes = min(config.num_classes, len(SHAPES))
        self.image_size = config.image_size
        self.num_channels = 3  # RGB for synthetic

        # Create datasets with different seeds for reproducibility
        train_transform = get_train_transforms(self.image_size, self.num_channels)
        eval_transform = get_eval_transforms(self.image_size, self.num_channels)

        self._train_dataset = SyntheticImageDataset(
            num_samples=config.num_samples_train,
            num_classes=self.num_classes,
            image_size=self.image_size,
            num_channels=self.num_channels,
            seed=42,  # Fixed seed for train
            transform=train_transform,
        )

        self._val_dataset = SyntheticImageDataset(
            num_samples=config.num_samples_val,
            num_classes=self.num_classes,
            image_size=self.image_size,
            num_channels=self.num_channels,
            seed=43,  # Different seed for val
            transform=eval_transform,
        )

        self._test_dataset = SyntheticImageDataset(
            num_samples=config.num_samples_test,
            num_classes=self.num_classes,
            image_size=self.image_size,
            num_channels=self.num_channels,
            seed=44,  # Different seed for test
            transform=eval_transform,
        )

        # Compute class counts
        train_counts = np.bincount(self._train_dataset.labels, minlength=self.num_classes)
        val_counts = np.bincount(self._val_dataset.labels, minlength=self.num_classes)
        test_counts = np.bincount(self._test_dataset.labels, minlength=self.num_classes)

        self._info = DatasetInfo(
            name="synthetic",
            num_classes=self.num_classes,
            num_channels=self.num_channels,
            image_size=(self.image_size, self.image_size),
            train_size=config.num_samples_train,
            val_size=config.num_samples_val,
            test_size=config.num_samples_test,
            class_names=SHAPES[: self.num_classes],
            class_counts={
                "train": train_counts,
                "val": val_counts,
                "test": test_counts,
            },
        )

    def get_train_dataset(self) -> Dataset:
        return self._train_dataset

    def get_val_dataset(self) -> Dataset:
        return self._val_dataset

    def get_test_dataset(self) -> Dataset:
        return self._test_dataset

    def get_info(self) -> DatasetInfo:
        return self._info
