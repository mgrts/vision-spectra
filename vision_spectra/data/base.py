"""
Base dataset abstractions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from vision_spectra.settings import DatasetConfig


@dataclass
class DatasetInfo:
    """Dataset metadata."""

    name: str
    num_classes: int
    num_channels: int
    image_size: tuple[int, int]
    train_size: int
    val_size: int
    test_size: int
    class_names: list[str] | None = None
    class_counts: dict[str, np.ndarray] | None = None  # {split: counts}


class BaseDataset(ABC):
    """
    Abstract base class for datasets.

    Subclasses must implement:
    - get_train_dataset()
    - get_val_dataset()
    - get_test_dataset()
    - get_info()
    """

    def __init__(self, config: DatasetConfig, data_dir: Path) -> None:
        self.config = config
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _pin_memory(self) -> bool:
        """Get pin_memory setting, disabled for MPS devices."""
        if not self.config.pin_memory:
            return False
        # MPS doesn't support pin_memory
        return not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        """Get training dataset."""
        pass

    @abstractmethod
    def get_val_dataset(self) -> Dataset:
        """Get validation dataset."""
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        """Get test dataset."""
        pass

    @abstractmethod
    def get_info(self) -> DatasetInfo:
        """Get dataset metadata."""
        pass

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.get_train_dataset(),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self._pin_memory,
            drop_last=True,
        )

    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.get_val_dataset(),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )

    def get_test_loader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.get_test_dataset(),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self._pin_memory,
            drop_last=False,
        )


def get_dataset(config: DatasetConfig, data_dir: Path) -> BaseDataset:
    """
    Factory function to get dataset by name.

    Args:
        config: Dataset configuration
        data_dir: Data directory path

    Returns:
        Dataset instance
    """
    from vision_spectra.data.medmnist import MedMNISTDataset
    from vision_spectra.data.synthetic import SyntheticDataset
    from vision_spectra.settings import DatasetName

    if config.name == DatasetName.SYNTHETIC:
        return SyntheticDataset(config, data_dir)
    else:
        # All other names are MedMNIST variants
        return MedMNISTDataset(config, data_dir)
