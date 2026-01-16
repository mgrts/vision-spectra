"""
MedMNIST dataset wrapper.

MedMNIST is a collection of medical image datasets for benchmarking.
All images are 28x28 grayscale or RGB, with various classification tasks.

Reference:
    Yang, J., et al. (2023). MedMNIST v2-A large-scale lightweight benchmark
    for 2D and 3D biomedical image classification. Scientific Data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vision_spectra.data.base import BaseDataset, DatasetInfo
from vision_spectra.data.transforms import get_eval_transforms, get_train_transforms

if TYPE_CHECKING:
    from vision_spectra.settings import DatasetConfig


# MedMNIST dataset configurations
MEDMNIST_INFO = {
    "pathmnist": {
        "module": "PathMNIST",
        "num_classes": 9,
        "num_channels": 3,
        "class_names": [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth muscle",
            "normal colon mucosa",
            "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium",
        ],
    },
    "pneumoniamnist": {
        "module": "PneumoniaMNIST",
        "num_classes": 2,
        "num_channels": 1,
        "class_names": ["normal", "pneumonia"],
    },
    "bloodmnist": {
        "module": "BloodMNIST",
        "num_classes": 8,
        "num_channels": 3,
        "class_names": [
            "basophil",
            "eosinophil",
            "erythroblast",
            "ig",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ],
    },
    "dermamnist": {
        "module": "DermaMNIST",
        "num_classes": 7,
        "num_channels": 3,
        "class_names": [
            "actinic keratoses",
            "basal cell carcinoma",
            "benign keratosis",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ],
    },
    "octmnist": {
        "module": "OCTMNIST",
        "num_classes": 4,
        "num_channels": 1,
        "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"],
    },
    "organamnist": {
        "module": "OrganAMNIST",
        "num_classes": 11,
        "num_channels": 1,
        "class_names": [
            "bladder",
            "femur-left",
            "femur-right",
            "heart",
            "kidney-left",
            "kidney-right",
            "liver",
            "lung-left",
            "lung-right",
            "spleen",
            "pancreas",
        ],
    },
}


class MedMNISTWrapper(Dataset):
    """Wrapper for MedMNIST datasets to apply transforms."""

    def __init__(
        self,
        base_dataset,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[idx]

        # MedMNIST returns numpy arrays
        if isinstance(image, np.ndarray):
            # Convert to PIL for transforms
            from PIL import Image

            image = Image.fromarray(image, mode="L") if image.ndim == 2 else Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Label is array of shape (1,) -> scalar
        if isinstance(label, np.ndarray):
            label = int(label.squeeze())

        return image, label


class MedMNISTDataset(BaseDataset):
    """MedMNIST dataset implementation."""

    def __init__(self, config: DatasetConfig, data_dir: Path) -> None:
        super().__init__(config, data_dir)

        dataset_name = config.name.value.lower()
        if dataset_name not in MEDMNIST_INFO:
            raise ValueError(
                f"Unknown MedMNIST dataset: {dataset_name}. "
                f"Available: {list(MEDMNIST_INFO.keys())}"
            )

        self.dataset_info = MEDMNIST_INFO[dataset_name]
        self._train_dataset: Dataset | None = None
        self._val_dataset: Dataset | None = None
        self._test_dataset: Dataset | None = None
        self._info: DatasetInfo | None = None

        # Load datasets
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load MedMNIST datasets."""
        import medmnist

        module_name = self.dataset_info["module"]
        dataset_class = getattr(medmnist, module_name)

        # Image size for transforms
        image_size = self.config.image_size
        num_channels = self.dataset_info["num_channels"]

        # Get transforms
        train_transform = get_train_transforms(image_size, num_channels)
        eval_transform = get_eval_transforms(image_size, num_channels)

        # Download and load datasets
        root = str(self.data_dir)

        train_base = dataset_class(split="train", download=True, root=root)
        val_base = dataset_class(split="val", download=True, root=root)
        test_base = dataset_class(split="test", download=True, root=root)

        self._train_dataset = MedMNISTWrapper(train_base, train_transform)
        self._val_dataset = MedMNISTWrapper(val_base, eval_transform)
        self._test_dataset = MedMNISTWrapper(test_base, eval_transform)

        # Compute class counts
        train_labels = train_base.labels.squeeze()
        val_labels = val_base.labels.squeeze()
        test_labels = test_base.labels.squeeze()

        num_classes = self.dataset_info["num_classes"]
        train_counts = np.bincount(train_labels, minlength=num_classes)
        val_counts = np.bincount(val_labels, minlength=num_classes)
        test_counts = np.bincount(test_labels, minlength=num_classes)

        self._info = DatasetInfo(
            name=self.config.name.value,
            num_classes=num_classes,
            num_channels=num_channels,
            image_size=(image_size, image_size),
            train_size=len(train_base),
            val_size=len(val_base),
            test_size=len(test_base),
            class_names=self.dataset_info["class_names"],
            class_counts={
                "train": train_counts,
                "val": val_counts,
                "test": test_counts,
            },
        )

    def get_train_dataset(self) -> Dataset:
        assert self._train_dataset is not None
        return self._train_dataset

    def get_val_dataset(self) -> Dataset:
        assert self._val_dataset is not None
        return self._val_dataset

    def get_test_dataset(self) -> Dataset:
        assert self._test_dataset is not None
        return self._test_dataset

    def get_info(self) -> DatasetInfo:
        assert self._info is not None
        return self._info


def download_medmnist(dataset_name: str, data_dir: Path) -> None:
    """
    Download a MedMNIST dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'pathmnist')
        data_dir: Directory to save data
    """
    import medmnist
    from loguru import logger

    dataset_name = dataset_name.lower()
    if dataset_name not in MEDMNIST_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(MEDMNIST_INFO.keys())}"
        )

    module_name = MEDMNIST_INFO[dataset_name]["module"]
    dataset_class = getattr(medmnist, module_name)

    data_dir.mkdir(parents=True, exist_ok=True)
    root = str(data_dir)

    logger.info(f"Downloading {dataset_name} to {data_dir}...")

    for split in ["train", "val", "test"]:
        logger.info(f"  Downloading {split} split...")
        dataset_class(split=split, download=True, root=root)

    logger.success(f"Successfully downloaded {dataset_name}!")
