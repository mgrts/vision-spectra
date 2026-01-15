"""Training modules for vision spectra experiments."""

from vision_spectra.training.base import BaseTrainer
from vision_spectra.training.classification import ClassificationTrainer
from vision_spectra.training.finetune import FinetuneTrainer
from vision_spectra.training.mim import MIMTrainer
from vision_spectra.training.multitask import MultitaskTrainer

__all__ = [
    "BaseTrainer",
    "ClassificationTrainer",
    "MIMTrainer",
    "FinetuneTrainer",
    "MultitaskTrainer",
]
