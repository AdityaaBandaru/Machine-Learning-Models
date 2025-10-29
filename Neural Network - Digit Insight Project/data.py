"""Dataset utilities for Digit Insight Studio."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


@dataclass
class DataConfig:
    batch_size: int = 128
    num_workers: int = 2
    augment: bool = True
    validation_split: float = 0.1
    root: str = "data"


def _build_transforms(train: bool, augment: bool) -> transforms.Compose:
    ops = []
    if train and augment:
        ops.extend(
            [
                transforms.RandomRotation(12, fill=(0,)),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
            ]
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    return transforms.Compose(ops)


def load_mnist(config: DataConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test dataloaders for MNIST."""
    root = Path(config.root)
    root.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=_build_transforms(True, config.augment),
    )
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=_build_transforms(False, False),
    )

    val_size = int(len(train_dataset) * config.validation_split)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(1337)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    return train_loader, val_loader, test_loader


def dataconfig_from_dict(data: Dict) -> DataConfig:
    return DataConfig(
        batch_size=data.get("batch_size", 128),
        num_workers=data.get("num_workers", 2),
        augment=data.get("augment", True),
        validation_split=data.get("validation_split", 0.1),
        root=data.get("root", "data"),
    )


__all__ = ["load_mnist", "DataConfig", "dataconfig_from_dict", "MNIST_MEAN", "MNIST_STD"]
