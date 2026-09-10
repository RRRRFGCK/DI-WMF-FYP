from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .models import SPECS


def _targets(dataset) -> Sequence[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [label for _, label in dataset.samples]
    if isinstance(targets, torch.Tensor):
        return targets.cpu().tolist()
    return list(targets)


def _stratified_indices(targets, val_fraction: float, train_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train_indices = []
    val_indices = []
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        remaining = indices[val_count:]
        train_count = max(1, int(round(len(remaining) * train_fraction)))
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(remaining[:train_count].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def _build_datasets(
    dataset_name: str,
    data_root: Path,
    sign_root: Path | None,
    augmentation: str = "none",
):
    if augmentation not in {"none", "cifar_standard"}:
        raise ValueError(f"Unknown augmentation policy: {augmentation}")
    if augmentation != "none" and dataset_name != "cifar10":
        raise ValueError("cifar_standard augmentation is only defined for CIFAR-10")
    spec = SPECS[dataset_name]
    if dataset_name in {"mnist", "fashion"}:
        transform = transforms.Compose(
            [transforms.Resize(spec.image_size), transforms.ToTensor()]
        )
        dataset_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
        train = dataset_cls(data_root, train=True, download=False, transform=transform)
        test = dataset_cls(data_root, train=False, download=False, transform=transform)
        return train, test
    if dataset_name == "cifar10":
        normalise = transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        )
        clean_transform = transforms.Compose(
            [
                transforms.Resize(spec.image_size),
                transforms.ToTensor(),
                normalise,
            ]
        )
        train_transform = clean_transform
        if augmentation == "cifar_standard":
            train_transform = transforms.Compose(
                [
                    transforms.RandomCrop(spec.image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalise,
                ]
            )
        train = datasets.CIFAR10(
            data_root, train=True, download=False, transform=train_transform
        )
        clean_train = datasets.CIFAR10(
            data_root, train=True, download=False, transform=clean_transform
        )
        test = datasets.CIFAR10(
            data_root, train=False, download=False, transform=clean_transform
        )
        return train, test, clean_train
    if dataset_name == "sign":
        if sign_root is None:
            raise ValueError("--sign-root is required for the sign dataset")
        transform = transforms.Compose(
            [transforms.Resize((spec.image_size, spec.image_size)), transforms.ToTensor()]
        )
        train, test = (
            datasets.ImageFolder(sign_root / "train_10", transform=transform),
            datasets.ImageFolder(sign_root / "test_10", transform=transform),
        )
        return train, test
    raise ValueError(dataset_name)


def build_test_dataset(
    dataset_name: str, data_root: Path, sign_root: Path | None
):
    """Build only the clean test dataset for checkpoint evaluation."""
    datasets_built = _build_datasets(dataset_name, data_root, sign_root)
    test_dataset = datasets_built[1]
    return test_dataset


def build_loaders(
    dataset_name: str,
    data_root: Path,
    sign_root: Path | None,
    batch_size: int,
    seed: int,
    train_fraction: float = 1.0,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
    augmentation: str = "none",
):
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    datasets_built = _build_datasets(
        dataset_name, data_root, sign_root, augmentation=augmentation
    )
    train_dataset, test_dataset = datasets_built[:2]
    # Initialisation and validation must remain deterministic and clean even
    # when the optimiser sees stochastic augmented views of the same indices.
    clean_train_dataset = (
        datasets_built[2] if len(datasets_built) == 3 else train_dataset
    )
    train_indices, val_indices = _stratified_indices(
        _targets(train_dataset), val_fraction, train_fraction, seed
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        common["persistent_workers"] = True
    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    fit_loader = DataLoader(
        Subset(clean_train_dataset, train_indices), shuffle=False, **common
    )
    val_loader = DataLoader(
        Subset(clean_train_dataset, val_indices), shuffle=False, **common
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, fit_loader, val_loader, test_loader
