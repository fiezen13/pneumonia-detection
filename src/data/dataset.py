"""PyTorch Dataset and DataLoader helpers for the pneumonia project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    build_eval_transforms,
    build_train_transforms,
)
from src.data.split import load_split

LABEL_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}
INDEX_TO_LABEL = {0: "NORMAL", 1: "PNEUMONIA"}


class ChestXrayDataset(Dataset):
    """Chest X-ray dataset backed by relative paths from data_split.json."""

    def __init__(
        self,
        data_dir: Path | str,
        rel_paths: list[str],
        transform: Callable | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        for rel_path in rel_paths:
            path = self.data_dir / rel_path
            label_name = Path(rel_path).parts[1]
            if label_name not in LABEL_TO_INDEX:
                raise ValueError(f"Unknown label in path: {rel_path}")
            self.samples.append((path, LABEL_TO_INDEX[label_name], label_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label, _ = self.samples[index]
        # Convert grayscale X-rays to RGB for ImageNet-pretrained models.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def label_counts(self) -> dict[str, int]:
        counts = {"NORMAL": 0, "PNEUMONIA": 0}
        for _, _, name in self.samples:
            counts[name] += 1
        return counts


def create_datasets(
    data_dir: Path | str,
    split_path: Path | str,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, ChestXrayDataset]:
    """Build train/val/test datasets from the saved split metadata."""
    split = load_split(split_path)
    root = Path(data_dir)

    train_tf = build_train_transforms(image_size=image_size)
    eval_tf = build_eval_transforms(image_size=image_size)

    return {
        "train": ChestXrayDataset(root, split["train"], transform=train_tf),
        "val": ChestXrayDataset(root, split["val"], transform=eval_tf),
        "test": ChestXrayDataset(root, split["test"], transform=eval_tf),
    }


def create_dataloaders(
    data_dir: Path | str,
    split_path: Path | str,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """Create train/val/test dataloaders. Shuffle only for train."""
    datasets = create_datasets(
        data_dir, split_path, image_size=image_size
    )

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
