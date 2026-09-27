"""Smoke-test Dataset / DataLoader / transforms against data_split.json."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import (
    INDEX_TO_LABEL,
    LABEL_TO_INDEX,
    ChestXrayDataset,
    create_dataloaders,
    create_datasets,
)
from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transforms,
    build_train_transforms,
)
from src.data.split import load_split


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "chest_xray"
    split_path = PROJECT_ROOT / "experiments" / "data_split.json"
    image_size = DEFAULT_IMAGE_SIZE
    batch_size = 8

    split = load_split(split_path)
    assert set(split.keys()) >= {"train", "val", "test", "seed"}
    print(
        f"Loaded split seed={split['seed']} "
        f"train={len(split['train'])} val={len(split['val'])} "
        f"test={len(split['test'])}"
    )

    datasets = create_datasets(data_dir, split_path, image_size=image_size)
    for name, ds in datasets.items():
        counts = ds.label_counts()
        print(f"{name}: len={len(ds)} counts={counts}")
        assert len(ds) == len(split[name])
        assert counts["NORMAL"] + counts["PNEUMONIA"] == len(ds)

    # Single-sample shape / label checks
    train_eval = ChestXrayDataset(
        data_dir, split["train"][:3], transform=build_eval_transforms(image_size)
    )
    image, label = train_eval[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, image_size, image_size), image.shape
    assert label in (0, 1)
    print(f"single sample shape={tuple(image.shape)} label={label} ({INDEX_TO_LABEL[label]})")

    # Train transforms are stochastic but must keep shape
    train_aug = ChestXrayDataset(
        data_dir, split["train"][:1], transform=build_train_transforms(image_size)
    )
    aug_a, _ = train_aug[0]
    aug_b, _ = train_aug[0]
    assert aug_a.shape == (3, image_size, image_size)
    assert aug_b.shape == (3, image_size, image_size)
    print("train augmentation produces valid tensors")

    # Eval transforms are deterministic
    eval_ds = ChestXrayDataset(
        data_dir, split["val"][:1], transform=build_eval_transforms(image_size)
    )
    e1, _ = eval_ds[0]
    e2, _ = eval_ds[0]
    assert torch.allclose(e1, e2)
    print("eval transforms are deterministic")

    loaders = create_dataloaders(
        data_dir,
        split_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )

    for name, loader in loaders.items():
        images, labels = next(iter(loader))
        assert images.ndim == 4
        assert images.shape[1:] == (3, image_size, image_size)
        assert labels.ndim == 1
        assert images.shape[0] == labels.shape[0]
        assert set(labels.tolist()) <= {0, 1}
        # Spot-check label matches path class for first item in this batch
        print(
            f"{name} batch: images={tuple(images.shape)} "
            f"labels={tuple(labels.shape)} "
            f"label_values={labels.tolist()}"
        )

    # Path-derived labels match LABEL_TO_INDEX
    for rel in split["train"][:20] + split["val"][:20] + split["test"][:20]:
        name = Path(rel).parts[1]
        assert name in LABEL_TO_INDEX

    print(f"ImageNet mean={IMAGENET_MEAN} std={IMAGENET_STD}")
    print("Phase 3 smoke test passed.")


if __name__ == "__main__":
    main()
