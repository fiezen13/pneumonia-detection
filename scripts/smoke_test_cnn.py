"""Smoke-test CustomCNN with a real DataLoader batch."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.dataset import create_dataloaders
from src.models.cnn import CustomCNN, count_parameters


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "chest_xray"
    split_path = PROJECT_ROOT / "experiments" / "data_split.json"
    batch_size = 8
    image_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_classes=2).to(device)
    model.eval()

    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    print(model)
    print(f"device={device}")
    print(f"parameters total={total_params:,} trainable={trainable_params:,}")

    loaders = create_dataloaders(
        data_dir,
        split_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )
    images, labels = next(iter(loaders["train"]))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(images)

    assert images.shape == (batch_size, 3, image_size, image_size), images.shape
    assert logits.shape == (batch_size, 2), logits.shape
    assert labels.shape == (batch_size,), labels.shape
    assert set(labels.tolist()) <= {0, 1}

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    assert probs.shape == (batch_size, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size, device=device), atol=1e-5)

    print(f"input shape={tuple(images.shape)}")
    print(f"label shape={tuple(labels.shape)} values={labels.tolist()}")
    print(f"logits shape={tuple(logits.shape)}")
    print(f"probs shape={tuple(probs.shape)}")
    print(f"preds={preds.tolist()}")
    print("Phase 4 smoke test passed.")


if __name__ == "__main__":
    main()
