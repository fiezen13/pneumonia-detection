"""Small smoke training run to verify the Phase 5 training engine."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.preprocessing import build_eval_transforms, build_train_transforms
from src.data.split import load_split
from src.models.cnn import CustomCNN
from src.training.metrics import compute_classification_metrics
from src.training.trainer import TrainConfig, Trainer, get_device, set_seed


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "chest_xray"
    split_path = PROJECT_ROOT / "experiments" / "data_split.json"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "smoke_train"
    image_size = 224
    batch_size = 4

    split = load_split(split_path)
    # Tiny balanced subsets — enough to exercise the loop on CPU quickly.
    train_normal = [p for p in split["train"] if "/NORMAL/" in p][:16]
    train_pneumonia = [p for p in split["train"] if "/PNEUMONIA/" in p][:16]
    val_normal = [p for p in split["val"] if "/NORMAL/" in p][:8]
    val_pneumonia = [p for p in split["val"] if "/PNEUMONIA/" in p][:8]
    train_paths = train_normal + train_pneumonia
    val_paths = val_normal + val_pneumonia
    assert len(train_paths) == 32 and len(val_paths) == 16

    train_ds = ChestXrayDataset(
        data_dir, train_paths, transform=build_train_transforms(image_size)
    )
    val_ds = ChestXrayDataset(
        data_dir, val_paths, transform=build_eval_transforms(image_size)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Metrics unit check
    m = compute_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0.1, 0.6, 0.7, 0.9],
    )
    assert m["false_negatives"] == 0
    assert m["false_positives"] == 1
    assert m["roc_auc"] is not None
    print("metrics helper OK")

    config = TrainConfig(
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=5,
        scheduler="cosine",
        seed=42,
        checkpoint_dir=str(ckpt_dir),
        run_name="cnn_smoke",
        monitor_metric="val_loss",
        maximize_monitor=False,
    )

    set_seed(config.seed)
    model = CustomCNN(num_classes=2)
    trainer = Trainer(model, train_loader, val_loader, config)
    assert trainer.device == get_device()
    print(f"device={trainer.device}")

    result = trainer.fit()
    assert result["best_epoch"] >= 1
    assert Path(result["best_checkpoint"]).is_file()
    assert Path(result["last_checkpoint"]).is_file()
    assert Path(result["history_path"]).is_file()
    assert len(result["history"]) == 2

    # Checkpoint reload
    fresh = CustomCNN(num_classes=2)
    reloaded = Trainer(fresh, train_loader, val_loader, config)
    ckpt = reloaded.load_checkpoint(result["best_checkpoint"])
    assert "model_state_dict" in ckpt
    reloaded.model.eval()
    images, labels = next(iter(val_loader))
    with torch.no_grad():
        logits = reloaded.model(images.to(reloaded.device))
    assert logits.shape == (images.size(0), 2)

    losses = [row["train_loss"] for row in result["history"]]
    print(f"train_loss history={losses}")
    print(f"best_epoch={result['best_epoch']} best_metric={result['best_metric']:.4f}")
    print(f"best_checkpoint={result['best_checkpoint']}")
    print("Phase 5 smoke training passed.")


if __name__ == "__main__":
    main()
