"""Smoke-test evaluation using the Phase 5 smoke CNN checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.preprocessing import build_eval_transforms
from src.data.split import load_split
from src.evaluation.evaluate import (
    evaluate_model,
    format_evaluation_summary,
    load_checkpoint,
    save_evaluation_report,
)
from src.models.cnn import CustomCNN
from src.training.metrics import compute_classification_metrics
from src.training.trainer import get_device


def _balanced_subset(paths: list[str], per_class: int) -> list[str]:
    normal = [p for p in paths if "/NORMAL/" in p][:per_class]
    pneumonia = [p for p in paths if "/PNEUMONIA/" in p][:per_class]
    assert len(normal) == per_class and len(pneumonia) == per_class
    return normal + pneumonia


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "chest_xray"
    split_path = PROJECT_ROOT / "experiments" / "data_split.json"
    checkpoint = PROJECT_ROOT / "checkpoints" / "smoke_train" / "cnn_smoke_best.pt"
    out_dir = PROJECT_ROOT / "experiments" / "smoke_eval"
    image_size = 224

    assert checkpoint.is_file(), f"Missing smoke checkpoint: {checkpoint}"

    # Unit: metrics helper is the single source of truth.
    unit = compute_classification_metrics([0, 1, 1], [0, 0, 1], [0.2, 0.4, 0.8])
    assert unit["false_negatives"] == 1
    assert "confusion_matrix" in unit
    print("metrics reuse OK")

    split = load_split(split_path)
    val_paths = _balanced_subset(split["val"], per_class=8)
    test_paths = _balanced_subset(split["test"], per_class=8)

    device = get_device()
    model = CustomCNN(num_classes=2)
    model, ckpt = load_checkpoint(model, checkpoint, device=device)
    assert "model_state_dict" in ckpt
    print(f"loaded checkpoint epoch={ckpt.get('epoch')} device={device}")

    eval_tf = build_eval_transforms(image_size)
    val_loader = DataLoader(
        ChestXrayDataset(data_dir, val_paths, transform=eval_tf),
        batch_size=4,
        shuffle=False,
    )
    test_loader = DataLoader(
        ChestXrayDataset(data_dir, test_paths, transform=eval_tf),
        batch_size=4,
        shuffle=False,
    )

    val_report = evaluate_model(
        model,
        val_loader,
        split="val",
        device=device,
        checkpoint_path=str(checkpoint),
    )
    test_report = evaluate_model(
        model,
        test_loader,
        split="test",
        device=device,
        checkpoint_path=str(checkpoint),
    )

    assert val_report["split"] == "val"
    assert test_report["split"] == "test"
    assert val_report["n_samples"] == 16
    assert test_report["n_samples"] == 16
    assert val_report["metrics"]["confusion_matrix"]
    assert "pneumonia_recall" in val_report
    assert "false_negatives" in val_report

    # Keep val/test reports as separate artifacts.
    val_out = save_evaluation_report(val_report, out_dir / "val_report.json")
    test_out = save_evaluation_report(test_report, out_dir / "test_report.json")

    print("--- VAL ---")
    print(format_evaluation_summary(val_report))
    print("--- TEST ---")
    print(format_evaluation_summary(test_report))
    print(f"Wrote {val_out}")
    print(f"Wrote {test_out}")
    print("Phase 6 smoke evaluation passed.")


if __name__ == "__main__":
    main()
