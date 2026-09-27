"""Reusable model evaluation on a chosen dataset split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.metrics import PNEUMONIA_INDEX, compute_classification_metrics
from src.training.trainer import get_device


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load ``model_state_dict`` from a Trainer checkpoint into ``model``."""
    resolved = get_device(str(device) if device is not None else None)
    checkpoint = torch.load(
        checkpoint_path, map_location=resolved, weights_only=False
    )
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"No model_state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, list]:
    """Run inference and collect labels, hard predictions, and P(PNEUMONIA)."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs[:, PNEUMONIA_INDEX].cpu().tolist())

    return {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    split: str,
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``model`` on one split and return a structured report.

    ``split`` should be ``\"val\"`` or ``\"test\"`` (or another explicit name).
    Metrics come from ``src.training.metrics.compute_classification_metrics``.
    """
    if not split:
        raise ValueError("split name must be a non-empty string (e.g. 'val' or 'test')")

    resolved = device or get_device()
    model = model.to(resolved)
    preds = collect_predictions(model, loader, resolved)
    metrics = compute_classification_metrics(
        preds["y_true"], preds["y_pred"], preds["y_prob"]
    )

    return {
        "split": split,
        "n_samples": len(preds["y_true"]),
        "checkpoint": checkpoint_path,
        "device": str(resolved),
        "metrics": metrics,
        # Spec priorities — surfaced explicitly for reports / interviews.
        "pneumonia_recall": metrics["recall_pneumonia"],
        "false_negatives": metrics["false_negatives"],
        "false_positives": metrics["false_positives"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def save_evaluation_report(report: dict[str, Any], output_path: Path | str) -> Path:
    """Write an evaluation report JSON (does not overwrite silently without path)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def format_evaluation_summary(report: dict[str, Any]) -> str:
    """Human-readable summary emphasizing pneumonia recall and false negatives."""
    m = report["metrics"]
    cm = report["confusion_matrix"]
    lines = [
        f"split={report['split']} n={report['n_samples']} device={report['device']}",
        f"accuracy={m['accuracy']:.4f}",
        f"f1_macro={m['f1_macro']:.4f}  f1_weighted={m['f1_weighted']:.4f}",
        (
            f"NORMAL  P={m['precision_normal']:.4f} "
            f"R={m['recall_normal']:.4f} F1={m['f1_normal']:.4f}"
        ),
        (
            f"PNEUMONIA  P={m['precision_pneumonia']:.4f} "
            f"R={m['recall_pneumonia']:.4f} F1={m['f1_pneumonia']:.4f}"
        ),
        f"pneumonia_recall={report['pneumonia_recall']:.4f}",
        f"false_negatives (PNEUMONIA→NORMAL)={report['false_negatives']}",
        f"false_positives (NORMAL→PNEUMONIA)={report['false_positives']}",
        f"roc_auc={m['roc_auc']}  pr_auc={m['pr_auc']}",
        f"confusion_matrix (rows=true NORMAL/PNEUMONIA, cols=pred)={cm}",
    ]
    return "\n".join(lines)
