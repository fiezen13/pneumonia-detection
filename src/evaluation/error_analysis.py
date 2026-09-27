"""Validation error analysis: per-sample records and FP/FN visualization grids."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data.dataset import INDEX_TO_LABEL
from src.evaluation.evaluate import collect_predictions
from src.training.metrics import compute_classification_metrics
from src.training.trainer import get_device


def error_type(y_true: int, y_pred: int) -> str:
    """Map a true/pred pair to TP/TN/FP/FN (positive = PNEUMONIA)."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"  # NORMAL → PNEUMONIA
    if y_true == 1 and y_pred == 0:
        return "FN"  # PNEUMONIA → NORMAL
    raise ValueError(f"Unexpected labels y_true={y_true}, y_pred={y_pred}")


def build_prediction_records(
    rel_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
) -> list[dict[str, Any]]:
    """Build one record per sample with path, labels, probability, error type."""
    if not (len(rel_paths) == len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("rel_paths / y_true / y_pred / y_prob length mismatch")

    records = []
    for rel, t, p, prob in zip(rel_paths, y_true, y_pred, y_prob):
        et = error_type(t, p)
        # Confidence in the predicted class.
        pred_confidence = float(prob) if p == 1 else float(1.0 - prob)
        records.append(
            {
                "rel_path": rel,
                "true_label": INDEX_TO_LABEL[t],
                "true_index": int(t),
                "predicted_label": INDEX_TO_LABEL[p],
                "predicted_index": int(p),
                "prob_pneumonia": float(prob),
                "pred_confidence": pred_confidence,
                "error_type": et,
            }
        )
    return records


def filter_and_sort_errors(
    records: list[dict[str, Any]],
    error_type_name: str,
) -> list[dict[str, Any]]:
    """Return FP or FN records sorted by wrong-prediction confidence (desc)."""
    subset = [r for r in records if r["error_type"] == error_type_name]
    return sorted(subset, key=lambda r: r["pred_confidence"], reverse=True)


def summarize_error_confidences(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Simple confidence stats for an FP or FN list."""
    if not errors:
        return {
            "count": 0,
            "prob_pneumonia_mean": None,
            "prob_pneumonia_min": None,
            "prob_pneumonia_max": None,
            "pred_confidence_mean": None,
            "pred_confidence_min": None,
            "pred_confidence_max": None,
        }
    probs = [r["prob_pneumonia"] for r in errors]
    confs = [r["pred_confidence"] for r in errors]
    return {
        "count": len(errors),
        "prob_pneumonia_mean": sum(probs) / len(probs),
        "prob_pneumonia_min": min(probs),
        "prob_pneumonia_max": max(probs),
        "pred_confidence_mean": sum(confs) / len(confs),
        "pred_confidence_min": min(confs),
        "pred_confidence_max": max(confs),
    }


@torch.no_grad()
def run_error_analysis(
    model: nn.Module,
    loader: DataLoader,
    rel_paths: list[str],
    *,
    split: str = "val",
    device: torch.device | None = None,
    checkpoint_path: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Collect val predictions and build FP/FN collections (no test usage)."""
    if split != "val":
        raise ValueError(
            f"Error analysis in this phase is validation-only; got split={split!r}"
        )

    resolved = device or get_device()
    model = model.to(resolved)
    preds = collect_predictions(model, loader, resolved)
    if len(rel_paths) != len(preds["y_true"]):
        raise ValueError("Dataset path list length does not match predictions")

    records = build_prediction_records(
        rel_paths, preds["y_true"], preds["y_pred"], preds["y_prob"]
    )
    metrics = compute_classification_metrics(
        preds["y_true"], preds["y_pred"], preds["y_prob"]
    )
    fps = filter_and_sort_errors(records, "FP")
    fns = filter_and_sort_errors(records, "FN")

    # Verify against confusion matrix [[TN, FP], [FN, TP]]
    cm = metrics["confusion_matrix"]
    tn, fp_cm, fn_cm, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    counts = {
        "TP": sum(1 for r in records if r["error_type"] == "TP"),
        "TN": sum(1 for r in records if r["error_type"] == "TN"),
        "FP": len(fps),
        "FN": len(fns),
    }
    if counts["FP"] != fp_cm or counts["FN"] != fn_cm:
        raise AssertionError(
            f"FP/FN mismatch vs confusion matrix: counts={counts} cm={cm}"
        )
    if counts["TP"] != tp or counts["TN"] != tn:
        raise AssertionError(
            f"TP/TN mismatch vs confusion matrix: counts={counts} cm={cm}"
        )

    return {
        "run_name": run_name,
        "split": split,
        "checkpoint": checkpoint_path,
        "n_samples": len(records),
        "confusion_matrix": cm,
        "counts": counts,
        "metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "recall_pneumonia": metrics["recall_pneumonia"],
            "precision_pneumonia": metrics["precision_pneumonia"],
        },
        "fp_confidence": summarize_error_confidences(fps),
        "fn_confidence": summarize_error_confidences(fns),
        "false_positives": fps,
        "false_negatives": fns,
        "records": records,
    }


def save_error_analysis(
    analysis: dict[str, Any],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Write JSON artifacts for one candidate run."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": out / "summary.json",
        "predictions": out / "predictions.json",
        "false_positives": out / "false_positives.json",
        "false_negatives": out / "false_negatives.json",
    }

    summary = {
        k: analysis[k]
        for k in (
            "run_name",
            "split",
            "checkpoint",
            "n_samples",
            "confusion_matrix",
            "counts",
            "metrics",
            "fp_confidence",
            "fn_confidence",
        )
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    paths["predictions"].write_text(
        json.dumps(analysis["records"], indent=2) + "\n"
    )
    paths["false_positives"].write_text(
        json.dumps(analysis["false_positives"], indent=2) + "\n"
    )
    paths["false_negatives"].write_text(
        json.dumps(analysis["false_negatives"], indent=2) + "\n"
    )
    return paths


def plot_error_grid(
    errors: list[dict[str, Any]],
    data_dir: Path | str,
    output_path: Path | str,
    *,
    title: str,
    max_images: int = 12,
    ncols: int = 4,
) -> Path | None:
    """Save a grid of representative FP/FN images with labels and probabilities.

    Does not add medical interpretations — only path, labels, and confidence.
    """
    if not errors:
        return None

    root = Path(data_dir)
    subset = errors[:max_images]
    n = len(subset)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        ax.axis("off")
        if idx >= n:
            continue
        rec = subset[idx]
        img = Image.open(root / rec["rel_path"]).convert("RGB")
        ax.imshow(img, cmap=None)
        ax.set_title(
            (
                f"{rec['error_type']}  conf={rec['pred_confidence']:.3f}\n"
                f"true={rec['true_label']} pred={rec['predicted_label']}\n"
                f"P(PNEU)={rec['prob_pneumonia']:.3f}\n"
                f"{Path(rec['rel_path']).name}"
            ),
            fontsize=8,
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
