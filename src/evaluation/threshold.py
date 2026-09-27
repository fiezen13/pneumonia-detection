"""Validation threshold analysis for binary pneumonia classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.training.metrics import compute_classification_metrics

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.10, 0.91, 0.10)]


def predict_at_threshold(
    y_prob: list[float] | np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Predict PNEUMONIA (1) when P(PNEUMONIA) >= threshold."""
    probs = np.asarray(y_prob, dtype=np.float64)
    return (probs >= threshold).astype(np.int64)


def evaluate_threshold(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute accuracy / PRF / CM / FN / FP at one probability threshold."""
    y_pred = predict_at_threshold(y_prob, threshold)
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    return {
        "threshold": float(threshold),
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "precision_pneumonia": metrics["precision_pneumonia"],
        "recall_pneumonia": metrics["recall_pneumonia"],
        "f1_pneumonia": metrics["f1_pneumonia"],
        "precision_normal": metrics["precision_normal"],
        "recall_normal": metrics["recall_normal"],
        "f1_normal": metrics["f1_normal"],
        "false_negatives": metrics["false_negatives"],
        "false_positives": metrics["false_positives"],
        "true_negatives": metrics["true_negatives"],
        "true_positives": metrics["true_positives"],
        "confusion_matrix": metrics["confusion_matrix"],
    }


def run_threshold_sweep(
    y_true: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a list of thresholds on validation predictions."""
    thr_list = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    return [evaluate_threshold(y_true, y_prob, t) for t in thr_list]


def select_threshold_max_pneumonia_f1(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Transparent validation selection rule.

    PROJECT_SPEC does not define a single threshold metric. This project
    prioritizes pneumonia detection quality, so we select the validation
    threshold that maximizes ``f1_pneumonia``. Ties break by:
      1) higher ``recall_pneumonia`` (fewer FN)
      2) lower threshold (more sensitive)

    This is a validation operating-point choice, not a clinical optimum.
    """
    if not rows:
        raise ValueError("No threshold rows to select from")

    best = sorted(
        rows,
        key=lambda r: (
            r["f1_pneumonia"],
            r["recall_pneumonia"],
            -r["threshold"],
        ),
        reverse=True,
    )[0]
    return {
        "rule": (
            "Maximize validation f1_pneumonia; "
            "tie-break: higher recall_pneumonia, then lower threshold"
        ),
        "selected": best,
        "note": (
            "Not a clinically optimal threshold; selected on validation only."
        ),
    }


def load_prediction_arrays(predictions_json: Path | str) -> dict[str, list]:
    """Load y_true / y_prob from an error-analysis predictions.json file."""
    records = json.loads(Path(predictions_json).read_text())
    return {
        "y_true": [int(r["true_index"]) for r in records],
        "y_prob": [float(r["prob_pneumonia"]) for r in records],
        "rel_paths": [r["rel_path"] for r in records],
    }


def plot_threshold_curves(
    rows: list[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str,
    selected_threshold: float | None = None,
) -> Path:
    """Plot pneumonia precision/recall/F1 and FN/FP vs threshold."""
    thresholds = [r["threshold"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(thresholds, [r["precision_pneumonia"] for r in rows], marker="o", label="precision_pneu")
    ax.plot(thresholds, [r["recall_pneumonia"] for r in rows], marker="o", label="recall_pneu")
    ax.plot(thresholds, [r["f1_pneumonia"] for r in rows], marker="o", label="f1_pneu")
    ax.plot(thresholds, [r["accuracy"] for r in rows], marker="o", label="accuracy", linestyle="--")
    if selected_threshold is not None:
        ax.axvline(selected_threshold, color="gray", linestyle=":", label=f"selected={selected_threshold}")
    ax.set_xlabel("P(PNEUMONIA) threshold")
    ax.set_ylabel("score")
    ax.set_title("Pneumonia metrics vs threshold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(thresholds, [r["false_negatives"] for r in rows], marker="o", label="FN (PNEU→NORMAL)")
    ax.plot(thresholds, [r["false_positives"] for r in rows], marker="o", label="FP (NORMAL→PNEU)")
    if selected_threshold is not None:
        ax.axvline(selected_threshold, color="gray", linestyle=":", label=f"selected={selected_threshold}")
    ax.set_xlabel("P(PNEUMONIA) threshold")
    ax.set_ylabel("count")
    ax.set_title("FN / FP vs threshold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def rows_to_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact table rows for JSON / printing."""
    keys = [
        "threshold",
        "accuracy",
        "precision_pneumonia",
        "recall_pneumonia",
        "f1_pneumonia",
        "false_negatives",
        "false_positives",
        "confusion_matrix",
    ]
    return [{k: r[k] for k in keys} for r in rows]
