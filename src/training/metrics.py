
"""Classification metrics used during validation and later evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Positive class for binary pneumonia detection.
PNEUMONIA_INDEX = 1


def compute_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_prob: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, per-class PRF, macro/weighted averages, AUCs, CM.

    ``y_prob`` should be P(PNEUMONIA) for each sample when provided.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    # rows = true, cols = pred: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(macro_p),
        "recall_macro": float(macro_r),
        "f1_macro": float(macro_f1),
        "precision_weighted": float(weighted_p),
        "recall_weighted": float(weighted_r),
        "f1_weighted": float(weighted_f1),
        "precision_normal": float(precision[0]),
        "recall_normal": float(recall[0]),
        "f1_normal": float(f1[0]),
        "support_normal": int(support[0]),
        "precision_pneumonia": float(precision[1]),
        "recall_pneumonia": float(recall[1]),
        "f1_pneumonia": float(f1[1]),
        "support_pneumonia": int(support[1]),
        "false_negatives": fn,  # PNEUMONIA → NORMAL
        "false_positives": fp,  # NORMAL → PNEUMONIA
        "true_negatives": tn,
        "true_positives": tp,
        "confusion_matrix": cm.tolist(),
        "roc_auc": None,
        "pr_auc": None,
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        # ROC/PR need both classes present in y_true.
        if len(np.unique(y_true_arr)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
            metrics["pr_auc"] = float(
                average_precision_score(y_true_arr, y_prob_arr)
            )

    return metrics
