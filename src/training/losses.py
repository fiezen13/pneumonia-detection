"""Loss helpers, including train-split class weights for imbalance handling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def compute_balanced_class_weights(
    rel_paths: list[str],
    *,
    label_order: tuple[str, ...] = ("NORMAL", "PNEUMONIA"),
) -> dict[str, Any]:
    """Compute inverse-frequency class weights from a list of relative paths.

    Uses only the provided paths (must be the training split). Formula:

        weight_c = N / (C * n_c)

    where N is the number of training samples, C is the number of classes,
    and n_c is the count of class c. This matches scikit-learn's
    ``compute_class_weight(class_weight='balanced')``.

    Labels are taken from path segment ``parts[1]`` (e.g. ``train/NORMAL/...``).
    """
    if not rel_paths:
        raise ValueError("rel_paths must be non-empty (training split only)")

    counts: Counter[str] = Counter()
    for rel in rel_paths:
        label = Path(rel).parts[1]
        if label not in label_order:
            raise ValueError(f"Unexpected label '{label}' in path {rel}")
        counts[label] += 1

    n_samples = sum(counts[label] for label in label_order)
    n_classes = len(label_order)
    if n_samples != len(rel_paths):
        raise ValueError("Mismatch between path count and class counts")

    weights = []
    per_class = {}
    for label in label_order:
        n_c = counts[label]
        if n_c <= 0:
            raise ValueError(f"Class {label} has zero samples in training split")
        w = n_samples / (n_classes * n_c)
        weights.append(float(w))
        per_class[label] = {
            "count": int(n_c),
            "weight": float(w),
        }

    return {
        "source_split": "train",
        "formula": "N / (C * n_c)",
        "n_samples": int(n_samples),
        "n_classes": int(n_classes),
        "label_order": list(label_order),
        "counts": {label: int(counts[label]) for label in label_order},
        "weights": weights,
        "per_class": per_class,
    }
