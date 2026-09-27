"""Reproducible train/validation split from the original training set.

Exact-duplicate groups (same SHA-256) are treated as atomic units so copies
cannot be separated across the new train and validation splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.audit import CLASSES, discover_images, hash_images

DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.15


def _build_content_groups(train_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique SHA-256 among original-train images."""
    if "sha256" not in train_df.columns:
        raise ValueError("train_df must include sha256")

    rows: list[dict[str, Any]] = []
    for sha, group in train_df.groupby("sha256", sort=True):
        labels = sorted(group["label"].unique())
        if len(labels) != 1:
            raise ValueError(
                f"Conflicting labels in content group {sha}: {labels}"
            )
        rel_paths = sorted(group["rel_path"].tolist())
        rows.append(
            {
                "sha256": sha,
                "label": labels[0],
                "n_files": len(rel_paths),
                "rel_paths": rel_paths,
            }
        )
    return pd.DataFrame(rows)


def create_group_aware_split(
    data_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
) -> dict[str, Any]:
    """Split original train into train/val; leave original test untouched.

    The original ``val/`` folder is intentionally ignored (too small).
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    root = Path(data_dir)
    inventory = discover_images(root)
    hashed = hash_images(inventory)

    original_train = hashed[hashed["split"] == "train"].copy()
    original_val = hashed[hashed["split"] == "val"].copy()
    original_test = hashed[hashed["split"] == "test"].copy()

    if original_train.empty:
        raise ValueError(f"No training images found under {root}")

    groups = _build_content_groups(original_train)
    multi_file = groups[groups["n_files"] > 1]

    group_train, group_val = train_test_split(
        groups,
        test_size=val_ratio,
        random_state=seed,
        stratify=groups["label"],
        shuffle=True,
    )

    train_paths = sorted(
        path for paths in group_train["rel_paths"] for path in paths
    )
    val_paths = sorted(
        path for paths in group_val["rel_paths"] for path in paths
    )
    test_paths = sorted(original_test["rel_path"].tolist())

    train_hashes = set(group_train["sha256"])
    val_hashes = set(group_val["sha256"])
    test_hashes = set(original_test["sha256"])

    def _counts(paths: list[str]) -> dict[str, int]:
        labels = [Path(p).parts[1] for p in paths]
        return {
            "NORMAL": labels.count("NORMAL"),
            "PNEUMONIA": labels.count("PNEUMONIA"),
            "total": len(paths),
        }

    def _group_counts(group_df: pd.DataFrame) -> dict[str, int]:
        return {
            "NORMAL": int((group_df["label"] == "NORMAL").sum()),
            "PNEUMONIA": int((group_df["label"] == "PNEUMONIA").sum()),
            "total_groups": int(len(group_df)),
            "total_files": int(group_df["n_files"].sum()),
        }

    train_val_overlap = sorted(train_hashes & val_hashes)
    train_test_overlap = sorted(train_hashes & test_hashes)
    val_test_overlap = sorted(val_hashes & test_hashes)
    path_overlap_train_val = sorted(set(train_paths) & set(val_paths))

    # Multi-file groups must sit entirely in one assigned split.
    assigned = pd.concat(
        [
            group_train.assign(assigned_split="train"),
            group_val.assign(assigned_split="val"),
        ],
        ignore_index=True,
    )
    multi_assigned = assigned[assigned["n_files"] > 1]
    split_by_hash = {
        sha: split
        for sha, split in zip(
            multi_assigned["sha256"], multi_assigned["assigned_split"]
        )
    }
    broken_groups = []
    for sha, group in original_train.groupby("sha256"):
        if len(group) <= 1:
            continue
        expected = split_by_hash[sha]
        member_splits = {
            "train" if rel in set(train_paths) else "val"
            for rel in group["rel_path"]
        }
        if member_splits != {expected}:
            broken_groups.append(sha)

    payload: dict[str, Any] = {
        "seed": seed,
        "val_ratio": val_ratio,
        "data_dir": str(root),
        "source_split": "train",
        "original_val_policy": "ignored",
        "original_val_count": int(len(original_val)),
        "test_policy": "untouched",
        "duplicate_policy": (
            "Exact SHA-256 duplicate groups from the original train set are "
            "assigned atomically to either train or val."
        ),
        "counts": {
            "train": _counts(train_paths),
            "val": _counts(val_paths),
            "test": _counts(test_paths),
            "original_train": _counts(sorted(original_train["rel_path"].tolist())),
            "original_val_ignored": _counts(
                sorted(original_val["rel_path"].tolist())
            ),
        },
        "content_groups": {
            "n_groups_total": int(len(groups)),
            "n_multi_file_groups": int(len(multi_file)),
            "train": _group_counts(group_train),
            "val": _group_counts(group_val),
        },
        "leakage_checks": {
            "train_val_shared_hashes": train_val_overlap,
            "train_test_shared_hashes": train_test_overlap,
            "val_test_shared_hashes": val_test_overlap,
            "train_val_shared_paths": path_overlap_train_val,
            "duplicate_groups_split_across_train_val": broken_groups,
            "passed": (
                not train_val_overlap
                and not train_test_overlap
                and not val_test_overlap
                and not path_overlap_train_val
                and not broken_groups
            ),
        },
        "coverage_checks": {
            "all_original_train_assigned": sorted(
                set(original_train["rel_path"])
            )
            == sorted(set(train_paths) | set(val_paths)),
            "n_original_train": int(len(original_train)),
            "n_assigned_train_val": int(len(train_paths) + len(val_paths)),
        },
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }
    return payload


def validate_split_payload(payload: dict[str, Any]) -> list[str]:
    """Return a list of validation error messages (empty if OK)."""
    errors: list[str] = []

    train = set(payload["train"])
    val = set(payload["val"])
    test = set(payload["test"])

    if train & val:
        errors.append(f"train/val path overlap: {len(train & val)}")
    if train & test:
        errors.append(f"train/test path overlap: {len(train & test)}")
    if val & test:
        errors.append(f"val/test path overlap: {len(val & test)}")

    checks = payload["leakage_checks"]
    if not checks["passed"]:
        errors.append("leakage_checks.passed is False")
    if not payload["coverage_checks"]["all_original_train_assigned"]:
        errors.append("not all original train files were assigned")

    for split_name in ("train", "val", "test"):
        counts = payload["counts"][split_name]
        expected = counts["NORMAL"] + counts["PNEUMONIA"]
        if counts["total"] != expected:
            errors.append(f"{split_name} counts inconsistent")
        if counts["total"] != len(payload[split_name]):
            errors.append(f"{split_name} list length != counts.total")

    for label in CLASSES:
        if payload["counts"]["train"][label] == 0:
            errors.append(f"train missing class {label}")
        if payload["counts"]["val"][label] == 0:
            errors.append(f"val missing class {label}")

    return errors


def save_split(payload: dict[str, Any], output_path: Path | str) -> Path:
    """Write split metadata JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return path


def load_split(path: Path | str) -> dict[str, Any]:
    """Load a previously saved split metadata file."""
    return json.loads(Path(path).read_text())
