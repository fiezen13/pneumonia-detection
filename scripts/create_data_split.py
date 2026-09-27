"""Create a reproducible train/validation split from the original train set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import (
    DEFAULT_SEED,
    DEFAULT_VAL_RATIO,
    create_group_aware_split,
    save_split,
    validate_split_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stratified train/val split from original train images, "
            "keeping exact-duplicate SHA-256 groups intact."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/chest_xray"),
        help="Path to chest_xray dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/data_split.json"),
        help="Where to write split metadata",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    args = parser.parse_args()

    payload = create_group_aware_split(
        args.data_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    errors = validate_split_payload(payload)
    if errors:
        raise SystemExit("Split validation failed:\n- " + "\n- ".join(errors))

    out = save_split(payload, args.output)

    counts = payload["counts"]
    groups = payload["content_groups"]
    checks = payload["leakage_checks"]

    print(f"Wrote {out}")
    print(f"seed={payload['seed']} val_ratio={payload['val_ratio']}")
    print(
        "train:",
        counts["train"],
        "| content groups:",
        groups["train"],
    )
    print(
        "val:",
        counts["val"],
        "| content groups:",
        groups["val"],
    )
    print("test (untouched):", counts["test"])
    print(
        "original val ignored:",
        counts["original_val_ignored"]["total"],
        "images",
    )
    print(
        "multi-file duplicate groups in source train:",
        groups["n_multi_file_groups"],
    )
    print("leakage_checks.passed:", checks["passed"])
    print(
        "coverage all_original_train_assigned:",
        payload["coverage_checks"]["all_original_train_assigned"],
    )


if __name__ == "__main__":
    main()
