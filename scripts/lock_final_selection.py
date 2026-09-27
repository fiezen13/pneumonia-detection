"""Lock the final model and validation-selected decision threshold.

Does not retrain, does not modify prior artifacts, and does not access the test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Candidates that completed validation threshold analysis (Phase 13).
CANDIDATE_CHECKPOINTS = {
    "resnet18_finetune": PROJECT_ROOT
    / "checkpoints"
    / "resnet18_finetune"
    / "resnet18_finetune_best.pt",
    "resnet18_finetune_weighted": PROJECT_ROOT
    / "checkpoints"
    / "resnet18_finetune_weighted"
    / "resnet18_finetune_weighted_best.pt",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def select_final_candidate(threshold_comparison: dict) -> dict:
    """Select final (model, threshold) from existing validation threshold results.

    Rule (aligned with threshold analysis; PROJECT_SPEC has no single metric):
      1. Restrict to ResNet18 fine-tune candidates with locked validation thresholds.
      2. Maximize validation pneumonia F1 at each candidate's selected threshold.
      3. Tie-break: higher pneumonia recall, then lower threshold, then fewer FP.

    Architecture context (already established on validation, argmax/0.5):
      ResNet18 partial fine-tune outperformed frozen ResNet18 and EfficientNet-B0.
    """
    candidates = threshold_comparison["candidates"]
    ranked = []
    for name, info in candidates.items():
        metrics = info["selected_metrics"]
        ranked.append(
            {
                "run_name": name,
                "threshold": float(info["selected_threshold"]),
                "metrics": metrics,
                "delta_vs_0_5": info["delta_vs_0_5"],
                "sort_key": (
                    float(metrics["f1_pneumonia"]),
                    float(metrics["recall_pneumonia"]),
                    -float(info["selected_threshold"]),
                    -int(metrics["false_positives"]),
                ),
            }
        )

    if not ranked:
        raise RuntimeError("No threshold-analysis candidates found")

    ranked.sort(key=lambda r: r["sort_key"], reverse=True)
    winner = ranked[0]
    return {
        "rule": (
            "Among ResNet18 fine-tune candidates with validation-selected "
            "thresholds, maximize validation f1_pneumonia; tie-break: higher "
            "recall_pneumonia, then lower threshold, then fewer FP. "
            "Not clinical; validation only."
        ),
        "winner": winner,
        "ranked": [
            {
                "run_name": r["run_name"],
                "threshold": r["threshold"],
                "f1_pneumonia": r["metrics"]["f1_pneumonia"],
                "recall_pneumonia": r["metrics"]["recall_pneumonia"],
                "false_negatives": r["metrics"]["false_negatives"],
                "false_positives": r["metrics"]["false_positives"],
            }
            for r in ranked
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lock final model + threshold from validation artifacts."
    )
    parser.add_argument(
        "--threshold-comparison",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "threshold_analysis"
        / "comparison_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "final_selection",
    )
    args = parser.parse_args()

    comparison = load_json(args.threshold_comparison)
    selection = select_final_candidate(comparison)
    winner = selection["winner"]
    run_name = winner["run_name"]
    threshold = winner["threshold"]
    checkpoint = CANDIDATE_CHECKPOINTS[run_name]

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    # Sanity: threshold analysis was validation-only.
    if comparison.get("split") != "val":
        raise RuntimeError("Threshold comparison split must be 'val'")

    manifest = {
        "status": "locked",
        "split_used_for_selection": "val",
        "test_accessed": False,
        "final_model": {
            "run_name": run_name,
            "architecture": "resnet18",
            "trainable_scope": "layer4_fc",
            "class_weighted_loss": run_name.endswith("_weighted"),
            "checkpoint": str(checkpoint),
            "checkpoint_relative": str(checkpoint.relative_to(PROJECT_ROOT)),
        },
        "final_threshold": {
            "probability_target": "PNEUMONIA",
            "value": threshold,
            "decision_rule": (
                "Predict PNEUMONIA if P(PNEUMONIA) >= threshold, else NORMAL"
            ),
        },
        "selection": {
            "metric": "validation f1_pneumonia (at validation-selected threshold)",
            "rule": selection["rule"],
            "source_artifact": str(
                args.threshold_comparison.relative_to(PROJECT_ROOT)
            ),
            "ranked_candidates": selection["ranked"],
            "selected_validation_metrics": winner["metrics"],
            "delta_vs_default_0_5": winner["delta_vs_0_5"],
        },
        "notes": [
            "Locked from existing validation artifacts only; no retraining.",
            "Previous experiment artifacts were not modified.",
            "Test set must remain unused until an explicit final test evaluation.",
            "Threshold is a validation operating point, not a clinical optimum.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "final_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Lightweight pointer file for later inference / test eval phases.
    lock_path = args.output_dir / "LOCKED"
    lock_path.write_text(
        (
            f"model={run_name}\n"
            f"checkpoint={checkpoint.relative_to(PROJECT_ROOT)}\n"
            f"threshold={threshold}\n"
            f"split=val\n"
            f"test_accessed=false\n"
        )
    )

    print("Final selection locked")
    print(f"  model={run_name}")
    print(f"  checkpoint={checkpoint}")
    print(f"  threshold={threshold}")
    print(f"  metric_rule=max validation f1_pneumonia")
    print(f"  split=val")
    print(f"  test_accessed=false")
    print(f"  wrote {manifest_path}")
    print(f"  wrote {lock_path}")


if __name__ == "__main__":
    main()
