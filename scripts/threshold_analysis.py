"""Validation threshold analysis for ResNet18 fine-tune candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.threshold import (
    DEFAULT_THRESHOLDS,
    load_prediction_arrays,
    plot_threshold_curves,
    rows_to_table,
    run_threshold_sweep,
    select_threshold_max_pneumonia_f1,
)

CANDIDATES = {
    "resnet18_finetune": {
        "predictions": PROJECT_ROOT
        / "experiments"
        / "error_analysis"
        / "resnet18_finetune"
        / "predictions.json",
        "expected_cm_at_0_5": [[198, 3], [15, 568]],
    },
    "resnet18_finetune_weighted": {
        "predictions": PROJECT_ROOT
        / "experiments"
        / "error_analysis"
        / "resnet18_finetune_weighted"
        / "predictions.json",
        "expected_cm_at_0_5": [[196, 5], [10, 573]],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only P(PNEUMONIA) threshold analysis."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "threshold_analysis",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "threshold_analysis",
    )
    args = parser.parse_args()
    args.json_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    comparison: dict = {
        "split": "val",
        "thresholds": DEFAULT_THRESHOLDS,
        "selection_rule": (
            "Maximize validation f1_pneumonia; "
            "tie-break: higher recall_pneumonia, then lower threshold. "
            "Not clinical; validation only; test unused."
        ),
        "candidates": {},
    }

    for run_name, meta in CANDIDATES.items():
        pred_path = meta["predictions"]
        assert pred_path.is_file(), f"Missing predictions: {pred_path}"
        arrays = load_prediction_arrays(pred_path)
        assert len(arrays["y_true"]) == 784

        rows = run_threshold_sweep(
            arrays["y_true"], arrays["y_prob"], DEFAULT_THRESHOLDS
        )
        row_0_5 = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-9)
        expected = meta["expected_cm_at_0_5"]
        assert row_0_5["confusion_matrix"] == expected, (
            f"{run_name} threshold=0.5 CM mismatch: "
            f"{row_0_5['confusion_matrix']} != {expected}"
        )

        selection = select_threshold_max_pneumonia_f1(rows)
        selected = selection["selected"]
        delta = {
            "threshold_default": 0.5,
            "threshold_selected": selected["threshold"],
            "fn_at_0_5": row_0_5["false_negatives"],
            "fp_at_0_5": row_0_5["false_positives"],
            "fn_at_selected": selected["false_negatives"],
            "fp_at_selected": selected["false_positives"],
            "fn_delta": selected["false_negatives"] - row_0_5["false_negatives"],
            "fp_delta": selected["false_positives"] - row_0_5["false_positives"],
            "recall_pneumonia_at_0_5": row_0_5["recall_pneumonia"],
            "recall_pneumonia_at_selected": selected["recall_pneumonia"],
            "f1_pneumonia_at_0_5": row_0_5["f1_pneumonia"],
            "f1_pneumonia_at_selected": selected["f1_pneumonia"],
        }

        fig_path = plot_threshold_curves(
            rows,
            args.figure_dir / f"{run_name}_threshold_curves.png",
            title=f"{run_name} — validation threshold trade-off",
            selected_threshold=selected["threshold"],
        )

        out = {
            "run_name": run_name,
            "split": "val",
            "n_samples": len(arrays["y_true"]),
            "predictions_source": str(pred_path),
            "selection": selection,
            "delta_vs_0_5": delta,
            "table": rows_to_table(rows),
            "rows_full": rows,
        }
        out_path = args.json_dir / f"{run_name}_threshold_analysis.json"
        out_path.write_text(json.dumps(out, indent=2) + "\n")

        print(f"\n=== {run_name} ===")
        print("threshold  acc   P_pneu  R_pneu  F1_pneu  FN   FP   CM")
        for r in rows_to_table(rows):
            marker = " *" if abs(r["threshold"] - selected["threshold"]) < 1e-9 else ""
            print(
                f"{r['threshold']:5.2f}     "
                f"{r['accuracy']:.4f} "
                f"{r['precision_pneumonia']:.4f} "
                f"{r['recall_pneumonia']:.4f} "
                f"{r['f1_pneumonia']:.4f}  "
                f"{r['false_negatives']:3d}  {r['false_positives']:3d}  "
                f"{r['confusion_matrix']}{marker}"
            )
        print(f"verified @0.5 CM={row_0_5['confusion_matrix']}")
        print(
            f"selected threshold={selected['threshold']} "
            f"F1_pneu={selected['f1_pneumonia']:.4f} "
            f"R_pneu={selected['recall_pneumonia']:.4f} "
            f"FN={selected['false_negatives']} FP={selected['false_positives']}"
        )
        print(
            f"vs 0.5: FN {delta['fn_at_0_5']}→{delta['fn_at_selected']} "
            f"(Δ{delta['fn_delta']:+d}), "
            f"FP {delta['fp_at_0_5']}→{delta['fp_at_selected']} "
            f"(Δ{delta['fp_delta']:+d})"
        )
        print(f"wrote {out_path}")
        print(f"wrote {fig_path}")

        comparison["candidates"][run_name] = {
            "selected_threshold": selected["threshold"],
            "selected_metrics": {
                k: selected[k]
                for k in (
                    "accuracy",
                    "precision_pneumonia",
                    "recall_pneumonia",
                    "f1_pneumonia",
                    "false_negatives",
                    "false_positives",
                    "confusion_matrix",
                )
            },
            "delta_vs_0_5": delta,
            "table": rows_to_table(rows),
            "figure": str(fig_path),
            "json": str(out_path),
        }

    comparison_path = args.json_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\nWrote {comparison_path}")
    print("Validation threshold analysis complete (test unused).")


if __name__ == "__main__":
    main()
