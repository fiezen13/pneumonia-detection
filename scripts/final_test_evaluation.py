"""One-time final TEST evaluation using the locked model and threshold only.

Does not retrain, does not change the threshold, and does not select another model.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset, INDEX_TO_LABEL
from src.data.preprocessing import build_eval_transforms
from src.data.split import load_split
from src.evaluation.evaluate import collect_predictions, load_checkpoint
from src.evaluation.threshold import evaluate_threshold, predict_at_threshold
from src.models.resnet import build_resnet18
from src.training.metrics import compute_classification_metrics
from src.training.trainer import get_device

EXPECTED_RUN_NAME = "resnet18_finetune"
EXPECTED_THRESHOLD = 0.30
EXPECTED_TEST_N = 624
EXPECTED_TEST_NORMAL = 234
EXPECTED_TEST_PNEUMONIA = 390


def plot_confusion_matrix(
    cm: list[list[int]],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Save a simple confusion-matrix figure."""
    mat = np.asarray(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["pred NORMAL", "pred PNEUMONIA"])
    ax.set_yticks([0, 1], ["true NORMAL", "true PNEUMONIA"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final one-time test evaluation from locked manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "final_selection"
        / "final_selection_manifest.json",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "data_split.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "chest_xray",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "final_evaluation",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "final_evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run_name = manifest["final_model"]["run_name"]
    threshold = float(manifest["final_threshold"]["value"])
    checkpoint = Path(manifest["final_model"]["checkpoint"])

    # Hard lock checks — refuse to proceed if the manifest does not match the lock.
    if run_name != EXPECTED_RUN_NAME:
        raise RuntimeError(
            f"Manifest model mismatch: {run_name} != {EXPECTED_RUN_NAME}"
        )
    if abs(threshold - EXPECTED_THRESHOLD) > 1e-9:
        raise RuntimeError(
            f"Manifest threshold mismatch: {threshold} != {EXPECTED_THRESHOLD}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Locked checkpoint missing: {checkpoint}")

    split = load_split(args.split_file)
    test_paths = split["test"]
    n_normal = sum(1 for p in test_paths if "/NORMAL/" in p)
    n_pneumonia = sum(1 for p in test_paths if "/PNEUMONIA/" in p)

    # Sanity checks for the held-out test split.
    assert len(test_paths) == EXPECTED_TEST_N, len(test_paths)
    assert n_normal == EXPECTED_TEST_NORMAL, n_normal
    assert n_pneumonia == EXPECTED_TEST_PNEUMONIA, n_pneumonia
    assert set(test_paths).isdisjoint(set(split["train"]))
    assert set(test_paths).isdisjoint(set(split["val"]))

    print("Sanity checks passed:")
    print(f"  model={run_name}")
    print(f"  checkpoint={checkpoint}")
    print(f"  threshold={threshold}")
    print(f"  split=test n={len(test_paths)} NORMAL={n_normal} PNEUMONIA={n_pneumonia}")

    device = get_device()
    model = build_resnet18(
        num_classes=2, pretrained=True, trainable_scope="layer4_fc"
    )
    model, ckpt = load_checkpoint(model, checkpoint, device=device)

    dataset = ChestXrayDataset(
        args.data_dir,
        test_paths,
        transform=build_eval_transforms(args.image_size),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    assert len(dataset) == EXPECTED_TEST_N

    preds = collect_predictions(model, loader, device)
    assert len(preds["y_true"]) == EXPECTED_TEST_N
    metrics_at_threshold = evaluate_threshold(
        preds["y_true"], preds["y_prob"], threshold
    )
    y_pred_thr = predict_at_threshold(preds["y_prob"], threshold)
    full_metrics = compute_classification_metrics(
        preds["y_true"], y_pred_thr, preds["y_prob"]
    )
    metrics_at_threshold["roc_auc"] = full_metrics["roc_auc"]
    metrics_at_threshold["pr_auc"] = full_metrics["pr_auc"]

    # Also record default-0.5 metrics for transparency (not for selection).
    metrics_at_0_5 = evaluate_threshold(preds["y_true"], preds["y_prob"], 0.5)

    accessed_at = datetime.now(timezone.utc).isoformat()
    report = {
        "evaluation_type": "final_test_once",
        "split": "test",
        "test_accessed": True,
        "accessed_at_utc": accessed_at,
        "locked_from_manifest": str(args.manifest.relative_to(PROJECT_ROOT)),
        "model": {
            "run_name": run_name,
            "checkpoint": str(checkpoint),
            "checkpoint_relative": str(checkpoint.relative_to(PROJECT_ROOT)),
            "architecture": manifest["final_model"]["architecture"],
            "trainable_scope": manifest["final_model"]["trainable_scope"],
            "checkpoint_epoch": ckpt.get("epoch"),
        },
        "threshold": {
            "value": threshold,
            "probability_target": "PNEUMONIA",
            "decision_rule": manifest["final_threshold"]["decision_rule"],
        },
        "sanity_checks": {
            "n_test": len(test_paths),
            "n_normal": n_normal,
            "n_pneumonia": n_pneumonia,
            "checkpoint_exists": True,
            "threshold_matches_lock": True,
            "split": "test",
        },
        "metrics": metrics_at_threshold,
        "metrics_at_default_0_5_for_reference_only": metrics_at_0_5,
        "notes": [
            "One-time final test evaluation using the locked model and threshold.",
            "No retraining, no threshold change, no post-hoc model selection.",
            "metrics_at_default_0_5_for_reference_only is not used for selection.",
            "Results are model performance on the dataset, not clinical claims.",
        ],
    }

    # Per-sample predictions at the locked threshold (for auditability).
    sample_rows = []
    for rel, yt, yp, prob, pred in zip(
        test_paths,
        preds["y_true"],
        preds["y_pred"],
        preds["y_prob"],
        y_pred_thr.tolist(),
    ):
        sample_rows.append(
            {
                "rel_path": rel,
                "true_label": INDEX_TO_LABEL[int(yt)],
                "argmax_pred_label": INDEX_TO_LABEL[int(yp)],
                "threshold_pred_label": INDEX_TO_LABEL[int(pred)],
                "prob_pneumonia": float(prob),
                "threshold": threshold,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "final_test_report.json"
    samples_path = args.output_dir / "final_test_predictions.json"
    cm_fig = plot_confusion_matrix(
        metrics_at_threshold["confusion_matrix"],
        args.figure_dir / "final_test_confusion_matrix.png",
        title=(
            f"Final TEST CM — {run_name}, threshold={threshold:.2f}\n"
            f"n={EXPECTED_TEST_N}"
        ),
    )

    report["artifacts"] = {
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "predictions": str(samples_path.relative_to(PROJECT_ROOT)),
        "confusion_matrix_figure": str(cm_fig.relative_to(PROJECT_ROOT)),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    samples_path.write_text(json.dumps(sample_rows, indent=2) + "\n")

    # Record that test was accessed on the selection manifest (append-only fields).
    manifest["test_accessed"] = True
    manifest["final_test_evaluation"] = {
        "accessed_at_utc": accessed_at,
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "threshold_used": threshold,
        "model_used": run_name,
        "n_test": EXPECTED_TEST_N,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    lock_path = args.manifest.parent / "LOCKED"
    lock_path.write_text(
        (
            f"model={run_name}\n"
            f"checkpoint={checkpoint.relative_to(PROJECT_ROOT)}\n"
            f"threshold={threshold}\n"
            f"split_used_for_selection=val\n"
            f"test_accessed=true\n"
            f"final_test_report={report_path.relative_to(PROJECT_ROOT)}\n"
        )
    )

    m = metrics_at_threshold
    print("\nFinal TEST metrics (locked threshold):")
    print(f"  accuracy={m['accuracy']:.4f}")
    print(
        f"  macro P/R/F1="
        f"{m['precision_macro']:.4f}/{m['recall_macro']:.4f}/{m['f1_macro']:.4f}"
    )
    print(
        f"  NORMAL P/R/F1="
        f"{m['precision_normal']:.4f}/{m['recall_normal']:.4f}/{m['f1_normal']:.4f}"
    )
    print(
        f"  PNEUMONIA P/R/F1="
        f"{m['precision_pneumonia']:.4f}/"
        f"{m['recall_pneumonia']:.4f}/"
        f"{m['f1_pneumonia']:.4f}"
    )
    print(f"  FN={m['false_negatives']} FP={m['false_positives']}")
    print(f"  CM={m['confusion_matrix']}")
    print(f"  ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}")
    print(f"wrote {report_path}")
    print(f"wrote {samples_path}")
    print(f"wrote {cm_fig}")
    print("test_accessed=true")


if __name__ == "__main__":
    main()
