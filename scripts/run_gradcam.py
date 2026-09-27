"""Generate Grad-CAM overlays for representative validation examples.

Uses ResNet18 partial fine-tuning (unweighted) as the selected architecture.
Validation images only — test set is never accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data.split import load_split
from src.evaluation.evaluate import load_checkpoint
from src.models.resnet import build_resnet18
from src.training.trainer import get_device
from src.visualization.gradcam import (
    GradCAM,
    get_resnet18_target_layer,
    prepare_input,
    save_gradcam_figure,
)

# Selected model: ResNet18 partial fine-tuning (best architecture vs EfficientNet).
RUN_NAME = "resnet18_finetune"
CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "resnet18_finetune"
    / "resnet18_finetune_best.pt"
)
PREDICTIONS = (
    PROJECT_ROOT
    / "experiments"
    / "error_analysis"
    / "resnet18_finetune"
    / "predictions.json"
)


def pick_representative(records: list[dict], error_type: str) -> dict:
    """Pick a confident example of the requested type from val predictions."""
    subset = [r for r in records if r["error_type"] == error_type]
    if not subset:
        raise ValueError(f"No validation records with error_type={error_type}")
    return sorted(subset, key=lambda r: r["pred_confidence"], reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grad-CAM on validation examples (no test access)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "chest_xray",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "gradcam",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "gradcam",
    )
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    # Guard: only use validation paths from the saved split + error analysis.
    split = load_split(PROJECT_ROOT / "experiments" / "data_split.json")
    val_set = set(split["val"])
    test_set = set(split["test"])

    records = json.loads(PREDICTIONS.read_text())
    examples = {
        "TP": pick_representative(records, "TP"),
        "TN": pick_representative(records, "TN"),
        "FP": pick_representative(records, "FP"),
        "FN": pick_representative(records, "FN"),
    }

    for et, rec in examples.items():
        rel = rec["rel_path"]
        if rel not in val_set:
            raise RuntimeError(f"{et} example not in validation split: {rel}")
        if rel in test_set:
            raise RuntimeError(f"Refusing test-set image: {rel}")

    device = get_device()
    model = build_resnet18(
        num_classes=2, pretrained=True, trainable_scope="layer4_fc"
    )
    model, _ = load_checkpoint(model, CHECKPOINT, device=device)
    target_layer, target_name = get_resnet18_target_layer(model)
    print(f"model={RUN_NAME} checkpoint={CHECKPOINT}")
    print(f"target_layer={target_name} type={type(target_layer).__name__}")
    print(f"device={device} test_accessed=False")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.json_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_name": RUN_NAME,
        "checkpoint": str(CHECKPOINT),
        "split": "val",
        "target_layer": target_name,
        "test_accessed": False,
        "disclaimer": (
            "Grad-CAM is an interpretability visualization only. "
            "It does not prove medical relevance of highlighted regions."
        ),
        "examples": {},
    }

    with GradCAM(model, target_layer) as cam:
        for error_type, rec in examples.items():
            rel = rec["rel_path"]
            image_path = args.data_dir / rel
            tensor, original = prepare_input(
                image_path, image_size=args.image_size, device=device
            )
            # Explain the model's predicted class (standard Grad-CAM usage).
            result = cam.generate(tensor, target_class=None)

            # Shape checks for ResNet18 layer4 on 224 input → 7x7 spatial map.
            act_shape = result["activation_shape"]
            heat_shape = result["heatmap_shape"]
            assert act_shape[0] == 512, act_shape  # ResNet18 layer4 channels
            assert heat_shape == (7, 7), heat_shape
            print(
                f"{error_type}: activation={act_shape} heatmap={heat_shape} "
                f"pred={result['pred_class']} "
                f"P(PNEU)={result['probs'][1]:.4f}"
            )

            pred_label = "PNEUMONIA" if result["pred_class"] == 1 else "NORMAL"
            # Consistency with saved error-analysis prediction.
            assert pred_label == rec["predicted_label"], (
                pred_label,
                rec["predicted_label"],
            )

            fig_path = save_gradcam_figure(
                original,
                result["heatmap"],
                args.figure_dir / f"{RUN_NAME}_{error_type}_gradcam.png",
                title=f"{RUN_NAME} — {error_type} (validation)",
                true_label=rec["true_label"],
                pred_label=pred_label,
                prob_pneumonia=float(result["probs"][1]),
                rel_path=rel,
                target_layer_name=target_name,
            )

            summary["examples"][error_type] = {
                "rel_path": rel,
                "absolute_path": str(image_path),
                "true_label": rec["true_label"],
                "predicted_label": pred_label,
                "prob_pneumonia": float(result["probs"][1]),
                "pred_confidence": rec["pred_confidence"],
                "activation_shape": list(act_shape),
                "heatmap_shape": list(heat_shape),
                "figure": str(fig_path),
                "in_validation_split": True,
                "in_test_split": False,
            }
            print(f"  wrote {fig_path}")

    out_json = args.json_dir / f"{RUN_NAME}_gradcam_summary.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_json}")
    print("Grad-CAM complete. test_accessed=False")


if __name__ == "__main__":
    main()
