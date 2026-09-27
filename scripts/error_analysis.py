"""Run validation-only error analysis for ResNet18 fine-tune candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.dataset import ChestXrayDataset
from src.data.preprocessing import build_eval_transforms
from src.data.split import load_split
from src.evaluation.evaluate import load_checkpoint
from src.evaluation.error_analysis import (
    plot_error_grid,
    run_error_analysis,
    save_error_analysis,
)
from src.models.resnet import build_resnet18
from src.training.trainer import get_device

CANDIDATES = {
    "resnet18_finetune": {
        "checkpoint": PROJECT_ROOT
        / "checkpoints"
        / "resnet18_finetune"
        / "resnet18_finetune_best.pt",
        "expected_fp": 3,
        "expected_fn": 15,
    },
    "resnet18_finetune_weighted": {
        "checkpoint": PROJECT_ROOT
        / "checkpoints"
        / "resnet18_finetune_weighted"
        / "resnet18_finetune_weighted_best.pt",
        "expected_fp": 5,
        "expected_fn": 10,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation error analysis for ResNet18 candidates."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "chest_xray",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "data_split.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "error_analysis",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "error_analysis",
    )
    parser.add_argument("--max-grid-images", type=int, default=12)
    args = parser.parse_args()

    split = load_split(args.split_file)
    val_paths = split["val"]
    # Explicitly ignore test.
    assert "test" in split

    device = get_device()
    eval_tf = build_eval_transforms(args.image_size)
    dataset = ChestXrayDataset(args.data_dir, val_paths, transform=eval_tf)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    # Keep path order aligned with DataLoader (no shuffle).
    rel_paths = list(val_paths)

    comparison = {"split": "val", "n_val": len(val_paths), "candidates": {}}

    for run_name, meta in CANDIDATES.items():
        ckpt_path = meta["checkpoint"]
        assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

        model = build_resnet18(
            num_classes=2, pretrained=True, trainable_scope="layer4_fc"
        )
        model, _ = load_checkpoint(model, ckpt_path, device=device)

        analysis = run_error_analysis(
            model,
            loader,
            rel_paths,
            split="val",
            device=device,
            checkpoint_path=str(ckpt_path),
            run_name=run_name,
        )

        # Verify against known confusion-matrix FP/FN from prior val reports.
        assert analysis["counts"]["FP"] == meta["expected_fp"], analysis["counts"]
        assert analysis["counts"]["FN"] == meta["expected_fn"], analysis["counts"]

        json_out = args.json_dir / run_name
        saved = save_error_analysis(analysis, json_out)

        fig_dir = args.figure_dir / run_name
        fp_fig = plot_error_grid(
            analysis["false_positives"],
            args.data_dir,
            fig_dir / "fp_grid.png",
            title=f"{run_name} — False Positives (NORMAL→PNEUMONIA), val",
            max_images=args.max_grid_images,
        )
        fn_fig = plot_error_grid(
            analysis["false_negatives"],
            args.data_dir,
            fig_dir / "fn_grid.png",
            title=f"{run_name} — False Negatives (PNEUMONIA→NORMAL), val",
            max_images=args.max_grid_images,
        )

        print(f"\n=== {run_name} ===")
        print(f"CM={analysis['confusion_matrix']} counts={analysis['counts']}")
        print(f"FP confidence: {analysis['fp_confidence']}")
        print(f"FN confidence: {analysis['fn_confidence']}")
        if analysis["false_positives"]:
            top_fp = analysis["false_positives"][0]
            print(
                f"most confident FP: {top_fp['rel_path']} "
                f"P(PNEU)={top_fp['prob_pneumonia']:.4f}"
            )
        if analysis["false_negatives"]:
            top_fn = analysis["false_negatives"][0]
            print(
                f"most confident FN: {top_fn['rel_path']} "
                f"P(PNEU)={top_fn['prob_pneumonia']:.4f} "
                f"conf={top_fn['pred_confidence']:.4f}"
            )
        print(f"wrote {saved['summary']}")
        print(f"wrote {fp_fig}")
        print(f"wrote {fn_fig}")

        comparison["candidates"][run_name] = {
            "counts": analysis["counts"],
            "confusion_matrix": analysis["confusion_matrix"],
            "fp_confidence": analysis["fp_confidence"],
            "fn_confidence": analysis["fn_confidence"],
            "artifacts": {
                "summary": str(saved["summary"]),
                "fp_json": str(saved["false_positives"]),
                "fn_json": str(saved["false_negatives"]),
                "fp_grid": str(fp_fig) if fp_fig else None,
                "fn_grid": str(fn_fig) if fn_fig else None,
            },
        }

    comparison_path = args.json_dir / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"\nWrote {comparison_path}")
    print("Validation error analysis complete (test unused).")


if __name__ == "__main__":
    main()
