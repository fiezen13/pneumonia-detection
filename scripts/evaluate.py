"""Evaluate a checkpoint on val or test without mixing the two."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.evaluation.evaluate import (
    evaluate_model,
    format_evaluation_summary,
    load_checkpoint,
    save_evaluation_report,
)
from src.models.cnn import CustomCNN
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet18
from src.training.trainer import get_device


def build_model(model_name: str):
    name = model_name.lower()
    if name in {"cnn", "custom_cnn", "customcnn"}:
        return CustomCNN(num_classes=2)
    if name == "resnet18_frozen":
        return build_resnet18(
            num_classes=2, pretrained=True, trainable_scope="fc"
        )
    if name in {"resnet18_finetune", "resnet18_partial", "resnet18"}:
        return build_resnet18(
            num_classes=2, pretrained=True, trainable_scope="layer4_fc"
        )
    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        return build_efficientnet_b0(
            num_classes=2,
            pretrained=True,
            trainable_scope="features7_8_classifier",
        )
    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Supported: cnn, resnet18_frozen, resnet18_finetune, efficientnet_b0."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on val or test."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        required=True,
        help="Evaluate exactly one held-out split (val or test).",
    )
    parser.add_argument("--model", type=str, default="cnn")
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path",
    )
    args = parser.parse_args()

    device = get_device()
    model = build_model(args.model)
    model, _ckpt = load_checkpoint(model, args.checkpoint, device=device)

    loaders = create_dataloaders(
        args.data_dir,
        args.split_file,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    loader = loaders[args.split]

    report = evaluate_model(
        model,
        loader,
        split=args.split,
        device=device,
        checkpoint_path=str(args.checkpoint),
    )
    print(format_evaluation_summary(report))

    if args.output is not None:
        out = save_evaluation_report(report, args.output)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
