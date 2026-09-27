#!/usr/bin/env python3
"""Single-image inference using the locked final model and threshold.

Usage:
  python scripts/inference.py --image path/to/xray.jpeg

Optional:
  python scripts/inference.py --image path/to/xray.jpeg \\
      --manifest experiments/final_selection/final_selection_manifest.json

Reads checkpoint path and decision threshold from the final-selection
manifest (not hard-coded). Uses the same evaluation preprocessing as
offline evaluation (``build_eval_transforms``).

This is an inference utility only — it does not access the test set,
retrain, or modify the locked selection.
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
from PIL import Image, UnidentifiedImageError

from src.data.preprocessing import DEFAULT_IMAGE_SIZE, build_eval_transforms
from src.evaluation.evaluate import load_checkpoint
from src.models.resnet import build_resnet18
from src.training.trainer import get_device

DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "final_selection" / "final_selection_manifest.json"
)


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Final-selection manifest not found: {path}")
    return json.loads(path.read_text())


def resolve_checkpoint(manifest: dict, project_root: Path) -> Path:
    """Prefer portable relative checkpoint path; fall back to absolute."""
    rel = manifest["final_model"].get("checkpoint_relative")
    if rel:
        candidate = project_root / rel
        if candidate.is_file():
            return candidate
    absolute = Path(manifest["final_model"]["checkpoint"])
    if absolute.is_file():
        return absolute
    raise FileNotFoundError(
        "Locked checkpoint not found. Tried "
        f"relative={rel!r} and absolute={manifest['final_model'].get('checkpoint')!r}"
    )


def build_locked_model(manifest: dict):
    """Reconstruct the locked architecture from manifest metadata."""
    architecture = manifest["final_model"]["architecture"]
    scope = manifest["final_model"]["trainable_scope"]
    if architecture != "resnet18":
        raise ValueError(
            f"Unsupported locked architecture '{architecture}'. "
            "This inference CLI currently supports resnet18 only."
        )
    if scope != "layer4_fc":
        raise ValueError(
            f"Unexpected trainable_scope '{scope}' in manifest; "
            "expected 'layer4_fc' for the locked ResNet18 fine-tune."
        )
    # Weights come from the checkpoint; pretrained init is only for structure.
    return build_resnet18(
        num_classes=2, pretrained=True, trainable_scope="layer4_fc"
    )


def predict_image(
    image_path: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | None = None,
) -> dict:
    """Run locked-model inference on one image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    manifest = load_manifest(manifest_path)
    checkpoint = resolve_checkpoint(manifest, PROJECT_ROOT)
    threshold = float(manifest["final_threshold"]["value"])
    run_name = manifest["final_model"]["run_name"]

    resolved_device = device or get_device()
    model = build_locked_model(manifest)
    model, _ = load_checkpoint(model, checkpoint, device=resolved_device)
    model.eval()

    try:
        image = Image.open(image_path)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot open image '{image_path}': {exc}") from exc

    image = image.convert("RGB")
    transform = build_eval_transforms(image_size)
    tensor = transform(image).unsqueeze(0).to(resolved_device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        prob_pneumonia = float(probs[1].item())

    label = "PNEUMONIA" if prob_pneumonia >= threshold else "NORMAL"
    return {
        "prediction": label,
        "prob_pneumonia": prob_pneumonia,
        "threshold": threshold,
        "model": run_name,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "image": str(image_path),
        "device": str(resolved_device),
    }


def format_result(result: dict) -> str:
    return (
        f"Prediction: {result['prediction']}\n"
        f"Probability: {result['prob_pneumonia']:.4f}\n"
        f"Threshold: {result['threshold']}\n"
        f"Model: {result['model']}\n"
        f"Checkpoint: {result['checkpoint']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer NORMAL/PNEUMONIA for one chest X-ray using the locked "
            "final model and threshold from the selection manifest."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to a single chest X-ray image",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to final_selection_manifest.json",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Input size (must match training/eval; default 224)",
    )
    args = parser.parse_args()

    try:
        result = predict_image(
            args.image,
            manifest_path=args.manifest,
            image_size=args.image_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_result(result))


if __name__ == "__main__":
    main()
