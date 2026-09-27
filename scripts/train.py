"""Train a model from a YAML config using the existing Trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import create_dataloaders
from src.data.split import load_split
from src.models.cnn import CustomCNN
from src.models.efficientnet import (
    assert_late_blocks_scope,
    build_efficientnet_b0,
)
from src.models.resnet import (
    assert_layer4_fc_scope,
    build_resnet18,
    describe_trainable_parameters,
)
from src.training.losses import compute_balanced_class_weights
from src.training.trainer import TrainConfig, Trainer


def build_model(model_name: str):
    name = model_name.lower()
    if name in {"cnn", "custom_cnn", "customcnn"}:
        return CustomCNN(num_classes=2)
    if name == "resnet18_frozen":
        return build_resnet18(
            num_classes=2, pretrained=True, trainable_scope="fc"
        )
    if name in {"resnet18_finetune", "resnet18_partial"}:
        return build_resnet18(
            num_classes=2, pretrained=True, trainable_scope="layer4_fc"
        )
    if name in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        return build_efficientnet_b0(
            num_classes=2,
            pretrained=True,
            trainable_scope="features7_8_classifier",
        )
    raise ValueError(f"Unknown model '{model_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train from a YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML config",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text())
    model_name = raw["model"]
    data_dir = PROJECT_ROOT / raw.get("data_dir", "data/chest_xray")
    split_file = PROJECT_ROOT / raw.get("split_file", "experiments/data_split.json")
    checkpoint_dir = PROJECT_ROOT / raw.get("checkpoint_dir", "checkpoints")

    model = build_model(model_name)
    param_info = describe_trainable_parameters(model)
    print(f"model={model_name}")
    print(
        f"parameters total={param_info['total_parameters']:,} "
        f"trainable={param_info['trainable_parameters']:,} "
        f"frozen={param_info['frozen_parameters']:,}"
    )
    print(f"trainable tensors ({param_info['n_trainable_param_tensors']}):")
    for name in param_info["trainable_param_names"]:
        print(f"  - {name}")

    lowered = model_name.lower()
    if lowered in {"resnet18_finetune", "resnet18_partial"}:
        assert_layer4_fc_scope(model)
        print("Verified trainable scope: layer4.* + fc.* only")
    if lowered in {"efficientnet_b0", "efficientnet-b0", "efficientnetb0"}:
        assert_late_blocks_scope(model)
        print(
            "Verified trainable scope: features.7.* + features.8.* + classifier.* only"
        )

    split = load_split(split_file)
    class_weight_info = None
    class_weights = None
    if bool(raw.get("use_class_weights", False)):
        # Train split only — never use val/test for weight estimation.
        class_weight_info = compute_balanced_class_weights(split["train"])
        class_weights = class_weight_info["weights"]
        print(
            "class_weights source=train_only "
            f"counts={class_weight_info['counts']} "
            f"formula={class_weight_info['formula']} "
            f"weights(NORMAL,PNEUMONIA)={class_weights}"
        )
        assert class_weight_info["n_samples"] == len(split["train"])
        assert class_weight_info["source_split"] == "train"

    loaders = create_dataloaders(
        data_dir,
        split_file,
        image_size=int(raw.get("image_size", 224)),
        batch_size=int(raw.get("batch_size", 16)),
        num_workers=int(raw.get("num_workers", 0)),
    )

    backbone_lr = raw.get("backbone_learning_rate", None)
    config = TrainConfig(
        epochs=int(raw.get("epochs", 10)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        backbone_learning_rate=(
            float(backbone_lr) if backbone_lr is not None else None
        ),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        early_stopping_patience=int(raw.get("early_stopping_patience", 5)),
        scheduler=str(raw.get("scheduler", "cosine")),
        seed=int(raw.get("seed", 42)),
        class_weights=class_weights,
        checkpoint_dir=str(checkpoint_dir),
        run_name=str(raw.get("run_name", "run")),
        monitor_metric=str(raw.get("monitor_metric", "val_loss")),
        maximize_monitor=bool(raw.get("maximize_monitor", False)),
    )
    print(
        f"optimizer LR head={config.learning_rate}"
        + (
            f" backbone={config.backbone_learning_rate}"
            if config.backbone_learning_rate is not None
            else ""
        )
        + (
            f" class_weights={config.class_weights}"
            if config.class_weights is not None
            else " class_weights=None"
        )
    )

    trainer = Trainer(model, loaders["train"], loaders["val"], config)
    print(
        f"device={trainer.device} train={len(loaders['train'].dataset)} "
        f"val={len(loaders['val'].dataset)}"
    )

    result = trainer.fit()

    param_info_save = {
        k: v for k, v in param_info.items() if k != "frozen_param_names"
    }
    summary = {
        "model": model_name,
        "config": raw,
        "param_info": param_info_save,
        "class_weight_info": class_weight_info,
        "train_result": {
            "best_epoch": result["best_epoch"],
            "best_metric": result["best_metric"],
            "monitor_metric": result["monitor_metric"],
            "best_checkpoint": result["best_checkpoint"],
            "last_checkpoint": result["last_checkpoint"],
            "history_path": result["history_path"],
            "device": result["device"],
            "history": result["history"],
        },
    }
    summary_path = checkpoint_dir / f"{config.run_name}_train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")
    print(
        f"Training done. best_epoch={result['best_epoch']} "
        f"{result['monitor_metric']}={result['best_metric']:.4f}"
    )


if __name__ == "__main__":
    main()
