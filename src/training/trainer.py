"""Reusable training / validation engine for pneumonia classifiers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.training.metrics import PNEUMONIA_INDEX, compute_classification_metrics


def _is_classifier_param(name: str) -> bool:
    """True for ResNet ``fc.*`` or EfficientNet ``classifier.*`` head params."""
    return name.startswith("fc.") or name.startswith("classifier.")


@dataclass
class TrainConfig:
    """Hyperparameters for one training run."""

    epochs: int = 10
    learning_rate: float = 1e-3
    # Optional lower LR for non-head trainable params (e.g. layer4).
    backbone_learning_rate: float | None = None
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    scheduler: str = "cosine"  # cosine | step | none
    step_size: int = 5
    step_gamma: float = 0.1
    seed: int = 42
    num_classes: int = 2
    class_weights: list[float] | None = None
    checkpoint_dir: str = "checkpoints"
    run_name: str = "run"
    monitor_metric: str = "val_loss"  # lower is better if ends with loss/fn
    maximize_monitor: bool | None = None
    device: str | None = None  # auto-detect when None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Select CUDA when available unless an explicit device is requested."""
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Train a classification model with validation, checkpointing, early stop."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
    ) -> None:
        self.config = config
        self.device = get_device(config.device)
        set_seed(config.seed)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        weights = None
        if config.class_weights is not None:
            weights = torch.tensor(
                config.class_weights, dtype=torch.float32, device=self.device
            )
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        trainable = [
            (name, p)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        ]
        if not trainable:
            raise ValueError("Model has no trainable parameters for the optimizer")

        if config.backbone_learning_rate is not None:
            backbone_params = [
                p for name, p in trainable if not _is_classifier_param(name)
            ]
            head_params = [
                p for name, p in trainable if _is_classifier_param(name)
            ]
            if not head_params:
                raise ValueError(
                    "backbone_learning_rate set but no trainable "
                    "classifier/fc parameters found"
                )
            param_groups = []
            if backbone_params:
                param_groups.append(
                    {
                        "params": backbone_params,
                        "lr": config.backbone_learning_rate,
                    }
                )
            param_groups.append(
                {"params": head_params, "lr": config.learning_rate}
            )
            self.optimizer = AdamW(
                param_groups, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = AdamW(
                [p for _, p in trainable],
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self.scheduler = self._build_scheduler()

        self.maximize = (
            config.maximize_monitor
            if config.maximize_monitor is not None
            else not config.monitor_metric.endswith(("loss", "false_negatives"))
        )
        self.best_metric = float("-inf") if self.maximize else float("inf")
        self.best_epoch = 0
        self.epochs_without_improve = 0
        self.history: list[dict[str, Any]] = []

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.checkpoint_dir / f"{config.run_name}_best.pt"
        self.last_path = self.checkpoint_dir / f"{config.run_name}_last.pt"

    def _build_scheduler(self):
        name = self.config.scheduler.lower()
        if name == "none":
            return None
        if name == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=max(1, self.config.epochs)
            )
        if name == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.step_gamma,
            )
        raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

    def _is_improvement(self, value: float) -> bool:
        if self.maximize:
            return value > self.best_metric
        return value < self.best_metric

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return {"train_loss": running_loss / max(1, n_samples)}

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        self.model.eval()
        running_loss = 0.0
        n_samples = 0
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[float] = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, PNEUMONIA_INDEX].cpu().tolist())

        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
        return {
            "val_loss": running_loss / max(1, n_samples),
            "val_accuracy": metrics["accuracy"],
            "val_precision_macro": metrics["precision_macro"],
            "val_recall_macro": metrics["recall_macro"],
            "val_f1_macro": metrics["f1_macro"],
            "val_precision_pneumonia": metrics["precision_pneumonia"],
            "val_recall_pneumonia": metrics["recall_pneumonia"],
            "val_f1_pneumonia": metrics["f1_pneumonia"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
            "val_false_negatives": metrics["false_negatives"],
            "val_confusion_matrix": metrics["confusion_matrix"],
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        is_best: bool,
        metrics: dict[str, Any],
    ) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "monitor_metric": self.config.monitor_metric,
            "metrics": metrics,
            "config": asdict(self.config),
            "is_best": is_best,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path | str) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        return checkpoint

    def fit(self) -> dict[str, Any]:
        """Run training with validation, checkpointing, and early stopping."""
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self.train_epoch()
            val_stats = self.validate()
            if self.scheduler is not None:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "learning_rate": lr,
                **train_stats,
                **{
                    k: v
                    for k, v in val_stats.items()
                    if k != "val_confusion_matrix"
                },
            }
            self.history.append(row)

            monitor_value = val_stats.get(self.config.monitor_metric)
            if monitor_value is None:
                raise KeyError(
                    f"monitor_metric '{self.config.monitor_metric}' "
                    f"not found in validation metrics"
                )

            improved = self._is_improvement(float(monitor_value))
            if improved:
                self.best_metric = float(monitor_value)
                self.best_epoch = epoch
                self.epochs_without_improve = 0
                self.save_checkpoint(
                    self.best_path,
                    epoch=epoch,
                    is_best=True,
                    metrics=val_stats,
                )
            else:
                self.epochs_without_improve += 1

            self.save_checkpoint(
                self.last_path,
                epoch=epoch,
                is_best=False,
                metrics=val_stats,
            )

            print(
                f"epoch {epoch}/{self.config.epochs} "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} "
                f"val_f1_macro={val_stats['val_f1_macro']:.4f} "
                f"val_recall_pneumonia={val_stats['val_recall_pneumonia']:.4f} "
                f"lr={lr:.6f}"
                + (" *" if improved else "")
            )

            if (
                self.config.early_stopping_patience > 0
                and self.epochs_without_improve
                >= self.config.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(best epoch {self.best_epoch}, "
                    f"{self.config.monitor_metric}={self.best_metric:.4f})"
                )
                break

        history_path = self.checkpoint_dir / f"{self.config.run_name}_history.json"
        history_path.write_text(json.dumps(self.history, indent=2) + "\n")

        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "monitor_metric": self.config.monitor_metric,
            "best_checkpoint": str(self.best_path),
            "last_checkpoint": str(self.last_path),
            "history_path": str(history_path),
            "history": self.history,
            "device": str(self.device),
        }
