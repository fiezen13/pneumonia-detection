"""EfficientNet-B0 for transfer learning / partial fine-tuning."""

from __future__ import annotations

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.models.cnn import count_parameters
from src.models.resnet import describe_trainable_parameters

# Analogous to ResNet18 layer4+fc: last MBConv stage + final conv + classifier.
TRAINABLE_CLASSIFIER = "classifier"
TRAINABLE_LATE_BLOCKS = "features7_8_classifier"
TRAINABLE_ALL = "all"


def build_efficientnet_b0(
    num_classes: int = 2,
    *,
    pretrained: bool = True,
    trainable_scope: str = TRAINABLE_LATE_BLOCKS,
) -> nn.Module:
    """Build EfficientNet-B0 with a 2-class classifier.

    Default ``trainable_scope`` matches the best ResNet18 setup: fine-tune
    late feature blocks + head while keeping early features frozen.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    apply_trainable_scope(model, trainable_scope)
    return model


def apply_trainable_scope(model: nn.Module, trainable_scope: str) -> None:
    """Set requires_grad for EfficientNet-B0 trainable scopes."""
    scope = trainable_scope.lower()
    if scope == TRAINABLE_CLASSIFIER:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("classifier.")
    elif scope == TRAINABLE_LATE_BLOCKS:
        for name, param in model.named_parameters():
            param.requires_grad = (
                name.startswith("features.7.")
                or name.startswith("features.8.")
                or name.startswith("classifier.")
            )
    elif scope == TRAINABLE_ALL:
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(
            f"Unknown trainable_scope={trainable_scope!r}. "
            f"Expected one of: {TRAINABLE_CLASSIFIER}, "
            f"{TRAINABLE_LATE_BLOCKS}, {TRAINABLE_ALL}"
        )


def assert_late_blocks_scope(model: nn.Module) -> None:
    """Fail fast unless trainable params are features.7/8 + classifier only."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]

    def _allowed(name: str) -> bool:
        return (
            name.startswith("features.7.")
            or name.startswith("features.8.")
            or name.startswith("classifier.")
        )

    bad_trainable = [n for n in trainable if not _allowed(n)]
    bad_frozen = [n for n in frozen if _allowed(n)]
    if bad_trainable or bad_frozen:
        raise AssertionError(
            "Trainable scope mismatch for EfficientNet late-block fine-tuning. "
            f"unexpected_trainable={bad_trainable} "
            f"unexpected_frozen_late_or_classifier={bad_frozen}"
        )

    for prefix in (
        "features.0.",
        "features.1.",
        "features.2.",
        "features.3.",
        "features.4.",
        "features.5.",
        "features.6.",
    ):
        if not any(n.startswith(prefix) for n in frozen):
            raise AssertionError(f"Expected frozen parameters under {prefix}")


def parameter_summary(model: nn.Module) -> dict[str, object]:
    """Reuse shared parameter summary helper."""
    return describe_trainable_parameters(model)


# Re-export count helper for local callers.
__all__ = [
    "TRAINABLE_ALL",
    "TRAINABLE_CLASSIFIER",
    "TRAINABLE_LATE_BLOCKS",
    "apply_trainable_scope",
    "assert_late_blocks_scope",
    "build_efficientnet_b0",
    "count_parameters",
    "parameter_summary",
]
