"""ResNet18 classifiers for transfer learning and fine-tuning experiments."""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from src.models.cnn import count_parameters

# Supported trainable scopes for ResNet18 experiments.
TRAINABLE_FC = "fc"
TRAINABLE_LAYER4_FC = "layer4_fc"
TRAINABLE_ALL = "all"


def build_resnet18(
    num_classes: int = 2,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    trainable_scope: str | None = None,
) -> nn.Module:
    """Build ResNet18 with a 2-class head.

    ``trainable_scope``:
      - ``fc``: only classification head (Phase 7)
      - ``layer4_fc``: layer4 + head (Phase 8)
      - ``all``: full network

    If ``trainable_scope`` is None, ``freeze_backbone=True`` maps to ``fc``
    and ``freeze_backbone=False`` maps to ``all``.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if trainable_scope is None:
        trainable_scope = TRAINABLE_FC if freeze_backbone else TRAINABLE_ALL

    apply_trainable_scope(model, trainable_scope)
    return model


def apply_trainable_scope(model: nn.Module, trainable_scope: str) -> None:
    """Set requires_grad according to the requested trainable scope."""
    scope = trainable_scope.lower()
    if scope == TRAINABLE_FC:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")
    elif scope == TRAINABLE_LAYER4_FC:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("layer4.") or name.startswith(
                "fc."
            )
    elif scope == TRAINABLE_ALL:
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(
            f"Unknown trainable_scope={trainable_scope!r}. "
            f"Expected one of: {TRAINABLE_FC}, {TRAINABLE_LAYER4_FC}, {TRAINABLE_ALL}"
        )


def describe_trainable_parameters(model: nn.Module) -> dict[str, object]:
    """Summarize frozen vs trainable parameters for reporting."""
    trainable_names = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    frozen_names = [
        name for name, p in model.named_parameters() if not p.requires_grad
    ]
    return {
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "frozen_parameters": count_parameters(model)
        - count_parameters(model, trainable_only=True),
        "trainable_param_names": trainable_names,
        "frozen_param_names": frozen_names,
        "n_frozen_param_tensors": len(frozen_names),
        "n_trainable_param_tensors": len(trainable_names),
    }


def assert_layer4_fc_scope(model: nn.Module) -> None:
    """Fail fast if trainable tensors are not exactly layer4.* and fc.*."""
    trainable = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    frozen = [
        name for name, p in model.named_parameters() if not p.requires_grad
    ]
    bad_trainable = [
        n
        for n in trainable
        if not (n.startswith("layer4.") or n.startswith("fc."))
    ]
    bad_frozen = [
        n
        for n in frozen
        if n.startswith("layer4.") or n.startswith("fc.")
    ]
    if bad_trainable or bad_frozen:
        raise AssertionError(
            "Trainable scope mismatch for layer4+fc fine-tuning. "
            f"unexpected_trainable={bad_trainable} "
            f"unexpected_frozen_layer4_or_fc={bad_frozen}"
        )
    required_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.")
    for prefix in required_prefixes:
        if not any(n.startswith(prefix) for n in frozen):
            raise AssertionError(f"Expected frozen parameters under {prefix}")
