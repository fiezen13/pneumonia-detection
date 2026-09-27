"""Small from-scratch CNN baseline for binary chest X-ray classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Lightweight CNN baseline.

    Four conv blocks downsample 224→14, then global average pooling and a
    linear classifier with 2 logits (NORMAL / PNEUMONIA).
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # 224 → 112
            ConvBlock(32, 64),            # 112 → 56
            ConvBlock(64, 128),           # 56 → 28
            ConvBlock(128, 256),          # 28 → 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
