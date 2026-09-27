"""Grad-CAM for ResNet-style CNNs (interpretability only, not clinical evidence)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.data.preprocessing import (
    DEFAULT_IMAGE_SIZE,
    build_eval_transforms,
)

class GradCAM:
    """Grad-CAM using gradients of a class score w.r.t. a convolutional feature map."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        *,
        target_class: int | None = None,
    ) -> dict[str, Any]:
        """Return Grad-CAM heatmap for one NCHW image tensor.

        If ``target_class`` is None, uses the predicted class (argmax).
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError("input_tensor must have shape (1, C, H, W)")

        self.model.zero_grad(set_to_none=True)
        # Enable input grads so backward hooks fire cleanly through frozen layers.
        x = input_tensor.detach().requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        class_idx = pred_class if target_class is None else int(target_class)

        score = logits[0, class_idx]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Hooks did not capture activations/gradients")

        activations = self._activations[0]  # (C, H, W)
        gradients = self._gradients[0]  # (C, H, W)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        if float(cam.max()) > 0:
            cam = cam / cam.max()

        heatmap = cam.detach().cpu().numpy().astype(np.float32)
        return {
            "heatmap": heatmap,
            "logits": logits.detach().cpu().numpy()[0],
            "probs": probs.detach().cpu().numpy()[0],
            "pred_class": pred_class,
            "target_class": class_idx,
            "activation_shape": tuple(activations.shape),
            "gradient_shape": tuple(gradients.shape),
            "heatmap_shape": tuple(heatmap.shape),
        }


def prepare_input(
    image_path: Path | str,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, Image.Image]:
    """Load an image for Grad-CAM: model tensor + original RGB PIL image."""
    path = Path(image_path)
    original = Image.open(path).convert("RGB")
    transform = build_eval_transforms(image_size)
    tensor = transform(original).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor, original


def overlay_heatmap(
    original: Image.Image,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Resize heatmap to the original image and blend with a jet colormap."""
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original.size, resample=Image.BILINEAR
    )
    heat_np = np.asarray(heat).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")
    colored = cmap(heat_np)[..., :3]  # RGB float
    base = np.asarray(original).astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * colored
    return np.clip(blended, 0.0, 1.0)


def save_gradcam_figure(
    original: Image.Image,
    heatmap: np.ndarray,
    output_path: Path | str,
    *,
    title: str,
    true_label: str,
    pred_label: str,
    prob_pneumonia: float,
    rel_path: str,
    target_layer_name: str,
) -> Path:
    """Save a 3-panel figure: original | heatmap | overlay (no clinical claims)."""
    overlay = overlay_heatmap(original, heatmap)
    heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        original.size, resample=Image.BILINEAR
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heat_img, cmap="jet")
    axes[1].set_title(f"Grad-CAM ({target_layer_name})")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        (
            f"{title}\n"
            f"true={true_label}  pred={pred_label}  "
            f"P(PNEUMONIA)={prob_pneumonia:.4f}\n"
            f"{rel_path}\n"
            "Interpretability only — not clinical evidence."
        ),
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def get_resnet18_target_layer(model: nn.Module) -> tuple[nn.Module, str]:
    """Return ResNet18's final convolutional block (``layer4``)."""
    if not hasattr(model, "layer4"):
        raise AttributeError("Model has no layer4; expected a torchvision ResNet")
    return model.layer4, "layer4"
