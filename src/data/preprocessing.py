"""Image transforms for chest X-ray classification."""

from __future__ import annotations

from torchvision import transforms

# ImageNet stats — required for pretrained torchvision backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_SIZE = 224


def build_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    horizontal_flip_p: float = 0.5,
    rotation_degrees: float = 10.0,
    translate: float = 0.05,
    scale_min: float = 0.95,
    scale_max: float = 1.05,
    brightness: float = 0.1,
    contrast: float = 0.1,
) -> transforms.Compose:
    """Mild training augmentation + ImageNet normalization."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=horizontal_flip_p),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=(translate, translate),
                scale=(scale_min, scale_max),
            ),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Deterministic resize + ImageNet normalization (val/test)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
