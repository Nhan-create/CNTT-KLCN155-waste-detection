"""Data Augmentation module for 10-Class Waste Detection (Phase 2).

Implements 4 formal ablation strategies using Albumentations:
1. Strategy 'none': Baseline identity transform (no augmentation).
2. Strategy 'geometric': HorizontalFlip, ShiftScaleRotate, RandomAffine with strict bbox bounds.
3. Strategy 'photometric': ColorJitter (brightness, contrast, saturation), GaussianBlur, MotionBlur.
4. Strategy 'combined': Geometric + Photometric transformations.

Scientific Note on Copy-Paste [5] (Ghiasi et al., 2021):
Copy-Paste strictly requires verified instance segmentation masks (polygons).
Rectangular bounding-box crops do NOT constitute valid foreground masks and introduce
severe visual edge artifacts. Since the dataset currently possesses bounding-box annotations only,
Copy-Paste is flagged as BLOCKED pending polygon mask annotations.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


COPY_PASTE_BLOCKED_REASON = (
    "BLOCKER: Copy-Paste [5] requires pixel-level foreground instance segmentation masks. "
    "Current dataset only provides bounding boxes. Rectangular crops must NOT be used as fake masks."
)


def get_detection_transform(strategy: str = "none") -> Any:
    """Return Albumentations Compose pipeline with YOLO bbox format."""
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations package is required for detection data augmentation")

    bbox_params = A.BboxParams(
        format="yolo",
        min_visibility=0.2,
        label_fields=["category_ids"],
        clip=True,
    )

    strategy = strategy.lower().strip()
    if strategy == "none":
        return A.Compose([], bbox_params=bbox_params)

    elif strategy == "geometric":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
        ]
        return A.Compose(transforms, bbox_params=bbox_params)

    elif strategy == "photometric":
        transforms = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.7,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
        ]
        return A.Compose(transforms, bbox_params=bbox_params)

    elif strategy == "combined":
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
        ]
        return A.Compose(transforms, bbox_params=bbox_params)

    elif strategy == "copy_paste":
        raise NotImplementedError(COPY_PASTE_BLOCKED_REASON)

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}. Choose from: none, geometric, photometric, combined")


def apply_augmentation(
    image: Image.Image | np.ndarray,
    boxes: list[list[float]],
    category_ids: list[int],
    strategy: str = "none",
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply synchronized image-bbox transformation.
    
    Args:
        image: PIL image or numpy array (H, W, C) in RGB.
        boxes: List of YOLO normalized boxes [cx, cy, w, h] in range [0, 1].
        category_ids: List of integer class IDs.
        strategy: 'none', 'geometric', 'photometric', or 'combined'.
        
    Returns:
        transformed_image (numpy uint8 array), transformed_boxes, transformed_category_ids.
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.asarray(image)

    transform = get_detection_transform(strategy)
    transformed = transform(
        image=img_np,
        bboxes=boxes,
        category_ids=category_ids,
    )
    return (
        transformed["image"],
        list(transformed["bboxes"]),
        list(transformed["category_ids"]),
    )
