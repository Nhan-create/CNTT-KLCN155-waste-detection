"""Data Augmentation module for 10-Class Waste Detection (Phase 2).

Implements 4 formal ablation strategies using Albumentations & Simple Copy-Paste:
1. Strategy 'none': Baseline identity transform (no augmentation).
2. Strategy 'geometric': HorizontalFlip, ShiftScaleRotate / Affine with strict bbox bounds.
3. Strategy 'photometric': ColorJitter (brightness, contrast, saturation), GaussianBlur, MotionBlur.
4. Strategy 'combined': Geometric + Photometric + Simple Copy-Paste (Ghiasi et al., 2021).

Scientific Verification Features:
- ReplayCompose telemetry: Records exact applied transforms, parameters, and random state.
- Split-isolated Copy-Paste: Uses verified polygon masks exclusively from train split (TACO).
- Forced Verification Mode: Enables p=1.0 deterministic verification tests without altering
  training stochastic probabilities.
- Synchronized YOLO bounding boxes with explicit box drop reasons.
"""

from __future__ import annotations

import random
from typing import Any
import numpy as np
from PIL import Image

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

from src.detection.copy_paste import DonorBank, apply_copy_paste


def get_detection_transform(strategy: str = "none", force_apply: bool = False) -> Any:
    """Return Albumentations ReplayCompose pipeline with YOLO bbox format.
    
    Args:
        strategy: 'none', 'geometric', 'photometric', or 'combined'.
        force_apply: If True, sets probabilities to 1.0 for verification testing.
                     If False (default), uses official stochastic training probabilities.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("albumentations package is required for detection data augmentation")

    bbox_params = A.BboxParams(
        format="yolo",
        min_visibility=0.2,
        label_fields=["category_ids"],
        clip=True,
    )

    strategy = strategy.lower().strip()
    p_main = 1.0 if force_apply else 0.5
    p_color = 1.0 if force_apply else 0.7
    p_blur = 1.0 if force_apply else 0.3

    if strategy == "none":
        return A.ReplayCompose([], bbox_params=bbox_params)

    elif strategy == "geometric":
        transforms = [
            A.HorizontalFlip(p=p_main),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=p_main,
            ),
        ]
        return A.ReplayCompose(transforms, bbox_params=bbox_params)

    elif strategy == "photometric":
        transforms = [
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=p_color,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=p_blur),
        ]
        return A.ReplayCompose(transforms, bbox_params=bbox_params)

    elif strategy == "combined":
        transforms = [
            A.HorizontalFlip(p=p_main),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=p_main,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=p_color,
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=p_blur),
        ]
        return A.ReplayCompose(transforms, bbox_params=bbox_params)

    elif strategy == "copy_paste":
        # Pure copy_paste uses identity for albumentations, handled in apply_augmentation
        return A.ReplayCompose([], bbox_params=bbox_params)

    else:
        raise ValueError(
            f"Unknown augmentation strategy: '{strategy}'. "
            f"Choose from: none, geometric, photometric, combined, copy_paste"
        )


def apply_augmentation(
    image: Image.Image | np.ndarray,
    boxes: list[list[float]],
    category_ids: list[int],
    strategy: str = "none",
    donor_bank: DonorBank | None = None,
    force_apply: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int]]:
    """Apply synchronized image-bbox transformation with official signature."""
    t_img, t_boxes, t_cats, _ = apply_augmentation_with_telemetry(
        image=image,
        boxes=boxes,
        category_ids=category_ids,
        strategy=strategy,
        donor_bank=donor_bank,
        force_apply=force_apply,
        seed=seed,
    )
    return t_img, t_boxes, t_cats


def apply_augmentation_with_telemetry(
    image: Image.Image | np.ndarray,
    boxes: list[list[float]],
    category_ids: list[int],
    strategy: str = "none",
    donor_bank: DonorBank | None = None,
    force_apply: bool = False,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int], dict[str, Any]]:
    """Apply transformation and return detailed audit telemetry.

    Returns:
        transformed_image, transformed_boxes, transformed_category_ids, telemetry_dict
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.asarray(image).copy()

    orig_box_count = len(boxes)
    norm_strategy = strategy.lower().strip()

    telemetry: dict[str, Any] = {
        "strategy": norm_strategy,
        "forced_verification_mode": force_apply,
        "seed": seed,
        "boxes_before": orig_box_count,
        "boxes_after": orig_box_count,
        "transforms_applied": [],
        "dropped_boxes_count": 0,
        "drop_reasons": [],
        "copy_paste_telemetry": None,
    }

    # 1. Apply Albumentations transform pipeline
    transform = get_detection_transform(norm_strategy, force_apply=force_apply)
    transformed = transform(
        image=img_np,
        bboxes=boxes,
        category_ids=category_ids,
    )

    out_img = transformed["image"]
    out_boxes = [list(b) for b in transformed["bboxes"]]
    out_cats = [int(c) for c in transformed["category_ids"]]

    # Extract Replay telemetry
    replay = transformed.get("replay", {})
    applied_list = []
    for t_info in replay.get("transforms", []):
        t_name = t_info.get("__class_fullname__", "").split(".")[-1]
        is_applied = t_info.get("applied", False)
        if is_applied:
            applied_list.append({
                "name": t_name,
                "params": t_info.get("params", {}),
            })
    telemetry["transforms_applied"] = applied_list

    # Check dropped boxes from Albumentations
    dropped_albu = orig_box_count - len(out_boxes)
    if dropped_albu > 0:
        telemetry["dropped_boxes_count"] += dropped_albu
        telemetry["drop_reasons"].append(
            f"{dropped_albu} box(es) dropped by Albumentations (visibility < 0.2 or clipped outside boundaries)"
        )

    # 2. Apply Copy-Paste if requested in strategy ('combined' or 'copy_paste')
    if norm_strategy in ("combined", "copy_paste") and donor_bank is not None and len(donor_bank) > 0:
        out_img, out_boxes, out_cats, cp_telem = apply_copy_paste(
            image=out_img,
            boxes=out_boxes,
            category_ids=out_cats,
            donor_bank=donor_bank,
            p=1.0 if force_apply else 0.5,
            force_apply=force_apply,
            seed=seed,
        )
        telemetry["copy_paste_telemetry"] = cp_telem
        if cp_telem.get("dropped_occluded_count", 0) > 0:
            telemetry["dropped_boxes_count"] += cp_telem["dropped_occluded_count"]
            telemetry["drop_reasons"].extend(cp_telem.get("drop_reasons", []))

    telemetry["boxes_after"] = len(out_boxes)
    return out_img, out_boxes, out_cats, telemetry
