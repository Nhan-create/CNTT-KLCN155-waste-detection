"""Tests for detection data augmentation pipeline."""

import numpy as np
import pytest
from src.detection.augmentation import apply_augmentation, get_detection_transform, COPY_PASTE_BLOCKED_REASON


def test_augmentation_strategies_preserve_bbox_bounds():
    # Synthetic test image 100x100 RGB
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [[0.5, 0.5, 0.2, 0.2]]
    cats = [0]

    for strategy in ("none", "geometric", "photometric", "combined"):
        t_img, t_boxes, t_cats = apply_augmentation(image, boxes, cats, strategy=strategy)
        assert t_img.shape == (100, 100, 3)
        assert len(t_boxes) == len(t_cats)
        for box in t_boxes:
            cx, cy, w, h = box
            assert 0.0 <= cx <= 1.0
            assert 0.0 <= cy <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0


def test_copy_paste_raises_not_implemented_with_reason():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [[0.5, 0.5, 0.2, 0.2]]
    cats = [0]

    with pytest.raises(NotImplementedError, match="BLOCKER: Copy-Paste"):
        apply_augmentation(image, boxes, cats, strategy="copy_paste")
