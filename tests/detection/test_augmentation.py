"""Tests for detection data augmentation pipeline and Copy-Paste module."""

import numpy as np
import pytest
from src.detection.augmentation import (
    apply_augmentation,
    apply_augmentation_with_telemetry,
    get_detection_transform,
)
from src.detection.copy_paste import DonorBank, DonorObject, apply_copy_paste


def test_augmentation_strategies_preserve_bbox_bounds():
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


def test_forced_verification_mode_guarantees_transform_execution():
    # Synthetic non-uniform image
    image = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
    boxes = [[0.5, 0.5, 0.3, 0.3]]
    cats = [1]

    # Test geometric in forced mode
    t_img_geo, t_boxes_geo, _, telem_geo = apply_augmentation_with_telemetry(
        image, boxes, cats, strategy="geometric", force_apply=True, seed=123
    )
    assert telem_geo["forced_verification_mode"] is True
    assert len(telem_geo["transforms_applied"]) > 0
    # Must produce pixel difference
    assert not np.array_equal(t_img_geo, image)

    # Test photometric in forced mode
    t_img_photo, t_boxes_photo, _, telem_photo = apply_augmentation_with_telemetry(
        image, boxes, cats, strategy="photometric", force_apply=True, seed=123
    )
    assert telem_photo["forced_verification_mode"] is True
    assert len(telem_photo["transforms_applied"]) > 0
    assert not np.array_equal(t_img_photo, image)


def test_telemetry_records_transform_details():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    boxes = [[0.5, 0.5, 0.2, 0.2]]
    cats = [0]

    _, _, _, telem = apply_augmentation_with_telemetry(
        image, boxes, cats, strategy="none"
    )
    assert telem["strategy"] == "none"
    assert telem["boxes_before"] == 1
    assert telem["boxes_after"] == 1
    assert telem["dropped_boxes_count"] == 0


def test_copy_paste_synthetic_donor_and_occlusion():
    bg_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Background has box covering center
    bg_boxes = [[0.5, 0.5, 0.4, 0.4]]
    bg_cats = [0]

    # Create synthetic donor: small 20x20 red square
    donor_rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    donor_rgba[:, :, 0] = 255  # Red
    donor_rgba[:, :, 3] = 255  # Solid Alpha
    donor_mask = np.ones((20, 20), dtype=bool)

    donor = DonorObject(
        image_rgba=donor_rgba,
        binary_mask=donor_mask,
        class_id=7,
        class_name="plastic",
        source_image_id="test_donor",
        original_bbox=[0, 0, 20, 20],
    )
    bank = DonorBank([donor])

    out_img, out_boxes, out_cats, telem = apply_copy_paste(
        bg_image, bg_boxes, bg_cats, bank, force_apply=True, max_paste=1
    )
    assert telem["applied"] is True
    assert telem["pasted_objects"] == 1
    assert 7 in out_cats
    assert len(out_boxes) >= 1
