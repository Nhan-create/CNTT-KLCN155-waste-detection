"""Unit tests for coordinate conversions, letterboxing, EXIF handling, and clipping."""

from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image

from src.detection.coordinates import (
    clip_and_validate_box,
    ensure_exif_orientation,
    letterbox_image,
    scale_coords_back,
    scale_coords_direct,
)
from src.detection.types import BoundingBox


def test_letterbox_preserves_aspect_ratio_horizontal_532x299() -> None:
    # 532 x 299 image letterboxed to 640 x 640
    orig = Image.new("RGB", (532, 299), color=(200, 200, 200))
    canvas, scale, (dx, dy) = letterbox_image(orig, target_size=640)

    assert canvas.size == (640, 640)
    expected_scale = min(640 / 532, 640 / 299)
    assert scale == pytest.approx(expected_scale, rel=1e-5)
    # Width is the constraining dimension (532 * scale ≈ 640), so dx ≈ 0, dy > 0
    assert dx == pytest.approx(0.0, abs=1.0)
    assert dy > 0.0


def test_letterbox_preserves_aspect_ratio_vertical_1080x1920() -> None:
    # 1080 x 1920 vertical phone photo letterboxed to 640 x 640
    orig = Image.new("RGB", (1080, 1920), color=(100, 150, 200))
    canvas, scale, (dx, dy) = letterbox_image(orig, target_size=640)

    assert canvas.size == (640, 640)
    expected_scale = min(640 / 1080, 640 / 1920)
    assert scale == pytest.approx(expected_scale, rel=1e-5)
    # Height is constraining dimension, so dy ≈ 0, dx > 0
    assert dy == pytest.approx(0.0, abs=1.0)
    assert dx > 0.0


def test_scale_coords_back_roundtrip_forward_and_inverse() -> None:
    orig_w, orig_h = 532, 299
    target_size = 640
    image = Image.new("RGB", (orig_w, orig_h), color=(50, 50, 50))
    _, scale, (dx, dy) = letterbox_image(image, target_size=target_size)

    # Define ground truth boxes in original pixel space
    original_boxes = [
        BoundingBox(50.0, 30.0, 200.0, 150.0),
        BoundingBox(10.0, 10.0, 500.0, 280.0),
    ]

    # Forward map to letterbox space: x_padded = x * scale + dx
    letterbox_boxes = [
        (
            box.x1 * scale + dx,
            box.y1 * scale + dy,
            box.x2 * scale + dx,
            box.y2 * scale + dy,
        )
        for box in original_boxes
    ]

    # Inverse map back to original pixel space
    restored = scale_coords_back(letterbox_boxes, scale, (dx, dy), (orig_w, orig_h))

    assert len(restored) == len(original_boxes)
    for orig_b, rest_b in zip(original_boxes, restored, strict=True):
        assert rest_b.x1 == pytest.approx(orig_b.x1, abs=0.5)
        assert rest_b.y1 == pytest.approx(orig_b.y1, abs=0.5)
        assert rest_b.x2 == pytest.approx(orig_b.x2, abs=0.5)
        assert rest_b.y2 == pytest.approx(orig_b.y2, abs=0.5)
        # Bounding box must be strictly within bounds
        assert 0.0 <= rest_b.x1 < rest_b.x2 <= float(orig_w)
        assert 0.0 <= rest_b.y1 < rest_b.y2 <= float(orig_h)


def test_scale_coords_back_eliminates_empty_and_nan_boxes() -> None:
    orig_w, orig_h = 640, 480
    scale = 1.0
    pad = (0.0, 0.0)

    boxes = [
        (10.0, 20.0, 100.0, 200.0),            # valid
        (float("nan"), 20.0, 100.0, 200.0),    # NaN x1
        (10.0, float("inf"), 100.0, 200.0),    # Inf y1
        (100.0, 20.0, 10.0, 200.0),            # inverted x1 >= x2
        (10.0, 200.0, 100.0, 20.0),            # inverted y1 >= y2
        (10.0, 20.0, 10.0, 200.0),             # zero width
        (700.0, 20.0, 800.0, 200.0),           # completely outside right
        (-100.0, 20.0, -10.0, 200.0),          # completely outside left
    ]

    restored = scale_coords_back(boxes, scale, pad, (orig_w, orig_h))
    assert len(restored) == 1
    assert restored[0].x1 == 10.0
    assert restored[0].y1 == 20.0
    assert restored[0].x2 == 100.0
    assert restored[0].y2 == 200.0


def test_scale_coords_back_clips_partially_outside_boxes() -> None:
    orig_w, orig_h = 200, 100
    scale = 1.0
    pad = (0.0, 0.0)

    # Box partially outside boundary [-20, -10, 250, 120]
    boxes = [(-20.0, -10.0, 250.0, 120.0)]
    restored = scale_coords_back(boxes, scale, pad, (orig_w, orig_h))

    assert len(restored) == 1
    box = restored[0]
    assert box.x1 == 0.0
    assert box.y1 == 0.0
    assert box.x2 == 200.0
    assert box.y2 == 100.0
    assert 0.0 <= box.x1 < box.x2 <= orig_w
    assert 0.0 <= box.y1 < box.y2 <= orig_h


def test_scale_coords_direct_ssdlite_mapping() -> None:
    # SSDLite direct stretch from 320x320 to 1920x1080
    orig_size = (1920, 1080)
    input_size = (320, 320)

    # A box at (32, 32, 160, 160) in 320x320
    boxes = [(32.0, 32.0, 160.0, 160.0)]
    restored = scale_coords_direct(boxes, input_size, orig_size)

    assert len(restored) == 1
    box = restored[0]
    # Expected: x * (1920/320) = x * 6, y * (1080/320) = y * 3.375
    assert box.x1 == pytest.approx(32.0 * 6.0)
    assert box.y1 == pytest.approx(32.0 * 3.375)
    assert box.x2 == pytest.approx(160.0 * 6.0)
    assert box.y2 == pytest.approx(160.0 * 3.375)
    assert 0.0 <= box.x1 < box.x2 <= 1920
    assert 0.0 <= box.y1 < box.y2 <= 1080


def test_clip_and_validate_box() -> None:
    # Valid box
    assert clip_and_validate_box((10, 10, 50, 50), 100, 100) == BoundingBox(10, 10, 50, 50)
    # Box exceeding border gets clipped
    assert clip_and_validate_box((-5, -10, 110, 120), 100, 100) == BoundingBox(0, 0, 100, 100)
    # Empty / inverted box returns None
    assert clip_and_validate_box((50, 50, 20, 20), 100, 100) is None
    # Out of bounds box returns None
    assert clip_and_validate_box((120, 10, 150, 50), 100, 100) is None
    # NaN returns None
    assert clip_and_validate_box((float("nan"), 10, 50, 50), 100, 100) is None


def test_ensure_exif_orientation() -> None:
    # Test that an image without EXIF remains intact in RGB
    raw = Image.new("RGBA", (80, 60), color=(255, 0, 0, 255))
    transposed = ensure_exif_orientation(raw)
    assert transposed.mode == "RGB"
    assert transposed.size == (80, 60)
