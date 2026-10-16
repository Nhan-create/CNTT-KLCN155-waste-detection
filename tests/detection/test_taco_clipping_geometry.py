"""
tests/detection/test_taco_clipping_geometry.py
----------------------------------------------
Mathematical proof and verification of bounding box geometry preservation
during COCO-to-YOLO clipping for out-of-bounds annotations (e.g. TACO y = -1.0).

Verifies that:
1. When y_1 is clipped from -1 to 0, height is reduced by exactly 1 pixel (h' = h - 1).
2. The bottom boundary y_2 = y + h is strictly preserved, preventing vertical box shift.
3. Left, right, and bottom borders in pixel space match ground truth object extent.
4. Degenerate boxes (e.g. completely outside image frame) are safely rejected (return None).
"""

import pytest
from scripts.collect_real_detection_v2 import coco_to_yolo


def yolo_to_pixel_xyxy(yolo_box, W, H):
    """Converts normalized YOLO [xc, yc, w, h] back to pixel [x1, y1, x2, y2]."""
    xc, yc, w, h = yolo_box
    x1 = (xc - w / 2.0) * W
    y1 = (yc - h / 2.0) * H
    x2 = (xc + w / 2.0) * W
    y2 = (yc + h / 2.0) * H
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def test_taco_0112_geometry_preservation():
    """
    taco_0112 (Image 112, Annotation 387):
    COCO bbox: [1225.0, -1.0, 107.0, 92.0] on 1824 x 1824 canvas.
    """
    W, H = 1824, 1824
    coco_box = [1225.0, -1.0, 107.0, 92.0]

    yolo_box = coco_to_yolo(coco_box, W, H)
    assert yolo_box is not None

    xc, yc, w, h = yolo_box

    # Mathematical checks on normalized values:
    # Clipped x1 = 1225.0, x2 = 1332.0 -> width = 107.0 -> w_norm = 107.0 / 1824
    assert w == pytest.approx(107.0 / 1824.0, rel=1e-5)
    # Clipped y1 = 0.0, y2 = 91.0 -> height = 91.0 -> h_norm = 91.0 / 1824 (NOT 92.0!)
    assert h == pytest.approx(91.0 / 1824.0, rel=1e-5)
    # Center y = (0 + 91) / 2 = 45.5 -> yc_norm = 45.5 / 1824
    assert yc == pytest.approx(45.5 / 1824.0, rel=1e-5)

    # Pixel space reconstruction checks:
    x1, y1, x2, y2 = yolo_to_pixel_xyxy(yolo_box, W, H)
    assert y1 == pytest.approx(0.0, abs=1e-4)        # clipped to top boundary
    assert y2 == pytest.approx(91.0, abs=1e-4)       # exact bottom preserved: -1 + 92 = 91
    assert x1 == pytest.approx(1225.0, abs=1e-4)     # left preserved
    assert x2 == pytest.approx(1332.0, abs=1e-4)     # right preserved: 1225 + 107 = 1332


def test_taco_0457_geometry_preservation():
    """
    taco_0457 (Image 457, Annotation 1621):
    COCO bbox: [1167.0, -1.0, 246.0, 269.0] on 4160 x 4160 canvas.
    """
    W, H = 4160, 4160
    coco_box = [1167.0, -1.0, 246.0, 269.0]

    yolo_box = coco_to_yolo(coco_box, W, H)
    assert yolo_box is not None

    xc, yc, w, h = yolo_box

    # Mathematical checks:
    # Clipped y1 = 0.0, y2 = 268.0 -> height = 268.0 (NOT 269.0!)
    assert h == pytest.approx(268.0 / 4160.0, rel=1e-5)
    assert yc == pytest.approx(134.0 / 4160.0, rel=1e-5)

    # Pixel space reconstruction:
    x1, y1, x2, y2 = yolo_to_pixel_xyxy(yolo_box, W, H)
    assert y1 == pytest.approx(0.0, abs=1e-4)
    assert y2 == pytest.approx(268.0, abs=1e-4)      # -1.0 + 269.0 = 268.0
    assert x1 == pytest.approx(1167.0, abs=1e-4)
    assert x2 == pytest.approx(1413.0, abs=1e-4)     # 1167 + 246 = 1413


def test_no_shift_vs_naive_clamping():
    """
    Proves that naive clamping (y_new = max(0, y) with unchanged height)
    causes an erroneous 1-pixel downward shift that extends beyond the true object bottom.
    """
    W, H = 1000, 1000
    y_raw = -1.0
    h_raw = 50.0
    true_bottom = y_raw + h_raw  # 49.0

    # Naive approach:
    naive_y1 = max(0.0, y_raw)   # 0.0
    naive_h = h_raw              # 50.0 (unchanged)
    naive_bottom = naive_y1 + naive_h  # 50.0 -> 1 PIXEL DRIFT!
    assert naive_bottom != true_bottom
    assert naive_bottom - true_bottom == 1.0  # Erroneous downward shift of 1 pixel

    # Correct geometry clipping in coco_to_yolo:
    yolo = coco_to_yolo([100.0, y_raw, 80.0, h_raw], W, H)
    _, _, _, y2_reconstructed = yolo_to_pixel_xyxy(yolo, W, H)
    assert y2_reconstructed == pytest.approx(true_bottom, abs=1e-4)


def test_boundary_clipping_all_edges():
    """Verify that clipping correctly handles right and bottom overflow."""
    W, H = 1000, 800
    # Box extending 10px past right edge and 15px past bottom edge
    coco_box = [900.0, 700.0, 110.0, 115.0]  # x2 = 1010, y2 = 815
    yolo = coco_to_yolo(coco_box, W, H)
    assert yolo is not None

    x1, y1, x2, y2 = yolo_to_pixel_xyxy(yolo, W, H)
    assert x1 == pytest.approx(900.0, abs=1e-4)
    assert y1 == pytest.approx(700.0, abs=1e-4)
    assert x2 == pytest.approx(1000.0, abs=1e-4)  # truncated to W
    assert y2 == pytest.approx(800.0, abs=1e-4)   # truncated to H


def test_degenerate_offscreen_boxes_rejected():
    """Verify that boxes completely outside the canvas return None."""
    W, H = 640, 480
    # Completely above image
    assert coco_to_yolo([100.0, -100.0, 50.0, 50.0], W, H) is None
    # Completely left of image
    assert coco_to_yolo([-200.0, 100.0, 50.0, 50.0], W, H) is None
    # Microscopic 1-pixel overlap rejected by degeneracy check (x2 - x1 <= 1)
    assert coco_to_yolo([-49.0, 100.0, 50.0, 50.0], W, H) is None  # overlap is 1 pixel
