# Mô tả: Kiểm thử vẽ khung bao phát hiện (Highlight Box / Bounding Box), nhãn tiếng Việt 10 lớp và xử lý hướng ảnh EXIF.
import io
import pytest
from PIL import Image

from src.detection.types import BoundingBox, Detection, DetectionResult
from src.detection.visualization import (
    DETECTION_CLASS_NAMES_VI,
    detection_label,
    draw_detections,
)


def test_draw_detections_returns_copy_with_box_pixels_changed() -> None:
    image = Image.new("RGB", (160, 120), "white")
    result = DetectionResult(
        (
            Detection(
                7,
                "plastic",
                0.92,
                BoundingBox(20, 30, 120, 100),
            ),
        ),
        160,
        120,
    )

    annotated = draw_detections(image, result)

    assert image.getpixel((20, 30)) == (255, 255, 255)
    assert annotated.getpixel((20, 30)) != image.getpixel((20, 30))
    assert annotated.size == image.size


def test_ten_vietnamese_labels_exact_match() -> None:
    expected = {
        0: ("battery", "Pin"),
        1: ("biological", "Rác hữu cơ"),
        2: ("cardboard", "Bìa carton"),
        3: ("clothes", "Quần áo"),
        4: ("glass", "Thủy tinh"),
        5: ("metal", "Kim loại"),
        6: ("paper", "Giấy"),
        7: ("plastic", "Nhựa"),
        8: ("shoes", "Giày dép"),
        9: ("trash", "Rác khác"),
    }

    assert len(DETECTION_CLASS_NAMES_VI) == 10
    for index, (name, vi_label) in expected.items():
        assert DETECTION_CLASS_NAMES_VI[index] == vi_label
        assert detection_label(name) == vi_label
        assert detection_label(index) == vi_label


def test_draw_detections_transposes_exif_orientation() -> None:
    # Simulate a phone camera image saved with EXIF Orientation tag 6 (Rotate 90 CW)
    raw = Image.new("RGB", (200, 100), "white")
    exif = raw.getexif()
    exif[0x0112] = 6  # Orientation tag = 6

    buffer = io.BytesIO()
    raw.save(buffer, format="JPEG", exif=exif)
    buffer.seek(0)
    image_with_exif = Image.open(buffer)

    # Detections are in the transposed coordinate space (100 x 200)
    result = DetectionResult(
        (
            Detection(0, "battery", 0.95, BoundingBox(10, 10, 80, 150)),
        ),
        image_width=100,
        image_height=200,
    )

    annotated = draw_detections(image_with_exif, result)
    # The canvas must be transposed to match the upright dimensions
    assert annotated.size == (100, 200)
    # The drawn box outline must be present on the annotated canvas
    assert annotated.getpixel((10, 10)) != (255, 255, 255)


def test_draw_detections_crisp_rectangle_outline_and_highlight() -> None:
    image = Image.new("RGB", (100, 100), "white")
    result = DetectionResult(
        (
            Detection(1, "biological", 0.88, BoundingBox(10, 10, 80, 80)),
        ),
        100,
        100,
    )

    # Test "box" style: clear bounding rectangle outline
    box_annotated = draw_detections(image, result, style="box")
    assert box_annotated.size == (100, 100)
    # Inside center of box should remain white in "box" style (not obscured by contour)
    assert box_annotated.getpixel((45, 45)) == (255, 255, 255)
    # Border of box must be drawn with color
    assert box_annotated.getpixel((10, 10)) != (255, 255, 255)

    # Test "highlight" style: subtle tint, crisp outline retained
    hl_annotated = draw_detections(image, result, style="highlight")
    assert hl_annotated.size == (100, 100)
    assert hl_annotated.getpixel((10, 10)) != (255, 255, 255)


def test_draw_detections_handles_boundary_boxes() -> None:
    image = Image.new("RGB", (100, 100), "white")
    # Box touching image boundaries: (0, 0, 100, 100)
    result = DetectionResult(
        (
            Detection(2, "cardboard", 0.75, BoundingBox(0, 0, 100, 100)),
        ),
        100,
        100,
    )
    annotated = draw_detections(image, result)
    assert annotated.size == (100, 100)
