import io
import zipfile

import pytest
from PIL import Image

from src.detection.types import BoundingBox, Detection, DetectionResult
from src.detection.visualization import DETECTION_CLASS_NAMES_VI
from src.app.detection_logic import (
    DetectionMediaError,
    analyze_image_file,
    archive_annotated_images,
    count_rows,
    detect_image_bytes,
    detection_rows,
)
from src.app.detection_service import DetectionService


class FakeDetector:
    def detect_pil(self, image: Image.Image) -> DetectionResult:
        assert image.mode == "RGB"
        return DetectionResult(
            detections=(
                Detection(4, "organic", 0.9, BoundingBox(5, 10, 60, 80)),
                Detection(5, "hazardous", 0.8, BoundingBox(70, 10, 130, 80)),
                Detection(4, "organic", 0.7, BoundingBox(5, 85, 60, 115)),
            ),
            image_width=image.width,
            image_height=image.height,
            inference_ms=12,
        )


class TenClassDetector:
    def detect_pil(self, image: Image.Image) -> DetectionResult:
        assert image.mode == "RGB"
        classes = (
            "battery", "biological", "cardboard", "clothes", "glass",
            "metal", "paper", "plastic", "shoes", "trash",
        )
        return DetectionResult(
            detections=tuple(
                Detection(idx, name, 0.85, BoundingBox(idx * 5, 5, idx * 5 + 10, 20))
                for idx, name in enumerate(classes)
            ),
            image_width=image.width,
            image_height=image.height,
            inference_ms=15,
        )


def test_annotated_output_counts_and_download_archive() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (160, 120), "white").save(buffer, format="PNG")
    output = analyze_image_file(
        "../../sample.png", buffer.getvalue(), DetectionService(FakeDetector())
    )

    assert output.processing_ms > 0
    assert count_rows(output.result) == [
        {"Loại rác": "Rác nguy hại", "Số lượng": 1},
        {"Loại rác": "Rác hữu cơ", "Số lượng": 2},
    ]
    assert detection_rows(output.result)[0]["Độ tin cậy"] == "90.0%"
    with Image.open(io.BytesIO(output.annotated_png)) as image:
        assert image.format == "PNG"
        assert image.size == (160, 120)
        assert image.getpixel((5, 60)) != (255, 255, 255)
    with zipfile.ZipFile(io.BytesIO(archive_annotated_images([output, output]))) as archive:
        assert archive.namelist() == ["001_sample_nhan_dien.png", "002_sample_nhan_dien.png"]
        assert archive.read(archive.namelist()[0]) == output.annotated_png


def test_corrupt_upload_is_rejected_before_prediction() -> None:
    with pytest.raises(DetectionMediaError, match="Không thể đọc ảnh"):
        detect_image_bytes(b"not an image", DetectionService(FakeDetector()))


def test_detect_image_bytes_handles_exif_orientation_tag() -> None:
    # 200 x 100 image with EXIF orientation 6 (90 CW rotation -> transposed to 100 x 200)
    raw = Image.new("RGB", (200, 100), "white")
    exif = raw.getexif()
    exif[0x0112] = 6

    buffer = io.BytesIO()
    raw.save(buffer, format="JPEG", exif=exif)

    service = DetectionService(FakeDetector())
    image, result = detect_image_bytes(buffer.getvalue(), service)

    # Transposed image should have width=100, height=200
    assert image.size == (100, 200)
    assert result.image_width == 100
    assert result.image_height == 200

    output = analyze_image_file("phone_photo.jpg", buffer.getvalue(), service)
    with Image.open(io.BytesIO(output.annotated_png)) as annotated:
        assert annotated.size == (100, 200)


def test_ten_classes_display_in_web_rows() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")
    service = DetectionService(TenClassDetector())

    output = analyze_image_file("multi_trash.png", buffer.getvalue(), service)
    rows = detection_rows(output.result)
    assert len(rows) == 10

    labels = [row["Loại rác"] for row in rows]
    expected_labels = list(DETECTION_CLASS_NAMES_VI)
    assert labels == expected_labels
