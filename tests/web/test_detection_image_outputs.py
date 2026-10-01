import io
import zipfile

import pytest
from PIL import Image

from src.detection.types import BoundingBox, Detection, DetectionResult
from src.web.detection_logic import (
    DetectionMediaError,
    analyze_image_file,
    archive_annotated_images,
    count_rows,
    detect_image_bytes,
    detection_rows,
)
from src.web.detection_service import DetectionService


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
