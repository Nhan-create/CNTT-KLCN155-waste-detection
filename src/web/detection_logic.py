"""Decode browser images once and run multi-object detection."""

from __future__ import annotations

import io
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError

from src.detection.types import DetectionResult
from src.detection.visualization import detection_label, draw_detections
from src.web.detection_service import DetectionService


class DetectionMediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageDetectionOutput:
    original_name: str
    result: DetectionResult
    annotated_png: bytes
    processing_ms: float


def detection_rows(result: DetectionResult) -> list[dict[str, object]]:
    return [
        {
            "Loại rác": detection_label(row.class_id),
            "Độ tin cậy": f"{row.confidence:.1%}",
            "Vị trí (x1, y1, x2, y2)": (
                f"({row.box.x1:.0f}, {row.box.y1:.0f}, "
                f"{row.box.x2:.0f}, {row.box.y2:.0f})"
            ),
        }
        for row in result.detections
    ]


def count_rows(result: DetectionResult) -> list[dict[str, object]]:
    counts = Counter(detection.class_id for detection in result.detections)
    return [
        {"Loại rác": detection_label(class_id), "Số lượng": count}
        for class_id, count in sorted(counts.items())
    ]


def annotated_filename(original_name: str) -> str:
    # Uploaded filenames are untrusted; archives must never contain directories.
    stem = PurePosixPath(original_name.replace("\\", "/")).stem
    return f"{stem or 'anh'}_nhan_dien.png"


def analyze_image_file(
    filename: str, payload: bytes, service: DetectionService
) -> ImageDetectionOutput:
    started = time.perf_counter()
    image, result = detect_image_bytes(payload, service)
    annotated = draw_detections(image, result)
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    return ImageDetectionOutput(
        original_name=filename,
        result=result,
        annotated_png=buffer.getvalue(),
        processing_ms=(time.perf_counter() - started) * 1000,
    )


def archive_annotated_images(outputs: list[ImageDetectionOutput]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, output in enumerate(outputs, start=1):
            archive.writestr(
                f"{index:03d}_{annotated_filename(output.original_name)}",
                output.annotated_png,
            )
    return buffer.getvalue()


def detect_image_bytes(
    payload: bytes, service: DetectionService
) -> tuple[Image.Image, DetectionResult]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DetectionMediaError(
            "Không thể đọc ảnh. Hãy dùng JPEG, PNG, WebP hoặc BMP hợp lệ."
        ) from error
    return image, service.detect(image)
