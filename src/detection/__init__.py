# Mô tả: Khởi tạo gói src.detection và cung cấp lối tắt import cho các kiểu dữ liệu và bộ phát hiện cốt lõi.
"""Multi-object waste detection interfaces and YOLO adapters."""

from .types import BoundingBox, Detection, DetectionResult
from .yolo import DetectionError, WasteDetector

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionError",
    "DetectionResult",
    "WasteDetector",
]
