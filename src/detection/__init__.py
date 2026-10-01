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
