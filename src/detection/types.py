# Mô tả: Định nghĩa hợp đồng dữ liệu đầu ra độc lập framework cho toàn bộ các mô hình phát hiện (Tầng 3A, 3B, Tầng 4, Tầng 5).
"""Framework-independent detection result types."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundingBox:
    """One pixel-space ``xyxy`` box in the original image."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("bounding box coordinates must be finite")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def is_valid(self, image_width: int | None = None, image_height: int | None = None) -> bool:
        """Check if box has strictly positive dimensions and fits within bounds."""
        if not (self.x2 > self.x1 and self.y2 > self.y1):
            return False
        if image_width is not None and not (0.0 <= self.x1 < self.x2 <= float(image_width)):
            return False
        if image_height is not None and not (0.0 <= self.y1 < self.y2 <= float(image_height)):
            return False
        return True

    def clip(self, image_width: int, image_height: int) -> BoundingBox | None:
        """Clip box strictly to [0, W] and [0, H]. Return None if box is empty, inverted, or outside."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            return None
        x1 = max(0.0, min(self.x1, float(image_width)))
        y1 = max(0.0, min(self.y1, float(image_height)))
        x2 = max(0.0, min(self.x2, float(image_width)))
        y2 = max(0.0, min(self.y2, float(image_height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)

    def clamp(self, image_width: int, image_height: int) -> BoundingBox:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be positive")
        x1 = min(max(self.x1, 0.0), float(image_width - 1))
        y1 = min(max(self.y1, 0.0), float(image_height - 1))
        x2 = min(max(self.x2, x1 + 1.0), float(image_width))
        y2 = min(max(self.y2, y1 + 1.0), float(image_height))
        return BoundingBox(x1, y1, x2, y2)


@dataclass(frozen=True)
class Detection:
    """A class, confidence, and bounding box for one visible waste object."""

    class_index: int
    class_id: str
    confidence: float
    box: BoundingBox

    def __post_init__(self) -> None:
        if self.class_index < 0:
            raise ValueError("class_index must be non-negative")
        if not self.class_id:
            raise ValueError("class_id must not be blank")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.box.width <= 0.0 or self.box.height <= 0.0:
            raise ValueError("bounding box must have positive width and height")


@dataclass(frozen=True)
class DetectionResult:
    """All detections for one image, in descending confidence order."""

    detections: tuple[Detection, ...]
    image_width: int
    image_height: int
    inference_ms: float | None = None

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.inference_ms is not None and (not math.isfinite(self.inference_ms) or self.inference_ms < 0.0):
            raise ValueError("inference_ms must be non-negative")
