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
