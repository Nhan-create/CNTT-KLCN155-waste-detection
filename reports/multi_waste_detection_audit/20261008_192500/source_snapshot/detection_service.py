"""Serialized access to a shared detector model."""

from __future__ import annotations

import threading
from typing import Protocol

from PIL import Image

from src.detection.types import DetectionResult


class Detector(Protocol):
    def detect_pil(self, image: Image.Image) -> DetectionResult: ...


class DetectionService:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self._lock = threading.RLock()

    def detect(self, image: Image.Image) -> DetectionResult:
        with self._lock:
            return self.detector.detect_pil(image)
