"""Real-time multi-object detection with short box stabilization."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import av
import cv2
import numpy as np
from PIL import Image

from src.detection.tracking import TemporalDetectionSmoother
from src.detection.types import DetectionResult
from src.detection.visualization import draw_detections
from src.web.detection_service import DetectionService


class LiveDetectionCallback:
    def __init__(
        self,
        service: DetectionService,
        *,
        inference_fps: float,
        tracker_iou_threshold: float,
        tracker_max_missed: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < inference_fps <= 30.0:
            raise ValueError("inference_fps must be greater than 0 and at most 30")
        self._service = service
        self._clock = clock
        self._interval = 1.0 / inference_fps
        self._lock = threading.RLock()
        self._next_inference_at: float | None = None
        self._latest: DetectionResult | None = None
        self._last_error: str | None = None
        self._smoother = TemporalDetectionSmoother(
            iou_threshold=tracker_iou_threshold,
            max_missed=tracker_max_missed,
        )

    @property
    def latest_result(self) -> DetectionResult | None:
        with self._lock:
            return self._latest

    def reset(self) -> None:
        with self._lock:
            self._next_inference_at = None
            self._latest = None
            self._last_error = None
            self._smoother.reset()

    def _claim_slot(self, now: float) -> bool:
        with self._lock:
            if self._next_inference_at is not None and now < self._next_inference_at:
                return False
            self._next_inference_at = now + self._interval
            return True

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._claim_slot(self._clock()):
            try:
                detected = self._service.detect(Image.fromarray(rgb))
                with self._lock:
                    self._latest = self._smoother.update(detected)
                    self._last_error = None
            except Exception as error:  # noqa: BLE001 - callback must return a frame
                with self._lock:
                    self._last_error = " ".join(str(error).split())[:96]

        with self._lock:
            latest = self._latest
            error_text = self._last_error
        if latest is not None and (
            latest.image_width == rgb.shape[1] and latest.image_height == rgb.shape[0]
        ):
            rgb = np.asarray(draw_detections(Image.fromarray(rgb), latest))
        annotated_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if error_text:
            cv2.putText(
                annotated_bgr,
                f"ERROR: {error_text}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
