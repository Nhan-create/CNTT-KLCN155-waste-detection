"""Sample uploaded video frames and detect every visible waste object."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO

import av
from PIL import Image

from src.detection.tracking import TemporalDetectionSmoother
from src.detection.types import DetectionResult
from src.web.detection_service import DetectionService
from src.web.video import (
    FrameSampler,
    SamplingPolicy,
    VideoAnalysisError,
    _frame_timestamp,
)


@dataclass(frozen=True)
class DetectionVideoSample:
    timestamp_seconds: float
    result: DetectionResult


@dataclass(frozen=True)
class DetectionVideoAnalysis:
    samples: tuple[DetectionVideoSample, ...]
    truncated: bool
    detection_counts: dict[str, int]


DetectionSampleCallback = Callable[[float, Image.Image, DetectionResult], None]


def analyze_detection_video(
    source: BinaryIO,
    service: DetectionService,
    policy: SamplingPolicy,
    *,
    tracker_iou_threshold: float,
    tracker_max_missed: int,
    on_sample: DetectionSampleCallback | None = None,
    open_container: Callable[[BinaryIO], Any] = av.open,
) -> DetectionVideoAnalysis:
    container: Any | None = None
    try:
        container = open_container(source)
        streams = list(container.streams.video)
        if not streams:
            raise VideoAnalysisError("Video không chứa luồng hình ảnh có thể đọc.")
        stream = streams[0]
        sampler = FrameSampler(policy)
        smoother = TemporalDetectionSmoother(
            iou_threshold=tracker_iou_threshold,
            max_missed=tracker_max_missed,
        )
        samples: list[DetectionVideoSample] = []
        counts: Counter[str] = Counter()
        for frame_index, frame in enumerate(container.decode(video=0)):
            timestamp = _frame_timestamp(
                frame, frame_index, stream, policy.fallback_fps
            )
            if not sampler.accept(timestamp):
                if sampler.limit_reached:
                    break
                continue
            image = frame.to_image().convert("RGB")
            result = smoother.update(service.detect(image))
            samples.append(DetectionVideoSample(timestamp, result))
            counts.update(row.class_id for row in result.detections)
            if on_sample is not None:
                on_sample(timestamp, image, result)
            if sampler.limit_reached:
                break
        if not samples:
            raise VideoAnalysisError("Không tìm thấy khung hình hợp lệ trong video.")
        return DetectionVideoAnalysis(
            tuple(samples), sampler.limit_reached, dict(counts)
        )
    except VideoAnalysisError:
        raise
    except Exception as error:
        raise VideoAnalysisError(
            f"Không thể đọc hoặc phân tích video: {error}"
        ) from error
    finally:
        if container is not None:
            container.close()
