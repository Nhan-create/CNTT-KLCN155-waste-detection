"""Light temporal stabilization for real-time detection boxes."""

from __future__ import annotations

from dataclasses import dataclass

from src.detection.types import BoundingBox, Detection, DetectionResult


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area + second.area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _blend_box(
    previous: BoundingBox, current: BoundingBox, alpha: float
) -> BoundingBox:
    return BoundingBox(
        previous.x1 * (1.0 - alpha) + current.x1 * alpha,
        previous.y1 * (1.0 - alpha) + current.y1 * alpha,
        previous.x2 * (1.0 - alpha) + current.x2 * alpha,
        previous.y2 * (1.0 - alpha) + current.y2 * alpha,
    )


@dataclass
class _Track:
    detection: Detection
    missed: int = 0


class TemporalDetectionSmoother:
    """Associate same-class boxes by IoU and smooth short camera jitter."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.30,
        smoothing_alpha: float = 0.65,
        max_missed: int = 1,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be greater than 0 and at most 1")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        self.iou_threshold = iou_threshold
        self.smoothing_alpha = smoothing_alpha
        self.max_missed = max_missed
        self._tracks: list[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, result: DetectionResult) -> DetectionResult:
        unmatched_tracks = set(range(len(self._tracks)))
        updated: list[_Track] = []
        for incoming in result.detections:
            candidates = [
                (box_iou(self._tracks[index].detection.box, incoming.box), index)
                for index in unmatched_tracks
                if self._tracks[index].detection.class_id == incoming.class_id
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_index >= 0 and best_iou >= self.iou_threshold:
                previous = self._tracks[best_index].detection
                unmatched_tracks.remove(best_index)
                alpha = self.smoothing_alpha
                detection = Detection(
                    class_index=incoming.class_index,
                    class_id=incoming.class_id,
                    confidence=previous.confidence * (1.0 - alpha)
                    + incoming.confidence * alpha,
                    box=_blend_box(previous.box, incoming.box, alpha),
                )
                updated.append(_Track(detection))
            else:
                updated.append(_Track(incoming))

        incoming_detections = tuple(result.detections)
        for index in unmatched_tracks:
            stale = self._tracks[index]
            stale.missed += 1
            replaced_by_new_class = any(
                incoming.class_id != stale.detection.class_id
                and box_iou(incoming.box, stale.detection.box) >= self.iou_threshold
                for incoming in incoming_detections
            )
            if stale.missed <= self.max_missed and not replaced_by_new_class:
                updated.append(stale)
        self._tracks = updated
        detections = tuple(
            sorted(
                (track.detection for track in updated), key=lambda row: -row.confidence
            )
        )
        return DetectionResult(
            detections,
            result.image_width,
            result.image_height,
            result.inference_ms,
        )
