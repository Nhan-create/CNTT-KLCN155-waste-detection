"""Class-wise WBF using the reference implementation by Solovyev et al."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Protocol

from PIL import Image

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult


class Detector(Protocol):
    def detect_pil(self, image: Image.Image) -> DetectionResult: ...


def fuse_detections(
    results: Sequence[DetectionResult],
    *,
    weights: Sequence[float] = (1.0, 1.0),
    iou_threshold: float = 0.55,
    confidence_threshold: float = 0.001,
    output_threshold: float = 0.0,
    max_detections: int = 100,
) -> DetectionResult:
    """Normalize original-image boxes, fuse within each class, then restore pixels.

    Coordinates use model_weight * confidence. Reference WBF's ``avg`` score
    penalizes boxes supported by only one model. Input and output thresholds are
    separate so AP can retain low-score proposals while the demo filters outputs.
    """
    if not results or len(results) != len(weights):
        raise ValueError("Provide one weight per detector result")
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Fusion weights must be finite and positive")
    for value in (iou_threshold, confidence_threshold, output_threshold):
        if not 0 <= value <= 1:
            raise ValueError("Fusion thresholds must lie in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    width, height = results[0].image_width, results[0].image_height
    boxes_list, scores_list, labels_list = [], [], []
    for result in results:
        if (result.image_width, result.image_height) != (width, height):
            raise ValueError("Cannot fuse boxes from different image dimensions")
        boxes, scores, labels = [], [], []
        for detection in result.detections:
            index = detection.class_index
            if index >= len(DETECTION_CLASS_NAMES) or detection.class_id != DETECTION_CLASS_NAMES[index]:
                raise ValueError("Fusion requires the six-class detection taxonomy")
            box = detection.box
            if not (0 <= box.x1 < box.x2 <= width and 0 <= box.y1 < box.y2 <= height):
                raise ValueError("Fusion box is outside its original image")
            boxes.append([box.x1 / width, box.y1 / height, box.x2 / width, box.y2 / height])
            scores.append(detection.confidence)
            labels.append(index)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to use Weighted Boxes Fusion") from error
    started = time.perf_counter()
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=list(weights),
        iou_thr=iou_threshold, skip_box_thr=confidence_threshold,
        conf_type="avg", allows_overflow=False,
    )
    fused = tuple(
        Detection(
            int(label), DETECTION_CLASS_NAMES[int(label)], min(1.0, float(score)),
            BoundingBox(float(box[0]) * width, float(box[1]) * height,
                        float(box[2]) * width, float(box[3]) * height),
        )
        for box, score, label in zip(boxes, scores, labels, strict=True)
        if score >= output_threshold
    )[:max_detections]
    elapsed = (time.perf_counter() - started) * 1000
    if all(result.inference_ms is not None for result in results):
        elapsed += sum(result.inference_ms for result in results)
    return DetectionResult(fused, width, height, elapsed)


class FusionDetector:
    def __init__(self, detectors: Sequence[Detector], **fusion_options) -> None:
        self.detectors = tuple(detectors)
        self.fusion_options = fusion_options

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        started = time.perf_counter()
        result = fuse_detections(
            [detector.detect_pil(image) for detector in self.detectors], **self.fusion_options,
        )
        return DetectionResult(result.detections, result.image_width, result.image_height,
                               (time.perf_counter() - started) * 1000)
