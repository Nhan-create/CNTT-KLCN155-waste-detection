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
                raise ValueError(f"Fusion requires the {len(DETECTION_CLASS_NAMES)}-class detection taxonomy")
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
    fused_rows: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        if score < output_threshold:
            continue
        raw_box = BoundingBox(
            float(box[0]) * width,
            float(box[1]) * height,
            float(box[2]) * width,
            float(box[3]) * height,
        )
        clipped = raw_box.clip(width, height)
        if clipped is None:
            continue
        label_idx = int(label)
        fused_rows.append(
            Detection(
                label_idx,
                DETECTION_CLASS_NAMES[label_idx],
                min(1.0, float(score)),
                clipped,
            )
        )
    fused = tuple(fused_rows[:max_detections])
    elapsed = (time.perf_counter() - started) * 1000
    if all(result.inference_ms is not None for result in results):
        elapsed += sum(result.inference_ms for result in results)
    return DetectionResult(fused, width, height, elapsed)


class FusionDetector:
    def __init__(self, detectors: Sequence[Detector], **fusion_options) -> None:
        self.detectors = tuple(detectors)
        self.fusion_options = fusion_options

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        from PIL import ImageOps

        prepared = ImageOps.exif_transpose(image).convert("RGB")
        started = time.perf_counter()
        result = fuse_detections(
            [detector.detect_pil(prepared) for detector in self.detectors], **self.fusion_options,
        )
        return DetectionResult(result.detections, result.image_width, result.image_height,
                               (time.perf_counter() - started) * 1000)


def box_iou(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def box_intersection_over_smaller(a: BoundingBox, b: BoundingBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 0 else 0.0


def class_wise_nms(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    max_detections: int | None = None,
) -> list[Detection]:
    """Greedy NMS applied independently per ``class_index``; output sorted by confidence."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must lie in [0, 1]")
    ordered = sorted(detections, key=lambda d: (-d.confidence, d.class_index))
    kept: list[Detection] = []
    for candidate in ordered:
        if all(
            other.class_index != candidate.class_index
            or box_iou(other.box, candidate.box) <= iou_threshold
            for other in kept
        ):
            kept.append(candidate)
            if max_detections is not None and len(kept) >= max_detections:
                break
    return kept

