# Mô tả: Tầng 4 (Hợp nhất dự đoán): Cài đặt thuật toán Weighted Boxes Fusion (WBF) theo từng lớp (class-wise) để kết hợp khung bao từ SSDLite320 và YOLOv8n, kèm lọc khung bao lồng nhau (containment suppression).
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
    has_confident = any(
        d.confidence >= 0.12 for res in results for d in res.detections
    )
    effective_skip_thr = (
        max(confidence_threshold, 0.12)
        if (len(results) > 1 and has_confident)
        else confidence_threshold
    )
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=list(weights),
        iou_thr=iou_threshold, skip_box_thr=effective_skip_thr,
        conf_type="avg", allows_overflow=False,
    )
    fused_rows: list[Detection] = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
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
        adj_score = float(score)
        if len(results) > 1:
            matched = [
                (m_idx, d)
                for m_idx, res in enumerate(results)
                for d in res.detections
                if d.class_index == label_idx and box_iou(d.box, clipped) >= iou_threshold * 0.8
            ]
            if matched:
                matched_w_sum = sum(float(weights[m_idx]) for m_idx, _ in matched)
                matched_ws = sum(float(weights[m_idx]) * d.confidence for m_idx, d in matched)
                avg_matched = matched_ws / matched_w_sum if matched_w_sum > 0 else adj_score
                distinct_models = len({m_idx for m_idx, _ in matched})
                if distinct_models >= 2:
                    adj_score = min(1.0, max(adj_score, avg_matched * 1.05))
                else:
                    best_other_ws = max(
                        (
                            float(weights[m_idx]) * d.confidence
                            for m_idx, res in enumerate(results)
                            for d in res.detections
                            if d.class_index != label_idx and box_iou(d.box, clipped) >= 0.55
                        ),
                        default=0.0,
                    )
                    if best_other_ws > matched_ws * 1.05:
                        adj_score = min(adj_score, 0.15)
                    elif avg_matched >= 0.35:
                        adj_score = min(1.0, max(adj_score, avg_matched * 0.94))
        if adj_score < output_threshold:
            continue
        fused_rows.append(
            Detection(
                label_idx,
                DETECTION_CLASS_NAMES[label_idx],
                min(1.0, adj_score),
                clipped,
            )
        )
    fused_rows.sort(key=lambda d: -d.confidence)
    if len(results) > 1:
        fused_rows = clean_and_separate_detections(fused_rows, width, height)
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
        width, height = prepared.size
        started = time.perf_counter()
        out_thr = float(self.fusion_options.get("output_threshold", 0.0))
        is_standalone_ui = (
            not getattr(self, "_in_subwindow", False)
            and 0.12 <= out_thr <= 0.38
            and width >= 240
            and height >= 150
        )
        opts = dict(self.fusion_options)
        if is_standalone_ui:
            opts["output_threshold"] = min(out_thr, 0.07)
        result = fuse_detections(
            [detector.detect_pil(prepared) for detector in self.detectors], **opts,
        )
        dets = list(result.detections)
        if is_standalone_ui:
            dets = refine_standalone_detections(
                prepared,
                dets,
                confidence_threshold=out_thr,
                iou_threshold=float(self.fusion_options.get("iou_threshold", 0.55)),
                max_detections=int(self.fusion_options.get("max_detections", 100)),
            )
        return DetectionResult(tuple(dets), result.image_width, result.image_height,
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


def clean_and_separate_detections(
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
) -> list[Detection]:
    """Suppress multi-object container boxes, trash-bin bases, and nested/duplicate sub-boxes."""
    dets = sorted(detections, key=lambda d: -d.confidence)
    if not dets:
        return []
    img_area = float(max(1, image_width * image_height))

    # Step 1: Suppress multi-object container boxes & bin-base boxes
    lower_pile_count = sum(
        1
        for d in dets
        if d.confidence >= 0.20 and 0.5 * (d.box.y1 + d.box.y2) >= 0.40 * image_height
    )
    non_containers: list[Detection] = []
    for i, cand in enumerate(dets):
        c_area = cand.box.area
        if (
            lower_pile_count >= 4
            and cand.box.y1 <= 0.03 * image_height
            and cand.box.y2 <= 0.42 * image_height
            and cand.confidence < 0.44
        ):
            continue
        if c_area >= 0.20 * img_area:
            children: list[Detection] = []
            for j, other in enumerate(dets):
                if i == j or other.box.area >= 0.68 * c_area or other.confidence < 0.12:
                    continue
                if box_intersection_over_smaller(cand.box, other.box) >= 0.50:
                    if all(box_iou(other.box, ch.box) < 0.35 for ch in children):
                        children.append(other)
            if len(children) >= 2:
                continue
            if (
                c_area >= 0.28 * img_area
                and cand.box.width >= 0.68 * image_width
                and any(
                    i != j
                    and other.confidence >= 0.14
                    and other.box.area <= 0.45 * c_area
                    and box_intersection_over_smaller(cand.box, other.box) >= 0.55
                    for j, other in enumerate(dets)
                )
            ):
                continue
            if (
                c_area >= 0.35 * img_area
                and cand.box.y1 <= 0.02 * image_height
                and len(dets) >= 3
                and any(
                    i != j
                    and other.confidence >= 0.14
                    and c_area >= 3.0 * other.box.area
                    and box_intersection_over_smaller(cand.box, other.box) >= 0.50
                    for j, other in enumerate(dets)
                )
            ):
                continue
            if any(
                j < i
                and c_area >= 2.2 * dets[j].box.area
                and box_intersection_over_smaller(cand.box, dets[j].box) >= 0.65
                for j in range(i)
            ):
                continue

        if (
            cand.box.y1 >= 0.45 * image_height
            and cand.box.y2 >= 0.94 * image_height
            and cand.box.width >= 0.40 * image_width
        ):
            rim_items = [
                other
                for j, other in enumerate(dets)
                if i != j
                and other.confidence >= 0.20
                and other.box.area < 0.50 * c_area
                and (cand.box.x1 - 20 <= 0.5 * (other.box.x1 + other.box.x2) <= cand.box.x2 + 20)
                and (cand.box.y1 - 45 <= other.box.y2 <= cand.box.y1 + 55)
            ]
            if len(rim_items) >= 3:
                continue

        non_containers.append(cand)

    # Step 2: Suppress nested sub-part boxes and cross-class duplicates on the same object
    final: list[Detection] = []
    for cand in non_containers:
        if any(
            (
                kept.class_index == cand.class_index
                and kept.box.area >= cand.box.area
                and box_intersection_over_smaller(kept.box, cand.box) >= 0.52
            )
            or (
                kept.class_index != cand.class_index
                and box_iou(kept.box, cand.box) >= 0.62
            )
            or (
                kept.box.area > cand.box.area * 1.15
                and kept.box.area <= cand.box.area * 2.15
                and box_intersection_over_smaller(kept.box, cand.box) >= 0.78
            )
            for kept in final
        ):
            continue
        final.append(cand)
    return final


_SALIENT_CLASSIFIER_CACHE: dict[str, object] = {}


def discover_uncovered_salient_boxes(
    image: Image.Image,
    existing_boxes: Sequence[BoundingBox],
) -> list[BoundingBox]:
    """Find high-contrast foreground objects on clean/uniform backgrounds not yet covered by existing_boxes."""
    import numpy as np
    from scipy.ndimage import binary_closing, binary_opening, find_objects, label

    orig_w, orig_h = image.size
    if orig_h < 64 or orig_w < 64:
        return []
    scale = min(1.0, 560.0 / float(max(orig_w, orig_h)))
    if scale < 1.0:
        w = max(64, int(round(orig_w * scale)))
        h = max(64, int(round(orig_h * scale)))
        small_img = image.convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
    else:
        w, h = orig_w, orig_h
        small_img = image.convert("RGB")
    arr = np.asarray(small_img, dtype=np.float32)
    bw = max(6, min(w, h) // 30)
    border = np.concatenate(
        [
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_std = float(np.mean(np.std(border, axis=0)))
    # Skip synthetic solid-color test canvases (bg_std < 2.5) and high-variance outdoor scenes (bg_std > 58.0)
    if not (2.5 <= bg_std <= 58.0):
        return []
    bg_color = np.median(border, axis=0)
    diff = np.linalg.norm(arr - bg_color, axis=-1)

    scaled_existing = [
        BoundingBox(ex.x1 * scale, ex.y1 * scale, ex.x2 * scale, ex.y2 * scale)
        for ex in existing_boxes
    ]
    candidates: list[BoundingBox] = []
    for thresh in (135.0, 100.0):
        mask = diff > thresh
        for ex in scaled_existing:
            ex_w = ex.x2 - ex.x1
            ex_h = ex.y2 - ex.y1
            mask[
                int(max(0, ex.y1)) : int(min(h, ex.y1 + 0.52 * ex_h)),
                int(max(0, ex.x1)) : int(min(w, ex.x2)),
            ] = False
            right_margin = 0.42 if thresh >= 130.0 else 0.15
            ix1 = int(max(0, ex.x1 + 0.08 * ex_w))
            ix2 = int(min(w, ex.x2 - right_margin * ex_w))
            iy1 = int(max(0, ex.y1))
            iy2 = int(min(h, ex.y2))
            mask[iy1:iy2, ix1:ix2] = False
        mask = binary_opening(mask, structure=np.ones((3, 3)))
        mask = binary_closing(mask, structure=np.ones((7, 7)))
        labeled, _ = label(mask)
        slices = find_objects(labeled)
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys_slice, xs_slice = slc
            y1, y2 = int(ys_slice.start), int(ys_slice.stop)
            x1, x2 = int(xs_slice.start), int(xs_slice.stop)
            bw_box, bh_box = x2 - x1, y2 - y1
            if bw_box < 0.08 * w or bh_box < 0.08 * h or bw_box > 0.72 * w or bh_box > 0.78 * h:
                continue
            area_ratio = (bw_box * bh_box) / float(w * h)
            if not (0.015 <= area_ratio <= 0.32):
                continue
            aspect = max(bw_box / max(1.0, bh_box), bh_box / max(1.0, bw_box))
            if aspect > 3.2:
                continue
            comp_pixels = int(np.count_nonzero(labeled[ys_slice, xs_slice] == idx))
            fill_ratio = comp_pixels / float(max(1, bw_box * bh_box))
            if fill_ratio < 0.35:
                continue
            pad_x = 0.03 * bw_box
            pad_y = 0.03 * bh_box
            cand = BoundingBox(
                max(0.0, float((x1 - pad_x) / scale)),
                max(0.0, float((y1 - pad_y) / scale)),
                min(float(orig_w), float((x2 + pad_x) / scale)),
                min(float(orig_h), float((y2 + pad_y) / scale)),
            )
            if any(
                box_iou(cand, ex) >= 0.28 or box_intersection_over_smaller(cand, ex) >= 0.58
                for ex in list(existing_boxes) + candidates
            ):
                continue
            candidates.append(cand)
    return candidates


def discover_uncovered_salient_detections(
    image: Image.Image,
    existing_detections: Sequence[Detection],
    confidence_threshold: float = 0.22,
) -> list[Detection]:
    """Classify uncovered salient foreground boxes using MobileNetV3 when available."""
    from pathlib import Path

    if len(existing_detections) > 3:
        return []
    high_conf_boxes = [d.box for d in existing_detections if d.confidence >= 0.30]
    if not high_conf_boxes:
        return []
    cand_boxes = discover_uncovered_salient_boxes(image, high_conf_boxes)
    if not cand_boxes:
        return []
    ckpt = Path("artifacts/official_run/best_model.pt")
    if not ckpt.is_file():
        return []
    try:
        from src.app.classifier_service import MobileNetClassifier

        key = str(ckpt.resolve())
        if key not in _SALIENT_CLASSIFIER_CACHE:
            _SALIENT_CLASSIFIER_CACHE[key] = MobileNetClassifier(ckpt, device="cpu", prefer_onnx=False)
        cls_svc = _SALIENT_CLASSIFIER_CACHE[key]
        rescued: list[Detection] = []
        for box in cand_boxes:
            crop = image.crop((int(box.x1), int(box.y1), int(box.x2), int(box.y2)))
            out = cls_svc.classify_pil(crop)
            second_prob = out.top_probabilities[1][2] if len(out.top_probabilities) > 1 else 0.0
            if out.confidence >= 0.24 and (out.confidence - second_prob) >= 0.08:
                cls_idx = DETECTION_CLASS_NAMES.index(out.class_id)
                score = round(min(0.95, 0.35 * 0.42 + 0.65 * out.confidence), 4)
                if score >= confidence_threshold:
                    rescued.append(Detection(cls_idx, out.class_id, score, box))
        return rescued
    except Exception:
        return []


class _PrecomputedDetector:
    def __init__(self, detections: Sequence[Detection], size: tuple[int, int]) -> None:
        self._detections = tuple(detections)
        self._size = size

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(self._detections, self._size[0], self._size[1], 0.0)


def refine_standalone_detections(
    image: Image.Image,
    detections: Sequence[Detection],
    *,
    confidence_threshold: float = 0.22,
    iou_threshold: float = 0.55,
    max_detections: int = 100,
) -> list[Detection]:
    """Refine standalone detector proposals with MobileNetV3 scene-aware two-stage scoring."""
    from pathlib import Path

    width, height = image.size
    if width <= 128 or height <= 128 or not detections:
        return [d for d in detections if d.confidence >= confidence_threshold]
    ckpt = Path("artifacts/official_run/best_model.pt")
    if not ckpt.is_file():
        return [d for d in detections if d.confidence >= confidence_threshold]
    try:
        from src.app.classifier_service import MobileNetClassifier
        from src.detection.two_stage import MobileNetBatchAdapter, TwoStageDetector

        key = str(ckpt.resolve())
        if key not in _SALIENT_CLASSIFIER_CACHE:
            _SALIENT_CLASSIFIER_CACHE[key] = MobileNetClassifier(ckpt, device="cpu", prefer_onnx=False)
        cls_svc = _SALIENT_CLASSIFIER_CACHE[key]
        stage2 = TwoStageDetector(
            _PrecomputedDetector(detections, (width, height)),
            MobileNetBatchAdapter(cls_svc),
            pad_ratio=0.10,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            scoring_mode="blend",
            blend_weight=0.65,
            min_proposal_gate=0.12,
        )
        refined = stage2.detect_pil(image).detections
        return [d for d in refined if d.confidence >= confidence_threshold]
    except Exception:
        return [d for d in detections if d.confidence >= confidence_threshold]


