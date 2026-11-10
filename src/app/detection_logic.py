# Mô tả: Lắp ráp chuỗi pipeline phát hiện hoàn chỉnh từ DetectionWebSettings (Base Detector ➔ TiledDetector ➔ TwoStageDetector ➔ PerClassThresholdDetector), giải mã ảnh tải lên từ trình duyệt (xử lý EXIF), vẽ khung bao và tạo bảng tổng hợp số lượng rác theo lớp.
"""Decode browser images once and run multi-object detection."""

from __future__ import annotations

import io
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePosixPath

from PIL import Image, ImageOps, UnidentifiedImageError

from src.detection.types import DetectionResult
from src.detection.visualization import detection_label, draw_detections
from src.app.detection_service import DetectionService


class DetectionMediaError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageDetectionOutput:
    original_name: str
    result: DetectionResult
    annotated_png: bytes
    processing_ms: float
    # Two-stage extras (RefinedDetection rows aligned with result.detections), if any.
    details: tuple = ()


def detection_rows(result: DetectionResult, details: tuple = ()) -> list[dict[str, object]]:
    rows = [
        {
            "Loại rác": detection_label(row.class_id),
            "Độ tin cậy": f"{row.confidence:.1%}",
            "Vị trí (x1, y1, x2, y2)": (
                f"({row.box.x1:.0f}, {row.box.y1:.0f}, "
                f"{row.box.x2:.0f}, {row.box.y2:.0f})"
            ),
        }
        for row in result.detections
    ]
    if details and len(details) == len(rows):
        for row, item in zip(rows, details, strict=True):
            row["Điểm phát hiện"] = f"{item.detector_score:.1%}"
            row["Xác suất phân loại"] = f"{item.classifier_prob:.1%}"
            row["Top-3 phân loại"] = ", ".join(
                f"{detection_label(class_id)} {prob:.0%}" for class_id, prob in item.top_k
            )
    return rows


def class_threshold_keep_indices(
    result: DetectionResult, default_threshold: float, per_class: dict[str, float],
    disabled_classes: tuple[str, ...] = (),
) -> list[int]:
    return [
        index for index, row in enumerate(result.detections)
        if row.class_id not in disabled_classes
        and row.confidence >= per_class.get(row.class_id, default_threshold)
    ]


def apply_class_thresholds(
    result: DetectionResult, default_threshold: float, per_class: dict[str, float],
    disabled_classes: tuple[str, ...] = (),
) -> DetectionResult:
    """Keep detections whose score >= their class threshold (fallback: default)."""
    keep = class_threshold_keep_indices(result, default_threshold, per_class, disabled_classes)
    return DetectionResult(
        tuple(result.detections[i] for i in keep),
        result.image_width, result.image_height, result.inference_ms,
    )


class PerClassThresholdDetector:
    """Final operating filter applied after tiling / two-stage refinement."""

    def __init__(self, detector, default_threshold: float,
                 per_class_thresholds: tuple[tuple[str, float], ...] = (),
                 disabled_classes: tuple[str, ...] = ()) -> None:
        self.detector = detector
        self.default_threshold = default_threshold
        self.per_class = dict(per_class_thresholds)
        self.disabled_classes = disabled_classes
        self.last_details: tuple = ()

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        is_tiled_standalone = (
            not hasattr(self.detector, "last_details")
            and hasattr(self.detector, "tile_size")
            and 0.12 <= self.default_threshold <= 0.38
            and image.width >= 240
            and image.height >= 150
        )
        inner = getattr(self.detector, "detector", None)
        sub_dets = [inner, *getattr(inner, "detectors", ())] if inner is not None else []
        prev_confs = [getattr(d, "confidence_threshold", None) for d in sub_dets]
        fusion_opts = getattr(inner, "fusion_options", None) if inner is not None else None
        prev_out_thr = fusion_opts.get("output_threshold") if isinstance(fusion_opts, dict) else None
        if is_tiled_standalone:
            if isinstance(fusion_opts, dict) and isinstance(prev_out_thr, (int, float)):
                fusion_opts["output_threshold"] = min(float(prev_out_thr), 0.07)
            for d, c in zip(sub_dets, prev_confs, strict=True):
                if isinstance(c, (int, float)):
                    try:
                        setattr(d, "confidence_threshold", min(float(c), 0.08))
                    except Exception:
                        pass
        try:
            result = self.detector.detect_pil(image)
        finally:
            if is_tiled_standalone:
                if isinstance(fusion_opts, dict) and prev_out_thr is not None:
                    fusion_opts["output_threshold"] = prev_out_thr
                for d, c in zip(sub_dets, prev_confs, strict=True):
                    if isinstance(c, (int, float)):
                        try:
                            setattr(d, "confidence_threshold", float(c))
                        except Exception:
                            pass
        details = tuple(getattr(self.detector, "last_details", ()) or ())
        if not hasattr(self.detector, "last_details") and result.detections:
            from src.detection.fusion import (
                clean_and_separate_detections,
                discover_uncovered_salient_detections,
                refine_standalone_detections,
            )
            dets = list(result.detections)
            rescued = discover_uncovered_salient_detections(
                image, dets, confidence_threshold=self.default_threshold
            )
            if rescued:
                dets = clean_and_separate_detections(
                    [*dets, *rescued], result.image_width, result.image_height
                )
            if is_tiled_standalone:
                dets = refine_standalone_detections(
                    image,
                    dets,
                    confidence_threshold=self.default_threshold,
                    iou_threshold=getattr(self.detector, "iou_threshold", 0.55),
                    max_detections=getattr(self.detector, "max_detections", 100),
                )
            result = DetectionResult(
                tuple(dets), result.image_width, result.image_height, result.inference_ms
            )
        keep = class_threshold_keep_indices(
            result, self.default_threshold, self.per_class, self.disabled_classes
        )
        self.last_details = (
            tuple(details[i] for i in keep) if len(details) == len(result.detections) else ()
        )
        return DetectionResult(
            tuple(result.detections[i] for i in keep),
            result.image_width, result.image_height, result.inference_ms,
        )


def build_detection_pipeline(base_detector, settings, classifier=None):
    """Wrap a raw detector with optional tiling / two-stage and the final per-class filter.

    With both flags off and no class filters the base detector is returned
    unchanged, preserving the pre-existing behaviour. The base detector should be
    created with ``detector_input_threshold(settings)`` so later filters see enough
    proposals.
    """
    disabled_classes = getattr(settings, "disabled_classes", ())
    if settings.thresholds_path is not None and (settings.per_class_thresholds or disabled_classes):
        from src.app.detection_settings import load_threshold_configuration

        _, profile_disabled = load_threshold_configuration(settings.thresholds_path, settings=settings)
        disabled_classes = tuple(sorted(set(disabled_classes) | set(profile_disabled)))
    if not (settings.enable_tiling or settings.enable_two_stage or settings.per_class_thresholds
            or disabled_classes):
        return base_detector
    detector = base_detector
    if settings.enable_tiling:
        from src.detection.tiling import TiledDetector

        detector = TiledDetector(
            detector, tile_size=settings.tile_size, overlap=settings.tile_overlap,
            min_side_to_tile=settings.tiling_min_side, iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
        )
    if settings.enable_two_stage:
        from src.detection.two_stage import MobileNetBatchAdapter, TwoStageDetector

        if classifier is None:
            from src.app.classifier_service import MobileNetClassifier

            classifier = MobileNetBatchAdapter(
                MobileNetClassifier(settings.classifier_path, device=settings.device, prefer_onnx=False)
            )
        elif not hasattr(classifier, "predict_proba"):
            classifier = MobileNetBatchAdapter(classifier)
        detector = TwoStageDetector(
            detector,
            classifier,
            pad_ratio=settings.two_stage_padding,
            iou_threshold=settings.iou_threshold,
            max_detections=settings.max_detections,
            scoring_mode=getattr(settings, "two_stage_scoring_mode", "product"),
            blend_weight=getattr(settings, "two_stage_blend_weight", 0.65),
            min_proposal_gate=getattr(settings, "two_stage_min_proposal_gate", 0.18),
        )
    return PerClassThresholdDetector(
        detector, settings.confidence_threshold, settings.per_class_thresholds, disabled_classes
    )


def count_rows(result: DetectionResult) -> list[dict[str, object]]:
    counts = Counter(detection.class_id for detection in result.detections)
    return [
        {"Loại rác": detection_label(class_id), "Số lượng": count}
        for class_id, count in sorted(counts.items())
    ]


def annotated_filename(original_name: str) -> str:
    # Uploaded filenames are untrusted; archives must never contain directories.
    stem = PurePosixPath(original_name.replace("\\", "/")).stem
    return f"{stem or 'anh'}_nhan_dien.png"


def analyze_image_file(
    filename: str, payload: bytes, service: DetectionService, style: str = "box"
) -> ImageDetectionOutput:
    started = time.perf_counter()
    image, result = detect_image_bytes(payload, service)
    details = tuple(getattr(service.detector, "last_details", ()) or ())
    if len(details) != len(result.detections):
        details = ()
    annotated = draw_detections(image, result, style=style)
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    return ImageDetectionOutput(
        original_name=filename,
        result=result,
        annotated_png=buffer.getvalue(),
        processing_ms=(time.perf_counter() - started) * 1000,
        details=details,
    )


def archive_annotated_images(outputs: list[ImageDetectionOutput]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, output in enumerate(outputs, start=1):
            archive.writestr(
                f"{index:03d}_{annotated_filename(output.original_name)}",
                output.annotated_png,
            )
    return buffer.getvalue()


def detect_image_bytes(
    payload: bytes, service: DetectionService
) -> tuple[Image.Image, DetectionResult]:
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DetectionMediaError(
            "Không thể đọc ảnh. Hãy dùng JPEG, PNG, WebP hoặc BMP hợp lệ."
        ) from error
    return image, service.detect(image)
