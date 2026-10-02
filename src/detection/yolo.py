"""Lazy Ultralytics YOLO adapter with strict waste-label validation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.detection.schema import DETECTION_CLASS_NAMES as CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult


class DetectionError(RuntimeError):
    """A detector model or inference result violates the runtime contract."""


def _load_ultralytics_model(model_path: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise DetectionError(
            "Thiếu ultralytics. Hãy chạy: pip install -r requirements.txt"
        ) from error
    return YOLO(model_path, task="detect")


def _ordered_names(values: Mapping[int, str] | Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, Mapping):
        try:
            return tuple(str(values[index]) for index in range(len(values)))
        except KeyError as error:
            raise DetectionError(
                "Detector class indices must be contiguous from zero"
            ) from error
    return tuple(str(value) for value in values)


def validate_yolov8n_architecture(model: Any, expected_classes: int | None = None) -> None:
    """Reject different YOLO generations and verify detection head classes."""
    network = getattr(model, "model", model)
    architecture = getattr(network, "yaml", {}) if isinstance(getattr(network, "yaml", None), dict) else {}
    filename = Path(str(architecture.get("yaml_file", ""))).stem.lower()
    scale = architecture.get("scale")
    depth = architecture.get("depth_multiple")
    width = architecture.get("width_multiple")
    model_name = Path(str(getattr(model, "model_name", "") or getattr(model, "ckpt_path", ""))).stem.lower()
    is_nano_scales = (depth == 0.33 and width == 0.25)
    is_v8n_name = "yolov8n" in model_name
    if not (filename == "yolov8n" or (filename == "yolov8" and scale == "n") or (is_v8n_name and is_nano_scales) or is_nano_scales):
        raise DetectionError("Giai đoạn 2 yêu cầu đúng kiến trúc YOLOv8n")

    if expected_classes is not None:
        yaml_nc = architecture.get("nc")
        if yaml_nc is not None and yaml_nc != expected_classes:
            raise DetectionError(
                f"Kiến trúc YOLOv8n có nc={yaml_nc}, yêu cầu chính xác {expected_classes} lớp rác"
            )
        net_model = getattr(network, "model", None)
        if net_model is not None and hasattr(net_model, "__len__") and len(net_model) > 0:
            try:
                head = net_model[-1]
                head_nc = getattr(head, "nc", None)
                if head_nc is not None and head_nc != expected_classes:
                    raise DetectionError(
                        f"Detection head có nc={head_nc}, yêu cầu chính xác {expected_classes} lớp rác giai đoạn 2"
                    )
                cv3 = getattr(head, "cv3", None)
                if cv3 is not None and hasattr(cv3, "__iter__"):
                    for i, cv_branch in enumerate(cv3):
                        out_conv = (
                            cv_branch[2]
                            if hasattr(cv_branch, "__getitem__") and len(cv_branch) > 2
                            else getattr(cv_branch, "conv", cv_branch)
                        )
                        weight = getattr(out_conv, "weight", None)
                        if weight is not None and weight.shape[0] != expected_classes:
                            raise DetectionError(
                                f"Detection head cv3[{i}] có {weight.shape[0]} output channels, yêu cầu chính xác {expected_classes} lớp"
                            )
            except (IndexError, TypeError):
                pass


def validate_yolo_metadata(model: Any) -> None:
    validate_yolov8n_architecture(model, expected_classes=len(CLASS_NAMES))
    network = getattr(model, "model", model)
    raw_names = getattr(model, "names", getattr(network, "names", None))
    if raw_names is not None:
        names = _ordered_names(raw_names)
        if names != CLASS_NAMES:
            raise DetectionError(
                f"Sai thứ tự lớp detector: {names}; yêu cầu chính xác {CLASS_NAMES}"
            )
    metadata = getattr(network, "phase2_metadata", {})
    if (
        metadata.get("backend") != "yolov8n"
        or metadata.get("trained") is not True
        or tuple(metadata.get("class_names", ())) != CLASS_NAMES
    ):
        raise DetectionError(f"Checkpoint YOLOv8n thiếu metadata huấn luyện {len(CLASS_NAMES)} lớp giai đoạn 2")


class WasteDetector:
    """Detect every visible waste object through one small, thread-safe-ready seam."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        device: str = "auto",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.55,
        image_size: int = 640,
        max_detections: int = 100,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self._model_factory = model_factory or _load_ultralytics_model
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise DetectionError(f"Không tìm thấy detector: {self.model_path}")
        model = self._model_factory(str(self.model_path))
        names = _ordered_names(model.names)
        if names != CLASS_NAMES:
            raise DetectionError(
                f"Sai thứ tự lớp detector: {names}; yêu cầu chính xác {CLASS_NAMES}"
            )
        validate_yolo_metadata(model)
        self._model = model
        return model

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        """Return clamped detections in coordinates of the EXIF-corrected RGB image."""

        model = self._ensure_loaded()
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        width, height = prepared.size
        device = None if self.device == "auto" else self.device
        started = time.perf_counter()
        results = model.predict(
            # Ultralytics interprets NumPy images as BGR; PIL is RGB.
            source=np.asarray(prepared)[:, :, ::-1].copy(),
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=self.max_detections,
            device=device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if len(results) != 1:
            raise DetectionError(
                f"Detector returned {len(results)} results for one image"
            )
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return DetectionResult((), width, height, elapsed_ms)

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        if not (len(xyxy) == len(confidences) == len(classes)):
            raise DetectionError("Detector returned inconsistent box tensors")

        detections: list[Detection] = []
        for coordinates, confidence, class_value in zip(
            xyxy, confidences, classes, strict=True
        ):
            class_index = int(class_value)
            if class_index < 0 or class_index >= len(CLASS_NAMES):
                raise DetectionError(
                    f"Detector returned invalid class index {class_index}"
                )
            box = BoundingBox(*(float(value) for value in coordinates)).clamp(
                width, height
            )
            detections.append(
                Detection(
                    class_index=class_index,
                    class_id=CLASS_NAMES[class_index],
                    confidence=float(confidence),
                    box=box,
                )
            )
        detections.sort(key=lambda row: (-row.confidence, row.class_index))
        return DetectionResult(tuple(detections), width, height, elapsed_ms)
