"""Runtime settings for the multi-object detection frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SSDLITE_PATH = Path(
    "artifacts/detection/waste-ssdlite320-mobilenetv3/weights/best.pt"
)
DEFAULT_YOLO_PATH = Path("artifacts/detection/waste-yolov8n/weights/best.pt")
DEFAULT_DETECTOR_PATH = DEFAULT_SSDLITE_PATH
DETECTION_BACKENDS = ("ssdlite", "yolov8n", "wbf")


@dataclass(frozen=True)
class DetectionWebSettings:
    model_path: Path
    backend: str = "ssdlite"
    secondary_model_path: Path | None = None
    device: str = "auto"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.55
    image_size: int = 640
    max_detections: int = 100
    fusion_weights: tuple[float, float] = (1.0, 1.0)
    fusion_iou_threshold: float = 0.55
    fusion_input_threshold: float = 0.01
    selection_path: Path | None = None
    video_sample_fps: float = 3.0
    live_inference_fps: float = 5.0
    tracker_iou_threshold: float = 0.30
    tracker_max_missed: int = 1


def validate_detection_settings(settings: DetectionWebSettings) -> None:
    if settings.backend not in DETECTION_BACKENDS:
        raise ValueError(f"backend must be one of {DETECTION_BACKENDS}")
    if not 0.0 <= settings.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= settings.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if settings.image_size <= 0 or settings.image_size % 32 != 0:
        raise ValueError("image_size must be a positive multiple of 32")
    if settings.max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if len(settings.fusion_weights) != 2 or any(
        not math.isfinite(value) or value <= 0 for value in settings.fusion_weights
    ):
        raise ValueError("fusion_weights must contain two finite positive values")
    if not 0.0 <= settings.fusion_iou_threshold <= 1.0:
        raise ValueError("fusion_iou_threshold must be between 0 and 1")
    if not 0.0 <= settings.fusion_input_threshold <= 1.0:
        raise ValueError("fusion_input_threshold must be between 0 and 1")
    if not 0.0 < settings.video_sample_fps <= 30.0:
        raise ValueError("video_sample_fps must be greater than 0 and at most 30")
    if not 0.0 < settings.live_inference_fps <= 30.0:
        raise ValueError("live_inference_fps must be greater than 0 and at most 30")
    if not 0.0 <= settings.tracker_iou_threshold <= 1.0:
        raise ValueError("tracker_iou_threshold must be between 0 and 1")
    if settings.tracker_max_missed < 0:
        raise ValueError("tracker_max_missed must be non-negative")


def _read_selection(path: Path) -> dict:
    """Accept validation-selected options only for the exact checkpoint pair."""
    try:
        selection = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(selection, dict):
            raise ValueError("selection must be an object")
        if selection.get("selected_on") != "val" or selection.get("backend") != "wbf":
            raise ValueError("selection must contain WBF parameters selected on val")
        for name in ("primary", "secondary"):
            checkpoint = Path(selection["models"][name])
            if not checkpoint.is_absolute():
                checkpoint = (path.parent / checkpoint).resolve()
            expected = selection["model_hashes"][name]
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"selection requires a SHA256 for {name}")
            hasher = hashlib.sha256()
            with checkpoint.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != expected.lower():
                raise ValueError(f"selection checkpoint hash mismatch: {name}")
            selection["models"][name] = checkpoint
        options = selection["detector_options"]
        if not isinstance(options, dict):
            raise ValueError("selection requires detector_options")
        # Require the full selected operating configuration; do not silently
        # substitute untuned defaults for missing manifest fields.
        for name in (
            "iou_threshold", "image_size", "max_detections", "fusion_weights",
            "fusion_iou_threshold", "fusion_input_threshold",
        ):
            if name not in options:
                raise ValueError(f"selection is missing detector_options.{name}")
        for name in ("image_size", "max_detections"):
            if not isinstance(options[name], int) or isinstance(options[name], bool):
                raise ValueError(f"selection detector_options.{name} must be an integer")
        for name in ("iou_threshold", "fusion_iou_threshold", "fusion_input_threshold"):
            options[name] = float(options[name])
        if not isinstance(options["fusion_weights"], list):
            raise ValueError("selection fusion_weights must be a list")
        options["fusion_weights"] = [float(value) for value in options["fusion_weights"]]
        selection["operating_confidence"] = float(selection["operating_confidence"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid validation selection: {error}") from error
    return selection


def parse_detection_settings(
    argv: Sequence[str], environ: Mapping[str, str]
) -> DetectionWebSettings:
    selection_parser = argparse.ArgumentParser(add_help=False)
    selection_parser.add_argument(
        "--selection", type=Path, default=environ.get("WASTE_DETECTOR_SELECTION")
    )
    selection_args, _ = selection_parser.parse_known_args(list(argv))
    selection = _read_selection(selection_args.selection) if selection_args.selection else None
    options = selection["detector_options"] if selection else {}
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--backend",
        default="wbf" if selection else environ.get("WASTE_DETECTOR_BACKEND", "ssdlite"),
    )
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--secondary-detector", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--confidence-threshold", type=float,
        default=selection["operating_confidence"] if selection else 0.35,
    )
    parser.add_argument("--iou-threshold", type=float, default=options.get("iou_threshold", 0.55))
    parser.add_argument("--image-size", type=int, default=options.get("image_size", 640))
    parser.add_argument("--max-detections", type=int, default=options.get("max_detections", 100))
    parser.add_argument("--video-sample-fps", type=float, default=3.0)
    parser.add_argument("--live-inference-fps", type=float, default=5.0)
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.30)
    parser.add_argument("--tracker-max-missed", type=int, default=1)
    values, _ = parser.parse_known_args(list(argv))
    model_path = values.detector
    if model_path is None and environ.get("WASTE_DETECTOR"):
        model_path = Path(environ["WASTE_DETECTOR"])
    secondary_path = values.secondary_detector
    if secondary_path is None and environ.get("WASTE_SECONDARY_DETECTOR"):
        secondary_path = Path(environ["WASTE_SECONDARY_DETECTOR"])
    if selection:
        if model_path is not None or secondary_path is not None or values.backend != "wbf":
            raise ValueError("--selection cannot be combined with checkpoint/backend overrides")
        model_path = selection["models"]["primary"]
        secondary_path = selection["models"]["secondary"]
    if values.backend == "wbf" and secondary_path is None:
        secondary_path = DEFAULT_YOLO_PATH
    default_path = (
        DEFAULT_YOLO_PATH if values.backend == "yolov8n" else DEFAULT_SSDLITE_PATH
    )
    settings = DetectionWebSettings(
        model_path=model_path or default_path,
        backend=values.backend,
        secondary_model_path=secondary_path,
        device=values.device,
        confidence_threshold=values.confidence_threshold,
        iou_threshold=values.iou_threshold,
        image_size=values.image_size,
        max_detections=values.max_detections,
        fusion_weights=tuple(float(value) for value in options.get("fusion_weights", (1.0, 1.0))),
        fusion_iou_threshold=float(options.get("fusion_iou_threshold", 0.55)),
        fusion_input_threshold=float(options.get("fusion_input_threshold", 0.01)),
        selection_path=selection_args.selection,
        video_sample_fps=values.video_sample_fps,
        live_inference_fps=values.live_inference_fps,
        tracker_iou_threshold=values.tracker_iou_threshold,
        tracker_max_missed=values.tracker_max_missed,
    )
    validate_detection_settings(settings)
    return settings


def detector_problem(settings: DetectionWebSettings) -> str | None:
    if not settings.model_path.is_file():
        return (
            "Chưa có mô hình đã huấn luyện cho 10 nhóm rác tại cấu hình hiện tại. "
            "Chức năng nhận diện sẽ dùng được sau khi bổ sung trọng số phù hợp."
        )
    if settings.backend == "wbf" and (
        settings.secondary_model_path is None
        or not settings.secondary_model_path.is_file()
    ):
        return (
            "Chế độ kết hợp cần đủ hai mô hình đã huấn luyện: "
            "SSDLite-MobileNetV3 và YOLOv8n."
        )
    return None


def paths_for_backend(
    settings: DetectionWebSettings, backend: str
) -> tuple[Path, Path | None]:
    """Keep supplied checkpoints when switching between individual/fused models."""
    if backend not in DETECTION_BACKENDS:
        raise ValueError(f"backend must be one of {DETECTION_BACKENDS}")
    ssdlite_path = (
        settings.model_path
        if settings.backend in ("ssdlite", "wbf")
        else DEFAULT_SSDLITE_PATH
    )
    yolo_path = (
        settings.model_path
        if settings.backend == "yolov8n"
        else settings.secondary_model_path or DEFAULT_YOLO_PATH
    )
    if backend == "yolov8n":
        return yolo_path, None
    return ssdlite_path, yolo_path if backend == "wbf" else None
