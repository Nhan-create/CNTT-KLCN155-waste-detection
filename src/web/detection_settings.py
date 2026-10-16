"""Runtime settings for the multi-object detection frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SSDLITE_V2_PATH = Path(
    "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt"
)
DEFAULT_SSDLITE_PATH = (
    DEFAULT_SSDLITE_V2_PATH
    if DEFAULT_SSDLITE_V2_PATH.exists()
    else Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt")
)
DEFAULT_YOLO_V2_PATH = Path(
    "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"
)
DEFAULT_YOLO_PATH = (
    DEFAULT_YOLO_V2_PATH
    if DEFAULT_YOLO_V2_PATH.exists()
    else Path("artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt")
)
DEFAULT_DETECTOR_PATH = DEFAULT_YOLO_PATH
DETECTION_BACKENDS = ("yolov8n", "ssdlite", "wbf")
DEFAULT_CLASSIFIER_PATH = Path("artifacts/official_run/best_model.pt")
DEFAULT_THRESHOLDS_PATH = Path("configs/detection_thresholds.json")


@dataclass(frozen=True)
class DetectionWebSettings:
    model_path: Path
    backend: str = "ssdlite"
    secondary_model_path: Path | None = None
    device: str = "auto"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.55
    # None -> detector reads imgsz from the checkpoint (YOLO train_args; SSDLite fixed 320).
    image_size: int | None = None
    max_detections: int = 100
    fusion_weights: tuple[float, float] = (1.0, 1.0)
    fusion_iou_threshold: float = 0.55
    fusion_input_threshold: float = 0.01
    selection_path: Path | None = None
    video_sample_fps: float = 3.0
    live_inference_fps: float = 5.0
    tracker_iou_threshold: float = 0.30
    tracker_max_missed: int = 1
    # Optional per-class operating thresholds as sorted (class_id, threshold) pairs
    # (tuple keeps the frozen settings hashable); missing classes use confidence_threshold.
    per_class_thresholds: tuple[tuple[str, float], ...] = ()
    thresholds_path: Path | None = None
    enable_tiling: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.2
    tiling_min_side: int = 400
    enable_two_stage: bool = False
    classifier_path: Path = DEFAULT_CLASSIFIER_PATH
    two_stage_padding: float = 0.10
    # Stage-1 proposal threshold when two-stage is on; the final filter is applied
    # to det_score * cls_prob with the (per-class) operating threshold.
    two_stage_proposal_threshold: float = 0.08
    two_stage_scoring_mode: str = "product"
    two_stage_blend_weight: float = 0.65
    two_stage_min_proposal_gate: float = 0.12


def class_threshold(settings: DetectionWebSettings, class_id: str) -> float:
    for name, value in settings.per_class_thresholds:
        if name == class_id:
            return value
    return settings.confidence_threshold


def detector_input_threshold(settings: DetectionWebSettings) -> float:
    """Lowest score the raw detector must keep so later filters can still apply."""
    values = [settings.confidence_threshold, *(v for _, v in settings.per_class_thresholds)]
    if settings.enable_two_stage:
        values.append(settings.two_stage_proposal_threshold)
    return min(values)


def load_per_class_thresholds(path: Path) -> tuple[tuple[str, float], ...]:
    """Read ``{"per_class_thresholds": {class_id: thr}}`` or a flat ``{class_id: thr}`` JSON."""
    from src.detection.schema import DETECTION_CLASS_NAMES

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid per-class thresholds file {path}: {error}") from error
    if isinstance(payload, dict) and "per_class_thresholds" in payload:
        payload = payload["per_class_thresholds"]
    if not isinstance(payload, dict):
        raise ValueError("per-class thresholds must be a JSON object")
    pairs = []
    for name, value in payload.items():
        if name not in DETECTION_CLASS_NAMES:
            raise ValueError(f"Unknown class in per-class thresholds: {name}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Threshold for {name} must be a number")
        value = float(value)
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"Threshold for {name} must be within [0, 1]")
        pairs.append((name, value))
    return tuple(sorted(pairs))



def validate_detection_settings(settings: DetectionWebSettings) -> None:
    if settings.backend not in DETECTION_BACKENDS:
        raise ValueError(f"backend must be one of {DETECTION_BACKENDS}")
    if not 0.0 <= settings.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= settings.iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if settings.image_size is not None and (
        settings.image_size <= 0 or settings.image_size % 32 != 0
    ):
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
    for name, value in settings.per_class_thresholds:
        if not (math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"per-class threshold for {name} must be between 0 and 1")
    if settings.tile_size <= 0 or settings.tiling_min_side <= 0:
        raise ValueError("tile_size and tiling_min_side must be positive")
    if not 0.0 <= settings.tile_overlap < 1.0:
        raise ValueError("tile_overlap must lie in [0, 1)")
    if not (math.isfinite(settings.two_stage_padding) and 0.0 <= settings.two_stage_padding <= 1.0):
        raise ValueError("two_stage_padding must lie in [0, 1]")
    if not 0.0 <= settings.two_stage_proposal_threshold <= 1.0:
        raise ValueError("two_stage_proposal_threshold must be between 0 and 1")
    if settings.two_stage_scoring_mode not in ("product", "blend"):
        raise ValueError("two_stage_scoring_mode must be 'product' or 'blend'")
    if not (math.isfinite(settings.two_stage_blend_weight) and 0.0 <= settings.two_stage_blend_weight <= 1.0):
        raise ValueError("two_stage_blend_weight must lie in [0, 1]")
    if not (math.isfinite(settings.two_stage_min_proposal_gate) and 0.0 <= settings.two_stage_min_proposal_gate <= 1.0):
        raise ValueError("two_stage_min_proposal_gate must lie in [0, 1]")


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
        default=selection["operating_confidence"] if selection else 0.45,
    )
    parser.add_argument("--iou-threshold", type=float, default=options.get("iou_threshold", 0.55))
    parser.add_argument(
        "--image-size", type=int,
        default=options.get("image_size", environ.get("WASTE_DETECTOR_IMAGE_SIZE")),
    )
    parser.add_argument("--max-detections", type=int, default=options.get("max_detections", 100))
    parser.add_argument("--video-sample-fps", type=float, default=3.0)
    parser.add_argument("--live-inference-fps", type=float, default=5.0)
    parser.add_argument("--tracker-iou-threshold", type=float, default=0.30)
    parser.add_argument("--tracker-max-missed", type=int, default=1)
    parser.add_argument("--thresholds", type=Path, default=environ.get("WASTE_DETECTOR_THRESHOLDS"))
    parser.add_argument(
        "--enable-tiling", action="store_true",
        default=environ.get("WASTE_ENABLE_TILING", "").lower() in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--enable-two-stage", action="store_true",
        default=environ.get("WASTE_ENABLE_TWO_STAGE", "").lower() in ("1", "true", "yes"),
    )
    parser.add_argument(
        "--two-stage-scoring-mode",
        choices=["product", "blend"],
        default=environ.get("WASTE_TWO_STAGE_SCORING_MODE", "product"),
    )
    parser.add_argument(
        "--two-stage-proposal-threshold",
        type=float,
        default=float(environ.get("WASTE_TWO_STAGE_PROPOSAL_THRESHOLD", 0.08)),
    )
    parser.add_argument(
        "--two-stage-min-proposal-gate",
        type=float,
        default=float(environ.get("WASTE_TWO_STAGE_MIN_PROPOSAL_GATE", 0.12)),
    )
    parser.add_argument("--classifier", type=Path, default=environ.get("WASTE_CLASSIFIER"))
    values, _ = parser.parse_known_args(list(argv))
    thresholds_path = values.thresholds
    if thresholds_path is None and DEFAULT_THRESHOLDS_PATH.is_file():
        thresholds_path = DEFAULT_THRESHOLDS_PATH
    per_class = load_per_class_thresholds(thresholds_path) if thresholds_path else ()
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
        per_class_thresholds=per_class,
        thresholds_path=thresholds_path,
        enable_tiling=bool(values.enable_tiling),
        enable_two_stage=bool(values.enable_two_stage),
        classifier_path=values.classifier or DEFAULT_CLASSIFIER_PATH,
        two_stage_scoring_mode=values.two_stage_scoring_mode,
        two_stage_proposal_threshold=values.two_stage_proposal_threshold,
        two_stage_min_proposal_gate=values.two_stage_min_proposal_gate,
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
