"""Per-class thresholds, opt-in pipeline flags and pipeline wiring."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult
from src.web.detection_logic import (
    PerClassThresholdDetector,
    analyze_image_file,
    apply_class_thresholds,
    build_detection_pipeline,
    detection_rows,
)
from src.web.detection_service import DetectionService
from src.web.detection_settings import (
    class_threshold,
    detector_input_threshold,
    load_per_class_thresholds,
    parse_detection_settings,
)


def _det(class_id: str, score: float, x: float = 0) -> Detection:
    return Detection(DETECTION_CLASS_NAMES.index(class_id), class_id, score,
                     BoundingBox(x, 0, x + 10, 10))


class Fixed:
    def __init__(self, rows) -> None:
        self.rows = tuple(rows)

    def detect_pil(self, image):
        return DetectionResult(self.rows, *image.size)


def test_defaults_keep_previous_behaviour() -> None:
    settings = parse_detection_settings([], {})
    assert settings.enable_tiling is False and settings.enable_two_stage is False
    assert settings.per_class_thresholds == ()
    assert detector_input_threshold(settings) == settings.confidence_threshold
    base = Fixed([])
    assert build_detection_pipeline(base, settings) is base


def test_flags_from_cli_and_env() -> None:
    settings = parse_detection_settings(["--enable-tiling"], {"WASTE_ENABLE_TWO_STAGE": "1"})
    assert settings.enable_tiling and settings.enable_two_stage
    assert detector_input_threshold(settings) == settings.two_stage_proposal_threshold


def test_per_class_thresholds_file_and_fallback(tmp_path: Path) -> None:
    path = tmp_path / "thr.json"
    path.write_text(json.dumps({"per_class_thresholds": {"battery": 0.7, "paper": 0.2}}), "utf-8")
    settings = parse_detection_settings(
        ["--thresholds", str(path), "--confidence-threshold", "0.5"], {}
    )
    assert settings.per_class_thresholds == (("battery", 0.7), ("paper", 0.2))
    assert class_threshold(settings, "battery") == 0.7
    assert class_threshold(settings, "paper") == 0.2
    assert class_threshold(settings, "glass") == 0.5
    assert detector_input_threshold(settings) == 0.2

    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"metal": 0.33}), "utf-8")
    assert load_per_class_thresholds(flat) == (("metal", 0.33),)


@pytest.mark.parametrize(
    "payload", [{"styrofoam": 0.3}, {"battery": 1.5}, {"battery": "high"}, {"battery": True}, [0.1]]
)
def test_invalid_threshold_files_are_rejected(tmp_path: Path, payload) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), "utf-8")
    with pytest.raises(ValueError):
        parse_detection_settings(["--thresholds", str(path)], {})


def test_apply_class_thresholds() -> None:
    result = DetectionResult(
        (_det("battery", 0.6), _det("paper", 0.25, 20), _det("glass", 0.45, 40),
         _det("glass", 0.55, 60)), 100, 100,
    )
    kept = apply_class_thresholds(result, 0.5, {"battery": 0.7, "paper": 0.2})
    assert [(d.class_id, d.confidence) for d in kept.detections] == [("paper", 0.25), ("glass", 0.55)]


def test_pipeline_with_per_class_only_filters_and_shows_rows(tmp_path: Path) -> None:
    path = tmp_path / "thr.json"
    path.write_text(json.dumps({"battery": 0.7}), "utf-8")
    settings = parse_detection_settings(["--thresholds", str(path), "--confidence-threshold", "0.4"], {})
    pipeline = build_detection_pipeline(
        Fixed([_det("battery", 0.6), _det("metal", 0.45, 20)]), settings
    )
    assert isinstance(pipeline, PerClassThresholdDetector)
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buffer, format="PNG")
    output = analyze_image_file("a.png", buffer.getvalue(), DetectionService(pipeline))
    assert [d.class_id for d in output.result.detections] == ["metal"]
    assert output.details == ()
    assert "Top-3 phân loại" not in detection_rows(output.result, output.details)[0]


def test_pipeline_two_stage_with_fake_classifier_exposes_top3() -> None:
    class Clf:
        class_names = list(DETECTION_CLASS_NAMES)

        def predict_proba(self, images):
            return [[0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0] for _ in images]

    settings = parse_detection_settings(["--enable-two-stage", "--confidence-threshold", "0.4"], {})
    pipeline = build_detection_pipeline(
        Fixed([_det("trash", 0.8), _det("trash", 0.3, 50)]), settings, classifier=Clf()
    )
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), (90, 90, 90)).save(buffer, format="PNG")
    output = analyze_image_file("b.png", buffer.getvalue(), DetectionService(pipeline))
    assert [(d.class_id, round(d.confidence, 3)) for d in output.result.detections] == [("metal", 0.72)]
    rows = detection_rows(output.result, output.details)
    assert rows[0]["Top-3 phân loại"].startswith(f"{rows[0]['Loại rác']} 90%")


def test_image_size_defaults_to_none_so_checkpoint_imgsz_is_used() -> None:
    assert parse_detection_settings([], {}).image_size is None
    assert parse_detection_settings(["--backend", "yolov8n"], {}).image_size is None


def test_image_size_only_when_user_specifies() -> None:
    assert parse_detection_settings(["--image-size", "416"], {}).image_size == 416
    assert parse_detection_settings([], {"WASTE_DETECTOR_IMAGE_SIZE": "320"}).image_size == 320
    assert parse_detection_settings(
        ["--image-size", "640"], {"WASTE_DETECTOR_IMAGE_SIZE": "320"}
    ).image_size == 640
    with pytest.raises(ValueError):
        parse_detection_settings(["--image-size", "300"], {})


def test_none_image_size_is_forwarded_to_factory(monkeypatch) -> None:
    import src.detection.factory as factory

    captured = {}

    class FakeYolo:
        def __init__(self, model_path, *, image_size, **options) -> None:
            captured["image_size"] = image_size

    import src.detection.yolo as yolo

    monkeypatch.setattr(yolo, "WasteDetector", FakeYolo)
    settings = parse_detection_settings(["--backend", "yolov8n"], {})
    factory.create_detector(settings.backend, settings.model_path, image_size=settings.image_size)
    assert captured["image_size"] is None


def test_tiled_and_two_stage_filter_subresolution_noise() -> None:
    from src.detection.tiling import TiledDetector
    from src.detection.two_stage import TwoStageDetector

    # 5x5 box is subresolution noise (< 8px)
    noise_det = Detection(9, "trash", 0.8, BoundingBox(10, 10, 15, 15))
    valid_det = Detection(7, "plastic", 0.8, BoundingBox(10, 10, 50, 50))

    tiled = TiledDetector(Fixed([noise_det, valid_det]), tile_size=100, min_side_to_tile=50)
    res = tiled.detect_pil(Image.new("RGB", (100, 100)))
    # Only the valid 40x40 box should survive, 5x5 noise box must be filtered out
    assert len(res.detections) == 1
    assert res.detections[0].class_id == "plastic"

