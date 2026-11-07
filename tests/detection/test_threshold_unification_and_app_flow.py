# Mô tả: Kiểm thử hợp nhất ngưỡng tin cậy (0.45 / 0.55 / 0.64), chống ghi đè ngầm trên giao diện Streamlit và kiểm chứng luồng ứng dụng thực tế (DetectionService -> analyze_image_file -> PNG overlay).
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from src.app.detection_logic import analyze_image_file, build_detection_pipeline
from src.app.detection_service import DetectionService
from src.app.detection_settings import (
    DEFAULT_YOLO_PATH,
    detector_input_threshold,
    parse_detection_settings,
)
from src.detection.factory import create_detector
from streamlit_app import (
    resolve_active_per_class_thresholds,
    resolve_slider_default_confidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_YOLOV8N_WEIGHTS = REPO_ROOT / DEFAULT_YOLO_PATH


@pytest.mark.parametrize(
    ("cli_args", "env", "expected_conf"),
    [
        ([], {}, 0.45),
        (["--confidence-threshold", "0.45"], {}, 0.45),
        (["--confidence-threshold", "0.55"], {}, 0.55),
        (["--confidence-threshold", "0.64"], {}, 0.64),
        ([], {"WASTE_CONFIDENCE_THRESHOLD": "0.55"}, 0.55),
        ([], {"WASTE_CONFIDENCE_THRESHOLD": "0.64"}, 0.64),
        ([], {"WASTE_DETECTOR_CONFIDENCE": "0.64"}, 0.64),
    ],
)
def test_parse_detection_settings_preserves_exact_thresholds(
    cli_args: list[str], env: dict[str, str], expected_conf: float
) -> None:
    settings = parse_detection_settings(cli_args, env)
    assert settings.confidence_threshold == pytest.approx(expected_conf)
    assert resolve_slider_default_confidence(settings) == pytest.approx(expected_conf)


@pytest.mark.parametrize("selected_conf", [0.45, 0.55, 0.64])
def test_parse_detection_settings_from_selection_json(
    tmp_path: Path, selected_conf: float
) -> None:
    dummy_ckpt = tmp_path / "dummy_yolov8.pt"
    dummy_ckpt.write_bytes(b"deterministic-checkpoint-bytes-for-selection-test")
    ckpt_sha256 = hashlib.sha256(dummy_ckpt.read_bytes()).hexdigest()

    selection_file = tmp_path / f"selection_{int(selected_conf * 100)}.json"
    selection_file.write_text(
        json.dumps(
            {
                "selected_on": "val",
                "backend": "yolov8n",
                "operating_confidence": selected_conf,
                "models": {"primary": str(dummy_ckpt)},
                "model_hashes": {"primary": ckpt_sha256},
                "detector_options": {
                    "iou_threshold": 0.50,
                    "image_size": 640,
                    "max_detections": 300,
                },
            }
        ),
        encoding="utf-8",
    )
    settings = parse_detection_settings(["--selection", str(selection_file)], {})
    assert settings.backend == "yolov8n"
    assert settings.confidence_threshold == pytest.approx(selected_conf)
    assert resolve_slider_default_confidence(settings) == pytest.approx(selected_conf)


def test_calibrated_per_class_thresholds_require_explicit_opt_in() -> None:
    settings = parse_detection_settings(["--confidence-threshold", "0.64"], {})
    assert resolve_active_per_class_thresholds(
        settings, 0.64, enable_calibrated_thresholds=False
    ) == ()

    calibrated = resolve_active_per_class_thresholds(
        settings, 0.64, enable_calibrated_thresholds=True
    )
    assert len(calibrated) == 10
    calibrated_map = dict(calibrated)
    assert calibrated_map["plastic"] == pytest.approx(0.64)
    assert calibrated_map["clothes"] == pytest.approx(0.896)
    assert calibrated_map["shoes"] == pytest.approx(0.896)


@pytest.mark.parametrize("target_conf", [0.45, 0.55, 0.64])
def test_streamlit_slider_does_not_silently_override_high_thresholds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target_conf: float
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WASTE_DETECTOR", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("WASTE_DETECTOR_BACKEND", "yolov8n")
    monkeypatch.setenv("WASTE_CONFIDENCE_THRESHOLD", f"{target_conf:.2f}")

    app = AppTest.from_file(str(REPO_ROOT / "streamlit_app.py")).run(timeout=20)
    assert not app.exception
    assert len(app.slider) == 1
    assert app.slider[0].value == pytest.approx(target_conf)
    assert app.slider[0].step == pytest.approx(0.01)
    assert len(app.checkbox) >= 1
    assert app.checkbox[0].value is False


def test_real_app_pipeline_inference_and_overlay_at_045_055_064(tmp_path: Path) -> None:
    if not ACTIVE_YOLOV8N_WEIGHTS.is_file():
        pytest.skip(f"Missing YOLOv8n checkpoint: {ACTIVE_YOLOV8N_WEIGHTS}")

    val_dir = REPO_ROOT / "data" / "detection_v5_clean" / "images" / "val"
    val_images = sorted(val_dir.glob("*.jpg"))
    if len(val_images) < 2:
        pytest.skip("Missing Validation V5 images")

    preferred = [
        val_dir / "oi_49d2640bef057acb.jpg",
        val_dir / "oi_023d6c9bc2d8b253.jpg",
    ]
    sample_paths = [
        p if p.is_file() else val_images[idx] for idx, p in enumerate(preferred)
    ]

    thresholds = (0.45, 0.55, 0.64)
    for img_path in sample_paths:
        payload = img_path.read_bytes()
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            expected_size = ImageOps.exif_transpose(opened).convert("RGB").size

        counts_by_threshold: dict[float, int] = {}
        for conf in thresholds:
            base_settings = parse_detection_settings(
                [
                    "--backend",
                    "yolov8n",
                    "--detector",
                    str(ACTIVE_YOLOV8N_WEIGHTS),
                    "--confidence-threshold",
                    f"{conf:.2f}",
                    "--device",
                    "cpu",
                ],
                {},
            )
            settings = replace(
                base_settings,
                enable_tiling=False,
                enable_two_stage=False,
                per_class_thresholds=(),
            )
            detector = create_detector(
                settings.backend,
                settings.model_path,
                device=settings.device,
                confidence_threshold=detector_input_threshold(settings),
                iou_threshold=settings.iou_threshold,
                image_size=settings.image_size,
                max_detections=settings.max_detections,
            )
            pipeline = build_detection_pipeline(detector, settings, classifier=None)
            service = DetectionService(pipeline)

            output = analyze_image_file(img_path.name, payload, service, style="box")
            assert (output.result.image_width, output.result.image_height) == expected_size

            with Image.open(io.BytesIO(output.annotated_png)) as rendered_img:
                rendered_img.load()
                assert rendered_img.size == expected_size

            for det in output.result.detections:
                assert det.confidence >= conf - 1e-6
                left, top, right, bottom = det.box.x1, det.box.y1, det.box.x2, det.box.y2
                assert 0 <= left < right <= expected_size[0]
                assert 0 <= top < bottom <= expected_size[1]

            out_png_path = tmp_path / f"{img_path.stem}_conf_{int(round(conf * 100)):03d}.png"
            out_png_path.write_bytes(output.annotated_png)
            counts_by_threshold[conf] = len(output.result.detections)

        assert counts_by_threshold[0.45] >= counts_by_threshold[0.55] >= counts_by_threshold[0.64]
