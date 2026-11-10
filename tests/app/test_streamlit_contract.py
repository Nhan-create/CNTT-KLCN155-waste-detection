# Mô tả: Kiểm thử hợp đồng cấu trúc giao diện Streamlit (streamlit_app.py) và các điểm kết nối dịch vụ suy luận.
from pathlib import Path

import pytest


def test_phase2_demo_is_static_image_detection_without_icons() -> None:
    source = Path("streamlit_app.py").read_text(encoding="utf-8")

    assert 'st.title("Nhận diện rác trong ảnh")' in source
    assert "create_detector" in source
    assert "accept_multiple_files=True" in source
    assert "LiveDetectionCallback" not in source
    assert "webrtc_streamer" not in source
    assert "Mô hình phân loại toàn bộ khung hình" not in source
    assert "♻" not in source
    assert "view.icon" not in source
    assert "row.icon" not in source
    assert "label.icon" not in source
    assert "latest.icon" not in source
    assert "page_icon=transparent_favicon()" in source
    for alert_call in (
        "st.info(",
        "st.warning(",
        "st.error(",
        "st.success(",
        "st.spinner(",
        "status.info(",
    ):
        assert alert_call not in source
    for label in ("Tải video", "Camera trực tiếp"):
        assert f'"{label}"' not in source
    assert "camera_input" not in source


def test_phase2_demo_opens_with_missing_weights_and_can_change_models(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WASTE_DETECTOR", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("WASTE_DETECTOR_BACKEND", "ssdlite")
    monkeypatch.delenv("WASTE_ENABLE_TILING", raising=False)
    monkeypatch.delenv("WASTE_ENABLE_TWO_STAGE", raising=False)
    root_app = Path(__file__).resolve().parent.parent.parent / "streamlit_app.py"
    app = AppTest.from_file(str(root_app)).run(timeout=20)

    assert not app.exception
    assert app.selectbox[0].value == "ssdlite"
    assert app.checkbox(key="detect-tiling").value is False
    assert app.checkbox(key="detect-refine-ssdlite").value is True
    assert "Chưa có mô hình đã huấn luyện" in " ".join(item.value for item in app.markdown)
    assert len(app.slider) == 1
    uploaders = app.get("file_uploader")
    assert len(uploaders) >= 1
    assert any("detect" in u.key or "detect" in u.label.lower() or u.accept_multiple_files for u in uploaders)
    app.selectbox[0].select("wbf").run()
    assert not app.exception
    assert app.selectbox[0].value == "wbf"
    assert app.checkbox(key="detect-tiling").value is False
    assert app.checkbox(key="detect-refine-wbf").value is True
    app.selectbox[0].select("yolov8n").run()
    assert not app.exception
    assert app.checkbox(key="detect-tiling").value is False
    assert app.checkbox(key="detect-refine-yolov8n").value is False
    app.checkbox(key="detect-tiling").check().run()
    assert not app.exception
    assert app.checkbox(key="detect-tiling").value is True


def test_yolo_can_switch_between_verified_and_original_weights(monkeypatch, tmp_path) -> None:
    from streamlit.testing.v1 import AppTest

    from src.app import detection_settings

    verified = tmp_path / "verified.pt"
    verified.touch()
    monkeypatch.setattr(detection_settings, "VERIFIED_YOLO_PATH", verified, raising=False)
    monkeypatch.delenv("WASTE_DETECTOR", raising=False)
    monkeypatch.delenv("WASTE_DETECTOR_SELECTION", raising=False)
    monkeypatch.setenv("WASTE_DETECTOR_BACKEND", "yolov8n")
    root_app = Path(__file__).resolve().parent.parent.parent / "streamlit_app.py"
    app = AppTest.from_file(str(root_app)).run(timeout=20)

    assert not app.exception
    assert app.checkbox(key="detect-yolo-verified").value is True
    assert any("Checkpoint đã được đánh giá" in caption.value for caption in app.caption)

    app.checkbox(key="detect-yolo-verified").uncheck().run()
    assert not app.exception
    assert any("Checkpoint gốc" in caption.value for caption in app.caption)

    app.selectbox[0].select("wbf").run()
    assert not app.exception
    assert all(widget.key != "detect-yolo-verified" for widget in app.checkbox)

    monkeypatch.setenv("WASTE_DETECTOR", str(detection_settings.DEFAULT_YOLO_PATH))
    pinned_app = AppTest.from_file(str(root_app)).run(timeout=20)
    assert not pinned_app.exception
    assert all(widget.key != "detect-yolo-verified" for widget in pinned_app.checkbox)


@pytest.mark.parametrize(
    ("scoring_mode", "expected_scores"),
    [("product", (0.728,)), ("blend", (0.8715, 0.35))],
)
def test_detection_click_respects_configured_two_stage_scoring(
    monkeypatch, tmp_path, scoring_mode, expected_scores
) -> None:
    import io

    import streamlit
    from PIL import Image
    from streamlit.testing.v1 import AppTest

    from src.app import classifier_service
    from src.detection import factory
    from src.detection.schema import DETECTION_CLASS_NAMES
    from src.detection.types import BoundingBox, Detection, DetectionResult

    checkpoint = tmp_path / "ssdlite.pt"
    checkpoint.touch()
    monkeypatch.setenv("WASTE_DETECTOR", str(checkpoint))
    monkeypatch.setenv("WASTE_DETECTOR_BACKEND", "ssdlite")
    monkeypatch.setenv("WASTE_CONFIDENCE_THRESHOLD", "0.29")
    monkeypatch.setenv("WASTE_TWO_STAGE_SCORING_MODE", scoring_mode)
    for name in ("WASTE_DETECTOR_SELECTION", "WASTE_ENABLE_TILING", "WASTE_ENABLE_TWO_STAGE"):
        monkeypatch.delenv(name, raising=False)

    payload = io.BytesIO()
    Image.new("RGB", (100, 100), (80, 90, 100)).save(payload, format="PNG")

    class UploadedImage:
        name = "bottles.png"

        def getvalue(self):
            return payload.getvalue()

    class Detector:
        def detect_pil(self, image):
            return DetectionResult(
                (
                    Detection(7, "plastic", 0.8, BoundingBox(0, 0, 40, 40)),
                    Detection(7, "plastic", 0.25, BoundingBox(60, 60, 95, 95)),
                ),
                *image.size,
            )

    class Classifier:
        class_names = DETECTION_CLASS_NAMES

        def predict_proba(self, images):
            return [[0, 0, 0, 0, 0, 0, 0.09, 0.91, 0, 0] for _ in images]

    monkeypatch.setattr(
        streamlit, "file_uploader",
        lambda *args, **kwargs: [UploadedImage()] if kwargs.get("accept_multiple_files") else None,
    )
    monkeypatch.setattr(factory, "create_detector", lambda *args, **kwargs: Detector())
    monkeypatch.setattr(classifier_service, "MobileNetClassifier", lambda *args, **kwargs: Classifier())
    streamlit.cache_resource.clear()
    try:
        root_app = Path(__file__).resolve().parent.parent.parent / "streamlit_app.py"
        app = AppTest.from_file(str(root_app)).run(timeout=60)
        assert not app.exception
        app.button[0].click().run(timeout=60)
        assert not app.exception

        output, = app.session_state["detection-results"]
        assert tuple(d.confidence for d in output.result.detections) == pytest.approx(expected_scores)
        assert output.details[0].detector_score == 0.8
        assert output.details[0].classifier_prob == 0.91
    finally:
        streamlit.cache_resource.clear()


def test_cached_pipeline_uses_configured_classifier_checkpoint_and_device(monkeypatch, tmp_path) -> None:
    from dataclasses import replace

    import streamlit
    from PIL import Image

    import streamlit_app
    from src.app.detection_settings import DEFAULT_CLASSIFIER_PATH, parse_detection_settings
    from src.detection import factory
    from src.detection.schema import DETECTION_CLASS_NAMES
    from src.detection.types import BoundingBox, Detection, DetectionResult

    model_path = tmp_path / "detector.pt"
    model_path.write_bytes(b"detector")
    classifier_path = tmp_path / "classifier.pt"
    classifier_path.write_text("plastic", encoding="utf-8")
    settings = replace(
        parse_detection_settings([], {}), model_path=model_path, classifier_path=classifier_path,
        device="cpu", enable_two_stage=True,
    )

    class Detector:
        def detect_pil(self, image):
            return DetectionResult(
                (Detection(7, "plastic", .8, BoundingBox(0, 0, 40, 40)),), *image.size
            )

    class Classifier:
        class_names = DETECTION_CLASS_NAMES

        def __init__(self, checkpoint_path=DEFAULT_CLASSIFIER_PATH, *, device="auto", prefer_onnx=True):
            self.label = (
                Path(checkpoint_path).read_text(encoding="utf-8")
                if Path(checkpoint_path) == classifier_path and device == "cpu" and not prefer_onnx
                else "metal"
            )

        def predict_proba(self, images):
            row = [0.0] * len(DETECTION_CLASS_NAMES)
            row[DETECTION_CLASS_NAMES.index(self.label)] = 1.0
            return [row for _ in images]

    monkeypatch.setattr(factory, "create_detector", lambda *args, **kwargs: Detector())
    monkeypatch.setattr(streamlit_app, "MobileNetClassifier", Classifier)

    def service():
        versions = tuple((path.stat().st_mtime_ns, path.stat().st_size)
                         for path in (model_path, classifier_path))
        return streamlit_app.cached_detector(
            settings.backend, str(model_path), None, settings.device,
            settings.confidence_threshold, settings.iou_threshold, settings.image_size,
            settings.max_detections, settings.fusion_weights, settings.fusion_iou_threshold,
            settings.fusion_input_threshold, versions, pipeline_settings=settings,
        )

    streamlit.cache_resource.clear()
    try:
        image = Image.new("RGB", (100, 100))
        assert service().detect(image).detections[0].class_id == "plastic"
        classifier_path.write_text("glass", encoding="utf-8")
        assert service().detect(image).detections[0].class_id == "glass"
    finally:
        streamlit.cache_resource.clear()
