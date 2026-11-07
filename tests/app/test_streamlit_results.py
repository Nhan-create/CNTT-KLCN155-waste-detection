"""Result panels keep their public presentation contract during extraction."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("has_detections", [False, True])
def test_detection_result_panel(tmp_path: Path, has_detections: bool) -> None:
    script = tmp_path / "detection_panel.py"
    script.write_text(
        f'''
import io
from PIL import Image
from src.app.detection_logic import ImageDetectionOutput
from src.detection.types import BoundingBox, Detection, DetectionResult
from streamlit_app import render_detection_result

buffer = io.BytesIO()
Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")
rows = (
    Detection(0, "battery", 0.9, BoundingBox(1, 1, 20, 20)),
    Detection(1, "biological", 0.8, BoundingBox(21, 1, 40, 20)),
    Detection(7, "plastic", 0.7, BoundingBox(41, 1, 60, 20)),
    Detection(8, "shoes", 0.6, BoundingBox(61, 1, 80, 20)),
) if {has_detections!r} else ()
output = ImageDetectionOutput(
    "sample.jpg", DetectionResult(rows, 100, 100), buffer.getvalue(), 12.0
)
render_detection_result(output, 0)
''',
        encoding="utf-8",
    )
    app = AppTest.from_file(script).run(timeout=60)

    assert not app.exception
    assert app.subheader[0].value == "Ảnh 1: sample.jpg"
    assert [metric.value for metric in app.metric] == [
        "4" if has_detections else "0", "12 ms"
    ]
    assert len(app.get("download_button")) == (2 if has_detections else 1)
    messages = " ".join(item.value for item in app.markdown)
    if has_detections:
        assert len(app.dataframe) == 2
        for guide in ("Rác hữu cơ", "Rác tái chế", "Rác nguy hại", "Thu gom riêng"):
            assert guide in messages
    else:
        assert len(app.dataframe) == 0
        assert "Không phát hiện vật thể rác" in messages


@pytest.mark.parametrize("confidence", [0.3, 0.9])
def test_classification_result_panel(tmp_path: Path, confidence: float) -> None:
    script = tmp_path / "classification_panel.py"
    script.write_text(
        f'''
from PIL import Image
from src.app.classifier_service import ClassificationOutput
from streamlit_app import render_classification_result

output = ClassificationOutput(
    "plastic", "Nhựa", {confidence!r}, [("plastic", "Nhựa", {confidence!r})],
    "Tái chế", "Chai nhựa sạch", 25.0,
)
render_classification_result(Image.new("RGB", (20, 20)), output, "bottle.jpg")
''',
        encoding="utf-8",
    )
    app = AppTest.from_file(script).run(timeout=60)

    assert not app.exception
    assert app.subheader[0].value == "Kết quả phân loại: bottle.jpg"
    assert [metric.value for metric in app.metric] == [
        "Nhựa", f"{confidence:.1%}", "25 ms"
    ]
    messages = " ".join(item.value for item in app.markdown)
    assert "Chai nhựa sạch" in messages
    assert ("Độ tin cậy thấp" in messages) == (confidence < 0.4)
