from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.schema import CLASS_NAMES
from src.detection.yolo import DetectionError, WasteDetector


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10, 20, 90, 70], [-5, -2, 200, 200]])
    conf = FakeTensor([0.82, 0.94])
    cls = FakeTensor([7, 5])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.names = {index: name for index, name in enumerate(CLASS_NAMES)}

    def predict(self, **kwargs):
        assert kwargs["source"].shape == (80, 100, 3)
        assert kwargs["imgsz"] == 640
        return [FakeResult()]


def test_adapter_maps_and_clamps_every_box(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.touch()
    detector = WasteDetector(
        model_path, device="cpu", model_factory=lambda _: FakeModel()
    )

    result = detector.detect_pil(Image.new("RGB", (100, 80)))

    assert [row.class_id for row in result.detections] == ["metal", "plastic"]
    assert result.detections[0].box.x1 == 0
    assert result.detections[0].box.x2 == 100


def test_adapter_rejects_semantically_wrong_label_order(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.touch()
    model = FakeModel()
    model.names = list(reversed(CLASS_NAMES))
    detector = WasteDetector(model_path, model_factory=lambda _: model)

    with pytest.raises(DetectionError, match="Sai thứ tự lớp"):
        detector.detect_pil(Image.new("RGB", (20, 20)))
