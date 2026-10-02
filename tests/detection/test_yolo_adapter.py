from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.detection.schema import DETECTION_CLASS_NAMES as CLASS_NAMES
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
    cls = FakeTensor([0, 2])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.names = {index: name for index, name in enumerate(CLASS_NAMES)}
        self.model = self
        self.yaml = {"scale": "n", "yaml_file": "yolov8n.yaml"}
        self.phase2_metadata = {
            "backend": "yolov8n",
            "trained": True,
            "class_names": list(CLASS_NAMES),
        }

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

    assert [row.class_id for row in result.detections] == ["cardboard", "battery"]
    assert result.detections[0].box.x1 == 0
    assert result.detections[0].box.x2 == 100
    assert result.detections[1].box.x1 == 10
    assert result.detections[1].box.x2 == 90


def test_adapter_rejects_semantically_wrong_label_order(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.touch()
    model = FakeModel()
    model.names = list(reversed(CLASS_NAMES))
    detector = WasteDetector(model_path, model_factory=lambda _: model)

    with pytest.raises(DetectionError, match="Sai thứ tự lớp"):
        detector.detect_pil(Image.new("RGB", (20, 20)))


def test_adapter_rejects_coco_80_class_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "coco80.pt"
    model_path.touch()
    model = FakeModel()
    model.names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}
    detector = WasteDetector(model_path, model_factory=lambda _: model)

    with pytest.raises(DetectionError, match="Sai thứ tự lớp"):
        detector.detect_pil(Image.new("RGB", (20, 20)))


def test_adapter_rejects_faked_names_with_80_class_head(tmp_path: Path) -> None:
    class FakeConv:
        def __init__(self, out_ch: int) -> None:
            self.weight = np.zeros((out_ch, 64, 1, 1))

    class FakeHead:
        def __init__(self, nc: int) -> None:
            self.nc = nc
            self.cv3 = [[None, None, FakeConv(nc)] for _ in range(3)]

    class FakeDetectionModel:
        def __init__(self, head: FakeHead) -> None:
            self.yaml = {"scale": "n", "yaml_file": "yolov8n.yaml", "nc": 80}
            self.model = [None, head]
            self.phase2_metadata = {
                "backend": "yolov8n",
                "trained": True,
                "class_names": list(CLASS_NAMES),
            }

    model_path = tmp_path / "faked_names_80_head.pt"
    model_path.touch()
    model = FakeModel()
    # names match 10 classes
    model.names = {index: name for index, name in enumerate(CLASS_NAMES)}
    # but actual network has 80-class head and yaml
    model.model = FakeDetectionModel(FakeHead(nc=80))
    detector = WasteDetector(model_path, model_factory=lambda _: model)

    with pytest.raises(DetectionError, match="(Detection head|Kiến trúc YOLOv8n)"):
        detector.detect_pil(Image.new("RGB", (20, 20)))



def test_adapter_rejects_invalid_predicted_class_index(tmp_path: Path) -> None:
    class BadBoxes(FakeBoxes):
        cls = FakeTensor([0, 10])  # 10 is out of bounds for 0-9 taxonomy

    class BadResult:
        boxes = BadBoxes()

    class BadPredictModel(FakeModel):
        def predict(self, **kwargs):
            return [BadResult()]

    model_path = tmp_path / "bad_pred.pt"
    model_path.touch()
    detector = WasteDetector(
        model_path, device="cpu", model_factory=lambda _: BadPredictModel()
    )

    with pytest.raises(DetectionError, match="invalid class index"):
        detector.detect_pil(Image.new("RGB", (100, 80)))

