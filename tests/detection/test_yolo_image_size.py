# Mô tả: Kiểm thử giữ nguyên độ phân giải suy luận gốc từ checkpoint YOLOv8n (imgsz=640) trừ khi truyền kích thước tường minh.
"""Inference image size must default to the checkpoint's training imgsz (QA anomaly follow-up)."""

from pathlib import Path

import numpy as np
from PIL import Image

from src.detection.factory import create_detector
from src.detection.schema import DETECTION_CLASS_NAMES as CLASS_NAMES
from src.detection.yolo import WasteDetector


class _Empty:
    boxes = None


class _TrainedAt320:
    def __init__(self) -> None:
        self.names = {index: name for index, name in enumerate(CLASS_NAMES)}
        self.model = self
        self.yaml = {"scale": "n", "yaml_file": "yolov8n.yaml"}
        self.phase2_metadata = {"backend": "yolov8n", "trained": True, "class_names": list(CLASS_NAMES)}
        self.ckpt = {"train_args": {"imgsz": 320}}
        self.seen_imgsz: list[int] = []

    def predict(self, **kwargs):
        self.seen_imgsz.append(kwargs["imgsz"])
        assert isinstance(kwargs["source"], np.ndarray)
        return [_Empty()]


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "best.pt"
    path.touch()
    return path


def test_default_image_size_follows_checkpoint_train_imgsz(tmp_path: Path) -> None:
    model = _TrainedAt320()
    detector = WasteDetector(_model_file(tmp_path), device="cpu", model_factory=lambda _: model)
    detector.detect_pil(Image.new("RGB", (50, 40)))
    assert model.seen_imgsz == [320]
    assert detector.image_size == 320


def test_explicit_image_size_is_respected(tmp_path: Path) -> None:
    model = _TrainedAt320()
    detector = WasteDetector(_model_file(tmp_path), device="cpu", image_size=640, model_factory=lambda _: model)
    detector.detect_pil(Image.new("RGB", (50, 40)))
    assert model.seen_imgsz == [640]
    assert detector.train_image_size == 320


def test_factory_does_not_force_640_for_yolo(tmp_path: Path) -> None:
    detector = create_detector("yolov8n", _model_file(tmp_path))
    assert detector.image_size is None
