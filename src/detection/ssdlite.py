"""Six-class SSDLite320-MobileNetV3 detector, separate from stage-1 classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult
from src.detection.yolo import DetectionError

ARCHITECTURE = "ssdlite320_mobilenet_v3_large"


def build_ssdlite(*, pretrained: bool = False, backbone_weights: Path | str | None = None) -> Any:
    """Reuse COCO backbone/regression weights and replace all classifier weights.

    Both model constructions use the same reduced-tail MobileNetV3 architecture.
    The default is download-free; only explicitly authorized training asks for COCO.
    If backbone_weights is provided, feature extractor weights are loaded from the stage-1 classifier.
    """
    try:
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )
    except ImportError as error:
        raise DetectionError("Cần cài torch và torchvision để dùng SSDLite") from error
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(DETECTION_CLASS_NAMES) + 1
    )
    if pretrained:
        state = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(
            progress=True, check_hash=True
        )
        reusable = {key: value for key, value in state.items()
                    if not key.startswith("head.classification_head.")}
        missing, unexpected = model.load_state_dict(reusable, strict=False)
        if unexpected or any(not key.startswith("head.classification_head.") for key in missing):
            raise DetectionError("COCO checkpoint không khớp kiến trúc SSDLite")

    if backbone_weights is not None:
        import torch
        bw_path = Path(backbone_weights)
        if not bw_path.is_file():
            raise DetectionError(f"Không tìm thấy checkpoint backbone classifier: {bw_path}")
        ckpt = torch.load(bw_path, map_location="cpu", weights_only=False)
        cls_sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        ssd_sd = model.state_dict()
        transfer_dict = {}
        for k, v in cls_sd.items():
            if not k.startswith("features."):
                continue
            parts = k.split(".")
            block_idx = int(parts[1])
            rest = ".".join(parts[2:])
            if block_idx <= 13:
                target_k = f"backbone.features.0.{block_idx}.{rest}"
            else:
                target_k = f"backbone.features.1.{block_idx - 14}.{rest}"
            if target_k in ssd_sd and ssd_sd[target_k].shape == v.shape:
                transfer_dict[target_k] = v
        missing, unexpected = model.load_state_dict(transfer_dict, strict=False)
        if unexpected:
            raise DetectionError(f"Unexpected keys while loading backbone weights: {unexpected}")
    return model


def resolve_device(value: str) -> Any:
    import torch

    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DetectionError("CUDA không khả dụng; chọn --device cpu hoặc cài PyTorch CUDA")
    return device


def tensor_detections(output: dict[str, Any], width: int, height: int,
                      *, confidence_threshold: float = 0.0,
                      max_detections: int = 100) -> DetectionResult:
    """Convert torchvision's 1-based foreground labels to shared 0-based IDs."""
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    if not len(boxes) == len(scores) == len(labels):
        raise DetectionError("SSDLite trả về các tensor không đồng nhất")
    rows = []
    for box_values, score, label in zip(boxes, scores, labels, strict=True):
        if not np.isfinite(score) or not np.all(np.isfinite(box_values)):
            raise DetectionError("SSDLite trả về tọa độ hoặc xác suất không hữu hạn")
        if not 1 <= int(label) <= len(DETECTION_CLASS_NAMES) or label != int(label):
            raise DetectionError(f"SSDLite trả về nhãn foreground không hợp lệ: {label}")
        if float(score) < confidence_threshold:
            continue
        box = BoundingBox(*(float(value) for value in box_values))
        if box.width <= 0 or box.height <= 0:
            continue
        class_index = int(label) - 1
        rows.append(Detection(class_index, DETECTION_CLASS_NAMES[class_index],
                              float(score), box.clamp(width, height)))
    rows.sort(key=lambda row: (-row.confidence, row.class_index))
    return DetectionResult(tuple(rows[:max_detections]), width, height)


class SSDLiteWasteDetector:
    """Lazy loader for trained, metadata-validated SSDLite checkpoints."""

    def __init__(self, model_path: Path | str, *, device: str = "auto",
                 confidence_threshold: float = 0.35, iou_threshold: float = 0.55,
                 image_size: int = 320, max_detections: int = 100) -> None:
        if image_size != 320:
            raise ValueError("SSDLite320 có kích thước đầu vào cố định 320")
        if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("confidence_threshold và iou_threshold phải thuộc [0, 1]")
        if max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.image_size = image_size
        self._model = None
        self._device = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        if not self.model_path.is_file():
            raise DetectionError(f"Không tìm thấy detector: {self.model_path}")
        try:
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            if (
                metadata.get("architecture") != ARCHITECTURE
                or metadata.get("backend") != "ssdlite"
                or tuple(metadata.get("class_names", ())) != DETECTION_CLASS_NAMES
                or metadata.get("background_index") != 0
                or metadata.get("trained") is not True
                or int(checkpoint.get("epoch", 0)) < 1
            ):
                raise DetectionError("Checkpoint SSDLite sai kiến trúc/nhãn hoặc chưa huấn luyện")
            model = build_ssdlite(pretrained=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except DetectionError:
            raise
        except Exception as error:
            raise DetectionError(f"Không đọc được checkpoint SSDLite: {error}") from error
        self._device = resolve_device(self.device)
        model.score_thresh = self.confidence_threshold
        model.nms_thresh = self.iou_threshold
        model.detections_per_img = self.max_detections
        self._model = model.to(self._device).eval()
        return self._model

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        model = self._ensure_loaded()
        prepared = ImageOps.exif_transpose(image).convert("RGB")
        tensor = pil_to_tensor(prepared).to(self._device, dtype=torch.float32) / 255.0
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model([tensor])[0]
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        elapsed = (time.perf_counter() - started) * 1000
        result = tensor_detections(output, *prepared.size,
                                   confidence_threshold=self.confidence_threshold,
                                   max_detections=self.max_detections)
        return DetectionResult(result.detections, *prepared.size, elapsed)
