# Mô tả: Kiểm thử vòng lặp huấn luyện SSDLite320-MobileNetV3 v2, nạp dữ liệu YoloBoxDataset và chọn checkpoint theo Validation.
"""Phase-3 SSDLite training data path: EXIF, SSD box augmentation, custom class count, COCO init."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite_train import YoloBoxDataset, build_ssd_augmentation, dataset_class_names


def _make_dataset(tmp_path: Path, names, *, exif_orientation: int | None = None) -> Path:
    root = tmp_path / "ds"
    (root / "images" / "real").mkdir(parents=True)
    (root / "labels" / "real").mkdir(parents=True)
    (root / "splits").mkdir()
    # Stored pixels: 60 wide x 40 tall; red object in the stored top-left corner.
    pixels = np.full((40, 60, 3), 128, dtype=np.uint8)
    pixels[0:10, 0:20] = (255, 0, 0)
    image = Image.fromarray(pixels)
    path = root / "images" / "real" / "a.jpg"
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        image.save(path, exif=exif.tobytes(), quality=95)
    else:
        image.save(path, quality=95)
    # Label expressed in the EXIF-corrected (displayed) frame.
    if exif_orientation == 6:  # rotate 90 CW for display -> 40 wide x 60 tall, red at top-right
        label = "0 0.875 0.1666667 0.25 0.3333333\n"
    else:
        label = "0 0.1666667 0.125 0.3333333 0.25\n"
    (root / "labels" / "real" / "a.txt").write_text(label, encoding="utf-8")
    for split in ("train", "val"):
        (root / "splits" / f"{split}.txt").write_text("images/real/a.jpg\n", encoding="utf-8")
    data = {"path": str(root), "train": "splits/train.txt", "val": "splits/val.txt",
            "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return yaml_path


def test_dataset_applies_exif_transpose(tmp_path: Path) -> None:
    data = _make_dataset(tmp_path, DETECTION_CLASS_NAMES, exif_orientation=6)
    image, target = YoloBoxDataset(data, "val")[0]
    assert tuple(image.shape) == (3, 60, 40)  # displayed frame, not stored 40x60
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert x1 == pytest.approx(30, abs=0.5) and x2 == pytest.approx(40, abs=0.5)
    assert y1 == pytest.approx(0, abs=0.5) and y2 == pytest.approx(20, abs=0.5)
    region = image[:, int(y1):int(y2), int(x1):int(x2)]
    assert region[0].mean() > 0.85 and region[1].mean() < 0.2  # box covers the red object
    assert target["labels"].tolist() == [1]  # foreground = class + 1


def test_dataset_supports_single_class_yaml(tmp_path: Path) -> None:
    data = _make_dataset(tmp_path, ["waste"])
    assert dataset_class_names(data) == ("waste",)
    _, target = YoloBoxDataset(data, "train")[0]
    assert target["labels"].tolist() == [1]


def test_ssd_augmentation_keeps_boxes_on_their_object() -> None:
    from torchvision import tv_tensors

    augment = build_ssd_augmentation(photometric=False)
    base = torch.full((3, 120, 160), 128, dtype=torch.uint8)
    base[:, 30:70, 50:110] = torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1)
    saw_zoom = saw_crop = False
    for seed in range(60):
        torch.manual_seed(seed)
        sample = augment({
            "image": tv_tensors.Image(base.clone()),
            "boxes": tv_tensors.BoundingBoxes([[50, 30, 110, 70]], format="XYXY", canvas_size=(120, 160)),
            "labels": torch.tensor([3]),
        })
        img, boxes = sample["image"], sample["boxes"]
        h, w = img.shape[-2:]
        saw_zoom |= h > 120 or w > 160
        saw_crop |= h < 120 or w < 160
        assert tuple(boxes.canvas_size) == (h, w)
        for box in boxes.tolist():
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            region = img[:, y1:y2, x1:x2].float()
            red = ((region[0] > 200) & (region[1] < 60)).float().mean()
            assert red > 0.9, f"seed {seed}: box {box} lost its object (red={red:.2f})"
        assert sample["labels"].tolist() == [3] * len(boxes)
    assert saw_zoom and saw_crop


def test_build_ssdlite_coco_init_without_classifier_backbone_override() -> None:
    from src.detection.ssdlite import build_ssdlite
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    model = build_ssdlite(pretrained=True, num_classes=1)
    coco = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(progress=False, check_hash=False)
    state = model.state_dict()
    for key in ("backbone.features.0.0.0.weight", "backbone.extra.0.1.0.weight",
                "head.regression_head.module_list.0.1.weight"):
        assert torch.equal(state[key], coco[key])
    assert state["head.classification_head.module_list.0.1.weight"].shape[0] == 2 * 6  # (1 fg + bg) x anchors
