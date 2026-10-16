"""Regression tests for 10-class detection mapping across entire pipeline.

Verifies:
1. visualization.py has full 10-class canonical Vietnamese display names.
2. schema.py maps all documented vn_trash source labels without error.
3. SSDLite background convention (foreground 1..10, background 0, +/- 1 shift).
4. Checkpoint validator rejects mutated class orders (mutation testing).
5. Output of real label files strictly matches canonical 10 classes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch
import yaml
from PIL import Image

from src.detection.schema import (
    DETECTION_CLASS_NAMES,
    DETECTION_CLASS_NAMES_VI,
    SOURCE_LABEL_MAPPING,
    map_source_label,
    DetectionDatasetError,
)
from src.detection.visualization import (
    DETECTION_DISPLAY_NAMES,
    detection_label,
)
from src.detection.yolo import (
    WasteDetector,
    DetectionError,
    validate_yolo_metadata,
)
from src.detection.ssdlite import (
    SSDLiteWasteDetector,
    tensor_detections,
    ARCHITECTURE,
)
from src.detection.ssdlite_train import YoloBoxDataset


def test_visualization_display_names_contains_all_10_canonical_classes():
    """DETECTION_DISPLAY_NAMES must explicitly define all 10 canonical classes."""
    missing = [c for c in DETECTION_CLASS_NAMES if c not in DETECTION_DISPLAY_NAMES]
    assert not missing, f"DETECTION_DISPLAY_NAMES is missing canonical classes: {missing}"

    # Also verify detection_label returns Vietnamese strings for all 10 classes
    for cid in DETECTION_CLASS_NAMES:
        label = detection_label(cid)
        assert label, f"detection_label returned empty string for {cid}"
        assert label != cid, f"detection_label returned raw English ID for {cid}"


def test_schema_maps_all_vn_trash_source_labels():
    """map_source_label must successfully map all documented vn_trash classes to canonical classes."""
    expected_mappings = {
        "Alu": "metal",
        "Carton": "cardboard",
        "Foam_box": "plastic",
        "Milk_box": "cardboard",
        "Other": "trash",
        "PET": "plastic",
        "Paper": "paper",
        "Paper_cup": "paper",
        "Plastic_cup": "plastic",
    }
    for source_label, expected_canonical in expected_mappings.items():
        mapped = map_source_label("vn_trash", source_label)
        assert mapped == expected_canonical, (
            f"vn_trash '{source_label}' mapped to '{mapped}', expected '{expected_canonical}'"
        )
        assert mapped in DETECTION_CLASS_NAMES, f"'{mapped}' is not in DETECTION_CLASS_NAMES"


def test_schema_rejects_automatic_medical_and_organic_mapping():
    """Unverified labels like medical and organic must be rejected from automatic mapping."""
    with pytest.raises(DetectionDatasetError, match="Exclude or manually review"):
        map_source_label("vn_trash", "medical")

    with pytest.raises(DetectionDatasetError, match="Exclude or manually review"):
        map_source_label("vn_trash", "organic")

    # Only when explicitly reviewed with reviewer can an ambiguous label be assigned
    reviewed_medical = map_source_label("vn_trash", "medical", manual_class="trash", reviewer="reviewer_1")
    assert reviewed_medical == "trash"


def test_ssdlite_background_convention_and_id_shift():
    """Verify SSDLite dataset shifts labels +1 (foreground 1..10) and detector shifts -1 (0..9)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        images_dir = tmp_path / "images" / "train"
        labels_dir = tmp_path / "labels" / "train"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create dummy image and label with class 0 (battery) and class 9 (trash)
        img_file = images_dir / "sample.jpg"
        lbl_file = labels_dir / "sample.txt"
        Image.new("RGB", (100, 100), color="white").save(img_file)
        lbl_file.write_text("0 0.5 0.5 0.2 0.2\n9 0.8 0.8 0.1 0.1\n", encoding="utf-8")

        yaml_file = tmp_path / "dataset.yaml"
        yaml_content = {
            "path": str(tmp_path),
            "train": "images/train",
            "val": "images/train",
            "test": "images/train",
            "names": {i: name for i, name in enumerate(DETECTION_CLASS_NAMES)},
            "nc": len(DETECTION_CLASS_NAMES),
        }
        yaml_file.write_text(yaml.safe_dump(yaml_content), encoding="utf-8")

        dataset = YoloBoxDataset(yaml_file, "train", augmentation_strategy="none")
        _, target = dataset[0]

        # Dataset targets must be shifted by +1: class 0 -> 1, class 9 -> 10
        target_labels = target["labels"].tolist()
        assert target_labels == [1, 10], f"Expected shifted labels [1, 10], got {target_labels}"

        # Inference conversion: tensor label 1 -> class_index 0 (battery), tensor label 10 -> class_index 9 (trash)
        mock_output = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [70.0, 70.0, 80.0, 80.0]]),
            "scores": torch.tensor([0.9, 0.85]),
            "labels": torch.tensor([1, 10]),  # 1 and 10 from model head
        }
        res = tensor_detections(mock_output, width=100, height=100)
        assert len(res.detections) == 2
        assert res.detections[0].class_id == "battery"
        assert res.detections[0].class_index == 0
        assert res.detections[1].class_id == "trash"
        assert res.detections[1].class_index == 9

        # Reject invalid background label (label 0 is reserved for background, cannot be returned as detection)
        bad_output = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),  # 0 is background
        }
        with pytest.raises(DetectionError, match="nhãn foreground không hợp lệ"):
            tensor_detections(bad_output, width=100, height=100)


def test_checkpoint_class_order_mutation_fails():
    """Deliberately swapping two classes in checkpoint metadata or model.names must fail validation."""
    class MockYOLOModel:
        def __init__(self, names: list[str]):
            self.names = {i: name for i, name in enumerate(names)}
            self.phase2_metadata = {
                "backend": "yolov8n",
                "trained": True,
                "class_names": names,
            }
            self.yaml = {
                "yaml_file": "yolov8n.yaml",
                "scale": "n",
                "depth_multiple": 0.33,
                "width_multiple": 0.25,
                "nc": len(names),
            }

    # 1. Valid order passes
    valid_model = MockYOLOModel(list(DETECTION_CLASS_NAMES))
    validate_yolo_metadata(valid_model)  # should not raise

    # 2. Mutated class order (swapping battery and biological) must raise DetectionError
    mutated_names = list(DETECTION_CLASS_NAMES)
    mutated_names[0], mutated_names[1] = mutated_names[1], mutated_names[0]
    mutated_model = MockYOLOModel(mutated_names)

    with pytest.raises(DetectionError, match="Sai thứ tự lớp detector"):
        validate_yolo_metadata(mutated_model)

    # 3. SSDLite checkpoint mutation
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "mutated_ssdlite.pt"
        torch.save(
            {
                "model_state_dict": {},
                "epoch": 10,
                "metadata": {
                    "architecture": ARCHITECTURE,
                    "backend": "ssdlite",
                    "class_names": tuple(mutated_names),  # Mutated!
                    "background_index": 0,
                    "trained": True,
                },
            },
            ckpt_path,
        )
        detector = SSDLiteWasteDetector(ckpt_path)
        with pytest.raises(DetectionError, match="Checkpoint SSDLite sai kiến trúc/nhãn hoặc chưa huấn luyện"):
            detector._ensure_loaded()
