"""
tests/test_detection_annotations.py
------------------------------------
Test suite for bounding box annotation integrity and validation.
Verifies that:
1. Real project annotations have 0 syntax or boundary errors.
2. The validator detects erroneous coordinates, out-of-range class IDs, malformed lines, and duplicates.
"""

import tempfile
import pytest
from pathlib import Path
from scripts.validate_detection_annotations import (
    validate_annotation_file,
    validate_dataset,
    calculate_iou
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_calculate_iou():
    # Exact same box
    box1 = [0.5, 0.5, 0.2, 0.2]
    assert calculate_iou(box1, box1) == pytest.approx(1.0)

    # Disjoint boxes
    box2 = [0.1, 0.1, 0.05, 0.05]
    assert calculate_iou(box1, box2) == 0.0

    # Half overlap
    box3 = [0.5, 0.5, 0.2, 0.2]
    box4 = [0.6, 0.5, 0.2, 0.2] # overlap width = 0.1, h = 0.2 => inter = 0.02. union = 0.04 + 0.04 - 0.02 = 0.06 => iou = 1/3
    assert calculate_iou(box3, box4) == pytest.approx(1.0 / 3.0)

def test_valid_annotation_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_file = Path(tmpdir) / "sample_valid.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n8 0.1 0.2 0.15 0.25\n", encoding="utf-8")
        
        errors, warnings, count = validate_annotation_file(label_file)
        assert len(errors) == 0
        assert len(warnings) == 0
        assert count == 2

def test_invalid_class_id():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_file = Path(tmpdir) / "sample_invalid_class.txt"
        # Class 10 is out of bounds [0, 9]
        label_file.write_text("10 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        
        errors, warnings, count = validate_annotation_file(label_file)
        assert len(errors) == 1
        assert errors[0]["type"] == "OUT_OF_BOUNDS_CLASS"

def test_out_of_range_coordinates():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_file = Path(tmpdir) / "sample_bad_coords.txt"
        # xc = 1.2, width = -0.1
        label_file.write_text("3 1.2 0.5 -0.1 0.3\n", encoding="utf-8")
        
        errors, warnings, count = validate_annotation_file(label_file)
        error_types = {e["type"] for e in errors}
        assert "CENTER_OUT_OF_RANGE" in error_types
        assert "DIMENSIONS_OUT_OF_RANGE" in error_types

def test_malformed_field_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_file = Path(tmpdir) / "sample_malformed.txt"
        # Only 4 fields
        label_file.write_text("1 0.2 0.3 0.4\n", encoding="utf-8")
        
        errors, warnings, count = validate_annotation_file(label_file)
        assert len(errors) == 1
        assert errors[0]["type"] == "INVALID_FIELD_COUNT"

def test_duplicate_box_warning():
    with tempfile.TemporaryDirectory() as tmpdir:
        label_file = Path(tmpdir) / "sample_duplicate.txt"
        label_file.write_text("2 0.4 0.4 0.3 0.3\n2 0.4 0.4 0.3 0.3\n", encoding="utf-8")
        
        errors, warnings, count = validate_annotation_file(label_file)
        assert len(errors) == 0
        assert len(warnings) >= 1
        assert any(w["type"] == "NEAR_DUPLICATE_BOX" for w in warnings)

def test_real_dataset_annotations():
    labels_dir = PROJECT_ROOT / "data" / "detection" / "labels"
    assert labels_dir.exists(), f"Labels directory {labels_dir} does not exist!"
    
    summary = validate_dataset(labels_dir)
    assert summary["total_files_checked"] == 1419
    assert summary["total_boxes_validated"] == 4602
    assert summary["total_errors"] == 0
    assert summary["passed"] is True
