import sys
"""
tests/qa/test_yolo_dataloader_batches.py
----------------------------------------
QA test verifying Ultralytics DataLoader batch generation, tensor shapes,
and coordinates validity across ALL 10 CLASSES on Dataset v3 candidate.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "model_v2" / "yolo_dataloader_verification_report.json"

@pytest.fixture(scope="module")
def yolo_report():
    if not REPORT_PATH.exists():
        import subprocess
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "verify_yolo_dataloader_batches.py")
        ], check=True)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_yolo_dataloader_report_verdict(yolo_report):
    """Verify YOLO DataLoader batch verification passed with zero errors."""
    assert yolo_report["status"] == "PASSED"
    assert yolo_report["errors_count"] == 0
    assert yolo_report["total_classes_declared"] == 10

def test_all_ten_classes_verified_in_batches(yolo_report):
    """Verify that every single class (0..9) was encountered and verified in the DataLoader."""
    assert yolo_report["all_10_classes_verified"] is True
    assert yolo_report["total_classes_verified"] == 10

    verified_ids = {entry["class_id"] for entry in yolo_report["classes_verification_table"]}
    assert verified_ids == set(range(10))

def test_sparse_classes_reported_transparently(yolo_report):
    """Verify battery and biological are tracked and verified in the DataLoader."""
    sparse = yolo_report["sparse_classes_status"]
    assert sparse["battery"]["verified_in_loader"] is True
    assert sparse["battery"]["train_instances"] == 1
    assert sparse["biological"]["verified_in_loader"] is True
    assert sparse["biological"]["train_instances"] == 3

def test_multi_object_augmentation_sample_present(yolo_report):
    """Verify multi-object sample with augmentations was captured and verified."""
    sample = yolo_report["multi_object_augmentation_sample"]
    assert sample is not None
    assert sample["num_boxes"] >= 5
    assert len(sample["boxes_preview"]) > 0
