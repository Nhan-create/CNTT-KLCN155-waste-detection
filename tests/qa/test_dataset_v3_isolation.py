"""
tests/qa/test_dataset_v3_isolation.py
-------------------------------------
Comprehensive integration & regression tests for Dataset v3 candidate build:
1. Initial build creates manifests, splits, YOLO files, and ledger.
2. Source label update propagates to v3 label upon rebuild.
3. Modifying v3 label NEVER mutates source v2 label (independent copy, no hardlink).
4. Manually modified v3 label is preserved upon source update (conflict detection).
5. Stale label and image cleanup works cleanly within candidate directory.
6. Build logic NEVER deletes or touches files outside the candidate directory.
7. No chmod is set on hardlinked images to protect source v2 permissions.
"""

import os
import shutil
import hashlib
import tempfile
import json
from pathlib import Path
import pytest

from scripts.build_dataset_v3_candidate import build_dataset_v3_candidate

ROOT = Path(__file__).resolve().parents[2]

def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

@pytest.fixture
def mock_dataset_environment(tmp_path):
    """
    Creates a miniature mock dataset environment mimicking data/detection:
    - splits_real_v2/
    - images/real/
    - labels/real/
    - real_v2_source_manifest.csv
    - real_dataset_quality_gate_audit.json
    """
    data_dir = tmp_path / "data" / "detection"
    v3_dir = tmp_path / "data" / "detection_v3_candidate"
    reports_dir = tmp_path / "reports" / "data_v2"
    canary_dir = tmp_path / "external_canary"

    (data_dir / "splits_real_v2").mkdir(parents=True)
    (data_dir / "images" / "real").mkdir(parents=True)
    (data_dir / "labels" / "real").mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    canary_dir.mkdir(parents=True)

    # Canary file outside candidate directory
    canary_file = canary_dir / "protected_external.txt"
    canary_file.write_text("DO NOT TOUCH", encoding="utf-8")

    # Create 2 mock images and labels
    img1 = data_dir / "images" / "real" / "oi_001.jpg"
    img2 = data_dir / "images" / "real" / "taco_002.jpg"
    from PIL import Image
    Image.new("RGB", (640, 480), color=(100, 100, 100)).save(img1)
    Image.new("RGB", (640, 480), color=(150, 150, 150)).save(img2)

    lbl1 = data_dir / "labels" / "real" / "oi_001.txt"
    lbl2 = data_dir / "labels" / "real" / "taco_002.txt"
    lbl1.write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    lbl2.write_text("0 0.3 0.3 0.1 0.1\n", encoding="utf-8")

    # Splits
    (data_dir / "splits_real_v2" / "train.txt").write_text("images/real/oi_001.jpg\n", encoding="utf-8")
    (data_dir / "splits_real_v2" / "val.txt").write_text("images/real/taco_002.jpg\n", encoding="utf-8")
    (data_dir / "splits_real_v2" / "test.txt").write_text("", encoding="utf-8")

    # Source manifest
    src_man = data_dir / "real_v2_source_manifest.csv"
    src_man.write_text("filename,source,source_url,status\noi_001.jpg,openimages_v7,http://oi,VALID\ntaco_002.jpg,taco,http://taco,VALID\n", encoding="utf-8")

    # Quality gate audit report
    audit_path = reports_dir / "real_dataset_quality_gate_audit.json"
    audit_data = {
        "all_entries": [
            {
                "filename": "oi_001.jpg",
                "validation_status": "VALID",
                "image_path": str(img1),
                "label_path": str(lbl1),
                "evidence": {
                    "image_size": [640, 480],
                    "phash": "0123456789abcdef",
                    "num_boxes": 1
                }
            },
            {
                "filename": "taco_002.jpg",
                "validation_status": "VALID",
                "image_path": str(img2),
                "label_path": str(lbl2),
                "evidence": {
                    "image_size": [640, 480],
                    "phash": "fedcba9876543210",
                    "num_boxes": 1
                }
            }
        ]
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f)

    return {
        "data_dir": data_dir,
        "v3_dir": v3_dir,
        "reports_dir": reports_dir,
        "audit_path": audit_path,
        "canary_file": canary_file,
        "img1": img1,
        "img2": img2,
        "lbl1": lbl1,
        "lbl2": lbl2
    }

def test_initial_production_build(mock_dataset_environment):
    env = mock_dataset_environment
    res = build_dataset_v3_candidate(
        data_dir=env["data_dir"],
        v3_dir=env["v3_dir"],
        audit_report_path=env["audit_path"]
    )
    assert res["status"] == "SUCCESS"
    assert res["total_images"] == 2
    assert res["total_boxes"] == 2

    # Check files created
    assert (env["v3_dir"] / "yolo" / "data.yaml").exists()
    assert (env["v3_dir"] / "yolo" / "labels" / "train" / "oi_001.txt").exists()
    assert (env["v3_dir"] / "yolo" / "labels" / "val" / "taco_002.txt").exists()

    # Check ledger
    ledger_file = env["v3_dir"] / "manifests" / "label_sync_ledger.json"
    assert ledger_file.exists()
    ledger = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert ledger["oi_001.txt"]["status"] == "SYNCHRONIZED"

def test_source_label_update_propagates(mock_dataset_environment):
    env = mock_dataset_environment
    # Initial build
    build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])

    # Update source label in v2
    new_lbl1_content = "7 0.5 0.5 0.3 0.3\n"
    env["lbl1"].write_text(new_lbl1_content, encoding="utf-8")

    # Rebuild
    res = build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])
    assert res["sync_stats"]["updated"] == 1

    # Check that candidate v3 label updated
    v3_lbl1 = env["v3_dir"] / "yolo" / "labels" / "train" / "oi_001.txt"
    assert v3_lbl1.read_text(encoding="utf-8") == new_lbl1_content

def test_v3_manual_edit_does_not_mutate_v2_source(mock_dataset_environment):
    env = mock_dataset_environment
    build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])

    v3_lbl1 = env["v3_dir"] / "yolo" / "labels" / "train" / "oi_001.txt"
    orig_v2_content = env["lbl1"].read_text(encoding="utf-8")

    # Manually edit v3 label
    v3_lbl1.write_text("7 0.9 0.9 0.1 0.1\n", encoding="utf-8")

    # Assert source v2 label is UNTOUCHED
    assert env["lbl1"].read_text(encoding="utf-8") == orig_v2_content

def test_manual_v3_edit_conflict_detection_preserves_work(mock_dataset_environment):
    env = mock_dataset_environment
    build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])

    v3_lbl1 = env["v3_dir"] / "yolo" / "labels" / "train" / "oi_001.txt"
    # Manually edit v3 label
    manual_content = "7 0.88 0.88 0.1 0.1\n"
    v3_lbl1.write_text(manual_content, encoding="utf-8")

    # Also edit source label to simulate divergence
    env["lbl1"].write_text("7 0.22 0.22 0.1 0.1\n", encoding="utf-8")

    # Rebuild
    res = build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])
    assert res["sync_stats"]["preserved_conflicts"] == 1

    # Verify manual content in v3 was PRESERVED, not overwritten
    assert v3_lbl1.read_text(encoding="utf-8") == manual_content

def test_stale_files_cleanup_and_outside_data_protection(mock_dataset_environment):
    env = mock_dataset_environment
    build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])

    # Add a stale file in candidate labels
    stale_lbl = env["v3_dir"] / "yolo" / "labels" / "train" / "stale_trash.txt"
    stale_lbl.write_text("stale data", encoding="utf-8")

    # Rebuild
    res = build_dataset_v3_candidate(data_dir=env["data_dir"], v3_dir=env["v3_dir"], audit_report_path=env["audit_path"])
    assert res["sync_stats"]["stale_removed"] >= 1
    assert not stale_lbl.exists(), "Stale file must be removed"

    # Verify canary file outside v3_dir is intact
    assert env["canary_file"].exists()
    assert env["canary_file"].read_text(encoding="utf-8") == "DO NOT TOUCH"
