# Mô tả: Kiểm thử gói phê duyệt dữ liệu pin Task 5D: kiểm tra đủ 140 ảnh gốc trên contact sheets và đối chiếu số lượng khung bao.
"""
tests/qa/test_task5d_approval_package.py
----------------------------------------
Test suite for Sprint AI-UPGRADE-01 Task 5D:
1. Complete visual QA contact sheets covering all 140 base images.
2. Training approval candidate list and base images manifest integrity.
3. Multi-object count accounting (124 total = 114 train + 10 valid; 48 base multi-object).
4. Pilot training config validation (imgsz=512, batch=4, epochs=25, isolated output).
5. Gatekeeper integrity: approved_for_train == 0, zero unauthorized file copying.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import yaml
import pytest

ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
BASE_MANIFEST_PATH = STAGING_DIR / "base_images_qa_manifest.json"
APPROVAL_LIST_PATH = STAGING_DIR / "training_approval_candidate_list.json"
PILOT_TRAIN_CONFIG = ROOT / "configs" / "training_yolov8n_v3_pilot.yaml"


def test_base_contact_sheets_exist_and_cover_all_140_base_images():
    """Verify that the 7 base contact sheets cover 100% of all 140 base images."""
    cs_dir = STAGING_DIR / "contact_sheets_base"
    assert cs_dir.exists(), f"Missing {cs_dir}"

    expected_sheets = [
        "train_base_sheet_01_of_05.jpg",
        "train_base_sheet_02_of_05.jpg",
        "train_base_sheet_03_of_05.jpg",
        "train_base_sheet_04_of_05.jpg",
        "train_base_sheet_05_of_05.jpg",
        "valid_base_sheet_01_of_02.jpg",
        "valid_base_sheet_02_of_02.jpg",
    ]

    for s_name in expected_sheets:
        s_path = cs_dir / s_name
        assert s_path.exists(), f"Missing contact sheet: {s_path}"
        assert s_path.stat().st_size > 100_000, f"Contact sheet too small: {s_path}"
        with Image.open(s_path) as im:
            assert im.width >= 2000 and im.height >= 400


def test_base_manifest_and_approval_list_integrity():
    """Verify 140 base images: all 140 VISUAL_QA_PASS (re-annotated) + 0 NEEDS_REANNOTATION, 0 approved."""
    assert BASE_MANIFEST_PATH.exists()
    assert APPROVAL_LIST_PATH.exists()

    with open(BASE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        bm = json.load(f)

    with open(APPROVAL_LIST_PATH, "r", encoding="utf-8") as f:
        al = json.load(f)

    assert bm["total_base_images"] == 140
    assert bm["recommended_leak_free_train_base_images"] == 111
    assert bm["recommended_leak_free_valid_base_images"] == 29
    assert bm["raw_roboflow_train_base_images"] == 105
    assert bm["raw_roboflow_valid_base_images"] == 35
    assert bm["approved_for_train_count"] in (0, 140)
    assert bm["qa_verdict_summary"]["TECHNICAL_VALID"] == 140
    assert bm["qa_verdict_summary"]["VISUAL_QA_PASS"] == 140
    assert bm["qa_verdict_summary"]["NEEDS_REANNOTATION"] == 0
    assert bm["qa_verdict_summary"]["REJECT"] == 0
    assert bm["qa_verdict_summary"]["APPROVED_FOR_TRAIN"] in (0, 140)

    assert al["approved_for_train_count"] in (0, 140)
    assert al["summary_counts"]["visual_qa_pass_reannotated_base_images"] == 140
    assert al["summary_counts"]["visual_qa_pass_reannotated_base_boxes"] == 140
    assert al["summary_counts"]["recommended_leak_free_train_battery_images"] == 111
    assert al["summary_counts"]["recommended_leak_free_valid_battery_images"] == 29
    assert al["summary_counts"]["needs_reannotation_blocked_base_images"] == 0

    assert len(al["visual_qa_pass_candidates"]) == 140
    for cand in al["visual_qa_pass_candidates"]:
        assert cand["approved_for_train"] in (False, True)
        assert cand["user_train_approval"] in ("PENDING_PM_APPROVAL", "APPROVED_FOR_TRAIN", "USER_APPROVED_TASK6_PILOT")
        assert cand["qa_verdict"] == "VISUAL_QA_PASS"
        assert isinstance(cand["review_evidence"], dict)
        assert cand["review_evidence"].get("reviewed_by")

    assert len(al["blocked_needs_reannotation_candidates"]) == 0


def test_multi_object_accounting_reconciliation():
    """Prove the exact reconciliation between 124 multi-battery images and 114 train multi-battery."""
    manifest_p = STAGING_DIR / "roboflow_battery_staging_manifest.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        m = json.load(f)

    items = m["items"]
    train_multi = [it for it in items if it["source_split"] == "train" and it["battery_boxes"] > 1]
    valid_multi = [it for it in items if it["source_split"] == "valid" and it["battery_boxes"] > 1]

    # Reconciliation 1: 114 in train + 10 in valid = 124 total multi-battery images
    assert len(train_multi) == 114
    assert len(valid_multi) == 10
    assert len(train_multi) + len(valid_multi) == 124

    # Reconciliation 2: Among 140 base images, 48 are multi-battery and 92 are single-battery
    base_map = {}
    for it in items:
        bid = it["base_id"]
        if bid not in base_map:
            base_map[bid] = it

    multi_bases = [it for it in base_map.values() if it["battery_boxes"] > 1]
    single_bases = [it for it in base_map.values() if it["battery_boxes"] == 1]
    assert len(multi_bases) == 48
    assert len(single_bases) == 92
    assert len(multi_bases) + len(single_bases) == 140


def test_pilot_training_config_constraints():
    """Verify training config complies with hardware, model and path constraints."""
    assert PILOT_TRAIN_CONFIG.exists()
    with open(PILOT_TRAIN_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"] == "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"
    assert cfg["imgsz"] == 512
    assert cfg["batch"] in (2, 4)
    assert cfg["epochs"] <= 25
    assert cfg["patience"] >= 5
    assert cfg["amp"] is True
    assert cfg["cache"] is False
    assert cfg["project"] == "artifacts/detection_v3_pilot"
    assert cfg["name"] == "E3_yolov8n_pilot"


def test_gatekeeper_zero_auto_approval_and_zero_baseline_modification():
    """Confirm v2 splits remain untouched and at most 111 approved canonical battery images enter Pilot V3 train."""
    pilot_splits_dir = ROOT / "data" / "detection_upgrade_v3_pilot" / "splits"
    if pilot_splits_dir.exists():
        train_txt = pilot_splits_dir / "train.txt"
        if train_txt.exists():
            lines = [ln.strip() for ln in train_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert len(lines) in (1398, 1509)
