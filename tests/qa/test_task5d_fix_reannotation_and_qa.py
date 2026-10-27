# Mô tả: Kiểm thử bản sửa lỗi gán nhãn lại lô 20 ảnh pin (Mini-Pilot 20) và chặn rò rỉ biến thể tăng cường qua các phiên.
"""
tests/qa/test_task5d_fix_reannotation_and_qa.py
-----------------------------------------------
Regression test suite for Sprint AI-UPGRADE-01 Task 5D Fix (Battery Object Annotation & QA State Machine):
1. Verifies the 20-image re-annotation mini-pilot (1 full-battery bbox per physical battery, 20 BEFORE/AFTER overlays, non-constant scaling proof).
2. Verifies augmentation & capture-session analysis (140 base_id groups, ORIGINAL_UNKNOWN, 1 representative per base_id, 0 specimen overlap in recommended leak-free split).
3. Verifies QA state machine (TECHNICAL_VALID vs VISUAL_QA_PASS vs NEEDS_REANNOTATION vs REJECT vs APPROVED_FOR_TRAIN; blocks VISUAL_QA_PASS without review_evidence; blocks automatic QA_PASS on raw import).
4. Verifies test split per-class box reconciliation (trash=139, sum=720).
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from PIL import Image
import yaml
import pytest

from src.detection.safe_dataset_builder import (
    validate_candidate_for_intake,
    verify_visual_qa_pass_evidence,
    validate_on_disk_sample,
    build_safe_pilot_dataset_staging,
)
from scripts.import_and_qa_roboflow_battery import process_roboflow_battery_archive

ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
REANN_DIR = STAGING_DIR / "reannotated_v1_minipilot20"
REANN_MANIFEST = REANN_DIR / "minipilot20_reannotation_manifest.json"
SESSION_ANALYSIS = STAGING_DIR / "augmentation_and_session_leakage_analysis.json"
BASE_MANIFEST = STAGING_DIR / "base_images_qa_manifest.json"
PILOT_DATASET_YAML = ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml"


def test_minipilot20_reannotation_manifest_and_on_disk_assets():
    """Verify 20 re-annotated images, labels, and side-by-side BEFORE/AFTER overlays."""
    assert REANN_MANIFEST.exists(), f"Missing {REANN_MANIFEST}"
    with open(REANN_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_reannotated_images"] == 20
    assert data["new_total_boxes"] == 20
    assert data["old_total_boxes"] == 41
    assert data["approved_for_train_count"] in (0, 20)

    items = data["items"]
    assert len(items) == 20

    # Must cover both train and valid
    splits = {it["source_split"] for it in items}
    assert splits == {"train", "valid"}

    # Must include PM's 3 specific examples
    pm_required = {
        "image-100-_png.rf.28fd8f243c7008b956a660fc9ffbb8d5.jpg",
        "image-2024-08-03T171736-085_png.rf.9fb8b6849dda134195cc731853cd4fbb.jpg",
        "image-2024-08-03T182055-514_png.rf.d231703a4a5f589d4d80b134bc232614.jpg",
    }
    present_fns = {it["filename"] for it in items}
    assert pm_required.issubset(present_fns)

    # Must cover 1, 2, 3, and 4 old boxes on 1 physical battery
    old_box_counts = {it["old_num_boxes"] for it in items}
    assert {1, 2, 3, 4}.issubset(old_box_counts)

    # Must cover all 3 source classes (0: Alkaline, 1: Alkaline_jp, 2: LR_Alkaline)
    src_classes = {ob["source_class_id"] for it in items for ob in it["old_boxes"]}
    assert src_classes == {0, 1, 2}

    # Verify non-constant scaling: area ratios and width/height ratios vary across physical batteries
    area_ratios = {it["old_to_new_area_ratio"] for it in items}
    assert len(area_ratios) >= 15, "Boxes must be individually measured, not scaled by a constant factor"

    for it in items:
        assert it["new_num_boxes"] == 1
        assert it["technical_status"] == "TECHNICAL_VALID"
        assert it["visual_qa"] == "VISUAL_QA_PASS"
        assert it["approved_for_train"] in (False, True)
        ev_ok, ev_err = verify_visual_qa_pass_evidence(it)
        assert ev_ok, ev_err

        img_p = STAGING_DIR / Path(it["image_path"])
        lbl_p = STAGING_DIR / Path(it["label_path"])
        ov_p = STAGING_DIR / Path(it["before_after_overlay_path"])

        assert img_p.exists()
        assert lbl_p.exists()
        assert ov_p.exists()

        # Validate YOLO label geometry on disk
        validate_on_disk_sample(img_p, lbl_p, expected_boxes=1, is_clean_negative=False)

        # Ensure full-battery bbox covers a substantial fraction of the 768x768 frame (> 8%)
        _, _, bw, bh = it["new_box"]["yolo_bbox"]
        assert bw * bh > 0.08, f"Full-battery box too small for {it['filename']}: {bw*bh}"

        # Verify side-by-side BEFORE/AFTER overlay dimensions
        with Image.open(ov_p) as oim:
            assert oim.width >= 1500 and oim.height >= 800


def test_augmentation_and_session_leakage_analysis():
    """Verify Task 3 grouping by base_id, ORIGINAL_UNKNOWN status, and specimen-isolated split."""
    assert SESSION_ANALYSIS.exists(), f"Missing {SESSION_ANALYSIS}"
    with open(SESSION_ANALYSIS, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    findings = analysis["roboflow_export_findings"]
    assert findings["total_files"] == 350
    assert findings["unique_base_ids"] == 140
    assert findings["raw_train_base_ids"] == 105
    assert findings["raw_valid_base_ids"] == 35

    groups = analysis["base_id_groups"]
    assert len(groups) == 140

    rep_count = 0
    excl_count = 0
    train_specimens = set()
    valid_specimens = set()

    for bg in groups:
        assert bg["original_status"] == "ORIGINAL_UNKNOWN"
        reps_in_group = [v for v in bg["variants"] if v["is_base_representative"] is True]
        assert len(reps_in_group) == 1
        for v in bg["variants"]:
            assert v["original_status"] == "ORIGINAL_UNKNOWN"
            if v["is_base_representative"]:
                assert v["variant_role"] == "BASE_REPRESENTATIVE"
                rep_count += 1
            else:
                assert v["variant_role"] == "AUGMENTED_VARIANT_EXCLUDED"
                excl_count += 1

        if bg["recommended_leak_free_split"] == "train":
            train_specimens.add(bg["specimen_cluster_id"])
        else:
            valid_specimens.add(bg["specimen_cluster_id"])

    assert rep_count == 140
    assert excl_count == 210
    # Zero physical battery specimen overlap between recommended train and valid splits
    assert train_specimens.isdisjoint(valid_specimens), (
        f"Specimen leakage detected between train and valid: {train_specimens & valid_specimens}"
    )


def test_qa_state_machine_blocks_unreviewed_and_needs_reannotation():
    """Verify QA state machine blocks NEEDS_REANNOTATION, TECHNICAL_VALID, and VISUAL_QA_PASS without review_evidence."""
    # 1. NEEDS_REANNOTATION must be blocked even if approval flags are True
    needs_reann_item = {
        "candidate_id": "raw_text_box_item",
        "num_boxes": 2,
        "technical_status": "TECHNICAL_VALID",
        "visual_qa": "NEEDS_REANNOTATION",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(needs_reann_item, require_approval=False)
    assert not ok
    assert "NEEDS_REANNOTATION" in reason

    # 2. TECHNICAL_VALID alone is NOT a visual QA pass
    tech_only_item = {
        "candidate_id": "tech_only_item",
        "num_boxes": 1,
        "technical_status": "TECHNICAL_VALID",
        "visual_qa": "TECHNICAL_VALID",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(tech_only_item, require_approval=False)
    assert not ok
    assert "TECHNICAL_VALID" in reason

    # 3. VISUAL_QA_PASS without review_evidence must be blocked
    fake_vqa_item = {
        "candidate_id": "unreviewed_auto_pass",
        "num_boxes": 1,
        "technical_status": "TECHNICAL_VALID",
        "visual_qa": "VISUAL_QA_PASS",
        "review_evidence": None,
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(fake_vqa_item, require_approval=False)
    assert not ok
    assert "review_evidence" in reason

    # 4. VISUAL_QA_PASS with incomplete review_evidence must be blocked
    partial_vqa_item = {
        "candidate_id": "partial_review_item",
        "num_boxes": 1,
        "technical_status": "TECHNICAL_VALID",
        "visual_qa": "VISUAL_QA_PASS",
        "review_evidence": {"reviewed_by": "Reviewer1", "review_timestamp": "2026-10-14", "review_note": ""},
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(partial_vqa_item, require_approval=False)
    assert not ok
    assert "review_note" in reason

    # 5. VISUAL_QA_PASS with complete review_evidence and dual approval passes
    valid_vqa_item = {
        "candidate_id": "verified_reannotated_item",
        "num_boxes": 1,
        "technical_status": "TECHNICAL_VALID",
        "visual_qa": "VISUAL_QA_PASS",
        "review_evidence": {
            "reviewed_by": "Human_QA_Reviewer",
            "review_timestamp": "2026-10-14T00:35:00Z",
            "review_note": "Full physical battery + terminals enclosed in 1 tight box.",
        },
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(valid_vqa_item, require_approval=True)
    assert ok, f"Expected valid item to pass, got: {reason}"


def test_importer_never_assigns_qa_pass_by_default(tmp_path: Path):
    """Prove process_roboflow_battery_archive assigns NEEDS_REANNOTATION and ORIGINAL_UNKNOWN by default."""
    zip_p = tmp_path / "raw_import_test.zip"
    staging_target = tmp_path / "staging_raw"

    im = Image.new("RGB", (120, 120), color=(180, 180, 180))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")

    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("data.yaml", "nc: 1\nnames: ['Alkaline']\n")
        zf.writestr("train/images/sample_bat_png.rf.1111111111111111.jpg", buf.getvalue())
        zf.writestr("train/labels/sample_bat_png.rf.1111111111111111.txt", "0 0.5 0.5 0.2 0.2\n")

    manifest = process_roboflow_battery_archive(zip_p, staging_target, root_dir=ROOT)
    assert len(manifest["items"]) == 1
    it = manifest["items"][0]

    assert it["technical_status"] == "TECHNICAL_VALID"
    assert it["original_status"] == "ORIGINAL_UNKNOWN"
    assert it["visual_qa"] == "NEEDS_REANNOTATION"
    assert it["qa_decision"] == "NEEDS_REANNOTATION"
    assert it["review_evidence"] is None
    assert it["approved_for_train"] is False
    assert manifest["qa_state_summary"]["VISUAL_QA_PASS"] == 0
    assert manifest["qa_state_summary"]["NEEDS_REANNOTATION"] == 1


def test_test_split_box_count_reconciliation():
    """Verify test split trash=139 and exact per-class sum=720 in detection_dataset_real_v3_pilot.yaml."""
    assert PILOT_DATASET_YAML.exists()
    with open(PILOT_DATASET_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    meta = cfg["metadata"]
    test_per_class = meta["per_class_boxes"]["test"]
    assert test_per_class["trash"] == 139, f"Expected test.trash=139, got {test_per_class['trash']}"
    assert sum(test_per_class.values()) == 720
    assert meta["split_boxes"]["test"] == 720
    assert sum(meta["per_class_boxes"]["train"].values()) == meta["split_boxes"]["train"] == 3963
    assert sum(meta["per_class_boxes"]["val"].values()) == meta["split_boxes"]["val"] == 901
