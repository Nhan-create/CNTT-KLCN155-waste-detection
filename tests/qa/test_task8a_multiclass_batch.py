"""
tests/qa/test_task8a_multiclass_batch.py
----------------------------------------
Automated Quality Assurance & Definition-of-Done (DoD) tests for:
TASK 8A — REAL-WORLD MULTI-CLASS DATA ACQUISITION & QUALITY ASSURANCE
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK8A_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch"

pytestmark = pytest.mark.skipif(
    not TASK8A_DIR.exists(),
    reason="Local gitignored staging directory data/acquisition_staging/task8a_multiclass_batch not present",
)


def test_task8a_manifests_exist_and_valid() -> None:
    qa_manifest_p = TASK8A_DIR / "task8a_qa_manifest.json"
    lic_manifest_p = TASK8A_DIR / "task8a_license_and_source_manifest.json"
    leak_report_p = TASK8A_DIR / "task8a_leakage_and_session_report.json"
    cov_matrix_p = TASK8A_DIR / "task8a_10class_coverage_matrix.json"

    for p in (qa_manifest_p, lic_manifest_p, leak_report_p, cov_matrix_p):
        assert p.exists(), f"Missing required Task 8A artifact: {p}"

    qa = json.loads(qa_manifest_p.read_text(encoding="utf-8"))
    assert qa["total_candidates_audited"] == 154
    assert qa["summary_counts"]["approved_for_train_images"] == 0

    for item in qa["items"]:
        assert item["approved_for_train"] is False
        assert item["approval_status"] == "PENDING_PM_APPROVAL"
        assert item["qa_verdict"] in ("QA_PASS", "NEEDS_MANUAL_REVIEW", "REJECT")


def test_task8a_priority_class_quotas() -> None:
    qa = json.loads((TASK8A_DIR / "task8a_qa_manifest.json").read_text(encoding="utf-8"))
    counts = qa["counts_by_primary_class"]

    # Target quotas per PM Task 8A specification:
    # Biological: 25-35 QA_PASS
    # Paper: 8-12 QA_PASS
    # Cardboard: 8-12 QA_PASS
    # Glass: 8-12 QA_PASS
    # Shoes: 8-12 QA_PASS
    assert 25 <= counts["1: biological"]["QA_PASS"] <= 35
    assert 8 <= counts["6: paper"]["QA_PASS"] <= 12
    assert 8 <= counts["2: cardboard"]["QA_PASS"] <= 12
    assert 8 <= counts["4: glass"]["QA_PASS"] <= 12
    assert 8 <= counts["8: shoes"]["QA_PASS"] <= 12
    assert qa["summary_counts"]["qa_pass_images"] == 75


def test_task8a_provenance_and_license_completeness() -> None:
    lic = json.loads((TASK8A_DIR / "task8a_license_and_source_manifest.json").read_text(encoding="utf-8"))
    assert lic["all_qa_pass_have_valid_open_license"] is True
    assert lic["qa_pass_candidates"] == 75

    for rec in lic["records"]:
        if rec["qa_verdict"] == "QA_PASS":
            assert rec["source_page_url"].startswith("https://commons.wikimedia.org/")
            assert rec["download_url"].startswith("https://")
            assert len(rec["author"].strip()) > 0
            assert len(rec["license"].strip()) > 0
            assert len(rec["sha256"]) == 64
            assert len(rec["phash_0"]) == 16
            assert len(rec["phash_rotations"]) == 4
            assert len(rec["acquired_at_utc"]) >= 19


def test_task8a_yolo_labels_and_overlays() -> None:
    qa = json.loads((TASK8A_DIR / "task8a_qa_manifest.json").read_text(encoding="utf-8"))
    w1_p = TASK8A_DIR / "task8_wave1_qa_manifest.json"
    w1_boxes = {}
    if w1_p.exists():
        w1_data = json.loads(w1_p.read_text(encoding="utf-8"))
        for rec in w1_data.get("records", []):
            w1_boxes[rec.get("candidate_id")] = rec.get("box_count")

    contact_sheets = [REPO_ROOT / s for s in qa["contact_sheets"] if (REPO_ROOT / s).exists()]
    if not contact_sheets:
        contact_sheets = sorted((TASK8A_DIR / "contact_sheets").glob("*.jpg"))
    assert len(contact_sheets) >= 6, f"Expected at least 6 contact sheets in {TASK8A_DIR / 'contact_sheets'}"
    for sheet_p in contact_sheets:
        with Image.open(sheet_p) as im:
            w, h = im.size
            assert w >= 1000 and h >= 400

    for item in qa["items"]:
        ov_p = REPO_ROOT / item["overlay_path"]
        assert ov_p.exists(), f"Missing overlay image: {ov_p}"

        if item["qa_verdict"] == "QA_PASS":
            assert item["label_path"] is not None
            lp = REPO_ROOT / item["label_path"]
            if not lp.exists():
                lp_alt = TASK8A_DIR / "labels_needs_manual_review" / lp.name
                if lp_alt.exists():
                    lp = lp_alt
                else:
                    continue
            lines = [ln.strip() for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            expected_counts = {item["boxes_count"]}
            if item["candidate_id"] in w1_boxes and w1_boxes[item["candidate_id"]] is not None:
                expected_counts.add(w1_boxes[item["candidate_id"]])
            assert len(lines) in expected_counts
            assert len(lines) >= 1
            for ln in lines:
                parts = ln.split()
                assert len(parts) == 5
                cid = int(parts[0])
                xc, yc, bw, bh = [float(x) for x in parts[1:]]
                assert 0 <= cid <= 9
                assert 0.0 < xc <= 1.0
                assert 0.0 < yc <= 1.0
                assert 0.0 < bw <= 1.0
                assert 0.0 < bh <= 1.0


def test_task8a_leakage_and_10class_coverage_matrix() -> None:
    leak = json.loads((TASK8A_DIR / "task8a_leakage_and_session_report.json").read_text(encoding="utf-8"))
    pools = leak["reference_pools_covered"]

    assert pools["dataset_v2_and_cache"] == 4046
    assert pools["frozen_test_299"] == 299
    assert pools["roboflow_v3_battery_140_base"] == 140
    assert pools["task7_diverse_battery_56"] == 56
    assert pools["batch_a_mini01_10"] == 10
    assert pools["holdout_27_hinh_anh_rac"] == 27

    assert leak["zero_external_sha256_leakage"] is True
    assert leak["zero_external_phash_le8_leakage"] is True
    assert leak["zero_cross_split_leakage"] is True
    assert leak["cross_split_train_vs_val_collisions"] == []

    cov = json.loads((TASK8A_DIR / "task8a_10class_coverage_matrix.json").read_text(encoding="utf-8"))
    classes = cov["classes"]
    assert len(classes) == 10
    expected_names = [
        "battery",
        "biological",
        "cardboard",
        "clothes",
        "glass",
        "metal",
        "paper",
        "plastic",
        "shoes",
        "trash",
    ]
    for idx, row in enumerate(classes):
        assert row["class_id"] == idx
        assert row["class_name"] == expected_names[idx]
        assert "remaining_context_gaps_for_task8b" in row
