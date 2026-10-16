import sys
"""
tests/qa/test_v3_splits_audit.py
--------------------------------
Regression tests for Phase 4B.1 P1 audit of v3 splits, 176 exclusions, 147 quarantine images,
and zero cross-split/holdout leakage.
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "reports" / "data_v2" / "v3_splits_reconciliation_audit.json"

@pytest.fixture(scope="module")
def audit_report():
    if not REPORT_PATH.exists():
        # Run generator script if report not found
        import subprocess
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts" / "audit_v3_splits_and_exclusions.py")
        ], check=True)
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def test_v3_candidate_split_counts(audit_report):
    """Verify v3 candidate splits have exactly 1,820 total images (1277 train, 269 val, 274 test)."""
    assert audit_report["dataset_v3_candidate_total_images"] == 1820
    reconcil = audit_report["splits_reconciliation"]
    assert reconcil["train"]["v3_count"] == 1277
    assert reconcil["val"]["v3_count"] == 269
    assert reconcil["test"]["v3_count"] == 274

def test_excluded_images_reconciliation(audit_report):
    """Verify exactly 176 images are excluded between v2 (1996) and v3 (1820)."""
    assert audit_report["total_excluded_images"] == 176
    assert audit_report["dataset_v2_total_images"] == 1996
    by_split = audit_report["excluded_176_images_summary"]["by_split"]
    assert by_split["train"] == 121
    assert by_split["val"] == 30
    assert by_split["test"] == 25

def test_critical_classes_preserved(audit_report):
    """Verify 0 boxes of battery and biological were removed in v3 candidate."""
    removed = audit_report["excluded_176_images_summary"]["class_boxes_removed"]
    assert removed["battery"] == 0, "Candidate v3 must not drop any battery boxes"
    assert removed["biological"] == 0, "Candidate v3 must not drop any biological boxes"

def test_unsplit_disk_quarantine_reconciliation(audit_report):
    """Verify exactly 147 un-split images on disk, matching quarantine reasons."""
    assert audit_report["total_unsplit_disk_images"] == 147
    reasons = audit_report["unsplit_147_disk_images_summary"]["by_quarantine_reason"]
    assert reasons["oi_human_present_in_use_object"] == 101
    assert reasons["review_status_REJECTED"] == 14
    assert reasons["empty_label"] == 13
    assert reasons["taco_orientation_mismatch"] == 10

def test_zero_leakage_verification(audit_report):
    """Verify zero cross-split SHA-256 collisions, zero holdout collisions, and zero pHash group leaks."""
    leak_info = audit_report["leakage_gates_verification"]
    assert leak_info["cross_split_sha256_collisions"] == 0
    assert leak_info["holdout_sha256_collisions"] == 0
    assert leak_info["cross_split_phash_group_leaks"] == 0
    assert leak_info["verdict"] == "VERIFIED_ZERO_LEAKAGE"
