"""
tests/qa/test_fail_closed_dataset_intake.py
-------------------------------------------
Regression tests verifying fail-closed dataset intake safety (Sprint AI-UPGRADE-01 Task 5A):

1. Proves QUARANTINED samples (wiki_bat_corroded_screwdriver) are blocked.
2. Proves PENDING_ANNOTATION samples (wiki_bio_food_scraps_bin_09) are blocked.
3. Proves 0-byte labels are NEVER treated as background unless explicitly QA_PASS_CLEAN_NEGATIVE.
4. Proves OBJECT_ONLY white surface samples (oak_hill_05, sidewalk_06) are segregated from in-the-wild discarded waste.
5. Proves blind filesystem directory scanning is strictly blocked by fail-closed guard.
6. Proves unapproved candidates (approved_for_train != True) yield 0 approved training samples.
7. Verifies the license terms and author attribution for wiki_bio_apple_core_01.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.detection.safe_dataset_builder import (
    validate_candidate_for_intake,
    scan_directory_blindly_guard,
    load_and_filter_manifest,
    UnmanifestedScanError,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data/acquisition_staging/batch_a_mini01/manifest.json"


def test_manifest_loads_and_has_10_candidates():
    """Verify manifest exists and contains exactly 10 candidates."""
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data.get("items", [])) == 10
    assert data.get("approved_for_train_count") == 0


def test_quarantined_screwdriver_is_strictly_blocked():
    """Prove wiki_bat_corroded_screwdriver cannot be accepted for dataset intake."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    screwdriver = next(c for c in data["items"] if c["candidate_id"] == "wiki_bat_corroded_screwdriver")
    
    # Even if someone artificially set approved_for_train = True
    screwdriver_tampered = dict(screwdriver)
    screwdriver_tampered["approved_for_train"] = True
    screwdriver_tampered["user_train_approval"] = "APPROVED_FOR_TRAIN"

    ok, reason = validate_candidate_for_intake(screwdriver_tampered, require_approval=False)
    assert not ok
    assert "QUARANTINED" in reason


def test_pending_annotation_bin_is_strictly_blocked():
    """Prove wiki_bio_food_scraps_bin_09 cannot be accepted for dataset intake."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    bin_item = next(c for c in data["items"] if c["candidate_id"] == "wiki_bio_food_scraps_bin_09")

    # Even if artificially approved
    bin_tampered = dict(bin_item)
    bin_tampered["approved_for_train"] = True
    bin_tampered["user_train_approval"] = "APPROVED_FOR_TRAIN"

    ok, reason = validate_candidate_for_intake(bin_tampered, require_approval=False)
    assert not ok
    assert "PENDING" in reason


def test_empty_label_file_not_clean_negative_is_blocked():
    """Prove a 0-box label file with a generic or incomplete status cannot be treated as clean negative."""
    bogus_item = {
        "candidate_id": "test_bogus_empty",
        "num_boxes": 0,
        "visual_qa": "QA_PASS_SOME_INCOMPLETE_ITEM",
        "scene_context": "REAL_DISCARDED_WASTE",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(bogus_item, require_approval=False)
    assert not ok
    assert "not QA_PASS_CLEAN_NEGATIVE" in reason


def test_clean_negatives_accepted_only_when_properly_certified():
    """Prove genuine clean negatives (brick wall, utility box corner) pass QA negative validation."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    brick = next(c for c in data["items"] if c["candidate_id"] == "wiki_neg_brick_wall_clean")
    util = next(c for c in data["items"] if c["candidate_id"] == "wiki_neg_utility_box_corner_05")

    # When approval is waived for QA logic check
    ok_b, _ = validate_candidate_for_intake(brick, require_approval=False)
    ok_u, _ = validate_candidate_for_intake(util, require_approval=False)
    assert ok_b
    assert ok_u

    # When approval is enforced, both are blocked because user hasn't approved them yet
    ok_b_app, reason_b = validate_candidate_for_intake(brick, require_approval=True)
    ok_u_app, reason_u = validate_candidate_for_intake(util, require_approval=True)
    assert not ok_b_app and "lacks explicit human/PM approval" in reason_b
    assert not ok_u_app and "lacks explicit human/PM approval" in reason_u


def test_isolated_object_segregation():
    """Prove white-surface banana peels are segregated from in-the-wild discarded waste."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    oak_hill = next(c for c in data["items"] if c["candidate_id"] == "wiki_bio_banana_peel_oak_hill_05")
    sidewalk = next(c for c in data["items"] if c["candidate_id"] == "wiki_bio_banana_peel_sidewalk_06")

    # For wild discarded waste (allow_isolated_objects=False): rejected
    ok_oak, reason_oak = validate_candidate_for_intake(oak_hill, require_approval=False, allow_isolated_objects=False)
    ok_side, reason_side = validate_candidate_for_intake(sidewalk, require_approval=False, allow_isolated_objects=False)
    assert not ok_oak
    assert "OBJECT_ONLY" in reason_oak
    assert not ok_side
    assert "OBJECT_ONLY" in reason_side

    # When isolated objects are explicitly permitted in a controlled isolated object slice:
    ok_oak_iso, _ = validate_candidate_for_intake(oak_hill, require_approval=False, allow_isolated_objects=True)
    ok_side_iso, _ = validate_candidate_for_intake(sidewalk, require_approval=False, allow_isolated_objects=True)
    assert ok_oak_iso
    assert ok_side_iso


def test_blind_directory_scan_guard():
    """Prove blind directory globbing raises UnmanifestedScanError unconditionally."""
    with pytest.raises(UnmanifestedScanError):
        scan_directory_blindly_guard(ROOT / "data/acquisition_staging/batch_a_mini01/labels")


def test_manifest_filter_yields_zero_approved_samples():
    """Prove filter of current batch_a_mini01 yields exactly 0 accepted training samples."""
    res = load_and_filter_manifest(MANIFEST_PATH, require_approval=True)
    assert res["total_items"] == 10
    assert res["accepted_count"] == 0
    assert res["rejected_count"] == 10


def test_apple_core_license_and_provenance():
    """Verify license terms and author attribution for wiki_bio_apple_core_01."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    apple = next(c for c in data["items"] if c["candidate_id"] == "wiki_bio_apple_core_01")
    assert apple["author"] == "Adrian Zion"
    assert apple["license"] == "Copyrighted free use"
    assert "Apple_core.jpg" in apple["source_page_url"]
    assert apple["num_boxes"] == 1
    assert apple["boxes"][0]["class_id"] == 1  # biological
