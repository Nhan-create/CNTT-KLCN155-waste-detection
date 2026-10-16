"""
tests/qa/test_split_contamination_safety.py
-------------------------------------------
QA Test Suite guaranteeing mathematical split safety and leak prevention:
1. Proves taco_0068.jpg and taco_0082.jpg (TEST split) can NEVER be exported to train.
2. Adversarial Test: Proves that even if metadata is spoofed to 'EXISTING_TRAIN',
   authoritative split cross-referencing strictly catches and blocks test images.
3. Proves the 8 unmapped Category 51 (rope & strings) empty labels can NEVER enter train.
4. Cryptographic Record Test: Proves quarantined annotations (taco_0072, taco_0092)
   require authentic approval records matching label SHA-256; boolean bypasses are rejected.
5. Proves naive training ingestion of data/detection_pilot_v3 is strictly blocked by quality gates.
6. Proves official train and test splits in splits_real_v2 have zero intersection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from src.detection.export_train_manifest import (
    validate_image_eligibility,
    filter_manifest_for_train,
    export_train_manifest,
    load_authoritative_splits,
    compute_file_sha256,
    SplitContaminationError,
    UnmappedWasteLeakageError,
    PendingApprovalError,
    ApprovalRecordVerificationError,
    UnverifiedSplitMembershipError,
    UnverifiedTrueNegativeError,
    MissingLabelVerificationError,
    TrainExportError,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT_MANIFEST_PATH = ROOT / "data" / "detection_pilot_v3" / "manifests" / "pilot_image_manifest.json"
PILOT_IMG_DIR = ROOT / "data" / "detection_pilot_v3" / "images"
PILOT_LBL_DIR = ROOT / "data" / "detection_pilot_v3" / "labels"
OUT_ELIGIBLE_PATH = ROOT / "reports" / "data_v2" / "pilot_train_eligible_manifest.json"


@pytest.fixture
def pilot_manifest() -> dict:
    assert PILOT_MANIFEST_PATH.is_file(), f"Missing {PILOT_MANIFEST_PATH}"
    with open(PILOT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_test_images_strictly_blocked_from_train_export(pilot_manifest):
    """Verify taco_0068 and taco_0082 trigger SplitContaminationError."""
    test_files = ["taco_0068.jpg", "taco_0082.jpg"]
    for fn in test_files:
        img_entry = next((im for im in pilot_manifest["images"] if im["filename"] == fn), None)
        assert img_entry is not None, f"Image {fn} not found in pilot manifest"

        is_ok, reason, err_type = validate_image_eligibility(img_entry)
        assert not is_ok
        assert err_type == "SPLIT_CONTAMINATION"
        assert "TEST split" in reason

        synthetic_manifest = {"images": [img_entry]}
        with pytest.raises(SplitContaminationError) as exc_info:
            filter_manifest_for_train(synthetic_manifest, raise_on_violation=True)
        assert fn in str(exc_info.value)
        assert "TEST split" in str(exc_info.value)


def test_adversarial_metadata_spoofing_blocked_by_authoritative_splits():
    """
    Adversarial Security Test:
    Intentionally spoof manifest metadata by labeling taco_0068.jpg and taco_0082.jpg
    as 'EXISTING_TRAIN' with 'APPROVED_TRAIN'.
    Verifies that the validator inspects the disk splits_real_v2/test.txt and blocks them.
    """
    for spoofed_fn in ["taco_0068.jpg", "taco_0082.jpg"]:
        spoofed_meta = {
            "filename": spoofed_fn,
            "width": 640,
            "height": 480,
            "category": "EXISTING_TRAIN",            # MALICIOUS SPOOF
            "official_split": "train",                # MALICIOUS SPOOF
            "qa_approval_status": "APPROVED_TRAIN",   # MALICIOUS SPOOF
            "boxes_count": 2,
            "is_hard_negative": False
        }

        # Must fail closed because fn is present in authoritative test.txt
        is_ok, reason, err_type = validate_image_eligibility(spoofed_meta)
        assert not is_ok, f"Adversarial spoof succeeded on {spoofed_fn}!"
        assert err_type == "SPLIT_CONTAMINATION"
        assert "authoritative TEST split" in reason or "splits_real_v2/test.txt" in reason

        with pytest.raises(SplitContaminationError):
            filter_manifest_for_train({"images": [spoofed_meta]}, raise_on_violation=True)


def test_unmapped_waste_images_strictly_blocked_from_train_export(pilot_manifest):
    """Verify 8 unmapped 0-byte images trigger UnmappedWasteLeakageError."""
    rope_stems = [
        "taco_0170", "taco_0431", "taco_0556", "taco_0909",
        "taco_0912", "taco_0916", "taco_0917", "taco_0925"
    ]
    for stem in rope_stems:
        fn = f"{stem}.jpg"
        img_entry = next((im for im in pilot_manifest["images"] if im["filename"] == fn), None)
        assert img_entry is not None, f"Image {fn} not found in pilot manifest"

        is_ok, reason, err_type = validate_image_eligibility(img_entry)
        assert not is_ok
        assert err_type == "UNMAPPED_WASTE_LEAKAGE"
        assert "Category 51" in reason or "unmapped waste" in reason

        synthetic_manifest = {"images": [img_entry]}
        with pytest.raises(UnmappedWasteLeakageError) as exc_info:
            filter_manifest_for_train(synthetic_manifest, raise_on_violation=True)
        assert fn in str(exc_info.value)


def test_proposed_biological_reannotations_require_valid_approval_record(pilot_manifest):
    """
    Verify that proposed biological annotations fail closed without an explicit approval record,
    and that approval records require matching label SHA-256.
    """
    pending_files = ["taco_0072.jpg", "taco_0092.jpg"]
    for fn in pending_files:
        img_entry = next((im for im in pilot_manifest["images"] if im["filename"] == fn), None)
        assert img_entry is not None, f"Image {fn} not found in pilot manifest"

        # 1. No approval ledger provided -> FAILS CLOSED
        is_ok, reason, err_type = validate_image_eligibility(img_entry)
        assert not is_ok
        assert err_type == "PENDING_APPROVAL"

        with pytest.raises(PendingApprovalError):
            filter_manifest_for_train({"images": [img_entry]}, raise_on_violation=True)

        # 2. Approval ledger with decision != APPROVED -> FAILS CLOSED
        rejected_ledger = {fn: {"decision": "REJECTED", "approved_by": "lead_qa"}}
        is_ok_rej, _, err_rej = validate_image_eligibility(img_entry, approval_ledger=rejected_ledger)
        assert not is_ok_rej
        assert err_rej == "INVALID_APPROVAL_RECORD"

        # 3. Approval ledger with mismatched SHA-256 -> FAILS CLOSED
        lbl_file = PILOT_LBL_DIR / f"{Path(fn).stem}.txt"
        assert lbl_file.is_file()
        mismatched_ledger = {
            fn: {
                "decision": "APPROVED",
                "label_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "approved_by": "lead_qa"
            }
        }
        is_ok_badhash, _, err_badhash = validate_image_eligibility(
            img_entry,
            approval_ledger=mismatched_ledger,
            labels_dir=PILOT_LBL_DIR
        )
        assert not is_ok_badhash
        assert err_badhash == "INVALID_APPROVAL_RECORD"

        # 4. Authentic Approval Record with matching SHA-256 -> SUCCEEDS
        actual_hash = compute_file_sha256(lbl_file)
        valid_ledger = {
            fn: {
                "decision": "APPROVED",
                "label_sha256": actual_hash,
                "approved_by": "lead_qa",
                "timestamp_utc": "2026-10-12T06:00:00Z"
            }
        }
        is_ok_valid, reason_valid, err_valid = validate_image_eligibility(
            img_entry,
            approval_ledger=valid_ledger,
            labels_dir=PILOT_LBL_DIR
        )
        assert is_ok_valid
        assert err_valid is None
        assert "lead_qa" in reason_valid


def test_naive_pilot_directory_loading_blocked():
    """Verify that pilot dataset validation reports TRAINING_BLOCKED."""
    report_path = ROOT / "reports" / "data_v2" / "pilot_quality_gates_report.json"
    assert report_path.is_file(), f"Missing report at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["verdict"] == "TRAINING_BLOCKED"
    assert report["gates"]["split_safety_gate"]["passed"] is False
    assert len(report["gates"]["split_safety_gate"]["violations"]) == 2
    assert report["gates"]["semantic_negative_gate"]["passed"] is False
    assert len(report["gates"]["semantic_negative_gate"]["violations"]) == 8
    assert report["gates"]["approval_status_gate"]["passed"] is False
    assert len(report["gates"]["approval_status_gate"]["violations"]) == 2


def test_pilot_train_eligible_manifest_contains_zero_test_or_unmapped_images(tmp_path, pilot_manifest):
    """Verify exported clean train manifest contains only verified train images."""
    out_file = tmp_path / "test_eligible.json"
    res = export_train_manifest(PILOT_MANIFEST_PATH, out_file)

    assert res["metadata"]["eligible_train_images_count"] == 3
    assert res["metadata"]["blocked_images_count"] == 12

    with open(out_file, "r", encoding="utf-8") as f:
        exported_data = json.load(f)

    exported_filenames = {img["filename"] for img in exported_data["images"]}
    assert "taco_0068.jpg" not in exported_filenames
    assert "taco_0082.jpg" not in exported_filenames
    for stem in ["taco_0170", "taco_0431", "taco_0556", "taco_0909", "taco_0912", "taco_0916", "taco_0917", "taco_0925"]:
        assert f"{stem}.jpg" not in exported_filenames


def test_official_splits_contain_zero_test_contamination():
    """Verify that official train and test split lists in splits_real_v2 have zero overlap."""
    splits_dir = ROOT / "data" / "detection" / "splits_real_v2"
    assert splits_dir.is_dir(), f"Missing splits dir at {splits_dir}"

    train_file = splits_dir / "train.txt"
    test_file = splits_dir / "test.txt"
    assert train_file.exists() and test_file.exists()

    train_lines = {Path(ln.strip()).name for ln in train_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    test_lines = {Path(ln.strip()).name for ln in test_file.read_text(encoding="utf-8").splitlines() if ln.strip()}

    intersection = train_lines.intersection(test_lines)
    assert len(intersection) == 0, f"Contamination detected in official splits: {intersection}"
    assert "taco_0068.jpg" in test_lines
    assert "taco_0082.jpg" in test_lines
    assert "taco_0068.jpg" not in train_lines
    assert "taco_0082.jpg" not in train_lines


def test_bypass_1_unverified_split_membership_blocked():
    """
    P0.1 Regression Test 1:
    Image not found in authoritative splits but metadata self-declares EXISTING_TRAIN.
    Must fail closed with UNVERIFIED_SPLIT_MEMBERSHIP.
    """
    phantom_meta = {
        "filename": "phantom_untracked_image_9999.jpg",
        "category": "EXISTING_TRAIN",
        "qa_approval_status": "APPROVED_TRAIN",
        "official_split": "train",
        "boxes_count": 1
    }
    is_ok, reason, err_type = validate_image_eligibility(phantom_meta)
    assert not is_ok
    assert err_type == "UNVERIFIED_SPLIT_MEMBERSHIP"
    assert "authoritative train split" in reason

    with pytest.raises(UnverifiedSplitMembershipError):
        filter_manifest_for_train({"images": [phantom_meta]}, raise_on_violation=True)


def test_bypass_2_unverified_true_negative_without_qa_record_blocked():
    """
    P0.1 Regression Test 2:
    Metadata self-declares VERIFIED_TRUE_NEGATIVE without QA approval record evidence.
    Must fail closed with UNVERIFIED_TRUE_NEGATIVE.
    """
    negative_meta = {
        "filename": "clean_sidewalk_001.jpg",
        "category": "VERIFIED_TRUE_NEGATIVE",
        "qa_approval_status": "VERIFIED_TRUE_NEGATIVE",
        "is_hard_negative": True,
        "boxes_count": 0
    }
    # No approval ledger provided
    is_ok, reason, err_type = validate_image_eligibility(negative_meta)
    assert not is_ok
    assert err_type == "UNVERIFIED_TRUE_NEGATIVE"
    assert "lacks QA approval record" in reason

    with pytest.raises(UnverifiedTrueNegativeError):
        filter_manifest_for_train({"images": [negative_meta]}, raise_on_violation=True)


def test_bypass_3_missing_labels_dir_verification_blocked():
    """
    P0.1 Regression Test 3:
    Approval ledger contains APPROVED + valid label_sha256, but labels_dir is not provided.
    Must fail closed with MISSING_LABEL_VERIFICATION.
    """
    fn = "taco_0072.jpg"
    valid_hash = "a" * 64
    ledger = {
        fn: {
            "decision": "APPROVED",
            "label_sha256": valid_hash,
            "approved_by": "senior_lead_qa"
        }
    }
    pending_meta = {
        "filename": fn,
        "category": "QUARANTINED",
        "qa_approval_status": "PENDING_USER_APPROVAL",
        "boxes_count": 1
    }
    # labels_dir=None -> MUST FAIL CLOSED
    is_ok, reason, err_type = validate_image_eligibility(
        pending_meta,
        approval_ledger=ledger,
        labels_dir=None
    )
    assert not is_ok
    assert err_type == "MISSING_LABEL_VERIFICATION"
    assert "labels_dir was not provided" in reason

    with pytest.raises(MissingLabelVerificationError):
        filter_manifest_for_train(
            {"images": [pending_meta]},
            approval_ledger=ledger,
            labels_dir=None,
            raise_on_violation=True
        )


def test_missing_splits_or_holdout_fails_closed(tmp_path):
    """
    P0.1 Regression Test 4:
    Missing split files or missing holdout manifest MUST fail closed by raising TrainExportError.
    """
    empty_splits = tmp_path / "empty_splits"
    empty_splits.mkdir()
    fake_holdout = tmp_path / "fake_holdout.json"

    # Non-existent holdout
    with pytest.raises(TrainExportError) as exc_info:
        load_authoritative_splits(splits_dir=empty_splits, holdout_manifest_path=fake_holdout)
    assert "missing" in str(exc_info.value).lower()

