"""
tests/qa/test_task5b_hotfixes_and_intake.py
-------------------------------------------
Unit and regression tests verifying Task 5B hotfixes:
1. Strict fail-closed allowlist in validate_candidate_for_intake:
   - Blocks QA_FAIL, REJECTED, SOURCE_UNVERIFIED, and empty visual_qa.
   - Blocks non-allowlisted positive QA statuses.
   - Requires QA_PASS_CLEAN_NEGATIVE for 0-box candidates.
   - Blocks QUARANTINED and PENDING annotations.
   - Enforces OBJECT_ONLY segregation and dual approval flags.
2. Roboflow Importer & Safe Extraction:
   - Rejects Zip Slip and Zip Bomb archives.
   - Validates data.yaml (nc == len(names), verified battery classes).
   - Validates YOLO bounding box geometry and coordinates bounds.
   - Detects internal duplicates inside archive.
   - Enforces PENDING_VISUAL_QA and approved_for_train=False.
3. Safe Dataset Builder Physical Staging:
   - Returns empty staging cleanly when 0 candidates are approved.
   - Performs physical staging only for approved candidates.
   - Blocks overwriting baseline/frozen datasets.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
import pytest
from PIL import Image
import yaml

from src.detection.safe_dataset_builder import (
    validate_candidate_for_intake,
    build_safe_pilot_dataset_staging,
    validate_on_disk_sample,
    OverwriteBaselineError,
    InvalidAnnotationError,
    InvalidImageError,
)
from scripts.import_and_qa_roboflow_battery import (
    validate_zip_archive_safety,
    safe_extract_untrusted_zip,
    parse_and_validate_data_yaml,
    validate_yolo_bbox,
    process_roboflow_battery_archive,
    ZipSafetyError,
    DataYamlError,
)

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# TASK 1 TESTS: FAIL-CLOSED ALLOWLIST IN validate_candidate_for_intake
# ============================================================================

def test_rejects_empty_visual_qa():
    """Prove validate_candidate_for_intake rejects candidates with empty visual_qa."""
    item = {
        "candidate_id": "test_empty_qa",
        "num_boxes": 1,
        "visual_qa": "",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(item, require_approval=False)
    assert not ok
    assert "empty or missing visual_qa" in reason


def test_rejects_qa_fail_and_rejected_statuses():
    """Prove validate_candidate_for_intake rejects QA_FAIL and REJECTED statuses even with valid boxes and approval."""
    for bad_status in ["QA_FAIL", "QA_FAILED", "REJECTED", "REJECTED_BLURRY", "REJECT"]:
        item = {
            "candidate_id": f"test_{bad_status.lower()}",
            "num_boxes": 2,
            "visual_qa": bad_status,
            "approved_for_train": True,
            "user_train_approval": "APPROVED_FOR_TRAIN",
        }
        ok, reason = validate_candidate_for_intake(item, require_approval=True)
        assert not ok, f"Expected rejection for visual_qa='{bad_status}'"
        assert "blocked visual_qa" in reason


def test_rejects_source_unverified_status():
    """Prove candidates with verification_status != 'VERIFIED' are rejected."""
    item = {
        "candidate_id": "test_unverified",
        "num_boxes": 1,
        "visual_qa": "QA_PASS",
        "verification_status": "SOURCE_UNVERIFIED",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(item, require_approval=True)
    assert not ok
    assert "unverified source status" in reason


def test_positive_candidate_requires_valid_allowlist_status():
    """Prove positive candidates are rejected if visual_qa is not in the explicit allowlist."""
    item = {
        "candidate_id": "test_not_allowlisted",
        "num_boxes": 1,
        "visual_qa": "QA_LOOKS_GOOD_TO_ME",  # Not in allowlist
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(item, require_approval=True)
    assert not ok
    assert "not in positive allowlist" in reason


def test_positive_candidate_accepts_valid_allowlist_status():
    """Prove positive candidates pass when visual_qa is in allowlist and approved."""
    for valid_qa in ["QA_PASS", "QA_PASS_SINGLE_OBJECT", "QA_PASS_MULTI_OBJECT"]:
        item = {
            "candidate_id": f"test_valid_{valid_qa}",
            "num_boxes": 1,
            "visual_qa": valid_qa,
            "approved_for_train": True,
            "user_train_approval": "APPROVED_FOR_TRAIN",
        }
        ok, reason = validate_candidate_for_intake(item, require_approval=True)
        assert ok, f"Expected pass for visual_qa='{valid_qa}', got: {reason}"


def test_clean_negative_requires_exact_qa_status():
    """Prove negative candidate (num_boxes == 0) MUST have QA_PASS_CLEAN_NEGATIVE."""
    # Invalid negative status
    bad_neg = {
        "candidate_id": "test_bad_negative",
        "num_boxes": 0,
        "visual_qa": "QA_PASS",  # Generic pass is NOT clean negative
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(bad_neg, require_approval=True)
    assert not ok
    assert "QA_PASS_CLEAN_NEGATIVE" in reason

    # Valid clean negative
    good_neg = {
        "candidate_id": "test_good_negative",
        "num_boxes": 0,
        "visual_qa": "QA_PASS_CLEAN_NEGATIVE",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    ok, reason = validate_candidate_for_intake(good_neg, require_approval=True)
    assert ok, f"Expected pass for valid clean negative, got: {reason}"


def test_quarantined_and_pending_always_blocked():
    """Prove QUARANTINED and PENDING candidates are rejected under all conditions."""
    for status in ["QUARANTINED_AMBIGUOUS_TOOL", "PENDING_ANNOTATION", "PENDING_REVIEW"]:
        item = {
            "candidate_id": f"test_{status}",
            "num_boxes": 0,
            "visual_qa": status,
            "approved_for_train": True,
            "user_train_approval": "APPROVED_FOR_TRAIN",
        }
        ok, reason = validate_candidate_for_intake(item, require_approval=False)
        assert not ok
        assert "blocked visual_qa" in reason


def test_isolated_object_requires_slice_flag():
    """Prove OBJECT_ONLY candidates are rejected unless allow_isolated_objects=True."""
    item = {
        "candidate_id": "test_studio_banana",
        "num_boxes": 1,
        "visual_qa": "QA_PASS_OBJECT_ONLY_WHITE_BG",
        "scene_context": "OBJECT_ONLY_WHITE_SURFACE",
        "approved_for_train": True,
        "user_train_approval": "APPROVED_FOR_TRAIN",
    }
    # Default (in-the-wild discarded waste dataset): must be rejected
    ok_default, reason_def = validate_candidate_for_intake(item, require_approval=True, allow_isolated_objects=False)
    assert not ok_default
    assert "Segregated from in-the-wild" in reason_def

    # When isolated objects are explicitly enabled for a controlled slice:
    ok_slice, _ = validate_candidate_for_intake(item, require_approval=True, allow_isolated_objects=True)
    assert ok_slice


def test_dual_approval_flags_enforced():
    """Prove both approved_for_train and user_train_approval must be valid."""
    base_item = {
        "candidate_id": "test_approval_flags",
        "num_boxes": 1,
        "visual_qa": "QA_PASS",
    }

    # Case 1: approved_for_train=False, user_train_approval="APPROVED_FOR_TRAIN"
    c1 = dict(base_item, approved_for_train=False, user_train_approval="APPROVED_FOR_TRAIN")
    ok, reason = validate_candidate_for_intake(c1, require_approval=True)
    assert not ok and "must be True" in reason

    # Case 2: approved_for_train=True, user_train_approval="PENDING_PM_APPROVAL"
    c2 = dict(base_item, approved_for_train=True, user_train_approval="PENDING_PM_APPROVAL")
    ok, reason = validate_candidate_for_intake(c2, require_approval=True)
    assert not ok and "must be 'APPROVED_FOR_TRAIN'" in reason

    # Case 3: Both valid
    c3 = dict(base_item, approved_for_train=True, user_train_approval="APPROVED_FOR_TRAIN")
    ok, _ = validate_candidate_for_intake(c3, require_approval=True)
    assert ok


# ============================================================================
# TASK 2 TESTS: ROBOFLOW IMPORTER & SAFE EXTRACTION
# ============================================================================

def test_zip_safety_rejects_zip_slip(tmp_path: Path):
    """Prove validate_zip_archive_safety raises ZipSafetyError on directory traversal."""
    bad_zip = tmp_path / "zip_slip.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../evil.txt", "malicious payload")

    with pytest.raises(ZipSafetyError) as exc_info:
        validate_zip_archive_safety(bad_zip)
    assert "path traversal '..'" in str(exc_info.value)


def test_zip_safety_rejects_excessive_file_count(tmp_path: Path):
    """Prove validate_zip_archive_safety blocks archives exceeding file count limit."""
    bomb_zip = tmp_path / "zip_bomb.zip"
    with zipfile.ZipFile(bomb_zip, "w") as zf:
        for i in range(15):
            zf.writestr(f"file_{i}.txt", "test")

    with pytest.raises(ZipSafetyError) as exc_info:
        validate_zip_archive_safety(bomb_zip, max_files=10)
    assert "file count 15 exceeds limit 10" in str(exc_info.value)


def test_parse_data_yaml_nc_mismatch(tmp_path: Path):
    """Prove parse_and_validate_data_yaml raises DataYamlError when nc != len(names)."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 2\nnames: ['battery']\n", encoding="utf-8")

    with pytest.raises(DataYamlError) as exc_info:
        parse_and_validate_data_yaml(yaml_file)
    assert "Class count mismatch" in str(exc_info.value)


def test_parse_data_yaml_requires_battery_class(tmp_path: Path):
    """Prove parse_and_validate_data_yaml raises DataYamlError if no battery class is defined."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 2\nnames: ['plastic', 'metal']\n", encoding="utf-8")

    with pytest.raises(DataYamlError) as exc_info:
        parse_and_validate_data_yaml(yaml_file)
    assert "No verified battery classes found" in str(exc_info.value)


def test_parse_data_yaml_maps_battery_to_class_0(tmp_path: Path):
    """Prove verified battery classes map to 0 and other classes map to None."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text("nc: 3\nnames: ['battery', 'cardboard', 'batteries']\n", encoding="utf-8")

    class_mapping, battery_cids, names = parse_and_validate_data_yaml(yaml_file)
    assert class_mapping[0] == 0  # battery -> 0
    assert class_mapping[1] is None  # cardboard -> None
    assert class_mapping[2] == 0  # batteries -> 0
    assert battery_cids == {0, 2}


def test_validate_yolo_bbox_token_count_and_bounds():
    """Verify YOLO bounding box validation rules."""
    valid_cids = {0, 1}
    battery_cids = {0}

    # Valid battery box
    ok, err, bbox = validate_yolo_bbox("0 0.5 0.5 0.2 0.3", valid_cids, battery_cids)
    assert ok
    assert bbox["is_battery"] is True
    assert bbox["target_class_id"] == 0

    # Non-battery class ID in valid classes
    ok, err, bbox = validate_yolo_bbox("1 0.5 0.5 0.2 0.3", valid_cids, battery_cids)
    assert ok
    assert bbox["is_battery"] is False
    assert bbox["target_class_id"] is None

    # Invalid token count
    ok, err, _ = validate_yolo_bbox("0 0.5 0.5 0.2", valid_cids, battery_cids)
    assert not ok and "EXACTLY_5_TOKENS_REQUIRED" in err

    # Out of bounds coordinates (x2 = 0.9 + 0.3/2 = 1.05 > 1.0)
    ok, err, _ = validate_yolo_bbox("0 0.9 0.5 0.3 0.2", valid_cids, battery_cids)
    assert not ok and "BBOX_EXCEEDS_IMAGE_BOUNDS" in err


def test_process_synthetic_roboflow_zip_fixture(tmp_path: Path):
    """Create a minimal synthetic Roboflow ZIP and test full importer pipeline."""
    # 1. Build synthetic archive
    zip_p = tmp_path / "synthetic_roboflow_battery.zip"
    staging_target = tmp_path / "staging_roboflow"

    # Create 2 small RGB images
    im1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
    im2 = Image.new("RGB", (100, 100), color=(0, 255, 0))

    b1 = io.BytesIO()
    im1.save(b1, format="JPEG")
    b2 = io.BytesIO()
    im2.save(b2, format="JPEG")

    data_yaml_content = "nc: 1\nnames: ['battery']\n"
    label1_content = "0 0.5 0.5 0.2 0.2\n"

    with zipfile.ZipFile(zip_p, "w") as zf:
        zf.writestr("data.yaml", data_yaml_content)
        zf.writestr("train/images/bat_01.jpg", b1.getvalue())
        zf.writestr("train/labels/bat_01.txt", label1_content)
        # bat_02 has identical image to bat_01 (internal exact duplicate test)
        zf.writestr("train/images/bat_02_dup.jpg", b1.getvalue())
        zf.writestr("train/labels/bat_02_dup.txt", label1_content)
        # bat_03 has no label (missing label test)
        zf.writestr("train/images/bat_03_nolbl.jpg", b2.getvalue())

    # 2. Run importer pipeline
    manifest = process_roboflow_battery_archive(zip_p, staging_target, root_dir=ROOT)

    # 3. Verify invariants
    assert manifest["total_images"] == 3
    assert manifest["approved_for_train_count"] == 0
    assert manifest["battery_boxes"] >= 1
    assert manifest["internal_duplicates"] >= 1

    # Check item statuses
    bat01 = next(it for it in manifest["items"] if it["candidate_id"] == "bat_01")
    assert bat01["qa_status"] == "PENDING_VISUAL_QA"
    assert bat01["approved_for_train"] is False

    bat02 = next(it for it in manifest["items"] if it["candidate_id"] == "bat_02_dup")
    assert bat02["qa_status"] == "INTERNAL_DUPLICATE_REJECTED"

    bat03 = next(it for it in manifest["items"] if it["candidate_id"] == "bat_03_nolbl")
    assert bat03["qa_status"] == "MISSING_LABEL"


# ============================================================================
# TASK 3 TESTS: PHYSICAL DATASET BUILDER STAGING OUTPUT
# ============================================================================

def test_builder_staging_with_zero_approved_candidates(tmp_path: Path):
    """Prove build_safe_pilot_dataset_staging returns EMPTY_STAGING status when no candidates approved."""
    manifest_p = tmp_path / "test_unapproved_manifest.json"
    manifest_p.write_text(json.dumps({
        "manifest_name": "TEST_UNAPPROVED",
        "items": [
            {
                "candidate_id": "c1",
                "num_boxes": 1,
                "visual_qa": "QA_PASS",
                "approved_for_train": False,
                "user_train_approval": "PENDING_PM_APPROVAL",
            }
        ]
    }), encoding="utf-8")

    out_dir = tmp_path / "pilot_staging_empty"
    res = build_safe_pilot_dataset_staging(manifest_p, out_dir, require_approval=True)

    assert res["status"] == "EMPTY_STAGING_NO_APPROVED_CANDIDATES"
    assert res["staged_images_count"] == 0
    assert res["staged_labels_count"] == 0
    assert len(list((out_dir / "images").glob("*"))) == 0


def test_builder_staging_blocks_overwriting_baseline(tmp_path: Path):
    """Prove build_safe_pilot_dataset_staging prevents targeting baseline datasets."""
    dummy_manifest = tmp_path / "manifest.json"
    dummy_manifest.write_text(json.dumps({"items": []}))

    protected_target = ROOT / "data/detection_real_v2"
    with pytest.raises(OverwriteBaselineError) as exc_info:
        build_safe_pilot_dataset_staging(dummy_manifest, protected_target)
    assert "matches protected baseline path" in str(exc_info.value)


def test_builder_staging_physically_copies_approved_candidates(tmp_path: Path):
    """Prove build_safe_pilot_dataset_staging physically copies approved images and labels."""
    src_dir = tmp_path / "source"
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create dummy image and label
    im = Image.new("RGB", (100, 100), color=(128, 128, 128))
    im.save(img_dir / "approved_01.jpg")
    (lbl_dir / "approved_01.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    manifest_p = src_dir / "manifest.json"
    manifest_p.write_text(json.dumps({
        "manifest_name": "TEST_APPROVED",
        "items": [
            {
                "candidate_id": "approved_01",
                "filename": "approved_01.jpg",
                "num_boxes": 1,
                "visual_qa": "QA_PASS",
                "approved_for_train": True,
                "user_train_approval": "APPROVED_FOR_TRAIN",
            }
        ]
    }), encoding="utf-8")

    dest_dir = tmp_path / "pilot_staging_active"
    res = build_safe_pilot_dataset_staging(manifest_p, dest_dir, require_approval=True)

    assert res["status"] == "STAGING_SUCCESS"
    assert res["staged_images_count"] == 1
    assert (dest_dir / "images" / "approved_01.jpg").exists()
    assert (dest_dir / "labels" / "approved_01.txt").exists()
    assert (dest_dir / "labels" / "approved_01.txt").read_text().strip() == "0 0.5 0.5 0.2 0.2"


def test_builder_staging_blocks_require_approval_false_bypass(tmp_path: Path):
    """
    FIX 1 Regression Test:
    Calling build_safe_pilot_dataset_staging with require_approval=False must raise SafeDatasetBuilderError.
    An unapproved candidate (visual_qa=QA_PASS, approved_for_train=False, user_train_approval=PENDING_PM_APPROVAL)
    must NEVER be staged under any circumstances.
    """
    from src.detection.safe_dataset_builder import SafeDatasetBuilderError

    src_dir = tmp_path / "bypass_src"
    img_dir = src_dir / "images"
    lbl_dir = src_dir / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    im = Image.new("RGB", (100, 100), color=(100, 100, 100))
    im.save(img_dir / "unapproved_01.jpg")
    (lbl_dir / "unapproved_01.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    manifest_p = src_dir / "manifest.json"
    manifest_p.write_text(json.dumps({
        "manifest_name": "TEST_BYPASS",
        "items": [
            {
                "candidate_id": "unapproved_01",
                "filename": "unapproved_01.jpg",
                "num_boxes": 1,
                "visual_qa": "QA_PASS",
                "approved_for_train": False,
                "user_train_approval": "PENDING_PM_APPROVAL",
            }
        ]
    }), encoding="utf-8")

    dest_dir = tmp_path / "bypass_dest"

    # Attempting to bypass gate with require_approval=False MUST raise SafeDatasetBuilderError
    with pytest.raises(SafeDatasetBuilderError) as exc_info:
        build_safe_pilot_dataset_staging(manifest_p, dest_dir, require_approval=False)
    assert "require_approval cannot be False" in str(exc_info.value)

    # Calling normally with require_approval=True results in empty staging (0 staged images)
    res = build_safe_pilot_dataset_staging(manifest_p, dest_dir, require_approval=True)
    assert res["status"] == "EMPTY_STAGING_NO_APPROVED_CANDIDATES"
    assert res["staged_images_count"] == 0
    assert not (dest_dir / "images" / "unapproved_01.jpg").exists()


def test_validate_on_disk_sample_boundary_exceed_rejected(tmp_path: Path):
    """
    FIX 2 Regression Test:
    Line '0 0.95 0.5 0.3 0.2' has x2 = 0.95 + 0.15 = 1.10 > 1.0.
    Must be rejected with InvalidAnnotationError (bbox exceeds bounds).
    """
    img = tmp_path / "sample_bounds.jpg"
    Image.new("RGB", (100, 100)).save(img)

    lbl = tmp_path / "sample_bounds.txt"
    lbl.write_text("0 0.95 0.5 0.3 0.2\n", encoding="utf-8")

    with pytest.raises(InvalidAnnotationError) as exc_info:
        validate_on_disk_sample(img, lbl, expected_boxes=1)
    assert "exceeds image bounds" in str(exc_info.value)
    assert "1.1000" in str(exc_info.value) or "1.1" in str(exc_info.value)


def test_validate_on_disk_sample_nan_and_inf_rejected(tmp_path: Path):
    """
    FIX 2 Regression Test:
    NaN or Infinity coordinates must be rejected with InvalidAnnotationError.
    """
    img = tmp_path / "sample_nan.jpg"
    Image.new("RGB", (100, 100)).save(img)

    # NaN xc
    lbl_nan = tmp_path / "sample_nan.txt"
    lbl_nan.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError) as exc_info:
        validate_on_disk_sample(img, lbl_nan, expected_boxes=1)
    assert "non-finite value" in str(exc_info.value)

    # Inf yc
    lbl_inf = tmp_path / "sample_inf.txt"
    lbl_inf.write_text("0 0.5 inf 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError) as exc_info:
        validate_on_disk_sample(img, lbl_inf, expected_boxes=1)
    assert "non-finite value" in str(exc_info.value)


def test_validate_on_disk_sample_invalid_dimensions_and_class_id(tmp_path: Path):
    """
    FIX 2 Regression Test:
    Rejects non-positive dimensions and out-of-range class IDs.
    """
    img = tmp_path / "sample_dims.jpg"
    Image.new("RGB", (100, 100)).save(img)

    # Zero width
    lbl_zero = tmp_path / "sample_zero.txt"
    lbl_zero.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError) as exc_info:
        validate_on_disk_sample(img, lbl_zero, expected_boxes=1)
    assert "non-positive dimensions" in str(exc_info.value)

    # Class ID out of range (10 not in 0..9)
    lbl_cid = tmp_path / "sample_cid.txt"
    lbl_cid.write_text("10 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(InvalidAnnotationError) as exc_info:
        validate_on_disk_sample(img, lbl_cid, expected_boxes=1)
    assert "invalid class ID 10" in str(exc_info.value)
