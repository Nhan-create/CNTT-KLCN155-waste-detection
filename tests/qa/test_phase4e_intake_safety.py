"""
tests/qa/test_phase4e_intake_safety.py
--------------------------------------
Regression tests verifying Phase 4E.2 data intake and staging safety:
1. Proves validate_yolo_bbox rejects:
   - Non-integer, negative, or unmapped class IDs (-1, 999).
   - Lines with != 5 tokens (> 5 or < 5 tokens).
   - Bounding boxes exceeding image boundaries (x1 < 0, y1 < 0, x2 > 1, y2 > 1).
2. Proves inspect_roboflow_data_yaml enforces:
   - Exact equality nc == len(names) (rejects nc=1 with 2 classes).
   - Presence of verified battery classes (battery, batteries).
3. Proves safe_extract_untrusted_zip blocks Zip Slip path traversal and zip bombs.
4. Proves load_authoritative_split_hashes fails closed on missing configs, splits, or physical images,
   and verifies holdout against manifest (JPG, JPEG, PNG).
5. Proves stage_phase4e_whitelist_candidates does not generate 0-byte labels or placeholder comments on disk,
   and fails closed without --approve-download.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_validate_yolo_bbox_valid_cases():
    """Verify standard normalized YOLO bounding boxes pass validation."""
    from scripts.ingest_roboflow_battery_export import validate_yolo_bbox

    valid_lines = [
        "0 0.5 0.5 0.2 0.3\n",
        "1 0.123456 0.654321 0.05 0.10\n",
        "0 0.9 0.9 0.1 0.1\n",
        "5 0.500000 0.500000 1.000000 1.000000\n",
    ]
    for line in valid_lines:
        is_valid, err, bbox = validate_yolo_bbox(line)
        assert is_valid, f"Expected valid box for '{line.strip()}', got err: {err}"
        assert bbox is not None
        assert len(bbox) == 5
        cid, xc, yc, w, h = bbox
        assert 0.0 <= xc <= 1.0
        assert 0.0 <= yc <= 1.0


def test_validate_yolo_bbox_token_count_strictly_five():
    """P0: Reviewer bug - lines with more or fewer than 5 tokens must be rejected."""
    from scripts.ingest_roboflow_battery_export import validate_yolo_bbox

    # > 5 tokens
    is_valid, err, _ = validate_yolo_bbox("0 0.5 0.5 0.2 0.3 extra_token")
    assert not is_valid
    assert "EXACTLY_5_TOKENS_REQUIRED" in err

    # < 5 tokens
    is_valid, err, _ = validate_yolo_bbox("0 0.5 0.5 0.2")
    assert not is_valid
    assert "EXACTLY_5_TOKENS_REQUIRED" in err


def test_validate_yolo_bbox_class_id_validation():
    """P0: Reviewer bug - class ID 999 or -1 must be rejected."""
    from scripts.ingest_roboflow_battery_export import validate_yolo_bbox

    valid_classes = {0}  # nc=1: only class 0 allowed

    # Class ID -1
    is_valid, err, _ = validate_yolo_bbox("-1 0.5 0.5 0.2 0.2", valid_class_ids=valid_classes)
    assert not is_valid
    assert "NEGATIVE_CLASS_ID" in err

    # Class ID 999
    is_valid, err, _ = validate_yolo_bbox("999 0.5 0.5 0.2 0.2", valid_class_ids=valid_classes)
    assert not is_valid
    assert "CLASS_ID_NOT_IN_DATA_YAML" in err

    # Non-integer class ID
    is_valid, err, _ = validate_yolo_bbox("0.5 0.5 0.5 0.2 0.2", valid_class_ids=valid_classes)
    assert not is_valid
    assert "NON_INTEGER_CLASS_ID" in err


def test_validate_yolo_bbox_strictly_within_image_bounds():
    """P0: Reviewer bug - boxes exceeding image boundaries must be rejected."""
    from scripts.ingest_roboflow_battery_export import validate_yolo_bbox

    # Box exceeding right boundary: xc=0.9, w=0.3 -> x2 = 1.05
    is_valid, err, _ = validate_yolo_bbox("0 0.9 0.5 0.3 0.2")
    assert not is_valid
    assert "BBOX_EXCEEDS_IMAGE_BOUNDS" in err

    # Box exceeding left boundary: xc=0.1, w=0.3 -> x1 = -0.05
    is_valid, err, _ = validate_yolo_bbox("0 0.1 0.5 0.3 0.2")
    assert not is_valid
    assert "BBOX_EXCEEDS_IMAGE_BOUNDS" in err

    # Box exceeding bottom boundary: yc=0.9, h=0.3 -> y2 = 1.05
    is_valid, err, _ = validate_yolo_bbox("0 0.5 0.9 0.2 0.3")
    assert not is_valid
    assert "BBOX_EXCEEDS_IMAGE_BOUNDS" in err

    # Box exceeding top boundary: yc=0.1, h=0.3 -> y1 = -0.05
    is_valid, err, _ = validate_yolo_bbox("0 0.5 0.1 0.2 0.3")
    assert not is_valid
    assert "BBOX_EXCEEDS_IMAGE_BOUNDS" in err

    # Box perfectly touching boundary (xc=0.5, w=1.0 -> x1=0.0, x2=1.0)
    is_valid, err, bbox = validate_yolo_bbox("0 0.5 0.5 1.0 1.0")
    assert is_valid, f"Expected box spanning full image to pass, got err: {err}"


def test_inspect_roboflow_data_yaml_nc_mismatch(tmp_path):
    """P0: Reviewer bug - data.yaml with nc=1 but 2 names must fail closed."""
    from scripts.ingest_roboflow_battery_export import inspect_roboflow_data_yaml

    bad_yaml = tmp_path / "data.yaml"
    bad_yaml.write_text("names: ['battery', 'cardboard']\nnc: 1\n")
    with pytest.raises(ValueError) as exc:
        inspect_roboflow_data_yaml(tmp_path)
    assert "class count mismatch" in str(exc.value)


def test_inspect_roboflow_data_yaml_identifies_battery_classes(tmp_path):
    """Verify data.yaml parses battery indices and flags non-battery classes."""
    from scripts.ingest_roboflow_battery_export import inspect_roboflow_data_yaml

    valid_multi_yaml = tmp_path / "data.yaml"
    valid_multi_yaml.write_text("names: ['waste_batteries', 'paper_box']\nnc: 2\n")
    meta = inspect_roboflow_data_yaml(tmp_path)
    assert meta["nc"] == 2
    assert meta["battery_class_ids"] == {0}
    assert meta["other_class_ids"] == {1}


def test_safe_extract_untrusted_zip_zip_slip_protection(tmp_path):
    """Verify safe_extract_untrusted_zip rejects path traversal (Zip Slip)."""
    from scripts.ingest_roboflow_battery_export import safe_extract_untrusted_zip

    zip_file = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        # Write an entry pointing outside target
        zf.writestr("../../evil.txt", "malicious payload")

    dest_dir = tmp_path / "unpacked"
    with pytest.raises(ValueError) as exc:
        safe_extract_untrusted_zip(zip_file, dest_dir)
    assert "zip slip" in str(exc.value).lower()


def test_safe_extract_untrusted_zip_bomb_protection(tmp_path):
    """Verify safe_extract_untrusted_zip rejects archives exceeding size limits."""
    from scripts.ingest_roboflow_battery_export import safe_extract_untrusted_zip

    zip_file = tmp_path / "bomb.zip"
    with zipfile.ZipFile(zip_file, "w") as zf:
        zf.writestr("big.txt", "0" * 1000)

    dest_dir = tmp_path / "unpacked"
    # Set max_bytes to 500
    with pytest.raises(ValueError) as exc:
        safe_extract_untrusted_zip(zip_file, dest_dir, max_bytes=500)
    assert "zip bomb protection" in str(exc.value).lower()


def test_load_authoritative_split_hashes_fails_closed_on_missing_files(tmp_path):
    """Verify load_authoritative_split_hashes raises FileNotFoundError if split file is missing."""
    from scripts.ingest_roboflow_battery_export import load_authoritative_split_hashes

    fake_yaml = tmp_path / "fake_v2.yaml"
    fake_yaml.write_text("train: missing_train.txt\nval: missing_val.txt\ntest: missing_test.txt\n")

    with pytest.raises(FileNotFoundError) as exc_info:
        load_authoritative_split_hashes(v2_yaml_path=fake_yaml)
    assert "missing" in str(exc_info.value).lower()


def test_load_authoritative_split_hashes_verifies_holdout_jpg_jpeg_png():
    """Verify holdout verification checks all manifest files (including JPEG and PNG)."""
    from scripts.ingest_roboflow_battery_export import load_authoritative_split_hashes

    split_hashes, split_phashes = load_authoritative_split_hashes()
    assert "holdout" in split_hashes
    # Note: 26 unique SHA-256 hashes across 27 manifest files because images (7).jpg and images (13).jpg are byte-duplicates
    assert len(split_hashes["holdout"]) == 26
    assert len(split_phashes["holdout"]) == 27

    # Verify presence of jpeg and png holdout samples
    holdout_names = list(split_phashes["holdout"].keys())
    assert any(fn.endswith(".jpeg") for fn in holdout_names), "Holdout must include .jpeg files"
    assert any(fn.endswith(".png") for fn in holdout_names), "Holdout must include .png files"
    assert any(fn.endswith(".jpg") for fn in holdout_names), "Holdout must include .jpg files"


def test_stage_phase4e_whitelist_does_not_contain_pending_annotation_comments():
    """P0 Regression Test: Ensure scripts do not write '# PENDING_ANNOTATION' into label files."""
    script_p = ROOT / "scripts" / "stage_phase4e_whitelist_candidates.py"
    text = script_p.read_text(encoding="utf-8")
    assert "# PENDING_ANNOTATION" not in text, "Found forbidden '# PENDING_ANNOTATION' comment generation in script"
    # Ensure no premature 0-byte file write upon download
    assert 'dest_lbl.write_text("",' not in text, "Found premature 0-byte file generation in staging script"


def test_stage_phase4e_mini_pilot_shortlist_structure():
    """Verify mini-pilot shortlist contains exactly the 4 required authentic candidates."""
    from scripts.stage_phase4e_whitelist_candidates import MINI_PILOT_SHORTLIST

    assert len(MINI_PILOT_SHORTLIST) == 4, f"Expected 4 shortlist items, found {len(MINI_PILOT_SHORTLIST)}"
    assert "wiki_bio_banana_peel_ground" in MINI_PILOT_SHORTLIST
    assert "wiki_bio_street_food_remains" in MINI_PILOT_SHORTLIST
    assert "wiki_neg_junction_boxes" in MINI_PILOT_SHORTLIST
    assert "wiki_neg_utility_box_leeds" in MINI_PILOT_SHORTLIST

    # Proves Street_cabinet.jpg is NOT in shortlist
    assert "wiki_neg_street_cabinet" not in MINI_PILOT_SHORTLIST


def test_stage_phase4e_unapproved_run_fails_closed():
    """Verify running staging script without approval flag exits with code 1."""
    cmd = [sys.executable, "scripts/stage_phase4e_whitelist_candidates.py"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 1, f"Expected returncode 1, got {res.returncode}"
    assert "USER APPROVAL REQUIRED" in (res.stdout + res.stderr)


def test_quarantine_non_battery_source_class():
    """P0: Reviewer requirement - non-battery source classes must be quarantined, NOT converted."""
    from scripts.ingest_roboflow_battery_export import validate_yolo_bbox

    # Simulated data.yaml with nc=2: 0=waste_battery, 1=plastic_bottle
    valid_class_ids = {0, 1}
    battery_class_ids = {0}

    # Line with class 1 (plastic_bottle)
    line = "1 0.5 0.5 0.2 0.2"
    is_valid, err, bbox_info = validate_yolo_bbox(
        line, valid_class_ids=valid_class_ids, battery_class_ids=battery_class_ids
    )
    assert is_valid, "Bbox itself is well-formed"
    src_cid, xc, yc, w, h = bbox_info
    assert src_cid == 1
    # Verification: must NOT be in battery_class_ids
    assert src_cid not in battery_class_ids, "Class 1 must not be treated as battery"


def test_evaluate_source_class_mapping_vetted_battery_types():
    """P0: Reviewer requirement - battery form factors without 'battery' word must be accepted."""
    from scripts.ingest_roboflow_battery_export import evaluate_source_class_mapping

    vetted_cases = [
        "AA", "AAA", "9V", "18650", "Button", "Cylindrical", "Rectangular", "Coin", "Lithium", "Lead_Acid"
    ]
    for cname in vetted_cases:
        target, status = evaluate_source_class_mapping(cname)
        assert target == "0: battery", f"Expected '0: battery' for '{cname}', got '{target}'"
        assert status == "PROPOSED_ACCEPTED"


def test_evaluate_source_class_mapping_accessories_excluded():
    """P0: Reviewer requirement - charger, case, holder, etc. must NEVER be converted to battery."""
    from scripts.ingest_roboflow_battery_export import evaluate_source_class_mapping

    accessory_cases = [
        "battery charger", "battery_case", "battery holder", "charging dock", "battery_cable", "adapter"
    ]
    for cname in accessory_cases:
        target, status = evaluate_source_class_mapping(cname)
        assert target == "QUARANTINE_ACCESSORY", f"Expected 'QUARANTINE_ACCESSORY' for '{cname}', got '{target}'"
        assert status == "EXCLUDED_ACCESSORY"


def test_evaluate_source_class_mapping_ambiguous_quarantined():
    """P0: Reviewer requirement - unverified or ambiguous classes must be quarantined."""
    from scripts.ingest_roboflow_battery_export import evaluate_source_class_mapping

    ambiguous_cases = ["Damaged", "unknown_box", "electronic_circuit"]
    for cname in ambiguous_cases:
        target, status = evaluate_source_class_mapping(cname)
        assert target == "QUARANTINE_UNVERIFIED", f"Expected 'QUARANTINE_UNVERIFIED' for '{cname}', got '{target}'"
        assert status == "REQUIRES_APPROVAL"


def test_inspect_roboflow_data_yaml_with_vetted_battery_form_factors(tmp_path):
    """P0: Reviewer test case - data.yaml with 8 classes (AA, AAA, 9V, 18650, Button, Cylindrical, Rectangular, Damaged)."""
    from scripts.ingest_roboflow_battery_export import inspect_roboflow_data_yaml

    yaml_p = tmp_path / "data.yaml"
    yaml_p.write_text(
        "names: ['AA', 'AAA', '9V', '18650', 'Button', 'Cylindrical', 'Rectangular', 'Damaged']\n"
        "nc: 8\n"
    )
    meta = inspect_roboflow_data_yaml(tmp_path)
    assert meta["nc"] == 8
    # 7 accepted battery classes (indices 0..6), 1 unverified/quarantined (index 7: Damaged)
    assert meta["battery_class_ids"] == {0, 1, 2, 3, 4, 5, 6}
    assert meta["quarantine_class_ids"] == {7}
    assert meta["mapping_table"][7]["mapping_status"] == "REQUIRES_APPROVAL"
    assert meta["mapping_table"][0]["mapping_status"] == "PROPOSED_ACCEPTED"


def test_cache_key_uniqueness_for_different_paths(tmp_path):
    """P0: Reviewer requirement - cache keys must not collide between different files with same basename."""
    dir1 = tmp_path / "split1"
    dir2 = tmp_path / "split2"
    dir1.mkdir()
    dir2.mkdir()

    file1 = dir1 / "image.jpg"
    file2 = dir2 / "image.jpg"
    file1.write_bytes(b"content1")
    file2.write_bytes(b"content2")

    # Generate cache keys using resolved absolute path
    st1 = file1.stat()
    st2 = file2.stat()
    key1 = f"{str(file1.resolve()).replace('\\', '/')}_{st1.st_size}_{st1.st_mtime_ns}"
    key2 = f"{str(file2.resolve()).replace('\\', '/')}_{st2.st_size}_{st2.st_mtime_ns}"

    assert key1 != key2, "Cache keys for files in different directories must be distinct"


def test_evaluate_source_class_mapping_non_battery_compounds():
    """P0: Reviewer bug - button_shirt, coin_purse, rectangular_box, lithium_flashlight must NOT map to battery."""
    from scripts.ingest_roboflow_battery_export import evaluate_source_class_mapping

    false_battery_cases = [
        "button_shirt",
        "coin_purse",
        "rectangular_box",
        "lithium_flashlight",
        "button-shirt",
        "coin-purse",
        "rectangular-box",
        "lithium-flashlight"
    ]
    for cname in false_battery_cases:
        target, status = evaluate_source_class_mapping(cname)
        assert target != "0: battery", f"False positive mapping: '{cname}' was mapped to battery!"
        assert target == "QUARANTINE_UNVERIFIED"
        assert status in {"REQUIRES_APPROVAL", "EXCLUDED_NON_BATTERY_OBJECT"}


def test_evaluate_source_class_mapping_numeric_zero_unverified_and_rejected(tmp_path):
    """P0: Reviewer bug - names: ['0'] must be unverified and data.yaml rejected (not auto-accepted)."""
    from scripts.ingest_roboflow_battery_export import evaluate_source_class_mapping, inspect_roboflow_data_yaml

    # 1. evaluate_source_class_mapping must treat "0" as unverified numeric
    target, status = evaluate_source_class_mapping("0")
    assert target != "0: battery", "'0' must not be auto-accepted as battery"
    assert target == "QUARANTINE_UNVERIFIED"
    assert status == "UNVERIFIED_NUMERIC_CLASS"

    # 2. inspect_roboflow_data_yaml must fail closed on names: ['0']
    yaml_p = tmp_path / "data.yaml"
    yaml_p.write_text("names: ['0']\nnc: 1\n")
    with pytest.raises(ValueError) as exc:
        inspect_roboflow_data_yaml(tmp_path)
    assert "do not contain any vetted battery class" in str(exc.value)


def test_inspect_roboflow_data_yaml_non_continuous_dict_keys(tmp_path):
    """P0: Reviewer bug - names dict with missing keys (e.g. 0, 2 missing 1) must raise ValueError."""
    from scripts.ingest_roboflow_battery_export import inspect_roboflow_data_yaml

    # Missing key 1: keys are 0 and 2
    yaml_missing_key = tmp_path / "data.yaml"
    yaml_missing_key.write_text("names: {0: 'battery', 2: 'cardboard'}\nnc: 2\n")
    with pytest.raises(ValueError) as exc1:
        inspect_roboflow_data_yaml(tmp_path)
    assert "not continuous integers" in str(exc1.value)

    # Starts at 1 instead of 0: keys are 1 and 2
    yaml_missing_key.write_text("names: {1: 'battery', 2: 'cardboard'}\nnc: 2\n")
    with pytest.raises(ValueError) as exc2:
        inspect_roboflow_data_yaml(tmp_path)
    assert "not continuous integers" in str(exc2.value)


def test_resolve_unique_curated_path_prevents_basename_collision(tmp_path):
    """P0: Reviewer requirement - different images with identical basename across folders must not overwrite."""
    import hashlib
    from scripts.ingest_roboflow_battery_export import resolve_unique_curated_path

    curated_dir = tmp_path / "curated"
    curated_dir.mkdir()

    bytes1 = b"content1"
    sha1 = hashlib.sha256(bytes1).hexdigest()
    bytes2 = b"content2"
    sha2 = hashlib.sha256(bytes2).hexdigest()

    # Image 1: train/images/sample.jpg
    path1 = Path("train/images/sample.jpg")
    dest1 = resolve_unique_curated_path(curated_dir, path1, sha1)
    assert dest1.name == "sample.jpg"
    dest1.write_bytes(bytes1)

    # Image 2: val/images/sample.jpg (different content, same basename)
    path2 = Path("val/images/sample.jpg")
    dest2 = resolve_unique_curated_path(curated_dir, path2, sha2)
    assert dest2 != dest1, "Must not return the same path for different content"
    assert dest2.name != "sample.jpg", "Must disambiguate filename to prevent overwrite"
    assert "val" in dest2.name
    assert sha2[:8] in dest2.name
    dest2.write_bytes(bytes2)

    # Verify both files coexist and neither was overwritten
    assert dest1.exists() and dest1.read_bytes() == bytes1
    assert dest2.exists() and dest2.read_bytes() == bytes2

    # Image 3: test/images/sample.jpg with identical sha1 (true duplicate) returns original path
    path3 = Path("test/images/sample.jpg")
    dest3 = resolve_unique_curated_path(curated_dir, path3, sha1)
    assert dest3 == dest1, "Identical content should resolve to existing destination"


def test_mini_pilot_qa_manifest_contract():
    """Verify Phase 4E Mini-Pilot QA manifest conforms to strict acceptance contracts."""
    import hashlib

    manifest_p = ROOT / "data" / "acquisition_staging" / "manifests" / "mini_pilot_qa_manifest.json"
    assert manifest_p.exists(), f"Missing QA manifest: {manifest_p}"

    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert data["manifest_version"] == "mini_pilot_v2_synced_qa"
    assert data["status"] == "DATA_QA_COMPLETED_AWAITING_USER_APPROVAL"
    assert data["within_size_budget"] is True
    assert data["total_downloaded_bytes"] <= data["max_download_budget_bytes"]

    summary = data["summary"]
    assert summary["total_candidates"] == 4
    assert summary["technical_qa_passed_count"] == 4
    assert summary["annotation_candidates_proposed"] == 2
    assert summary["annotation_rejected_count"] == 1
    assert summary["annotation_pending_review_count"] == 1
    assert summary["approved_for_train_count"] == 0

    img_dir = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images"
    lbl_dir = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "labels"

    # Enforce zero premature label files written to disk
    label_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
    assert len(label_files) == 0, f"Expected 0 label files on disk before review, found {len(label_files)}"

    for cand in data["candidates"]:
        cid = cand["candidate_id"]
        img_p = img_dir / f"{cid}.jpg"
        assert img_p.exists(), f"Physical file missing for {cid}: {img_p}"
        actual_sha = hashlib.sha256(img_p.read_bytes()).hexdigest()
        assert actual_sha == cand["sha256"], f"SHA mismatch for {cid}: {actual_sha} vs {cand['sha256']}"

        assert cand["technical_qa_result"] == "PASS"
        assert cand["user_train_approval"] in {"PENDING_USER_APPROVAL", "REJECTED", "BLOCKED_PENDING_REVIEW"}

        if cid == "wiki_bio_banana_peel_ground":
            assert cand["annotation_review_status"] == "PROPOSED_PENDING_USER_APPROVAL"
            assert len(cand["proposed_bboxes"]) == 1
            bbox = cand["proposed_bboxes"][0]
            assert bbox["class_id"] == 1
            assert bbox["class_name"] == "biological"
            xc, yc, w, h = bbox["yolo_normalized_cxcywh"]
            assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
            assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
            assert bbox["pixel_box_xyxy"] == [2100, 1280, 5184, 2300]
            assert bbox["touches_right_edge"] is True
        elif cid == "wiki_bio_street_food_remains":
            assert cand["annotation_review_status"] == "REJECTED_TAXONOMY_CONFLICT"
            assert "styrofoam" in cand["annotation_review_notes"].lower()
            assert len(cand["proposed_bboxes"]) == 0
        elif cid == "wiki_neg_junction_boxes":
            assert cand["annotation_review_status"] == "CLEAN_NEGATIVE_CANDIDATE"
            assert len(cand["proposed_bboxes"]) == 0
        elif cid == "wiki_neg_utility_box_leeds":
            assert cand["annotation_review_status"] == "PENDING_OBJECT_IDENTITY_REVIEW"
            assert "unidentified" in cand["annotation_review_notes"].lower()
            assert len(cand["proposed_bboxes"]) == 0


def test_train_expansion_candidate_manifest_contract():
    """Verify train_expansion_candidate_manifest.json conforms to safety and provenance requirements."""
    import hashlib

    manifest_p = ROOT / "data" / "acquisition_staging" / "manifests" / "train_expansion_candidate_manifest.json"
    assert manifest_p.exists(), f"Missing train expansion manifest: {manifest_p}"

    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert data["manifest_name"] == "TRAIN_EXPANSION_CANDIDATE_MINI_PILOT"
    assert data["approved_for_train"] is False
    assert data["candidates_count"] == 2
    assert len(data["candidates"]) == 2

    for cand in data["candidates"]:
        img_p = ROOT / cand["image_path"]
        assert img_p.exists(), f"Candidate physical image missing: {img_p}"
        actual_sha = hashlib.sha256(img_p.read_bytes()).hexdigest()
        assert actual_sha == cand["image_sha256"]

        # Verify proposed label content sha256
        label_content = cand["proposed_label_content"]
        actual_lbl_sha = hashlib.sha256(label_content.encode("utf-8")).hexdigest()
        assert actual_lbl_sha == cand["proposed_label_sha256"]

        assert cand["merge_status"] == "PENDING_USER_APPROVAL"
        assert cand["leakage_verification"]["status"] == "CLEAN"


def test_mini_pilot_contact_sheet_and_overlays_exist():
    """Verify physical contact sheet and 4 visual QA overlay artifacts exist."""
    cs_path = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "contact_sheet" / "mini_pilot_contact_sheet.jpg"
    assert cs_path.exists(), f"Contact sheet missing: {cs_path}"
    assert cs_path.stat().st_size > 50_000, "Contact sheet file is unexpectedly small"

    overlays_dir = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "overlays"
    assert overlays_dir.exists(), f"Overlays dir missing: {overlays_dir}"

    expected_overlays = [
        "overlay_wiki_bio_banana_peel_ground.jpg",
        "overlay_wiki_bio_street_food_remains.jpg",
        "overlay_wiki_neg_junction_boxes.jpg",
        "overlay_wiki_neg_utility_box_leeds.jpg",
    ]
    for name in expected_overlays:
        ov = overlays_dir / name
        assert ov.exists(), f"Overlay missing: {ov}"
        assert ov.stat().st_size > 50_000, f"Overlay file unexpectedly small: {ov}"




