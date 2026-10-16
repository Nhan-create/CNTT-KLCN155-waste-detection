"""
tests/qa/test_task5c_battery_qa.py
----------------------------------
Automated QA verification suite for Sprint AI-UPGRADE-01 Task 5C:
1. Real data.yaml parsing for Roboflow alkaline battery subtypes.
2. Verified Roboflow battery staging manifest integrity (350 images, 505 boxes).
3. Contact sheet and sample overlay physical asset verification.
4. Zero leakage against frozen baseline splits and external holdout.
5. Strict fail-closed gatekeeping: 0 approved for train until explicit human sign-off.
"""

from __future__ import annotations

import json
from pathlib import Path
from PIL import Image
import pytest

from scripts.import_and_qa_roboflow_battery import (
    parse_and_validate_data_yaml,
    VERIFIED_BATTERY_CLASS_NAMES,
)
from src.detection.safe_dataset_builder import (
    validate_candidate_for_intake,
    build_safe_pilot_dataset_staging,
    SafeDatasetBuilderError,
)

ROOT = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "roboflow_battery"
MANIFEST_PATH = STAGING_DIR / "roboflow_battery_staging_manifest.json"


def test_parse_data_yaml_alkaline_subtypes_mapped_to_class_0(tmp_path: Path):
    """Prove all 3 Roboflow waste-battery-only classes map to project class 0: battery."""
    yaml_file = tmp_path / "data.yaml"
    yaml_file.write_text(
        "nc: 3\nnames: ['Alkaline', 'Alkaline_jp', 'LR_Alkaline']\n",
        encoding="utf-8",
    )
    class_mapping, battery_cids, names = parse_and_validate_data_yaml(yaml_file)
    assert class_mapping[0] == 0  # Alkaline -> 0
    assert class_mapping[1] == 0  # Alkaline_jp -> 0
    assert class_mapping[2] == 0  # LR_Alkaline -> 0
    assert battery_cids == {0, 1, 2}
    assert len(names) == 3


def test_roboflow_battery_manifest_and_counts():
    """Verify authoritative counts in the generated roboflow_battery_staging_manifest.json."""
    assert MANIFEST_PATH.exists(), f"Missing staging manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_images"] == 350
    assert manifest["total_boxes"] == 505
    assert manifest["battery_boxes"] == 505
    assert manifest["non_battery_boxes"] == 0
    assert manifest["external_leaks"] == 0
    assert manifest["approved_for_train_count"] in (0, 140)

    splits = manifest["source_splits"]
    assert splits["train_images"] == 315
    assert splits["valid_images"] == 35
    assert splits["unique_base_images"] == 140

    items = manifest["items"]
    assert len(items) == 350

    for it in items:
        if it.get("is_base_representative"):
            assert it["approved_for_train"] in (False, True)
            assert it["user_train_approval"] in ("PENDING_PM_APPROVAL", "USER_APPROVED_TASK6_PILOT")
        else:
            assert it["approved_for_train"] is False
            assert it["user_train_approval"] == "PENDING_PM_APPROVAL"
        assert it["decode_ok"] is True
        assert it["technical_status"] == "TECHNICAL_VALID"
        assert it["original_status"] == "ORIGINAL_UNKNOWN"
        assert it["qa_decision"] in ("VISUAL_QA_PASS", "NEEDS_REANNOTATION")
        assert it["visual_qa"] in ("VISUAL_QA_PASS", "NEEDS_REANNOTATION")
        if it["visual_qa"] == "VISUAL_QA_PASS":
            assert isinstance(it["review_evidence"], dict)
            assert it["review_evidence"].get("reviewed_by")
        else:
            assert it["review_evidence"] is None
        assert it["battery_boxes"] > 0
        assert not it["external_leakage"]


def test_contact_sheets_and_overlays_exist_and_valid():
    """Verify that all generated contact sheets and overlays exist on disk and decode as valid images."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cs_rel_paths = manifest.get("contact_sheets", [])
    assert len(cs_rel_paths) >= 10, f"Expected >= 10 contact sheets, found {len(cs_rel_paths)}"

    for cs_rel in cs_rel_paths:
        cs_full = STAGING_DIR / cs_rel
        assert cs_full.exists(), f"Missing contact sheet: {cs_full}"
        assert cs_full.stat().st_size > 50_000, f"Contact sheet too small: {cs_full}"
        with Image.open(cs_full) as im:
            assert im.width > 1000 and im.height > 1000

    overlays = manifest.get("sample_overlays", [])
    assert len(overlays) >= 5, f"Expected >= 5 sample overlays, found {len(overlays)}"
    for ov_rel in overlays:
        ov_full = STAGING_DIR / ov_rel
        assert ov_full.exists(), f"Missing overlay: {ov_full}"
        with Image.open(ov_full) as im:
            assert im.width > 0 and im.height > 0


def test_zero_leakage_against_authoritative_reference_cache():
    """Assert zero SHA-256 or pHash collision against frozen train/val/test/holdout splits."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for it in manifest["items"]:
        assert it["external_leakage"] is False, f"Leaked item detected: {it['candidate_id']}"


def test_fail_closed_gatekeeper_blocks_staging_without_human_approval(tmp_path: Path):
    """Verify safe dataset builder strictly refuses to stage unapproved or NEEDS_REANNOTATION battery items."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Pick 5 unapproved augmented variant items directly from manifest
    sample_candidates = [it for it in manifest["items"] if not it.get("is_base_representative")][:5]

    # Intake validation without approved_for_train=True MUST fail
    for cand in sample_candidates:
        ok, reason = validate_candidate_for_intake(cand, require_approval=True)
        assert not ok
        assert ("approved_for_train" in reason) or ("NEEDS_REANNOTATION" in reason)

    # Builder staging must return empty staging when candidates are unapproved
    mini_manifest = tmp_path / "test_battery_manifest.json"
    mini_manifest.write_text(json.dumps({"items": sample_candidates}), encoding="utf-8")

    result = build_safe_pilot_dataset_staging(
        manifest_path=mini_manifest,
        output_dir=tmp_path / "staging_out",
        require_approval=True,
    )
    assert result["status"] == "EMPTY_STAGING_NO_APPROVED_CANDIDATES"
    assert result["staged_images_count"] == 0
    assert result["staged_labels_count"] == 0
