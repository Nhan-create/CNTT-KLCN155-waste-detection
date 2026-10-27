"""
Unit and regression tests for Task 4: Real Image Acquisition & QA.
Verifies:
1. Physical existence of all 10 acquired images in staging.
2. Hard byte limit compliance (total <= 50 MiB).
3. Image decodability and valid RGB mode.
4. Correct YOLO format annotations in staging/labels/.
5. Strict 0-box negative label files for clean background textures.
6. Zero training approval gatekeeper (approved_for_train == False).
7. Baseline reconciliation artifact integrity (BASELINE_CORRECTION.md).
"""

import json
from pathlib import Path
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = ROOT / "data/acquisition_staging/batch_a_mini01"
IMG_DIR = STAGING_DIR / "images"
LBL_DIR = STAGING_DIR / "labels"
MANIFEST_PATH = STAGING_DIR / "manifest.json"
MAX_ALLOWED_BYTES = 50 * 1024 * 1024  # 50 MiB

EXPECTED_CANDIDATE_IDS = [
    "wiki_bat_corroded_screwdriver",
    "wiki_bio_apple_core_01",
    "wiki_bio_banana_peel_bucharest_02",
    "wiki_bio_banana_peel_sidewalk_06",
    "wiki_bio_banana_peel_bus_stop_03",
    "wiki_bio_banana_peel_oak_hill_05",
    "wiki_bio_food_scraps_bin_09",
    "wiki_neg_utility_box_corner_05",
    "wiki_neg_brick_wall_clean",
    "wiki_multi_banana_and_cup_01"
]

def test_downloaded_images_exist_and_within_limit():
    assert IMG_DIR.exists(), f"Image staging directory does not exist: {IMG_DIR}"
    
    total_bytes = 0
    for cid in EXPECTED_CANDIDATE_IDS:
        matched = list(IMG_DIR.glob(f"{cid}.*"))
        assert len(matched) == 1, f"Expected exactly 1 image for {cid}, found {len(matched)}"
        img_p = matched[0]
        size = img_p.stat().st_size
        total_bytes += size
        assert size > 0, f"Image {img_p.name} has 0 bytes"
        
        # Test PIL decode
        with Image.open(img_p) as im:
            assert im.size[0] > 0 and im.size[1] > 0
            assert im.mode in ["RGB", "RGBA"]
            
    assert total_bytes <= MAX_ALLOWED_BYTES, f"Total size {total_bytes} exceeds 50 MiB limit!"
    assert total_bytes == 38506683, f"Total size {total_bytes} does not match expected 38,506,683 bytes"

def test_labels_format_and_clean_negatives():
    assert LBL_DIR.exists(), f"Label staging directory does not exist: {LBL_DIR}"
    
    empty_label_cids = {
        "wiki_neg_utility_box_corner_05",
        "wiki_neg_brick_wall_clean",
        "wiki_bat_corroded_screwdriver",  # QUARANTINED: ambiguous battery/motor assembly
        "wiki_bio_food_scraps_bin_09",    # PENDING_ANNOTATION: multi-object compost bin
    }
    
    total_boxes = 0
    for cid in EXPECTED_CANDIDATE_IDS:
        lbl_p = LBL_DIR / f"{cid}.txt"
        assert lbl_p.exists(), f"Missing label file: {lbl_p}"
        
        content = lbl_p.read_text(encoding="utf-8").strip()
        
        if cid in empty_label_cids:
            assert len(content) == 0, f"Candidate {cid} must have 0-byte label file, got: '{content}'"
        else:
            assert len(content) > 0, f"Positive candidate {cid} must have non-empty annotations"
            lines = content.split("\n")
            total_boxes += len(lines)
            for line in lines:
                parts = line.strip().split()
                assert len(parts) == 5, f"Invalid YOLO format line in {cid}.txt: {line}"
                cls_id = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:]]
                assert 0 <= cls_id <= 9, f"Invalid class ID {cls_id} in {cid}.txt"
                assert 0.0 <= xc <= 1.0, f"Invalid xc {xc} in {cid}.txt"
                assert 0.0 <= yc <= 1.0, f"Invalid yc {yc} in {cid}.txt"
                assert 0.0 < w <= 1.0, f"Invalid w {w} in {cid}.txt"
                assert 0.0 < h <= 1.0, f"Invalid h {h} in {cid}.txt"
                
    assert total_boxes == 7, f"Expected exactly 7 active bounding boxes across mini-batch, got {total_boxes}"

def test_strict_zero_approval_dod():
    assert MANIFEST_PATH.exists(), f"Missing manifest file: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["approved_for_train_count"] == 0, "Strict DoD violated: approved_for_train_count must be 0"
    assert data["total_boxes"] == 7, f"Expected total_boxes=7 in manifest, got {data.get('total_boxes')}"
    assert len(data["items"]) == 10, f"Expected 10 items in manifest, got {len(data['items'])}"
    for item in data["items"]:
        assert item["approved_for_train"] is False, f"Item {item['candidate_id']} has approved_for_train=True!"
        assert item.get("user_train_approval") == "PENDING_PM_APPROVAL" or item.get("training_approval_status") == "PENDING_PM_APPROVAL"
        assert item["technical_valid"] is True
        assert "visual_qa" in item

def test_baseline_correction_artifact():
    corr_path = ROOT / "reports/archive/task_reports/BASELINE_CORRECTION.md"
    if not corr_path.exists():
        corr_path = ROOT / "BASELINE_CORRECTION.md"
    assert corr_path.exists(), "Missing BASELINE_CORRECTION.md"
    content = corr_path.read_text(encoding="utf-8")
    
    # Must document provenance of all key figures
    assert "0.1380" in content
    assert "0.0498" in content
    assert "0.0877" in content or "8.77%" in content
    assert "0.0572" in content
    assert "eval_val_test.json" in content
    assert "7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8" in content
    assert "c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7" in content
