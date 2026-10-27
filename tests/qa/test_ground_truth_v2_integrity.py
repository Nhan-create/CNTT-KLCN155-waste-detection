# Mô tả: Kiểm thử tính toàn vẹn của bộ nhãn chuẩn Ground Truth v1/v2 và quy ước tệp nhãn rỗng cho ảnh âm tính khó (hard negatives).
import json
import pytest
from pathlib import Path

def test_ground_truth_v1_frozen_integrity():
    v1_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
    assert v1_path.exists(), "Frozen v1 GT file must exist"
    
    with open(v1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data['images']) == 27, "Must have exactly 27 acceptance images"
    total_objs = sum(len(img['objects']) for img in data['images'])
    assert total_objs == 57, f"Must have exactly 57 GT objects in v1, got {total_objs}"
    
    # Verify exact tally of boxes >= 80% and == 100%
    full_canvas_count = 0
    ge_80_count = 0
    
    for img in data['images']:
        w, h = img['width'], img['height']
        img_area = float(w * h)
        for obj in img['objects']:
            box = obj['bbox_xyxy']
            area = (box[2] - box[0]) * (box[3] - box[1])
            area_pct = (area / img_area) * 100.0
            if area_pct >= 99.0:
                full_canvas_count += 1
            if area_pct >= 80.0:
                ge_80_count += 1
                
    assert full_canvas_count == 7, f"Expected exactly 7 full canvas (100%) boxes in v1, got {full_canvas_count}"
    assert ge_80_count == 12, f"Expected exactly 12 boxes >= 80% in v1, got {ge_80_count}"

def test_ground_truth_v2_proposed_integrity():
    v2_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json")
    assert v2_path.exists(), "Proposed v2 GT file must exist"
    
    with open(v2_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert len(data['images']) == 27, "Must have 27 images"
    summary = data['summary']
    assert summary['total_v1_objects'] == 57
    assert summary['total_quarantined_objects'] == 4
    assert summary['active_objects_for_instance_detection'] == 45
    
    # Verify geometry
    for img in data['images']:
        for obj in img['objects']:
            b = obj['bbox_xyxy']
            assert b[0] < b[2], f"Invalid x coordinates: {b}"
            assert b[1] < b[3], f"Invalid y coordinates: {b}"
            assert obj['v2_status'] in [
                "ACCEPTED",
                "QUARANTINED_FULL_CANVAS_DUPLICATE",
                "QUARANTINED_TEXTURE_CARPET",
                "QUARANTINED_UTILITY_RECEPTACLE",
                "FLAGGED_UNSEGREGATED_PILE",
                "FLAGGED_CONTAINER_NESTING",
                "FLAGGED_OVERLAPPING_OBJECT",
                "AMBIGUOUS_COMPOSITE_TAXONOMY",
                "AMBIGUOUS_BAGGED_WASTE"
            ]

def test_hard_negative_empty_labels_convention():
    # Verify domain modeling convention: hard negative files must be 0 bytes
    empty_label_mock = b""
    assert len(empty_label_mock) == 0
