# Mô tả: Kiểm thử module đánh giá nghiệm thu DoD (acceptance_evaluator.py), tính toán IoU và kiểm tra tính toàn vẹn Ground Truth.
import json
import pytest
from pathlib import Path
from reports.qa_acceptance.acceptance_evaluator import (
    QAAcceptanceEvaluator,
    compute_iou,
    compute_intersection_over_gt,
    validate_bbox
)

MANIFEST_PATH = Path("reports/qa_acceptance/manifest_hinh_anh_rac.json")
GT_PATH = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")

def test_manifest_integrity():
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["metadata"]["total_files_scanned"] == 27
    assert data["metadata"]["valid_images_count"] == 27
    assert data["metadata"]["corrupt_images_count"] == 0
    assert len(data["files"]) == 27
    
    for item in data["files"]:
        assert item["status"] == "VALID"
        assert item["pil_decode"]["success"] is True
        assert item["opencv_decode"]["success"] is True
        assert len(item["sha256"]) == 64

def test_ground_truth_integrity():
    assert GT_PATH.exists(), f"Missing GT at {GT_PATH}"
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
    
    assert gt["metadata"]["total_images"] == 27
    assert gt["metadata"]["reviewer"] == "Agent_D_QA"
    assert len(gt["images"]) == 27
    
    all_obj_ids = set()
    for img in gt["images"]:
        w, h = img["width"], img["height"]
        assert w > 0 and h > 0
        for obj in img["objects"]:
            oid = obj["object_id"]
            assert oid not in all_obj_ids, f"Duplicate object ID: {oid}"
            all_obj_ids.add(oid)
            
            cname = obj["class_name"]
            cid = obj["class_id"]
            assert cid in range(10)
            assert gt["metadata"]["classes"][cid] == cname
            assert obj["reviewer"] == "Agent_D_QA"
            assert len(obj["review_rationale"]) > 5
            
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h

def test_compute_iou_and_inter_gt():
    b1 = [0, 0, 100, 100]
    b2 = [0, 0, 100, 100]
    assert compute_iou(b1, b2) == 1.0
    assert compute_intersection_over_gt(b1, b2) == 1.0

    # Half overlap
    b3 = [0, 0, 100, 50]
    # Inter = 100*50 = 5000, union = 10000 + 5000 - 5000 = 10000 -> IoU = 0.5
    assert compute_iou(b1, b3) == 0.5
    # Inter / area(b1) = 5000 / 10000 = 0.5
    assert compute_intersection_over_gt(b3, b1) == 0.5
    # Inter / area(b3) = 5000 / 5000 = 1.0 (b1 encloses b3)
    assert compute_intersection_over_gt(b1, b3) == 1.0

def test_validate_bbox():
    ok, _ = validate_bbox([10, 10, 50, 50], 100, 100)
    assert ok is True
    
    # Inverted x
    bad, msg = validate_bbox([50, 10, 10, 50], 100, 100)
    assert bad is False
    
    # Out of bounds
    bad, msg = validate_bbox([10, 10, 150, 50], 100, 100)
    assert bad is False

def test_evaluator_perfect_predictions():
    evaluator = QAAcceptanceEvaluator(GT_PATH, MANIFEST_PATH)
    
    # Create perfect predictions by mirroring GT
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
        
    perfect_preds = {
        "images": [
            {
                "filename": img["filename"],
                "objects": [
                    {
                        "object_id": f"pred_{obj['object_id']}",
                        "class_name": obj["class_name"],
                        "class_id": obj["class_id"],
                        "bbox_xyxy": list(obj["bbox_xyxy"]),
                        "confidence": 0.99
                    }
                    for obj in img["objects"]
                ]
            }
            for img in gt["images"]
        ]
    }
    
    summary = evaluator.evaluate(perfect_preds)
    assert summary["dod_status"] == "PASS"
    assert summary["acceptance_criteria"]["fn_actual"] == 0
    assert summary["acceptance_criteria"]["fp_actual"] == 0
    assert summary["acceptance_criteria"]["wrong_class_actual"] == 0
    assert summary["acceptance_criteria"]["iou_violations_count"] == 0
    assert summary["acceptance_criteria"]["inter_gt_violations_count"] == 0
    assert summary["overall_statistics"]["evaluated_images"] == 27
    assert summary["overall_statistics"]["mean_iou_matched"] == 1.0

def test_evaluator_catches_fn():
    evaluator = QAAcceptanceEvaluator(GT_PATH, MANIFEST_PATH)
    # Empty predictions
    empty_preds = {"images": [{"filename": f["filename"], "objects": []} for f in evaluator.manifest_data["files"]]}
    summary = evaluator.evaluate(empty_preds)
    assert summary["dod_status"] == "FAIL"
    assert summary["acceptance_criteria"]["fn_actual"] > 0
    assert summary["acceptance_criteria"]["fn_passed"] is False

def test_evaluator_catches_wrong_class():
    evaluator = QAAcceptanceEvaluator(GT_PATH, MANIFEST_PATH)
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
        
    preds = {
        "images": [
            {
                "filename": img["filename"],
                "objects": [
                    {
                        "object_id": f"pred_{obj['object_id']}",
                        "class_name": "trash",
                        "class_id": 9,  # corrupt all to trash (9)
                        "bbox_xyxy": list(obj["bbox_xyxy"]),
                        "confidence": 0.99
                    }
                    for obj in img["objects"]
                ]
            }
            for img in gt["images"]
        ]
    }
    summary = evaluator.evaluate(preds)
    assert summary["dod_status"] == "FAIL"
    assert summary["acceptance_criteria"]["wrong_class_actual"] > 0
    assert summary["acceptance_criteria"]["wrong_class_passed"] is False

def test_evaluator_catches_iou_violation():
    evaluator = QAAcceptanceEvaluator(GT_PATH, MANIFEST_PATH)
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)
        
    # Shift boxes slightly so IoU drops below 0.75
    shifted_preds = {
        "images": [
            {
                "filename": img["filename"],
                "objects": [
                    {
                        "object_id": f"pred_{obj['object_id']}",
                        "class_name": obj["class_name"],
                        "class_id": obj["class_id"],
                        "bbox_xyxy": [
                            obj["bbox_xyxy"][0] + (obj["bbox_xyxy"][2] - obj["bbox_xyxy"][0]) * 0.3,
                            obj["bbox_xyxy"][1],
                            obj["bbox_xyxy"][2],
                            obj["bbox_xyxy"][3]
                        ],
                        "confidence": 0.99
                    }
                    for obj in img["objects"]
                ]
            }
            for img in gt["images"]
        ]
    }
    summary = evaluator.evaluate(shifted_preds)
    assert summary["dod_status"] == "FAIL"
    assert summary["acceptance_criteria"]["iou_violations_count"] > 0
