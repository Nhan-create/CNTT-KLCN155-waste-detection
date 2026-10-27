# Mô tả: Kiểm thử logic khớp khung bao cùng lớp và chính sách tính dương tính giả trên ảnh âm tính khi so sánh pipeline v1 và v2.
"""
tests/qa/test_compare_eval_v1_vs_v2.py
---------------------------------------
Regression tests for the thresholded, one-to-one object matcher
supporting NEGATIVE, EXCLUDED, and IGNORE REGION evaluation policies.
"""

import json
from pathlib import Path
from scripts.compare_eval_v1_vs_v2 import evaluate_pipeline

ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "reports" / "qa_acceptance"


def test_best_match_must_have_the_prediction_class():
    gt = {"images": [{"filename": "case.jpg", "objects": [
        {"class_id": 7, "bbox_xyxy": [0, 0, 100, 100]},
        {"class_id": 6, "bbox_xyxy": [0, 0, 80, 80]},
    ]}]}
    raw = {"images": [{"filename": "case.jpg", "predictions": [
        {"class_id": 6, "confidence": 0.9, "bbox_xyxy": [0, 0, 100, 100]},
    ]}]}

    result = evaluate_pipeline(raw, gt)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)


def test_real_before_quick_fix_v1_matching():
    raw = json.loads((QA / "raw_predictions_before_quick_fix.json").read_text(encoding="utf-8"))
    gt = json.loads((QA / "ground_truth_hinh_anh_rac.json").read_text(encoding="utf-8"))

    result = evaluate_pipeline(raw, gt, conf_thresh=0.25, iou_thresh=0.5)
    assert (result["total_gt"], result["total_preds"]) == (57, 21)
    assert (result["tp"], result["fp"], result["fn"]) == (13, 8, 44)


def test_negative_benchmark_policy_counts_fp():
    """Negative benchmark images have 0 active GT; any detection must be penalized as FP."""
    gt = {
        "metadata": {"version": "v2.0-proposed"},
        "images": [{
            "filename": "clean_street.jpg",
            "evaluation_policy": "NEGATIVE_BENCHMARK",
            "objects": [
                {"object_id": "bin01", "class_id": 7, "bbox_xyxy": [10, 10, 50, 50],
                 "v2_status": "QUARANTINED_UTILITY_RECEPTACLE", "is_active_for_instance_eval": False}
            ]
        }]
    }
    raw = {
        "images": [{
            "filename": "clean_street.jpg",
            "predictions": [
                {"class_id": 7, "confidence": 0.8, "bbox_xyxy": [12, 12, 48, 48]}
            ]
        }]
    }
    result = evaluate_pipeline(raw, gt, receptacle_policy="negative")
    assert result["total_gt"] == 0
    assert result["total_preds"] == 1
    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_excluded_image_policy_discards_predictions():
    """Excluded images (e.g. unsegregated bulk dumps) should not count GT or predictions."""
    gt = {
        "metadata": {"version": "v2.0-proposed"},
        "images": [{
            "filename": "dump_pile.jpg",
            "evaluation_policy": "UNSEGREGATED_PILE_EXCLUDED",
            "objects": [
                {"object_id": "pile01", "class_id": 2, "bbox_xyxy": [0, 0, 500, 500],
                 "v2_status": "FLAGGED_UNSEGREGATED_PILE", "is_active_for_instance_eval": False}
            ]
        }]
    }
    raw = {
        "images": [{
            "filename": "dump_pile.jpg",
            "predictions": [
                {"class_id": 2, "confidence": 0.85, "bbox_xyxy": [10, 10, 200, 200]}
            ]
        }]
    }
    result = evaluate_pipeline(raw, gt, pile_policy="excluded")
    assert result["total_gt"] == 0
    assert result["total_preds"] == 0
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0


def test_ignore_region_policy_discards_contained_prediction():
    """Predictions falling inside designated ignore-regions should be discarded rather than penalized."""
    gt = {
        "metadata": {"version": "v2.0-proposed"},
        "images": [{
            "filename": "dump_pile.jpg",
            "evaluation_policy": "UNSEGREGATED_PILE_EXCLUDED",
            "objects": [
                {"object_id": "pile01", "class_id": 2, "bbox_xyxy": [0, 0, 500, 500],
                 "v2_status": "FLAGGED_UNSEGREGATED_PILE", "is_active_for_instance_eval": False,
                 "is_ignore_region": True}
            ]
        }]
    }
    raw = {
        "images": [{
            "filename": "dump_pile.jpg",
            "predictions": [
                {"class_id": 2, "confidence": 0.85, "bbox_xyxy": [50, 50, 200, 200]}  # inside pile
            ]
        }]
    }
    result = evaluate_pipeline(raw, gt, pile_policy="ignore_region")
    assert result["total_gt"] == 0
    assert result["total_preds"] == 0
    assert result["ignored_preds"] == 1
    assert result["tp"] == 0
    assert result["fp"] == 0


def test_ignore_region_policy_penalizes_outside_prediction():
    """Predictions falling outside ignore-regions on a pile image should still count as FP."""
    gt = {
        "metadata": {"version": "v2.0-proposed"},
        "images": [{
            "filename": "dump_pile.jpg",
            "evaluation_policy": "UNSEGREGATED_PILE_EXCLUDED",
            "objects": [
                {"object_id": "pile01", "class_id": 2, "bbox_xyxy": [0, 0, 200, 200],  # pile only in top-left
                 "v2_status": "FLAGGED_UNSEGREGATED_PILE", "is_active_for_instance_eval": False,
                 "is_ignore_region": True}
            ]
        }]
    }
    raw = {
        "images": [{
            "filename": "dump_pile.jpg",
            "predictions": [
                {"class_id": 5, "confidence": 0.85, "bbox_xyxy": [300, 300, 450, 450]}  # completely outside pile
            ]
        }]
    }
    result = evaluate_pipeline(raw, gt, pile_policy="ignore_region")
    assert result["total_gt"] == 0
    assert result["total_preds"] == 1
    assert result["ignored_preds"] == 0
    assert result["fp"] == 1


def test_v2_proposed_policy_on_real_predictions():
    """Verify exact counts on real SSDLite After Quick Fix predictions with proposed candidate policies."""
    raw = json.loads((QA / "raw_predictions_after_quick_fix.json").read_text(encoding="utf-8"))
    gt = json.loads((QA / "ground_truth_hinh_anh_rac_v2_proposed.json").read_text(encoding="utf-8"))

    # Proposed Candidate Policy (pile excluded, receptacle negative)
    result = evaluate_pipeline(raw, gt, conf_thresh=0.25, iou_thresh=0.5,
                               v2_filter="all_active", pile_policy="excluded", receptacle_policy="negative")
    assert result["total_gt"] == 45
    assert result["total_preds"] == 10
    assert result["tp"] == 7
    assert result["fp"] == 3
    assert result["fn"] == 38
    assert result["precision"] == 70.0
    assert result["recall"] == 15.56
