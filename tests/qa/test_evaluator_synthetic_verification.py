"""Synthetic verification tests proving mathematical consistency of detection metrics."""

import math
import pytest
from reports.qa_acceptance.practical_metrics import (
    average_precision_101,
    compute_ap_table,
    precision_recall_f1,
)

def test_mathematical_contradiction_proof():
    """Prove that (Precision=1.8%, Recall=10.5%, FP=23, GT=57) is mathematically impossible.
    
    Proof:
    Total GT = 57.
    Recall = TP / GT = 10.526% => TP = round(0.10526 * 57) = 6.
    If FP = 23, Precision = TP / (TP + FP) = 6 / (6 + 23) = 6 / 29 = 20.69% != 1.8%.
    If Precision = 1.8% and TP = 6, FP = (6 / 0.018) - 6 = 327.33.
    If FP = 23 and TP = 2 (SSDLite raw), Precision = 2 / (2 + 23) = 8.0%, Recall = 2 / 57 = 3.51%.
    """
    total_gt = 57
    recall_exp05 = 0.10526
    tp_exp05 = round(recall_exp05 * total_gt)
    assert tp_exp05 == 6
    
    # Under FP = 23, precision cannot be 1.8%
    fp_stated = 23
    precision_with_fp23 = tp_exp05 / (tp_exp05 + fp_stated)
    assert precision_with_fp23 == pytest.approx(6 / 29, rel=1e-3)
    assert precision_with_fp23 > 0.20  # ~20.7%, nowhere near 1.8%
    
    # Under Precision = 1.8%, FP must be ~327
    target_prec = 0.018
    implied_fp = (tp_exp05 / target_prec) - tp_exp05
    assert implied_fp == pytest.approx(327.33, rel=1e-2)
    
    # SSDLite Raw baseline reality: TP=2, FP=23
    tp_ssdlite_raw = 2
    assert tp_ssdlite_raw / total_gt == pytest.approx(0.0351, rel=1e-2)  # Recall 3.5%
    assert tp_ssdlite_raw / (tp_ssdlite_raw + fp_stated) == pytest.approx(0.080, rel=1e-2)  # Precision 8.0%

def test_iou_threshold_sensitivity_synthetic():
    """Demonstrate how changing IoU threshold from 0.10 to 0.20 to 0.50 affects TP/FP counts."""
    # Box A: [0, 0, 100, 100], area = 10000
    # Box B: [0, 0, 100, 60],  area = 6000, inter = 6000, union = 10000 => IoU = 0.60
    # Box C: [0, 0, 100, 30],  area = 3000, inter = 3000, union = 10000 => IoU = 0.30
    # Box D: [0, 0, 100, 15],  area = 1500, inter = 1500, union = 10000 => IoU = 0.15
    gt = {
        "synthetic.jpg": {
            "objects": [
                {"class_id": 0, "bbox_xyxy": [0, 0, 100, 100]}
            ]
        }
    }
    
    # Prediction with IoU = 0.15
    pred_d = {
        "synthetic.jpg": [
            {"class_id": 0, "confidence": 0.90, "bbox_xyxy": [0, 0, 100, 15]}
        ]
    }
    # Prediction with IoU = 0.30
    pred_c = {
        "synthetic.jpg": [
            {"class_id": 0, "confidence": 0.90, "bbox_xyxy": [0, 0, 100, 30]}
        ]
    }
    # Prediction with IoU = 0.60
    pred_b = {
        "synthetic.jpg": [
            {"class_id": 0, "confidence": 0.90, "bbox_xyxy": [0, 0, 100, 60]}
        ]
    }
    
    # At IoU = 0.50 (practical_metrics)
    r_d_50 = precision_recall_f1(gt, pred_d, conf=0.5, iou_thr=0.5)
    r_c_50 = precision_recall_f1(gt, pred_c, conf=0.5, iou_thr=0.5)
    r_b_50 = precision_recall_f1(gt, pred_b, conf=0.5, iou_thr=0.5)
    assert (r_d_50["tp"], r_d_50["fp"]) == (0, 1)  # IoU 0.15 rejected at 0.50
    assert (r_c_50["tp"], r_c_50["fp"]) == (0, 1)  # IoU 0.30 rejected at 0.50
    assert (r_b_50["tp"], r_b_50["fp"]) == (1, 0)  # IoU 0.60 accepted at 0.50
    
    # At IoU = 0.20 (run_baseline_evaluation)
    r_d_20 = precision_recall_f1(gt, pred_d, conf=0.5, iou_thr=0.20)
    r_c_20 = precision_recall_f1(gt, pred_c, conf=0.5, iou_thr=0.20)
    assert (r_d_20["tp"], r_d_20["fp"]) == (0, 1)  # IoU 0.15 rejected at 0.20
    assert (r_c_20["tp"], r_c_20["fp"]) == (1, 0)  # IoU 0.30 accepted at 0.20!
    
    # At IoU = 0.10 (acceptance_evaluator matcher threshold)
    r_d_10 = precision_recall_f1(gt, pred_d, conf=0.5, iou_thr=0.10)
    assert (r_d_10["tp"], r_d_10["fp"]) == (1, 0)  # IoU 0.15 accepted at 0.10!
