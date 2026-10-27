# Mô tả: Kiểm thử đơn vị có tính tay đối chứng cho module tính chỉ số thực nghiệm AP và Precision/Recall trên tập holdout.
"""Hand-computed unit tests for reports/qa_acceptance/practical_metrics.py."""
import json
import math
from pathlib import Path

import pytest

from reports.qa_acceptance.practical_metrics import (
    BACKGROUND_INDEX,
    average_precision_101,
    compute_ap_table,
    evaluate_practical,
    load_ground_truth,
    load_predictions,
    localization_class_analysis,
    precision_recall_f1,
    predictions_from_baseline,
    run,
)


def P(c, conf, box):
    return {"class_id": c, "confidence": conf, "bbox_xyxy": box}


def G(*objs):
    return {"objects": [{"class_id": c, "bbox_xyxy": b} for c, b in objs]}


BOX_A = [0, 0, 10, 10]
BOX_B = [20, 20, 30, 30]
BOX_C = [50, 50, 60, 60]


# ----------------------------- AP primitives ------------------------------ #
def test_ap101_hand_computed():
    # TP, FP, TP with 2 positives: recall .5,.5,1 ; precision 1,.5,.667
    # envelope 1,.667,.667 -> 51 points at 1.0, 50 points at 2/3
    ap = average_precision_101([0.9, 0.8, 0.7], [True, False, True], 2)
    assert ap == pytest.approx((51 * 1.0 + 50 * (2 / 3)) / 101)


def test_ap101_edge_cases():
    assert average_precision_101([], [], 3) == 0.0
    assert math.isnan(average_precision_101([0.5], [False], 0))
    assert average_precision_101([0.9, 0.8], [True, True], 2) == pytest.approx(1.0)


def test_perfect_predictions_ap_is_one():
    gt = {"a.jpg": G((0, BOX_A), (4, BOX_B)), "b.jpg": G((7, BOX_C))}
    preds = {"a.jpg": [P(0, 0.9, BOX_A), P(4, 0.8, BOX_B)], "b.jpg": [P(7, 0.7, BOX_C)]}
    t = compute_ap_table(gt, preds)
    assert t["mAP50"] == pytest.approx(1.0)
    assert t["mAP50_95"] == pytest.approx(1.0)
    assert t["class_agnostic_AP50"] == pytest.approx(1.0)
    assert t["classes_in_mean"] == 3  # classes without GT excluded
    assert t["per_class"]["metal"]["ap50"] is None
    r = evaluate_practical(gt, preds, conf=0.25)
    assert r["practical_acceptance"]["status"] == "PASS"
    assert r["operating_point"]["class_aware"]["f1"] == pytest.approx(1.0)


def test_no_predictions_gives_zero():
    gt = {"a.jpg": G((0, BOX_A))}
    t = compute_ap_table(gt, {"a.jpg": []})
    assert t["mAP50"] == 0.0 and t["mAP50_95"] == 0.0 and t["class_agnostic_AP50"] == 0.0
    prf = precision_recall_f1(gt, {"a.jpg": []}, conf=0.0)
    assert (prf["tp"], prf["fp"], prf["fn"], prf["precision"], prf["recall"]) == (0, 0, 1, 0.0, 0.0)


def test_missing_image_counts_as_fn_and_fails_coverage():
    gt = {"a.jpg": G((0, BOX_A)), "b.jpg": G((1, BOX_B))}
    r = evaluate_practical(gt, {"a.jpg": [P(0, 0.9, BOX_A)]}, conf=0.25)
    assert r["missing_prediction_images"] == ["b.jpg"]
    assert r["operating_point"]["class_aware"]["recall"] == pytest.approx(0.5)
    assert r["practical_acceptance"]["checks"]["coverage_all_images_predicted"]["passed"] is False


# ----------------------------- wrong class -------------------------------- #
def test_wrong_class_localised_but_misclassified():
    gt = {"a.jpg": G((4, BOX_A))}            # glass
    preds = {"a.jpg": [P(5, 0.9, BOX_A)]}    # metal, same box
    t = compute_ap_table(gt, preds)
    assert t["per_class"]["glass"]["ap50"] == 0.0
    assert t["per_class"]["metal"]["ap50"] is None  # no metal GT
    assert t["mAP50"] == 0.0
    assert t["class_agnostic_AP50"] == pytest.approx(1.0)
    loc = localization_class_analysis(gt, preds, conf=0.0)
    assert loc["matched_pairs"] == 1 and loc["class_accuracy_on_matched"] == 0.0
    cm = loc["confusion_matrix"]["matrix"]
    assert cm[4][5] == 1 and sum(map(sum, cm)) == 1
    prf = precision_recall_f1(gt, preds, conf=0.0)
    assert (prf["tp"], prf["fp"], prf["fn"]) == (0, 1, 1)
    agn = precision_recall_f1(gt, preds, conf=0.0, agnostic=True)
    assert (agn["tp"], agn["fp"], agn["fn"]) == (1, 0, 0)


def test_confusion_matrix_background_cells():
    gt = {"a.jpg": G((0, BOX_A), (2, BOX_B))}
    preds = {"a.jpg": [P(0, 0.9, BOX_A), P(9, 0.8, BOX_C)]}  # BOX_B missed, BOX_C spurious
    loc = localization_class_analysis(gt, preds, conf=0.0)
    cm = loc["confusion_matrix"]["matrix"]
    assert cm[0][0] == 1
    assert cm[2][BACKGROUND_INDEX] == 1
    assert cm[BACKGROUND_INDEX][9] == 1
    assert loc["class_accuracy_on_matched"] == pytest.approx(1.0)


# ----------------------------- duplicate box ------------------------------ #
def test_duplicate_box_is_one_tp_one_fp():
    gt = {"a.jpg": G((0, BOX_A))}
    preds = {"a.jpg": [P(0, 0.9, BOX_A), P(0, 0.8, BOX_A)]}
    # Highest-confidence copy is TP at recall 1 with precision 1 -> AP = 1
    assert compute_ap_table(gt, preds)["mAP50"] == pytest.approx(1.0)
    prf = precision_recall_f1(gt, preds, conf=0.0)
    assert (prf["tp"], prf["fp"], prf["fn"]) == (1, 1, 0)
    assert prf["precision"] == pytest.approx(0.5) and prf["recall"] == pytest.approx(1.0)
    assert prf["f1"] == pytest.approx(2 / 3)
    loc = localization_class_analysis(gt, preds, conf=0.0)
    assert loc["matched_pairs"] == 1
    assert loc["confusion_matrix"]["matrix"][BACKGROUND_INDEX][0] == 1


# ----------------------------- IoU thresholds ----------------------------- #
def test_map50_95_partial_iou():
    gt = {"a.jpg": G((0, BOX_A))}
    preds = {"a.jpg": [P(0, 0.9, [0, 0, 10, 6.2])]}  # IoU 0.62 -> matches t=.50,.55,.60
    t = compute_ap_table(gt, preds)
    assert t["mAP50"] == pytest.approx(1.0)
    assert t["per_class"]["battery"]["ap75"] == 0.0
    assert t["mAP50_95"] == pytest.approx(0.3)


def test_iou_below_half_is_not_a_match():
    gt = {"a.jpg": G((0, BOX_A))}
    preds = {"a.jpg": [P(0, 0.9, [0, 0, 10, 4])]}  # IoU 0.4
    assert compute_ap_table(gt, preds)["mAP50"] == 0.0
    assert localization_class_analysis(gt, preds, conf=0.0)["matched_pairs"] == 0


# ----------------------------- confidence thresholds ---------------------- #
def test_multiple_confidence_thresholds():
    gt = {"a.jpg": G((0, BOX_A), (0, BOX_B))}
    preds = {"a.jpg": [P(0, 0.9, BOX_A), P(0, 0.5, BOX_C), P(0, 0.3, BOX_B)]}
    hi = precision_recall_f1(gt, preds, conf=0.4)
    assert (hi["tp"], hi["fp"], hi["fn"]) == (1, 1, 1)
    assert hi["precision"] == pytest.approx(0.5) and hi["recall"] == pytest.approx(0.5)
    lo = precision_recall_f1(gt, preds, conf=0.2)
    assert (lo["tp"], lo["fp"], lo["fn"]) == (2, 1, 0)
    assert lo["precision"] == pytest.approx(2 / 3) and lo["recall"] == pytest.approx(1.0)
    top = precision_recall_f1(gt, preds, conf=0.95)
    assert (top["tp"], top["fp"], top["fn"]) == (0, 0, 2)
    r = evaluate_practical(gt, preds, conf=0.2, sweep=(0.2, 0.4))
    assert [s["recall"] for s in r["confidence_sweep"]] == pytest.approx([1.0, 0.5])
    # AP: TP(.9), FP(.5), TP(.3) with 2 positives
    assert r["ap"]["mAP50"] == pytest.approx((51 + 50 * (2 / 3)) / 101)


def test_custom_criteria_fail():
    gt = {"a.jpg": G((0, BOX_A), (0, BOX_B))}
    preds = {"a.jpg": [P(0, 0.9, BOX_A)]}
    r = evaluate_practical(gt, preds, conf=0.25, criteria={"min_recall_at_iou50": 0.6})
    assert r["practical_acceptance"]["checks"]["recall_at_iou50"]["passed"] is False
    assert r["practical_acceptance"]["status"] == "FAIL"
    r2 = evaluate_practical(gt, preds, conf=0.25, criteria={"min_recall_at_iou50": 0.5})
    assert r2["practical_acceptance"]["status"] == "PASS"


# ----------------------------- IO / formats ------------------------------- #
def test_baseline_reconstruction_and_formats(tmp_path: Path):
    baseline = {
        "backend": "x", "confidence_threshold": 0.2, "cases": [{
            "filename": "a.jpg", "pred_count": 2,
            "matching": {
                "matched_pairs": [{"pred_class_id": 0, "pred_confidence": 0.9,
                                   "pred_bbox": BOX_A}],
                "false_positives": [{"class_id": 3, "class_name": "clothes",
                                     "confidence": 0.4, "bbox_xyxy": BOX_B}],
            }}]}
    preds = predictions_from_baseline(baseline)
    assert preds["a.jpg"] == [P(0, 0.9, [0.0, 0.0, 10.0, 10.0]), P(3, 0.4, [20.0, 20.0, 30.0, 30.0])]
    baseline["cases"][0]["pred_count"] = 3
    with pytest.raises(ValueError):
        predictions_from_baseline(baseline)

    simple = tmp_path / "simple.json"
    simple.write_text(json.dumps({"a.jpg": [P(0, 0.9, BOX_A)]}), encoding="utf-8")
    p, info = load_predictions(simple)
    assert info["format"] == "simple" and p["a.jpg"][0]["class_id"] == 0

    images = tmp_path / "images.json"
    images.write_text(json.dumps({"images": [{"filename": "a.jpg",
                                              "objects": [P(1, 0.5, BOX_A)]}]}), encoding="utf-8")
    p, info = load_predictions(images)
    assert info["format"] == "evaluator_images" and p["a.jpg"][0]["class_id"] == 1


def test_ground_truth_dedupe(tmp_path: Path):
    gt_file = tmp_path / "gt.json"
    obj = {"class_id": 0, "class_name": "battery", "bbox_xyxy": BOX_A}
    gt_file.write_text(json.dumps({"metadata": {}, "images": [
        {"filename": "x.jpg", "sha256": "s1", "objects": [obj]},
        {"filename": "y.jpg", "sha256": "s1", "objects": [obj]},
        {"filename": "z.jpg", "sha256": "s2", "objects": []},
    ]}), encoding="utf-8")
    full, ex = load_ground_truth(gt_file)
    assert len(full) == 3 and ex == []
    dd, ex = load_ground_truth(gt_file, dedupe_sha256=True)
    assert sorted(dd) == ["x.jpg", "z.jpg"] and ex == ["y.jpg"]


def test_real_holdout_dedupe_drops_one_image():
    gt, ex = load_ground_truth(dedupe_sha256=True)
    assert len(gt) == 26 and len(ex) == 1


def test_real_baseline_reconstructs():
    path = Path("reports/qa_acceptance/baseline_results_ssdlite.json")
    if not path.exists():
        pytest.skip("baseline file missing")
    res = run(path, None, conf=0.25)
    assert res["views"]["all_27"]["n_images"] == 27
    assert res["views"]["dedup_sha256"]["n_images"] == 26
    assert res["views"]["all_27"]["warnings"]
