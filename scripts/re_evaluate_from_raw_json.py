import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from reports.qa_acceptance.practical_metrics import (
    load_ground_truth,
    compute_ap_table,
    precision_recall_f1,
    localization_class_analysis,
    CLASS_NAMES,
)

def evaluate_raw_file(raw_json_path, gt_dict, operating_conf=0.25):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct predictions dict {filename: [pred, ...]}
    preds_dict = {}
    for img_entry in data["images"]:
        fn = img_entry["filename"]
        preds_dict[fn] = [
            {
                "class_id": p["class_id"],
                "confidence": p["confidence"],
                "bbox_xyxy": p["bbox_xyxy"],
            }
            for p in img_entry["predictions"]
        ]

    # Evaluate using practical_metrics standard functions
    prf_aware = precision_recall_f1(gt_dict, preds_dict, conf=operating_conf, iou_thr=0.50, agnostic=False)
    prf_agn = precision_recall_f1(gt_dict, preds_dict, conf=operating_conf, iou_thr=0.50, agnostic=True)
    loc_analysis = localization_class_analysis(gt_dict, preds_dict, conf=operating_conf, iou_thr=0.50)
    ap_res = compute_ap_table(gt_dict, preds_dict)

    return {
        "pipeline_name": data.get("pipeline_name"),
        "raw_json_path": str(raw_json_path),
        "operating_conf": operating_conf,
        "total_generated_boxes": data["summary_metrics"]["total_predictions_generated"],
        "filtered_low_conf": data["summary_metrics"]["predictions_filtered_low_conf"],
        "boxes_ge_conf": prf_aware["tp"] + prf_aware["fp"],
        "class_aware": {
            "tp": prf_aware["tp"],
            "fp": prf_aware["fp"],
            "fn": prf_aware["fn"],
            "precision": round(prf_aware["precision"], 4),
            "recall": round(prf_aware["recall"], 4),
            "f1": round(prf_aware["f1"], 4),
        },
        "class_agnostic": {
            "tp": prf_agn["tp"],
            "fp": prf_agn["fp"],
            "fn": prf_agn["fn"],
            "precision": round(prf_agn["precision"], 4),
            "recall": round(prf_agn["recall"], 4),
            "f1": round(prf_agn["f1"], 4),
        },
        "localization": {
            "matched_pairs": loc_analysis["matched_pairs"],
            "correct_class_pairs": loc_analysis["correct_class_pairs"],
            "class_accuracy": round(loc_analysis["class_accuracy_on_matched"], 4),
        },
        "coco_ap": {
            "mAP50": round(ap_res["mAP50"], 4),
            "mAP50_95": round(ap_res["mAP50_95"], 4),
            "class_agnostic_AP50": round(ap_res["class_agnostic_AP50"], 4),
        }
    }

def main():
    gt_dict, _ = load_ground_truth(PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
    out_dir = PROJECT_ROOT / "reports/qa_acceptance"

    raw_files = [
        out_dir / "raw_predictions_before_quick_fix.json",
        out_dir / "raw_predictions_after_quick_fix.json",
        out_dir / "raw_predictions_yolov8n_e1.json",
    ]

    all_results = []
    print("================================================================================")
    print("RE-EVALUATING STRICTLY FROM RAW PREDICTIONS JSON FILES (ZERO MARKDOWN RELIANCE)")
    print("================================================================================")

    for rf in raw_files:
        if not rf.exists():
            print(f"Error: {rf} does not exist yet!")
            continue
        res = evaluate_raw_file(rf, gt_dict, operating_conf=0.25)
        all_results.append(res)

        print(f"\n--- Model / Pipeline: {res['pipeline_name']} ---")
        print(f"Source file: {rf.name}")
        print(f"Total boxes generated: {res['total_generated_boxes']} | Filtered (<0.25): {res['filtered_low_conf']} | Evaluated (>=0.25): {res['boxes_ge_conf']}")
        ca = res["class_aware"]
        print(f"Class-Aware  (Conf>=0.25, IoU>=0.50): TP={ca['tp']}, FP={ca['fp']}, FN={ca['fn']} | Precision={ca['precision']*100:.2f}%, Recall={ca['recall']*100:.2f}%, F1={ca['f1']*100:.2f}%")
        cagn = res["class_agnostic"]
        print(f"Class-Agnostic(Conf>=0.25, IoU>=0.50): TP={cagn['tp']}, FP={cagn['fp']}, FN={cagn['fn']} | Precision={cagn['precision']*100:.2f}%, Recall={cagn['recall']*100:.2f}%, F1={cagn['f1']*100:.2f}%")
        loc = res["localization"]
        print(f"Localization Accuracy: {loc['correct_class_pairs']}/{loc['matched_pairs']} ({loc['class_accuracy']*100:.2f}%)")
        cap = res["coco_ap"]
        print(f"COCO AP: mAP50={cap['mAP50']:.4f}, mAP50-95={cap['mAP50_95']:.4f}, Class-Agnostic AP50={cap['class_agnostic_AP50']:.4f}")

    eval_out = out_dir / "verified_evaluation_summary_from_raw.json"
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary successfully written to: {eval_out}")

if __name__ == "__main__":
    main()
