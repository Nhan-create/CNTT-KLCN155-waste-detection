import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from PIL import Image

from src.detection.ssdlite import SSDLiteWasteDetector
from src.app.classifier_service import MobileNetClassifier
from src.detection.two_stage import TwoStageDetector, MobileNetBatchAdapter
from src.detection.tiling import TiledDetector
from src.detection.yolo import WasteDetector
from reports.qa_acceptance.practical_metrics import (
    load_ground_truth,
    box_iou,
    CLASS_NAMES,
)

def compute_matching(pred_objs, gt_objs, conf_threshold=0.25, iou_threshold=0.50):
    """Perform exact class-aware greedy matching at conf >= conf_threshold and IoU >= iou_threshold.
    
    Returns:
    - enriched_preds: list of pred dicts with matching_status, matched_gt_id, matched_iou
    - enriched_gts: list of gt dicts with matched_by_pred_id
    """
    enriched_preds = []
    for p in pred_objs:
        enriched_preds.append({
            **p,
            "status": "PENDING",
            "matched_gt_id": None,
            "matched_gt_class": None,
            "matched_iou": 0.0
        })

    enriched_gts = []
    for g in gt_objs:
        enriched_gts.append({
            **g,
            "matched_by_pred_id": None,
            "status": "UNMATCHED_FN"
        })

    # Filter by confidence
    candidates = []
    for pi, p in enumerate(enriched_preds):
        if p["confidence"] < conf_threshold:
            p["status"] = "FILTERED_CONF"
            continue
        for gi, g in enumerate(enriched_gts):
            if p["class_id"] == g["class_id"]:
                iou = box_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou >= iou_threshold:
                    candidates.append((p["confidence"], iou, pi, gi))

    # Sort greedy by confidence desc, then IoU desc
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    matched_p = set()
    matched_g = set()
    for conf, iou, pi, gi in candidates:
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        
        enriched_preds[pi]["status"] = "MATCHED_TP"
        enriched_preds[pi]["matched_gt_id"] = enriched_gts[gi].get("object_id", f"gt_{gi}")
        enriched_preds[pi]["matched_gt_class"] = enriched_gts[gi].get("class_name", str(enriched_gts[gi]["class_id"]))
        enriched_preds[pi]["matched_iou"] = round(float(iou), 4)

        enriched_gts[gi]["status"] = "MATCHED_TP"
        enriched_gts[gi]["matched_by_pred_id"] = pi

    # Mark remaining >= conf_threshold as FP
    for pi, p in enumerate(enriched_preds):
        if p["status"] == "PENDING":
            p["status"] = "UNMATCHED_FP"
            # Find best overlapping GT regardless of class to provide audit context
            best_iou = 0.0
            best_gi = None
            for gi, g in enumerate(enriched_gts):
                iou = box_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi is not None and best_iou >= 0.10:
                p["matched_gt_id"] = enriched_gts[best_gi].get("object_id", f"gt_{best_gi}")
                p["matched_gt_class"] = enriched_gts[best_gi].get("class_name", str(enriched_gts[best_gi]["class_id"]))
                p["matched_iou"] = round(float(best_iou), 4)
                if p["matched_gt_class"] != p["class_name"] and best_iou >= iou_threshold:
                    p["status"] = "WRONG_CLASS_FP"

    return enriched_preds, enriched_gts

def run_export():
    gt_path = PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_by_filename = {item["filename"]: item for item in gt_data["images"]}

    ckpt_det = PROJECT_ROOT / "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt"
    ckpt_cls = PROJECT_ROOT / "artifacts/official_run/best_model.pt"
    ckpt_yolo = PROJECT_ROOT / "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"

    base_det = SSDLiteWasteDetector(ckpt_det, confidence_threshold=0.20, image_size=320, device="cpu")
    classifier = MobileNetClassifier(ckpt_cls, device="cpu")
    adapter = MobileNetBatchAdapter(classifier)

    # Config 1: Before
    ts_before = TwoStageDetector(base_det, adapter, scoring_mode="blend", blend_weight=0.65, min_proposal_gate=0.0, min_score=0.20)
    tiled_before = TiledDetector(ts_before, tile_size=640, overlap=0.20, min_side_to_tile=640, adaptive=True)

    # Config 2: After
    ts_after = TwoStageDetector(base_det, adapter, scoring_mode="product", min_score=0.20)
    tiled_after = TiledDetector(ts_after, tile_size=640, overlap=0.20, min_side_to_tile=800, adaptive=True)

    # Config 3: YOLOv8n E1
    yolo_det = WasteDetector(ckpt_yolo, confidence_threshold=0.20, image_size=640, device="cpu")

    pipelines = [
        ("raw_predictions_before_quick_fix.json", "BEFORE_QUICK_FIX (Tiling 640 + TwoStage Blend)", tiled_before),
        ("raw_predictions_after_quick_fix.json", "AFTER_QUICK_FIX (Tiling 800 + TwoStage Product)", tiled_after),
        ("raw_predictions_yolov8n_e1.json", "YOLOV8N_E1_RAW", yolo_det),
    ]

    out_dir = PROJECT_ROOT / "reports/qa_acceptance"

    for out_filename, pipe_name, pipe in pipelines:
        print(f"\n=======================================================")
        print(f"Exporting raw predictions for: {pipe_name}")
        images_result = []

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_filtered = 0

        for fn, gt_item in gt_by_filename.items():
            img_p = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout"))) / fn
            with Image.open(img_p) as im:
                res = pipe.detect_pil(im)

            preds = [
                {
                    "pred_id": i,
                    "class_id": d.class_index,
                    "class_name": d.class_id,
                    "confidence": round(float(d.confidence), 4),
                    "bbox_xyxy": [round(float(v), 2) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)]
                }
                for i, d in enumerate(res.detections)
            ]

            enriched_preds, enriched_gts = compute_matching(
                preds, gt_item["objects"], conf_threshold=0.25, iou_threshold=0.50
            )

            tp_cnt = sum(1 for p in enriched_preds if p["status"] == "MATCHED_TP")
            fp_cnt = sum(1 for p in enriched_preds if p["status"] in ("UNMATCHED_FP", "WRONG_CLASS_FP"))
            fn_cnt = sum(1 for g in enriched_gts if g["status"] == "UNMATCHED_FN")
            filt_cnt = sum(1 for p in enriched_preds if p["status"] == "FILTERED_CONF")

            total_tp += tp_cnt
            total_fp += fp_cnt
            total_fn += fn_cnt
            total_filtered += filt_cnt

            images_result.append({
                "filename": fn,
                "image_width": gt_item["width"],
                "image_height": gt_item["height"],
                "metrics_at_conf025_iou050": {
                    "tp": tp_cnt,
                    "fp": fp_cnt,
                    "fn": fn_cnt,
                    "filtered_low_conf": filt_cnt,
                    "total_predictions_generated": len(preds)
                },
                "predictions": enriched_preds,
                "ground_truth": enriched_gts
            })

        out_data = {
            "pipeline_name": pipe_name,
            "evaluation_thresholds": {
                "operating_confidence": 0.25,
                "matching_iou": 0.50
            },
            "summary_metrics": {
                "total_images": len(images_result),
                "total_gt_objects": 57,
                "total_predictions_generated": sum(img["metrics_at_conf025_iou050"]["total_predictions_generated"] for img in images_result),
                "predictions_filtered_low_conf": total_filtered,
                "evaluated_predictions_ge_025": total_tp + total_fp,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": round(total_tp / (total_tp + total_fp), 4) if (total_tp + total_fp) > 0 else 0.0,
                "recall": round(total_tp / 57, 4),
                "f1": round(2 * total_tp / (2 * total_tp + total_fp + total_fn), 4) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
            },
            "images": images_result
        }

        out_file = out_dir / out_filename
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

        print(f"Saved: {out_file}")
        print(f"Summary: Generated={out_data['summary_metrics']['total_predictions_generated']}, "
              f"LowConfFiltered={total_filtered}, Evaluated(TP+FP)={total_tp+total_fp} (TP={total_tp}, FP={total_fp}), "
              f"FN={total_fn}, P={out_data['summary_metrics']['precision']}, R={out_data['summary_metrics']['recall']}")

if __name__ == "__main__":
    run_export()
