"""
Execute rigorous, end-to-end baseline evaluation on data/holdout (27 images, 57 GT objects)
for both SSDLite320-MobileNetV3 and YOLOv8n.
Measures latency, FPS, precision, recall, F1, per-class metrics, mAP50, mAP50-95,
false positives, false negatives, and exports overlay images + standalone visual dashboard.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.detection.schema import DETECTION_CLASS_NAMES, DETECTION_CLASS_NAMES_VI
CLASS_NAMES_VI_DICT = dict(zip(DETECTION_CLASS_NAMES, DETECTION_CLASS_NAMES_VI))
from src.detection.ssdlite import SSDLiteWasteDetector
from src.detection.yolo import WasteDetector
from reports.qa_acceptance.practical_metrics import (
    load_ground_truth,
    compute_ap_table,
    precision_recall_f1,
    localization_class_analysis,
    box_iou,
    CLASS_NAMES,
    NUM_CLASSES,
)
from reports.qa_acceptance.acceptance_evaluator import QAAcceptanceEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GT_PATH = PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json"
MANIFEST_PATH = PROJECT_ROOT / "reports/qa_acceptance/manifest_hinh_anh_rac.json"
OUTPUT_DIR = PROJECT_ROOT / "reports/baseline_demo_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: "#e6194B",  # battery: red
    1: "#3cb44b",  # biological: green
    2: "#ffe119",  # cardboard: yellow
    3: "#4363d8",  # clothes: blue
    4: "#f58231",  # glass: orange
    5: "#911eb4",  # metal: purple
    6: "#42d4f4",  # paper: cyan
    7: "#f032e6",  # plastic: magenta
    8: "#bfef45",  # shoes: lime
    9: "#fabed4",  # trash: pink
}


def draw_gt_and_pred_overlay(
    image: Image.Image,
    gt_objs: List[Dict[str, Any]],
    pred_objs: List[Dict[str, Any]],
    title: str = "",
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    
    # Try font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Draw Ground Truth boxes in GREEN outline
    for obj in gt_objs:
        b = obj["bbox_xyxy"]
        c_name = obj.get("class_name", CLASS_NAMES[obj["class_id"]])
        draw.rectangle(b, outline="#00ff00", width=3)
        label = f"GT: {c_name}"
        draw.text((b[0] + 3, b[1] + 3), label, fill="#00ff00", font=font)

    # Draw Predictions (conf >= 0.25) in solid colored boxes
    for p in pred_objs:
        if p["confidence"] < 0.25:
            continue
        b = p["bbox_xyxy"]
        cid = p["class_id"]
        c_name = CLASS_NAMES[cid]
        color = CLASS_COLORS.get(cid, "#ffffff")
        draw.rectangle(b, outline=color, width=2)
        txt = f"{c_name} {p['confidence']:.2f}"
        
        # text background
        bbox_txt = draw.textbbox((b[0], max(0, b[1] - 18)), txt, font=font)
        draw.rectangle(bbox_txt, fill=color)
        draw.text((b[0] + 2, max(0, b[1] - 18)), txt, fill="#000000", font=font)

    return canvas


def run_model_inference(detector, manifest_files, device_name):
    # Warmup
    dummy = Image.new("RGB", (640, 640), "white")
    detector.detect_pil(dummy)

    latencies = []
    preds_by_file = {}

    for f_info in manifest_files:
        p = Path(f_info["absolute_path"])
        with Image.open(p) as opened:
            prepared = ImageOps.exif_transpose(opened).convert("RGB")

        # Synchronize for CUDA timing if applicable
        start_t = time.perf_counter()
        res = detector.detect_pil(prepared)
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        latencies.append(elapsed_ms)

        pred_list = []
        for det in res.detections:
            pred_list.append({
                "class_id": det.class_index,
                "confidence": round(float(det.confidence), 4),
                "bbox_xyxy": [round(float(v), 2) for v in (det.box.x1, det.box.y1, det.box.x2, det.box.y2)],
            })
        preds_by_file[f_info["filename"]] = pred_list

    return preds_by_file, latencies


def compute_per_class_prf(gt_dict, preds_dict, conf=0.25, iou_thr=0.50):
    per_class_stats = {}
    for c in range(NUM_CLASSES):
        c_name = CLASS_NAMES[c]
        tp = 0
        fp = 0
        fn = 0
        n_gt = 0
        for fn_name, g in gt_dict.items():
            g_objs = [o for o in g["objects"] if o["class_id"] == c]
            p_objs = [p for p in preds_dict.get(fn_name, []) if p["class_id"] == c and p["confidence"] >= conf]
            n_gt += len(g_objs)
            matched_g = set()
            for p in p_objs:
                best_iou = 0.0
                best_gi = None
                for gi, go in enumerate(g_objs):
                    if gi in matched_g:
                        continue
                    iou = box_iou(p["bbox_xyxy"], go["bbox_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_gi is not None and best_iou >= iou_thr:
                    matched_g.add(best_gi)
                    tp += 1
                else:
                    fp += 1
            fn += (len(g_objs) - len(matched_g))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class_stats[c_name] = {
            "class_id": c,
            "vietnamese": CLASS_NAMES_VI_DICT.get(c_name, c_name),
            "n_gt": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    return per_class_stats


def extract_detailed_errors(gt_dict, preds_dict, conf=0.25, iou_thr=0.50):
    false_negatives = []
    false_positives = []
    true_positives = []

    for fn_name, g in gt_dict.items():
        g_objs = g["objects"]
        p_objs = [p for p in preds_dict.get(fn_name, []) if p["confidence"] >= conf]

        matched_p = set()
        matched_g = set()

        # Build candidate matches (same class, IoU >= 0.5)
        candidates = []
        for pi, p in enumerate(p_objs):
            for gi, go in enumerate(g_objs):
                if p["class_id"] == go["class_id"]:
                    iou = box_iou(p["bbox_xyxy"], go["bbox_xyxy"])
                    if iou >= iou_thr:
                        candidates.append((p["confidence"], iou, pi, gi))

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for _, iou, pi, gi in candidates:
            if pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi)
            matched_g.add(gi)
            true_positives.append({
                "filename": fn_name,
                "class_id": g_objs[gi]["class_id"],
                "class_name": CLASS_NAMES[g_objs[gi]["class_id"]],
                "confidence": p_objs[pi]["confidence"],
                "iou": round(float(iou), 3),
                "gt_bbox": g_objs[gi]["bbox_xyxy"],
                "pred_bbox": p_objs[pi]["bbox_xyxy"],
            })

        # Unmatched GT -> FN
        for gi, go in enumerate(g_objs):
            if gi not in matched_g:
                false_negatives.append({
                    "filename": fn_name,
                    "gt_object_id": go.get("object_id", f"gt_{gi}"),
                    "class_id": go["class_id"],
                    "class_name": CLASS_NAMES[go["class_id"]],
                    "gt_bbox": go["bbox_xyxy"],
                    "reason": "Missed by detector (no matching pred box with same class and IoU>=0.50)",
                })

        # Unmatched Pred -> FP
        for pi, p in enumerate(p_objs):
            if pi not in matched_p:
                # check if there is an overlapping GT of different class
                best_iou = 0.0
                best_g_class = None
                for go in g_objs:
                    iou = box_iou(p["bbox_xyxy"], go["bbox_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_g_class = CLASS_NAMES[go["class_id"]]

                error_type = "Wrong class confusion" if best_iou >= iou_thr else "Background hallucination"
                false_positives.append({
                    "filename": fn_name,
                    "predicted_class": CLASS_NAMES[p["class_id"]],
                    "confidence": p["confidence"],
                    "pred_bbox": p["bbox_xyxy"],
                    "overlapping_gt_class": best_g_class if best_iou >= 0.10 else None,
                    "overlapping_iou": round(float(best_iou), 3) if best_iou >= 0.10 else 0.0,
                    "error_type": error_type,
                })

    return true_positives, false_negatives, false_positives


def main():
    print("================================================================================")
    print("STARTING RIGOROUS BASELINE EVALUATION (TASK 3 & TASK 4)")
    print("================================================================================")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    manifest_files = manifest_data["files"]

    gt_dict, _ = load_ground_truth(GT_PATH, dedupe_sha256=False)

    ckpt_yolo = PROJECT_ROOT / "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"
    ckpt_ssd = PROJECT_ROOT / "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt"

    models = [
        ("ssdlite", "SSDLite320-MobileNetV3 (Mô hình chính)", SSDLiteWasteDetector(ckpt_ssd, confidence_threshold=0.01, image_size=320, device="cuda"), ckpt_ssd),
        ("yolov8n", "YOLOv8n (Mô hình đối chứng)", WasteDetector(ckpt_yolo, confidence_threshold=0.01, image_size=640, device="cuda"), ckpt_yolo),
    ]

    all_eval_results = {}

    for model_key, model_display_name, detector, ckpt_path in models:
        print(f"\n>>> Running inference for {model_display_name}...")
        overlay_sub = OUTPUT_DIR / f"overlays_{model_key}"
        overlay_sub.mkdir(parents=True, exist_ok=True)

        preds_by_file, latencies = run_model_inference(detector, manifest_files, "cuda")

        # Also measure CPU latency on 5 images to document CPU vs GPU
        detector_cpu = (
            SSDLiteWasteDetector(ckpt_path, confidence_threshold=0.01, image_size=320, device="cpu")
            if model_key == "ssdlite"
            else WasteDetector(ckpt_path, confidence_threshold=0.01, image_size=640, device="cpu")
        )
        _, cpu_latencies = run_model_inference(detector_cpu, manifest_files[:5], "cpu")
        avg_cpu_lat = float(np.mean(cpu_latencies))

        # Metrics calculation
        ap_res = compute_ap_table(gt_dict, preds_by_file)
        prf_025 = precision_recall_f1(gt_dict, preds_by_file, conf=0.25, iou_thr=0.50, agnostic=False)
        prf_agn = precision_recall_f1(gt_dict, preds_by_file, conf=0.25, iou_thr=0.50, agnostic=True)
        loc_res = localization_class_analysis(gt_dict, preds_by_file, conf=0.25, iou_thr=0.50)
        per_class_prf = compute_per_class_prf(gt_dict, preds_by_file, conf=0.25, iou_thr=0.50)
        tps, fns, fps = extract_detailed_errors(gt_dict, preds_by_file, conf=0.25, iou_thr=0.50)

        # Merge AP into per_class_prf
        for c_name, entry in per_class_prf.items():
            ap_entry = ap_res["per_class"].get(c_name, {})
            entry["ap50"] = round(ap_entry["ap50"], 4) if ap_entry.get("ap50") is not None else 0.0
            entry["ap50_95"] = round(ap_entry["ap50_95"], 4) if ap_entry.get("ap50_95") is not None else 0.0

        # Strict DoD evaluation
        strict_evaluator = QAAcceptanceEvaluator(GT_PATH, MANIFEST_PATH)
        images_for_strict = [
            {
                "filename": fn,
                "objects": [
                    {**p, "class_name": CLASS_NAMES[p["class_id"]]}
                    for p in plist if p["confidence"] >= 0.25
                ],
            }
            for fn, plist in preds_by_file.items()
        ]
        strict_res = strict_evaluator.evaluate({"images": images_for_strict})

        # Save overlays
        print(f"Generating overlays for {model_key}...")
        for f_info in manifest_files:
            fn = f_info["filename"]
            img_p = Path(f_info["absolute_path"])
            with Image.open(img_p) as im:
                canvas = ImageOps.exif_transpose(im).convert("RGB")
            gts = gt_dict[fn]["objects"]
            preds = preds_by_file.get(fn, [])
            overlay = draw_gt_and_pred_overlay(canvas, gts, preds, title=f"{model_key} on {fn}")
            overlay.save(overlay_sub / f"{Path(fn).stem}_overlay_{model_key}.jpg", quality=90)

        mean_gpu_lat = float(np.mean(latencies))
        fps_gpu = round(1000.0 / mean_gpu_lat, 1) if mean_gpu_lat > 0 else 0.0
        fps_cpu = round(1000.0 / avg_cpu_lat, 1) if avg_cpu_lat > 0 else 0.0

        model_summary = {
            "model_key": model_key,
            "display_name": model_display_name,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
            "latency": {
                "gpu_mean_ms": round(mean_gpu_lat, 2),
                "gpu_min_ms": round(float(np.min(latencies)), 2),
                "gpu_max_ms": round(float(np.max(latencies)), 2),
                "gpu_fps": fps_gpu,
                "cpu_sample_mean_ms": round(avg_cpu_lat, 2),
                "cpu_fps": fps_cpu,
            },
            "overall_metrics_conf025_iou050": {
                "total_gt_objects": 57,
                "true_positives": prf_025["tp"],
                "false_positives": prf_025["fp"],
                "false_negatives": prf_025["fn"],
                "precision": round(prf_025["precision"] * 100, 2),
                "recall": round(prf_025["recall"] * 100, 2),
                "f1": round(prf_025["f1"] * 100, 2),
            },
            "class_agnostic_conf025_iou050": {
                "tp": prf_agn["tp"],
                "fp": prf_agn["fp"],
                "fn": prf_agn["fn"],
                "precision": round(prf_agn["precision"] * 100, 2),
                "recall": round(prf_agn["recall"] * 100, 2),
                "f1": round(prf_agn["f1"] * 100, 2),
            },
            "coco_map": {
                "mAP50": round(ap_res["mAP50"] * 100, 2),
                "mAP50_95": round(ap_res["mAP50_95"] * 100, 2),
                "class_agnostic_AP50": round(ap_res["class_agnostic_AP50"] * 100, 2),
            },
            "localization_accuracy": {
                "matched_pairs": loc_res["matched_pairs"],
                "correct_class_pairs": loc_res["correct_class_pairs"],
                "class_accuracy": round(loc_res["class_accuracy_on_matched"] * 100, 2),
            },
            "strict_dod": {
                "status": strict_res["dod_status"],
                "passed": strict_res["dod_status"] == "PASSED",
                "acceptance_criteria": strict_res.get("acceptance_criteria"),
                "overall_statistics": strict_res.get("overall_statistics"),
            },
            "per_class_metrics": per_class_prf,
            "false_negatives_count": len(fns),
            "false_positives_count": len(fps),
            "false_negatives_sample": fns,
            "false_positives_sample": fps,
        }

        all_eval_results[model_key] = model_summary

    # Save complete JSON
    out_json = OUTPUT_DIR / "baseline_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved authoritative baseline report to {out_json}")

    # TASK 2: Specific Functional Test Cases
    print("\n>>> Testing TASK 2 specific cases...")
    task2_dir = OUTPUT_DIR / "task2_cases"
    task2_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        ("single_object", "Một vật thể", (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(PROJECT_ROOT / "data" / "holdout"))) / "battery-sample.png")),
        ("multi_objects", "Nhiều vật thể cùng lúc", (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(PROJECT_ROOT / "data" / "holdout"))) / "images (17).jpg")),
        ("small_far_objects", "Vật thể nhỏ và xa", (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(PROJECT_ROOT / "data" / "holdout"))) / "download (2).jpg")),
        ("no_waste", "Ảnh không có rác", PROJECT_ROOT / "data/detection/images/real/taco_0170.jpg"),
        ("utility_box", "Ảnh tủ điện dễ bị nhận nhầm", PROJECT_ROOT / "scratch_utility_box_bottom_right.jpg"),
    ]

    task2_results = []
    ssd_det = SSDLiteWasteDetector(ckpt_ssd, confidence_threshold=0.25, image_size=320, device="cuda")
    yolo_det = WasteDetector(ckpt_yolo, confidence_threshold=0.25, image_size=640, device="cuda")

    for case_id, case_name, img_path in test_cases:
        img_path = Path(img_path).resolve()

        with Image.open(img_path) as im:
            img = ImageOps.exif_transpose(im).convert("RGB")

        # SSD inference
        t0 = time.perf_counter()
        ssd_res = ssd_det.detect_pil(img)
        ssd_time = (time.perf_counter() - t0) * 1000

        # YOLO inference
        t0 = time.perf_counter()
        yolo_res = yolo_det.detect_pil(img)
        yolo_time = (time.perf_counter() - t0) * 1000

        ssd_dets = [
            {"class": d.class_id, "vietnamese": CLASS_NAMES_VI_DICT.get(d.class_id, d.class_id), "conf": round(float(d.confidence), 3), "box": [round(float(v), 1) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)]}
            for d in ssd_res.detections
        ]
        yolo_dets = [
            {"class": d.class_id, "vietnamese": CLASS_NAMES_VI_DICT.get(d.class_id, d.class_id), "conf": round(float(d.confidence), 3), "box": [round(float(v), 1) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)]}
            for d in yolo_res.detections
        ]

        # Draw overlays
        def draw_simple_overlay(im_in, dets, model_lbl):
            c = im_in.copy()
            d = ImageDraw.Draw(c)
            try:
                f = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                f = ImageFont.load_default()
            for obj in dets:
                b = obj["box"]
                lbl = f"{obj['vietnamese']} ({obj['conf']:.2f})"
                d.rectangle(b, outline="#ff3333" if "yolo" in model_lbl.lower() else "#3388ff", width=3)
                tb = d.textbbox((b[0], max(0, b[1] - 20)), lbl, font=f)
                d.rectangle(tb, fill="#ff3333" if "yolo" in model_lbl.lower() else "#3388ff")
                d.text((b[0] + 2, max(0, b[1] - 20)), lbl, fill="#ffffff", font=f)
            return c

        ssd_img = draw_simple_overlay(img, ssd_dets, "SSDLite")
        yolo_img = draw_simple_overlay(img, yolo_dets, "YOLOv8n")

        ssd_p = task2_dir / f"{case_id}_ssdlite.jpg"
        yolo_p = task2_dir / f"{case_id}_yolov8n.jpg"
        orig_p = task2_dir / f"{case_id}_original.jpg"

        img.save(orig_p, quality=90)
        ssd_img.save(ssd_p, quality=90)
        yolo_img.save(yolo_p, quality=90)

        task2_results.append({
            "case_id": case_id,
            "case_name": case_name,
            "filename": img_path.name,
            "ssdlite": {
                "latency_ms": round(ssd_time, 1),
                "fps": round(1000.0 / ssd_time, 1),
                "detections": ssd_dets,
                "count": len(ssd_dets),
                "image_rel": f"task2_cases/{ssd_p.name}",
            },
            "yolov8n": {
                "latency_ms": round(yolo_time, 1),
                "fps": round(1000.0 / yolo_time, 1),
                "detections": yolo_dets,
                "count": len(yolo_dets),
                "image_rel": f"task2_cases/{yolo_p.name}",
            },
            "original_image_rel": f"task2_cases/{orig_p.name}",
        })

    with open(OUTPUT_DIR / "task2_results.json", "w", encoding="utf-8") as f:
        json.dump(task2_results, f, indent=2, ensure_ascii=False)

    print("TASK 2 functional test cases completed and saved!")


if __name__ == "__main__":
    main()
