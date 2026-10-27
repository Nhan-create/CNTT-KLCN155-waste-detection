# Mô tả: Chạy đánh giá đối chứng cơ sở (baseline evaluation) trên tập data/holdout cho cả SSDLite320 và YOLOv8n.
"""
Run Authoritative Baseline Evaluation on data/holdout for both SSDLite320 and YOLOv8n.
Computes 1-to-1 matching with Ground Truth, per-object metrics, and DoD compliance verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.schema import DETECTION_CLASS_NAMES, DETECTION_CLASS_NAMES_VI
from src.detection.ssdlite import SSDLiteWasteDetector
from src.detection.yolo import WasteDetector
from src.detection.visualization import draw_detections, detection_label
from src.detection.types import BoundingBox, Detection, DetectionResult


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_iou(box1: list[float], box2: list[float]) -> tuple[float, float]:
    """Compute IoU and (intersection_area / gt_area) coverage for two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0
    gt_coverage = inter_area / area2 if area2 > 0 else 0.0
    return iou, gt_coverage


def match_predictions_to_gt(
    pred_detections: list[dict[str, Any]],
    gt_objects: list[dict[str, Any]],
    iou_match_threshold: float = 0.50,
) -> dict[str, Any]:
    """1-to-1 greedy matching between predictions and GT objects sorted by confidence/IoU."""
    unmatched_gt = list(range(len(gt_objects)))
    unmatched_pred = list(range(len(pred_detections)))

    pairs = []
    # Calculate candidate matrix
    candidates = []
    for p_idx in unmatched_pred:
        p_box = pred_detections[p_idx]["bbox_xyxy"]
        for g_idx in unmatched_gt:
            g_box = gt_objects[g_idx]["bbox_xyxy"]
            iou, cov = compute_iou(p_box, g_box)
            candidates.append((iou, cov, p_idx, g_idx))

    # Sort descending by IoU
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_p = set()
    matched_g = set()

    for iou, cov, p_idx, g_idx in candidates:
        if p_idx in matched_p or g_idx in matched_g:
            continue
        if iou >= iou_match_threshold:
            matched_p.add(p_idx)
            matched_g.add(g_idx)
            pred = pred_detections[p_idx]
            gt = gt_objects[g_idx]
            is_class_correct = (pred["class_id"] == gt["class_id"])
            pairs.append({
                "pred_index": p_idx,
                "gt_object_id": gt["object_id"],
                "gt_class_id": gt["class_id"],
                "gt_class_name": gt["class_name"],
                "gt_bbox": gt["bbox_xyxy"],
                "pred_class_id": pred["class_id"],
                "pred_class_name": pred["class_name"],
                "pred_confidence": pred["confidence"],
                "pred_bbox": pred["bbox_xyxy"],
                "iou": iou,
                "gt_coverage": cov,
                "is_class_correct": is_class_correct,
                "meets_dod_box": (iou >= 0.75 and cov >= 0.99),
            })

    unmatched_gt_objects = [gt_objects[g_idx] for g_idx in range(len(gt_objects)) if g_idx not in matched_g]
    unmatched_predictions = [pred_detections[p_idx] for p_idx in range(len(pred_detections)) if p_idx not in matched_p]

    return {
        "matched_pairs": pairs,
        "false_negatives": unmatched_gt_objects,
        "false_positives": unmatched_predictions,
        "tp_count": sum(1 for p in pairs if p["is_class_correct"]),
        "wrong_class_count": sum(1 for p in pairs if not p["is_class_correct"]),
        "fn_count": len(unmatched_gt_objects),
        "fp_count": len(unmatched_predictions),
    }


def run_evaluation(
    backend: str,
    model_path: Path,
    manifest_path: Path,
    gt_path: Path,
    output_dir: Path,
    confidence_threshold: float = 0.20,
    iou_threshold: float = 0.45,
    device: str = "auto",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / f"overlays_{backend}"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    gt_by_filename = {item["filename"]: item["objects"] for item in ground_truth["images"]}

    ckpt_hash = sha256_file(model_path)
    print(f"Loading {backend} model from {model_path} (SHA-256: {ckpt_hash[:12]}...)")

    if backend == "ssdlite":
        detector = SSDLiteWasteDetector(
            model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=320,
        )
    elif backend == "yolov8n":
        detector = WasteDetector(
            model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=640,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    case_results = []
    total_images = len(manifest["files"])
    total_gt_objects = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_wrong_class = 0
    box_failures = 0

    print(f"Starting evaluation on {total_images} images with backend={backend}...")

    for file_info in manifest["files"]:
        filename = file_info["filename"]
        img_path = Path(file_info["absolute_path"])
        gt_objects = gt_by_filename.get(filename, [])
        total_gt_objects += len(gt_objects)

        with Image.open(img_path) as opened:
            prepared = ImageOps.exif_transpose(opened).convert("RGB")
            w, h = prepared.size

        # Run inference
        start_t = time.perf_counter()
        result = detector.detect_pil(prepared)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        # Extract predictions
        preds = []
        for det in result.detections:
            box = det.box
            preds.append({
                "class_id": det.class_index,
                "class_name": det.class_id,
                "confidence": det.confidence,
                "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
            })

        # Match to GT
        matching = match_predictions_to_gt(preds, gt_objects, iou_match_threshold=0.20)
        total_tp += matching["tp_count"]
        total_wrong_class += matching["wrong_class_count"]
        total_fn += matching["fn_count"]
        total_fp += matching["fp_count"]

        for pair in matching["matched_pairs"]:
            if not pair["meets_dod_box"]:
                box_failures += 1

        # Render overlay image
        annotated = draw_detections(prepared, result, style="box")
        annotated_path = overlay_dir / f"{Path(filename).stem}_eval_{backend}.png"
        annotated.save(annotated_path)

        case_results.append({
            "filename": filename,
            "image_size": [w, h],
            "inference_ms": elapsed_ms,
            "gt_count": len(gt_objects),
            "pred_count": len(preds),
            "matching": matching,
            "overlay_path": str(annotated_path),
        })

    eval_summary = {
        "backend": backend,
        "model_path": str(model_path),
        "checkpoint_sha256": ckpt_hash,
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "total_images": total_images,
        "total_gt_objects": total_gt_objects,
        "metrics": {
            "true_positives": total_tp,
            "false_negatives": total_fn,
            "false_positives": total_fp,
            "wrong_class": total_wrong_class,
            "box_dod_failures": box_failures,
            "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
            "recall": total_tp / total_gt_objects if total_gt_objects > 0 else 0.0,
        },
        "dod_compliance": {
            "image_coverage_100pct": (len(case_results) == total_images),
            "zero_fn": (total_fn == 0),
            "zero_wrong_class": (total_wrong_class == 0),
            "zero_fp": (total_fp == 0),
            "zero_box_failures": (box_failures == 0),
            "overall_dod_passed": (
                total_fn == 0 and total_wrong_class == 0 and total_fp == 0 and box_failures == 0
            ),
        },
        "cases": case_results,
    }

    out_file = output_dir / f"baseline_results_{backend}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"Evaluation complete for {backend}! Results saved to {out_file}")
    print(f"Summary: TP={total_tp}, FN={total_fn}, FP={total_fp}, WrongClass={total_wrong_class}, BoxFail={box_failures}")
    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ssdlite", "yolov8n"], default="ssdlite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="reports/qa_acceptance/manifest_hinh_anh_rac.json")
    parser.add_argument("--gt", type=str, default="reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
    parser.add_argument("--out-dir", type=str, default="reports/qa_acceptance")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_evaluation(
        backend=args.backend,
        model_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        gt_path=Path(args.gt),
        output_dir=Path(args.out_dir),
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
