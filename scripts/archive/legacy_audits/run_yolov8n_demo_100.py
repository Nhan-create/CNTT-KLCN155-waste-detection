# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Chạy thử nghiệm, tái lập đánh giá hoặc so sánh mô hình cho `run_yolov8n_demo_100`.
"""Run YOLOv8n demo inference on 100 candidate images.

Checkpoint: artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt
Settings:
  confidence = 0.25
  tiling = OFF
  two-stage = OFF
  per-class thresholds = OFF
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.detection.types import DetectionResult
from src.detection.visualization import draw_detections
from src.detection.yolo import WasteDetector

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout"))) / "demo_multiobject_100_20261012")
ORIGINALS_DIR = BASE_DIR / "originals"
OVERLAYS_DIR = BASE_DIR / "overlays"
CONTACT_SHEETS_DIR = BASE_DIR / "contact_sheets"
COMPARISONS_DIR = CONTACT_SHEETS_DIR / "single_comparisons"
MANIFESTS_DIR = BASE_DIR / "manifests"

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CKPT = "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt"
CANDIDATE_AUDIT_JSON = MANIFESTS_DIR / "candidate_100_download_audit.json"


def create_side_by_side(orig: Image.Image, overlay: Image.Image, candidate_id: str, class_name: str, num_dets: int) -> Image.Image:
    """Create a side-by-side comparison with a top title banner."""
    # Resize both to a common display height, e.g., 600px
    target_h = 600
    def resize_h(im: Image.Image) -> Image.Image:
        w, h = im.size
        new_w = max(1, int(w * (target_h / h)))
        return im.resize((new_w, target_h), Image.Resampling.LANCZOS)

    im1 = resize_h(orig)
    im2 = resize_h(overlay)

    banner_h = 40
    total_w = im1.width + im2.width + 10
    total_h = target_h + banner_h

    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 27))
    draw = ImageDraw.Draw(canvas)

    # Banner title
    title = f"{candidate_id} | Class mục tiêu: {class_name} | YOLOv8n phát hiện: {num_dets} boxes"
    draw.text((15, 10), title, fill=(255, 255, 255))
    draw.text((15, banner_h - 14), "ẢNH GỐC (Original)", fill=(180, 180, 180))
    draw.text((im1.width + 25, banner_h - 14), "DỰ ĐOÁN YOLOV8N (Overlay - conf >= 0.25)", fill=(56, 189, 248))

    canvas.paste(im1, (0, banner_h))
    canvas.paste(im2, (im1.width + 10, banner_h))

    return canvas


def main() -> None:
    print(f"Loading candidate audit from: {CANDIDATE_AUDIT_JSON}")
    with open(CANDIDATE_AUDIT_JSON, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    items = audit_data["items"]
    print(f"Total candidate images to infer: {len(items)}")

    print(f"Loading YOLOv8n model from: {MODEL_CKPT}")
    detector = WasteDetector(
        model_path=MODEL_CKPT,
        confidence_threshold=0.25,
        iou_threshold=0.55,
    )

    # Warmup
    print("Warming up model...")
    warmup_img = Image.new("RGB", (640, 640), (128, 128, 128))
    detector.detect_pil(warmup_img)
    print("Model warmup complete.")

    predictions_records = []
    class_detect_counts: dict[str, int] = {}
    latencies: list[float] = []

    print("\nRunning inference on 100 images...")
    for idx, item in enumerate(items, 1):
        cid = item["candidate_id"]
        class_id = item["target_class_id"]
        class_name = item["target_class_name"]
        local_filename = item["local_filename"]
        img_path = ORIGINALS_DIR / local_filename

        if not img_path.exists():
            print(f"[{idx}/100] ERROR: Image not found: {img_path}")
            continue

        try:
            with Image.open(img_path) as im:
                orig_img = im.convert("RGB")
        except Exception as e:
            print(f"[{idx}/100] ERROR reading {img_path}: {e}")
            continue

        res: DetectionResult = detector.detect_pil(orig_img)
        latencies.append(res.inference_ms)

        # Draw overlay
        overlay = draw_detections(orig_img, res)
        overlay_path = OVERLAYS_DIR / f"{cid}_overlay.jpg"
        overlay.save(overlay_path, "JPEG", quality=95)

        # Side by side comparison
        sbs = create_side_by_side(orig_img, overlay, cid, class_name, len(res.detections))
        sbs_path = COMPARISONS_DIR / f"{cid}_side_by_side.jpg"
        sbs.save(sbs_path, "JPEG", quality=90)

        boxes_record = []
        for d in res.detections:
            c_name = d.class_id
            class_detect_counts[c_name] = class_detect_counts.get(c_name, 0) + 1
            boxes_record.append({
                "class_index": d.class_index,
                "class_name": d.class_id,
                "confidence": round(float(d.confidence), 4),
                "box_xyxy": [round(d.box.x1, 1), round(d.box.y1, 1), round(d.box.x2, 1), round(d.box.y2, 1)],
            })

        pred_entry = {
            "candidate_id": cid,
            "target_class_id": class_id,
            "target_class_name": class_name,
            "image_filename": local_filename,
            "image_width": res.image_width,
            "image_height": res.image_height,
            "inference_ms": round(res.inference_ms, 2),
            "num_detections": len(res.detections),
            "detections": boxes_record,
            "overlay_path": str(overlay_path),
            "side_by_side_path": str(sbs_path),
        }
        predictions_records.append(pred_entry)

        print(
            f"[{idx}/100] {cid} ({class_name}): {len(res.detections)} dets, "
            f"{res.inference_ms:.1f}ms -> saved overlay & side-by-side"
        )

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print(f"\nInference completed: {len(predictions_records)} images.")
    print(f"Average latency: {avg_lat:.2f} ms")
    print(f"Detections by class: {class_detect_counts}")

    out_json = MANIFESTS_DIR / "predictions_yolov8n_raw.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "YOLOv8n-10cls",
                "checkpoint": MODEL_CKPT,
                "confidence_threshold": 0.25,
                "iou_threshold": 0.55,
                "average_inference_ms": round(avg_lat, 2),
                "total_detections": sum(class_detect_counts.values()),
                "detections_by_class": class_detect_counts,
                "predictions": predictions_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved raw predictions to: {out_json}")


if __name__ == "__main__":
    main()
