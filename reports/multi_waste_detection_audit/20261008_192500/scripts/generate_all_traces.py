#!/usr/bin/env python3
# Mô tả: Sinh toàn bộ nhật ký truy vết dự đoán (traces) và ảnh vẽ khung bao cho đợt kiểm toán đa đối tượng 20261008_192500.
"""
scripts/generate_all_traces.py
------------------------------
Generates end-to-end technical traces for CASE_001, CASE_002, CASE_003, CASE_004.
Strictly adheres to technical data logging without subjective validity assertions:
- Input image metadata & decode (EXIF, SHA-256, dimensions, color mode)
- Preprocessing steps and actual runtime transform attributes
- Real PyTorch forward pass tensor measurements (shape, dtype, device, min/max, norm, SHA-256)
- Detector candidate pools (conf=0.01) and target thresholds (0.20, 0.45)
- WBF input candidate clusters and fusion outputs
- Production UI / service state at both default (0.45) and low (0.20) operating thresholds
- Correct linking between threshold, detection count, and overlay visualization
- Strict preflight verification: raises FileNotFoundError and exits non-zero if any required image is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor

# Ensure repo root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection.factory import create_detector
from src.detection.fusion import fuse_detections
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite import SSDLiteWasteDetector, build_ssdlite
from src.detection.yolo import WasteDetector
import ultralytics

PALETTE = [
    "#E6194B", "#3CBA54", "#FFE119", "#4363D8", "#F58231",
    "#911EB4", "#42D4F4", "#F032E6", "#BFEF45", "#FABED4"
]

CASE_DEFINITIONS = [
    {
        "case_id": "CASE_001",
        "file_name": "rac-1562828175-1562828212-3852-1562828313.webp",
        "scenario": "Ảnh người dùng báo lỗi (Bãi chai nhựa/thủy tinh, nền nhà)",
        "ground_truth_note": "Nhiều chai nhựa/thủy tinh rải rác trên nền nhà; không có quần áo",
        "dest_raw": "case_001_raw.webp"
    },
    {
        "case_id": "CASE_002",
        "file_name": "rac-thai-huu-co-1.jpg",
        "scenario": "Rác hữu cơ thực tế (Vỏ hoa quả, rau củ sinh hoạt)",
        "ground_truth_note": "Rác hữu cơ / sinh học; không có quần áo hay pin",
        "dest_raw": "case_002_raw.jpg"
    },
    {
        "case_id": "CASE_003",
        "file_name": "rac-thai-quan-ao-4.jpg",
        "scenario": "Rác quần áo phế thải thực tế (Đống quần áo cũ ngoài trời)",
        "ground_truth_note": "Quần áo cũ phế thải thải bỏ; không có người mặc",
        "dest_raw": "case_003_raw.jpg"
    },
    {
        "case_id": "CASE_004",
        "file_name": "download (2).jpg",
        "scenario": "Túi rác ven đường (Rác thải túi nilon, phế thải sinh hoạt ven đường đất)",
        "ground_truth_note": "Túi rác nilon và phế thải sinh hoạt ven đường đất; rác nhựa/túi nilon",
        "dest_raw": "case_004_raw.jpg"
    }
]


def compute_sha256(path_or_bytes: Path | bytes) -> str:
    if isinstance(path_or_bytes, bytes):
        return hashlib.sha256(path_or_bytes).hexdigest()
    sha = hashlib.sha256()
    with open(path_or_bytes, "rb") as fp:
        while chunk := fp.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def render_overlay(image: Image.Image, detections, title: str) -> Image.Image:
    im = image.convert("RGB").copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in detections:
        color = PALETTE[d.class_index % len(PALETTE)]
        box = d.box
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=2)
        label_text = f"{d.class_id} {round(d.confidence * 100):.0f}%"
        text_bbox = draw.textbbox((box.x1, max(0, box.y1 - 14)), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box.x1, max(0, box.y1 - 14)), label_text, fill="white", font=font)

    draw.rectangle([0, 0, min(im.width, 420), 22], fill="black")
    draw.text((5, 4), title, fill="yellow", font=font)
    return im


def format_detections(dets):
    return [
        {
            "class_index": int(d.class_index),
            "class_name": str(d.class_id),
            "confidence": round(float(d.confidence), 4),
            "confidence_percent": f"{round(float(d.confidence) * 100):.0f}%",
            "box_xyxy": [round(d.box.x1, 1), round(d.box.y1, 1), round(d.box.x2, 1), round(d.box.y2, 1)],
            "box_area_px": round(float(d.box.area), 1)
        }
        for d in dets
    ]


def det_summary_str(dets):
    if not dets:
        return "0 box"
    parts = [f"{d.class_id}:{round(d.confidence * 100):.0f}%" for d in dets[:4]]
    suffix = "..." if len(dets) > 4 else ""
    return f"{len(dets)} box (" + ", ".join(parts) + suffix + ")"


def main():
    parser = argparse.ArgumentParser(description="Generate E2E audit traces with real PyTorch measurements.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Root path of the repository")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR.parent, help="Directory to save traces and images")
    parser.add_argument("--data-dir", type=Path, default=Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[4] / "data" / "holdout"))), help="Directory containing original images")
    parser.add_argument("--yolo-weights", type=Path, default=None, help="Path to YOLOv8n checkpoint")
    parser.add_argument("--ssdlite-weights", type=Path, default=None, help="Path to SSDLite checkpoint")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    images_dir = output_dir / "images"
    traces_dir = output_dir / "traces"
    images_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    yolo_weights = args.yolo_weights or (repo_root / "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt")
    ssdlite_weights = args.ssdlite_weights or (repo_root / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt")

    if not yolo_weights.is_file():
        raise FileNotFoundError(f"YOLO checkpoint missing: {yolo_weights}")
    if not ssdlite_weights.is_file():
        raise FileNotFoundError(f"SSDLite checkpoint missing: {ssdlite_weights}")

    yolo_sha = compute_sha256(yolo_weights)
    ssd_sha = compute_sha256(ssdlite_weights)

    # Preflight Check: verify all 4 case images exist before proceeding
    resolved_cases = []
    missing_cases = []
    for c in CASE_DEFINITIONS:
        # Check in data_dir, then in output_dir/images
        candidate_primary = data_dir / c["file_name"]
        candidate_fallback = images_dir / c["dest_raw"]
        if candidate_primary.is_file():
            resolved_cases.append((c, candidate_primary))
        elif candidate_fallback.is_file():
            resolved_cases.append((c, candidate_fallback))
        else:
            missing_cases.append(c["file_name"])

    if missing_cases:
        raise FileNotFoundError(
            f"Preflight check failed! Missing required case images: {missing_cases}. "
            f"Searched in {data_dir} and {images_dir}."
        )

    print(f"[PREFLIGHT OK] All {len(resolved_cases)} cases verified. Loading PyTorch detectors...")

    # Load high-level detectors
    yolo_001 = WasteDetector(yolo_weights, confidence_threshold=0.01)
    yolo_020 = WasteDetector(yolo_weights, confidence_threshold=0.20)
    yolo_045 = WasteDetector(yolo_weights, confidence_threshold=0.45)

    ssdlite_001 = SSDLiteWasteDetector(ssdlite_weights, confidence_threshold=0.01)
    ssdlite_020 = SSDLiteWasteDetector(ssdlite_weights, confidence_threshold=0.20)
    ssdlite_045 = SSDLiteWasteDetector(ssdlite_weights, confidence_threshold=0.45)

    # Load raw models for actual tensor forward measurement
    raw_yolo_model = ultralytics.YOLO(str(yolo_weights)).model.eval()
    raw_ssd_model = build_ssdlite(pretrained=False).eval()
    ssd_ckpt = torch.load(ssdlite_weights, map_location="cpu", weights_only=True)
    raw_ssd_model.load_state_dict(ssd_ckpt["model_state_dict"])

    cases_csv_rows = []

    for c_def, src_path in resolved_cases:
        c_id = c_def["case_id"]
        dest_raw_path = images_dir / c_def["dest_raw"]
        if src_path != dest_raw_path:
            shutil.copy2(src_path, dest_raw_path)

        file_bytes = dest_raw_path.read_bytes()
        img_sha256 = hashlib.sha256(file_bytes).hexdigest()

        with Image.open(dest_raw_path) as opened:
            img_format = opened.format or dest_raw_path.suffix.replace(".", "").upper()
            orig_w, orig_h = opened.size
            exif_info = getattr(opened, "_getexif", lambda: None)()
            orientation = exif_info.get(274, 1) if isinstance(exif_info, dict) else 1
            color_mode = opened.mode
            prepared = ImageOps.exif_transpose(opened).convert("RGB")

        # 1. Forward measurements through raw PyTorch models
        # 1A. YOLO raw forward measurement
        im0 = np.asarray(prepared)[:, :, ::-1]  # BGR
        from ultralytics.data.augment import LetterBox
        im_letterbox = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=im0)
        tensor_yolo = torch.from_numpy(im_letterbox).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            yolo_forward_out = raw_yolo_model(tensor_yolo)
            if isinstance(yolo_forward_out, (tuple, list)):
                raw_yolo_tensor = yolo_forward_out[0]
            else:
                raw_yolo_tensor = yolo_forward_out

        raw_yolo_np = raw_yolo_tensor.detach().cpu().numpy()
        yolo_tensor_stats = {
            "forward_shape": list(raw_yolo_tensor.shape),
            "dtype": str(raw_yolo_tensor.dtype),
            "device": str(raw_yolo_tensor.device),
            "min_val": round(float(raw_yolo_tensor.min().item()), 5),
            "max_val": round(float(raw_yolo_tensor.max().item()), 5),
            "l2_norm": round(float(torch.norm(raw_yolo_tensor).item()), 3),
            "tensor_bytes_sha256": hashlib.sha256(raw_yolo_np.tobytes()).hexdigest(),
            "channels_semantics": "4 bounding box parameters (cx, cy, w, h) + 10 class logits",
            "grid_anchors_count": int(raw_yolo_tensor.shape[-1])
        }

        # 1B. SSDLite raw forward measurement
        tensor_ssd = pil_to_tensor(prepared).float() / 255.0
        with torch.no_grad():
            t_imgs, _ = raw_ssd_model.transform([tensor_ssd])
            ssd_features = raw_ssd_model.backbone(t_imgs.tensors)
            ssd_head_outs = raw_ssd_model.head(list(ssd_features.values()))

        cls_logits = ssd_head_outs["cls_logits"]
        bbox_reg = ssd_head_outs["bbox_regression"]
        cls_logits_np = cls_logits.detach().cpu().numpy()
        bbox_reg_np = bbox_reg.detach().cpu().numpy()

        ssdlite_tensor_stats = {
            "transformed_shape": list(t_imgs.tensors.shape),
            "cls_logits_shape": list(cls_logits.shape),
            "bbox_regression_shape": list(bbox_reg.shape),
            "dtype": str(cls_logits.dtype),
            "device": str(cls_logits.device),
            "cls_logits_min": round(float(cls_logits.min().item()), 5),
            "cls_logits_max": round(float(cls_logits.max().item()), 5),
            "cls_logits_sha256": hashlib.sha256(cls_logits_np.tobytes()).hexdigest(),
            "bbox_regression_sha256": hashlib.sha256(bbox_reg_np.tobytes()).hexdigest(),
            "classes_total": int(cls_logits.shape[-1]),
            "anchors_total": int(cls_logits.shape[1]),
            "runtime_transform_attributes": {
                "image_mean": [float(v) for v in raw_ssd_model.transform.image_mean],
                "image_std": [float(v) for v in raw_ssd_model.transform.image_std],
                "min_size": list(raw_ssd_model.transform.min_size),
                "max_size": int(raw_ssd_model.transform.max_size)
            }
        }

        # 2. Run inference across thresholds
        res_yolo_001 = yolo_001.detect_pil(prepared)
        res_yolo_020 = yolo_020.detect_pil(prepared)
        res_yolo_045 = yolo_045.detect_pil(prepared)

        res_ssd_001 = ssdlite_001.detect_pil(prepared)
        res_ssd_020 = ssdlite_020.detect_pil(prepared)
        res_ssd_045 = ssdlite_045.detect_pil(prepared)

        # WBF at target thresholds
        # Note: UI passes outputs of sub-models at target threshold to fusion
        res_wbf_020 = fuse_detections([res_yolo_020, res_ssd_020], output_threshold=0.20)
        res_wbf_045 = fuse_detections([res_yolo_045, res_ssd_045], output_threshold=0.45)
        # Also compute WBF from low candidates (pipeline evaluation behavior)
        res_wbf_from_low_020 = fuse_detections([res_yolo_001, res_ssd_001], output_threshold=0.20)
        res_wbf_from_low_045 = fuse_detections([res_yolo_001, res_ssd_001], output_threshold=0.45)

        # 3. Render overlays
        overlay_yolo_020 = render_overlay(prepared, res_yolo_020.detections, f"{c_id} - YOLOv8n @ conf=0.20 ({len(res_yolo_020.detections)} det)")
        overlay_yolo_020.save(images_dir / f"{c_id}_yolo_conf020.jpg", quality=95)

        overlay_yolo_045 = render_overlay(prepared, res_yolo_045.detections, f"{c_id} - YOLOv8n @ conf=0.45 ({len(res_yolo_045.detections)} det)")
        overlay_yolo_045.save(images_dir / f"{c_id}_yolo_conf045.jpg", quality=95)

        overlay_ssd_020 = render_overlay(prepared, res_ssd_020.detections, f"{c_id} - SSDLite @ conf=0.20 ({len(res_ssd_020.detections)} det)")
        overlay_ssd_020.save(images_dir / f"{c_id}_ssdlite_conf020.jpg", quality=95)

        overlay_ssd_045 = render_overlay(prepared, res_ssd_045.detections, f"{c_id} - SSDLite @ conf=0.45 ({len(res_ssd_045.detections)} det)")
        overlay_ssd_045.save(images_dir / f"{c_id}_ssdlite_conf045.jpg", quality=95)

        overlay_wbf_020 = render_overlay(prepared, res_wbf_020.detections, f"{c_id} - WBF @ conf=0.20 ({len(res_wbf_020.detections)} det)")
        overlay_wbf_020.save(images_dir / f"{c_id}_wbf_conf020.jpg", quality=95)

        overlay_wbf_045 = render_overlay(prepared, res_wbf_045.detections, f"{c_id} - WBF @ conf=0.45 ({len(res_wbf_045.detections)} det)")
        overlay_wbf_045.save(images_dir / f"{c_id}_wbf_conf045.jpg", quality=95)

        # 4. Construct E2E Trace JSON
        execution_time_iso = datetime.now(timezone(timedelta(hours=7))).isoformat()
        trace_data = {
            "trace_version": "3.0.0",
            "case_id": c_id,
            "run_id": f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{c_id.lower()}",
            "timestamp": execution_time_iso,
            "models_provenance": {
                "yolov8n_checkpoint": str(yolo_weights),
                "yolov8n_sha256": yolo_sha,
                "ssdlite_checkpoint": str(ssdlite_weights),
                "ssdlite_sha256": ssd_sha
            },
            "input_image": {
                "file_name": c_def["file_name"],
                "raw_file_path": str(dest_raw_path),
                "sha256": img_sha256,
                "format": img_format,
                "dimensions": {"width": orig_w, "height": orig_h},
                "exif_orientation": orientation,
                "color_mode": color_mode
            },
            "preprocessing_measurements": {
                "yolo_pipeline": {
                    "conversion": "PIL.Image (RGB) -> np.asarray[:, :, ::-1] (BGR)",
                    "letterbox_shape": list(im_letterbox.shape),
                    "normalized_tensor_shape": list(tensor_yolo.shape),
                    "normalization_range": "[0.0, 1.0]"
                },
                "ssdlite_pipeline": {
                    "conversion": "pil_to_tensor(RGB) / 255.0 to torch.float32",
                    "internal_transform": ssdlite_tensor_stats["runtime_transform_attributes"]
                }
            },
            "measured_raw_model_tensors": {
                "yolov8n": yolo_tensor_stats,
                "ssdlite320": ssdlite_tensor_stats
            },
            "postprocessing_and_filtering": {
                "yolov8n": {
                    "candidate_pool_conf001_count": len(res_yolo_001.detections),
                    "detections_conf020": format_detections(res_yolo_020.detections),
                    "detections_conf045": format_detections(res_yolo_045.detections)
                },
                "ssdlite320": {
                    "candidate_pool_conf001_count": len(res_ssd_001.detections),
                    "detections_conf020": format_detections(res_ssd_020.detections),
                    "detections_conf045": format_detections(res_ssd_045.detections)
                }
            },
            "weighted_boxes_fusion": {
                "parameters": {
                    "weights": [1.0, 1.0],
                    "iou_thr": 0.55,
                    "skip_box_thr": 0.001,
                    "conf_type": "avg"
                },
                "fused_from_target_thresholds": {
                    "conf020": {
                        "inputs": {"yolo_count": len(res_yolo_020.detections), "ssdlite_count": len(res_ssd_020.detections)},
                        "fused_detections": format_detections(res_wbf_020.detections)
                    },
                    "conf045": {
                        "inputs": {"yolo_count": len(res_yolo_045.detections), "ssdlite_count": len(res_ssd_045.detections)},
                        "fused_detections": format_detections(res_wbf_045.detections)
                    }
                },
                "fused_from_low_candidate_pool": {
                    "conf020": format_detections(res_wbf_from_low_020.detections),
                    "conf045": format_detections(res_wbf_from_low_045.detections)
                }
            },
            "service_and_ui_view": {
                "conf045_operating_state": {
                    "threshold": 0.45,
                    "yolo_count": len(res_yolo_045.detections),
                    "ssdlite_count": len(res_ssd_045.detections),
                    "wbf_count": len(res_wbf_045.detections),
                    "yolo_overlay_file": f"images/{c_id}_yolo_conf045.jpg",
                    "ssdlite_overlay_file": f"images/{c_id}_ssdlite_conf045.jpg",
                    "wbf_overlay_file": f"images/{c_id}_wbf_conf045.jpg"
                },
                "conf020_operating_state": {
                    "threshold": 0.20,
                    "yolo_count": len(res_yolo_020.detections),
                    "ssdlite_count": len(res_ssd_020.detections),
                    "wbf_count": len(res_wbf_020.detections),
                    "yolo_overlay_file": f"images/{c_id}_yolo_conf020.jpg",
                    "ssdlite_overlay_file": f"images/{c_id}_ssdlite_conf020.jpg",
                    "wbf_overlay_file": f"images/{c_id}_wbf_conf020.jpg"
                }
            }
        }

        trace_path = traces_dir / f"{c_id.lower()}_trace.json"
        with open(trace_path, "w", encoding="utf-8") as fp:
            json.dump(trace_data, fp, indent=2, ensure_ascii=False)
        print(f"Saved verified trace: {trace_path}")

        cases_csv_rows.append({
            "case_id": c_id,
            "file_name": c_def["file_name"],
            "sha256": img_sha256,
            "scenario": c_def["scenario"],
            "ground_truth_note": c_def["ground_truth_note"],
            "yolo_conf020": det_summary_str(res_yolo_020.detections),
            "yolo_conf045": det_summary_str(res_yolo_045.detections),
            "ssdlite_conf020": det_summary_str(res_ssd_020.detections),
            "ssdlite_conf045": det_summary_str(res_ssd_045.detections),
            "wbf_conf020": det_summary_str(res_wbf_020.detections),
            "wbf_conf045": det_summary_str(res_wbf_045.detections),
            "overlay_conf020": f"images/{c_id}_yolo_conf020.jpg",
            "overlay_conf045": f"images/{c_id}_yolo_conf045.jpg"
        })

    import pandas as pd
    df_cases = pd.DataFrame(cases_csv_rows)
    csv_path = output_dir / "cases.csv"
    df_cases.to_csv(csv_path, index=False)
    print(f"Saved verified cases.csv: {csv_path}")


if __name__ == "__main__":
    main()
