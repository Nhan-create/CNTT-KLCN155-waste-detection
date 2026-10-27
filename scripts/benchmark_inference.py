# Mô tả: Đo thời gian suy luận trung bình (ms/ảnh) và FPS của SSDLite320 và YOLOv8n.
"""Inference benchmark and testing script for SSDLite and YOLOv8n models.

Supports CPU and CUDA execution, confidence threshold tuning, and batch/folder evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.factory import create_detector
from src.detection.schema import DETECTION_CLASS_NAMES


def benchmark_model(
    backend: str,
    model_path: str | Path,
    image_paths: list[Path],
    device: str = "cpu",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    image_size: int | None = None,
    warmup_runs: int = 2,
) -> dict[str, Any]:
    """Run inference benchmark on a list of images using specified detector and device."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    # Set default image size if not provided
    if image_size is None:
        image_size = 320 if backend == "ssdlite" else 640

    detector = create_detector(
        backend=backend,
        model_path=model_path,
        device=device,
        confidence_threshold=conf_thresh,
        iou_threshold=iou_thresh,
        image_size=image_size,
    )

    if not image_paths:
        return {"error": "No images provided"}

    # Warmup
    warmup_img = Image.open(image_paths[0])
    for _ in range(warmup_runs):
        _ = detector.detect_pil(warmup_img)

    results = []
    latencies = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as pil_img:
                t0 = time.perf_counter()
                det_res = detector.detect_pil(pil_img)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

            latencies.append(elapsed_ms)
            detections_list = [
                {
                    "class_index": d.class_index,
                    "class_name": d.class_id,
                    "confidence": round(float(d.confidence), 4),
                    "box_xyxy": [
                        round(float(d.box.x1), 1),
                        round(float(d.box.y1), 1),
                        round(float(d.box.x2), 1),
                        round(float(d.box.y2), 1),
                    ],
                }
                for d in det_res.detections
            ]
            results.append(
                {
                    "image": img_path.name,
                    "image_size": [det_res.width, det_res.height],
                    "latency_ms": round(elapsed_ms, 2),
                    "detections_count": len(detections_list),
                    "detections": detections_list,
                }
            )
        except Exception as e:
            results.append({"image": img_path.name, "error": str(e)})

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    return {
        "backend": backend,
        "model_path": str(model_path),
        "device": device,
        "image_size": image_size,
        "confidence_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "total_images": len(image_paths),
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test and benchmark waste detectors")
    parser.add_argument(
        "--backend",
        choices=["ssdlite", "yolov8n", "both"],
        default="both",
        help="Backend detector to test",
    )
    parser.add_argument(
        "--ssdlite-path",
        default="artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt",
        help="Path to SSDLite checkpoint",
    )
    parser.add_argument(
        "--yolo-path",
        default="artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt",
        help="Path to YOLOv8n checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Hardware execution device",
    )
    parser.add_argument(
        "--images-dir",
        default="data/holdout",
        help="Path to image directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Limit number of images to test (default: 5 for quick check)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--output-json",
        default="reports/inference_benchmark_sample.json",
        help="Output file for benchmark results",
    )
    args = parser.parse_args()

    # Find images
    img_dir = Path(args.images_dir)
    if not img_dir.is_dir():
        print(f"Directory {img_dir} does not exist!")
        sys.exit(1)

    all_images = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
    )
    if not all_images:
        print(f"No valid images found in {img_dir}!")
        sys.exit(1)

    target_images = all_images[: args.limit] if args.limit > 0 else all_images
    print(f"Found {len(all_images)} images. Running evaluation on {len(target_images)} images.")

    devices = ["cpu"]
    import torch
    if torch.cuda.is_available() and args.device in ["cuda", "both"]:
        devices.append("cuda:0")
    if args.device == "cuda" and "cuda:0" in devices:
        devices = ["cuda:0"]
    elif args.device == "cpu":
        devices = ["cpu"]

    backends = []
    if args.backend in ["ssdlite", "both"]:
        backends.append(("ssdlite", args.ssdlite_path))
    if args.backend in ["yolov8n", "both"]:
        backends.append(("yolov8n", args.yolo_path))

    summary_report = []

    for backend_name, model_file in backends:
        for dev in devices:
            print(f"\n---> Benchmarking {backend_name.upper()} on {dev} (ckpt: {model_file})...")
            try:
                rep = benchmark_model(
                    backend=backend_name,
                    model_path=model_file,
                    image_paths=target_images,
                    device=dev,
                    conf_thresh=args.conf,
                )
                print(
                    f"     Done! Avg Latency: {rep['avg_latency_ms']} ms | FPS: {rep['fps']} | Total Detections: {sum(r.get('detections_count', 0) for r in rep['results'])}"
                )
                summary_report.append(rep)
            except Exception as e:
                print(f"     FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Save output
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {out_p}")


if __name__ == "__main__":
    main()
