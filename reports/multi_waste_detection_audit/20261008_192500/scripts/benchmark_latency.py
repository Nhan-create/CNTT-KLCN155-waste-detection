#!/usr/bin/env python3
# Mô tả: Đo độ trễ suy luận (latency) và mức tiêu thụ bộ nhớ của các mô hình phát hiện trong đợt kiểm toán 20261008_192500.
"""
scripts/benchmark_latency.py
----------------------------
Measures latency distribution (min, max, mean, std, p50, p90, p95, FPS),
raw timing samples, and measured memory consumption of SSDLite320, YOLOv8n,
and WBF Ensemble on REAL-WORLD images (not dummy grey images).

Adheres to PM Review F09:
- Uses real image input (rac-1562...313.webp)
- Logs all 50 individual raw timing samples
- Performs actual memory measurement (CUDA allocated/reserved or CPU RSS via psutil)
- Drops unverified VRAM 481 MB claim.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import psutil
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import hashlib
from src.detection.factory import create_detector

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def measure_memory(device: str) -> dict:
    mem_info = {}
    proc = psutil.Process()
    mem_info["cpu_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
    if device == "cuda" and torch.cuda.is_available():
        mem_info["cuda_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        mem_info["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        mem_info["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
    else:
        mem_info["cuda_available"] = False
    return mem_info

def benchmark_model(backend: str, model_path: Path, secondary_path: Path | None,
                    image_path: Path, device: str = "cpu",
                    num_warmup: int = 10, num_runs: int = 50) -> dict:
    print(f"\n--- Benchmarking {backend} on {device} ({num_runs} runs) ---")
    
    # Measure memory before loading
    mem_before = measure_memory(device)
    
    detector = create_detector(
        backend,
        model_path=model_path,
        secondary_model_path=secondary_path,
        device=device,
        confidence_threshold=0.20
    )
    
    mem_loaded = measure_memory(device)
    
    # Load real image
    real_img = Image.open(image_path).convert("RGB")
    width, height = real_img.size
    print(f"Loaded real image: {image_path.name} ({width}x{height})")
    
    # Warmup
    for _ in range(num_warmup):
        detector.detect_pil(real_img)
        
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # Reset CUDA peak memory stats if applicable
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    raw_samples_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        detector.detect_pil(real_img)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        raw_samples_ms.append(round((t1 - t0) * 1000.0, 3))
        
    samples_np = np.array(raw_samples_ms)
    mem_after = measure_memory(device)
    
    return {
        "backend": backend,
        "device": device,
        "torch_threads": torch.get_num_threads(),
        "input_image": {
            "file_name": image_path.name,
            "dimensions": f"{width}x{height}",
            "sha256": compute_sha256(image_path),
            "is_real_image": True
        },
        "runs_count": num_runs,
        "warmup_runs": num_warmup,
        "metrics_ms": {
            "mean": round(float(np.mean(samples_np)), 2),
            "std": round(float(np.std(samples_np)), 2),
            "min": round(float(np.min(samples_np)), 2),
            "max": round(float(np.max(samples_np)), 2),
            "p50_median": round(float(np.percentile(samples_np, 50)), 2),
            "p90": round(float(np.percentile(samples_np, 90)), 2),
            "p95": round(float(np.percentile(samples_np, 95)), 2),
            "fps": round(float(1000.0 / np.mean(samples_np)), 1)
        },
        "memory_measurement": {
            "measurement_type": "process_resident_set_size_snapshot",
            "cpu_rss_before_load_mb": mem_before["cpu_rss_mb"],
            "cpu_rss_loaded_mb": mem_loaded["cpu_rss_mb"],
            "cpu_rss_end_mb": mem_after["cpu_rss_mb"],
            "cuda_allocated_mb": mem_after.get("cuda_allocated_mb", None),
            "cuda_max_allocated_mb": mem_after.get("cuda_max_allocated_mb", None),
            "note": "cpu_rss_end_mb reflects overall process memory footprint snapshot at run completion, not an isolated model peak."
        },
        "raw_samples_ms": raw_samples_ms
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark detector latency on real images")
    parser.add_argument("--image-path", type=str,
                        default=str(WORKSPACE_ROOT / "reports/multi_waste_detection_audit/20261008_192500/images/case_001_raw.webp"))
    parser.add_argument("--output-json", type=str,
                        default=str(WORKSPACE_ROOT / "reports/multi_waste_detection_audit/20261008_192500/metrics/benchmark_latency.json"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_weights = WORKSPACE_ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"
    ssd_weights = WORKSPACE_ROOT / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"

    results = {}
    
    # 1. YOLOv8n
    results["yolov8n"] = benchmark_model(
        "yolov8n", yolo_weights, None, image_path, device=args.device, num_runs=args.runs
    )
    
    # 2. SSDLite320
    results["ssdlite"] = benchmark_model(
        "ssdlite", ssd_weights, None, image_path, device=args.device, num_runs=args.runs
    )
    
    # 3. WBF Ensemble (Primary=SSDLite, Secondary=YOLO)
    results["wbf"] = benchmark_model(
        "wbf", ssd_weights, yolo_weights, image_path, device=args.device, num_runs=args.runs
    )

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nAll benchmarks complete! Results written to: {output_path}")

if __name__ == "__main__":
    main()
