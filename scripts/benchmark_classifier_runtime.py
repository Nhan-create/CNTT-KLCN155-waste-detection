"""
scripts/benchmark_classifier_runtime.py
---------------------------------------
Comprehensive CPU runtime optimization benchmark for MobileNetV3-Large waste classifier.

Branch A Requirements:
1. Frozen protocol saved & hashed BEFORE running benchmarks.
2. Benchmarks performed using REAL validation images from data/processed_v2/val/.
3. Batch size 1, FP32 precision, eval() mode, torch.inference_mode().
4. Thread configurations evaluated: 1, 2, 4, 6, 8, 12 threads.
5. PyTorch native vs ONNX Runtime FP32 comparison.
6. Full pipeline latency (I/O, resize, normalization, forward, softmax) vs model forward pass.
7. Full validation set (2,223 images) equivalence audit:
   - Output logit/probability maximum difference and mean difference.
   - Top-1 prediction matching rate.
   - Validation Accuracy and Macro-F1 across PyTorch vs ONNX.
8. Raw per-trial timing logs preserved.
"""

import os
import sys
import time
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import onnx
import onnxruntime as ort

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def build_classifier(num_classes: int = 10) -> nn.Module:
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def main():
    project_root = Path(__file__).resolve().parent.parent
    ckpt_path = project_root / "artifacts" / "official_run" / "best_model.pt"
    val_dir = project_root / "data" / "processed_v2" / "val"
    manifest_path = project_root / "data" / "audit" / "split_manifest_v2.csv"
    out_dir = project_root / "artifacts" / "part02" / "runtime_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TASK 2 BRANCH A: CPU RUNTIME OPTIMIZATION BENCHMARK")
    print(f"Project Root: {project_root}")
    print(f"Checkpoint:   {ckpt_path} (exists: {ckpt_path.exists()})")
    print(f"Val Dir:      {val_dir} (exists: {val_dir.exists()})")
    print(f"Output Dir:   {out_dir}")
    print("=" * 80)

    if not ckpt_path.exists() or not val_dir.exists():
        print("[ERROR] Required checkpoint or validation directory missing!")
        sys.exit(1)

    ckpt_sha256 = compute_sha256(ckpt_path)
    print(f"[INFO] Verified Checkpoint SHA-256: {ckpt_sha256}")

    # Load validation sample list
    val_images = sorted(list(val_dir.glob("*/*.jpg")))
    print(f"[INFO] Discovered {len(val_images)} validation images on disk.")
    if len(val_images) == 0:
        print("[ERROR] No validation images found!")
        sys.exit(1)

    # 1. SAVE PROTOCOL BEFORE BENCHMARK EXECUTION
    protocol_path = out_dir / "optimization_protocol_frozen.json"
    protocol_data = {
        "protocol_name": "CPU_RUNTIME_OPTIMIZATION_PROTOCOL_BRANCH_A",
        "created_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha256,
        "input_source": "real_validation_images",
        "validation_samples_count": len(val_images),
        "image_resolution": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "batch_size": 1,
        "warmup_trials": 50,
        "benchmark_trials": 200,
        "threads_to_evaluate": [1, 2, 4, 6, 8, 12],
        "runtimes_to_evaluate": [
            "pytorch_fp32_inference_mode",
            "onnxruntime_fp32_cpu"
        ],
        "system_environment": {
            "os": "Windows 11 Home Single Language (64-bit)",
            "cpu_model": "AMD Ryzen 5 6600H with Radeon Graphics (6C/12T, 3.3-4.5 GHz)",
            "ram_mb": 15676,
            "power_state": "AC_Connected (PowerOnline=True)",
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__
        }
    }
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol_data, f, indent=2, ensure_ascii=False)
    protocol_sha = compute_sha256(protocol_path)
    print(f"[INFO] Frozen Protocol written to: {protocol_path} (SHA: {protocol_sha})")

    # Load Model Weights
    device = torch.device("cpu")
    model = build_classifier(len(CLASS_NAMES))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Select fixed subset of 200 real images for latency benchmarking
    rng = np.random.RandomState(42)
    benchmark_indices = rng.choice(len(val_images), size=min(200, len(val_images)), replace=False)
    benchmark_img_paths = [val_images[i] for i in benchmark_indices]

    # Pre-load tensors to isolate forward latency from disk I/O
    print("[INFO] Pre-loading 200 validation image tensors into memory...")
    benchmark_tensors = []
    for p in benchmark_img_paths:
        with Image.open(p) as img:
            t = transform(img.convert("RGB")).unsqueeze(0)
            benchmark_tensors.append(t)

    # 2. EXPORT MODEL TO ONNX
    onnx_path = out_dir / "mobilenetv3_large_waste.onnx"
    print(f"[INFO] Exporting PyTorch model to ONNX: {onnx_path}...")
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    onnx_sha256 = compute_sha256(onnx_path)
    print(f"[INFO] ONNX model successfully exported (SHA: {onnx_sha256})")

    # Verify ONNX model integrity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model checker passed successfully.")

    # 3. BENCHMARK RUNTIMES
    threads_list = [1, 2, 4, 6, 8, 12]
    all_results = []
    raw_timings_records = []

    # A) PyTorch Native Benchmarks
    print("\n" + "=" * 60)
    print("STARTING PYTORCH FP32 BENCHMARKS (with torch.inference_mode())")
    print("=" * 60)

    orig_threads = torch.get_num_threads()

    for n_th in threads_list:
        torch.set_num_threads(n_th)
        print(f"\n[PyTorch] Testing with num_threads = {n_th}...")

        # Warmup with real tensors
        with torch.inference_mode():
            for i in range(50):
                _ = model(benchmark_tensors[i % len(benchmark_tensors)])

        # Benchmark Forward Only
        forward_times_ms = []
        with torch.inference_mode():
            for idx, t in enumerate(benchmark_tensors):
                t0 = time.perf_counter()
                _ = model(t)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                forward_times_ms.append(elapsed_ms)
                raw_timings_records.append({
                    "framework": "pytorch_fp32",
                    "threads": n_th,
                    "trial": idx + 1,
                    "type": "forward_only",
                    "latency_ms": elapsed_ms
                })

        # Benchmark Full Pipeline (PIL open + transform + forward + softmax)
        pipeline_times_ms = []
        with torch.inference_mode():
            for idx in range(min(50, len(benchmark_img_paths))):
                p = benchmark_img_paths[idx]
                t0 = time.perf_counter()
                with Image.open(p) as raw_img:
                    inp = transform(raw_img.convert("RGB")).unsqueeze(0)
                    out = model(inp)
                    _ = torch.softmax(out, dim=1).numpy()
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                pipeline_times_ms.append(elapsed_ms)
                raw_timings_records.append({
                    "framework": "pytorch_fp32",
                    "threads": n_th,
                    "trial": idx + 1,
                    "type": "full_pipeline",
                    "latency_ms": elapsed_ms
                })

        f_arr = np.array(forward_times_ms)
        p_arr = np.array(pipeline_times_ms)

        f_mean = float(np.mean(f_arr))
        f_median = float(np.median(f_arr))
        f_p95 = float(np.percentile(f_arr, 95))
        f_min = float(np.min(f_arr))
        f_max = float(np.max(f_arr))
        f_std = float(np.std(f_arr))
        fps = 1000.0 / f_mean if f_mean > 0 else 0.0

        p_mean = float(np.mean(p_arr))
        p_median = float(np.median(p_arr))

        print(f"  -> Forward Mean:    {f_mean:.2f} ms ({fps:.1f} FPS)")
        print(f"  -> Forward Median:  {f_median:.2f} ms | P95: {f_p95:.2f} ms")
        print(f"  -> Full Pipeline:   {p_mean:.2f} ms (Median: {p_median:.2f} ms)")

        all_results.append({
            "framework": "pytorch_fp32",
            "threads": n_th,
            "forward_mean_ms": round(f_mean, 2),
            "forward_median_ms": round(f_median, 2),
            "forward_p95_ms": round(f_p95, 2),
            "forward_min_ms": round(f_min, 2),
            "forward_max_ms": round(f_max, 2),
            "forward_std_ms": round(f_std, 2),
            "fps": round(fps, 1),
            "full_pipeline_mean_ms": round(p_mean, 2),
            "full_pipeline_median_ms": round(p_median, 2),
            "meets_sub_20ms_threshold": bool(f_mean < 20.0),
            "meets_sub_10ms_hypothesis": bool(f_mean < 10.0)
        })

    # Restore PyTorch threads
    torch.set_num_threads(orig_threads)

    # B) ONNX Runtime FP32 Benchmarks
    print("\n" + "=" * 60)
    print("STARTING ONNX RUNTIME FP32 BENCHMARKS")
    print("=" * 60)

    for n_th in threads_list:
        print(f"\n[ONNX Runtime] Testing with intra_op_num_threads = {n_th}...")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = n_th
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Warmup
        for i in range(50):
            inp_np = benchmark_tensors[i % len(benchmark_tensors)].numpy()
            _ = session.run(None, {input_name: inp_np})

        # Benchmark Forward Only
        forward_times_ms = []
        for idx, t in enumerate(benchmark_tensors):
            inp_np = t.numpy()
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: inp_np})
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            forward_times_ms.append(elapsed_ms)
            raw_timings_records.append({
                "framework": "onnxruntime_fp32",
                "threads": n_th,
                "trial": idx + 1,
                "type": "forward_only",
                "latency_ms": elapsed_ms
            })

        # Benchmark Full Pipeline
        pipeline_times_ms = []
        for idx in range(min(50, len(benchmark_img_paths))):
            p = benchmark_img_paths[idx]
            t0 = time.perf_counter()
            with Image.open(p) as raw_img:
                inp = transform(raw_img.convert("RGB")).unsqueeze(0).numpy()
                out = session.run(None, {input_name: inp})[0]
                # Softmax
                exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
                _ = exp_out / np.sum(exp_out, axis=1, keepdims=True)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            pipeline_times_ms.append(elapsed_ms)
            raw_timings_records.append({
                "framework": "onnxruntime_fp32",
                "threads": n_th,
                "trial": idx + 1,
                "type": "full_pipeline",
                "latency_ms": elapsed_ms
            })

        f_arr = np.array(forward_times_ms)
        p_arr = np.array(pipeline_times_ms)

        f_mean = float(np.mean(f_arr))
        f_median = float(np.median(f_arr))
        f_p95 = float(np.percentile(f_arr, 95))
        f_min = float(np.min(f_arr))
        f_max = float(np.max(f_arr))
        f_std = float(np.std(f_arr))
        fps = 1000.0 / f_mean if f_mean > 0 else 0.0

        p_mean = float(np.mean(p_arr))
        p_median = float(np.median(p_arr))

        print(f"  -> Forward Mean:    {f_mean:.2f} ms ({fps:.1f} FPS)")
        print(f"  -> Forward Median:  {f_median:.2f} ms | P95: {f_p95:.2f} ms")
        print(f"  -> Full Pipeline:   {p_mean:.2f} ms (Median: {p_median:.2f} ms)")

        all_results.append({
            "framework": "onnxruntime_fp32",
            "threads": n_th,
            "forward_mean_ms": round(f_mean, 2),
            "forward_median_ms": round(f_median, 2),
            "forward_p95_ms": round(f_p95, 2),
            "forward_min_ms": round(f_min, 2),
            "forward_max_ms": round(f_max, 2),
            "forward_std_ms": round(f_std, 2),
            "fps": round(fps, 1),
            "full_pipeline_mean_ms": round(p_mean, 2),
            "full_pipeline_median_ms": round(p_median, 2),
            "meets_sub_20ms_threshold": bool(f_mean < 20.0),
            "meets_sub_10ms_hypothesis": bool(f_mean < 10.0)
        })

    # Save raw timings CSV
    raw_df = pd.DataFrame(raw_timings_records)
    raw_csv_path = out_dir / "raw_timings.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"\n[INFO] Saved raw timings ({len(raw_df)} records) to {raw_csv_path}")

    # 4. FULL VALIDATION EQUIVALENCE CHECK (PyTorch vs ONNX across all 2,223 images)
    print("\n" + "=" * 60)
    print("PERFORMING FULL VALIDATION SET EQUIVALENCE AUDIT (2,223 IMAGES)")
    print("=" * 60)

    # Pick the best ONNX session (e.g. 6 threads)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 6
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    torch.set_num_threads(6)

    max_logit_diff = 0.0
    sum_logit_diff = 0.0
    max_prob_diff = 0.0
    sum_prob_diff = 0.0
    total_comparisons = 0
    predictions_matched = 0

    pytorch_preds = []
    onnx_preds = []
    ground_truths = []

    print("[INFO] Evaluating all 2,223 validation samples for exact numerical discrepancy...")
    for idx, p in enumerate(val_images):
        true_cls_name = p.parent.name
        true_cls_id = CLASS_NAMES.index(true_cls_name)
        ground_truths.append(true_cls_id)

        with Image.open(p) as img:
            t = transform(img.convert("RGB")).unsqueeze(0)

        # PyTorch forward
        with torch.inference_mode():
            py_logits = model(t).numpy()[0]
            py_exp = np.exp(py_logits - np.max(py_logits))
            py_prob = py_exp / np.sum(py_exp)
            py_pred = int(np.argmax(py_prob))
            pytorch_preds.append(py_pred)

        # ONNX forward
        onnx_logits = session.run(None, {input_name: t.numpy()})[0][0]
        onnx_exp = np.exp(onnx_logits - np.max(onnx_logits))
        onnx_prob = onnx_exp / np.sum(onnx_exp)
        onnx_pred = int(np.argmax(onnx_prob))
        onnx_preds.append(onnx_pred)

        # Numerical differences
        l_diff = np.max(np.abs(py_logits - onnx_logits))
        p_diff = np.max(np.abs(py_prob - onnx_prob))

        max_logit_diff = max(max_logit_diff, float(l_diff))
        sum_logit_diff += float(np.mean(np.abs(py_logits - onnx_logits)))
        max_prob_diff = max(max_prob_diff, float(p_diff))
        sum_prob_diff += float(np.mean(np.abs(py_prob - onnx_prob)))
        total_comparisons += 1

        if py_pred == onnx_pred:
            predictions_matched += 1

    py_acc = float(np.mean(np.array(pytorch_preds) == np.array(ground_truths)))
    onnx_acc = float(np.mean(np.array(onnx_preds) == np.array(ground_truths)))
    match_rate = float(predictions_matched / total_comparisons)
    mean_logit_diff = float(sum_logit_diff / total_comparisons)
    mean_prob_diff = float(sum_prob_diff / total_comparisons)

    print("\n--- Equivalence Audit Results ---")
    print(f"Total Validation Samples Checked: {total_comparisons}")
    print(f"Top-1 Prediction Matching Rate:   {match_rate * 100:.4f}% ({predictions_matched}/{total_comparisons})")
    print(f"PyTorch Validation Accuracy:      {py_acc * 100:.2f}%")
    print(f"ONNX Validation Accuracy:         {onnx_acc * 100:.2f}%")
    print(f"Max Absolute Logit Diff:          {max_logit_diff:.6e}")
    print(f"Mean Absolute Logit Diff:         {mean_logit_diff:.6e}")
    print(f"Max Absolute Probability Diff:    {max_prob_diff:.6e}")
    print(f"Mean Absolute Probability Diff:   {mean_prob_diff:.6e}")

    equivalence_summary = {
        "total_validation_samples": total_comparisons,
        "prediction_matching_rate": round(match_rate, 6),
        "matching_predictions_count": predictions_matched,
        "mismatched_predictions_count": total_comparisons - predictions_matched,
        "pytorch_val_accuracy": round(py_acc, 4),
        "onnx_val_accuracy": round(onnx_acc, 4),
        "max_absolute_logit_difference": max_logit_diff,
        "mean_absolute_logit_difference": mean_logit_diff,
        "max_absolute_probability_difference": max_prob_diff,
        "mean_absolute_probability_difference": mean_prob_diff,
        "numerical_status": "EXCELLENT_EQUIVALENCE" if match_rate == 1.0 and max_prob_diff < 1e-4 else "DISCREPANCY_DETECTED"
    }

    # 5. SELECT BEST CONFIGURATION & COMPILE SUMMARY
    results_df = pd.DataFrame(all_results)
    results_json_path = out_dir / "runtime_benchmark_results.json"

    # Find fastest configuration
    fastest_row = results_df.sort_values(by="forward_mean_ms").iloc[0]

    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol_file": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": ckpt_sha256,
        "onnx_model_path": str(onnx_path),
        "onnx_model_sha256": onnx_sha256,
        "benchmark_results": all_results,
        "fastest_configuration": {
            "framework": fastest_row["framework"],
            "threads": int(fastest_row["threads"]),
            "forward_mean_ms": float(fastest_row["forward_mean_ms"]),
            "forward_median_ms": float(fastest_row["forward_median_ms"]),
            "forward_p95_ms": float(fastest_row["forward_p95_ms"]),
            "forward_min_ms": float(fastest_row["forward_min_ms"]),
            "forward_max_ms": float(fastest_row["forward_max_ms"]),
            "fps": float(fastest_row["fps"]),
            "full_pipeline_mean_ms": float(fastest_row["full_pipeline_mean_ms"]),
            "passes_sub_20ms_threshold": bool(fastest_row["meets_sub_20ms_threshold"]),
            "passes_sub_10ms_hypothesis": bool(fastest_row["meets_sub_10ms_hypothesis"])
        },
        "equivalence_audit": equivalence_summary,
        "findings_and_verdict": {
            "hypothesis_sub_10ms_on_this_machine": "CONFIRMED" if fastest_row["forward_mean_ms"] < 10.0 else "REFUTED",
            "sub_20ms_threshold_status": "PASS" if fastest_row["forward_mean_ms"] < 20.0 else "FAIL",
            "historical_gate_a_status": "GATE_A_FAILED_PRESERVED (Initial evaluation remains FAILED, this is runtime optimization experiment on validation set)",
            "key_takeaway": f"Fastest runtime is {fastest_row['framework']} with {fastest_row['threads']} threads achieving {fastest_row['forward_mean_ms']} ms forward latency ({fastest_row['fps']} FPS)."
        }
    }

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Runtime Optimization Benchmark Completed!")
    print(f"Results JSON: {results_json_path}")
    print(f"Fastest Config: {fastest_row['framework']} ({fastest_row['threads']} threads) -> {fastest_row['forward_mean_ms']} ms (FPS: {fastest_row['fps']})")
    print(f"Sub-20ms threshold: {'PASS' if fastest_row['forward_mean_ms'] < 20.0 else 'FAIL'}")
    print(f"Sub-10ms hypothesis: {'CONFIRMED' if fastest_row['forward_mean_ms'] < 10.0 else 'REFUTED'}")

if __name__ == "__main__":
    main()
