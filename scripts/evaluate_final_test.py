r"""Official Gate A Independent Final Test Evaluation Script for MobileNetV3.

Protocol & Verification:
1. Dynamically calculates checkpoint and split manifest SHA-256.
2. Measures CPU forward latency & full pipeline latency (batch size 1, 4 threads, 50 warmup, 200 trials).
3. Reads 100% of physical test images from disk and hashes every file byte stream directly.
4. Evaluates MobileNetV3 checkpoint on the strictly isolated Final Test Set (2,223 images).
5. Computes all Gate A metrics: Top-1 Accuracy, Macro-F1, Battery Recall, Min Class Recall.
6. Exports test predictions table, confusion matrix, error analysis, and structured JSON metrics report.
7. Evaluates Gate A pass/fail criteria strictly against predefined frozen thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "official_run" / "best_model.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "audit" / "split_manifest_v2.csv"
if (PROJECT_ROOT / "data" / "processed_v2" / "test").exists():
    DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "processed_v2" / "test"
elif Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2\test").exists():
    DEFAULT_TEST_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2\test")
else:
    DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "processed_v2" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "part02" / "gate_a"

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]

# Gate A Acceptance Criteria Thresholds
GATE_A_THRESHOLDS = {
    "min_accuracy": 0.880,
    "min_macro_f1": 0.850,
    "min_battery_recall": 0.880,
    "min_class_recall": 0.800,
    "max_cpu_latency_ms": 20.0,
    "expected_test_samples": 2223,
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_classifier(num_classes: int = 10) -> nn.Module:
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def benchmark_cpu_latency(model: nn.Module, sample_img_path: Path, num_threads: int = 4, warmup: int = 50, trials: int = 200) -> dict:
    print(f"\n[CPU Latency Benchmark Protocol]:")
    print(f"  - Threads: {num_threads} | Batch Size: 1 | Warmup: {warmup} | Trials: {trials}")

    orig_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    cpu_model = build_classifier(len(CLASS_NAMES))
    cpu_model.load_state_dict(model.state_dict())
    cpu_model.eval()

    dummy_tensor = torch.randn(1, 3, 224, 224)

    # Warmup forward
    with torch.no_grad():
        for _ in range(warmup):
            _ = cpu_model(dummy_tensor)

    # Benchmark forward pass
    forward_times = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.perf_counter()
            _ = cpu_model(dummy_tensor)
            t1 = time.perf_counter()
            forward_times.append((t1 - t0) * 1000.0)

    forward_times = np.array(forward_times)
    mean_lat = float(np.mean(forward_times))
    median_lat = float(np.median(forward_times))
    p95_lat = float(np.percentile(forward_times, 95))
    min_lat = float(np.min(forward_times))
    max_lat = float(np.max(forward_times))
    std_lat = float(np.std(forward_times))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    # Benchmark full pipeline (read image + transform + forward + softmax)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pipeline_times = []
    with torch.no_grad():
        for _ in range(min(trials, 50)):
            t0 = time.perf_counter()
            img = Image.open(sample_img_path).convert("RGB")
            inp = tf(img).unsqueeze(0)
            out = cpu_model(inp)
            _ = torch.softmax(out, dim=1).numpy()
            t1 = time.perf_counter()
            pipeline_times.append((t1 - t0) * 1000.0)

    mean_pipe = float(np.mean(pipeline_times))
    median_pipe = float(np.median(pipeline_times))

    torch.set_num_threads(orig_threads)

    print(f"  -> Forward Mean:    {mean_lat:.2f} ms ({fps:.1f} FPS)")
    print(f"  -> Forward Median:  {median_lat:.2f} ms | P95: {p95_lat:.2f} ms (Min: {min_lat:.2f} ms, Max: {max_lat:.2f} ms)")
    print(f"  -> Full Pipeline:   {mean_pipe:.2f} ms (Median: {median_pipe:.2f} ms)")

    return {
        "num_threads": num_threads,
        "batch_size": 1,
        "warmup_trials": warmup,
        "benchmark_trials": trials,
        "forward_mean_ms": round(mean_lat, 2),
        "forward_median_ms": round(median_lat, 2),
        "forward_p95_ms": round(p95_lat, 2),
        "forward_min_ms": round(min_lat, 2),
        "forward_max_ms": round(max_lat, 2),
        "forward_std_ms": round(std_lat, 2),
        "forward_fps": round(fps, 1),
        "full_pipeline_mean_ms": round(mean_pipe, 2),
        "full_pipeline_median_ms": round(median_pipe, 2),
        "passes_latency_threshold": bool(mean_lat < GATE_A_THRESHOLDS["max_cpu_latency_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3 on Final Test Set (Gate A).")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--warmup-trials", type=int, default=50)
    parser.add_argument("--benchmark-trials", type=int, default=200)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("=" * 80)
    print("TASK 2: OFFICIAL GATE A EVALUATION ON STRICTLY ISOLATED FINAL TEST SET")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print("=" * 80)

    # 1. Integrity Verification
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        return 1
    if not args.test_dir.exists():
        print(f"ERROR: Final Test directory not found: {args.test_dir}")
        return 1
    if not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}")
        return 1

    ckpt_sha256 = compute_sha256(args.checkpoint)
    manifest_sha256 = compute_sha256(args.manifest)
    manifest = pd.read_csv(args.manifest)
    test_manifest = manifest[manifest["split"] == "test"].copy()

    print(f"\n[1] Protocol & Resource Provenance:")
    print(f"  - Checkpoint Path:   {args.checkpoint}")
    print(f"  - Checkpoint SHA256: {ckpt_sha256}")
    print(f"  - Manifest Path:     {args.manifest}")
    print(f"  - Manifest SHA256:   {manifest_sha256}")
    print(f"  - Test samples in manifest: {len(test_manifest)}")

    assert len(test_manifest) == GATE_A_THRESHOLDS["expected_test_samples"], (
        f"Manifest test count {len(test_manifest)} != Expected {GATE_A_THRESHOLDS['expected_test_samples']}"
    )

    device = torch.device(args.device)
    use_amp = (device.type == "cuda")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n[2] Runtime Environment:")
    print(f"  - Evaluation Device: {device} ({gpu_name})")
    print(f"  - Precision:         {'AMP fp16 (autocast)' if use_amp else 'FP32'}")
    print(f"  - Batch Size:        {args.batch_size}")

    # 2. Build Dataset & DataLoader
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_dataset = datasets.ImageFolder(str(args.test_dir), transform=test_transform)
    assert test_dataset.classes == CLASS_NAMES, f"Classes mismatch: {test_dataset.classes} vs {CLASS_NAMES}"
    print(f"\n[3] Final Test Dataset on Disk:")
    print(f"  - Samples on disk: {len(test_dataset)}")
    assert len(test_dataset) == len(test_manifest), (
        f"Disk count {len(test_dataset)} != Manifest count {len(test_manifest)}"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"\n[4] Loaded Checkpoint Metadata:")
    print(f"  - Trained Epoch:        {ckpt.get('epoch')}")
    print(f"  - Baseline Val Acc:     {ckpt.get('val_acc')*100:.2f}%")
    print(f"  - Baseline Val Macro-F1:{ckpt.get('val_macro_f1'):.4f}")

    model = build_classifier(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # 4. CPU Latency Benchmark
    sample_img_path = Path(test_dataset.samples[0][0])
    latency_stats = benchmark_cpu_latency(
        model,
        sample_img_path,
        num_threads=args.cpu_threads,
        warmup=args.warmup_trials,
        trials=args.benchmark_trials,
    )

    # 5. Full Final Test Inference
    print("\n[5] Running Final Test Inference on All 2,223 Images...")
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 6. Physical Disk Hash Verification & Sample Prediction Table
    sample_paths = [Path(p) for p, _ in test_dataset.samples]
    test_manifest_sha_map = test_manifest.set_index("filename")["sha256"].to_dict()

    print("\n[6] Computing Direct Disk SHA-256 for all 2,223 Test Images...")
    t_hash_start = time.perf_counter()
    prediction_records = []
    disk_hash_mismatches = 0
    missing_manifest_count = 0

    for i in range(len(test_dataset)):
        sample_p = sample_paths[i]
        fname = sample_p.name
        t_cls = CLASS_NAMES[all_targets[i]]
        p_cls = CLASS_NAMES[all_preds[i]]
        conf = float(all_probs[i][all_preds[i]])
        is_corr = bool(all_targets[i] == all_preds[i])

        # Read directly from disk
        actual_disk_sha256 = compute_sha256(sample_p)

        # Cross-reference with manifest
        expected_manifest_sha = test_manifest_sha_map.get(fname)
        if expected_manifest_sha is None:
            missing_manifest_count += 1
        elif actual_disk_sha256 != expected_manifest_sha:
            disk_hash_mismatches += 1

        prediction_records.append({
            "sample_index": i + 1,
            "filename": fname,
            "true_label": t_cls,
            "predicted_label": p_cls,
            "confidence": round(conf, 4),
            "is_correct": is_corr,
            "sha256": actual_disk_sha256,
            "full_path": str(sample_p),
        })

    t_hash_elapsed = time.perf_counter() - t_hash_start
    print(f"  - Direct disk SHA-256 computed for {len(prediction_records)} samples in {t_hash_elapsed:.2f}s")
    print(f"  - Hash mismatches against manifest: {disk_hash_mismatches}")
    print(f"  - Missing from manifest:           {missing_manifest_count}")
    assert disk_hash_mismatches == 0, f"ERROR: {disk_hash_mismatches} test images do not match manifest SHA-256!"
    assert missing_manifest_count == 0, f"ERROR: {missing_manifest_count} images missing from manifest!"

    pred_df = pd.DataFrame(prediction_records)
    pred_csv_path = args.output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"  -> Saved full test prediction table to: {pred_csv_path} ({len(pred_df)} rows)")

    # 7. Compute Quantitative Metrics
    test_acc = float(accuracy_score(all_targets, all_preds))
    test_macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    correct_count = int(np.sum(all_targets == all_preds))
    error_count = len(all_targets) - correct_count

    print(f"\n[7] Final Test Evaluation Results (Unseen Data):")
    print(f"  - Total Test Samples:   {len(all_targets)}")
    print(f"  - Correct Predictions:  {correct_count} / {len(all_targets)}")
    print(f"  - Total Errors:         {error_count} / {len(all_targets)} ({error_count/len(all_targets)*100:.2f}%)")
    print(f"  - Top-1 Accuracy:       {test_acc*100:.2f}% (Threshold: >= {GATE_A_THRESHOLDS['min_accuracy']*100:.1f}%)")
    print(f"  - Macro-Averaged F1:    {test_macro_f1:.4f} (Threshold: >= {GATE_A_THRESHOLDS['min_macro_f1']:.3f})")

    # Per-Class Breakdown
    report_dict = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    print(f"\n[8] Per-Class Breakdown on Final Test Set:")
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 62)
    per_class_summary = {}
    recalls = []

    for c in CLASS_NAMES:
        c_p = report_dict[c]["precision"]
        c_r = report_dict[c]["recall"]
        c_f = report_dict[c]["f1-score"]
        c_sup = int(report_dict[c]["support"])
        recalls.append(c_r)
        print(f"{c:<12} | {c_p*100:>9.2f}% | {c_r*100:>9.2f}% | {c_f:>10.4f} | {c_sup:>8d}")
        per_class_summary[c] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1_score": round(c_f, 4),
            "support": c_sup,
        }

    battery_recall = per_class_summary["battery"]["recall"]
    min_class_recall = float(np.min(recalls))
    min_recall_class = CLASS_NAMES[int(np.argmin(recalls))]

    print(f"\n[9] Specific Gate Criteria Verification:")
    print(f"  - Battery Recall:    {battery_recall*100:.2f}% (Threshold: >= {GATE_A_THRESHOLDS['min_battery_recall']*100:.1f}%)")
    print(f"  - Min Class Recall:  {min_class_recall*100:.2f}% [{min_recall_class}] (Threshold: >= {GATE_A_THRESHOLDS['min_class_recall']*100:.1f}%)")
    print(f"  - CPU Forward Mean:  {latency_stats['forward_mean_ms']:.2f} ms (Threshold: < {GATE_A_THRESHOLDS['max_cpu_latency_ms']:.1f} ms)")
    print(f"  - Byte Mismatches:   {disk_hash_mismatches} (Threshold: == 0)")

    # 8. Check Gate A Criteria
    gate_checks = {
        "accuracy_criterion": {
            "metric": "top1_accuracy",
            "observed": round(test_acc, 4),
            "threshold": GATE_A_THRESHOLDS["min_accuracy"],
            "passed": bool(test_acc >= GATE_A_THRESHOLDS["min_accuracy"]),
        },
        "macro_f1_criterion": {
            "metric": "macro_f1",
            "observed": round(test_macro_f1, 4),
            "threshold": GATE_A_THRESHOLDS["min_macro_f1"],
            "passed": bool(test_macro_f1 >= GATE_A_THRESHOLDS["min_macro_f1"]),
        },
        "battery_recall_criterion": {
            "metric": "battery_recall",
            "observed": round(battery_recall, 4),
            "threshold": GATE_A_THRESHOLDS["min_battery_recall"],
            "passed": bool(battery_recall >= GATE_A_THRESHOLDS["min_battery_recall"]),
        },
        "min_class_recall_criterion": {
            "metric": "min_class_recall",
            "observed": round(min_class_recall, 4),
            "worst_class": min_recall_class,
            "threshold": GATE_A_THRESHOLDS["min_class_recall"],
            "passed": bool(min_class_recall >= GATE_A_THRESHOLDS["min_class_recall"]),
        },
        "cpu_latency_criterion": {
            "metric": "forward_mean_latency_ms",
            "observed": latency_stats["forward_mean_ms"],
            "threshold": GATE_A_THRESHOLDS["max_cpu_latency_ms"],
            "passed": bool(latency_stats["forward_mean_ms"] < GATE_A_THRESHOLDS["max_cpu_latency_ms"]),
        },
        "data_integrity_criterion": {
            "metric": "disk_sha256_mismatches",
            "observed": disk_hash_mismatches,
            "threshold": 0,
            "passed": bool(disk_hash_mismatches == 0),
        },
    }

    all_passed = all(check["passed"] for check in gate_checks.values())
    gate_status = "GATE_A_PASSED" if all_passed else "GATE_A_FAILED"

    print("\n" + "=" * 80)
    print(f"GATE A FINAL STATUS: {gate_status}")
    for k, v in gate_checks.items():
        res_str = "PASS" if v["passed"] else "FAIL"
        print(f"  [{res_str}] {k}: observed={v['observed']} vs threshold={v['threshold']}")
    print("=" * 80)

    # 9. Confusion Matrix & Error Analysis
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = args.output_dir / "test_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"  -> Saved Confusion Matrix to: {cm_path}")

    error_df = pred_df[~pred_df["is_correct"]].sort_values(by="confidence", ascending=False)
    error_csv_path = args.output_dir / "test_error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    print(f"  -> Saved Error Analysis to:    {error_csv_path} ({len(error_df)} errors)")

    # 10. Summary JSON Export
    t_total = time.perf_counter() - t_start
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "TASK 2 GATE A EVALUATION",
        "gate_status": gate_status,
        "all_criteria_passed": all_passed,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": ckpt_sha256,
            "trained_epoch": ckpt.get("epoch"),
        },
        "manifest": {
            "path": str(args.manifest),
            "sha256": manifest_sha256,
            "test_samples_count": len(test_manifest),
        },
        "runtime": {
            "device": str(device),
            "gpu_name": gpu_name,
            "precision": "AMP fp16" if use_amp else "FP32",
            "batch_size": args.batch_size,
            "elapsed_time_seconds": round(t_total, 2),
        },
        "cpu_latency": latency_stats,
        "metrics": {
            "top1_accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "correct_predictions": correct_count,
            "total_errors": error_count,
            "battery_recall": round(battery_recall, 4),
            "min_class_recall": round(min_class_recall, 4),
            "min_class_name": min_recall_class,
        },
        "gate_checks": gate_checks,
        "disk_hash_verification": {
            "total_samples_hashed_from_disk": len(prediction_records),
            "hash_calculation_method": "direct_disk_read_sha256",
            "mismatches_against_manifest": disk_hash_mismatches,
            "missing_from_manifest": missing_manifest_count,
            "passed": bool(disk_hash_mismatches == 0 and missing_manifest_count == 0),
        },
        "per_class": per_class_summary,
    }

    summary_json_path = args.output_dir / "gate_a_metrics.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved Gate A Summary to:   {summary_json_path}")

    # Export frozen protocol record
    frozen_protocol = {
        "protocol_name": "GATE_A_FROZEN_EVALUATION_PROTOCOL",
        "frozen_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoint_sha256": ckpt_sha256,
        "manifest_sha256": manifest_sha256,
        "class_names": CLASS_NAMES,
        "image_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "gate_thresholds": GATE_A_THRESHOLDS,
        "cpu_latency_setup": {
            "threads": args.cpu_threads,
            "warmup": args.warmup_trials,
            "trials": args.benchmark_trials,
            "batch_size": 1,
        },
    }
    protocol_json_path = args.output_dir / "gate_a_protocol_frozen.json"
    with open(protocol_json_path, "w", encoding="utf-8") as f:
        json.dump(frozen_protocol, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved Frozen Protocol to:  {protocol_json_path}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
