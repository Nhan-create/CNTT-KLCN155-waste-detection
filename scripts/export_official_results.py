r"""Export Official Results, Confusion Matrix, and Error Analysis for P1.6.

Loads the saved best checkpoint from `artifacts/official_run/best_model.pt`,
evaluates on `data/processed_v2/val` (leaving test set locked and untouched),
and exports all required deliverables.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

# UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

PROJECT_ROOT = Path("D:/CNTT-KLCN155-waste-detection")
VAL_DIR = PROJECT_ROOT / "data" / "processed_v2" / "val"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "official_run" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "official_run"

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

# History extracted from official run execution (task-744)
TRAINING_HISTORY = [
    {"epoch": 1, "phase": 1, "train_loss": 0.9685, "train_acc": 0.8275, "val_loss": 0.7757, "val_acc": 0.9051, "val_macro_f1": 0.8997, "epoch_time_sec": 120.6, "peak_vram_mb": 312.7},
    {"epoch": 2, "phase": 1, "train_loss": 0.7756, "train_acc": 0.9078, "val_loss": 0.7463, "val_acc": 0.9253, "val_macro_f1": 0.9215, "epoch_time_sec": 127.2, "peak_vram_mb": 312.7},
    {"epoch": 3, "phase": 1, "train_loss": 0.7227, "train_acc": 0.9348, "val_loss": 0.7530, "val_acc": 0.9208, "val_macro_f1": 0.9140, "epoch_time_sec": 128.9, "peak_vram_mb": 312.7},
    {"epoch": 4, "phase": 2, "train_loss": 0.6554, "train_acc": 0.9616, "val_loss": 0.6841, "val_acc": 0.9433, "val_macro_f1": 0.9375, "epoch_time_sec": 114.0, "peak_vram_mb": 1514.5},
    {"epoch": 5, "phase": 2, "train_loss": 0.5933, "train_acc": 0.9877, "val_loss": 0.6681, "val_acc": 0.9465, "val_macro_f1": 0.9415, "epoch_time_sec": 115.7, "peak_vram_mb": 1514.5},
    {"epoch": 6, "phase": 2, "train_loss": 0.5659, "train_acc": 0.9945, "val_loss": 0.6496, "val_acc": 0.9510, "val_macro_f1": 0.9450, "epoch_time_sec": 119.3, "peak_vram_mb": 1514.5},
    {"epoch": 7, "phase": 2, "train_loss": 0.5503, "train_acc": 0.9977, "val_loss": 0.6373, "val_acc": 0.9595, "val_macro_f1": 0.9527, "epoch_time_sec": 131.9, "peak_vram_mb": 1514.5},
    {"epoch": 8, "phase": 2, "train_loss": 0.5396, "train_acc": 0.9992, "val_loss": 0.6312, "val_acc": 0.9586, "val_macro_f1": 0.9514, "epoch_time_sec": 143.7, "peak_vram_mb": 1514.5},
    {"epoch": 9, "phase": 2, "train_loss": 0.5340, "train_acc": 0.9991, "val_loss": 0.6285, "val_acc": 0.9586, "val_macro_f1": 0.9516, "epoch_time_sec": 130.6, "peak_vram_mb": 1514.5},
    {"epoch": 10, "phase": 2, "train_loss": 0.5312, "train_acc": 0.9994, "val_loss": 0.6263, "val_acc": 0.9613, "val_macro_f1": 0.9555, "epoch_time_sec": 139.8, "peak_vram_mb": 1514.5},
    {"epoch": 11, "phase": 2, "train_loss": 0.5286, "train_acc": 0.9997, "val_loss": 0.6256, "val_acc": 0.9604, "val_macro_f1": 0.9548, "epoch_time_sec": 135.4, "peak_vram_mb": 1514.5},
    {"epoch": 12, "phase": 2, "train_loss": 0.5282, "train_acc": 0.9992, "val_loss": 0.6248, "val_acc": 0.9618, "val_macro_f1": 0.9559, "epoch_time_sec": 142.2, "peak_vram_mb": 1514.5},
]


def build_model(num_classes: int = 10) -> nn.Module:
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main():
    print("=" * 80)
    print("TASK P1.6: EXPORT OFFICIAL DELIVERABLES & DETAILED VALIDATION METRICS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Validation Samples: {len(val_ds)} from {VAL_DIR}", flush=True)
    assert val_ds.classes == CLASS_NAMES, f"Classes mismatch: {val_ds.classes} vs {CLASS_NAMES}"

    # Load Checkpoint
    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}", flush=True)
    print(f"  - Checkpoint Epoch: {ckpt['epoch']}", flush=True)
    print(f"  - Checkpoint Val Macro F1: {ckpt['val_macro_f1']:.4f}", flush=True)
    print(f"  - Checkpoint Val Accuracy: {ckpt['val_acc']*100:.2f}%", flush=True)
    print(f"  - Checkpoint Val Loss: {ckpt['val_loss']:.4f}", flush=True)

    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Run full validation inference
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    print(f"\nEmpirical Evaluation Results:", flush=True)
    print(f"  - Accuracy: {acc*100:.2f}% ({np.sum(all_targets == all_preds)} / {len(all_targets)})", flush=True)
    print(f"  - Macro F1: {macro_f1:.4f}", flush=True)

    diff_acc = abs(acc - ckpt["val_acc"])
    diff_f1 = abs(macro_f1 - ckpt["val_macro_f1"])
    print(f"Difference with checkpoint: acc={diff_acc:.2e}, f1={diff_f1:.2e}", flush=True)
    assert diff_acc < 1e-3, f"Accuracy mismatch: {acc} vs {ckpt['val_acc']}"
    assert diff_f1 < 1e-3, f"Macro F1 mismatch: {macro_f1} vs {ckpt['val_macro_f1']}"
    print("Verification PASSED!", flush=True)

    # 1. Classification Report & Per-Class Metrics
    report_dict = classification_report(
        all_targets, all_preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    print("\nPer-Class Breakdown:")
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 62)
    per_class_summary = {}
    for c in CLASS_NAMES:
        c_p = report_dict[c]["precision"]
        c_r = report_dict[c]["recall"]
        c_f = report_dict[c]["f1-score"]
        c_sup = int(report_dict[c]["support"])
        print(f"{c:<12} | {c_p*100:>9.2f}% | {c_r*100:>9.2f}% | {c_f:>10.4f} | {c_sup:>8d}")
        per_class_summary[c] = {
            "precision": round(c_p, 4),
            "recall": round(c_r, 4),
            "f1_score": round(c_f, 4),
            "support": c_sup,
        }

    # 2. Confusion Matrix CSV
    cm = confusion_matrix(all_targets, all_preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = OUTPUT_DIR / "val_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"\n[Saved] Confusion Matrix: {cm_path}")

    # 3. Error Analysis CSV
    val_samples_paths = [Path(p) for p, _ in val_ds.samples]
    error_records = []
    for i in range(len(all_targets)):
        if all_targets[i] != all_preds[i]:
            true_cls = CLASS_NAMES[all_targets[i]]
            pred_cls = CLASS_NAMES[all_preds[i]]
            conf = float(all_probs[i][all_preds[i]])
            img_rel_path = val_samples_paths[i].name
            error_records.append({
                "filename": img_rel_path,
                "true_label": true_cls,
                "predicted_label": pred_cls,
                "confidence": round(conf, 4),
                "full_path": str(val_samples_paths[i]),
            })

    error_df = pd.DataFrame(error_records).sort_values(by="confidence", ascending=False)
    error_csv_path = OUTPUT_DIR / "val_error_analysis.csv"
    error_df.to_csv(error_csv_path, index=False)
    print(f"[Saved] Error Analysis: {error_csv_path} ({len(error_df)} errors / {len(val_ds)} total = {len(error_df)/len(val_ds)*100:.2f}%)")

    # 4. Training History CSV
    history_csv_path = OUTPUT_DIR / "training_history.csv"
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(TRAINING_HISTORY[0].keys()))
        writer.writeheader()
        writer.writerows(TRAINING_HISTORY)
    print(f"[Saved] Training History: {history_csv_path}")

    # 5. Deliverable Summary JSON
    metrics_summary_path = OUTPUT_DIR / "official_training_metrics.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "TASK P1.6 OFFICIAL MOBILENETV3 TRAINING",
        "status": "COMPLETED",
        "data_split_version": "v2_zero_leakage",
        "split_manifest_v2_sha256": "1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96",
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1) if torch.cuda.is_available() else 0,
            "peak_vram_mb": 1514.5,
        },
        "dataset": {
            "train_samples": 10383,
            "val_samples": len(val_ds),
            "test_samples": 2223,
            "test_status": "STRICTLY_LOCKED_AND_ISOLATED (0% SNOOPING, ZERO LEAKAGE)",
        },
        "training_strategy": {
            "model_name": "mobilenet_v3_large",
            "pretrained_weights": "ImageNet-1K (Torchvision DEFAULT)",
            "batch_size": 64,
            "loss_function": "CrossEntropyLoss(label_smoothing=0.1)",
            "optimizer": "AdamW",
            "phase1_warmup_epochs": 3,
            "phase2_finetune_epochs": 9,
            "total_epochs_trained": 12,
            "early_stopping_patience": 4,
            "early_stopping_triggered": False,
        },
        "best_checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "epoch": ckpt["epoch"],
            "phase": ckpt["phase"],
            "val_accuracy": round(float(acc), 4),
            "val_macro_f1": round(float(macro_f1), 4),
            "val_loss": round(float(ckpt["val_loss"]), 4),
            "file_size_mb": round(CHECKPOINT_PATH.stat().st_size / (1024 ** 2), 2),
            "reload_test_passed": True,
        },
        "per_class_validation_metrics": per_class_summary,
        "validation_errors_count": len(error_df),
        "history": TRAINING_HISTORY,
    }

    with open(metrics_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Metrics Summary JSON: {metrics_summary_path}")

    print("\nAll deliverables generated and verified successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
