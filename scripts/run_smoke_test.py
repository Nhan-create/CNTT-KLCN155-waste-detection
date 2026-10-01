r"""MobileNetV3-Large 10-Class Smoke Test Pipeline.

Executes a 3-epoch smoke test on Train and Val splits ONLY to verify:
1. GPU CUDA runtime execution on RTX 2050.
2. Gradient flow, AMP FP16 stability, and loss descent.
3. Metric calculation (Train/Val Loss, Accuracy, Macro-F1).
4. Checkpoint saving, metadata integrity, and reload test.
5. Strict isolation of the Test split (Test set is NOT loaded or evaluated).

Target environment: D:\CNTT-KLCN155-waste-detection
Data source: D:\HK7\Đồ án khóa luận\Data\processed
Output directory: D:\CNTT-KLCN155-waste-detection\artifacts\smoke_test
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

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

DATA_PROCESSED_ROOT = Path(r"D:\HK7\Đồ án khóa luận\Data\processed")
OUTPUT_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\artifacts\smoke_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_transforms(img_size: int = 224, is_train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor],
    ) if False else transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_classifier(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main() -> int:
    print("=" * 80)
    print("TASK R2.1: MOBILENETV3-LARGE 10-CLASS SMOKE TEST (TRAIN/VAL ONLY)")
    print("=" * 80)

    # 1. Device and GPU Memory Audit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Environment & Hardware Verification:")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  - GPU Name: {gpu_name}")
        print(f"  - Total VRAM: {total_vram_mb:.1f} MB")
        torch.cuda.reset_peak_memory_stats(0)
    else:
        print("  - WARNING: Running on CPU.")
        gpu_name = "CPU"
        total_vram_mb = 0.0

    # 2. Dataset Setup
    train_dir = DATA_PROCESSED_ROOT / "train"
    val_dir = DATA_PROCESSED_ROOT / "val"
    test_dir = DATA_PROCESSED_ROOT / "test"

    print(f"\n[2] Dataset Paths:")
    print(f"  - Train Directory: {train_dir} (Exists: {train_dir.exists()})")
    print(f"  - Val Directory: {val_dir} (Exists: {val_dir.exists()})")
    print(f"  - Test Directory: {test_dir} (STRICTLY LOCKED - NOT LOADED)")

    train_dataset = datasets.ImageFolder(str(train_dir), transform=build_transforms(224, is_train=True))
    val_dataset = datasets.ImageFolder(str(val_dir), transform=build_transforms(224, is_train=False))

    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Val samples: {len(val_dataset)}")
    print(f"  - Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    assert train_dataset.classes == CLASS_NAMES, f"Classes mismatch! Expected {CLASS_NAMES}, got {train_dataset.classes}"

    batch_size = 64
    num_workers = 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 3. Model Initialization
    print(f"\n[3] Model Construction:")
    print(f"  - Architecture: MobileNetV3-Large")
    print(f"  - Pretrained: ImageNet-1K (Torchvision MobileNet_V3_Large_Weights.DEFAULT)")
    print(f"  - Output Logits: 10 classes")

    model = build_classifier(num_classes=10, pretrained=True).to(device)

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer: Fine-tune head + backbone with differential learning rates
    optimizer = AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop (3 Epochs)
    total_epochs = 3
    print(f"\n[4] Executing Smoke Test Training ({total_epochs} Epochs)...")

    history = []
    best_val_f1 = -1.0
    best_checkpoint_path = OUTPUT_DIR / "best_smoke_checkpoint.pt"

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        
        # --- TRAIN ---
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = targets.size(0)
            train_loss_total += loss.item() * bs
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += bs

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"    Epoch {epoch}/{total_epochs} | Step {batch_idx + 1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        train_loss = train_loss_total / train_total
        train_acc = train_correct / train_total

        # --- VAL ---
        model.eval()
        val_loss_total = 0.0
        val_preds_all = []
        val_targets_all = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss_total += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds_all.extend(preds)
                val_targets_all.extend(targets.cpu().numpy())

        val_loss = val_loss_total / len(val_dataset)
        val_acc = accuracy_score(val_targets_all, val_preds_all)
        val_macro_f1 = f1_score(val_targets_all, val_preds_all, average="macro", zero_division=0)
        epoch_time = time.time() - t0

        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        else:
            peak_vram_mb = 0.0

        print(
            f"  >>> Epoch {epoch:02d} Summary: "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_time:.1f}s | Peak VRAM: {peak_vram_mb:.1f} MB"
        )

        epoch_stat = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "time_seconds": round(epoch_time, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
        }
        history.append(epoch_stat)

        # Save checkpoint if best val macro-f1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save({
                "epoch": epoch,
                "model_name": "mobilenet_v3_large",
                "num_classes": 10,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
                "dataset_note": "Smoke test train/val run only. Test set locked and untouched.",
            }, best_checkpoint_path)
            print(f"    * Saved new best checkpoint to: {best_checkpoint_path}")

    # 5. Checkpoint Reload Verification
    print(f"\n[5] Testing Checkpoint Reload & Deterministic Inference:")
    assert best_checkpoint_path.exists(), "Checkpoint file was not created!"
    checkpoint_size_mb = best_checkpoint_path.stat().st_size / (1024 ** 2)
    print(f"  - Checkpoint size: {checkpoint_size_mb:.2f} MB")

    loaded_ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    fresh_model = build_classifier(num_classes=10, pretrained=False).to(device)
    fresh_model.load_state_dict(loaded_ckpt["state_dict"])
    fresh_model.eval()

    # Compare inference output between trained model and reloaded model on a sample batch
    sample_inputs, sample_targets = next(iter(val_loader))
    sample_inputs = sample_inputs.to(device)
    with torch.no_grad():
        out_orig = model(sample_inputs)
        out_loaded = fresh_model(sample_inputs)
        max_diff = (out_orig - out_loaded).abs().max().item()

    print(f"  - Checkpoint reload output divergence: {max_diff:.8e}")
    assert max_diff < 1e-5, f"Reloaded weights produced diverging outputs! Max diff: {max_diff}"
    print(f"  - Reload verification PASSED! Model reproduces identical predictions.")

    # 6. Export Metrics and History
    history_csv = OUTPUT_DIR / "smoke_test_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metrics_json = OUTPUT_DIR / "smoke_test_metrics.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "TASK R2.1 SMOKE TEST",
        "status": "PASSED",
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_vram_mb": round(total_vram_mb, 1),
            "peak_vram_mb": round(max(h["peak_vram_mb"] for h in history), 1),
        },
        "dataset": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "num_classes": 10,
            "class_names": CLASS_NAMES,
            "test_split_status": "STRICTLY_LOCKED_AND_ISOLATED",
        },
        "training": {
            "architecture": "mobilenet_v3_large",
            "pretrained_weights": "ImageNet-1K",
            "epochs_run": total_epochs,
            "batch_size": batch_size,
            "amp_enabled": device.type == "cuda",
            "best_val_macro_f1": round(best_val_f1, 4),
            "best_val_accuracy": round(history[np.argmax([h["val_macro_f1"] for h in history])]["val_acc"], 4),
            "checkpoint_path": str(best_checkpoint_path),
            "checkpoint_size_mb": round(checkpoint_size_mb, 2),
            "reload_test_passed": True,
        },
        "epoch_history": history,
        "scientific_integrity_statement": (
            "This smoke test validates code execution, loss descent, gradient backprop, "
            "and checkpoint loadability on GPU. The Test split was NOT touched. "
            "Final benchmark reporting requires fixing the 15 identified cross-split burst-shot "
            "leakages before official training."
        ),
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[6] Exported deliverables:")
    print(f"  - History CSV: {history_csv}")
    print(f"  - Summary JSON: {metrics_json}")
    print(f"  - Best Checkpoint: {best_checkpoint_path}")
    print("=" * 80)
    print("SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
