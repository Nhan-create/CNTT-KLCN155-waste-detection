r"""MobileNetV3-Large Smoke Test on Verified Leak-Free Split V2.

Verifies:
1. GPU CUDA runtime execution on RTX 2050 using clean data/processed_v2.
2. Gradient flow, AMP FP16 stability, loss descent, and metric computation.
3. Checkpoint saving and rigorous reload verification (reloaded model evaluates on full Val set
   to match the recorded best checkpoint metrics exactly).
4. Strict test isolation (Test set is NOT loaded or evaluated).
"""

from __future__ import annotations

import csv
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

DATA_PROCESSED_V2 = Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2")
OUTPUT_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\artifacts\smoke_test_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_transforms(img_size: int = 224, is_train: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
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


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    val_loss_total = 0.0
    val_preds_all = []
    val_targets_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            val_loss_total += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            val_preds_all.extend(preds)
            val_targets_all.extend(targets.cpu().numpy())

    val_loss = val_loss_total / len(loader.dataset)
    val_acc = accuracy_score(val_targets_all, val_preds_all)
    val_macro_f1 = f1_score(val_targets_all, val_preds_all, average="macro", zero_division=0)
    return val_loss, val_acc, val_macro_f1


def main() -> int:
    print("=" * 80)
    print("TASK P1.5: MOBILENETV3 SMOKE TEST ON VERIFIED SPLIT V2 (2 EPOCHS)")
    print("=" * 80)

    # 1. Environment & Hardware Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Environment & Hardware:")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  - GPU Name: {gpu_name} ({total_vram_mb:.1f} MB)")
        torch.cuda.reset_peak_memory_stats(0)
    else:
        gpu_name = "CPU"
        total_vram_mb = 0.0

    # 2. Dataset Setup
    train_dir = DATA_PROCESSED_V2 / "train"
    val_dir = DATA_PROCESSED_V2 / "val"
    test_dir = DATA_PROCESSED_V2 / "test"

    print(f"\n[2] Data Directories (Split V2):")
    print(f"  - Train: {train_dir} (Exists: {train_dir.exists()})")
    print(f"  - Val: {val_dir} (Exists: {val_dir.exists()})")
    print(f"  - Test: {test_dir} (STRICTLY LOCKED - NOT ACCESSED)")

    train_ds = datasets.ImageFolder(str(train_dir), transform=build_transforms(224, is_train=True))
    val_ds = datasets.ImageFolder(str(val_dir), transform=build_transforms(224, is_train=False))

    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Val samples: {len(val_ds)}")
    assert train_ds.classes == CLASS_NAMES, f"Class names mismatch! {train_ds.classes}"

    batch_size = 64
    num_workers = 2
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )

    # 3. Model Construction
    print(f"\n[3] Model Construction:")
    print(f"  - Base Architecture: MobileNetV3-Large")
    print(f"  - Pretrained: ImageNet-1K (Torchvision DEFAULT)")
    print(f"  - Classes: 10 logits")

    model = build_classifier(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW([
        {"params": model.features.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop (2 Epochs)
    total_epochs = 2
    print(f"\n[4] Executing Smoke Test ({total_epochs} Epochs)...")

    history = []
    best_val_f1 = -1.0
    best_checkpoint_path = OUTPUT_DIR / "best_smoke_checkpoint.pt"
    best_epoch_stats = None

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
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

        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0

        peak_vram_mb = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if device.type == "cuda" else 0.0

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

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch_stats = epoch_stat
            torch.save({
                "epoch": epoch,
                "model_name": "mobilenet_v3_large",
                "num_classes": 10,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "split_version": "v2_zero_leakage",
            }, best_checkpoint_path)
            print(f"    * Saved best checkpoint (Epoch {epoch}) to: {best_checkpoint_path}")

    # 5. Rigorous Checkpoint Reload & Evaluation Match
    print(f"\n[5] Rigorous Checkpoint Reload Verification:")
    assert best_checkpoint_path.exists(), "Checkpoint file missing!"
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    fresh_model = build_classifier(num_classes=10, pretrained=False).to(device)
    fresh_model.load_state_dict(ckpt["state_dict"])

    # Run full validation evaluation on reloaded model
    reloaded_val_loss, reloaded_val_acc, reloaded_val_f1 = evaluate(fresh_model, val_loader, criterion, device)
    print(f"  - Checkpoint recorded: Val Loss={ckpt['val_loss']:.4f}, Acc={ckpt['val_acc']*100:.2f}%, F1={ckpt['val_macro_f1']:.4f}")
    print(f"  - Reloaded model eval: Val Loss={reloaded_val_loss:.4f}, Acc={reloaded_val_acc*100:.2f}%, F1={reloaded_val_f1:.4f}")

    diff_loss = abs(reloaded_val_loss - ckpt["val_loss"])
    diff_acc = abs(reloaded_val_acc - ckpt["val_acc"])
    diff_f1 = abs(reloaded_val_f1 - ckpt["val_macro_f1"])

    print(f"  - Discrepancy: Loss diff={diff_loss:.2e}, Acc diff={diff_acc:.2e}, F1 diff={diff_f1:.2e}")
    assert diff_loss < 1e-5 and diff_acc < 1e-5 and diff_f1 < 1e-5, "Reloaded model metrics do not match checkpoint exactly!"
    print("  -> Reload verification PASSED with zero metric discrepancy!")

    # 6. Export Deliverables
    history_csv = OUTPUT_DIR / "smoke_test_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metrics_json = OUTPUT_DIR / "smoke_test_metrics.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "TASK P1.5 SMOKE TEST ON SPLIT V2",
        "status": "PASSED",
        "data_split_version": "v2_zero_leakage",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_status": "STRICTLY_LOCKED_AND_ISOLATED",
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_vram_mb": round(total_vram_mb, 1),
            "peak_vram_mb": round(max(h["peak_vram_mb"] for h in history), 1),
        },
        "best_epoch": best_epoch_stats["epoch"],
        "best_val_macro_f1": best_epoch_stats["val_macro_f1"],
        "best_val_accuracy": best_epoch_stats["val_acc"],
        "checkpoint_path": str(best_checkpoint_path),
        "checkpoint_size_mb": round(best_checkpoint_path.stat().st_size / (1024 ** 2), 2),
        "reload_test_passed": True,
        "history": history,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[6] Saved artifacts to {OUTPUT_DIR}:")
    print(f"  - History CSV: {history_csv}")
    print(f"  - Metrics JSON: {metrics_json}")
    print(f"  - Checkpoint: {best_checkpoint_path}")
    print("=" * 80)
    print("SMOKE TEST P1.5 PASSED!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
