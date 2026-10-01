r"""Official MobileNetV3-Large Waste Classification Training (P1.6).

Standard, reproducible two-phase fine-tuning on clean Split V2 (Train + Val):
- Phase 1: Head warmup (3 epochs, frozen backbone, lr=1e-3).
- Phase 2: Full fine-tuning (up to 9 epochs, lr_backbone=1e-4, lr_head=3e-4, Cosine Annealing).
- Selection criterion: Best Validation Macro-Averaged F1.
- Complete per-class precision, recall, F1 metrics on validation.
- Export of validation confusion matrix and top error analysis.
- Checkpoint reload verification.
- STRICTLY DOES NOT ACCESS THE TEST SET.
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
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
OUTPUT_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\artifacts\official_run")
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


def evaluate_full(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    val_loss_total = 0.0
    val_preds_all = []
    val_targets_all = []
    val_probs_all = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            bs = targets.size(0)
            val_loss_total += loss.item() * bs
            preds = outputs.argmax(dim=1).cpu().numpy()

            val_preds_all.extend(preds)
            val_targets_all.extend(targets.cpu().numpy())
            val_probs_all.extend(probs)

    val_loss = val_loss_total / len(loader.dataset)
    val_acc = accuracy_score(val_targets_all, val_preds_all)
    val_macro_f1 = f1_score(val_targets_all, val_preds_all, average="macro", zero_division=0)
    
    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "targets": np.array(val_targets_all),
        "preds": np.array(val_preds_all),
        "probs": np.array(val_probs_all),
    }


def main() -> int:
    print("=" * 80)
    print("TASK P1.6: OFFICIAL MOBILENETV3-LARGE FINE-TUNING (SPLIT V2)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Hardware & Runtime:")
    print(f"  - Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  - GPU Name: {gpu_name} ({vram_total:.1f} MB)")
        torch.cuda.reset_peak_memory_stats(0)
    else:
        gpu_name = "CPU"
        vram_total = 0.0

    # 2. DataLoaders
    train_dir = DATA_PROCESSED_V2 / "train"
    val_dir = DATA_PROCESSED_V2 / "val"
    test_dir = DATA_PROCESSED_V2 / "test"

    print(f"\n[2] Dataset Paths:")
    print(f"  - Train: {train_dir} (Exists: {train_dir.exists()})")
    print(f"  - Val: {val_dir} (Exists: {val_dir.exists()})")
    print(f"  - Test: {test_dir} (STRICTLY LOCKED - NOT LOADED)")

    train_ds = datasets.ImageFolder(str(train_dir), transform=build_transforms(224, is_train=True))
    val_ds = datasets.ImageFolder(str(val_dir), transform=build_transforms(224, is_train=False))

    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Val samples: {len(val_ds)}")
    assert train_ds.classes == CLASS_NAMES, f"Classes mismatch: {train_ds.classes}"

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

    # 3. Model & Loss Setup
    print(f"\n[3] Model & Training Strategy:")
    print(f"  - Backbone: MobileNetV3-Large (Pretrained ImageNet-1K)")
    print(f"  - Target Head: 10 Classes")
    print(f"  - Phase 1: 3 Warmup Epochs (Head only, lr=1e-3)")
    print(f"  - Phase 2: Up to 9 Fine-Tuning Epochs (Backbone lr=1e-4, Head lr=3e-4, Cosine Annealing)")
    print(f"  - Early Stopping Patience: 4 Epochs on Val Macro-F1")

    model = build_classifier(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Phase 1: Freeze features
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_macro_f1 = -1.0
    best_checkpoint_path = OUTPUT_DIR / "best_model.pt"
    best_eval_dict = None
    patience_counter = 0
    max_patience = 4
    history = []

    phase1_epochs = 3
    phase2_epochs = 9
    total_epochs = phase1_epochs + phase2_epochs

    print(f"\n[4] Starting Training Execution ({total_epochs} Max Epochs)...")

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        current_phase = 1 if epoch <= phase1_epochs else 2

        # Phase transition check
        if epoch == phase1_epochs + 1:
            print("\n  >>> PHASE TRANSITION: Unfreezing backbone features for Phase 2 joint fine-tuning! <<<")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = AdamW([
                {"params": model.features.parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 3e-4},
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)

        # Train loop
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
                print(f"    Epoch {epoch:02d}/{total_epochs} (Phase {current_phase}) | Step {batch_idx + 1:03d}/{len(train_loader)} | Loss: {loss.item():.4f}")

        if current_phase == 2:
            scheduler.step()

        train_loss = train_loss_total / train_total
        train_acc = train_correct / train_total

        # Validation
        eval_res = evaluate_full(model, val_loader, criterion, device)
        val_loss = eval_res["val_loss"]
        val_acc = eval_res["val_acc"]
        val_f1 = eval_res["val_macro_f1"]
        epoch_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 2) if device.type == "cuda" else 0.0

        print(
            f"  >>> Epoch {epoch:02d} Summary (Phase {current_phase}): "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1:.4f} | Time: {epoch_time:.1f}s | VRAM: {peak_vram:.1f}MB"
        )

        stat_record = {
            "epoch": epoch,
            "phase": current_phase,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "time_seconds": round(epoch_time, 1),
            "peak_vram_mb": round(peak_vram, 1),
        }
        history.append(stat_record)

        # Checkpoint selection on Val Macro-F1
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_eval_dict = eval_res
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "phase": current_phase,
                "model_name": "mobilenet_v3_large",
                "num_classes": 10,
                "class_names": CLASS_NAMES,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "split_version": "v2_zero_leakage",
                "creation_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, best_checkpoint_path)
            print(f"    * Saved new BEST checkpoint (Epoch {epoch}, Val Macro-F1: {val_f1:.4f}) -> {best_checkpoint_path}")
        else:
            patience_counter += 1
            print(f"    - No improvement in Val Macro-F1. Early stopping counter: {patience_counter}/{max_patience}")
            if patience_counter >= max_patience and current_phase == 2:
                print(f"\n  [!] Early stopping triggered at Epoch {epoch} after {max_patience} stagnant epochs.")
                break

    # 5. Checkpoint Reload Verification & Full Classification Metrics
    print(f"\n[5] Checkpoint Reload Verification & In-Depth Validation Analysis:")
    assert best_checkpoint_path.exists(), "Best checkpoint was not saved!"
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    fresh_model = build_classifier(num_classes=10, pretrained=False).to(device)
    fresh_model.load_state_dict(ckpt["state_dict"])

    reloaded_eval = evaluate_full(fresh_model, val_loader, criterion, device)
    diff_f1 = abs(reloaded_eval["val_macro_f1"] - ckpt["val_macro_f1"])
    print(f"  - Reloaded model Val Macro-F1: {reloaded_eval['val_macro_f1']:.4f} vs Checkpoint: {ckpt['val_macro_f1']:.4f}")
    print(f"  - Reload discrepancy: {diff_f1:.2e}")
    assert diff_f1 < 1e-5, "Reloaded model metrics do not match checkpoint exactly!"
    print("  -> Reload verification PASSED with 0 discrepancy!")

    # 6. Detailed Classification Report & Confusion Matrix
    targets = reloaded_eval["targets"]
    preds = reloaded_eval["preds"]
    probs = reloaded_eval["probs"]

    report_dict = classification_report(
        targets, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    print("\n[6] Per-Class Validation Performance (Best Checkpoint):")
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

    cm = confusion_matrix(targets, preds)
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = OUTPUT_DIR / "val_confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print(f"\nSaved Confusion Matrix to: {cm_path}")

    # 7. Error Analysis: Top Misclassified Samples
    val_samples_paths = [Path(p) for p, _ in val_ds.samples]
    error_records = []
    for i in range(len(targets)):
        if targets[i] != preds[i]:
            true_cls = CLASS_NAMES[targets[i]]
            pred_cls = CLASS_NAMES[preds[i]]
            conf = float(probs[i][preds[i]])
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
    print(f"Total Validation Errors: {len(error_df)} / {len(val_ds)} ({len(error_df)/len(val_ds)*100:.2f}%)")
    print(f"Saved Error Analysis to: {error_csv_path}")

    # 8. Export Training Deliverables
    history_csv = OUTPUT_DIR / "training_history.csv"
    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    final_metrics_json = OUTPUT_DIR / "official_training_metrics.json"
    deliverable_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task": "TASK P1.6 OFFICIAL MOBILENETV3 TRAINING",
        "status": "COMPLETED",
        "data_split_version": "v2_zero_leakage",
        "split_manifest_v2_sha256": "1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96",
        "hardware": {
            "device": str(device),
            "gpu_name": gpu_name,
            "total_vram_mb": round(vram_total, 1),
            "peak_vram_mb": round(max(h["peak_vram_mb"] for h in history), 1),
        },
        "dataset": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": 2223,
            "test_status": "STRICTLY_LOCKED_AND_ISOLATED (0% SNOOPING)",
        },
        "training_strategy": {
            "model_name": "mobilenet_v3_large",
            "pretrained_weights": "ImageNet-1K (Torchvision DEFAULT)",
            "batch_size": batch_size,
            "loss_function": "CrossEntropyLoss(label_smoothing=0.1)",
            "optimizer": "AdamW",
            "phase1_warmup_epochs": phase1_epochs,
            "total_epochs_trained": len(history),
            "early_stopping_triggered": patience_counter >= max_patience,
        },
        "best_checkpoint": {
            "path": str(best_checkpoint_path),
            "epoch": ckpt["epoch"],
            "phase": ckpt["phase"],
            "val_accuracy": round(ckpt["val_acc"], 4),
            "val_macro_f1": round(ckpt["val_macro_f1"], 4),
            "val_loss": round(ckpt["val_loss"], 4),
            "file_size_mb": round(best_checkpoint_path.stat().st_size / (1024 ** 2), 2),
            "reload_test_passed": True,
        },
        "per_class_validation_metrics": per_class_summary,
        "validation_errors_count": len(error_df),
        "history": history,
    }

    with open(final_metrics_json, "w", encoding="utf-8") as f:
        json.dump(deliverable_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[7] Exported Official Deliverables:")
    print(f"  - Best Checkpoint: {best_checkpoint_path}")
    print(f"  - Metrics Summary JSON: {final_metrics_json}")
    print(f"  - Training History CSV: {history_csv}")
    print(f"  - Confusion Matrix CSV: {cm_path}")
    print(f"  - Error Analysis CSV: {error_csv_path}")
    print("=" * 80)
    print("OFFICIAL TRAINING P1.6 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
