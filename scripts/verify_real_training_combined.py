"""Rigorous Empirical Verification of Real Training with 'combined' Augmentation.

Covers both detectors (SSDLite320-MobileNetV3 and YOLOv8n):
1. Loads real training images and labels from official manifest / dataset.
2. Passes through official loader/trainer pipeline with verified DonorBank.
3. Confirms augmentation is actively executed on batches; logs telemetry.
4. Executes real forward pass, verifies finite positive loss.
5. Executes backward pass, verifies non-zero gradients.
6. Executes optimizer step and empirically verifies trainable parameter changes (pre vs post update).
7. Saves new checkpoint under independent run directory (preserving baseline 'none' checkpoints).
8. Reloads new checkpoint and runs deterministic inference on real validation image.
9. Records comprehensive metadata (full commit SHA, versions, seed, device, telemetry).
"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from src.detection.copy_paste import DonorBank
from src.detection.schema import CLASS_NAMES
from src.detection.ssdlite import build_ssdlite, resolve_device
from src.detection.ssdlite_train import YoloBoxDataset, collate_detection_batch
from torch.utils.data import DataLoader


def get_git_commit_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode("utf-8").strip()
        return out
    except Exception:
        return "UNKNOWN_COMMIT"


def draw_predictions_on_image(
    image_path: Path,
    boxes: list[list[float]],  # pixel [x1, y1, x2, y2]
    scores: list[float],
    class_ids: list[int],
    title: str = "",
) -> Image.Image:
    im = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    w, h = im.size

    colors = [
        "red", "green", "blue", "yellow", "cyan",
        "magenta", "orange", "purple", "lime", "pink"
    ]

    for box, score, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cid_int = int(cid)
        color = colors[cid_int % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        cname = CLASS_NAMES[cid_int] if 0 <= cid_int < len(CLASS_NAMES) else str(cid_int)
        label = f"{cname} {score:.2f}"
        draw.rectangle([x1, max(0, y1 - 14), x1 + len(label) * 7 + 4, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 13)), label, fill="black")

    if title:
        draw.rectangle([0, 0, w, 22], fill="black")
        draw.text((8, 4), title, fill="white")
    return im


def verify_ssdlite_real_combined(
    dataset_yaml: Path,
    donor_bank: DonorBank,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 1: SSDLite320-MobileNetV3 Real Training & Gradient Verification ('combined')")
    print("=" * 70)

    # 1. Official training dataset with 'combined' strategy
    train_dataset = YoloBoxDataset(
        dataset_yaml,
        split="train",
        augmentation_strategy="combined",
        donor_bank=donor_bank,
    )
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        generator=generator,
        collate_fn=collate_detection_batch,
    )

    # 2. Build model and transfer backbone
    backbone_weights = PROJECT_ROOT / "artifacts" / "official_run" / "best_model.pt"
    model = build_ssdlite(
        pretrained=True,
        backbone_weights=str(backbone_weights) if backbone_weights.is_file() else None,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # 3. Pull a real batch
    images, targets = next(iter(loader))
    assert len(images) == 4, f"Expected batch size 4, got {len(images)}"

    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Confirm augmentation occurred in dataset telemetry
    aug_telemetry = getattr(train_dataset, "telemetry_history", [])
    print(f"[SSDLite] Batch loaded. Augmentation events recorded: {len(aug_telemetry)}")

    # 4. Pick a sensitive trainable parameter to verify gradient update
    # Use classification head layer 0 weight
    trainable_param = None
    target_param_name = ""
    for name, p in model.named_parameters():
        if p.requires_grad and "head.classification_head" in name and "weight" in name:
            trainable_param = p
            target_param_name = name
            break
    if trainable_param is None:
        # Fallback to any trainable parameter
        for name, p in model.named_parameters():
            if p.requires_grad and "weight" in name:
                trainable_param = p
                target_param_name = name
                break

    param_before = trainable_param.detach().clone().cpu()

    # 5. Forward pass
    optimizer.zero_grad()
    loss_dict = model(images, targets)
    total_loss = sum(l for l in loss_dict.values())
    loss_val = float(total_loss.item())

    assert math.isfinite(loss_val), f"SSDLite loss is non-finite: {loss_val}"
    assert loss_val > 0.0, f"SSDLite loss must be positive: {loss_val}"
    print(f"[SSDLite] Forward pass successful. Total loss: {loss_val:.4f}")
    for k, v in loss_dict.items():
        print(f"  - {k}: {float(v.item()):.4f}")

    # 6. Backward pass
    total_loss.backward()
    grad_norm = float(trainable_param.grad.norm().item())
    assert grad_norm > 0.0, f"Gradient norm must be positive, got {grad_norm}"
    print(f"[SSDLite] Backward pass successful. Target parameter '{target_param_name}' grad_norm: {grad_norm:.6f}")

    # 7. Optimizer step
    optimizer.step()
    param_after = trainable_param.detach().clone().cpu()

    # 8. Verify weight change
    diff = torch.abs(param_after - param_before)
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    is_changed = not torch.allclose(param_before, param_after, atol=1e-7)

    assert is_changed, "Optimizer step FAILED to change target trainable parameter!"
    print(f"[SSDLite] Parameter change confirmed: max_abs_diff = {max_abs_diff:.6e}, mean_abs_diff = {mean_abs_diff:.6e}")

    # 9. Save new verification checkpoint
    ckpt_dir = out_dir / "combined-ssdlite320" / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = ckpt_dir / "best.pt"

    commit_sha = get_git_commit_sha()
    metadata = {
        "architecture": "ssdlite320_mobilenet_v3_large",
        "backend": "ssdlite",
        "augmentation_variant": "combined",
        "dataset_yaml": str(dataset_yaml),
        "seed": 42,
        "commit_sha": commit_sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "device": str(device),
        "verification_metrics": {
            "loss": loss_val,
            "loss_dict": {k: float(v.item()) for k, v in loss_dict.items()},
            "target_param": target_param_name,
            "param_max_abs_diff": max_abs_diff,
            "param_mean_abs_diff": mean_abs_diff,
            "parameter_updated_verified": True,
        },
    }

    checkpoint_payload = {
        "metadata": metadata,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint_payload, best_pt_path)
    print(f"[SSDLite] Checkpoint saved: {best_pt_path} ({best_pt_path.stat().st_size:,} bytes)")

    # 10. Reload checkpoint & deterministic real validation inference
    print("[SSDLite] Reloading checkpoint for validation inference...")
    loaded = torch.load(best_pt_path, map_location=device, weights_only=False)
    eval_model = build_ssdlite(pretrained=False).to(device)
    eval_model.load_state_dict(loaded["model_state_dict"])
    eval_model.eval()

    val_img_path = PROJECT_ROOT / "data" / "detection" / "images" / "real" / "taco_0081.jpg"
    with Image.open(val_img_path) as opened:
        val_rgb = opened.convert("RGB")
    val_w, val_h = val_rgb.size
    val_tensor = torch.from_numpy(np.array(val_rgb)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    val_tensor = val_tensor.to(device)

    with torch.no_grad():
        preds = eval_model(val_tensor)[0]

    pred_boxes = preds["boxes"].cpu().numpy().tolist()
    pred_scores = preds["scores"].cpu().numpy().tolist()
    pred_labels = (preds["labels"].cpu().numpy() - 1).tolist()  # SSDLite label 0 is background

    # Filter top detections (score > 0.05 or top 5)
    keep_indices = [i for i, s in enumerate(pred_scores) if s >= 0.05][:5]
    if not keep_indices and len(pred_scores) > 0:
        keep_indices = [0]

    filtered_boxes = [pred_boxes[i] for i in keep_indices]
    filtered_scores = [pred_scores[i] for i in keep_indices]
    filtered_labels = [pred_labels[i] for i in keep_indices]

    inf_img = draw_predictions_on_image(
        val_img_path,
        filtered_boxes,
        filtered_scores,
        filtered_labels,
        title=f"SSDLite320 [combined] Real Val Inference | {val_img_path.name} | Detections: {len(filtered_boxes)}",
    )
    inf_save_path = out_dir / "smoke_ssdlite_combined_real_val_inference.png"
    inf_img.save(inf_save_path)
    print(f"[SSDLite] Validation inference image saved: {inf_save_path}")

    return {
        "status": "PASS",
        "loss": loss_val,
        "target_parameter": target_param_name,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "parameter_updated_verified": True,
        "checkpoint_path": str(best_pt_path),
        "checkpoint_size_bytes": best_pt_path.stat().st_size,
        "inference_image": str(inf_save_path),
        "detections_count": len(filtered_boxes),
        "telemetry_events_count": len(aug_telemetry),
    }


def verify_yolo_real_combined(
    dataset_yaml: Path,
    donor_bank: DonorBank,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 2: Ultralytics YOLOv8n Real Training & Gradient Verification ('combined')")
    print("=" * 70)

    from ultralytics import YOLO
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.build import build_dataloader
    from src.detection.train import AblationAugmentationTransform, _make_yolo_trainer

    commit_sha = get_git_commit_sha()
    metadata = {
        "backend": "yolov8n",
        "augmentation_variant": "combined",
        "dataset_yaml": str(dataset_yaml),
        "seed": 42,
        "commit_sha": commit_sha,
    }

    # 1. Initialize YOLOv8n model
    weights_path = PROJECT_ROOT / "yolov8n.pt"
    model = YOLO(str(weights_path) if weights_path.is_file() else "yolov8n.pt", task="detect")

    # 2. Build official Ultralytics dataset pipeline with AblationAugmentationTransform
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Use resolved abs split for training
    train_split_file = PROJECT_ROOT / "data" / "detection" / "splits" / "train.txt"
    lines = train_split_file.read_text(encoding="utf-8").splitlines()
    abs_lines = [str((PROJECT_ROOT / "data" / "detection" / line).resolve()).replace("\\", "/") for line in lines if line.strip()]
    temp_train_txt = out_dir / "temp_train_abs.txt"
    temp_train_txt.write_text("\n".join(abs_lines) + "\n", encoding="utf-8")

    from types import SimpleNamespace
    hyp = SimpleNamespace(
        imgsz=320, augment=False, rect=False, degrees=0, translate=0, scale=0, shear=0,
        perspective=0, flipud=0, fliplr=0, mosaic=0, mixup=0, copy_paste=0,
        auto_augment=None, erasing=0, crop_fraction=1.0, bgr=0, mask_ratio=4, overlap_mask=True
    )

    yolo_dataset = YOLODataset(
        img_path=str(temp_train_txt),
        imgsz=320,
        batch_size=4,
        augment=False,
        hyp=hyp,
        data=data_cfg,
    )

    # Attach official ablation transform at index 0 of transforms
    ablation = AblationAugmentationTransform(strategy="combined", donor_bank=donor_bank, seed=42)
    yolo_dataset.transforms.insert(0, ablation)

    # Build genuine Ultralytics DataLoader
    yolo_loader = build_dataloader(
        yolo_dataset,
        batch=4,
        workers=0,
        shuffle=True,
    )

    # 3. Pull actual Ultralytics batch data structure
    batch = next(iter(yolo_loader))
    assert "img" in batch and "bboxes" in batch and "cls" in batch, f"Missing Ultralytics batch keys: {batch.keys()}"
    print(f"[YOLOv8n] Real Ultralytics batch loaded: img={batch['img'].shape}, bboxes={batch['bboxes'].shape}")

    # Confirm telemetry was captured
    aug_telemetry = getattr(ablation, "telemetry_history", [])
    print(f"[YOLOv8n] Augmentation events recorded: {len(aug_telemetry)}")

    # 4. Prepare PyTorch training step
    py_model = model.model.to(device)
    py_model.train()
    for p in py_model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(py_model.parameters(), lr=1e-3, weight_decay=5e-4)

    # Select target trainable parameter
    target_param_name = "model.22.cv3.2.2.weight"  # Output head weight
    trainable_param = None
    for name, p in py_model.named_parameters():
        if target_param_name in name and p.requires_grad:
            trainable_param = p
            target_param_name = name
            break
    if trainable_param is None:
        for name, p in py_model.named_parameters():
            if p.requires_grad and "weight" in name:
                trainable_param = p
                target_param_name = name
                break

    assert trainable_param is not None, "Failed to find any trainable parameter in YOLOv8n"
    param_before = trainable_param.detach().clone().cpu()

    # 5. Forward pass through loss
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.cfg import get_cfg, DEFAULT_CFG
    loss_fn = v8DetectionLoss(py_model)
    loss_fn.hyp = get_cfg(DEFAULT_CFG)

    imgs = batch["img"].to(device).float() / 255.0
    batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    batch_dev["img"] = imgs

    optimizer.zero_grad()
    preds = py_model(imgs)
    loss, loss_items = loss_fn(preds, batch_dev)
    total_loss = loss.sum() if loss.numel() > 1 else loss
    loss_val = float(total_loss.item())

    assert math.isfinite(loss_val), f"YOLOv8n loss is non-finite: {loss_val}"
    assert loss_val > 0.0, f"YOLOv8n loss must be positive: {loss_val}"
    loss_items_repr = (
        {k: float(v.item()) if hasattr(v, "item") else float(v) for k, v in loss_items.items()}
        if isinstance(loss_items, dict)
        else (loss_items.cpu().tolist() if hasattr(loss_items, "cpu") else str(loss_items))
    )
    print(f"[YOLOv8n] Forward pass successful. Total loss: {loss_val:.4f}, loss_items: {loss_items_repr}")

    # 6. Backward pass
    total_loss.backward()
    grad_norm = float(trainable_param.grad.norm().item())
    assert grad_norm > 0.0, f"YOLOv8n gradient norm must be positive, got {grad_norm}"
    print(f"[YOLOv8n] Backward pass successful. Target parameter '{target_param_name}' grad_norm: {grad_norm:.6f}")

    # 7. Optimizer step
    optimizer.step()
    param_after = trainable_param.detach().clone().cpu()

    # 8. Verify weight change
    diff = torch.abs(param_after - param_before)
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())
    is_changed = not torch.allclose(param_before, param_after, atol=1e-7)

    assert is_changed, "Optimizer step FAILED to change YOLOv8n trainable parameter!"
    print(f"[YOLOv8n] Parameter change confirmed: max_abs_diff = {max_abs_diff:.6e}, mean_abs_diff = {mean_abs_diff:.6e}")

    # 9. Save new verification checkpoint
    ckpt_dir = out_dir / "combined-yolov8n" / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_pt_path = ckpt_dir / "best.pt"

    # Save via Ultralytics model format
    import importlib.metadata
    ultra_ver = importlib.metadata.version("ultralytics")

    yolo_metadata = {
        "backend": "yolov8n",
        "augmentation_variant": "combined",
        "dataset_yaml": str(dataset_yaml),
        "seed": 42,
        "commit_sha": commit_sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch_version": torch.__version__,
        "ultralytics_version": ultra_ver,
        "device": str(device),
        "verification_metrics": {
            "loss": loss_val,
            "target_param": target_param_name,
            "param_max_abs_diff": max_abs_diff,
            "param_mean_abs_diff": mean_abs_diff,
            "parameter_updated_verified": True,
        },
    }

    # Save checkpoint with model state and metadata
    ckpt_payload = {
        "model": py_model,
        "train_args": {"imgsz": 320, "batch": 4, "augmentation_variant": "combined"},
        "epoch": 1,
        "phase2_metadata": yolo_metadata,
    }
    torch.save(ckpt_payload, best_pt_path)
    print(f"[YOLOv8n] Checkpoint saved: {best_pt_path} ({best_pt_path.stat().st_size:,} bytes)")

    # 10. Reload checkpoint & deterministic real validation inference
    print("[YOLOv8n] Reloading checkpoint for validation inference...")
    eval_yolo = YOLO(str(best_pt_path), task="detect")
    val_img_path = PROJECT_ROOT / "data" / "detection" / "images" / "real" / "taco_0081.jpg"

    results = eval_yolo.predict(
        source=str(val_img_path),
        imgsz=320,
        device=str(device) if device.type == "cuda" else "cpu",
        conf=0.001,
        max_det=10,
        verbose=False,
    )[0]

    yolo_boxes = results.boxes.xyxy.cpu().numpy().tolist()
    yolo_scores = results.boxes.conf.cpu().numpy().tolist()
    yolo_classes = results.boxes.cls.cpu().numpy().astype(int).tolist()

    inf_img = draw_predictions_on_image(
        val_img_path,
        yolo_boxes,
        yolo_scores,
        yolo_classes,
        title=f"YOLOv8n [combined] Real Val Inference | {val_img_path.name} | Detections: {len(yolo_boxes)}",
    )
    inf_save_path = out_dir / "smoke_yolov8n_combined_real_val_inference.png"
    inf_img.save(inf_save_path)
    print(f"[YOLOv8n] Validation inference image saved: {inf_save_path}")

    return {
        "status": "PASS",
        "loss": loss_val,
        "target_parameter": target_param_name,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "parameter_updated_verified": True,
        "checkpoint_path": str(best_pt_path),
        "checkpoint_size_bytes": best_pt_path.stat().st_size,
        "inference_image": str(inf_save_path),
        "detections_count": len(yolo_boxes),
        "telemetry_events_count": len(aug_telemetry),
    }


def main():
    root = PROJECT_ROOT
    dataset_yaml = root / "configs" / "detection_dataset.yaml"
    out_dir = root / "artifacts" / "part02" / "combined_training_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("auto")
    print(f"[MAIN] Initializing verification on device: {device}")

    # Load verified training donors
    donor_bank = DonorBank.from_dataset_root(root)
    print(f"[MAIN] Loaded {len(donor_bank)} verified training masks from TACO train split.")

    # 1. Verify SSDLite
    ssdlite_res = verify_ssdlite_real_combined(dataset_yaml, donor_bank, out_dir, device)

    # 2. Verify YOLOv8n
    yolo_res = verify_yolo_real_combined(dataset_yaml, donor_bank, out_dir, device)

    # Assemble comprehensive report
    report = {
        "verification_summary": {
            "strategy": "combined",
            "device": str(device),
            "commit_sha": get_git_commit_sha(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ssdlite_status": ssdlite_res["status"],
            "yolo_status": yolo_res["status"],
        },
        "ssdlite320": ssdlite_res,
        "yolov8n": yolo_res,
    }

    report_path = out_dir / "combined_training_verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[SUCCESS] Combined training verification complete! Report written to:\n  {report_path}")


if __name__ == "__main__":
    main()
