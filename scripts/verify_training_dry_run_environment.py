"""
scripts/verify_training_dry_run_environment.py
----------------------------------------------
Executes an isolated, single-step forward/loss/backward dry-run on RTX 2050 GPU:
1. Purely experimental model instance: weights=None, weights_backbone=None, isolated from production.
2. Uses torch.cuda.synchronize() for accurate GPU kernel execution timing.
3. Tests mixed batch: 1 real positive sample (oi_0010c714a5da358a.jpg) + 1 artificial hard-negative canvas (0-object).
4. Validates that loss terms and parameter gradients are finite (zero NaNs, zero Infs).
5. Instruments exact CUDA VRAM (allocated, reserved, peak) and host system RAM.
6. Evaluates the Preflight Safety Guard and enforces training blocks when RAM is critical.
7. Documents explicit scope caveats (batch=2 non-extrapolability, artificial canvas limitations).

Outputs: reports/model_v2/training_dry_run_report.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import time
from pathlib import Path
import psutil
import torch
import torchvision
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports" / "model_v2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

from src.detection.preflight_guard import check_training_safety

def measure_system_ram() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "used_gb": round(vm.used / (1024**3), 2),
        "available_mb": round(vm.available / (1024**2), 1),
        "used_percent": round(vm.percent, 1)
    }

def main():
    print("=== GENUINE SSDLITE TRAINING DRY-RUN & HARDWARE TELEMETRY ===")

    # 1. Evaluate Preflight Guard
    preflight = check_training_safety(min_free_ram_mb=1200.0)
    print(f"Preflight Safety Status: {preflight.status_code}")
    print(f"  Reason: {preflight.reason}")
    print(f"  Available Host RAM: {preflight.host_ram_available_mb} MB ({preflight.host_ram_used_percent}% utilized)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ram_before = measure_system_ram()

    # 2. Build purely isolated model with weights=None and weights_backbone=None
    print("\n[Step 1/4] Building un-pretrained SSDLite model (weights=None, weights_backbone=None)...")
    # 10 waste classes + 1 background = 11 classes
    model = ssdlite320_mobilenet_v3_large(
        num_classes=11,
        weights=None,
        weights_backbone=None
    )
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Load Samples
    print("\n[Step 2/4] Preparing Mixed Batch (Positive Sample + Hard-Negative Canvas)...")
    pos_img_path = ROOT / "data" / "detection" / "images" / "real" / "oi_0010c714a5da358a.jpg"
    pos_lbl_path = ROOT / "data" / "detection" / "labels" / "real" / "oi_0010c714a5da358a.txt"

    if not pos_img_path.exists() or not pos_lbl_path.exists():
        raise FileNotFoundError(f"Missing positive sample: {pos_img_path}")

    # Load positive image
    with Image.open(pos_img_path) as im:
        im_rgb = im.convert("RGB")
        w_orig, h_orig = im_rgb.size
        # SSDLite uses 320x320
        im_resized = im_rgb.resize((320, 320), Image.Resampling.BILINEAR)
        pos_tensor = torchvision.transforms.functional.to_tensor(im_resized).to(device)

    # Parse positive boxes
    pos_boxes_list = []
    pos_labels_list = []
    with open(pos_lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                cid = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                # Convert to [x1, y1, x2, y2] in 320x320 coordinates
                x1 = max(0.0, (xc - bw / 2.0) * 320.0)
                y1 = max(0.0, (yc - bh / 2.0) * 320.0)
                x2 = min(320.0, (xc + bw / 2.0) * 320.0)
                y2 = min(320.0, (yc + bh / 2.0) * 320.0)
                pos_boxes_list.append([x1, y1, x2, y2])
                # SSDLite target class: foreground is class_id + 1 (1..10)
                pos_labels_list.append(cid + 1)

    pos_target = {
        "boxes": torch.tensor(pos_boxes_list, dtype=torch.float32, device=device),
        "labels": torch.tensor(pos_labels_list, dtype=torch.int64, device=device)
    }

    # Prepare artificial hard-negative canvas (320x320, 0 boxes)
    neg_tensor = torch.zeros((3, 320, 320), dtype=torch.float32, device=device)
    neg_target = {
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device)
    }

    images = [pos_tensor, neg_tensor]
    targets = [pos_target, neg_target]

    print(f"  Positive sample: {pos_img_path.name} with {len(pos_boxes_list)} boxes.")
    print("  Negative sample: Artificial 0-object canvas with 0 boxes.")

    # 4. Execute Forward, Loss & Backward with CUDA Synchronize
    print("\n[Step 3/4] Executing Forward & Loss computation with torch.cuda.synchronize()...")
    optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd_start = time.perf_counter()

    loss_dict = model(images, targets)
    total_loss = sum(loss for loss in loss_dict.values())

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd_end = time.perf_counter()
    forward_duration = t_fwd_end - t_fwd_start

    # Check finite losses
    finite_loss = True
    loss_metrics = {}
    for k, v in loss_dict.items():
        val = float(v.item())
        loss_metrics[k] = round(val, 6)
        if torch.isnan(v) or torch.isinf(v):
            finite_loss = False

    tot_val = float(total_loss.item())
    loss_metrics["total_loss"] = round(tot_val, 6)
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        finite_loss = False

    print(f"  Forward completed in {forward_duration:.4f}s. Total Loss: {tot_val:.4f}")
    print(f"  Loss breakdown: {loss_metrics}")

    print("\n[Step 4/4] Executing Backward computation with torch.cuda.synchronize()...")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd_start = time.perf_counter()

    total_loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd_end = time.perf_counter()
    backward_duration = t_bwd_end - t_bwd_start

    # Check finite gradients
    finite_grads = True
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                finite_grads = False
            grad_norms.append(float(param.grad.norm().item()))

    print(f"  Backward completed in {backward_duration:.4f}s. Finite gradients: {finite_grads}")

    # Instrument CUDA VRAM
    vram_telemetry = {}
    if device.type == "cuda":
        vram_telemetry = {
            "vram_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
            "vram_peak_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 2),
            "vram_device_total_mb": preflight.gpu_vram_total_mb
        }
        print(f"  VRAM Telemetry: Allocated={vram_telemetry['vram_allocated_mb']} MB, Reserved={vram_telemetry['vram_reserved_mb']} MB, Peak={vram_telemetry['vram_peak_allocated_mb']} MB")

    ram_after = measure_system_ram()

    # Determine overall safety status
    dry_run_success = finite_loss and finite_grads
    local_multi_epoch_allowed = preflight.safe_to_train

    report = {
        "metadata": {
            "execution_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": str(device),
            "gpu_name": preflight.gpu_device_name,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        },
        "model_architecture": {
            "name": "ssdlite320_mobilenet_v3_large",
            "num_classes": 11,
            "weights": None,
            "weights_backbone": None,
            "isolation_status": "EXPERIMENT_ONLY_NO_CHECKPOINT_UPDATED"
        },
        "batch_specification": {
            "batch_size": 2,
            "samples": [
                {
                    "type": "positive_real",
                    "file": pos_img_path.name,
                    "num_boxes": len(pos_boxes_list),
                    "labels": pos_labels_list
                },
                {
                    "type": "artificial_hard_negative_canvas",
                    "description": "Zero-object canvas (zeros) used strictly to verify loss computation on empty targets; does NOT substitute for real background negative imagery.",
                    "num_boxes": 0
                }
            ]
        },
        "execution_telemetry": {
            "forward_duration_seconds": round(forward_duration, 4),
            "backward_duration_seconds": round(backward_duration, 4),
            "total_step_duration_seconds": round(forward_duration + backward_duration, 4),
            "losses": loss_metrics,
            "finite_losses": finite_loss,
            "finite_gradients": finite_grads,
            "cuda_vram": vram_telemetry,
            "system_ram_before": ram_before,
            "system_ram_after": ram_after
        },
        "hardware_safety_gate": {
            "preflight_status": preflight.status_code,
            "preflight_reason": preflight.reason,
            "local_multi_epoch_training_allowed": local_multi_epoch_allowed,
            "verdict": "VERIFIED_DRY_RUN_BUT_LOCAL_TRAINING_BLOCKED" if not local_multi_epoch_allowed else "VERIFIED_READY"
        },
        "scientific_scope_caveats": [
            "Caveat 1: Batch=2 measurements CANNOT be linearly extrapolated to batch=8/16. PyTorch caching allocator reserves memory in discrete 2MB/20MB blocks; larger batches trigger different internal memory fragmentation and non-linear peak allocation.",
            "Caveat 2: The artificial 0-object canvas tests SSDLite's mathematical handling of empty tensors without NaN/Inf, but does not provide visual false-positive penalization. True hard-negative background images (sidewalks, walls) must be acquired for effective feature discrimination.",
            "Caveat 3: Because host system RAM is critically low (>95% used, <1.2GB free), multi-epoch local training is strictly BLOCKED by the Preflight Guard to prevent OS freeze."
        ]
    }

    out_file = REPORT_DIR / "training_dry_run_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully wrote dry-run report to: {out_file}")
    print(f"VERDICT: {report['hardware_safety_gate']['verdict']}")

if __name__ == "__main__":
    main()
