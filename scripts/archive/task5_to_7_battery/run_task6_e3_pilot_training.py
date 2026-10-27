# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Chạy thử nghiệm, tái lập đánh giá hoặc so sánh mô hình cho `run_task6_e3_pilot_training`.
"""
scripts/run_task6_e3_pilot_training.py
--------------------------------------
Sprint AI-UPGRADE-01 — Task 6 (Task 3):
Executes the single authorized YOLOv8n Pilot V3 fine-tuning run (E3):
- Base checkpoint: artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt (verified immutable via SHA-256)
- Dataset: configs/detection_dataset_real_v3_pilot.yaml (1,509 train / 328 val / 299 test)
- Hyperparameters: imgsz=512, epochs=25, patience=5, batch=4 (OOM fallback to batch=2),
  optimizer=AdamW, lr0=0.001, amp=True, cache=False, workers=2 (Windows MP fallback to 0), seed=42
- Output directory: artifacts/detection_v3_pilot/E3_yolov8n_pilot/
- Captures per-epoch telemetry (losses, mAP50, mAP50-95, precision, recall, LR, GPU VRAM) and raw log.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
E1_WEIGHTS = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
TRAIN_CONFIG_P = ROOT / "configs" / "training_yolov8n_v3_pilot.yaml"
DATASET_YAML_P = ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml"
PROJECT_DIR = ROOT / "artifacts" / "detection_v3_pilot"
RUN_NAME = "E3_yolov8n_pilot"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_e3_training() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("ABORT: torch.cuda.is_available() is False. Refusing to run CPU training.")

    gpu_name = torch.cuda.get_device_name(0)
    if "2050" not in gpu_name:
        raise RuntimeError(f"ABORT: Expected RTX 2050 GPU, got '{gpu_name}'.")

    assert E1_WEIGHTS.exists(), f"Missing E1 checkpoint: {E1_WEIGHTS}"
    e1_sha_before = sha256_file(E1_WEIGHTS)

    cfg = yaml.safe_load(TRAIN_CONFIG_P.read_text(encoding="utf-8"))
    epoch_records: List[Dict[str, Any]] = []
    log_lines: List[str] = []

    def log_msg(msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log_msg(f"Starting Task 6 E3 YOLOv8n Pilot Fine-Tuning on GPU: {gpu_name}")
    log_msg(f"E1 Base Checkpoint: {E1_WEIGHTS} (SHA-256: {e1_sha_before})")
    log_msg(f"Dataset YAML: {DATASET_YAML_P}")

    def on_fit_epoch_end(trainer: Any) -> None:
        ep = int(trainer.epoch) + 1
        tloss = trainer.tloss
        if isinstance(tloss, dict):
            losses = [float(v.detach().cpu().item() if hasattr(v, "detach") else v) for v in tloss.values()]
        elif hasattr(tloss, "detach"):
            losses = [float(x) for x in tloss.detach().cpu().flatten().tolist()]
        elif isinstance(tloss, (list, tuple)):
            losses = [float(x) for x in tloss]
        else:
            losses = [0.0, 0.0, 0.0]

        for val in losses:
            if math.isnan(val) or math.isinf(val):
                raise RuntimeError(f"ABORT: NaN or Inf loss detected at epoch {ep}: {losses}")

        metrics = trainer.metrics or {}
        mp = float(metrics.get("metrics/precision(B)", 0.0))
        mr = float(metrics.get("metrics/recall(B)", 0.0))
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
        lr_pg0 = float(trainer.lr.get("lr/pg0", 0.0)) if isinstance(trainer.lr, dict) else 0.0
        vram_alloc_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 1)
        vram_max_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 1)
        vram_reserved_mb = round(torch.cuda.memory_reserved(0) / (1024 * 1024), 1)

        rec = {
            "epoch": ep,
            "total_epochs_configured": int(trainer.epochs),
            "box_loss": round(losses[0], 5) if len(losses) > 0 else 0.0,
            "cls_loss": round(losses[1], 5) if len(losses) > 1 else 0.0,
            "dfl_loss": round(losses[2], 5) if len(losses) > 2 else 0.0,
            "val_precision": round(mp, 5),
            "val_recall": round(mr, 5),
            "val_mAP50": round(map50, 5),
            "val_mAP50_95": round(map50_95, 5),
            "lr_pg0": lr_pg0,
            "gpu_vram_allocated_mb": vram_alloc_mb,
            "gpu_vram_max_allocated_mb": vram_max_mb,
            "gpu_vram_reserved_mb": vram_reserved_mb,
        }
        epoch_records.append(rec)
        log_msg(
            f"Epoch {ep:02d}/{trainer.epochs:02d} | "
            f"box={rec['box_loss']:.4f} cls={rec['cls_loss']:.4f} dfl={rec['dfl_loss']:.4f} | "
            f"P={mp:.4f} R={mr:.4f} mAP50={map50:.4f} mAP50-95={map50_95:.4f} | "
            f"lr={lr_pg0:.6f} | VRAM_max={vram_max_mb:.1f}MB (reserved={vram_reserved_mb:.1f}MB)"
        )

    batch_to_use = int(cfg.get("batch", 4))
    workers_to_use = int(cfg.get("workers", 2))
    fallback_notes: List[str] = []

    train_kwargs = {
        "data": str(DATASET_YAML_P),
        "epochs": int(cfg.get("epochs", 25)),
        "patience": int(cfg.get("patience", 5)),
        "batch": batch_to_use,
        "imgsz": int(cfg.get("imgsz", 512)),
        "device": 0,
        "workers": workers_to_use,
        "amp": bool(cfg.get("amp", True)),
        "cache": False,
        "optimizer": str(cfg.get("optimizer", "AdamW")),
        "lr0": float(cfg.get("lr0", 0.001)),
        "lrf": float(cfg.get("lrf", 0.01)),
        "momentum": float(cfg.get("momentum", 0.937)),
        "weight_decay": float(cfg.get("weight_decay", 0.0005)),
        "warmup_epochs": float(cfg.get("warmup_epochs", 3.0)),
        "cos_lr": bool(cfg.get("cos_lr", True)),
        "hsv_h": float(cfg.get("hsv_h", 0.015)),
        "hsv_s": float(cfg.get("hsv_s", 0.5)),
        "hsv_v": float(cfg.get("hsv_v", 0.3)),
        "degrees": float(cfg.get("degrees", 10.0)),
        "translate": float(cfg.get("translate", 0.1)),
        "scale": float(cfg.get("scale", 0.3)),
        "shear": float(cfg.get("shear", 0.0)),
        "perspective": float(cfg.get("perspective", 0.0)),
        "flipud": float(cfg.get("flipud", 0.2)),
        "fliplr": float(cfg.get("fliplr", 0.5)),
        "mosaic": float(cfg.get("mosaic", 0.5)),
        "mixup": float(cfg.get("mixup", 0.0)),
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": True,
        "pretrained": True,
        "seed": int(cfg.get("seed", 42)),
        "deterministic": True,
        "save": True,
        "val": True,
        "plots": True,
    }

    start_t = time.time()
    try:
        model = YOLO(str(E1_WEIGHTS))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(**train_kwargs)
    except RuntimeError as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str and batch_to_use == 4:
            note = "CUDA Out of Memory at batch=4; cleared CUDA cache and fell back to batch=2 per Task 3.3."
            log_msg(note)
            fallback_notes.append(note)
            gc.collect()
            torch.cuda.empty_cache()
            epoch_records.clear()
            train_kwargs["batch"] = 2
            model = YOLO(str(E1_WEIGHTS))
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.train(**train_kwargs)
        elif ("multiprocessing" in err_str or "broken pipe" in err_str or "paging file" in err_str) and workers_to_use > 0:
            note = f"Windows DataLoader multiprocessing error with workers={workers_to_use}; fell back to workers=0 per Task 3.3."
            log_msg(note)
            fallback_notes.append(note)
            gc.collect()
            torch.cuda.empty_cache()
            epoch_records.clear()
            train_kwargs["workers"] = 0
            model = YOLO(str(E1_WEIGHTS))
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            model.train(**train_kwargs)
        else:
            raise

    elapsed_sec = round(time.time() - start_t, 2)
    out_dir = PROJECT_DIR / RUN_NAME
    best_pt = out_dir / "weights" / "best.pt"
    last_pt = out_dir / "weights" / "last.pt"
    results_csv = out_dir / "results.csv"
    args_yaml = out_dir / "args.yaml"

    assert best_pt.exists(), f"Missing {best_pt}"
    assert last_pt.exists(), f"Missing {last_pt}"
    assert results_csv.exists(), f"Missing {results_csv}"
    assert args_yaml.exists(), f"Missing {args_yaml}"

    # Verify E1 checkpoint was NOT modified
    e1_sha_after = sha256_file(E1_WEIGHTS)
    assert e1_sha_before == e1_sha_after, "CRITICAL: E1 checkpoint SHA-256 changed!"

    # Clean up any temporary Ultralytics .cache files in data/detection/labels/
    for cache_f in [
        ROOT / "data" / "detection" / "labels" / "real.cache",
        ROOT / "data" / "detection_upgrade_v3_pilot" / "labels" / "roboflow_battery.cache",
    ]:
        if cache_f.exists():
            cache_f.unlink()

    # Determine best epoch and early stopping status from epoch_records / results.csv
    best_ep_rec = max(
        epoch_records,
        key=lambda r: 0.1 * r["val_mAP50"] + 0.9 * r["val_mAP50_95"],
    ) if epoch_records else {}
    completed_epochs = len(epoch_records)
    early_stopped = completed_epochs < int(train_kwargs["epochs"])
    early_stop_reason = (
        f"EarlyStopping triggered at epoch {completed_epochs} (patience={train_kwargs['patience']} epochs without fitness improvement after best epoch {best_ep_rec.get('epoch')})."
        if early_stopped
        else f"Completed all {completed_epochs} configured epochs (best fitness at epoch {best_ep_rec.get('epoch')})."
    )

    log_msg(f"Training finished in {elapsed_sec}s. Completed epochs: {completed_epochs}. {early_stop_reason}")
    log_msg(f"Best epoch: {best_ep_rec.get('epoch')} | Best val mAP50={best_ep_rec.get('val_mAP50')} | Best val mAP50-95={best_ep_rec.get('val_mAP50_95')}")
    log_msg(f"best.pt SHA-256: {sha256_file(best_pt)}")
    log_msg(f"last.pt SHA-256: {sha256_file(last_pt)}")
    log_msg(f"E1 best.pt SHA-256 preserved: {e1_sha_after == e1_sha_before} ({e1_sha_after})")

    (out_dir / "training_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    telemetry = {
        "experiment_id": "E3_yolov8n_pilot",
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_sec,
        "gpu_device": gpu_name,
        "e1_base_checkpoint": str(E1_WEIGHTS.relative_to(ROOT)),
        "e1_sha256_before": e1_sha_before,
        "e1_sha256_after": e1_sha_after,
        "e1_checkpoint_preserved": e1_sha_before == e1_sha_after,
        "e3_best_pt": str(best_pt.relative_to(ROOT)),
        "e3_best_pt_sha256": sha256_file(best_pt),
        "e3_last_pt": str(last_pt.relative_to(ROOT)),
        "e3_last_pt_sha256": sha256_file(last_pt),
        "configured_epochs": int(train_kwargs["epochs"]),
        "completed_epochs": completed_epochs,
        "patience": int(train_kwargs["patience"]),
        "batch_size_used": int(train_kwargs["batch"]),
        "workers_used": int(train_kwargs["workers"]),
        "fallback_notes": fallback_notes,
        "early_stopped": early_stopped,
        "early_stopping_reason": early_stop_reason,
        "best_epoch_summary": best_ep_rec,
        "epochs": epoch_records,
    }
    (out_dir / "epoch_telemetry.json").write_text(json.dumps(telemetry, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_e3_training())
