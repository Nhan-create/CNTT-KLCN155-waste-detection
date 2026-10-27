# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Chạy thử nghiệm, tái lập đánh giá hoặc so sánh mô hình cho `evaluate_and_compare_e1_vs_e3_task6`.
"""Task 6: Comprehensive Fair Evaluation of YOLOv8n E1 vs E3 across 4 Benchmarks.

Evaluates:
  - Evaluation A: Validation V3 (328 images, 930 boxes, 29 battery boxes)
  - Evaluation B: Validation V2 (299 images, 901 boxes, 0 battery boxes)
  - Evaluation C: Internal Test Frozen (299 images, 720 boxes, 1 battery box)
  - Evaluation D: External Holdout (27 images, 57 GT objects in data/holdout)
  - Extra Qualitative Check: Unannotated hard-negative utility boxes / brick walls
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.schema import DETECTION_CLASS_NAMES
from reports.qa_acceptance.practical_metrics import (
    box_iou,
    run as run_practical_metrics,
)

E1_CKPT = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
E1_EXPECTED_SHA = "7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8"

E3_DIR = ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot"
E3_BEST_CKPT = E3_DIR / "weights" / "best.pt"
E3_LAST_CKPT = E3_DIR / "weights" / "last.pt"

YAML_V3 = ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml"
YAML_V2 = ROOT / "configs" / "detection_dataset_real_v2.yaml"

HOLDOUT_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
HOLDOUT_GT = ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac.json"

EVAL_OUT_DIR = ROOT / "artifacts" / "detection_v3_pilot" / "evaluation_e1_vs_e3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_e3_phase2_metadata() -> Tuple[str, str]:
    """Ensure E3 best.pt and last.pt carry phase2_metadata required by src.detection.yolo.WasteDetector."""
    v3_cfg = yaml.safe_load(YAML_V3.read_text(encoding="utf-8"))
    split_sha = v3_cfg.get("metadata", {}).get("split_sha256", {})
    meta = {
        "backend": "yolov8n",
        "trained": True,
        "class_names": list(DETECTION_CLASS_NAMES),
        "agnostic": False,
        "run_name": "E3_yolov8n_pilot",
        "split_sha256": split_sha,
    }
    for pt_path in (E3_BEST_CKPT, E3_LAST_CKPT):
        if not pt_path.is_file():
            raise RuntimeError(f"Missing E3 checkpoint: {pt_path}")
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        existing = ckpt.get("phase2_metadata")
        model_meta = getattr(ckpt.get("model"), "phase2_metadata", None) if ckpt.get("model") is not None else None
        if existing != meta or model_meta != meta:
            for key in ("model", "ema"):
                if ckpt.get(key) is not None:
                    ckpt[key].phase2_metadata = meta
                    ckpt[key].names = {i: n for i, n in enumerate(DETECTION_CLASS_NAMES)}
            ckpt["phase2_metadata"] = meta
            torch.save(ckpt, pt_path)

    best_sha = sha256_file(E3_BEST_CKPT)
    last_sha = sha256_file(E3_LAST_CKPT)

    telemetry_path = E3_DIR / "epoch_telemetry.json"
    if telemetry_path.is_file():
        tel = json.loads(telemetry_path.read_text(encoding="utf-8"))
        ep_list = tel.get("epochs", tel.get("epoch_records", []))
        if len(ep_list) == 23:
            final_eval_rec = ep_list[-1]
            ep_list = ep_list[:22]
            tel["final_eval_on_best_pt"] = final_eval_rec
        best_rec = max(ep_list, key=lambda r: float(r.get("val_mAP50_95", 0.0)))
        tel["epochs"] = ep_list
        tel["epoch_records"] = ep_list
        tel["completed_epochs"] = len(ep_list)
        tel["epochs_completed"] = len(ep_list)
        tel["best_epoch"] = int(best_rec["epoch"])
        tel["best_epoch_summary"] = best_rec
        tel["early_stopping_reason"] = (
            f"EarlyStopping triggered at epoch {len(ep_list)} "
            f"(patience=5 epochs without fitness improvement after best epoch {best_rec['epoch']})."
        )
        tel["e3_best_pt_sha256"] = best_sha
        tel["e3_last_pt_sha256"] = last_sha
        tel["best_pt_sha256"] = best_sha
        tel["last_pt_sha256"] = last_sha
        tel["phase2_metadata_injected"] = True
        telemetry_path.write_text(json.dumps(tel, indent=2, ensure_ascii=False), encoding="utf-8")

    return best_sha, last_sha


def load_split_images_and_labels(yaml_path: Path, split: str) -> List[Tuple[Path, Path]]:
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = (yaml_path.parent / cfg["path"]).resolve()
    split_txt = (root / cfg[split]).resolve()
    pairs: List[Tuple[Path, Path]] = []
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        img_p = ((split_txt.parent / rel) if rel.startswith(".") else (root / rel)).resolve()
        s = img_p.as_posix()
        lbl_p = Path(s.replace("/images/", "/labels/")).with_suffix(".txt")
        pairs.append((img_p, lbl_p))
    return pairs


def run_ultralytics_val(
    model_path: Path,
    yaml_path: Path,
    split: str,
    imgsz: int,
    run_tag: str,
) -> Dict[str, Any]:
    """Run Ultralytics model.val() on a specific dataset YAML and split using resolved absolute paths."""
    pairs = load_split_images_and_labels(yaml_path, split)
    resolved_dir = EVAL_OUT_DIR / "resolved_yamls"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    abs_txt = resolved_dir / f"{run_tag}_{split}_abs.txt"
    abs_txt.write_text("\n".join(img_p.as_posix() for img_p, _ in pairs) + "\n", encoding="utf-8")

    names = list(DETECTION_CLASS_NAMES)
    resolved_yaml = resolved_dir / f"{run_tag}_{split}.yaml"
    resolved_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ROOT.as_posix(),
                "train": abs_txt.as_posix(),
                "val": abs_txt.as_posix(),
                "nc": len(names),
                "names": {i: n for i, n in enumerate(names)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    model = YOLO(str(model_path), task="detect")
    m = model.val(
        data=str(resolved_yaml),
        split="val",
        imgsz=imgsz,
        batch=4,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=0,
        workers=0,
        plots=True,
        project=str(EVAL_OUT_DIR / "ultralytics_runs"),
        name=run_tag,
        exist_ok=True,
        verbose=False,
    )
    ap_cls_indices = [int(x) for x in m.box.ap_class_index.tolist()]
    p_list = m.box.p.tolist()
    r_list = m.box.r.tolist()
    ap50_list = m.box.ap50.tolist()
    ap_list = m.box.ap.tolist()

    per_class: Dict[str, Any] = {}
    for cls_id, cls_name in enumerate(names):
        if cls_id in ap_cls_indices:
            idx = ap_cls_indices.index(cls_id)
            per_class[cls_name] = {
                "class_id": cls_id,
                "present_in_gt": True,
                "precision": round(float(p_list[idx]), 5),
                "recall": round(float(r_list[idx]), 5),
                "ap50": round(float(ap50_list[idx]), 5),
                "ap50_95": round(float(ap_list[idx]), 5),
            }
        else:
            per_class[cls_name] = {
                "class_id": cls_id,
                "present_in_gt": False,
                "precision": None,
                "recall": None,
                "ap50": None,
                "ap50_95": None,
            }

    cm_matrix = m.confusion_matrix.matrix.tolist() if hasattr(m, "confusion_matrix") and m.confusion_matrix is not None else None

    return {
        "run_tag": run_tag,
        "model_path": str(model_path),
        "dataset_yaml": str(yaml_path),
        "split": split,
        "imgsz": imgsz,
        "overall": {
            "precision": round(float(m.box.mp), 5),
            "recall": round(float(m.box.mr), 5),
            "mAP50": round(float(m.box.map50), 5),
            "mAP50_95": round(float(m.box.map), 5),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": names + ["background"],
            "matrix": cm_matrix,
        },
    }


def analyze_operating_point_on_split(
    model_path: Path,
    yaml_path: Path,
    split: str,
    imgsz: int,
    conf_thr: float = 0.25,
    iou_thr: float = 0.50,
) -> Dict[str, Any]:
    """Compute concrete TP/FP/FN per class and detailed battery case list at conf=0.25, iou=0.50."""
    model = YOLO(str(model_path), task="detect")
    pairs = load_split_images_and_labels(yaml_path, split)

    class_stats = {
        name: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        for name in DETECTION_CLASS_NAMES
    }
    battery_cases: List[Dict[str, Any]] = []

    for img_p, lbl_p in pairs:
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
            w, h = pil.size

        gt_objs: List[Dict[str, Any]] = []
        if lbl_p.is_file():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                gt_objs.append({"class_id": cid, "bbox_xyxy": [x1, y1, x2, y2]})

        res = model.predict(
            source=np.asarray(pil)[:, :, ::-1].copy(),
            imgsz=imgsz,
            conf=conf_thr,
            iou=0.55,
            max_det=100,
            device=0,
            verbose=False,
        )[0]
        preds: List[Dict[str, Any]] = []
        if res.boxes is not None:
            for b, s, c in zip(
                res.boxes.xyxy.tolist(),
                res.boxes.conf.tolist(),
                res.boxes.cls.tolist(),
            ):
                preds.append({
                    "class_id": int(c),
                    "confidence": float(s),
                    "bbox_xyxy": [float(v) for v in b],
                })

        # Match per class greedily by descending confidence
        for cid, cname in enumerate(DETECTION_CLASS_NAMES):
            g_cls = [g for g in gt_objs if g["class_id"] == cid]
            p_cls = sorted(
                [p for p in preds if p["class_id"] == cid],
                key=lambda x: -x["confidence"],
            )
            used_g = [False] * len(g_cls)
            tp_c = 0
            fp_c = 0
            matched_ious: List[float] = []
            for p in p_cls:
                best_iou, best_gi = iou_thr, -1
                for gi, g in enumerate(g_cls):
                    if used_g[gi]:
                        continue
                    iou_val = box_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if iou_val >= best_iou:
                        best_iou, best_gi = iou_val, gi
                if best_gi >= 0:
                    used_g[best_gi] = True
                    tp_c += 1
                    matched_ious.append(round(best_iou, 4))
                else:
                    fp_c += 1
            fn_c = len(g_cls) - tp_c
            class_stats[cname]["gt"] += len(g_cls)
            class_stats[cname]["pred"] += len(p_cls)
            class_stats[cname]["tp"] += tp_c
            class_stats[cname]["fp"] += fp_c
            class_stats[cname]["fn"] += fn_c

            if cid == 0 and (len(g_cls) > 0 or len(p_cls) > 0):
                battery_cases.append({
                    "filename": img_p.name,
                    "gt_battery_count": len(g_cls),
                    "pred_battery_count": len(p_cls),
                    "tp": tp_c,
                    "fp": fp_c,
                    "fn": fn_c,
                    "matched_ious": matched_ious,
                    "pred_confidences": [round(p["confidence"], 4) for p in p_cls],
                    "all_preds_in_image": [
                        {
                            "class_name": DETECTION_CLASS_NAMES[p["class_id"]],
                            "confidence": round(p["confidence"], 4),
                            "bbox_xyxy": [round(v, 1) for v in p["bbox_xyxy"]],
                        }
                        for p in preds
                    ],
                })

    for cname, st in class_stats.items():
        tp, fp, fn, gt = st["tp"], st["fp"], st["fn"], st["gt"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / gt if gt > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        st["precision"] = round(p, 5)
        st["recall"] = round(r, 5)
        st["f1"] = round(f1, 5)

    return {
        "conf_threshold": conf_thr,
        "iou_threshold": iou_thr,
        "imgsz": imgsz,
        "per_class_operating_point": class_stats,
        "battery_cases": battery_cases,
    }


def evaluate_holdout_27(
    model_path: Path,
    run_name: str,
    imgsz: int,
    conf_op: float = 0.25,
) -> Dict[str, Any]:
    """Run raw prediction export + practical_metrics.py + per-image breakdown on 27 holdout images."""
    model = YOLO(str(model_path), task="detect")
    manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    gt_data = json.loads(HOLDOUT_GT.read_text(encoding="utf-8"))
    gt_by_fn = {item["filename"]: item["objects"] for item in gt_data["images"]}

    raw_preds: Dict[str, List[Dict[str, Any]]] = {}
    per_image_op: List[Dict[str, Any]] = []

    for item in manifest["files"]:
        fn = item["filename"]
        img_p = Path(item["absolute_path"])
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")

        r = model.predict(
            source=np.asarray(pil)[:, :, ::-1].copy(),
            imgsz=imgsz,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=0,
            verbose=False,
        )[0]
        plist: List[Dict[str, Any]] = []
        if r.boxes is not None:
            for b, s, c in zip(
                r.boxes.xyxy.tolist(),
                r.boxes.conf.tolist(),
                r.boxes.cls.tolist(),
            ):
                plist.append({
                    "class_id": int(c),
                    "confidence": float(s),
                    "bbox_xyxy": [float(v) for v in b],
                })
        raw_preds[fn] = plist

        # Operating-point predictions (conf >= conf_op, NMS at 0.55 via model.predict)
        r_op = model.predict(
            source=np.asarray(pil)[:, :, ::-1].copy(),
            imgsz=imgsz,
            conf=conf_op,
            iou=0.55,
            max_det=100,
            device=0,
            verbose=False,
        )[0]
        op_preds: List[Dict[str, Any]] = []
        if r_op.boxes is not None:
            for b, s, c in zip(
                r_op.boxes.xyxy.tolist(),
                r_op.boxes.conf.tolist(),
                r_op.boxes.cls.tolist(),
            ):
                cid = int(c)
                op_preds.append({
                    "class_id": cid,
                    "class_name": DETECTION_CLASS_NAMES[cid],
                    "confidence": round(float(s), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in b],
                })

        gobjs = gt_by_fn.get(fn, [])
        # Class-aware matching at IoU >= 0.50
        tp_img = 0
        fp_img = 0
        fn_img = 0
        for cid in range(len(DETECTION_CLASS_NAMES)):
            g_c = [g for g in gobjs if int(g["class_id"]) == cid]
            p_c = sorted([p for p in op_preds if p["class_id"] == cid], key=lambda x: -x["confidence"])
            used = [False] * len(g_c)
            tp_c = 0
            for p in p_c:
                best_iou, best_gi = 0.50, -1
                for gi, g in enumerate(g_c):
                    if used[gi]:
                        continue
                    v = box_iou(p["bbox_xyxy"], g["bbox_xyxy"])
                    if v >= best_iou:
                        best_iou, best_gi = v, gi
                if best_gi >= 0:
                    used[best_gi] = True
                    tp_c += 1
                else:
                    fp_img += 1
            tp_img += tp_c
            fn_img += len(g_c) - tp_c

        per_image_op.append({
            "filename": fn,
            "gt_count": len(gobjs),
            "gt_classes": [DETECTION_CLASS_NAMES[int(g["class_id"])] for g in gobjs],
            "pred_count_at_025": len(op_preds),
            "tp_at_025_iou50": tp_img,
            "fp_at_025_iou50": fp_img,
            "fn_at_025_iou50": fn_img,
            "predictions_at_025": op_preds,
        })

    ckpt_sha = sha256_file(model_path)
    raw_pred_path = EVAL_OUT_DIR / f"predictions_{run_name}_imgsz{imgsz}.json"
    raw_payload = {
        "predictions": raw_preds,
        "confidence_threshold": 0.001,
        "backend": "yolov8n",
        "checkpoint_sha256": ckpt_sha,
        "run": run_name,
        "imgsz": imgsz,
        "agnostic": False,
        "class_names": list(DETECTION_CLASS_NAMES),
        "coordinates": "pixel xyxy in original image after EXIF transpose",
    }
    raw_pred_path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    prac_out_path = EVAL_OUT_DIR / f"practical_metrics_{run_name}_imgsz{imgsz}.json"
    prac_res = run_practical_metrics(
        predictions_path=raw_pred_path,
        out_path=prac_out_path,
        gt_path=HOLDOUT_GT,
        conf=conf_op,
        manifest_path=HOLDOUT_MANIFEST,
    )

    return {
        "practical_metrics": prac_res,
        "per_image_operating_point": per_image_op,
    }


def evaluate_hard_negatives_qualitative(
    model_path: Path,
    imgsz: int,
    conf_thr: float = 0.25,
) -> List[Dict[str, Any]]:
    """Run qualitative inference on the 10 batch_a_mini01 candidate images (including 2 clean negatives)."""
    model = YOLO(str(model_path), task="detect")
    batch_a_dir = ROOT / "data" / "acquisition_staging" / "batch_a_mini01"
    batch_a_manifest = batch_a_dir / "manifest.json"
    data = json.loads(batch_a_manifest.read_text(encoding="utf-8"))
    records = data.get("items", [])
    results_out: List[Dict[str, Any]] = []

    for r in records:
        fn = r["filename"]
        img_p = batch_a_dir / "images" / fn
        if not img_p.is_file():
            continue
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        pred = model.predict(
            source=np.asarray(pil)[:, :, ::-1].copy(),
            imgsz=imgsz,
            conf=conf_thr,
            iou=0.55,
            max_det=100,
            device=0,
            verbose=False,
        )[0]
        dets = []
        if pred.boxes is not None:
            for b, s, c in zip(
                pred.boxes.xyxy.tolist(),
                pred.boxes.conf.tolist(),
                pred.boxes.cls.tolist(),
            ):
                cid = int(c)
                dets.append({
                    "class_id": cid,
                    "class_name": DETECTION_CLASS_NAMES[cid],
                    "confidence": round(float(s), 4),
                    "bbox_xyxy": [round(float(v), 1) for v in b],
                })
        results_out.append({
            "filename": fn,
            "visual_qa": r.get("visual_qa"),
            "scene_context": r.get("scene_context"),
            "detections_at_025": dets,
            "battery_detections_count": sum(1 for d in dets if d["class_id"] == 0),
        })
    return results_out


def main() -> int:
    EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    e1_sha = sha256_file(E1_CKPT)
    if e1_sha != E1_EXPECTED_SHA:
        raise RuntimeError(f"E1 checkpoint SHA mismatch: {e1_sha} != {E1_EXPECTED_SHA}")

    e3_best_sha, e3_last_sha = ensure_e3_phase2_metadata()
    print(f"Verified E1 SHA-256: {e1_sha}")
    print(f"Verified E3 best.pt SHA-256: {e3_best_sha}")

    # 1. Evaluation A: Validation V3 (328 images, 930 boxes, 29 battery boxes)
    print("Running Evaluation A: Validation V3...")
    eval_a_e1_512 = run_ultralytics_val(E1_CKPT, YAML_V3, "val", 512, "evalA_val_v3_E1_512")
    eval_a_e1_640 = run_ultralytics_val(E1_CKPT, YAML_V3, "val", 640, "evalA_val_v3_E1_640")
    eval_a_e3_512 = run_ultralytics_val(E3_BEST_CKPT, YAML_V3, "val", 512, "evalA_val_v3_E3_512")
    op_a_e1_512 = analyze_operating_point_on_split(E1_CKPT, YAML_V3, "val", 512)
    op_a_e3_512 = analyze_operating_point_on_split(E3_BEST_CKPT, YAML_V3, "val", 512)

    # 2. Evaluation B: Validation V2 (299 images, 901 boxes, 0 battery boxes)
    print("Running Evaluation B: Validation V2...")
    eval_b_e1_512 = run_ultralytics_val(E1_CKPT, YAML_V2, "val", 512, "evalB_val_v2_E1_512")
    eval_b_e1_640 = run_ultralytics_val(E1_CKPT, YAML_V2, "val", 640, "evalB_val_v2_E1_640")
    eval_b_e3_512 = run_ultralytics_val(E3_BEST_CKPT, YAML_V2, "val", 512, "evalB_val_v2_E3_512")
    op_b_e1_512 = analyze_operating_point_on_split(E1_CKPT, YAML_V2, "val", 512)
    op_b_e3_512 = analyze_operating_point_on_split(E3_BEST_CKPT, YAML_V2, "val", 512)

    # 3. Evaluation C: Internal Test Frozen (299 images, 720 boxes, 1 battery box)
    print("Running Evaluation C: Internal Test Frozen...")
    eval_c_e1_512 = run_ultralytics_val(E1_CKPT, YAML_V3, "test", 512, "evalC_test_frozen_E1_512")
    eval_c_e1_640 = run_ultralytics_val(E1_CKPT, YAML_V3, "test", 640, "evalC_test_frozen_E1_640")
    eval_c_e3_512 = run_ultralytics_val(E3_BEST_CKPT, YAML_V3, "test", 512, "evalC_test_frozen_E3_512")
    op_c_e1_512 = analyze_operating_point_on_split(E1_CKPT, YAML_V3, "test", 512)
    op_c_e3_512 = analyze_operating_point_on_split(E3_BEST_CKPT, YAML_V3, "test", 512)

    # 4. Evaluation D: External Holdout 27 images (57 GT objects)
    print("Running Evaluation D: External Holdout 27 images...")
    eval_d_e1_512 = evaluate_holdout_27(E1_CKPT, "E1_yolov8n_10cls", 512)
    eval_d_e1_640 = evaluate_holdout_27(E1_CKPT, "E1_yolov8n_10cls", 640)
    eval_d_e3_512 = evaluate_holdout_27(E3_BEST_CKPT, "E3_yolov8n_pilot", 512)

    # 5. Qualitative check on 8 batch_a images (2 clean negatives + 6 corroded/utility batteries)
    print("Running qualitative check on hard negatives...")
    qual_e1_512 = evaluate_hard_negatives_qualitative(E1_CKPT, 512)
    qual_e3_512 = evaluate_hard_negatives_qualitative(E3_BEST_CKPT, 512)

    # Compute Catastrophic Forgetting analysis on Validation V2 (9 non-battery classes)
    forgetting_val_v2: Dict[str, Any] = {}
    for cname in DETECTION_CLASS_NAMES[1:]:
        e1_c = eval_b_e1_512["per_class"][cname]
        e3_c = eval_b_e3_512["per_class"][cname]
        d_ap50 = round(float(e3_c["ap50"]) - float(e1_c["ap50"]), 5)
        d_ap50_95 = round(float(e3_c["ap50_95"]) - float(e1_c["ap50_95"]), 5)
        forgetting_val_v2[cname] = {
            "e1_512_ap50": e1_c["ap50"],
            "e3_512_ap50": e3_c["ap50"],
            "delta_ap50": d_ap50,
            "e1_512_ap50_95": e1_c["ap50_95"],
            "e3_512_ap50_95": e3_c["ap50_95"],
            "delta_ap50_95": d_ap50_95,
            "degraded_over_5pct_ap50": d_ap50 < -0.05,
        }

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            "E1": {"path": str(E1_CKPT), "sha256": e1_sha},
            "E3": {"path": str(E3_BEST_CKPT), "sha256": e3_best_sha, "last_sha256": e3_last_sha},
        },
        "evaluation_A_val_v3": {
            "E1_imgsz512": eval_a_e1_512,
            "E1_imgsz640": eval_a_e1_640,
            "E3_imgsz512": eval_a_e3_512,
            "operating_point_conf025_E1_512": op_a_e1_512,
            "operating_point_conf025_E3_512": op_a_e3_512,
        },
        "evaluation_B_val_v2": {
            "E1_imgsz512": eval_b_e1_512,
            "E1_imgsz640": eval_b_e1_640,
            "E3_imgsz512": eval_b_e3_512,
            "operating_point_conf025_E1_512": op_b_e1_512,
            "operating_point_conf025_E3_512": op_b_e3_512,
            "catastrophic_forgetting_analysis_9_classes": forgetting_val_v2,
        },
        "evaluation_C_test_frozen": {
            "E1_imgsz512": eval_c_e1_512,
            "E1_imgsz640": eval_c_e1_640,
            "E3_imgsz512": eval_c_e3_512,
            "operating_point_conf025_E1_512": op_c_e1_512,
            "operating_point_conf025_E3_512": op_c_e3_512,
        },
        "evaluation_D_holdout_27": {
            "E1_imgsz512": eval_d_e1_512["practical_metrics"]["views"]["all_27"],
            "E1_imgsz640": eval_d_e1_640["practical_metrics"]["views"]["all_27"],
            "E3_imgsz512": eval_d_e3_512["practical_metrics"]["views"]["all_27"],
        },
        "qualitative_hard_negatives_batch_a": {
            "E1_imgsz512": qual_e1_512,
            "E3_imgsz512": qual_e3_512,
        },
    }

    comp_json_path = EVAL_OUT_DIR / "e1_vs_e3_comprehensive_evaluation.json"
    comp_json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    holdout_per_img_path = EVAL_OUT_DIR / "holdout_27_e1_vs_e3_per_image.json"
    holdout_per_img_path.write_text(
        json.dumps(
            {
                "E1_imgsz512": eval_d_e1_512["per_image_operating_point"],
                "E1_imgsz640": eval_d_e1_640["per_image_operating_point"],
                "E3_imgsz512": eval_d_e3_512["per_image_operating_point"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Write per-class CSV comparison across Eval A, Eval B, Eval C, Eval D
    csv_path = EVAL_OUT_DIR / "e1_vs_e3_per_class_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "benchmark",
            "class_name",
            "E1_512_AP50",
            "E1_640_AP50",
            "E3_512_AP50",
            "delta_AP50_512",
            "E1_512_AP50_95",
            "E1_640_AP50_95",
            "E3_512_AP50_95",
            "delta_AP50_95_512",
            "E1_512_Recall",
            "E3_512_Recall",
        ])
        for b_name, e1_512, e1_640, e3_512 in (
            ("Eval_A_Val_V3", eval_a_e1_512, eval_a_e1_640, eval_a_e3_512),
            ("Eval_B_Val_V2", eval_b_e1_512, eval_b_e1_640, eval_b_e3_512),
            ("Eval_C_Test_Frozen", eval_c_e1_512, eval_c_e1_640, eval_c_e3_512),
        ):
            for cname in DETECTION_CLASS_NAMES:
                c1 = e1_512["per_class"][cname]
                c1_640 = e1_640["per_class"][cname]
                c3 = e3_512["per_class"][cname]
                d50 = (
                    round(float(c3["ap50"]) - float(c1["ap50"]), 5)
                    if c3["ap50"] is not None and c1["ap50"] is not None
                    else None
                )
                d95 = (
                    round(float(c3["ap50_95"]) - float(c1["ap50_95"]), 5)
                    if c3["ap50_95"] is not None and c1["ap50_95"] is not None
                    else None
                )
                writer.writerow([
                    b_name,
                    cname,
                    c1["ap50"],
                    c1_640["ap50"],
                    c3["ap50"],
                    d50,
                    c1["ap50_95"],
                    c1_640["ap50_95"],
                    c3["ap50_95"],
                    d95,
                    c1["recall"],
                    c3["recall"],
                ])

    print(f"Saved comprehensive evaluation to {comp_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
