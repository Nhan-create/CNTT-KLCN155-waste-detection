"""Probe: compare raw Ultralytics predictions with WasteDetector wrapper (CPU only).

Usage: .venv\\Scripts\\python.exe reports/model_v2/scripts/yolo_box_anomaly_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.detection.yolo import WasteDetector  # noqa: E402

CKPT = ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"
TRAIN_LIST = ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/train-val-train-abs.txt"
VAL_LIST = ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/train-val-val-abs.txt"
QA_DIR = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout")))
QA_IMAGES = ["base64-1717231151649602366964.jpeg", "Phan-Loai-Rac-Tai-Ng-01.jpg", "images (10).jpg", "rac-thai-quan-ao-4.jpg"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_yolo_gt(img_path: Path, w: int, h: int):
    label = Path(str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    rows = []
    if label.is_file():
        for line in label.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            rows.append((c, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return rows


def raw_predict(model, pil, imgsz, conf=0.2):
    r = model.predict(source=np.asarray(pil)[:, :, ::-1].copy(), imgsz=imgsz, conf=conf, iou=0.45, device="cpu", verbose=False)[0]
    return [(int(c), float(s), [round(float(v), 1) for v in b]) for b, s, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]


def summarize(preds, gt=None):
    out = []
    for c, s, b in preds:
        row = {"cls": c, "conf": round(s, 3), "xyxy": b, "w": round(b[2] - b[0], 1), "h": round(b[3] - b[1], 1)}
        if gt:
            best = max(gt, key=lambda g: iou(b, g[1]))
            row["best_gt_iou"] = round(iou(b, best[1]), 3)
            row["best_gt_cls"] = best[0]
        out.append(row)
    return out


def main():
    from ultralytics import YOLO

    model = YOLO(str(CKPT), task="detect")
    report = {"checkpoint": str(CKPT), "train_imgsz": model.ckpt.get("train_args", {}).get("imgsz"), "cases": []}
    train = [Path(p) for p in TRAIN_LIST.read_text().split()]
    val = [Path(p) for p in VAL_LIST.read_text().split()]
    real_train = [p for p in train if "/real/" in p.as_posix()][:3]
    syn_train = [p for p in train if "/synthetic/" in p.as_posix()][:3]
    samples = [("qa", QA_DIR / n) for n in QA_IMAGES] + [("train_real", p) for p in real_train] + \
              [("train_synthetic", p) for p in syn_train] + [("val_real", p) for p in val[:3]]
    wrappers = {s: WasteDetector(CKPT, device="cpu", confidence_threshold=0.2, iou_threshold=0.45, image_size=s) for s in (320, 640)}
    for group, path in samples:
        with Image.open(path) as im:
            pil = ImageOps.exif_transpose(im).convert("RGB")
        w, h = pil.size
        gt = load_yolo_gt(path, w, h) if group != "qa" else None
        case = {"group": group, "path": str(path), "size": [w, h], "gt": [{"cls": c, "xyxy": [round(v, 1) for v in b]} for c, b in (gt or [])]}
        for s in (320, 640):
            raw = raw_predict(model, pil, s)
            wr = wrappers[s].detect_pil(pil)
            wr_list = [(d.class_index, d.confidence, [round(d.box.x1, 1), round(d.box.y1, 1), round(d.box.x2, 1), round(d.box.y2, 1)]) for d in wr.detections]
            raw_sorted = sorted(raw, key=lambda r: (-r[1], r[0]))
            max_diff = max((max(abs(a - b) for a, b in zip(r[2], q[2])) for r, q in zip(raw_sorted, wr_list)), default=0.0)
            case[f"imgsz{s}"] = {
                "raw": summarize(raw, gt), "wrapper_n": len(wr_list), "raw_n": len(raw),
                "raw_vs_wrapper_max_abs_px": round(max_diff, 2),
            }
        report["cases"].append(case)
    out = ROOT / "reports/model_v2/yolo_box_anomaly_probe.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for c in report["cases"]:
        print(c["group"], Path(c["path"]).name, c["size"], "GT:", [(g["cls"], g["xyxy"]) for g in c["gt"]][:4])
        for s in (320, 640):
            k = c[f"imgsz{s}"]
            print(f"  imgsz={s} raw_n={k['raw_n']} wrap_n={k['wrapper_n']} diff={k['raw_vs_wrapper_max_abs_px']}")
            for r in k["raw"][:5]:
                print("    ", r)


if __name__ == "__main__":
    main()
