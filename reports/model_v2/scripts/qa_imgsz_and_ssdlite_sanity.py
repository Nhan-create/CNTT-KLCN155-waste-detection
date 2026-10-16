"""Evaluate YOLO imgsz 320 vs 640 on the 27 QA images and SSDLite sanity on train images (CPU).

Reuses QA's matching functions read-only; writes only to reports/model_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports/qa_acceptance"))

from run_baseline_evaluation import match_predictions_to_gt  # noqa: E402
from src.detection.ssdlite import SSDLiteWasteDetector  # noqa: E402
from src.detection.yolo import WasteDetector  # noqa: E402

YOLO_CKPT = ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"
SSD_CKPT = ROOT / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"
SSD_TRAIN_LIST = ROOT / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/train-val-train-abs.txt"


def load(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def preds_of(result):
    return [{"class_id": d.class_index, "class_name": d.class_id, "confidence": d.confidence,
             "bbox_xyxy": [d.box.x1, d.box.y1, d.box.x2, d.box.y2]} for d in result.detections]


def qa_eval(detector, manifest, gt_by_name, iou_thr):
    tot = {"tp": 0, "wrong_class": 0, "fn": 0, "fp": 0, "pred": 0, "loc_hits": 0}
    rel_areas = []
    for f in manifest["files"]:
        pil = load(f["absolute_path"])
        preds = preds_of(detector.detect_pil(pil))
        w, h = pil.size
        rel_areas += [((p["bbox_xyxy"][2] - p["bbox_xyxy"][0]) * (p["bbox_xyxy"][3] - p["bbox_xyxy"][1])) / (w * h) for p in preds]
        m = match_predictions_to_gt(preds, gt_by_name.get(f["filename"], []), iou_match_threshold=iou_thr)
        tot["tp"] += m["tp_count"]; tot["wrong_class"] += m["wrong_class_count"]
        tot["fn"] += m["fn_count"]; tot["fp"] += m["fp_count"]; tot["pred"] += len(preds)
        tot["loc_hits"] += len(m["matched_pairs"])
    rel_areas.sort()
    tot["median_pred_area_frac"] = round(rel_areas[len(rel_areas) // 2], 4) if rel_areas else None
    return tot


def yolo_gt(img_path: Path, w, h):
    label = Path(str(img_path).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
    out = []
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) == 5:
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            out.append({"object_id": len(out), "class_id": c, "class_name": str(c),
                        "bbox_xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


def main():
    manifest = json.loads((ROOT / "reports/qa_acceptance/manifest_hinh_anh_rac.json").read_text(encoding="utf-8"))
    gt = json.loads((ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json").read_text(encoding="utf-8"))
    gt_by_name = {i["filename"]: i["objects"] for i in gt["images"]}
    report = {"qa_yolo": {}, "qa_ssdlite": {}, "ssdlite_train_sanity": {}}
    for s in (320, 640):
        det = WasteDetector(YOLO_CKPT, device="cpu", confidence_threshold=0.2, iou_threshold=0.45, image_size=s)
        for thr in (0.2, 0.5):
            report["qa_yolo"][f"imgsz{s}_iou{thr}"] = qa_eval(det, manifest, gt_by_name, thr)
    ssd = SSDLiteWasteDetector(SSD_CKPT, device="cpu", confidence_threshold=0.2, iou_threshold=0.45)
    for thr in (0.2, 0.5):
        report["qa_ssdlite"][f"iou{thr}"] = qa_eval(ssd, manifest, gt_by_name, thr)
    # SSDLite sanity on train images with labels: class mapping (label-1) + coordinates.
    train = [Path(p) for p in SSD_TRAIN_LIST.read_text().split()]
    picks = [p for p in train if "/real/" in p.as_posix()][:6] + [p for p in train if "/synthetic/" in p.as_posix()][:6]
    tot = {"gt": 0, "loc_hits_iou0.5": 0, "class_correct": 0, "pred": 0}
    for p in picks:
        pil = load(p)
        g = yolo_gt(p, *pil.size)
        preds = preds_of(ssd.detect_pil(pil))
        m = match_predictions_to_gt(preds, g, iou_match_threshold=0.5)
        tot["gt"] += len(g); tot["pred"] += len(preds)
        tot["loc_hits_iou0.5"] += len(m["matched_pairs"]); tot["class_correct"] += m["tp_count"]
    report["ssdlite_train_sanity"] = {"images": [str(p) for p in picks], **tot}
    out = ROOT / "reports/model_v2/qa_imgsz_and_ssdlite_sanity.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "ssdlite_train_sanity"}, indent=1))
    print({k: v for k, v in report["ssdlite_train_sanity"].items() if k != "images"})


if __name__ == "__main__":
    main()
