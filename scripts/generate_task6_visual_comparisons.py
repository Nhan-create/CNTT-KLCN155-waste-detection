"""Task 5: Generate Side-by-Side Visual Comparisons (Original/GT | YOLOv8n E1 | YOLOv8n E3).

Covers all 5 required visual inspection groups:
  1. Single battery on table / simple background (Validation V3)
  2. Angled, corroded, or small batteries (Validation V3 + corroded samples)
  3. Utility boxes, brick walls, and surfaces prone to false positives (Hard Negatives)
  4. Mixed multi-class waste images & Holdout battery images
  5. Holdout images with previous E1 False Negatives (FN) or False Positives (FP)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detection.schema import DETECTION_CLASS_NAMES

E1_CKPT = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
E3_CKPT = ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot" / "weights" / "best.pt"
YAML_V3 = ROOT / "configs" / "detection_dataset_real_v3_pilot.yaml"
HOLDOUT_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
HOLDOUT_GT = ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac.json"

VIS_OUT_DIR = ROOT / "artifacts" / "detection_v3_pilot" / "visual_comparison_e1_vs_e3"

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 230, 118),    # battery - bright green
    1: (139, 195, 74),   # biological
    2: (255, 193, 7),    # cardboard
    3: (233, 30, 99),    # clothes
    4: (0, 188, 212),    # glass
    5: (158, 158, 158),  # metal
    6: (63, 81, 181),    # paper
    7: (255, 87, 34),    # plastic
    8: (156, 39, 176),   # shoes
    9: (244, 67, 54),    # trash
}


def load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ("arialbd.ttf", "arial.ttf", "consola.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def predict_boxes(
    model: YOLO,
    pil_img: Image.Image,
    imgsz: int = 512,
    conf: float = 0.25,
    iou: float = 0.55,
) -> List[Dict[str, Any]]:
    res = model.predict(
        source=np.asarray(pil_img)[:, :, ::-1].copy(),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=100,
        device=0,
        verbose=False,
    )[0]
    out: List[Dict[str, Any]] = []
    if res.boxes is not None:
        for b, s, c in zip(
            res.boxes.xyxy.tolist(),
            res.boxes.conf.tolist(),
            res.boxes.cls.tolist(),
        ):
            cid = int(c)
            out.append({
                "class_id": cid,
                "class_name": DETECTION_CLASS_NAMES[cid],
                "confidence": float(s),
                "bbox_xyxy": [float(v) for v in b],
            })
    return out


def draw_boxes_on_panel(
    pil_img: Image.Image,
    boxes: List[Dict[str, Any]],
    is_gt: bool = False,
) -> Image.Image:
    canvas = pil_img.copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(13, min(22, int(min(canvas.size) * 0.032))))

    for b in boxes:
        cid = int(b["class_id"])
        cname = b.get("class_name", DETECTION_CLASS_NAMES[cid])
        x1, y1, x2, y2 = b["bbox_xyxy"]
        color = (255, 215, 0) if is_gt else CLASS_COLORS.get(cid, (0, 255, 0))
        width = max(2, int(min(canvas.size) * 0.006))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        if is_gt:
            label = f"GT: {cname}"
        else:
            conf = float(b.get("confidence", 1.0))
            label = f"{cname} {conf:.2f}"

        bbox_txt = draw.textbbox((0, 0), label, font=font)
        tw = bbox_txt[2] - bbox_txt[0]
        th = bbox_txt[3] - bbox_txt[1]
        ty1 = max(0.0, y1 - th - 6)
        draw.rectangle([x1, ty1, x1 + tw + 8, ty1 + th + 6], fill=color)
        draw.text((x1 + 4, ty1 + 2), label, fill=(0, 0, 0), font=font)

    return canvas


def build_three_panel_comparison(
    orig_img: Image.Image,
    gt_boxes: Optional[List[Dict[str, Any]]],
    e1_boxes: List[Dict[str, Any]],
    e3_boxes: List[Dict[str, Any]],
    title_text: str,
    subtitle_text: str,
    target_panel_w: int = 480,
) -> Image.Image:
    w, h = orig_img.size
    scale = target_panel_w / float(w)
    panel_w = target_panel_w
    panel_h = max(240, int(round(h * scale)))

    def scale_boxes(blist: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not blist:
            return []
        scaled = []
        for item in blist:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            scaled.append({
                **item,
                "bbox_xyxy": [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
            })
        return scaled

    base_resized = orig_img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    p1 = draw_boxes_on_panel(base_resized, scale_boxes(gt_boxes), is_gt=True) if gt_boxes is not None else base_resized
    p2 = draw_boxes_on_panel(base_resized, scale_boxes(e1_boxes), is_gt=False)
    p3 = draw_boxes_on_panel(base_resized, scale_boxes(e3_boxes), is_gt=False)

    header_h = 82
    col_header_h = 36
    pad = 12
    total_w = panel_w * 3 + pad * 4
    total_h = header_h + col_header_h + panel_h + pad * 2

    sheet = Image.new("RGB", (total_w, total_h), (24, 26, 31))
    draw = ImageDraw.Draw(sheet)
    font_title = load_font(18)
    font_sub = load_font(14)
    font_col = load_font(15)

    draw.text((pad, 12), title_text, fill=(255, 255, 255), font=font_title)
    draw.text((pad, 42), subtitle_text, fill=(190, 200, 215), font=font_sub)

    gt_hdr = (
        f"1. ẢNH GỐC + GT ({len(gt_boxes)} boxes)"
        if gt_boxes is not None
        else "1. ẢNH GỐC (Unannotated Hard Negative)"
    )
    e1_hdr = f"2. YOLOV8N E1 (Baseline v2) — {len(e1_boxes)} preds"
    e3_hdr = f"3. YOLOV8N E3 (Pilot v3) — {len(e3_boxes)} preds"

    for idx, (panel_im, hdr_txt, hdr_color) in enumerate([
        (p1, gt_hdr, (255, 215, 0)),
        (p2, e1_hdr, (255, 138, 101)),
        (p3, e3_hdr, (0, 230, 118)),
    ]):
        x0 = pad + idx * (panel_w + pad)
        y0 = header_h
        draw.rectangle([x0, y0, x0 + panel_w, y0 + col_header_h], fill=(38, 42, 50))
        draw.text((x0 + 10, y0 + 8), hdr_txt, fill=hdr_color, font=font_col)
        sheet.paste(panel_im, (x0, y0 + col_header_h))

    return sheet


def load_yolo_gt_boxes(lbl_p: Path, w: int, h: int) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    if not lbl_p.is_file():
        return boxes
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
        boxes.append({
            "class_id": cid,
            "class_name": DETECTION_CLASS_NAMES[cid],
            "bbox_xyxy": [x1, y1, x2, y2],
        })
    return boxes


def stack_panels_vertically(panels: List[Image.Image], title: str) -> Image.Image:
    max_w = max(p.width for p in panels)
    header_h = 56
    gap = 10
    total_h = header_h + sum(p.height for p in panels) + gap * (len(panels) + 1)
    canvas = Image.new("RGB", (max_w + 20, total_h), (15, 17, 21))
    draw = ImageDraw.Draw(canvas)
    font_h = load_font(22)
    draw.text((16, 16), title, fill=(255, 255, 255), font=font_h)
    y = header_h + gap
    for p in panels:
        canvas.paste(p, (10, y))
        y += p.height + gap
    return canvas


def main() -> int:
    VIS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_e1 = YOLO(str(E1_CKPT), task="detect")
    model_e3 = YOLO(str(E3_CKPT), task="detect")

    # Identify Validation V3 battery images (29 images)
    v3_cfg = yaml.safe_load(YAML_V3.read_text(encoding="utf-8"))
    v3_root = (YAML_V3.parent / v3_cfg["path"]).resolve()
    val_txt = (v3_root / v3_cfg["val"]).resolve()
    val_battery_items: List[Tuple[Path, Path]] = []
    for line in val_txt.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        img_p = ((val_txt.parent / rel) if rel.startswith(".") else (v3_root / rel)).resolve()
        if ".rf." in img_p.name or "_rf_" in img_p.name:
            lbl_p = Path(img_p.as_posix().replace("/images/", "/labels/")).with_suffix(".txt")
            val_battery_items.append((img_p, lbl_p))

    val_battery_items.sort(key=lambda x: x[0].name)

    manifest_records: List[Dict[str, Any]] = []
    group1_2_3_panels: List[Image.Image] = []
    group4_5_panels: List[Image.Image] = []

    # Group 1: Single battery on table / simple background (first 5 val battery images)
    for idx, (img_p, lbl_p) in enumerate(val_battery_items[:5], start=1):
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        gt_b = load_yolo_gt_boxes(lbl_p, *pil.size)
        e1_b = predict_boxes(model_e1, pil, imgsz=512, conf=0.25)
        e3_b = predict_boxes(model_e3, pil, imgsz=512, conf=0.25)
        panel = build_three_panel_comparison(
            orig_img=pil,
            gt_boxes=gt_b,
            e1_boxes=e1_b,
            e3_boxes=e3_b,
            title_text=f"[Group 1 - Pin đơn trên bàn] {img_p.name}",
            subtitle_text="Split: Validation V3 | imgsz=512, conf=0.25, iou=0.55",
        )
        out_name = f"group1_table_battery_{idx:02d}_{img_p.stem[:36]}.jpg"
        out_p = VIS_OUT_DIR / out_name
        panel.save(out_p, quality=92)
        group1_2_3_panels.append(panel)
        manifest_records.append({
            "group": "Group 1: Pin đơn trên bàn / nền đơn giản",
            "filename": img_p.name,
            "panel_file": out_name,
            "gt_boxes": len(gt_b),
            "e1_preds": e1_b,
            "e3_preds": e3_b,
        })

    # Group 2: Angled / small / corroded batteries (4 angled/small from Val V3 + 3 corroded from batch_a)
    for idx, (img_p, lbl_p) in enumerate(val_battery_items[-4:], start=1):
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        gt_b = load_yolo_gt_boxes(lbl_p, *pil.size)
        e1_b = predict_boxes(model_e1, pil, imgsz=512, conf=0.25)
        e3_b = predict_boxes(model_e3, pil, imgsz=512, conf=0.25)
        panel = build_three_panel_comparison(
            orig_img=pil,
            gt_boxes=gt_b,
            e1_boxes=e1_b,
            e3_boxes=e3_b,
            title_text=f"[Group 2 - Pin nghiêng / góc khó Val V3] {img_p.name}",
            subtitle_text="Split: Validation V3 | imgsz=512, conf=0.25, iou=0.55",
        )
        out_name = f"group2_angled_battery_{idx:02d}_{img_p.stem[:36]}.jpg"
        out_p = VIS_OUT_DIR / out_name
        panel.save(out_p, quality=92)
        group1_2_3_panels.append(panel)
        manifest_records.append({
            "group": "Group 2: Pin nghiêng, pin rỉ sét, pin nhỏ",
            "filename": img_p.name,
            "panel_file": out_name,
            "gt_boxes": len(gt_b),
            "e1_preds": e1_b,
            "e3_preds": e3_b,
        })

    batch_a_dir = ROOT / "data" / "acquisition_staging" / "batch_a_mini01"
    extra_group2_files = [
        (ROOT / "data" / "detection" / "images" / "real" / "battery238.jpg", ROOT / "data" / "detection" / "labels" / "real" / "battery238.txt", "Internal Test Frozen battery sample"),
        (batch_a_dir / "images" / "wiki_bat_corroded_screwdriver.jpg", None, "Quarantined corroded battery tool assembly"),
    ]
    for idx, (img_p, lbl_p, note_str) in enumerate(extra_group2_files, start=1):
        if not img_p.is_file():
            continue
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        gt_b = load_yolo_gt_boxes(lbl_p, *pil.size) if lbl_p is not None else None
        e1_b = predict_boxes(model_e1, pil, imgsz=512, conf=0.25)
        e3_b = predict_boxes(model_e3, pil, imgsz=512, conf=0.25)
        panel = build_three_panel_comparison(
            orig_img=pil,
            gt_boxes=gt_b,
            e1_boxes=e1_b,
            e3_boxes=e3_b,
            title_text=f"[Group 2 - Pin rỉ sét / Test Frozen] {img_p.name}",
            subtitle_text=f"{note_str} | imgsz=512, conf=0.25, iou=0.55",
        )
        out_name = f"group2_corroded_{idx:02d}_{img_p.stem}.jpg"
        panel.save(VIS_OUT_DIR / out_name, quality=92)
        group1_2_3_panels.append(panel)
        manifest_records.append({
            "group": "Group 2: Pin nghiêng, pin rỉ sét, pin nhỏ",
            "filename": img_p.name,
            "panel_file": out_name,
            "gt_boxes": len(gt_b) if gt_b is not None else 0,
            "e1_preds": e1_b,
            "e3_preds": e3_b,
        })

    # Group 3: Utility boxes, brick walls, and surfaces prone to false positives (Hard Negatives)
    hard_neg_files = [
        (batch_a_dir / "images" / "wiki_neg_brick_wall_clean.jpg", None),
        (batch_a_dir / "images" / "wiki_neg_utility_box_corner_05.jpg", None),
    ]
    for idx, (img_p, lbl_p) in enumerate(hard_neg_files, start=1):
        if not img_p.is_file():
            continue
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        gt_b = load_yolo_gt_boxes(lbl_p, *pil.size) if lbl_p is not None else None
        e1_b = predict_boxes(model_e1, pil, imgsz=512, conf=0.25)
        e3_b = predict_boxes(model_e3, pil, imgsz=512, conf=0.25)
        panel = build_three_panel_comparison(
            orig_img=pil,
            gt_boxes=gt_b,
            e1_boxes=e1_b,
            e3_boxes=e3_b,
            title_text=f"[Group 3 - Tường gạch / Hộp điện Hard Negative] {img_p.name}",
            subtitle_text="Qualitative Hard Negative Check (Out-of-train) | imgsz=512, conf=0.25, iou=0.55",
        )
        out_name = f"group3_hard_negative_{idx:02d}_{img_p.stem}.jpg"
        panel.save(VIS_OUT_DIR / out_name, quality=92)
        group1_2_3_panels.append(panel)
        manifest_records.append({
            "group": "Group 3: Hộp điện, tường, bề mặt dễ nhầm là pin (Hard Negative)",
            "filename": img_p.name,
            "panel_file": out_name,
            "gt_boxes": len(gt_b) if gt_b is not None else 0,
            "e1_preds": e1_b,
            "e3_preds": e3_b,
        })

    # Group 4 & Group 5: External Holdout (battery images, mixed waste, and previous E1 FN/FP cases)
    holdout_manifest = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    holdout_gt = json.loads(HOLDOUT_GT.read_text(encoding="utf-8"))
    gt_by_fn = {item["filename"]: item["objects"] for item in holdout_gt["images"]}

    # Select key holdout images covering batteries, mixed waste, and E1 FN/FP cases
    selected_holdout = [
        ("battery-sample.png", "Group 4: Holdout Pin & Rác hỗn hợp"),
        ("pin-160724628783677618625.jpg", "Group 4: Holdout Pin & Rác hỗn hợp"),
        ("anh-rac-thai-nhua.jpg", "Group 4: Holdout Pin & Rác hỗn hợp"),
        ("phan-loai-rac-thai-sinh-hoat-1.jpg", "Group 4: Holdout Pin & Rác hỗn hợp"),
        ("rac-thai-sinh-hoat.jpg", "Group 4: Holdout Pin & Rác hỗn hợp"),
        ("cardboard-boxes.jpg", "Group 5: Holdout từng bị FN hoặc FP ở E1"),
        ("glass-bottles.jpg", "Group 5: Holdout từng bị FN hoặc FP ở E1"),
        ("metal-cans.jpg", "Group 5: Holdout từng bị FN hoặc FP ở E1"),
        ("old-shoes.jpg", "Group 5: Holdout từng bị FN hoặc FP ở E1"),
        ("plastic-bottles.jpg", "Group 5: Holdout từng bị FN hoặc FP ở E1"),
    ]
    selected_map = dict(selected_holdout)

    for idx, item in enumerate(holdout_manifest["files"], start=1):
        fn = item["filename"]
        img_p = Path(item["absolute_path"])
        with Image.open(img_p) as opened:
            pil = ImageOps.exif_transpose(opened).convert("RGB")
        gobjs = [
            {
                "class_id": int(o["class_id"]),
                "class_name": DETECTION_CLASS_NAMES[int(o["class_id"])],
                "bbox_xyxy": [float(v) for v in o["bbox_xyxy"]],
            }
            for o in gt_by_fn.get(fn, [])
        ]
        e1_b = predict_boxes(model_e1, pil, imgsz=512, conf=0.25)
        e3_b = predict_boxes(model_e3, pil, imgsz=512, conf=0.25)

        is_battery_holdout = any(g["class_id"] == 0 for g in gobjs)
        is_mixed = len(set(g["class_id"] for g in gobjs)) >= 2 or len(gobjs) >= 3
        grp_label = selected_map.get(
            fn,
            "Group 4: Holdout Pin & Rác hỗn hợp" if (is_battery_holdout or is_mixed) else "Group 5: Holdout từng bị FN hoặc FP ở E1",
        )

        panel = build_three_panel_comparison(
            orig_img=pil,
            gt_boxes=gobjs,
            e1_boxes=e1_b,
            e3_boxes=e3_b,
            title_text=f"[{grp_label}] {fn}",
            subtitle_text="External Holdout (D:\\Hinh_Anh_Rac) | imgsz=512, conf=0.25, iou=0.55",
        )
        safe_stem = Path(fn).stem[:32].replace(" ", "_")
        out_name = f"holdout_{idx:02d}_{safe_stem}.jpg"
        panel.save(VIS_OUT_DIR / out_name, quality=92)
        if is_battery_holdout or idx <= 12:
            group4_5_panels.append(panel)
        manifest_records.append({
            "group": grp_label,
            "filename": fn,
            "panel_file": out_name,
            "gt_boxes": len(gobjs),
            "e1_preds": e1_b,
            "e3_preds": e3_b,
        })

    # Save overview contact sheets
    cs1 = stack_panels_vertically(
        group1_2_3_panels,
        "TASK 6 VISUAL COMPARISON — GROUPS 1, 2, 3 (BATTERY VAL V3, CORRODED & HARD NEGATIVES)",
    )
    cs1.save(VIS_OUT_DIR / "contact_sheet_groups_1_2_3_battery_and_hard_negatives.jpg", quality=90)

    cs2 = stack_panels_vertically(
        group4_5_panels,
        "TASK 6 VISUAL COMPARISON — GROUPS 4 & 5 (EXTERNAL HOLDOUT BATTERY, MIXED WASTE & FN/FP)",
    )
    cs2.save(VIS_OUT_DIR / "contact_sheet_groups_4_5_holdout_and_mixed_waste.jpg", quality=90)

    manifest_path = VIS_OUT_DIR / "visual_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "total_panels": len(manifest_records),
                "contact_sheets": [
                    "contact_sheet_groups_1_2_3_battery_and_hard_negatives.jpg",
                    "contact_sheet_groups_4_5_holdout_and_mixed_waste.jpg",
                ],
                "panels": manifest_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Generated {len(manifest_records)} 3-panel comparisons in {VIS_OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
