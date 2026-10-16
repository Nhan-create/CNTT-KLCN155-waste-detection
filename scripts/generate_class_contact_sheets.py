"""Generate high quality class-by-class contact sheets and an overall summary contact sheet.

Destination: D:\Hinh_Anh_Rac\demo_multiobject_100_20261012\contact_sheets\
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[1] / "data" / "holdout"))) / "demo_multiobject_100_20261012")
ORIGINALS_DIR = BASE_DIR / "originals"
OVERLAYS_DIR = BASE_DIR / "overlays"
CONTACT_SHEETS_DIR = BASE_DIR / "contact_sheets"
MANIFESTS_DIR = BASE_DIR / "manifests"

PREDICTIONS_JSON = MANIFESTS_DIR / "predictions_yolov8n_raw.json"
AUDIT_JSON = MANIFESTS_DIR / "candidate_100_download_audit.json"

CLASS_NAMES_VI = {
    "battery": "Pin",
    "biological": "Rác hữu cơ / Thực phẩm",
    "cardboard": "Bìa carton",
    "clothes": "Quần áo / Vải vóc",
    "glass": "Thủy tinh / Chai lọ",
    "metal": "Kim loại / Lon hộp",
    "paper": "Giấy",
    "plastic": "Nhựa / Chai nhựa",
    "shoes": "Giày dép",
    "trash": "Rác hỗn hợp / Khác",
}


def fit_image(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Aspect-preserving thumbnail inside target bounding box with letterboxing."""
    im_copy = im.copy()
    im_copy.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (20, 20, 24))
    offset_x = (target_w - im_copy.width) // 2
    offset_y = (target_h - im_copy.height) // 2
    canvas.paste(im_copy, (offset_x, offset_y))
    return canvas


def build_card(orig_path: Path, overlay_path: Path, item_info: dict, pred_info: dict, card_w: int = 700, card_h: int = 320) -> Image.Image:
    card = Image.new("RGB", (card_w, card_h), (30, 32, 38))
    draw = ImageDraw.Draw(card)

    cid = item_info["candidate_id"]
    cname = item_info["target_class_name"]
    title = item_info["commons_file_title"].replace("File:", "")[:35]
    num_dets = pred_info.get("num_detections", 0)
    dets = pred_info.get("detections", [])
    classes_str = ", ".join(sorted(set(d["class_name"] for d in dets))) if dets else "None"

    # Header bar
    header_h = 42
    draw.rectangle([(0, 0), (card_w, header_h)], fill=(40, 44, 52))
    draw.text((10, 6), f"{cid} | {title}", fill=(255, 255, 255))
    draw.text((10, 22), f"Class: {cname} ({CLASS_NAMES_VI.get(cname, '')})", fill=(148, 163, 184))

    badge_text = f"YOLO: {num_dets} boxes [{classes_str}]"
    badge_color = (34, 197, 94) if num_dets > 0 else (239, 68, 68)
    draw.text((card_w - 280, 14), badge_text, fill=badge_color)

    # Images area
    img_area_w = (card_w - 15) // 2
    img_area_h = card_h - header_h - 10

    try:
        with Image.open(orig_path) as im:
            orig_fitted = fit_image(im, img_area_w, img_area_h)
        with Image.open(overlay_path) as im:
            over_fitted = fit_image(im, img_area_w, img_area_h)
    except Exception as e:
        orig_fitted = Image.new("RGB", (img_area_w, img_area_h), (80, 0, 0))
        over_fitted = Image.new("RGB", (img_area_w, img_area_h), (80, 0, 0))

    # Labels below images
    d_orig = ImageDraw.Draw(orig_fitted)
    d_orig.rectangle([(0, 0), (70, 16)], fill=(0, 0, 0))
    d_orig.text((4, 2), "ORIGINAL", fill=(200, 200, 200))

    d_over = ImageDraw.Draw(over_fitted)
    d_over.rectangle([(0, 0), (85, 16)], fill=(0, 0, 0))
    d_over.text((4, 2), "YOLO PREDICT", fill=(56, 189, 248))

    card.paste(orig_fitted, (5, header_h + 5))
    card.paste(over_fitted, (10 + img_area_w, header_h + 5))

    return card


def generate_class_contact_sheet(group_idx: int, class_name: str, group_items: list, group_preds: list) -> Path:
    card_w = 700
    card_h = 320
    margin = 15
    header_h = 70

    cols = 2
    rows = 5
    sheet_w = cols * card_w + (cols + 1) * margin
    sheet_h = header_h + rows * card_h + (rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)

    # Title Banner
    vi_name = CLASS_NAMES_VI.get(class_name, "")
    draw.rectangle([(0, 0), (sheet_w, header_h)], fill=(30, 41, 59))
    draw.text(
        (25, 15),
        f"NHÓM {group_idx:02d}: {class_name.upper()} ({vi_name}) — 10 ẢNH ỨNG VIÊN WIKIMEDIA COMMONS",
        fill=(255, 255, 255),
    )
    draw.text(
        (25, 42),
        "So sánh trực quan: Ảnh gốc Wikimedia (Trái) vs Dự đoán YOLOv8n (Phải - conf >= 0.25)",
        fill=(148, 163, 184),
    )

    for i in range(len(group_items)):
        item = group_items[i]
        pred = group_preds[i]
        cid = item["candidate_id"]

        orig_path = ORIGINALS_DIR / f"{cid}.jpg"
        over_path = OVERLAYS_DIR / f"{cid}_overlay.jpg"

        card = build_card(orig_path, over_path, item, pred, card_w, card_h)

        r = i // cols
        c = i % cols
        pos_x = margin + c * (card_w + margin)
        pos_y = header_h + margin + r * (card_h + margin)
        sheet.paste(card, (pos_x, pos_y))

    out_file = CONTACT_SHEETS_DIR / f"contact_sheet_{group_idx:02d}_{class_name}.jpg"
    sheet.save(out_file, "JPEG", quality=90)
    print(f"Generated class contact sheet: {out_file.name} ({sheet_w}x{sheet_h})")
    return out_file


def generate_overview_contact_sheet(items: list, preds: list) -> Path:
    """1 representative image per class (10 images total in a 5x2 grid)."""
    card_w = 700
    card_h = 320
    margin = 15
    header_h = 80

    cols = 2
    rows = 5
    sheet_w = cols * card_w + (cols + 1) * margin
    sheet_h = header_h + rows * card_h + (rows + 1) * margin

    sheet = Image.new("RGB", (sheet_w, sheet_h), (18, 18, 22))
    draw = ImageDraw.Draw(sheet)

    draw.rectangle([(0, 0), (sheet_w, header_h)], fill=(15, 23, 42))
    draw.text(
        (25, 15),
        "TỔNG HỢP KIỂM ĐỊNH 10 LỚP RÁC THỰC TẾ — YOLOV8N BENCHMARK OVERVIEW",
        fill=(255, 255, 255),
    )
    draw.text(
        (25, 45),
        "Bộ 100 ảnh kiểm định từ Wikimedia Commons — Minh họa đại diện 10 lớp rác theo quy chuẩn CNTT-KLCN155",
        fill=(148, 163, 184),
    )

    # Pick 1 representative item per group (e.g. item 1 or 2 of each class)
    for g in range(10):
        # find best representative with detections or clean view
        group_items = items[g * 10 : (g + 1) * 10]
        group_preds = preds[g * 10 : (g + 1) * 10]

        # pick first or one with dets
        chosen_idx = 0
        for idx_cand, p in enumerate(group_preds):
            if p.get("num_detections", 0) > 0:
                chosen_idx = idx_cand
                break

        item = group_items[chosen_idx]
        pred = group_preds[chosen_idx]
        cid = item["candidate_id"]

        orig_path = ORIGINALS_DIR / f"{cid}.jpg"
        over_path = OVERLAYS_DIR / f"{cid}_overlay.jpg"
        card = build_card(orig_path, over_path, item, pred, card_w, card_h)

        r = g // cols
        c = g % cols
        pos_x = margin + c * (card_w + margin)
        pos_y = header_h + margin + r * (card_h + margin)
        sheet.paste(card, (pos_x, pos_y))

    out_file = CONTACT_SHEETS_DIR / "contact_sheet_overview_10_classes.jpg"
    sheet.save(out_file, "JPEG", quality=90)
    print(f"Generated overview contact sheet: {out_file.name}")
    return out_file


def main() -> None:
    print(f"Loading data from {AUDIT_JSON} and {PREDICTIONS_JSON}...")
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    with open(PREDICTIONS_JSON, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    items = audit_data["items"]
    preds = pred_data["predictions"]

    items_by_id = {it["candidate_id"]: it for it in items}
    preds_by_id = {p["candidate_id"]: p for p in preds}

    # Generate 10 class contact sheets
    for g in range(10):
        group_items = items[g * 10 : (g + 1) * 10]
        group_preds = [preds_by_id[it["candidate_id"]] for it in group_items]
        cname = group_items[0]["target_class_name"]
        generate_class_contact_sheet(g, cname, group_items, group_preds)

    # Generate overview
    generate_overview_contact_sheet(items, [preds_by_id[it["candidate_id"]] for it in items])
    print("\nAll contact sheets generated successfully!")


if __name__ == "__main__":
    main()
