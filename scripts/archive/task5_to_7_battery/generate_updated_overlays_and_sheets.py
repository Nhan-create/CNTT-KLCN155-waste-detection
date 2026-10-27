"""
Generate updated visual overlays and contact sheets for Mini-Batch 01.
Includes exact bounding boxes, status banners, watermarks for quarantined/pending images,
and 5x2 contact sheets.
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def main():
    root = Path(__file__).resolve().parents[3]
    staging = root / "data/acquisition_staging/batch_a_mini01"
    img_dir = staging / "images"
    manifest_path = staging / "manifest.json"
    overlay_dir = root / "reports/source_verification/overlays"
    contact_dir = root / "reports/source_verification/contact_sheets"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    CLASS_COLORS = {
        0: "#e6194B",  # battery: red
        1: "#3cb44b",  # biological: green
        2: "#ffe119",  # cardboard: yellow
        3: "#4363d8",  # clothes: blue
        4: "#f58231",  # glass: orange
        5: "#911eb4",  # metal: purple
        6: "#42d4f4",  # paper: cyan
        7: "#f032e6",  # plastic: magenta
        8: "#bfef45",  # shoes: lime
        9: "#fabed4",  # trash: pink
    }

    try:
        font = ImageFont.truetype("arial.ttf", 36)
        banner_font = ImageFont.truetype("arial.ttf", 44)
        small_font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        banner_font = font
        small_font = font

    annotated_thumbs = []
    raw_thumbs = []

    for item in manifest["items"]:
        cid = item["candidate_id"]
        img_p = img_dir / item["filename"]
        im = Image.open(img_p)
        w, h = im.size
        
        # Raw thumb for contact sheet
        raw_thumb_im = im.copy()
        raw_thumb_im.thumbnail((500, 500))
        canvas_raw = Image.new("RGB", (520, 560), (30, 30, 30))
        paste_x = (520 - raw_thumb_im.width) // 2
        paste_y = 45 + (500 - raw_thumb_im.height) // 2
        canvas_raw.paste(raw_thumb_im, (paste_x, paste_y))
        d_raw = ImageDraw.Draw(canvas_raw)
        d_raw.text((15, 12), f"{cid[:32]}", fill=(240, 240, 240), font=small_font)
        raw_thumbs.append(canvas_raw)
        
        # Annotated overlay
        vis = im.copy()
        draw = ImageDraw.Draw(vis)
        
        # Header banner
        banner_h = max(80, int(h * 0.05))
        draw.rectangle([0, 0, w, banner_h], fill=(20, 20, 20))
        status_text = f"{cid} | QA: {item['visual_qa']} | Context: {item['scene_context']}"
        banner_color = (100, 255, 100) if "QA_PASS" in item["visual_qa"] else (255, 200, 50)
        draw.text((20, 15), status_text, fill=banner_color, font=banner_font)
        
        # Draw boxes
        for b in item["boxes"]:
            cls_id = b["class_id"]
            cls_name = b["class_name"]
            xc, yc, bw, bh = b["yolo_bbox"]
            xmin = int((xc - bw/2) * w)
            xmax = int((xc + bw/2) * w)
            ymin = int((yc - bh/2) * h)
            ymax = int((yc + bh/2) * h)
            col = CLASS_COLORS.get(cls_id, "#00ff00")
            line_w = max(4, int(min(w, h) * 0.005))
            draw.rectangle([xmin, ymin, xmax, ymax], outline=col, width=line_w)
            lbl_tag = f"{cls_id}: {cls_name}"
            bbox_txt = draw.textbbox((xmin, max(banner_h, ymin - 45)), lbl_tag, font=font)
            draw.rectangle([bbox_txt[0]-4, bbox_txt[1]-4, bbox_txt[2]+4, bbox_txt[3]+4], fill=col)
            draw.text((xmin, max(banner_h, ymin - 45)), lbl_tag, fill=(0, 0, 0), font=font)
            
        # Draw informative watermark boxes for non-object scenes
        if "QUARANTINED" in item["visual_qa"]:
            w_box = [int(w*0.08), int(h*0.44), int(w*0.92), int(h*0.56)]
            draw.rectangle(w_box, fill=(180, 0, 0))
            draw.text((w_box[0] + 20, w_box[1] + 25), "QUARANTINED: AMBIGUOUS BATTERY ASSEMBLY", fill=(255, 255, 255), font=banner_font)
        elif "PENDING_ANNOTATION" in item["visual_qa"]:
            w_box = [int(w*0.08), int(h*0.44), int(w*0.92), int(h*0.56)]
            draw.rectangle(w_box, fill=(200, 120, 0))
            draw.text((w_box[0] + 20, w_box[1] + 25), "PENDING ANNOTATION: MULTI-OBJECT COMPOST BIN", fill=(255, 255, 255), font=banner_font)
        elif "CLEAN_NEGATIVE" in item["visual_qa"]:
            w_box = [int(w*0.08), int(h*0.44), int(w*0.92), int(h*0.56)]
            draw.rectangle(w_box, fill=(0, 100, 160))
            draw.text((w_box[0] + 20, w_box[1] + 25), "CLEAN NEGATIVE: 0 LITTER (BACKGROUND TEXTURE)", fill=(255, 255, 255), font=banner_font)
            
        out_overlay_path = overlay_dir / f"{cid}_annotated.jpg"
        vis.save(out_overlay_path, quality=92)
        print(f"Saved overlay: {out_overlay_path.name}")
        
        # Annotated thumb for contact sheet
        ann_thumb_im = vis.copy()
        ann_thumb_im.thumbnail((500, 500))
        canvas_ann = Image.new("RGB", (520, 560), (30, 30, 30))
        paste_x = (520 - ann_thumb_im.width) // 2
        paste_y = 45 + (500 - ann_thumb_im.height) // 2
        canvas_ann.paste(ann_thumb_im, (paste_x, paste_y))
        d_ann = ImageDraw.Draw(canvas_ann)
        d_ann.text((15, 12), f"{cid[:28]} ({len(item['boxes'])}b)", fill=(240, 240, 240), font=small_font)
        annotated_thumbs.append(canvas_ann)

    # 5x2 contact sheets
    grid_w = 520 * 5
    grid_h = 560 * 2
    sheet_raw = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
    sheet_ann = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))

    for idx in range(10):
        gx = (idx % 5) * 520
        gy = (idx // 5) * 560
        sheet_raw.paste(raw_thumbs[idx], (gx, gy))
        sheet_ann.paste(annotated_thumbs[idx], (gx, gy))

    sheet_raw.save(contact_dir / "minibatch01_contact_sheet.jpg", quality=90)
    sheet_ann.save(contact_dir / "minibatch01_annotated_contact_sheet.jpg", quality=90)
    print("Successfully regenerated contact sheets (raw & annotated).")

if __name__ == "__main__":
    main()
