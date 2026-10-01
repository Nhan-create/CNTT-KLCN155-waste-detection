"""
scripts/audit_real_detection_images.py
--------------------------------------
Audits all 114 OpenImages real images currently in data/detection/images/real/.
Correlates with original OpenImages V7 validation annotations to identify:
1. People, clothing, and body parts present in the image.
2. Context: Worn shoes vs discarded waste.
3. Missing annotations (specifically Clothing - Class 3).
4. Generates a per-image audit CSV and visual contact sheet.
"""

import os
import sys
import csv
import glob
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

def main():
    project_root = Path(__file__).resolve().parent.parent
    real_img_dir = project_root / "data" / "detection" / "images" / "real"
    real_lbl_dir = project_root / "data" / "detection" / "labels" / "real"
    oi_box_csv = Path(r"D:\waste-training\openimages-validation-boxes.csv")
    oi_classes_csv = Path(r"D:\waste-training\openimages-boxable-classes.csv")
    out_dir = project_root / "artifacts" / "part02" / "real_data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Auditing real images in: {real_img_dir}")
    img_files = sorted(list(real_img_dir.glob("*.jpg")))
    print(f"[INFO] Found {len(img_files)} real images.")

    # Load OpenImages class dictionary
    class_desc = {}
    if oi_classes_csv.exists():
        with open(oi_classes_csv, mode="r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    class_desc[row[0]] = row[1]

    # Map image ID -> OpenImages full annotations
    oi_boxes_by_id = {}
    if oi_box_csv.exists():
        with open(oi_box_csv, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                img_id = row[0]
                cls_code = row[2]
                name = class_desc.get(cls_code, cls_code)
                if img_id not in oi_boxes_by_id:
                    oi_boxes_by_id[img_id] = []
                oi_boxes_by_id[img_id].append(name)

    audit_records = []
    verdict_counter = Counter()

    for img_path in img_files:
        stem = img_path.stem
        oi_id = stem.replace("oi_", "")
        lbl_path = real_lbl_dir / f"{stem}.txt"

        # Count project boxes
        proj_boxes = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        proj_boxes.append(int(parts[0]))

        oi_tags = oi_boxes_by_id.get(oi_id, [])
        has_person = any(t in ["Person", "Man", "Woman", "Girl", "Human body", "Human leg"] for t in oi_tags)
        has_clothing = any(t in ["Clothing", "Dress", "Pants", "Trousers", "Jeans", "Coat", "Suit", "Jacket"] for t in oi_tags)
        has_footwear = any(t in ["Footwear", "Boot", "Sandal"] for t in oi_tags)

        # Determine verdict
        if has_person or has_clothing:
            verdict = "UNSUITABLE_SHOES_BEING_WORN"
            notes = f"Contains human/clothing ({', '.join(set(oi_tags) & {'Person', 'Man', 'Woman', 'Girl', 'Clothing'})}). Shoes are actively worn by people, not discarded waste. Missing Class 3 (clothes) annotations."
        else:
            verdict = "REQUIRES_MANUAL_INSPECTION"
            notes = f"No explicit person/clothing tag in OpenImages metadata. Tags: {', '.join(list(set(oi_tags))[:5])}."

        verdict_counter[verdict] += 1
        audit_records.append({
            "image_id": stem,
            "filename": img_path.name,
            "project_boxes_count": len(proj_boxes),
            "project_classes": list(set(proj_boxes)),
            "openimages_tags_count": len(oi_tags),
            "has_person_detected": has_person,
            "has_clothing_detected": has_clothing,
            "suitability_verdict": verdict,
            "audit_notes": notes
        })

    # Save audit CSV
    audit_csv_path = out_dir / "real_images_audit_table.csv"
    with open(audit_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "filename", "project_boxes_count", "project_classes",
            "openimages_tags_count", "has_person_detected", "has_clothing_detected",
            "suitability_verdict", "audit_notes"
        ])
        writer.writeheader()
        writer.writerows(audit_records)

    print(f"\n[DONE] Audit Table saved to: {audit_csv_path}")
    print("\n--- Summary of 114 Real Images Suitability ---")
    for v, c in verdict_counter.items():
        print(f"  {v:<35}: {c} / {len(img_files)} ({c/len(img_files)*100:.1f}%)")

    # Generate Contact Sheets (4x4 grid of images)
    print("\n[INFO] Generating visual contact sheet for PM review...")
    contact_sheet_path = out_dir / "real_images_contact_sheet_sample.jpg"
    thumb_w, thumb_h = 240, 180
    cols, rows = 4, 4
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    sample_imgs = img_files[:16]
    for idx, p in enumerate(sample_imgs):
        r_idx = idx // cols
        c_idx = idx % cols
        x_off = c_idx * thumb_w
        y_off = r_idx * thumb_h

        with Image.open(p) as im:
            thumb = im.convert("RGB").resize((thumb_w, thumb_h - 25))
            sheet.paste(thumb, (x_off, y_off))

        record = audit_records[idx]
        tag_text = f"{record['image_id'][:12]}.. | {'WORN' if record['has_person_detected'] else 'INSPECT'}"
        draw.rectangle([x_off, y_off + thumb_h - 25, x_off + thumb_w, y_off + thumb_h], fill=(0, 0, 0))
        draw.text((x_off + 5, y_off + thumb_h - 20), tag_text, fill=(255, 200, 0))

    sheet.save(contact_sheet_path, quality=90)
    print(f"[INFO] Contact sheet saved to: {contact_sheet_path}")

if __name__ == "__main__":
    main()
