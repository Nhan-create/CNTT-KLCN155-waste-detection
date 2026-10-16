import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "detection"
SPLITS_DIR = DATA_DIR / "splits_real_v2"

print("=== FULL AUTOMATED DATASET INTEGRITY AUDIT (real_v2) ===")

splits = ["train.txt", "val.txt", "test.txt"]
all_split_images = {}
for sp in splits:
    with open(SPLITS_DIR / sp, "r", encoding="utf-8") as f:
        imgs = [l.strip() for l in f if l.strip()]
        all_split_images[sp] = imgs
        print(f"Loaded {len(imgs)} images from {sp}")

total_images = sum(len(v) for v in all_split_images.values())
print(f"Total split images: {total_images}")

# Tracking issues
unreadable_images = []
missing_labels = []
empty_labels = []
malformed_format_lines = []
invalid_class_ids = []
zero_or_negative_dim_boxes = []
out_of_bounds_boxes = []
duplicate_boxes_in_image = []

# For cross-split leakage
image_sha256_by_split = defaultdict(dict)
image_sha256_to_paths = defaultdict(list)

epsilon = 1e-6
total_boxes = 0

for sp, img_list in all_split_images.items():
    for rel_path in img_list:
        img_p = DATA_DIR / rel_path
        
        # 1. Check image readable & compute hash
        if not img_p.exists():
            unreadable_images.append((str(rel_path), sp, "FILE_NOT_FOUND"))
            continue
            
        try:
            with Image.open(img_p) as im:
                im.verify()
            # reopen for size
            with Image.open(img_p) as im:
                w, h = im.size
        except Exception as e:
            unreadable_images.append((str(rel_path), sp, f"CORRUPT_IMAGE: {e}"))
            continue
            
        with open(img_p, "rb") as f:
            hsh = hashlib.sha256(f.read()).hexdigest()
        image_sha256_by_split[sp][hsh] = rel_path
        image_sha256_to_paths[hsh].append((sp, rel_path))
        
        # 2. Check label file
        parts = list(Path(rel_path).parts)
        if parts[0] == "images":
            parts[0] = "labels"
        lbl_rel = Path(*parts).with_suffix(".txt")
        lbl_p = DATA_DIR / lbl_rel
        
        if not lbl_p.exists():
            missing_labels.append((str(lbl_rel), sp, "LABEL_MISSING"))
            continue
            
        with open(lbl_p, "r", encoding="utf-8") as lf:
            raw_lines = [l.strip() for l in lf]
            
        lines = [l for l in raw_lines if l]
        if len(lines) == 0:
            empty_labels.append((str(lbl_rel), sp))
            continue
            
        seen_boxes = set()
        for line_idx, l in enumerate(lines, 1):
            toks = l.split()
            if len(toks) != 5:
                malformed_format_lines.append((str(lbl_rel), line_idx, l, f"Expected 5 tokens, got {len(toks)}"))
                continue
                
            try:
                cid_val = float(toks[0])
                cid = int(cid_val)
                xc, yc, bw, bh = float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
            except ValueError as e:
                malformed_format_lines.append((str(lbl_rel), line_idx, l, f"Non-numeric value: {e}"))
                continue
                
            total_boxes += 1
            
            # Check class ID
            if cid_val != cid or not (0 <= cid <= 9):
                invalid_class_ids.append((str(lbl_rel), line_idx, cid, "Class ID outside [0..9]"))
                
            # Check dimensions > 0
            if bw <= 0 or bh <= 0:
                zero_or_negative_dim_boxes.append((str(lbl_rel), line_idx, bw, bh, "Width or height <= 0"))
                
            # Check center within [0, 1]
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                out_of_bounds_boxes.append((str(lbl_rel), line_idx, (xc, yc, bw, bh), "Center outside [0, 1]"))
                
            # Check bounding box bounds
            x1 = xc - bw / 2.0
            y1 = yc - bh / 2.0
            x2 = xc + bw / 2.0
            y2 = yc + bh / 2.0
            
            if x1 < -epsilon or y1 < -epsilon or x2 > 1.0 + epsilon or y2 > 1.0 + epsilon:
                out_of_bounds_boxes.append((str(lbl_rel), line_idx, (x1, y1, x2, y2), f"BBox out of bounds: x1={x1:.6f}, y1={y1:.6f}, x2={x2:.6f}, y2={y2:.6f}"))
                
            # Check duplicate boxes in same file
            box_tuple = (cid, round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6))
            if box_tuple in seen_boxes:
                duplicate_boxes_in_image.append((str(lbl_rel), line_idx, box_tuple, "DUPLICATE_BOX_IN_SAME_IMAGE"))
            seen_boxes.add(box_tuple)

# Check cross-split leakage
cross_split_leakages = []
for hsh, occurrences in image_sha256_to_paths.items():
    distinct_splits = set(occ[0] for occ in occurrences)
    if len(distinct_splits) > 1:
        cross_split_leakages.append((hsh, occurrences))

print("\n=== AUDIT RESULTS SUMMARY ===")
print(f"Total images scanned: {total_images}")
print(f"Total boxes scanned: {total_boxes}")
print(f"1. Unreadable/Corrupt images: {len(unreadable_images)}")
print(f"2. Missing label files: {len(missing_labels)}")
print(f"3. Empty labels (Negative images): {len(empty_labels)}")
print(f"4. Malformed label lines: {len(malformed_format_lines)}")
print(f"5. Invalid Class IDs: {len(invalid_class_ids)}")
print(f"6. Zero or negative dimension boxes: {len(zero_or_negative_dim_boxes)}")
print(f"7. Out-of-bounds boxes (tolerance 1e-6): {len(out_of_bounds_boxes)}")
for ob in out_of_bounds_boxes[:10]:
    print(f"   - {ob[0]}:{ob[1]} -> {ob[3]}")
print(f"8. Duplicate boxes in same image: {len(duplicate_boxes_in_image)}")
for db in duplicate_boxes_in_image[:10]:
    print(f"   - {db[0]}:{db[1]} -> {db[2]}")
print(f"9. Exact SHA-256 cross-split leakage: {len(cross_split_leakages)}")

# Save detailed JSON report
report_path = PROJECT_ROOT / "reports" / "DATASET_INTEGRITY_AUDIT_REPORT.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_images": total_images,
        "total_boxes": total_boxes,
        "unreadable_images_count": len(unreadable_images),
        "missing_labels_count": len(missing_labels),
        "empty_labels_count": len(empty_labels),
        "malformed_lines_count": len(malformed_format_lines),
        "invalid_class_ids_count": len(invalid_class_ids),
        "zero_or_negative_dims_count": len(zero_or_negative_dim_boxes),
        "out_of_bounds_count": len(out_of_bounds_boxes),
        "duplicate_boxes_count": len(duplicate_boxes_in_image),
        "cross_split_leakage_count": len(cross_split_leakages),
        "unreadable_images": unreadable_images,
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "malformed_format_lines": malformed_format_lines,
        "invalid_class_ids": invalid_class_ids,
        "zero_or_negative_dim_boxes": zero_or_negative_dim_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "duplicate_boxes_in_image": duplicate_boxes_in_image,
        "cross_split_leakages": cross_split_leakages
    }, f, indent=2, ensure_ascii=False)

print(f"\nDetailed report saved to: {report_path}")
