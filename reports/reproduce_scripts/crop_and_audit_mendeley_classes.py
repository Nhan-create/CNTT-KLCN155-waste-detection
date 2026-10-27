# Mô tả: Cắt ảnh từng vật thể mẫu theo từng lớp của nguồn Mendeley để phục vụ kiểm toán trực quan ánh xạ nhãn.
"""
Crop sample objects for each Mendeley class (0, 1, 2, 3, 4, 5, 6, 8, 9)
with exact provenance:
- actual_stem: Real image stem without loop-variable leakage
- img_path & img_sha256: Path and SHA-256 of the source image
- mendeley_label_path & line_number: Annotation source file and exact line number
- raw_line: Original annotation line text
- v1_label_path: Corresponding dataset-v1 label path
- bbox & crop_size: Bounding box coordinates and pixel crop size

Also outputs explicit stats for matched, unmatched, and duplicate files.
"""

import hashlib
import json
from pathlib import Path
from PIL import Image

mendeley_base = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "waste-training"))) / "mendeley-labels")
v1_img_base = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "waste-training"))) / "dataset-v1/images")
v1_lbl_base = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[2] / "data" / "external" / "waste-training"))) / "dataset-v1/labels")
output_crops_dir = (Path(__file__).resolve().parents[2] / "reports/mendeley_crops")
output_crops_dir.mkdir(parents=True, exist_ok=True)

# 1. Build index of v1 images
v1_images = {}
for split in ["train", "val", "test"]:
    p = v1_img_base / split
    if p.exists():
        for img_p in p.glob("syn_*.jpg"):
            stem = img_p.stem
            if stem.startswith("syn_syn_"):
                clean_stem = "syn_" + stem[8:]
            else:
                clean_stem = stem
            v1_images[clean_stem] = img_p

# Also index v1 labels
v1_labels = {}
for split in ["train", "val", "test"]:
    p = v1_lbl_base / split
    if p.exists():
        for lbl_p in p.glob("syn_*.txt"):
            stem = lbl_p.stem
            if stem.startswith("syn_syn_"):
                clean_stem = "syn_" + stem[8:]
            else:
                clean_stem = stem
            v1_labels[clean_stem] = lbl_p

# 2. Collect boxes per Mendeley class and record match statistics
boxes_by_class = {c: [] for c in range(10)}
matched_files = set()
unmatched_files = set()
duplicate_files = set()
seen_stems = set()

for split in ["train", "val"]:
    for f in sorted((mendeley_base / split).glob("syn_*.txt")):
        stem = f.stem
        if stem in seen_stems:
            duplicate_files.add(stem)
        seen_stems.add(stem)

        if stem not in v1_images:
            unmatched_files.add(stem)
            continue
        
        matched_files.add(stem)
        img_p = v1_images[stem]
        v1_lbl_p = v1_labels.get(stem)
        
        with open(f, "r", encoding="utf-8") as lf:
            for lidx, line in enumerate(lf):
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                area = bbox[2] * bbox[3]
                boxes_by_class[cls_id].append({
                    "orig_class": cls_id,
                    "stem": stem,
                    "mendeley_label_path": str(f),
                    "line_number": lidx + 1,
                    "raw_line": line.strip(),
                    "img_path": str(img_p),
                    "v1_label_path": str(v1_lbl_p) if v1_lbl_p else "",
                    "bbox": bbox,
                    "area": area
                })

print(f"=== Provenance & Matching Statistics ===")
print(f"Total synthetic files indexed in dataset-v1: {len(v1_images)}")
print(f"Matched Mendeley files: {len(matched_files)}")
print(f"Unmatched Mendeley files: {len(unmatched_files)}")
print(f"Duplicate Mendeley file stems: {len(duplicate_files)}")

# 3. Compute sha256 cache
sha_cache = {}
def get_sha256(filepath):
    if filepath not in sha_cache:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha_cache[filepath] = h.hexdigest()
    return sha_cache[filepath]

# 4. Generate crops: 8 largest per class for high visibility
manifest = []
for c in range(10):
    c_boxes = boxes_by_class[c]
    if not c_boxes:
        print(f"  Class {c}: 0 boxes (Omitted/Empty)")
        continue
    print(f"  Class {c}: {len(c_boxes)} boxes total")
    # Sort by area descending
    c_boxes_sorted = sorted(c_boxes, key=lambda x: x["area"], reverse=True)
    for sample_idx, item in enumerate(c_boxes_sorted[:8]):
        actual_stem = item["stem"]
        img_path = Path(item["img_path"])
        img_sha = get_sha256(img_path)
        
        xc, yc, w, h = item["bbox"]
        with Image.open(img_path) as img:
            W, H = img.size
            x1 = max(0, int((xc - w/2) * W))
            y1 = max(0, int((yc - h/2) * H))
            x2 = min(W, int((xc + w/2) * W))
            y2 = min(H, int((yc + h/2) * H))
            
            if x2 > x1 and y2 > y1:
                crop = img.crop((x1, y1, x2, y2))
                crop_name = f"mendeley_c{c}_sample{sample_idx}_{actual_stem}.jpg"
                crop_file = output_crops_dir / crop_name
                crop.save(crop_file, quality=95)
                
                manifest.append({
                    "orig_class": c,
                    "sample_idx": sample_idx,
                    "actual_stem": actual_stem,
                    "crop_filename": crop_name,
                    "crop_path": str(crop_file),
                    "img_path": str(img_path),
                    "img_sha256": img_sha,
                    "mendeley_label_path": item["mendeley_label_path"],
                    "line_number": item["line_number"],
                    "raw_line": item["raw_line"],
                    "v1_label_path": item["v1_label_path"],
                    "bbox": [xc, yc, w, h],
                    "crop_size": [x2 - x1, y2 - y1]
                })

manifest_path = output_crops_dir / "crop_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n[DONE] Saved {len(manifest)} crops with verified stems and full provenance to {output_crops_dir}")
