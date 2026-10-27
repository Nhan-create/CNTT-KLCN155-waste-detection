# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Kiểm toán nguồn gốc, cụm pHash và tính toàn vẹn dữ liệu cho `audit_taco_coco_provenance`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

coco_path = Path("data/audit/taco_annotations_raw.json")
with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

print(f"Loaded COCO annotations from {coco_path}")
print(f"Total images in COCO: {len(coco_data['images'])}")
print(f"Total annotations in COCO: {len(coco_data['annotations'])}")

# Build image lookup by filename or id
img_by_id = {img['id']: img for img in coco_data['images']}
img_by_fn = {Path(img['file_name']).name: img for img in coco_data['images']}

# Check target images
for name in ["000112", "000457", "taco_0112", "taco_0457", "batch_1/000112", "batch_7/000057"]:
    for k, img in img_by_id.items():
        fn = img['file_name']
        if "112" in fn or "457" in fn or name in fn:
            print(f"Image match: id={k}, file_name={fn}, w={img['width']}, h={img['height']}")

# Find annotations for the images matching 112 and 457
target_img_ids = []
for k, img in img_by_id.items():
    fn = img['file_name']
    # Check if this image corresponds to taco_0112 or taco_0457
    if fn.endswith("/000112.jpg") or fn.endswith("000112.jpg") or fn.endswith("/000457.jpg") or fn.endswith("000457.jpg"):
        target_img_ids.append((k, fn, img['width'], img['height']))

print("\n--- Target TACO Images Found in COCO ---")
for tid, tfn, tw, th in target_img_ids:
    print(f"ID {tid}: {tfn} ({tw}x{th})")
    annos = [a for a in coco_data['annotations'] if a['image_id'] == tid]
    print(f"  Annotations count: {len(annos)}")
    for idx, a in enumerate(annos, 1):
        bbox = a['bbox'] # [x, y, width, height] in COCO
        cat_id = a['category_id']
        x, y, w, h = bbox
        xc = (x + w / 2.0) / tw
        yc = (y + h / 2.0) / th
        bw = w / tw
        bh = h / th
        x1_norm = xc - bw / 2.0
        y1_norm = yc - bh / 2.0
        x2_norm = xc + bw / 2.0
        y2_norm = yc + bh / 2.0
        print(f"    [{idx}] cat={cat_id} | COCO bbox=[{x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}]")
        print(f"         YOLO norm=[{xc:.6f}, {yc:.6f}, {bw:.6f}, {bh:.6f}] | y1_norm={y1_norm:.6f}")
