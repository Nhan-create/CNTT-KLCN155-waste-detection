import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

coco_path = Path("data/audit/taco_annotations_raw.json")
with open(coco_path, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

img_112 = next((img for img in coco_data['images'] if img['id'] == 112), None)
img_457 = next((img for img in coco_data['images'] if img['id'] == 457), None)

print(f"Image 112 in COCO: {img_112}")
print(f"Image 457 in COCO: {img_457}")

annos_112 = [a for a in coco_data['annotations'] if a['image_id'] == 112]
annos_457 = [a for a in coco_data['annotations'] if a['image_id'] == 457]

print(f"\nImage 112 has {len(annos_112)} annotations in COCO:")
for idx, a in enumerate(annos_112, 1):
    bbox = a['bbox'] # [x, y, w, h]
    x, y, w, h = bbox
    tw, th = img_112['width'], img_112['height']
    xc = (x + w / 2.0) / tw
    yc = (y + h / 2.0) / th
    bw = w / tw
    bh = h / th
    y1_norm = yc - bh / 2.0
    print(f"  [{idx:2d}] id={a['id']} cat={a['category_id']:2d} | COCO bbox=[{x:7.2f}, {y:7.2f}, {w:7.2f}, {h:7.2f}] | y1_norm={y1_norm:10.6f}")

print(f"\nImage 457 has {len(annos_457)} annotations in COCO:")
for idx, a in enumerate(annos_457, 1):
    bbox = a['bbox']
    x, y, w, h = bbox
    tw, th = img_457['width'], img_457['height']
    xc = (x + w / 2.0) / tw
    yc = (y + h / 2.0) / th
    bw = w / tw
    bh = h / th
    y1_norm = yc - bh / 2.0
    print(f"  [{idx:2d}] id={a['id']} cat={a['category_id']:2d} | COCO bbox=[{x:7.2f}, {y:7.2f}, {w:7.2f}, {h:7.2f}] | y1_norm={y1_norm:10.6f}")
