import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

gt_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

print(f"Total images: {len(gt_data['images'])}")
total_objs = sum(len(img['objects']) for img in gt_data['images'])
print(f"Total GT objects: {total_objs}")

boxes = []
for img in gt_data['images']:
    fn = img['filename']
    w, h = img['width'], img['height']
    img_area = float(w * h)
    for obj in img['objects']:
        box = obj['bbox_xyxy']
        bw = float(box[2] - box[0])
        bh = float(box[3] - box[1])
        box_area = bw * bh
        area_ratio = box_area / img_area
        is_exact_full = (box[0] == 0 and box[1] == 0 and (box[2] == w or box[2] == w - 1 or box[2] >= w * 0.99) and (box[3] == h or box[3] == h - 1 or box[3] >= h * 0.99))
        
        boxes.append({
            'filename': fn,
            'object_id': obj.get('object_id'),
            'class_name': obj.get('class_name'),
            'class_id': obj.get('class_id'),
            'bbox': box,
            'img_w': w,
            'img_h': h,
            'box_w': bw,
            'box_h': bh,
            'area_ratio': area_ratio,
            'area_pct': area_ratio * 100.0,
            'is_exact_full': is_exact_full,
            'rationale': obj.get('review_rationale', '')
        })

# Sort descending by area ratio
boxes.sort(key=lambda x: -x['area_ratio'])

print("\n=== ALL 57 GT BOXES RANKED BY AREA RATIO ===")
for idx, b in enumerate(boxes, 1):
    print(f"{idx:2d}. {b['filename']} | {b['object_id']} | {b['class_name']} | box={b['bbox']} | img=({b['img_w']}x{b['img_h']}) | area={b['area_pct']:.2f}% | full={b['is_exact_full']}")

# Threshold counts
ge_100 = [b for b in boxes if b['area_pct'] >= 99.0]
ge_80 = [b for b in boxes if b['area_pct'] >= 80.0]
ge_50 = [b for b in boxes if b['area_pct'] >= 50.0]
ge_25 = [b for b in boxes if b['area_pct'] >= 25.0]

print(f"\nSummary Counts:")
print(f"Total objects: {len(boxes)}")
print(f"Area >= 99% (Full image canvas): {len(ge_100)}")
for b in ge_100:
    print(f"  - {b['filename']} ({b['object_id']}): {b['area_pct']:.2f}% | {b['class_name']} | {b['bbox']}")

print(f"\nArea >= 80%: {len(ge_80)}")
for b in ge_80:
    print(f"  - {b['filename']} ({b['object_id']}): {b['area_pct']:.2f}% | {b['class_name']} | {b['bbox']}")

print(f"\nArea >= 50%: {len(ge_50)}")
print(f"Area >= 25%: {len(ge_25)}")
