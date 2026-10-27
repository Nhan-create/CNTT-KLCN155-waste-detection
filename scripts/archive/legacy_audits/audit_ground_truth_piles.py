# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Kiểm toán nguồn gốc, cụm pHash và tính toàn vẹn dữ liệu cho `audit_ground_truth_piles`.
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

large_boxes = []
mixed_piles = []

for img in gt_data['images']:
    fn = img['filename']
    w, h = img['width'], img['height']
    img_area = w * h
    
    for obj in img['objects']:
        box = obj['bbox_xyxy']
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        box_area = bw * bh
        area_ratio = box_area / img_area
        rationale = obj.get('review_rationale', '')
        cname = obj.get('class_name', '')
        
        # Check large box
        if area_ratio >= 0.25:
            large_boxes.append({
                'filename': fn,
                'object_id': obj.get('object_id'),
                'class_name': cname,
                'bbox': box,
                'area_ratio': area_ratio,
                'rationale': rationale
            })
            
        # Check mixed / pile keywords in rationale
        keywords = ['đống', 'hỗn hợp', 'bãi', 'nhiều', 'trải', 'toàn bộ', 'pile', 'mixed', 'dump']
        if any(k in rationale.lower() for k in keywords):
            mixed_piles.append({
                'filename': fn,
                'object_id': obj.get('object_id'),
                'class_name': cname,
                'bbox': box,
                'area_ratio': area_ratio,
                'rationale': rationale
            })

print(f"\n--- LARGE BOUNDING BOXES (Area >= 25% of image) [Total: {len(large_boxes)}] ---")
for lb in sorted(large_boxes, key=lambda x: -x['area_ratio']):
    print(f"* {lb['filename']} [{lb['object_id']}] ({lb['class_name']}): {lb['area_ratio']*100:.1f}% img area | box={lb['bbox']}")
    print(f"  Rationale: {lb['rationale']}")

print(f"\n--- MIXED / AGGREGATE WASTE PILES IN GT [Total: {len(mixed_piles)}] ---")
for mp in mixed_piles:
    print(f"* {mp['filename']} [{mp['object_id']}] ({mp['class_name']}): area={mp['area_ratio']*100:.1f}% | box={mp['bbox']}")
    print(f"  Rationale: {mp['rationale']}")
