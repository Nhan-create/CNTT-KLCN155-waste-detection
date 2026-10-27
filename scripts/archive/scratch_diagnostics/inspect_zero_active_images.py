# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Kiểm tra chi tiết dữ liệu, khung bao và thuộc tính của `zero active images`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

v2_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json")
with open(v2_path, "r", encoding="utf-8") as f:
    v2_data = json.load(f)

zero_active_images = []
for img in v2_data['images']:
    fn = img['filename']
    active_objs = [o for o in img['objects'] if o.get('is_active_for_instance_eval', False)]
    if len(active_objs) == 0:
        zero_active_images.append((fn, img['objects']))

print(f"Total images with zero active objects in GT v2: {len(zero_active_images)}")
for fn, objs in zero_active_images:
    print(f"\n* Image: {fn}")
    for o in objs:
        print(f"    [{o['object_id']}] class={o['class_name']} | status={o['v2_status']} | action={o['v2_action']}")
        print(f"         note: {o.get('v2_audit_note')}")
