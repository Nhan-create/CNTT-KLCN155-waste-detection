# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Kiểm tra chi tiết dữ liệu, khung bao và thuộc tính của `all gt objects`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from PIL import Image

gt_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
img_dir = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[3] / "data" / "holdout")))

with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

classes_count = {}
for img in gt_data['images']:
    fn = img['filename']
    img_path = img_dir / fn
    exists = img_path.exists()
    actual_size = None
    if exists:
        try:
            with Image.open(img_path) as im:
                actual_size = im.size
        except Exception as e:
            actual_size = str(e)
            
    w, h = img['width'], img['height']
    size_match = (actual_size == (w, h)) if actual_size else False
    
    print(f"\n=======================================================")
    print(f"IMAGE: {fn} | JSON size: {w}x{h} | Disk size: {actual_size} (Match: {size_match})")
    print(f"Objects count: {len(img['objects'])}")
    
    for obj in img['objects']:
        oid = obj.get('object_id')
        cname = obj.get('class_name')
        cid = obj.get('class_id')
        box = obj['bbox_xyxy']
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        area = bw * bh
        area_pct = (area / (w * h)) * 100.0
        rat = obj.get('review_rationale', '')
        classes_count[cname] = classes_count.get(cname, 0) + 1
        
        print(f"  [{oid}] {cname} (id={cid}) | box={box} (w={bw}, h={bh}) | area={area_pct:.1f}%")
        print(f"       Rationale: {rat}")

print("\n=======================================================")
print("CLASS DISTRIBUTION IN GT (57 objects):")
for c, cnt in sorted(classes_count.items(), key=lambda x: -x[1]):
    print(f"  {c:12s}: {cnt:2d} ({cnt/57.0*100.1:.1f}%)")
