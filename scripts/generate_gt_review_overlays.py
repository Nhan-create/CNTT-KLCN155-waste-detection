import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
import cv2
import numpy as np

gt_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
img_dir = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[1] / "data" / "holdout")))
out_overlay_dir = Path("reports/qa_acceptance/gt_review_overlays")
out_overlay_dir.mkdir(parents=True, exist_ok=True)

with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

# Colors in BGR
COLOR_CRITICAL = (0, 0, 255)     # Red (Must fix / severe flaw)
COLOR_HIGH = (0, 140, 255)       # Orange (Large pile / texture carpet)
COLOR_AMBIGUOUS = (0, 255, 255)  # Yellow (Overlap / nesting / taxonomy edge case)
COLOR_VALID = (0, 255, 0)        # Green (Clear discrete object)

suspicious_table = []
all_annotations_analyzed = []

for img_info in gt_data['images']:
    fn = img_info['filename']
    w, h = img_info['width'], img_info['height']
    img_area = float(w * h)
    img_path = img_dir / fn
    
    if not img_path.exists():
        continue
        
    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        continue
        
    overlay = img_cv.copy()
    objects = img_info['objects']
    num_objs = len(objects)
    
    for obj in objects:
        oid = obj.get('object_id')
        cname = obj.get('class_name')
        cid = obj.get('class_id')
        box = obj['bbox_xyxy']
        bw = float(box[2] - box[0])
        bh = float(box[3] - box[1])
        area = bw * bh
        area_pct = (area / img_area) * 100.0
        rationale = obj.get('review_rationale', '')
        
        severity = "VALID"
        color = COLOR_VALID
        issue_category = "VALID_OBJECT"
        issue_desc = "Annotation rõ ràng, vật thể rác đơn lẻ hợp lệ."
        proposed_fix = "Giữ nguyên"
        certainty = "Cao"
        
        # Rule 1: 100% canvas box
        if area_pct >= 99.0:
            severity = "CRITICAL"
            color = COLOR_CRITICAL
            if num_objs > 1:
                issue_category = "DUPLICATE_FULL_CANVAS_OVER_INDIVIDUALS"
                issue_desc = f"Box chiếm 100% diện tích ảnh ({w}x{h}), bao trùm trùng đè lên các vật thể đơn lẻ đã được gắn nhãn chi tiết bên trong."
                proposed_fix = "Xóa bỏ box toàn ảnh; chỉ giữ lại các box vật thể đơn lẻ bên trong."
                certainty = "Tuyệt đối (100%)"
            else:
                issue_category = "FULL_CANVAS_UNSEGREGATED_PILE"
                issue_desc = f"Box chiếm 100% diện tích ảnh ({w}x{h}), đóng khung toàn bộ frame chứa đống rác cận cảnh/hỗn hợp không phân tách biên vật thể."
                proposed_fix = "Không phù hợp object detection; chuyển thành classification hoặc chia nhỏ thành bounding box từng vật thể cụ thể."
                certainty = "Rất cao (95%)"
                
        # Rule 2: Municipal utility container mislabeled as waste
        elif fn in ["images (18).jpg", "images (8).jpg"]:
            severity = "CRITICAL"
            color = COLOR_CRITICAL
            issue_category = "UTILITY_CONTAINER_MISLABELED_AS_WASTE"
            if fn == "images (18).jpg":
                issue_desc = f"Thùng rác công cộng 2 bánh xe màu xanh lá cây ({area_pct:.1f}% diện tích) bị gán nhãn thành rác plastic cần thu gom."
                proposed_fix = "Quarantine khỏi GT rác thải; xếp vào class 'trash_bin' (hạ tầng đô thị) hoặc đưa vào Negative background."
                certainty = "Tuyệt đối (100%)"
            else:
                issue_desc = f"Sọt/giỏ đựng rác văn phòng đục lỗ màu xanh lam ({area_pct:.1f}% diện tích) bị gán nhãn thành rác plastic cần thu gom."
                proposed_fix = "Quarantine khỏi GT rác thải; xếp vào class 'trash_bin' hoặc đưa vào Negative background."
                certainty = "Tuyệt đối (100%)"
                
        # Rule 3: Texture carpet / massive area >= 80%
        elif area_pct >= 80.0:
            severity = "HIGH"
            color = COLOR_HIGH
            if oid == "img01_obj07":
                severity = "CRITICAL"
                color = COLOR_CRITICAL
                issue_category = "TEXTURE_CARPET_OVER_OBJECTS"
                issue_desc = f"Box thảm vụn kính vỡ phủ toàn bộ nền cát ({area_pct:.1f}%), trong khi 4 chai/hũ thủy tinh đơn lẻ đã được đánh dấu bên trong."
                proposed_fix = "Xóa bỏ box thảm bề mặt img01_obj07; giữ lại các chai/hũ thủy tinh đơn lẻ."
                certainty = "Tuyệt đối (100%)"
            elif fn == "photo-5-1642556540067542343380.webp":
                issue_category = "MASSIVE_LANDFILL_PILE"
                issue_desc = f"Box ngọn núi bãi rác quần áo dệt may khổng lồ chiếm {area_pct:.1f}% diện tích frame."
                proposed_fix = "Đánh dấu dạng waste_pile hoặc chia thành các cụm vật thể xác định thay vì đóng khung toàn cảnh."
                certainty = "Cao"
            elif fn == "images (19).jpg":
                issue_category = "COMPOST_PILE"
                issue_desc = f"Đống rác hữu cơ ủ phân compost chiếm {area_pct:.1f}% diện tích."
                proposed_fix = "Xem xét giữ lại nếu chấp nhận class biological_pile, hoặc khoanh vùng lõi rác hữu cơ thực tế."
                certainty = "Trung bình"
                
        # Rule 4: Overlapping / nested box in same frame
        elif fn == "images (12).jpg" and oid == "img06_obj01":
            severity = "AMBIGUOUS"
            color = COLOR_AMBIGUOUS
            issue_category = "CONTAINER_OBJECT_NESTING"
            issue_desc = f"Thùng carton ({area_pct:.1f}%) bao chứa toàn bộ tập hồ sơ giấy bên trong ({objects[1]['class_name']} 42.2%). Hai box lồng nhau."
            proposed_fix = "Chuẩn hóa quy ước: chỉ gán nhãn vật thể chứa (thùng carton) hoặc phân ranh giới rõ giữa vỏ hộp và ruột giấy."
            certainty = "Cao"
            
        elif fn == "images (10).jpg" and oid == "img04_obj01":
            severity = "AMBIGUOUS"
            color = COLOR_AMBIGUOUS
            issue_category = "NESTED_OVERLAPPING_CLOTHES"
            issue_desc = f"Box quần áo lớn ({area_pct:.1f}%) bao trùm lên box quần áo nhỏ ({objects[1]['bbox_xyxy']}, 22.3%) trong cùng 1 khối vải."
            proposed_fix = "Quy ước thống nhất: gán nhãn 1 box duy nhất cho cả khối hoặc tách rời từng bộ phận không chồng lấn."
            certainty = "Cao"
            
        elif fn == "OIP (1).webp" and oid == "img20_obj01":
            severity = "AMBIGUOUS"
            color = COLOR_AMBIGUOUS
            issue_category = "OVERLAPPING_PILES"
            issue_desc = f"Hai box đống quần áo sa mạc chồng lấn nhau ({area_pct:.1f}% và {objects[1]['bbox_xyxy']})."
            proposed_fix = "Điều chỉnh tọa độ để tách biệt 2 gò rác sa mạc rõ ràng."
            certainty = "Trung bình"
            
        # Rule 5: Taxonomy boundary edge case
        elif fn == "Phan-Loai-Rac-Tai-Ng-01.jpg" and oid == "img22_obj04":
            severity = "AMBIGUOUS"
            color = COLOR_AMBIGUOUS
            issue_category = "COMPOSITE_MATERIAL_TAXONOMY"
            issue_desc = "Vỏ hộp sữa TH true MILK (Tetra Pak) là vật liệu phức hợp (giấy + nhôm + màng PE), hiện đang gán nhãn cardboard."
            proposed_fix = "Quy chuẩn taxonomy: quy định Tetra Pak thống nhất thuộc paper hay cardboard (hoặc class composite)."
            certainty = "Cao"
            
        elif fn == "Phan-Loai-Rac-Tai-Ng-01.jpg" and oid in ["img22_obj01", "img22_obj02"]:
            severity = "AMBIGUOUS"
            color = COLOR_AMBIGUOUS
            issue_category = "GARBAGE_BAG_VS_CONTENTS"
            issue_desc = f"Bao tải/túi ni lông chứa rác ({cname}, {area_pct:.1f}%) được gán nhãn plastic theo vỏ túi thay vì trash (rác hỗn hợp bên trong)."
            proposed_fix = "Làm rõ taxonomy: túi rác đóng gói chứa nhiều loại rác nên gán nhãn plastic (theo vỏ ngoài) hay trash (theo bản chất)."
            certainty = "Trung bình"

        # Record suspicious items
        if severity in ["CRITICAL", "HIGH", "AMBIGUOUS"]:
            suspicious_table.append({
                'filename': fn,
                'object_id': oid,
                'class_name': cname,
                'class_id': cid,
                'bbox': box,
                'area_pct': round(area_pct, 2),
                'severity': severity,
                'issue_category': issue_category,
                'issue_desc': issue_desc,
                'proposed_fix': proposed_fix,
                'certainty': certainty
            })
            
        all_annotations_analyzed.append({
            'filename': fn,
            'object_id': oid,
            'class_name': cname,
            'class_id': cid,
            'bbox': box,
            'area_pct': round(area_pct, 2),
            'severity': severity,
            'issue_category': issue_category,
            'issue_desc': issue_desc,
            'proposed_fix': proposed_fix,
            'certainty': certainty
        })
        
        # Draw box on overlay
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        thickness = 3 if severity == "CRITICAL" else (2 if severity in ["HIGH", "AMBIGUOUS"] else 1)
        cv2.rectangle(overlay, pt1, pt2, color, thickness)
        
        # Label
        lbl_text = f"[{oid}] {cname} {area_pct:.1f}% ({severity})"
        t_size = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        c_y1 = max(0, pt1[1] - 18)
        cv2.rectangle(overlay, (pt1[0], c_y1), (pt1[0] + t_size[0] + 6, c_y1 + 18), color, -1)
        cv2.putText(overlay, lbl_text, (pt1[0] + 3, c_y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)
        
    out_img_path = out_overlay_dir / f"review_{fn}.jpg"
    cv2.imwrite(str(out_img_path), overlay)

print(f"Generated {len(gt_data['images'])} review overlay images in: {out_overlay_dir}")
print(f"Total annotations analyzed: {len(all_annotations_analyzed)}")
print(f"Total suspicious annotations flagged: {len(suspicious_table)}")

crit = [s for s in suspicious_table if s['severity'] == 'CRITICAL']
high = [s for s in suspicious_table if s['severity'] == 'HIGH']
ambi = [s for s in suspicious_table if s['severity'] == 'AMBIGUOUS']

print(f"  - CRITICAL: {len(crit)}")
print(f"  - HIGH:     {len(high)}")
print(f"  - AMBIGUOUS:{len(ambi)}")

with open("reports/qa_acceptance/suspicious_gt_audit_table.json", "w", encoding="utf-8") as f:
    json.dump({
        'total_objects': len(all_annotations_analyzed),
        'suspicious_count': len(suspicious_table),
        'critical_count': len(crit),
        'high_count': len(high),
        'ambiguous_count': len(ambi),
        'suspicious_items': suspicious_table,
        'all_items': all_annotations_analyzed
    }, f, indent=2, ensure_ascii=False)

print("Saved reports/qa_acceptance/suspicious_gt_audit_table.json successfully.")
