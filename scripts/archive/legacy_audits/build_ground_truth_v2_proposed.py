# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `build_ground_truth_v2_proposed`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from copy import deepcopy

v1_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
v2_path = Path("reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json")

with open(v1_path, "r", encoding="utf-8") as f:
    v1_data = json.load(f)

v2_data = {
    "metadata": {
        "dataset_name": "Hinh_Anh_Rac_Independent_Ground_Truth",
        "version": "v2.0-proposed-standardized",
        "parent_version": "v1.0-frozen",
        "created_at": "2026-10-11T22:30:00Z",
        "description": "Standardized Ground Truth resolving full-canvas duplicates, surface carpets, utility receptacles, and unsegregated piles.",
        "evaluator": "Agent_D_QA_Standardization",
        "frozen_v1_preserved_at": "reports/qa_acceptance/ground_truth_hinh_anh_rac.json"
    },
    "taxonomy_guidelines": {
        "receptacles_vs_waste": "Trash bins, wheeled dumpsters, and indoor baskets are municipal fixtures, not waste to be collected.",
        "material_vs_object": "Specific object types (battery, shoes, clothes) supersede raw materials (metal, plastic).",
        "discrete_vs_pile": "Discrete objects must have clear boundaries. Piles covering >50% without separable items are classified as waste_pile, segregated from instance detection.",
        "bagged_waste": "Garbage bags filled with heterogeneous trash are classified as trash; empty discarded film/carrier bags are plastic."
    },
    "images": []
}

total_v1_objs = 0
total_v2_accepted = 0
total_quarantined = 0
total_flagged_pile = 0
total_flagged_nested = 0
total_ambiguous = 0

for img in v1_data['images']:
    fn = img['filename']
    w, h = img['width'], img['height']
    img_area = float(w * h)
    
    if fn in ["images (8).jpg", "images (18).jpg"]:
        eval_policy = "NEGATIVE_BENCHMARK"
    elif fn in ["images (6).jpg", "images (7).jpg", "images (9).jpg", "images (13).jpg", "OIP.webp", "rac-thai-huu-co-1.jpg", "photo-5-1642556540067542343380.webp", "images (19).jpg"]:
        eval_policy = "UNSEGREGATED_PILE_EXCLUDED"
    else:
        eval_policy = "INSTANCE_BENCHMARK"

    img_v2 = {
        "filename": fn,
        "width": w,
        "height": h,
        "sha256": img.get("sha256"),
        "status": "VALID_IMAGE",
        "evaluation_policy": eval_policy,
        "objects": []
    }
    
    for obj in img['objects']:
        total_v1_objs += 1
        obj_id = obj['object_id']
        cname = obj['class_name']
        cid = obj['class_id']
        box = obj['bbox_xyxy']
        bw = float(box[2] - box[0])
        bh = float(box[3] - box[1])
        area = bw * bh
        area_pct = (area / img_area) * 100.0
        rat = obj.get('review_rationale', '')
        
        status = "ACCEPTED"
        v2_action = "KEEP"
        audit_note = "Valid discrete waste object annotation."
        
        # 1. 100% duplicate canvas box over individual objects
        if fn == "download (2).jpg" and obj_id == "img03_obj03":
            status = "QUARANTINED_FULL_CANVAS_DUPLICATE"
            v2_action = "REMOVE_FROM_EVAL"
            audit_note = "Box 100% toàn ảnh bao trùm lên 2 chai nhựa đơn lẻ đã được đánh dấu bên trong. Xóa bỏ để tránh phạt trùng lặp."
            total_quarantined += 1
            
        # 2. Surface glass carpet
        elif fn == "base64-1717231151649602366964.jpeg" and obj_id == "img01_obj07":
            status = "QUARANTINED_TEXTURE_CARPET"
            v2_action = "REMOVE_FROM_EVAL"
            audit_note = "Box thảm vụn kính chiếm 85.38% nền cát bao trùm 4 chai/hũ kính đơn lẻ. Xóa bỏ để tập trung vào vật thể thực tế."
            total_quarantined += 1
            
        # 3. Municipal utility receptacles
        elif fn == "images (18).jpg" and obj_id == "img11_obj01":
            status = "QUARANTINED_UTILITY_RECEPTACLE"
            v2_action = "REMOVE_FROM_EVAL"
            audit_note = "Thùng rác công cộng có bánh xe (thiết bị thu gom đô thị), không phải rác thải cần nhặt."
            total_quarantined += 1
            
        elif fn == "images (8).jpg" and obj_id == "img18_obj01":
            status = "QUARANTINED_UTILITY_RECEPTACLE"
            v2_action = "REMOVE_FROM_EVAL"
            audit_note = "Giỏ đựng rác văn phòng (thiết bị chứa rác nội thất), không phải rác thải bừa bãi."
            total_quarantined += 1
            
        # 4. Unsegregated full-canvas piles (100% or near 100%)
        elif fn in ["images (6).jpg", "images (7).jpg", "images (9).jpg", "images (13).jpg", "OIP.webp", "rac-thai-huu-co-1.jpg", "photo-5-1642556540067542343380.webp", "images (19).jpg"]:
            status = "FLAGGED_UNSEGREGATED_PILE"
            v2_action = "SEPARATE_PILE_EVALUATION"
            audit_note = f"Đống rác/bãi rác cận cảnh chiếm {area_pct:.1f}% toàn bộ khung hình. Đánh dấu để đánh giá chế độ pile detection hoặc segmentation."
            total_flagged_pile += 1
            
        # 5. Nested container/overlapping objects
        elif fn == "images (12).jpg" and obj_id == "img06_obj01":
            status = "FLAGGED_CONTAINER_NESTING"
            v2_action = "REFINE_BOUNDARY"
            audit_note = "Thùng carton bên ngoài chứa tập giấy bên trong. Giữ lại nhưng lưu ý quan hệ cha-con (container-contained)."
            total_flagged_nested += 1
            
        elif fn == "images (10).jpg" and obj_id == "img04_obj01":
            status = "FLAGGED_OVERLAPPING_OBJECT"
            v2_action = "REFINE_BOUNDARY"
            audit_note = "Box đống quần áo lớn bao trùm quần áo nhỏ bên trong."
            total_flagged_nested += 1
            
        elif fn == "OIP (1).webp" and obj_id == "img20_obj01":
            status = "FLAGGED_OVERLAPPING_OBJECT"
            v2_action = "REFINE_BOUNDARY"
            audit_note = "Đống quần áo sa mạc chồng lấn một phần với gò bên cạnh."
            total_flagged_nested += 1
            
        # 6. Taxonomy ambiguity
        elif fn == "Phan-Loai-Rac-Tai-Ng-01.jpg" and obj_id == "img22_obj04":
            status = "AMBIGUOUS_COMPOSITE_TAXONOMY"
            v2_action = "STANDARDIZE_TAXONOMY"
            audit_note = "Vỏ hộp sữa TH true MILK (Tetra Pak) đa lớp (giấy/nhôm/PE). Đề xuất chuẩn hóa là cardboard theo vật liệu cứng chính."
            total_ambiguous += 1
            
        elif fn == "Phan-Loai-Rac-Tai-Ng-01.jpg" and obj_id in ["img22_obj01", "img22_obj02"]:
            status = "AMBIGUOUS_BAGGED_WASTE"
            v2_action = "STANDARDIZE_TAXONOMY"
            audit_note = f"Túi rác ni lông ({cname}) chứa rác hỗn hợp. Đề xuất chuẩn hóa thành trash nếu chứa rác, plastic nếu là túi rỗng."
            total_ambiguous += 1
            
        else:
            total_v2_accepted += 1
            
        new_obj = deepcopy(obj)
        new_obj['area_percentage'] = round(area_pct, 2)
        new_obj['v2_status'] = status
        new_obj['v2_action'] = v2_action
        new_obj['v2_audit_note'] = audit_note
        new_obj['is_active_for_instance_eval'] = (status in ["ACCEPTED", "FLAGGED_CONTAINER_NESTING", "FLAGGED_OVERLAPPING_OBJECT", "AMBIGUOUS_COMPOSITE_TAXONOMY", "AMBIGUOUS_BAGGED_WASTE"])
        new_obj['is_ignore_region'] = (status == "FLAGGED_UNSEGREGATED_PILE")
        img_v2['objects'].append(new_obj)
        
    v2_data['images'].append(img_v2)

# Summary
v2_data['summary'] = {
    "total_images": len(v2_data['images']),
    "total_v1_objects": total_v1_objs,
    "total_v2_accepted_discrete": total_v2_accepted,
    "total_quarantined_objects": total_quarantined,
    "total_unsegregated_piles": total_flagged_pile,
    "total_nested_overlapping": total_flagged_nested,
    "total_ambiguous_taxonomy": total_ambiguous,
    "active_objects_for_instance_detection": sum(
        sum(1 for o in img['objects'] if o['is_active_for_instance_eval'])
        for img in v2_data['images']
    )
}

with open(v2_path, "w", encoding="utf-8") as f:
    json.dump(v2_data, f, indent=2, ensure_ascii=False)

print(f"Generated Ground Truth v2 Proposed at: {v2_path}")
print(f"Total images: {len(v2_data['images'])}")
print(f"Total V1 objects: {total_v1_objs}")
print(f"  - Quarantined (Full-canvas duplicate / Surface carpet / Utility receptacle): {total_quarantined}")
print(f"  - Flagged unsegregated piles (>80-100% bulk dump): {total_flagged_pile}")
print(f"  - Flagged nested / container: {total_flagged_nested}")
print(f"  - Ambiguous taxonomy: {total_ambiguous}")
print(f"  - Accepted discrete objects: {total_v2_accepted}")
print(f"Active objects for instance evaluation in v2: {v2_data['summary']['active_objects_for_instance_detection']}")
