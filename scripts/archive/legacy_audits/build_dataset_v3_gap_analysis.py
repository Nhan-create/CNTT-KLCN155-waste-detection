# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `build_dataset_v3_gap_analysis`.
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

data_dir = Path("data/detection")
splits = ['train', 'val', 'test']
classes = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# Source names
source_manifest_path = data_dir / "real_v2_source_manifest.csv"
split_manifest_path = data_dir / "real_v2_split_manifest.csv"

# 1. Parse splits distribution
stats = {s: Counter() for s in splits}
img_stats = {s: 0 for s in splits}
img_source_stats = {s: Counter() for s in splits}
class_per_source = defaultdict(Counter)

# Read split manifest
split_meta = {}
with open(split_manifest_path, "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        split_meta[row['filename']] = row

for s in splits:
    split_file = data_dir / "splits_real_v2" / f"{s}.txt"
    with open(split_file, "r", encoding="utf-8") as f:
        img_paths = [l.strip() for l in f if l.strip()]
    img_stats[s] = len(img_paths)
    
    for p in img_paths:
        fn = Path(p).name
        src = split_meta.get(fn, {}).get('source', 'unknown')
        img_source_stats[s][src] += 1
        
        parts = list(Path(p).parts)
        if parts[0] == 'images':
            parts[0] = 'labels'
        lbl_path = data_dir / Path(*parts).with_suffix('.txt')
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    if line.strip():
                        toks = line.strip().split()
                        cid = int(toks[0])
                        stats[s][cid] += 1
                        class_per_source[cid][src] += 1

tot_train_boxes = sum(stats['train'].values())
tot_val_boxes = sum(stats['val'].values())
tot_test_boxes = sum(stats['test'].values())
grand_total_boxes = tot_train_boxes + tot_val_boxes + tot_test_boxes

# 2. Data gaps definition and priorities
gap_analysis = []
priorities = {
    'battery': ('P0 (Critical)', 'Chỉ có 1 box trong tập train (0.03%), 0 box trong val. Hoàn toàn đói nhãn. Mô hình không thể học đặc trưng nhận diện pin, nhầm lẫn sang shoes/metal.'),
    'biological': ('P0 (Critical)', 'Chỉ có 3 box trong tập train (0.08%), 2 box trong val. Đói nhãn nghiêm trọng. Dễ nhầm với nền đất đá hoặc trash.'),
    'cardboard': ('P1 (High)', '169 train boxes (4.26%). Thiếu các góc nhìn bìa carton đa dạng, vỏ hộp bị gập, ướt, rách.'),
    'clothes': ('P1 (High)', '368 train boxes (9.29%). 100% từ Open Images người mặc quần áo thời trang, không phải quần áo rác vứt ngoài đường.'),
    'glass': ('P1 (High)', '213 train boxes (5.37%). Thiếu chai thủy tinh vỡ, mảnh vỡ và chai trong suốt trên nền phức tạp.'),
    'metal': ('P1 (High)', '420 train boxes (10.60%). Chủ yếu là lon nước ngọt; thiếu kim loại rỉ sét, nắp hộp đồ hộp, phế liệu sắc nhọn.'),
    'paper': ('P1 (High)', '160 train boxes (4.04%). Thiếu giấy báo vò nát, khăn giấy bẩn, biên lai nhăn nheo.'),
    'plastic': ('P2 (Dominant)', '1,560 train boxes (39.36%). Chiếm tỷ trọng áp đảo gần 40%, gây thiên vị phân loại (bias) lên tất cả các vật thể khác.'),
    'shoes': ('P1 (High)', '215 train boxes (5.43%). Nguồn Open Images người đang đi giày/giày thời trang; thiếu giày cũ nát, rách đế, dép lê một chiếc trên đường.'),
    'trash': ('P2 (Dominant)', '854 train boxes (21.55%). Chiếm 21.55%, thường là nhãn gom chung (catch-all) gây nhiễu cho các lớp vật liệu cụ thể.')
}

for cid, cname in enumerate(classes):
    tr = stats['train'][cid]
    va = stats['val'][cid]
    te = stats['test'][cid]
    tot = tr + va + te
    pct = (tr / tot_train_boxes) * 100
    prio, desc = priorities[cname]
    gap_analysis.append({
        'class_id': cid,
        'class_name': cname,
        'train_boxes': tr,
        'val_boxes': va,
        'test_boxes': te,
        'total_boxes': tot,
        'train_percent': round(pct, 2),
        'sources': dict(class_per_source[cid]),
        'priority': prio,
        'gap_description': desc
    })

# 3. Hard-Negative candidates specification
hard_negative_spec = {
    'current_negative_images_in_train': 0,
    'current_negative_ratio': "0.0%",
    'target_negative_ratio': "5.0% - 10.0%",
    'target_negative_count': "75 - 150 images",
    'problem_statement': "Tập train hiện tại có 0 ảnh negative. Mọi ảnh đều có ít nhất 1 box rác. Do đó mô hình luôn dự đoán rác trên mọi khung hình, gây False Positive trầm trọng trên tường, biển quảng cáo, tủ điện, vỉa hè.",
    'candidate_categories': [
        {
            'category': 'clean_sidewalk_and_road',
            'vietnamese_name': 'Vỉa hè và lòng đường sạch',
            'recommended_count': 30,
            'rationale': 'Giảm FP nhận diện vết nứt, bóng râm đường phố thành rác',
            'license_requirement': 'CC-BY hoặc Public Domain / Tự chụp'
        },
        {
            'category': 'building_facades_and_brick_walls',
            'vietnamese_name': 'Tường gạch và mặt tiền tòa nhà',
            'recommended_count': 30,
            'rationale': 'Khắc phục triệt để lỗi SSDLite/YOLO nhận diện biển hiệu/tòa nhà thành cardboard/paper',
            'license_requirement': 'CC-BY hoặc Public Domain / Tự chụp'
        },
        {
            'category': 'utility_boxes_and_transformers',
            'vietnamese_name': 'Tủ điện kỹ thuật màu xanh và hộp cáp',
            'recommended_count': 25,
            'rationale': 'Khắc phục lỗi tủ điện màu xanh bị nhận nhầm thành bìa carton 48%',
            'license_requirement': 'CC-BY hoặc Public Domain / Tự chụp'
        },
        {
            'category': 'empty_public_trash_bins',
            'vietnamese_name': 'Thùng rác công cộng sạch và giỏ rác rỗng',
            'recommended_count': 25,
            'rationale': 'Khắc phục việc mô hình nhận diện bản thân chiếc thùng rác thành rác plastic cần thu gom',
            'license_requirement': 'CC-BY hoặc Public Domain / Tự chụp'
        },
        {
            'category': 'indoor_furniture_and_floors',
            'vietnamese_name': 'Sàn nhà, bàn ghế và nội thất sạch',
            'recommended_count': 20,
            'rationale': 'Dạy mô hình nhận biết không gian sinh hoạt sạch không có rác',
            'license_requirement': 'CC-BY hoặc Public Domain / Tự chụp'
        }
    ],
    'download_policy': 'TUYỆT ĐỐI KHÔNG TỰ Ý DOWNLOAD DỮ LIỆU NGOÀI LỚN KHI CHƯA ĐƯỢC PHÊ DUYỆT. Thiết kế sẵn danh mục và schema để tích hợp khi được chấp thuận.'
}

# 4. Summary output
out_data = {
    'dataset_version': 'real_v2_to_v3_transition',
    'total_images': {
        'train': img_stats['train'],
        'val': img_stats['val'],
        'test': img_stats['test'],
        'total': sum(img_stats.values())
    },
    'image_sources_by_split': {
        s: dict(img_source_stats[s]) for s in splits
    },
    'total_boxes': {
        'train': tot_train_boxes,
        'val': tot_val_boxes,
        'test': tot_test_boxes,
        'grand_total': grand_total_boxes
    },
    'class_gap_analysis': gap_analysis,
    'hard_negative_specification': hard_negative_spec
}

out_file = Path("reports/data_v2/dataset_v3_gap_analysis.json")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"Generated {out_file} successfully.")
