# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Kiểm toán nguồn gốc, cụm pHash và tính toàn vẹn dữ liệu cho `audit_openimages_provenance`.
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
from pathlib import Path
from collections import Counter

manifest_path = Path("data/audit/real_detection_source_manifest.csv")
data_dir = Path("data/detection")

# Read manifest records
manifest_records = {}
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest_records[row['filename']] = row

print(f"Total records in real_detection_source_manifest.csv: {len(manifest_records)}")

# Read all files in splits_real_v2
split_files = {}
for sp in ["train.txt", "val.txt", "test.txt"]:
    sp_path = data_dir / "splits_real_v2" / sp
    with open(sp_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                fn = Path(p).name
                split_files[fn] = sp

print(f"Total images in splits_real_v2: {len(split_files)}")

# Check openimages files in splits_real_v2
oi_in_splits = {fn: sp for fn, sp in split_files.items() if "openimages" in fn or "oi_" in fn or manifest_records.get(fn, {}).get('source') == 'openimages_v7'}

print(f"Total Open Images files in splits_real_v2: {len(oi_in_splits)}")

missing_in_manifest = []
box_mismatches = []
provenance_table = []

for fn, sp in split_files.items():
    lbl_candidates = list((data_dir / "labels" / "real").rglob(f"{Path(fn).stem}.txt"))
    actual_boxes = 0
    actual_classes = Counter()
    if lbl_candidates and lbl_candidates[0].exists():
        with open(lbl_candidates[0], "r", encoding="utf-8") as lf:
            lines = [l.strip() for l in lf if l.strip()]
            actual_boxes = len(lines)
            for l in lines:
                actual_classes[int(l.split()[0])] += 1
                
    m_row = manifest_records.get(fn)
    if not m_row:
        if "openimages" in fn or "oi_" in fn:
            missing_in_manifest.append(fn)
            provenance_table.append({
                'filename': fn,
                'source': 'openimages_v7 (inferred from name)',
                'expected_boxes': 'UNKNOWN (Missing in CSV)',
                'actual_boxes': actual_boxes,
                'actual_classes': dict(actual_classes),
                'split': sp,
                'error': 'MISSING_SOURCE_MANIFEST_RECORD',
                'recommended_fix': 'Truy vấn OpenImages v7 metadata theo ImageID để phục hồi provenance và license; đối soát box.'
            })
    else:
        src = m_row.get('source', '')
        if src == 'openimages_v7' or 'openimages' in fn:
            expected_boxes = int(m_row.get('box_count', 0))
            if expected_boxes != actual_boxes:
                box_mismatches.append(fn)
                provenance_table.append({
                    'filename': fn,
                    'source': src,
                    'expected_boxes': expected_boxes,
                    'actual_boxes': actual_boxes,
                    'actual_classes': dict(actual_classes),
                    'split': sp,
                    'error': f'BOX_COUNT_MISMATCH (Expected {expected_boxes} vs Actual {actual_boxes})',
                    'recommended_fix': 'Kiểm tra xem nhãn thực tế có bị lọc theo class 10 lớp hay bị rơi box khi chuyển đổi sang YOLO.'
                })

print(f"\n1. Open Images missing in manifest ({len(missing_in_manifest)} images):")
for fn in missing_in_manifest:
    print(f"  - {fn}")

print(f"\n2. Open Images with box count mismatch ({len(box_mismatches)} images):")
for fn in box_mismatches:
    print(f"  - {fn}")

# Output table
print("\n=== OPEN IMAGES AUDIT TABLE ===")
print(f"{'Filename':35s} | {'Source':15s} | {'Exp':5s} | {'Act':5s} | {'Error'}")
print("-" * 110)
for r in provenance_table:
    print(f"{r['filename']:35s} | {str(r['source']):15s} | {str(r['expected_boxes']):5s} | {str(r['actual_boxes']):5s} | {r['error']}")
