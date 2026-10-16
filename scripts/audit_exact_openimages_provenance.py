import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
from pathlib import Path
from collections import Counter

manifest_path = Path("data/detection/real_v2_source_manifest.csv")
data_dir = Path("data/detection")

manifest_records = {}
with open(manifest_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        manifest_records[row['filename']] = row

print(f"Total records in data/detection/real_v2_source_manifest.csv: {len(manifest_records)}")

# Read all files in splits_real_v2
split_images = {}
for sp in ["train.txt", "val.txt", "test.txt"]:
    with open(data_dir / "splits_real_v2" / sp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                fn = Path(line).name
                split_images[fn] = sp

print(f"Total images in splits_real_v2: {len(split_images)}")

missing_in_source = []
box_mismatches = []
provenance_findings = []

for fn, sp in split_images.items():
    # Read actual label file
    lbl_file = data_dir / "labels" / "real" / f"{Path(fn).stem}.txt"
    actual_boxes = 0
    actual_classes = Counter()
    if lbl_file.exists():
        with open(lbl_file, "r", encoding="utf-8") as lf:
            lines = [l.strip() for l in lf if l.strip()]
            actual_boxes = len(lines)
            for l in lines:
                actual_classes[int(l.split()[0])] += 1
                
    if fn not in manifest_records:
        missing_in_source.append((fn, sp, actual_boxes, dict(actual_classes)))
        provenance_findings.append({
            'filename': fn,
            'split': sp,
            'source': 'UNKNOWN_MISSING_IN_MANIFEST',
            'expected_boxes': 'N/A',
            'actual_boxes': actual_boxes,
            'actual_classes': dict(actual_classes),
            'issue': 'MISSING_IN_SOURCE_MANIFEST',
            'fix': 'Cần tra cứu metadata nguồn OpenImages v7 theo ImageID để bổ sung bản ghi.'
        })
    else:
        rec = manifest_records[fn]
        exp_boxes = int(rec.get('num_boxes', rec.get('box_count', -1)))
        if exp_boxes != actual_boxes:
            box_mismatches.append((fn, sp, exp_boxes, actual_boxes, dict(actual_classes), rec.get('source')))
            provenance_findings.append({
                'filename': fn,
                'split': sp,
                'source': rec.get('source'),
                'expected_boxes': exp_boxes,
                'actual_boxes': actual_boxes,
                'actual_classes': dict(actual_classes),
                'issue': f'BOX_COUNT_MISMATCH (Expected {exp_boxes} vs Actual {actual_boxes})',
                'fix': 'Kiểm tra xem nhãn thực tế có bị lọc theo class 10 lớp hay bị rơi box khi chuyển đổi sang YOLO.'
            })

print(f"\n1. Images missing in source manifest: {len(missing_in_source)}")
for fn, sp, act, cls_dist in missing_in_source:
    print(f"  - {fn} ({sp}): {act} actual boxes | classes={cls_dist}")

print(f"\n2. Images with box count mismatch: {len(box_mismatches)}")
for fn, sp, exp, act, cls_dist, src in box_mismatches:
    print(f"  - {fn} ({sp}, {src}): expected={exp} vs actual={act} | classes={cls_dist}")
