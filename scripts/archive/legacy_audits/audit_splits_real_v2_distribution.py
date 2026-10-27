import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from collections import Counter

data_dir = Path("data/detection")
splits = ['train.txt', 'val.txt', 'test.txt']

class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

print("=== AUDIT OF REAL V2 SPLITS (data/detection/splits_real_v2) ===")

for sp in splits:
    split_file = data_dir / "splits_real_v2" / sp
    if not split_file.exists():
        print(f"Missing {split_file}")
        continue
    with open(split_file, "r") as f:
        img_rel_paths = [line.strip() for line in f if line.strip()]
    
    class_counts = Counter()
    total_boxes = 0
    missing_labels = 0
    empty_labels = 0
    
    for p in img_rel_paths:
        # e.g. images/real/taco/batch_1/000000.jpg -> labels/real/taco/batch_1/000000.txt
        img_path = data_dir / p
        # replace images with labels and .ext with .txt
        parts = list(Path(p).parts)
        if parts[0] == 'images':
            parts[0] = 'labels'
        lbl_rel = Path(*parts).with_suffix('.txt')
        lbl_path = data_dir / lbl_rel
        
        if not lbl_path.exists():
            missing_labels += 1
            continue
            
        with open(lbl_path, "r") as lf:
            lines = [l.strip() for l in lf if l.strip()]
        
        if len(lines) == 0:
            empty_labels += 1
            
        for l in lines:
            toks = l.split()
            cid = int(toks[0])
            class_counts[cid] += 1
            total_boxes += 1
            
    print(f"\n--- Split: {sp} ({len(img_rel_paths)} images, {total_boxes} boxes) ---")
    print(f"Missing labels: {missing_labels}, Empty labels (background/negatives): {empty_labels}")
    for cid in range(10):
        cname = class_names[cid]
        cnt = class_counts.get(cid, 0)
        pct = (cnt / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  [{cid}] {cname:12s}: {cnt:5d} ({pct:5.2f}%)")
