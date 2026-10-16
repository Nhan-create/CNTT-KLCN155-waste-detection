import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from PIL import Image

data_dir = Path("data/detection")

for fname, line_num in [("taco_0112", 12), ("taco_0457", 5)]:
    lbl_file = data_dir / "labels" / "real" / f"{fname}.txt"
    img_candidates = list((data_dir / "images" / "real").rglob(f"{fname}.*"))
    
    print(f"\n==================== {fname} ====================")
    print(f"Label file: {lbl_file}")
    
    img_path = img_candidates[0] if img_candidates else None
    print(f"Image path: {img_path}")
    img_w, img_h = None, None
    if img_path and img_path.exists():
        with Image.open(img_path) as im:
            img_w, img_h = im.size
            print(f"Image dimensions (W x H): {img_w} x {img_h}")
            
    with open(lbl_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    print(f"Total lines in label file: {len(lines)}")
    if line_num <= len(lines):
        target_line = lines[line_num - 1]
        toks = target_line.split()
        cid = int(toks[0])
        xc, yc, bw, bh = float(toks[1]), float(toks[2]), float(toks[3]), float(toks[4])
        
        # Calculate normalized corners
        x1_norm = xc - bw / 2.0
        y1_norm = yc - bh / 2.0
        x2_norm = xc + bw / 2.0
        y2_norm = yc + bh / 2.0
        
        print(f"Line {line_num}: {target_line}")
        print(f"  Class ID: {cid}")
        print(f"  Normalized [x1, y1, x2, y2]: [{x1_norm:.6f}, {y1_norm:.6f}, {x2_norm:.6f}, {y2_norm:.6f}]")
        print(f"  Out of bounds check:")
        print(f"    x1 < 0: {x1_norm < 0} (val={x1_norm:.6f})")
        print(f"    y1 < 0: {y1_norm < 0} (val={y1_norm:.6f})")
        print(f"    x2 > 1: {x2_norm > 1} (val={x2_norm:.6f})")
        print(f"    y2 > 1: {y2_norm > 1} (val={y2_norm:.6f})")
        
        if img_w and img_h:
            x1_px = x1_norm * img_w
            y1_px = y1_norm * img_h
            x2_px = x2_norm * img_w
            y2_px = y2_norm * img_h
            print(f"  Pixel coords: [{x1_px:.2f}, {y1_px:.2f}, {x2_px:.2f}, {y2_px:.2f}]")
