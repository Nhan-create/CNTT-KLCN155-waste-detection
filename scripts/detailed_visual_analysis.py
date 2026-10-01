"""Analyze the 18 candidate pairs in detail to determine:
1. Exact visual characteristics of each image.
2. Background color, foreground shape, color differences.
3. Determine whether they are:
   - SAME_IMAGE_RESIZED / SAME_IMAGE_CROPPED
   - BURST_SHOT_SAME_OBJECT (same physical item in same setup)
   - SAME_OBJECT_DIFFERENT_VIEW (same physical item rotated)
   - DIFFERENT_OBJECTS_SAME_CLASS (different objects that share similar shape/background)
   - DIFFERENT_OBJECTS_DIFFERENT_CLASS (visual coincidence / cross-class)
   - LABEL_CONFLICT (same or near-identical object with conflicting class labels)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = Path(r"D:\HK7\Đồ án khóa luận")
CANDIDATES_PATH = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\cross_split_phash_candidates.csv")

def analyze_image(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    h, w, c = arr.shape
    # Corner pixels as background proxy
    corners = np.concatenate([
        arr[0:10, 0:10, :].reshape(-1, 3),
        arr[0:10, -10:, :].reshape(-1, 3),
        arr[-10:, 0:10, :].reshape(-1, 3),
        arr[-10:, -10:, :].reshape(-1, 3),
    ], axis=0)
    bg_mean = corners.mean(axis=0)
    center = arr[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75), :].reshape(-1, 3)
    fg_mean = center.mean(axis=0)
    return {
        "size": f"{w}x{h}",
        "aspect_ratio": round(w / h, 2),
        "bg_color": [round(x, 1) for x in bg_mean],
        "center_color": [round(x, 1) for x in fg_mean],
    }

def main():
    df = pd.read_csv(CANDIDATES_PATH)
    
    for idx, row in df.iterrows():
        p1 = BASE_DIR / row["path1"]
        p2 = BASE_DIR / row["path2"]
        im1 = Image.open(p1)
        im2 = Image.open(p2)
        stat1 = analyze_image(im1)
        stat2 = analyze_image(im2)
        
        # Check if one is a resized version of another
        im1_r = im1.convert("RGB").resize((128, 128))
        im2_r = im2.convert("RGB").resize((128, 128))
        arr1 = np.array(im1_r, dtype=np.float32)
        arr2 = np.array(im2_r, dtype=np.float32)
        diff = np.abs(arr1 - arr2)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))
        
        print(f"=== PAIR {idx+1:02d} ===")
        print(f"File 1: {row['file1']} ({row['class1']}, {row['split1']}) -> Size: {stat1['size']}, BG: {stat1['bg_color']}")
        print(f"File 2: {row['file2']} ({row['class2']}, {row['split2']}) -> Size: {stat2['size']}, BG: {stat2['bg_color']}")
        print(f"pHash Hamming: {row['hamming_dist']} | Resized Mean Diff: {mean_diff:.2f} | Max Diff: {max_diff:.2f}")

if __name__ == "__main__":
    main()
