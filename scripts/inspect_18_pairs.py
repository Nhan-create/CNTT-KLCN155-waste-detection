"""Deep Inspection of 18 Cross-Split pHash Candidate Pairs.
Analyzes image resolution, modes, pixel stats, and visual content.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

CANDIDATES_PATH = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\cross_split_phash_candidates.csv")
BASE_DIR = Path(r"D:\HK7\Đồ án khóa luận")

def main():
    df = pd.read_csv(CANDIDATES_PATH)
    print(f"Loaded {len(df)} candidate pairs from {CANDIDATES_PATH}")
    
    for idx, row in df.iterrows():
        p1 = BASE_DIR / row["path1"]
        p2 = BASE_DIR / row["path2"]
        print(f"\n--- PAIR {idx+1:02d} ---")
        print(f"File 1: {row['file1']} | Class: {row['class1']} | Split: {row['split1']} | Source Path: {row['path1']}")
        print(f"File 2: {row['file2']} | Class: {row['class2']} | Split: {row['split2']} | Source Path: {row['path2']}")
        print(f"Hamming Dist: {row['hamming_dist']}")
        
        if not p1.exists() or not p2.exists():
            print(f"ERROR: Files not found! p1={p1.exists()}, p2={p2.exists()}")
            continue
            
        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")
        print(f"Image 1 Size: {im1.size} | Image 2 Size: {im2.size}")
        
        # Resize to common 224x224 to calculate precise pixel difference
        arr1 = np.array(im1.resize((224, 224)), dtype=np.float32)
        arr2 = np.array(im2.resize((224, 224)), dtype=np.float32)
        
        mae = float(np.mean(np.abs(arr1 - arr2)))
        corr = float(np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1])
        print(f"Resized Pixel MAE: {mae:.2f} (scale 0-255) | Pearson Correlation: {corr:.4f}")

if __name__ == "__main__":
    main()
