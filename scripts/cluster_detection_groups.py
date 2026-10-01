"""
scripts/cluster_detection_groups.py
-----------------------------------
Generates Group IDs for the multi-object detection dataset (1,419 images)
based on perceptual hash (pHash) Hamming distance <= 4 and Disjoint Set Union (DSU).
Ensures near-duplicate images and sequence captures share the same group_id.
Updates data/detection/manifest_detection_v1.csv.
"""

import sys
from pathlib import Path
import pandas as pd
from PIL import Image

def compute_phash(img_path: Path, hash_size: int = 8) -> int:
    with Image.open(img_path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        phash_int = 0
        for p in pixels:
            phash_int = (phash_int << 1) | (1 if p > avg else 0)
        return phash_int

def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")

class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def main():
    project_root = Path(__file__).resolve().parent.parent
    detection_dir = project_root / "data" / "detection"
    manifest_path = detection_dir / "manifest_detection_v1.csv"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    print(f"[INFO] Loaded manifest: {len(df)} rows.")

    # Compute pHash for each image
    print("[INFO] Computing pHash for all detection images...")
    phashes = []
    for idx, row in df.iterrows():
        img_path = detection_dir / row["relative_image_path"]
        if not img_path.exists():
            print(f"[ERROR] Missing image: {img_path}")
            sys.exit(1)
        ph = compute_phash(img_path)
        phashes.append(ph)

    n = len(df)
    dsu = DSU(n)

    # Cluster images with Hamming distance <= 4
    print("[INFO] Clustering images with pHash Hamming distance <= 4...")
    near_dups_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Only compare if same source type (synthetic with synthetic, real with real)
            if df.at[i, "is_synthetic"] == df.at[j, "is_synthetic"]:
                dist = hamming_distance(phashes[i], phashes[j])
                if dist <= 4:
                    dsu.union(i, j)
                    near_dups_count += 1

    print(f"[INFO] Discovered {near_dups_count} near-duplicate pairs (Hamming <= 4).")

    # Assign group IDs
    group_map = {}
    group_ids = []
    for i in range(n):
        root = dsu.find(i)
        if root not in group_map:
            prefix = "syn_grp" if df.at[root, "is_synthetic"] else "real_grp"
            group_map[root] = f"{prefix}_{len(group_map)+1:04d}"
        group_ids.append(group_map[root])

    df["group_id"] = group_ids
    df["phash"] = [f"{p:016x}" for p in phashes]

    # Save updated manifest
    df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Updated manifest with {len(group_map)} unique group IDs: {manifest_path}")

    # Summary
    grp_counts = df["group_id"].value_counts()
    multi_img_groups = grp_counts[grp_counts > 1]
    print(f"Total Unique Groups: {len(group_map)}")
    print(f"Groups with >1 image: {len(multi_img_groups)} (containing {multi_img_groups.sum()} images)")

if __name__ == "__main__":
    main()
