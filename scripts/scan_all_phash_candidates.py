"""Scan all pairwise pHash distances <= 4 across ALL clean images in Split V2.

Does not restrict to same class.
Reports:
1. Total candidate pairs (Hamming distance <= 4).
2. Breakdown by same-class vs cross-class.
3. Breakdown by same-split vs cross-split.
4. Detailed table of all cross-class candidate pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path("D:/CNTT-KLCN155-waste-detection")
MANIFEST_PATH = PROJECT_ROOT / "data" / "audit" / "split_manifest_v2.csv"
DATASET_INFO_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_info.csv"
BASE_DATA_DIR = Path("D:/HK7/Đồ án khóa luận")


def hex_to_bits(hex_str: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8))


def main():
    print("=" * 80)
    print("DATASET-WIDE PAIRWISE PHASH SCAN (ALL CLASSES, ALL SPLITS)")
    print("=" * 80)

    manifest = pd.read_csv(MANIFEST_PATH)
    info = pd.read_csv(DATASET_INFO_PATH)

    def get_proc_name(p: str) -> str:
        parts = p.replace("\\", "/").split("/")
        src = parts[2]
        fname = parts[-1]
        return f"{src}_{fname}"

    info["clean_name"] = info["path"].apply(get_proc_name)
    phash_map = info.set_index("clean_name")["phash"].to_dict()
    raw_path_map = info.set_index("clean_name")["path"].to_dict()

    manifest["phash"] = manifest["filename"].map(phash_map)
    manifest["raw_rel_path"] = manifest["filename"].map(raw_path_map)

    missing_phash = manifest["phash"].isna().sum()
    print(f"Manifest rows: {len(manifest)} | Missing pHash: {missing_phash}")
    assert missing_phash == 0, f"Found {missing_phash} rows missing pHash!"

    bits = np.array([hex_to_bits(h) for h in manifest["phash"]], dtype=np.uint8)
    N = len(bits)
    print(f"Loaded {N} pHash signatures (64-bit). Starting vectorized chunked scan...")

    filenames = manifest["filename"].values
    splits = manifest["split"].values
    classes = manifest["unified_label_10"].values
    raw_rel_paths = manifest["raw_rel_path"].values

    chunk_size = 2000
    candidates = []

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        b_sub = bits[i_start:i_end]
        # Hamming distance: sum of differing bits
        diff = (b_sub[:, None, :] != bits[None, :, :]).sum(axis=2)
        rows, cols = np.where(diff <= 4)
        for r, c in zip(rows, cols):
            g_r = i_start + r
            if g_r < c:  # strictly upper triangle
                candidates.append({
                    "idx1": g_r,
                    "idx2": c,
                    "file1": filenames[g_r],
                    "file2": filenames[c],
                    "split1": splits[g_r],
                    "split2": splits[c],
                    "class1": classes[g_r],
                    "class2": classes[c],
                    "raw_path1": raw_rel_paths[g_r],
                    "raw_path2": raw_rel_paths[c],
                    "hamming_dist": int(diff[r, c]),
                    "same_class": bool(classes[g_r] == classes[c]),
                    "same_split": bool(splits[g_r] == splits[c]),
                })

    df = pd.DataFrame(candidates)
    print(f"\nTotal candidate pairs with Hamming distance <= 4: {len(df)}")
    print(f"  - Same-class pairs: {len(df[df['same_class']])}")
    print(f"  - Cross-class pairs: {len(df[~df['same_class']])}")
    print(f"  - Same-split pairs: {len(df[df['same_split']])}")
    print(f"  - Cross-split pairs: {len(df[~df['same_split']])}")

    # Breakdown by category
    same_cls_same_split = len(df[df['same_class'] & df['same_split']])
    same_cls_cross_split = len(df[df['same_class'] & ~df['same_split']])
    cross_cls_same_split = len(df[~df['same_class'] & df['same_split']])
    cross_cls_cross_split = len(df[~df['same_class'] & ~df['same_split']])

    print("\nFour-Way Breakdown:")
    print(f"  1. Same Class, Same Split:   {same_cls_same_split}")
    print(f"  2. Same Class, Cross Split:  {same_cls_cross_split} (LEAKAGE if same object)")
    print(f"  3. Cross Class, Same Split:  {cross_cls_same_split}")
    print(f"  4. Cross Class, Cross Split: {cross_cls_cross_split}")

    # Export full candidates table
    out_csv = PROJECT_ROOT / "data" / "audit" / "all_dataset_phash_candidates.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved all candidates to: {out_csv}")

    # Inspect all Cross-Class candidates
    cross_cls_df = df[~df["same_class"]].copy()
    print(f"\nFound {len(cross_cls_df)} Cross-Class candidate pairs across entire dataset:")
    for idx, row in cross_cls_df.iterrows():
        print(f"  Pair ({row['class1']}/{row['split1']} vs {row['class2']}/{row['split2']}) dist={row['hamming_dist']}: {row['file1']} vs {row['file2']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
