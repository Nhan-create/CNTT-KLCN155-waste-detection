"""Deterministic Group-Stratified Split Generation (Split V2).

Solves data leakage by:
1. Identifying all connected near-duplicate components (pHash <= 4 within class).
2. Isolating label conflict samples (Pair 10: vn_trash_train_cardboard438 vs garbage_v2_paper_582).
3. Assigning all connected components atomically to ONE split using Stratified Group Split (Seed 42, 70/15/15).
4. Exporting data/audit/split_manifest_v2.csv.
5. Materializing physical directory at data/processed_v2 using NTFS hardlinks.
6. Verifying zero cross-split leakage for all grouped clusters.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_V1_PATH = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\split_manifest.csv")
DATASET_INFO_PATH = Path(r"D:\CNTT-KLCN155-waste-detection\data\metadata\dataset_info.csv")
OUTPUT_MANIFEST_V2 = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\split_manifest_v2.csv")
OUTPUT_PROCESSED_V2 = Path(r"D:\CNTT-KLCN155-waste-detection\data\processed_v2")
AUDIT_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit")

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


class DisjointSetUnion:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def hex_to_bits(hex_str: str) -> np.ndarray:
    byte_vals = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(byte_vals, dtype=np.uint8))


def main() -> int:
    print("=" * 80)
    print("GENERATING LEAK-FREE SPLIT V2 (GROUP-STRATIFIED SPLIT, SEED 42)")
    print("=" * 80)

    # 1. Load data
    manifest_v1 = pd.read_csv(MANIFEST_V1_PATH)
    info = pd.read_csv(DATASET_INFO_PATH)

    def get_proc_name(row):
        parts = row["path"].replace("\\", "/").split("/")
        src = parts[2]
        fname = parts[-1]
        return f"{src}_{fname}"

    info["clean_name"] = info.apply(get_proc_name, axis=1)
    phash_dict = info.set_index("clean_name")["phash"].to_dict()
    raw_path_dict = info.set_index("clean_name")["path"].to_dict()

    manifest_v1["phash"] = manifest_v1["filename"].map(phash_dict)
    manifest_v1["raw_path"] = manifest_v1["filename"].map(raw_path_dict)

    print(f"Loaded {len(manifest_v1)} records from Manifest V1.")

    # 2. Quarantine Label Conflict Samples
    # Pair 10: vn_trash_train_cardboard438.jpg (Carton->cardboard) vs garbage_v2_paper_582.jpg (paper)
    # These two images depict the exact same physical cylinder cup with conflicting ground truth labels.
    quarantine_files = {"vn_trash_train_cardboard438.jpg", "garbage_v2_paper_582.jpg"}
    quarantine_records = manifest_v1[manifest_v1["filename"].isin(quarantine_files)].copy()
    quarantine_records["quarantine_reason"] = "CROSS_DATASET_LABEL_CONFLICT_SAME_OBJECT"
    quarantine_path = AUDIT_DIR / "quarantined_samples.csv"
    quarantine_records.to_csv(quarantine_path, index=False)
    print(f"Quarantined {len(quarantine_records)} conflicting label samples -> {quarantine_path}")

    clean_manifest = manifest_v1[~manifest_v1["filename"].isin(quarantine_files)].copy().reset_index(drop=True)
    print(f"Clean samples remaining for Split V2: {len(clean_manifest)}")

    # 3. Build Connected Components (DSU) using bitwise pHash <= 4 within each class
    dsu = DisjointSetUnion()
    for fn in clean_manifest["filename"]:
        dsu.find(fn)

    bits = np.array([hex_to_bits(h) for h in clean_manifest["phash"]], dtype=np.uint8)
    classes = clean_manifest["unified_label_10"].values
    filenames = clean_manifest["filename"].values

    total_merged_pairs = 0
    for c in CLASS_NAMES:
        c_mask = np.where(classes == c)[0]
        c_bits = bits[c_mask]
        c_files = filenames[c_mask]
        n_c = len(c_mask)

        chunk_size = 500
        for i_start in range(0, n_c, chunk_size):
            i_end = min(i_start + chunk_size, n_c)
            sub_bits = c_bits[i_start:i_end]
            diff = np.bitwise_xor(sub_bits[:, None, :], c_bits[None, :, :]).sum(axis=2)
            rows, cols = np.where(diff <= 4)
            for r, col in zip(rows, cols):
                global_r = i_start + r
                global_c = col
                if global_r < global_c:
                    f1, f2 = c_files[global_r], c_files[global_c]
                    dsu.union(f1, f2)
                    total_merged_pairs += 1

    print(f"Total within-class near-duplicate pairs grouped into atomic clusters: {total_merged_pairs}")

    # Map each file to its root cluster ID
    clean_manifest["cluster_root"] = clean_manifest["filename"].apply(dsu.find)
    cluster_counts = clean_manifest["cluster_root"].value_counts()
    multi_item_clusters = cluster_counts[cluster_counts > 1]
    print(f"Total unique clusters: {clean_manifest['cluster_root'].nunique()}")
    print(f"Clusters containing >= 2 images: {len(multi_item_clusters)}")

    # 4. Stratified Group Partition (70% Train, 15% Val, 15% Test) with Seed 42
    rng = np.random.RandomState(42)
    clean_manifest["new_split"] = ""

    # Partition group-by-group per class to preserve class balance
    for c in CLASS_NAMES:
        class_df = clean_manifest[clean_manifest["unified_label_10"] == c]
        # Get unique cluster roots and their sizes
        cluster_sizes = class_df.groupby("cluster_root").size().to_dict()
        cluster_list = list(cluster_sizes.keys())
        rng.shuffle(cluster_list)

        total_class_samples = len(class_df)
        target_val = int(round(total_class_samples * 0.15))
        target_test = int(round(total_class_samples * 0.15))

        val_clusters = []
        test_clusters = []
        train_clusters = []

        cur_val = 0
        cur_test = 0

        for clust in cluster_list:
            sz = cluster_sizes[clust]
            if cur_val + sz <= target_val:
                val_clusters.append(clust)
                cur_val += sz
            elif cur_test + sz <= target_test:
                test_clusters.append(clust)
                cur_test += sz
            else:
                train_clusters.append(clust)

        # Assign splits
        val_mask = class_df["cluster_root"].isin(val_clusters)
        test_mask = class_df["cluster_root"].isin(test_clusters)
        train_mask = class_df["cluster_root"].isin(train_clusters)

        clean_manifest.loc[class_df[val_mask].index, "new_split"] = "val"
        clean_manifest.loc[class_df[test_mask].index, "new_split"] = "test"
        clean_manifest.loc[class_df[train_mask].index, "new_split"] = "train"

    # Verify counts
    split_summary = clean_manifest["new_split"].value_counts().to_dict()
    print(f"\nSplit V2 sample counts: {split_summary}")
    print(f"Total: {sum(split_summary.values())}")
    for s, count in split_summary.items():
        pct = count / len(clean_manifest) * 100
        print(f"  - {s}: {count} ({pct:.2f}%)")

    # 5. Verify Zero Leakage on Split V2
    print("\nVerifying Zero-Leakage on Split V2...")
    train_files = set(clean_manifest[clean_manifest["new_split"] == "train"]["filename"])
    val_files = set(clean_manifest[clean_manifest["new_split"] == "val"]["filename"])
    test_files = set(clean_manifest[clean_manifest["new_split"] == "test"]["filename"])

    train_clusters = set(clean_manifest[clean_manifest["new_split"] == "train"]["cluster_root"])
    val_clusters = set(clean_manifest[clean_manifest["new_split"] == "val"]["cluster_root"])
    test_clusters = set(clean_manifest[clean_manifest["new_split"] == "test"]["cluster_root"])

    c_train_val = len(train_clusters & val_clusters)
    c_train_test = len(train_clusters & test_clusters)
    c_val_test = len(val_clusters & test_clusters)

    print(f"  - Cluster overlap Train & Val: {c_train_val}")
    print(f"  - Cluster overlap Train & Test: {c_train_test}")
    print(f"  - Cluster overlap Val & Test: {c_val_test}")
    assert c_train_val == 0 and c_train_test == 0 and c_val_test == 0, "Cluster leakage detected!"
    print("VERIFIED: 100% of multi-image clusters are strictly contained in single splits!")

    # 6. Save Manifest V2
    clean_manifest["split"] = clean_manifest["new_split"]
    clean_manifest["group_id"] = clean_manifest["cluster_root"].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:12])
    output_df = clean_manifest[[
        "split",
        "unified_label_10",
        "filename",
        "path",
        "sha256",
        "size_bytes",
        "group_id"
    ]].sort_values(by=["split", "unified_label_10", "filename"]).reset_index(drop=True)

    output_df.to_csv(OUTPUT_MANIFEST_V2, index=False)
    print(f"\nSaved Split Manifest V2 ({len(output_df)} rows) to: {OUTPUT_MANIFEST_V2}")

    # 7. Materialize physical directory data/processed_v2 using NTFS hardlinks
    print(f"\nMaterializing physical directory at {OUTPUT_PROCESSED_V2}...")
    if OUTPUT_PROCESSED_V2.exists():
        shutil.rmtree(OUTPUT_PROCESSED_V2)

    for split in ("train", "val", "test"):
        for cls in CLASS_NAMES:
            (OUTPUT_PROCESSED_V2 / split / cls).mkdir(parents=True, exist_ok=True)

    base_dir = Path(r"D:\HK7\Đồ án khóa luận")
    link_count = 0
    for _, row in output_df.iterrows():
        # Source physical path from existing Data/processed or raw
        orig_proc_path = Path(row["path"])
        if not orig_proc_path.exists():
            # fallback to raw path
            raw_p = base_dir / info.set_index("clean_name")["path"].to_dict()[row["filename"]]
            src_file = raw_p
        else:
            src_file = orig_proc_path

        dst_file = OUTPUT_PROCESSED_V2 / row["split"] / row["unified_label_10"] / row["filename"]
        os.link(src_file, dst_file)
        link_count += 1

    print(f"Materialized {link_count} hardlinks in {OUTPUT_PROCESSED_V2} successfully!")

    # 8. Save Split V2 Summary JSON
    class_split_breakdown = {}
    for c in CLASS_NAMES:
        c_df = output_df[output_df["unified_label_10"] == c]
        class_split_breakdown[c] = {
            "train": int((c_df["split"] == "train").sum()),
            "val": int((c_df["split"] == "val").sum()),
            "test": int((c_df["split"] == "test").sum()),
            "total": len(c_df),
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "version": "v2",
        "seed": 42,
        "method": "Stratified Group Split (Atomic Near-Duplicate Component Grouping)",
        "total_samples": len(output_df),
        "quarantined_samples": len(quarantine_records),
        "split_counts": split_summary,
        "cluster_verification": {
            "train_val_cluster_overlap": c_train_val,
            "train_test_cluster_overlap": c_train_test,
            "val_test_cluster_overlap": c_val_test,
            "zero_cluster_leakage": True,
        },
        "class_breakdown": class_split_breakdown,
    }

    summary_path = AUDIT_DIR / "split_v2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved Split V2 summary to: {summary_path}")
    print("=" * 80)
    print("SPLIT V2 GENERATION COMPLETE: ZERO LEAKAGE VERIFIED!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
