"""Find All Near-Duplicate Clusters and Transitive Chains.
Computes bitwise Hamming distance <= 4 on all clean images.
Identifies:
1. Cross-split pairs vs Within-split pairs.
2. Connected components (transitive chains) that span across splits.
3. Quantifies how many clusters cross splits in the legacy split.
"""

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import networkx as nx

MANIFEST_PATH = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\split_manifest.csv")
DATASET_INFO_PATH = Path(r"D:\HK7\Đồ án khóa luận\Data\metadata\dataset_info.csv")
OUTPUT_DIR = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit")

def hex_to_bits(hex_str: str) -> np.ndarray:
    byte_vals = bytes.fromhex(hex_str)
    return np.unpackbits(np.frombuffer(byte_vals, dtype=np.uint8))

def main():
    print("=" * 80)
    print("TRANSITIVE CHAIN & CONNECTED COMPONENT AUDIT (pHash <= 4)")
    print("=" * 80)
    
    manifest = pd.read_csv(MANIFEST_PATH)
    info = pd.read_csv(DATASET_INFO_PATH)
    
    def get_proc_name(row):
        parts = row["path"].replace("\\", "/").split("/")
        src = parts[2]
        fname = parts[-1]
        return f"{src}_{fname}"
        
    info["clean_name"] = info.apply(get_proc_name, axis=1)
    phash_dict = info.set_index("clean_name")["phash"].to_dict()
    
    manifest["phash"] = manifest["filename"].map(phash_dict)
    valid_manifest = manifest.dropna(subset=["phash"]).copy().reset_index(drop=True)
    print(f"Total manifest rows: {len(manifest)} | With valid pHash: {len(valid_manifest)}")
    
    # Check if any image is missing pHash
    missing = manifest[manifest["phash"].isna()]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images missing pHash in dataset_info!")
    else:
        print("VERIFIED: 100% of manifest images have valid pHash.")
        
    # Vectorized bit unpack
    bits = np.array([hex_to_bits(h) for h in valid_manifest["phash"]], dtype=np.uint8)
    splits = valid_manifest["split"].values
    filenames = valid_manifest["filename"].values
    classes = valid_manifest["unified_label_10"].values
    
    print("\nScanning pairwise Hamming distances (grouped by class to optimize memory)...")
    
    G = nx.Graph()
    # Add all nodes
    for i, fn in enumerate(filenames):
        G.add_node(fn, split=splits[i], class_name=classes[i], index=i)
        
    cross_split_edges = []
    within_split_edges = []
    
    # Process by class first (most burst shots are within class)
    unique_classes = np.unique(classes)
    for c in unique_classes:
        c_mask = np.where(classes == c)[0]
        c_bits = bits[c_mask]
        c_splits = splits[c_mask]
        c_files = filenames[c_mask]
        n_c = len(c_mask)
        
        # Chunked dot-product for Hamming distance
        chunk_size = 500
        for i_start in range(0, n_c, chunk_size):
            i_end = min(i_start + chunk_size, n_c)
            sub_bits = c_bits[i_start:i_end]
            # dist = 64 - 2*(sub_bits @ c_bits.T) + sum_bits... or XOR:
            # using unpacked uint8: sum of abs difference
            diff = np.bitwise_xor(sub_bits[:, None, :], c_bits[None, :, :]).sum(axis=2)
            
            rows, cols = np.where(diff <= 4)
            for r, col in zip(rows, cols):
                global_r = i_start + r
                global_c = col
                if global_r < global_c: # upper triangle
                    f1, f2 = c_files[global_r], c_files[global_c]
                    s1, s2 = c_splits[global_r], c_splits[global_c]
                    d = int(diff[r, col])
                    G.add_edge(f1, f2, distance=d, same_class=True)
                    if s1 != s2:
                        cross_split_edges.append((f1, f2, s1, s2, d, c))
                    else:
                        within_split_edges.append((f1, f2, s1, s2, d, c))
                        
    print(f"\n[Results within same classes]")
    print(f"  - Total edges (pHash <= 4): {G.number_of_edges()}")
    print(f"  - Cross-split edges: {len(cross_split_edges)}")
    print(f"  - Within-split edges: {len(within_split_edges)}")
    
    # Connected components
    components = list(nx.connected_components(G))
    multi_node_components = [c for c in components if len(c) > 1]
    print(f"  - Total connected components: {len(components)}")
    print(f"  - Multi-image clusters: {len(multi_node_components)}")
    
    # Analyze clusters that cross splits
    cross_split_clusters = []
    for comp in multi_node_components:
        comp_splits = set(valid_manifest[valid_manifest["filename"].isin(comp)]["split"])
        if len(comp_splits) > 1:
            cross_split_clusters.append({
                "cluster_size": len(comp),
                "splits_involved": list(comp_splits),
                "sample_files": list(comp)[:5]
            })
            
    print(f"  - Clusters spanning across splits in legacy split: {len(cross_split_clusters)}")
    for i, c in enumerate(cross_split_clusters[:10]):
        print(f"    * Cluster {i+1}: size={c['cluster_size']}, splits={c['splits_involved']}")
        
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_images": len(valid_manifest),
        "total_edges_le_4": G.number_of_edges(),
        "cross_split_edges": len(cross_split_edges),
        "within_split_edges": len(within_split_edges),
        "multi_node_clusters": len(multi_node_components),
        "cross_split_clusters_count": len(cross_split_clusters),
        "cross_split_clusters_details": cross_split_clusters
    }
    
    with open(OUTPUT_DIR / "transitive_chains_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved report to {OUTPUT_DIR / 'transitive_chains_report.json'}")

if __name__ == "__main__":
    main()
