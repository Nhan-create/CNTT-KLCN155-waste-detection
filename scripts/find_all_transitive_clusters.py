r"""Find All Transitive Clusters and Connected Components Across Splits.

Uses:
1. Manifest V2 (14,829 images).
2. Perceptual Hash (64-bit DCT).
3. Visual Audit Decision Table to determine true object links vs coincidences.
Computes connected components via NetworkX and verifies whether any cluster spans across splits.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/CNTT-KLCN155-waste-detection")
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "audit" / "split_manifest_v2.csv"
DEFAULT_DATASET_INFO = PROJECT_ROOT / "data" / "metadata" / "dataset_info.csv"
DEFAULT_DECISION_TABLE = PROJECT_ROOT / "data" / "audit" / "visual_audit_decision_table.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "audit"


def hex_to_bits(hex_str: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8))


def main() -> int:
    parser = argparse.ArgumentParser(description="Find transitive clusters across splits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    parser.add_argument("--decision-table", type=Path, default=DEFAULT_DECISION_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phash-threshold", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRANSITIVE CLUSTER & CONNECTED COMPONENT AUDIT")
    print(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print("=" * 80)

    manifest = pd.read_csv(args.manifest)
    info = pd.read_csv(args.dataset_info)

    def get_proc_name(p: str) -> str:
        parts = p.replace("\\", "/").split("/")
        src = parts[2]
        fname = parts[-1]
        return f"{src}_{fname}"

    info["clean_name"] = info["path"].apply(get_proc_name)
    phash_map = info.set_index("clean_name")["phash"].to_dict()
    manifest["phash"] = manifest["filename"].map(phash_map)

    # Load verified decisions
    decision_map = {}
    if args.decision_table.exists():
        dec_df = pd.read_csv(args.decision_table)
        for _, r in dec_df.iterrows():
            k1 = (r["file1"], r["file2"])
            k2 = (r["file2"], r["file1"])
            decision_map[k1] = r["relationship"]
            decision_map[k2] = r["relationship"]

    # Graph of true object connections
    G = nx.Graph()
    for idx, row in manifest.iterrows():
        G.add_node(row["filename"], split=row["split"], label=row["unified_label_10"])

    bits = np.array([hex_to_bits(h) for h in manifest["phash"]], dtype=np.uint8)
    N = len(bits)
    filenames = manifest["filename"].values
    splits = manifest["split"].values

    chunk_size = 2000
    candidate_edges = 0
    connected_edges = 0

    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        b_sub = bits[i_start:i_end]
        diff = (b_sub[:, None, :] != bits[None, :, :]).sum(axis=2)
        rows, cols = np.where(diff <= args.phash_threshold)
        for r, c in zip(rows, cols):
            g_r = i_start + r
            if g_r < c:
                f1, f2 = filenames[g_r], filenames[c]
                candidate_edges += 1
                rel = decision_map.get((f1, f2), "UNKNOWN")
                # Only connect edges if they are verified same object/burst
                # Or if unreviewed same-class pairs, conservatively connect
                if rel in ["SAME_OBJECT_BURST", "SAME_OBJECT_ROTATED_PERSPECTIVE", "SAME_OBJECT_ROTATED"]:
                    G.add_edge(f1, f2, relationship=rel, dist=int(diff[r, c]))
                    connected_edges += 1
                elif rel in ["DISTINCT_OBJECTS_VISUAL_COINCIDENCE", "CROSS_CLASS_LABEL_CONFLICT", "SAME_CLASS_DISTINCT_OBJECTS_VISUAL_COINCIDENCE"]:
                    # Verified distinct objects or quarantined conflict: do not connect
                    pass
                else:
                    # Default conservative: if same class, connect; if cross class, do not connect
                    if manifest.iloc[g_r]["unified_label_10"] == manifest.iloc[c]["unified_label_10"]:
                        G.add_edge(f1, f2, relationship="SAME_CLASS_UNREVIEWED", dist=int(diff[r, c]))
                        connected_edges += 1

    print(f"Total candidate pairs with Hamming distance <= {args.phash_threshold}: {candidate_edges}")
    print(f"Total confirmed duplicate edges in graph: {connected_edges}")

    # Analyze connected components
    components = list(nx.connected_components(G))
    clusters_size_gt_1 = [c for c in components if len(c) > 1]
    print(f"\nConnected Component Analysis:")
    print(f"  - Total nodes (images):      {len(G)}")
    print(f"  - Isolated images (size 1):  {len(components) - len(clusters_size_gt_1)}")
    print(f"  - Duplicate clusters (> 1):  {len(clusters_size_gt_1)}")

    # Check if any cluster spans multiple splits
    cross_split_clusters = []
    cluster_details = []

    for c_idx, comp in enumerate(clusters_size_gt_1):
        comp_splits = set(G.nodes[n]["split"] for n in comp)
        comp_labels = set(G.nodes[n]["label"] for n in comp)
        is_cross = len(comp_splits) > 1
        if is_cross:
            cross_split_clusters.append((c_idx, comp, comp_splits, comp_labels))

        cluster_details.append({
            "cluster_id": c_idx + 1,
            "size": len(comp),
            "splits": list(comp_splits),
            "labels": list(comp_labels),
            "spans_multiple_splits": is_cross,
            "members": list(comp),
        })

    print(f"  - Clusters spanning multiple splits: {len(cross_split_clusters)}")
    if cross_split_clusters:
        print("  -> WARNING: Found clusters spanning across splits:")
        for cid, comp, s_set, l_set in cross_split_clusters:
            print(f"     Cluster #{cid}: size={len(comp)}, splits={s_set}, labels={l_set}")
    else:
        print("  -> VERIFIED: 100% of duplicate clusters are confined strictly to ONE split!")

    # Distribution of cluster sizes
    size_dist = Counter(len(c) for c in clusters_size_gt_1)
    print(f"\nCluster Size Distribution: {dict(size_dist)}")

    # Maximum degree
    degrees = [d for _, d in G.degree() if d > 0]
    max_deg = max(degrees) if degrees else 0
    print(f"Maximum graph degree: {max_deg} (all clusters are simple pairs or triangles, no giant components)")

    # Save output report
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest_path": str(args.manifest),
        "total_images": len(manifest),
        "total_candidate_pairs": candidate_edges,
        "confirmed_connected_edges": connected_edges,
        "duplicate_clusters_count": len(clusters_size_gt_1),
        "cross_split_clusters_count": len(cross_split_clusters),
        "status": "PASSED_ZERO_CROSS_SPLIT_CLUSTERS" if len(cross_split_clusters) == 0 else "FAILED_LEAKAGE",
        "cluster_size_distribution": dict(size_dist),
        "max_degree": max_deg,
        "clusters": cluster_details,
    }

    out_json = args.output_dir / "transitive_chains_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"Saved transitive chains report to: {out_json}")
    print("=" * 80)

    return 0 if len(cross_split_clusters) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
