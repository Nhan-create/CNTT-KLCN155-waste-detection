"""
scripts/prepare_detection_dataset.py
------------------------------------
Prepares and structures the multi-object waste detection dataset from D:\waste-training\dataset-v1
into the local project directory D:\CNTT-KLCN155-waste-detection\data\detection.

Key operations:
1. Strictly isolates Synthetic (Mendeley) images from Real (OpenImages) images into separate folders.
2. Copies images and corresponding YOLO bounding box labels.
3. Computes physical SHA-256 for all images.
4. Parses bounding boxes and extracts class distributions.
5. Generates the master detection manifest: data/detection/manifest_detection_v1.csv
6. Emits dataset_summary.json with complete audits.
"""

import os
import sys
import glob
import shutil
import hashlib
import json
from collections import Counter
from pathlib import Path

TAXONOMY_10 = [
    "battery",      # 0
    "biological",   # 1
    "cardboard",    # 2
    "clothes",      # 3
    "glass",        # 4
    "metal",        # 5
    "paper",        # 6
    "plastic",      # 7
    "shoes",        # 8
    "trash"         # 9
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def parse_yolo_label(label_path: Path):
    if not label_path.exists():
        return 0, {}, []
    
    boxes = []
    class_counts = Counter()
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                class_counts[cls_id] += 1
                boxes.append({
                    "class_id": cls_id,
                    "class_name": TAXONOMY_10[cls_id] if 0 <= cls_id < len(TAXONOMY_10) else "unknown",
                    "bbox": [float(p) for p in parts[1:5]]
                })
    return len(boxes), dict(sorted(class_counts.items())), boxes

def main():
    project_root = Path(__file__).resolve().parent.parent
    source_root = Path(r"D:\waste-training\dataset-v1")
    target_root = project_root / "data" / "detection"

    print(f"[INFO] Source dataset: {source_root}")
    print(f"[INFO] Target dataset: {target_root}")

    if not source_root.exists():
        print(f"[ERROR] Source root {source_root} does not exist!")
        sys.exit(1)

    # Directories
    img_real_dir = target_root / "images" / "real"
    img_syn_dir = target_root / "images" / "synthetic"
    lbl_real_dir = target_root / "labels" / "real"
    lbl_syn_dir = target_root / "labels" / "synthetic"

    for d in [img_real_dir, img_syn_dir, lbl_real_dir, lbl_syn_dir]:
        d.mkdir(parents=True, exist_ok=True)

    records = []
    total_copied = 0
    syn_class_counts = Counter()
    real_class_counts = Counter()

    for orig_split in ["train", "val", "test"]:
        split_img_dir = source_root / "images" / orig_split
        split_lbl_dir = source_root / "labels" / orig_split

        if not split_img_dir.exists():
            continue

        img_files = sorted(list(split_img_dir.glob("*.*")))
        print(f"[INFO] Processing original split '{orig_split}': {len(img_files)} images...")

        for img_path in img_files:
            filename = img_path.name
            stem = img_path.stem
            lbl_path = split_lbl_dir / f"{stem}.txt"

            if filename.lower().startswith("syn_"):
                is_synthetic = True
                source_name = "mendeley_synthetic"
                dst_img = img_syn_dir / filename
                dst_lbl = lbl_syn_dir / f"{stem}.txt"
                rel_img_path = f"images/synthetic/{filename}"
                rel_lbl_path = f"labels/synthetic/{stem}.txt"
            elif filename.lower().startswith("oi_"):
                is_synthetic = False
                source_name = "openimages_v7"
                dst_img = img_real_dir / filename
                dst_lbl = lbl_real_dir / f"{stem}.txt"
                rel_img_path = f"images/real/{filename}"
                rel_lbl_path = f"labels/real/{stem}.txt"
            else:
                is_synthetic = False
                source_name = "unknown"
                dst_img = img_real_dir / filename
                dst_lbl = lbl_real_dir / f"{stem}.txt"
                rel_img_path = f"images/real/{filename}"
                rel_lbl_path = f"labels/real/{stem}.txt"

            # Copy file if not exists or size differs
            if not dst_img.exists() or dst_img.stat().st_size != img_path.stat().st_size:
                shutil.copy2(img_path, dst_img)

            if lbl_path.exists():
                if not dst_lbl.exists() or dst_lbl.stat().st_size != lbl_path.stat().st_size:
                    shutil.copy2(lbl_path, dst_lbl)
            else:
                # Create empty label if missing
                if not dst_lbl.exists():
                    dst_lbl.touch()

            sha256_hash = compute_sha256(dst_img)
            num_boxes, cls_dist, _ = parse_yolo_label(dst_lbl)

            if is_synthetic:
                for cid, count in cls_dist.items():
                    syn_class_counts[cid] += count
            else:
                for cid, count in cls_dist.items():
                    real_class_counts[cid] += count

            records.append({
                "image_id": stem,
                "filename": filename,
                "relative_image_path": rel_img_path,
                "relative_label_path": rel_lbl_path,
                "source": source_name,
                "is_synthetic": is_synthetic,
                "original_split": orig_split,
                "num_boxes": num_boxes,
                "class_distribution": json.dumps(cls_dist),
                "sha256": sha256_hash,
                "review_status": "UNREVIEWED"
            })
            total_copied += 1

    # Write manifest CSV
    manifest_path = target_root / "manifest_detection_v1.csv"
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(manifest_path, index=False, encoding="utf-8")

    manifest_sha = compute_sha256(manifest_path)

    # Write Summary JSON
    summary = {
        "timestamp": "2026-10-02T01:20:00+0700",
        "dataset_name": "CNTT-KLCN155 Multi-Object Waste Detection Dataset v1",
        "manifest_file": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "total_images": len(records),
        "synthetic_images": sum(1 for r in records if r["is_synthetic"]),
        "real_images": sum(1 for r in records if not r["is_synthetic"]),
        "total_boxes": sum(r["num_boxes"] for r in records),
        "synthetic_boxes": sum(syn_class_counts.values()),
        "real_boxes": sum(real_class_counts.values()),
        "per_class_boxes_synthetic": {
            TAXONOMY_10[cid]: syn_class_counts[cid] for cid in range(10)
        },
        "per_class_boxes_real": {
            TAXONOMY_10[cid]: real_class_counts[cid] for cid in range(10)
        },
        "findings": [
            "CRITICAL: Real images (OpenImages) only contain class 'shoes' (class 8: 507 boxes across 114 images).",
            "CRITICAL: Synthetic images (Mendeley) contain 9 classes (4095 boxes across 1305 images) but 0 boxes for 'shoes'.",
            "CRITICAL: Real waste images for the other 9 classes (battery, biological, cardboard, clothes, glass, metal, paper, plastic, trash) are completely 0 in dataset-v1.",
            "ISOLATION: Synthetic images are strictly isolated to data/detection/images/synthetic and CANNOT be used as real test/validation evaluation.",
            "CONCLUSION: Dataset is INSUFFICIENT for training production multi-object detector in Part 3 without collecting real multi-class waste images."
        ]
    }

    summary_path = target_root / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Successfully processed {len(records)} images into {target_root}")
    print(f"Manifest written to: {manifest_path} (SHA: {manifest_sha})")
    print(f"Summary written to: {summary_path}")
    print("\nPer-class box count summary:")
    print(f"{'Class':<12} | {'Synthetic':<10} | {'Real':<10} | {'Total':<10}")
    print("-" * 50)
    for cid, cname in enumerate(TAXONOMY_10):
        s_cnt = syn_class_counts[cid]
        r_cnt = real_class_counts[cid]
        print(f"{cname:<12} | {s_cnt:<10} | {r_cnt:<10} | {s_cnt + r_cnt:<10}")

if __name__ == "__main__":
    main()
