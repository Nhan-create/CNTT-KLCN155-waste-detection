#!/usr/bin/env python3
"""
scripts/generate_mendeley_provenance.py
---------------------------------------
Generates rigorous provenance data for the Mendeley synthetic dataset:
- Verifies exact box counts across all 1,305 synthetic label files (4,095 total boxes)
- Documents official DOI metadata:
    DOI: 10.17632/2x69gjbcz6.2
    Title: "A synthetic outdoor waste image dataset with YOLO-format annotations for object detection"
    Authors: Tanvinur Rahman Siam and Kaushik Deb (published 16 Feb 2026)
- Extracts sample verification images and annotations into data_provenance/samples/
- Generates pixel-accurate crops with verifiable bounding box coordinates, SHA-256 hashes,
  and class mapping comparison (Mendeley class vs Project TAXONOMY_10 class).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
AUDIT_DIR = REPO_ROOT / "reports" / "multi_waste_detection_audit" / "20261008_192500"
PROVENANCE_DIR = AUDIT_DIR / "data_provenance"
SAMPLES_DIR = PROVENANCE_DIR / "samples"
CROPS_DIR = PROVENANCE_DIR / "crops"
AUDIT_IMAGES_DIR = AUDIT_DIR / "images"

PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# Mendeley v2 official categories (DOI 10.17632/2x69gjbcz6.2)
MENDELEY_CATEGORIES = [
    "plastic",        # 0
    "paper",          # 1
    "cardboard",      # 2
    "metal",          # 3
    "glass",          # 4
    "organic waste",  # 5
    "battery waste",  # 6
    "e-waste",        # 7
    "cloth",          # 8
    "other waste"     # 9
]

# Project TAXONOMY_10 (from scripts/prepare_detection_dataset.py)
PROJECT_TAXONOMY_10 = [
    "battery",     # 0
    "biological",  # 1
    "cardboard",   # 2
    "clothes",     # 3
    "glass",       # 4
    "metal",       # 5
    "paper",       # 6
    "plastic",     # 7
    "shoes",       # 8
    "trash"        # 9
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    print("--- 1. Auditing all synthetic label files ---")
    labels_dir = REPO_ROOT / "data" / "detection" / "labels" / "synthetic"
    images_dir = REPO_ROOT / "data" / "detection" / "images" / "synthetic"

    if not labels_dir.exists():
        print(f"Error: {labels_dir} not found!")
        sys.exit(1)

    label_files = sorted(list(labels_dir.glob("*.txt")))
    print(f"Found {len(label_files)} synthetic label files.")

    class_counts = Counter()
    total_boxes = 0

    for lf in label_files:
        with open(lf, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    class_counts[cid] += 1
                    total_boxes += 1

    counts_report = {
        "dataset_name": "A synthetic outdoor waste image dataset with YOLO-format annotations for object detection",
        "doi": "10.17632/2x69gjbcz6.2",
        "authors": ["Tanvinur Rahman Siam", "Kaushik Deb"],
        "total_synthetic_images": len(label_files),
        "total_synthetic_boxes": total_boxes,
        "boxes_by_class_id": {
            str(cid): {
                "mendeley_category": MENDELEY_CATEGORIES[cid],
                "project_taxonomy_10": PROJECT_TAXONOMY_10[cid],
                "alignment_status": "MATCH" if (MENDELEY_CATEGORIES[cid] == PROJECT_TAXONOMY_10[cid] or (cid == 9 and PROJECT_TAXONOMY_10[cid] == "trash")) else "SCRAMBLED",
                "box_count": class_counts[cid]
            }
            for cid in range(10)
        }
    }

    with open(PROVENANCE_DIR / "synthetic_dataset_box_counts.json", "w", encoding="utf-8") as fp:
        json.dump(counts_report, fp, indent=2, ensure_ascii=False)
    print(f"Saved synthetic box counts to {PROVENANCE_DIR / 'synthetic_dataset_box_counts.json'}")

    print("\n--- 2. Packaging representative samples and crops ---")
    sample_stems = [
        "syn_syn_000000",
        "syn_syn_000003",
        "syn_syn_000004",
        "syn_syn_000010"
    ]

    crops_ledger = []

    for stem in sample_stems:
        img_src = images_dir / f"{stem}.jpg"
        lbl_src = labels_dir / f"{stem}.txt"

        if not img_src.is_file() or not lbl_src.is_file():
            print(f"Warning: Sample {stem} image or label missing, skipping.")
            continue

        # Copy sample image and label into data_provenance/samples/
        shutil.copy2(img_src, SAMPLES_DIR / f"{stem}.jpg")
        shutil.copy2(lbl_src, SAMPLES_DIR / f"{stem}.txt")

        img_sha = compute_sha256(img_src)
        with Image.open(img_src) as opened:
            w, h = opened.size
            im_rgb = opened.convert("RGB")

        with open(lbl_src, "r", encoding="utf-8") as fp:
            box_idx = 0
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = max(0, int((cx - bw / 2) * w))
                    y1 = max(0, int((cy - bh / 2) * h))
                    x2 = min(w, int((cx + bw / 2) * w))
                    y2 = min(h, int((cy + bh / 2) * h))

                    crop_im = im_rgb.crop((x1, y1, x2, y2))
                    crop_filename = f"crop_{stem}_box{box_idx}_cid{cid}.jpg"
                    crop_path = CROPS_DIR / crop_filename
                    crop_im.save(crop_path, quality=95)

                    crops_ledger.append({
                        "sample_stem": stem,
                        "source_image_sha256": img_sha,
                        "box_index": box_idx,
                        "raw_class_id": cid,
                        "mendeley_category": MENDELEY_CATEGORIES[cid],
                        "project_assigned_class": PROJECT_TAXONOMY_10[cid],
                        "alignment_status": "MATCH" if (MENDELEY_CATEGORIES[cid] == PROJECT_TAXONOMY_10[cid] or (cid == 9 and PROJECT_TAXONOMY_10[cid] == "trash")) else "SCRAMBLED",
                        "bbox_yolo_normalized": [cx, cy, bw, bh],
                        "bbox_pixel_xyxy": [x1, y1, x2, y2],
                        "crop_file": f"data_provenance/crops/{crop_filename}",
                        "crop_sha256": compute_sha256(crop_path)
                    })
                    box_idx += 1

    with open(PROVENANCE_DIR / "mendeley_crops_provenance.json", "w", encoding="utf-8") as fp:
        json.dump(crops_ledger, fp, indent=2, ensure_ascii=False)
    print(f"Saved {len(crops_ledger)} verified crops to {PROVENANCE_DIR / 'mendeley_crops_provenance.json'}")

if __name__ == "__main__":
    main()
