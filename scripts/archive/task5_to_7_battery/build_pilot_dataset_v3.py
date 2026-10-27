"""
scripts/build_pilot_dataset_v3.py
---------------------------------
Builds the isolated Pilot Dataset v3 in data/detection_pilot_v3/:
1. Independent directory isolation (images/, labels/, manifests/).
2. Never modifies or overwrites Dataset v2, Dataset v3 candidate, or Holdout.
3. Ingests QA re-annotated biological instances (taco_0072, taco_0092) with PROPOSED_PENDING_USER_APPROVAL status.
4. Identifies existing split membership (taco_0068 & taco_0082 in TEST; taco_0841, taco_0924, taco_0456 in TRAIN).
5. Documents the 8 unmapped Rope & strings images as REJECTED_AS_NEGATIVE.
6. Emits SHA-256 and pHash manifests with honest, transparent QA statuses.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parents[3]
SRC_IMG_DIR = ROOT / "data" / "detection" / "images" / "real"
SRC_LBL_DIR = ROOT / "data" / "detection" / "labels" / "real"

PILOT_ROOT = ROOT / "data" / "detection_pilot_v3"
PILOT_IMG_DIR = PILOT_ROOT / "images"
PILOT_LBL_DIR = PILOT_ROOT / "labels"
PILOT_MAN_DIR = PILOT_ROOT / "manifests"

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

PILOT_ITEMS = [
    # Biological (re-annotated proposed)
    {
        "filename": "taco_0072.jpg",
        "category": "QUARANTINED",
        "official_split": "quarantined",
        "custom_labels": [
            "7 0.428500 0.485000 0.585000 0.510000",
            "1 0.478000 0.432000 0.385000 0.285000"
        ],
        "qa_status": "PROPOSED_PENDING_USER_APPROVAL",
        "qa_notes": "Recovered from quarantine. Apple core boundary tightened; plastic packaging isolated."
    },
    {
        "filename": "taco_0092.jpg",
        "category": "QUARANTINED",
        "official_split": "quarantined",
        "custom_labels": [
            "1 0.475000 0.648000 0.178000 0.215000"
        ],
        "qa_status": "PROPOSED_PENDING_USER_APPROVAL",
        "qa_notes": "Recovered from quarantine. Concrete background padding removed."
    },
    # Biological (verified context from splits)
    {
        "filename": "taco_0068.jpg",
        "category": "EXISTING_TEST",
        "official_split": "test",
        "custom_labels": None,
        "qa_status": "VERIFIED_EXISTING_TEST_DO_NOT_MERGE",
        "qa_notes": "Belongs to TEST split. Preserved in test to prevent test contamination."
    },
    {
        "filename": "taco_0841.jpg",
        "category": "EXISTING_TRAIN",
        "official_split": "train",
        "custom_labels": None,
        "qa_status": "VERIFIED_EXISTING_TRAIN",
        "qa_notes": "Already part of official train split."
    },
    {
        "filename": "taco_0924.jpg",
        "category": "EXISTING_TRAIN",
        "official_split": "train",
        "custom_labels": None,
        "qa_status": "VERIFIED_EXISTING_TRAIN",
        "qa_notes": "Already part of official train split."
    },
    # Battery (verified context from splits)
    {
        "filename": "taco_0456.jpg",
        "category": "EXISTING_TRAIN",
        "official_split": "train",
        "custom_labels": None,
        "qa_status": "VERIFIED_EXISTING_TRAIN",
        "qa_notes": "Already part of official train split (1 battery)."
    },
    {
        "filename": "taco_0082.jpg",
        "category": "EXISTING_TEST",
        "official_split": "test",
        "custom_labels": None,
        "qa_status": "VERIFIED_EXISTING_TEST_DO_NOT_MERGE",
        "qa_notes": "Contains genuine button cell battery (Anno 207) but belongs to TEST split. Do not merge to train."
    },
    # Unmapped waste images previously thought to be hard negatives
    {
        "filename": "taco_0170.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 578: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0431.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 1551: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0556.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 1786: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0909.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 2817: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0912.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 2820: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0916.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 2826, 2827: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0917.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 2828: Category 51 (Rope & strings). Not a clean negative."
    },
    {
        "filename": "taco_0925.jpg",
        "category": "EMPTY_LABEL_UNMAPPED_WASTE",
        "official_split": "none",
        "custom_labels": [],
        "qa_status": "REJECTED_AS_NEGATIVE_CONTAINS_ROPE_WASTE",
        "qa_notes": "Raw COCO Anno 2839: Category 51 (Rope & strings). Not a clean negative."
    }
]

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    print("=== REBUILDING PILOT DATASET V3 WITH HONEST METADATA ===")
    PILOT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_MAN_DIR.mkdir(parents=True, exist_ok=True)

    image_manifest = []
    annotation_manifest = []
    anno_counter = 1

    for item in PILOT_ITEMS:
        fn = item["filename"]
        src_img = SRC_IMG_DIR / fn
        dst_img = PILOT_IMG_DIR / fn
        shutil.copy2(src_img, dst_img)

        img_bytes = dst_img.read_bytes()
        sha256 = sha256_bytes(img_bytes)
        with Image.open(dst_img) as im:
            w, h = im.size
            phash = str(imagehash.phash(im.convert("RGB")))

        dst_lbl = PILOT_LBL_DIR / f"{Path(fn).stem}.txt"
        if item["custom_labels"] is not None:
            if len(item["custom_labels"]) == 0:
                dst_lbl.write_text("", encoding="utf-8")
            else:
                dst_lbl.write_text("\n".join(item["custom_labels"]) + "\n", encoding="utf-8")
        else:
            src_lbl = SRC_LBL_DIR / f"{Path(fn).stem}.txt"
            shutil.copy2(src_lbl, dst_lbl)

        lbl_content = dst_lbl.read_text(encoding="utf-8").strip()
        lines = [ln.strip() for ln in lbl_content.splitlines() if ln.strip()]
        boxes_in_image = len(lines)

        for ln in lines:
            parts = ln.split()
            cid = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"unknown_{cid}"
            area = bw * bh

            is_proposed = item["category"] == "QUARANTINED"
            annotation_manifest.append({
                "annotation_id": f"anno_{anno_counter:04d}",
                "image_filename": fn,
                "class_id": cid,
                "class_name": cname,
                "bbox_normalized": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
                "area_normalized": round(area, 6),
                "is_reannotated": is_proposed,
                "qa_approval_status": "PROPOSED_PENDING_USER_APPROVAL" if is_proposed else "EXISTING_SPLIT_RECORD"
            })
            anno_counter += 1

        image_manifest.append({
            "filename": fn,
            "width": w,
            "height": h,
            "sha256": sha256,
            "phash": phash,
            "category": item["category"],
            "official_split": item["official_split"],
            "boxes_count": boxes_in_image,
            "is_hard_negative": False,  # Corrected: None of them are verified negatives!
            "qa_notes": item["qa_notes"],
            "qa_approval_status": item["qa_status"]
        })

    # Save manifests
    img_man_path = PILOT_MAN_DIR / "pilot_image_manifest.json"
    with open(img_man_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "generated_at_utc": "2026-10-12T02:00:00Z",
                "pilot_version": "v3-pilot-reconciled",
                "total_images": len(image_manifest),
                "total_annotations": len(annotation_manifest),
                "existing_train_images": sum(1 for im in image_manifest if im["category"] == "EXISTING_TRAIN"),
                "existing_test_images": sum(1 for im in image_manifest if im["category"] == "EXISTING_TEST"),
                "quarantined_proposed_images": sum(1 for im in image_manifest if im["category"] == "QUARANTINED"),
                "rejected_negative_candidates": sum(1 for im in image_manifest if im["category"] == "EMPTY_LABEL_UNMAPPED_WASTE"),
                "verified_true_negatives": 0,
                "eligible_new_train_images": 2,
                "policy": "KEEP_PILOT_ISOLATED. Zero automatic merging into official train."
            },
            "images": image_manifest
        }, f, indent=2, ensure_ascii=False)

    anno_man_path = PILOT_MAN_DIR / "pilot_annotation_manifest.json"
    with open(anno_man_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "generated_at_utc": "2026-10-12T02:00:00Z",
                "total_annotations": len(annotation_manifest),
                "class_distribution": {
                    cname: sum(1 for a in annotation_manifest if a["class_name"] == cname)
                    for cname in CLASS_NAMES
                }
            },
            "annotations": annotation_manifest
        }, f, indent=2, ensure_ascii=False)

    print(f"Rebuilt Pilot Dataset Manifests at: {PILOT_MAN_DIR}")

if __name__ == "__main__":
    main()
