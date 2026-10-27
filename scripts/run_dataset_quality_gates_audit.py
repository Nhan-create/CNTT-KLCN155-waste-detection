# Mô tả: Chạy kiểm toán tự động các cổng chất lượng dữ liệu trên bộ dữ liệu phát hiện.
"""
scripts/run_dataset_quality_gates_audit.py
-----------------------------------------
Executes Quality Gates on real workspace images and annotations.
Audits:
- All 1,996 images in splits_real_v2 (train, val, test)
- The 147 real images not currently in splits
- Exact SHA-256 and perceptual pHash against 27 holdout images in data/holdout

Generates:
- reports/data_v2/real_dataset_quality_gate_audit.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import time
from pathlib import Path
from typing import Dict
from PIL import Image
import imagehash

# Import Quality Gates library
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset_v3_quality_gates import (
    validate_dataset_entry,
    STATUS_VALID,
    STATUS_FLAG_FOR_REVIEW,
    STATUS_INVALID
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "detection"
REPORTS_DIR = ROOT / "reports" / "data_v2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HOLDOUT_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"

print("[1/4] Loading Holdout images for SHA-256 and pHash baseline...")
holdout_sha256 = {}
holdout_phashes = {}

if HOLDOUT_MANIFEST.exists():
    with open(HOLDOUT_MANIFEST, "r", encoding="utf-8") as f:
        hm = json.load(f)
    for entry in hm.get("files", []):
        fn = entry.get("filename")
        sha = entry.get("sha256")
        abs_p = entry.get("absolute_path")
        if fn and sha:
            holdout_sha256[fn] = sha
        if abs_p and Path(abs_p).exists():
            try:
                with Image.open(abs_p) as img:
                    holdout_phashes[fn] = imagehash.phash(img)
            except Exception as e:
                print(f"  Warning: failed to compute pHash for holdout {fn}: {e}")

print(f"  -> Loaded {len(holdout_sha256)} holdout SHA-256 and {len(holdout_phashes)} holdout pHashes.")

# 2. Gather dataset files to audit
print("[2/4] Gathering images to audit...")
splits_images = {}
for sp in ["train", "val", "test"]:
    sp_file = DATA_DIR / "splits_real_v2" / f"{sp}.txt"
    if sp_file.exists():
        with open(sp_file, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    fn = Path(l.strip()).name
                    splits_images[fn] = sp

all_real_imgs = sorted((DATA_DIR / "images" / "real").glob("*.jpg"))
print(f"  -> Total real images on disk: {len(all_real_imgs)}")
print(f"  -> In splits_real_v2: {len(splits_images)}")
print(f"  -> Outside splits_real_v2: {len(all_real_imgs) - len(splits_images)}")

# 3. Execute Quality Gates Audit
print("[3/4] Running Quality Gates validation across real files...")
t0 = time.time()
audit_results = []
status_counts = {STATUS_VALID: 0, STATUS_FLAG_FOR_REVIEW: 0, STATUS_INVALID: 0}
category_counts = {}

for idx, img_p in enumerate(all_real_imgs):
    fn = img_p.name
    lbl_p = (DATA_DIR / "labels" / "real" / fn).with_suffix(".txt")
    
    # Run entry validation
    entry_audit = validate_dataset_entry(
        image_path=img_p,
        label_path=lbl_p if lbl_p.exists() else None,
        is_negative=False,
        num_classes=10,
        holdout_sha256_set=holdout_sha256,
        holdout_phash_set=holdout_phashes
    )
    
    # Add split tag
    entry_audit["split"] = splits_images.get(fn, "unassigned_outside_split")
    
    st = entry_audit["validation_status"]
    status_counts[st] += 1
    cat = entry_audit["error_category"]
    category_counts[cat] = category_counts.get(cat, 0) + 1
    
    audit_results.append(entry_audit)
    
    if (idx + 1) % 500 == 0 or (idx + 1) == len(all_real_imgs):
        print(f"  Progress: {idx + 1}/{len(all_real_imgs)} audited... ({time.time() - t0:.1f}s)")

elapsed = time.time() - t0
print(f"\n[4/4] Audit Complete in {elapsed:.2f}s:")
print(f"  VALID:           {status_counts[STATUS_VALID]}")
print(f"  FLAG_FOR_REVIEW: {status_counts[STATUS_FLAG_FOR_REVIEW]}")
print(f"  INVALID:         {status_counts[STATUS_INVALID]}")
print(f"Category breakdown: {category_counts}")

output_report = {
    "audit_metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_images_audited": len(all_real_imgs),
        "in_splits_count": len(splits_images),
        "outside_splits_count": len(all_real_imgs) - len(splits_images),
        "holdout_leakage_check": {
            "holdout_count": len(holdout_sha256),
            "holdout_phash_count": len(holdout_phashes),
            "max_hamming_threshold": 8
        },
        "duration_seconds": round(elapsed, 2)
    },
    "summary_counts": {
        "status": status_counts,
        "categories": category_counts
    },
    "flagged_and_invalid_entries": [
        res for res in audit_results if res["validation_status"] != STATUS_VALID
    ],
    "all_entries": audit_results
}

out_file = REPORTS_DIR / "real_dataset_quality_gate_audit.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output_report, f, indent=2, ensure_ascii=False)

print(f"\nSaved structured audit report to: {out_file} ({out_file.stat().st_size:,} bytes)")
