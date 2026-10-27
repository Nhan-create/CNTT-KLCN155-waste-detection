"""
scripts/build_phase4a_delivery.py
---------------------------------
Packages all Phase 4A verified artifacts, reports, raw predictions, GT files,
evaluators, patch evidence, and test logs into:
~/Downloads/PHASE4A_DATASET_PREPARATION_EVIDENCE.zip
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import zipfile
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
ZIP_OUTPUT = DOWNLOADS_DIR / "PHASE4A_DATASET_PREPARATION_EVIDENCE.zip"

FILES_TO_PACK = [
    # 1. Master Reports
    ("reports/PHASE4A_DATASET_PREPARATION.md", "PHASE4A_DATASET_PREPARATION.md"),
    ("reports/ANTI_PHASE3_FINAL_VERIFICATION.md", "reports/ANTI_PHASE3_FINAL_VERIFICATION.md"),
    ("reports/GT_V2_REVIEW_FOR_APPROVAL.md", "reports/GT_V2_REVIEW_FOR_APPROVAL.md"),
    
    # 2. Raw Predictions & Benchmarks
    ("reports/qa_acceptance/raw_predictions_before_quick_fix.json", "qa_acceptance/raw_predictions_before_quick_fix.json"),
    ("reports/qa_acceptance/raw_predictions_after_quick_fix.json", "qa_acceptance/raw_predictions_after_quick_fix.json"),
    ("reports/qa_acceptance/raw_predictions_yolov8n_e1.json", "qa_acceptance/raw_predictions_yolov8n_e1.json"),
    ("reports/qa_acceptance/verified_evaluation_summary_from_raw.json", "qa_acceptance/verified_evaluation_summary_from_raw.json"),
    
    # 3. Ground Truth Definitions
    ("reports/qa_acceptance/ground_truth_hinh_anh_rac.json", "qa_acceptance/ground_truth_hinh_anh_rac_v1_frozen.json"),
    ("reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json", "qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json"),
    
    # 4. Evaluator & Quality Gates Code
    ("scripts/compare_eval_v1_vs_v2.py", "scripts/compare_eval_v1_vs_v2.py"),
    ("scripts/validate_dataset_v3_quality_gates.py", "scripts/validate_dataset_v3_quality_gates.py"),
    ("scripts/build_dataset_v3_gap_analysis.py", "scripts/build_dataset_v3_gap_analysis.py"),
    ("scripts/audit_splits_real_v2_distribution.py", "scripts/audit_splits_real_v2_distribution.py"),
    ("tests/qa/test_dataset_v3_quality_gates.py", "tests/qa/test_dataset_v3_quality_gates.py"),
    ("tests/qa/test_compare_eval_v1_vs_v2.py", "tests/qa/test_compare_eval_v1_vs_v2.py"),
    ("tests/qa/test_ground_truth_v2_integrity.py", "tests/qa/test_ground_truth_v2_integrity.py"),
    
    # 5. Patch Evidence
    ("reports/patches/patch_oi_2734e597675d8a97.diff", "patches/patch_oi_2734e597675d8a97.diff"),
    ("reports/patches/patch_oi_2734e597675d8a97_evidence.json", "patches/patch_oi_2734e597675d8a97_evidence.json"),
    
    # 6. Dataset v3 Preparation & Manifests
    ("reports/data_v2/dataset_v3_gap_analysis.json", "data_v3/dataset_v3_gap_analysis.json"),
    ("configs/detection_dataset_real_v3_candidate.yaml", "configs/detection_dataset_real_v3_candidate.yaml"),
    ("data/detection/candidate_hard_negatives_manifest.csv", "data_v3/candidate_hard_negatives_manifest.csv"),
    
    # 7. Test Evidence
    ("reports/test_evidence/tests_summary.json", "test_evidence/tests_summary.json"),
    ("reports/test_evidence/pytest_collection.log", "test_evidence/pytest_collection.log"),
    ("reports/test_evidence/pytest_targeted.log", "test_evidence/pytest_targeted.log"),
    ("reports/test_evidence/python_environment.json", "test_evidence/python_environment.json"),
]

# Add disputed objects visual overlays if available
overlay_dir = ROOT / "reports" / "qa_acceptance" / "gt_review_overlays"
if overlay_dir.exists():
    disputed_images = [
        "review_images (12).jpg.jpg",
        "review_images (10).jpg.jpg",
        "review_OIP (1).webp.jpg",
        "review_Phan-Loai-Rac-Tai-Ng-01.jpg.jpg",
        "review_images (18).jpg.jpg",
        "review_images (8).jpg.jpg"
    ]
    for d_img in disputed_images:
        p = overlay_dir / d_img
        if p.exists():
            FILES_TO_PACK.append((str(p.relative_to(ROOT)), f"visual_overlays/{d_img}"))

print(f"Building Phase 4A Delivery Package: {ZIP_OUTPUT}")
packed_count = 0
with zipfile.ZipFile(ZIP_OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for rel_src, arc_name in FILES_TO_PACK:
        src_path = ROOT / rel_src
        if src_path.exists():
            zipf.write(src_path, arc_name)
            packed_count += 1
            print(f"  + Added: {arc_name} ({src_path.stat().st_size:,} bytes)")
        else:
            print(f"  ! WARNING: Missing {src_path}")

# Compute SHA-256 of resulting zip
sha256 = hashlib.sha256(ZIP_OUTPUT.read_bytes()).hexdigest()
size_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)

print("\n" + "="*70)
print(f"DELIVERY PACKAGE CREATED SUCCESSFULLY!")
print(f"Location: {ZIP_OUTPUT}")
print(f"Files Packed: {packed_count}")
print(f"Size: {size_mb:.2f} MB ({ZIP_OUTPUT.stat().st_size:,} bytes)")
print(f"SHA-256: {sha256}")
print("="*70)
