# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Đóng gói hồ sơ minh chứng, manifest và tệp nén bàn giao cho `build_phase4b_delivery`.
"""
scripts/build_phase4b_delivery.py
---------------------------------
Packages all Phase 4B verified artifacts, source code, regression tests,
logs, manifests, YOLO config, and dry-run report into:
~/Downloads/PHASE4B_DATASET_BUILD_EVIDENCE.zip
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
ZIP_OUTPUT = DOWNLOADS_DIR / "PHASE4B_DATASET_BUILD_EVIDENCE.zip"

FILES_TO_PACK = [
    # 1. Master Reports
    ("reports/PHASE4B_DATASET_BUILD_READINESS.md", "PHASE4B_DATASET_BUILD_READINESS.md"),
    ("reports/PHASE4A_DATASET_PREPARATION.md", "reports/PHASE4A_DATASET_PREPARATION.md"),
    
    # 2. Source Code Changes
    ("scripts/validate_dataset_v3_quality_gates.py", "scripts/validate_dataset_v3_quality_gates.py"),
    ("scripts/run_dataset_quality_gates_audit.py", "scripts/run_dataset_quality_gates_audit.py"),
    ("scripts/build_dataset_v3_candidate.py", "scripts/build_dataset_v3_candidate.py"),
    ("scripts/verify_yolo_v3_compatibility.py", "scripts/verify_yolo_v3_compatibility.py"),
    ("scripts/verify_training_dry_run_environment.py", "scripts/verify_training_dry_run_environment.py"),
    
    # 3. Regression Tests
    ("tests/qa/test_dataset_v3_quality_gates.py", "tests/qa/test_dataset_v3_quality_gates.py"),
    ("tests/qa/test_yolo_v3_compatibility.py", "tests/qa/test_yolo_v3_compatibility.py"),
    ("tests/qa/test_compare_eval_v1_vs_v2.py", "tests/qa/test_compare_eval_v1_vs_v2.py"),
    ("tests/qa/test_ground_truth_v2_integrity.py", "tests/qa/test_ground_truth_v2_integrity.py"),
    
    # 4. Raw Test Evidence & Environment
    ("reports/test_evidence/tests_summary.json", "test_evidence/tests_summary.json"),
    ("reports/test_evidence/pytest_collection.log", "test_evidence/pytest_collection.log"),
    ("reports/test_evidence/pytest_targeted.log", "test_evidence/pytest_targeted.log"),
    ("reports/test_evidence/python_environment.json", "test_evidence/python_environment.json"),
    
    # 5. Dataset v3 Candidate Manifests & Splits
    ("data/detection_v3_candidate/manifests/candidate_image_manifest.csv", "data_v3/candidate_image_manifest.csv"),
    ("data/detection_v3_candidate/manifests/candidate_annotation_manifest.csv", "data_v3/candidate_annotation_manifest.csv"),
    ("data/detection_v3_candidate/manifests/dataset_provenance_report.json", "data_v3/dataset_provenance_report.json"),
    ("data/detection_v3_candidate/manifests/dataset_integrity_report.json", "data_v3/dataset_integrity_report.json"),
    ("data/detection_v3_candidate/splits/train.txt", "data_v3/splits/train.txt"),
    ("data/detection_v3_candidate/splits/val.txt", "data_v3/splits/val.txt"),
    ("data/detection_v3_candidate/splits/test.txt", "data_v3/splits/test.txt"),
    
    # 6. Quality Gate Audit Report (Real Dataset 2,143 images)
    ("reports/data_v2/real_dataset_quality_gate_audit.json", "data_v3/real_dataset_quality_gate_audit.json"),
    
    # 7. Configs & YOLO Compatibility
    ("configs/detection_dataset_real_v3_candidate.yaml", "configs/detection_dataset_real_v3_candidate.yaml"),
    ("data/detection_v3_candidate/yolo/data.yaml", "yolo/data.yaml"),
    
    # 8. Training Dry Run Report
    ("reports/model_v2/training_dry_run_report.json", "reports/training_dry_run_report.json"),
]

print(f"Building Phase 4B Delivery Package: {ZIP_OUTPUT}")
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

sha256 = hashlib.sha256(ZIP_OUTPUT.read_bytes()).hexdigest()
size_mb = ZIP_OUTPUT.stat().st_size / (1024 * 1024)

print("\n" + "="*70)
print(f"DELIVERY PACKAGE CREATED SUCCESSFULLY!")
print(f"Location: {ZIP_OUTPUT}")
print(f"Files Packed: {packed_count}")
print(f"Size: {size_mb:.2f} MB ({ZIP_OUTPUT.stat().st_size:,} bytes)")
print(f"SHA-256: {sha256}")
print("="*70)
