"""
scripts/build_phase4b2_delivery.py
-----------------------------------
Packages all Phase 4B.2 verified artifacts, source code, regression tests,
logs, manifests, and authoritative metrics into:
~/Downloads/PHASE4B2_FINAL_ENGINEERING_EVIDENCE.zip
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
ZIP_OUTPUT = DOWNLOADS_DIR / "PHASE4B2_FINAL_ENGINEERING_EVIDENCE.zip"

FILES_TO_PACK = [
    # 1. Master Reports
    ("reports/PHASE4B2_FINAL_ENGINEERING.md", "PHASE4B2_FINAL_ENGINEERING.md"),
    ("reports/data_v2/authoritative_dataset_metrics.md", "reports/authoritative_dataset_metrics.md"),
    ("reports/data_v2/candidate_expansion_sources.md", "reports/candidate_expansion_sources.md"),

    # 2. Source Code & Scripts
    ("src/detection/preflight_guard.py", "src/detection/preflight_guard.py"),
    ("scripts/build_dataset_v3_candidate.py", "scripts/build_dataset_v3_candidate.py"),
    ("scripts/generate_authoritative_dataset_metrics.py", "scripts/generate_authoritative_dataset_metrics.py"),
    ("scripts/verify_phash_cross_split_and_holdout.py", "scripts/verify_phash_cross_split_and_holdout.py"),
    ("scripts/verify_training_dry_run_environment.py", "scripts/verify_training_dry_run_environment.py"),
    ("scripts/verify_yolo_dataloader_batches.py", "scripts/verify_yolo_dataloader_batches.py"),
    ("scripts/build_expansion_source_catalog.py", "scripts/build_expansion_source_catalog.py"),

    # 3. Targeted QA Regression Tests
    ("tests/qa/test_dataset_v3_isolation.py", "tests/qa/test_dataset_v3_isolation.py"),
    ("tests/qa/test_authoritative_dataset_metrics.py", "tests/qa/test_authoritative_dataset_metrics.py"),
    ("tests/qa/test_phash_leakage_audit.py", "tests/qa/test_phash_leakage_audit.py"),
    ("tests/qa/test_training_dry_run_api.py", "tests/qa/test_training_dry_run_api.py"),
    ("tests/qa/test_yolo_dataloader_batches.py", "tests/qa/test_yolo_dataloader_batches.py"),

    # 4. JSON Reports & Metrics
    ("reports/data_v2/authoritative_dataset_metrics.json", "reports/authoritative_dataset_metrics.json"),
    ("reports/data_v2/phash_leakage_audit_report.json", "reports/phash_leakage_audit_report.json"),
    ("reports/model_v2/training_dry_run_report.json", "reports/training_dry_run_report.json"),
    ("reports/model_v2/yolo_dataloader_verification_report.json", "reports/yolo_dataloader_verification_report.json"),
    ("reports/data_v2/candidate_expansion_sources.json", "reports/candidate_expansion_sources.json"),

    # 5. Raw Test Evidence & Environment
    ("reports/test_evidence/tests_summary.json", "test_evidence/tests_summary.json"),
    ("reports/test_evidence/pytest_collection.log", "test_evidence/pytest_collection.log"),
    ("reports/test_evidence/pytest_targeted.log", "test_evidence/pytest_targeted.log"),
    ("reports/test_evidence/python_environment.json", "test_evidence/python_environment.json"),

    # 6. Candidate Manifests, Splits & Ledger
    ("data/detection_v3_candidate/manifests/candidate_image_manifest.csv", "data_v3/candidate_image_manifest.csv"),
    ("data/detection_v3_candidate/manifests/candidate_annotation_manifest.csv", "data_v3/candidate_annotation_manifest.csv"),
    ("data/detection_v3_candidate/manifests/label_sync_ledger.json", "data_v3/label_sync_ledger.json"),
    ("data/detection_v3_candidate/splits/train.txt", "data_v3/splits/train.txt"),
    ("data/detection_v3_candidate/splits/val.txt", "data_v3/splits/val.txt"),
    ("data/detection_v3_candidate/splits/test.txt", "data_v3/splits/test.txt"),
    ("data/detection_v3_candidate/yolo/data.yaml", "data_v3/yolo/data.yaml"),
]

def main():
    print(f"Building Phase 4B.2 Delivery Package: {ZIP_OUTPUT}")
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
    print("PHASE 4B.2 DELIVERY PACKAGE CREATED SUCCESSFULLY!")
    print(f"Location: {ZIP_OUTPUT}")
    print(f"Files Packed: {packed_count}")
    print(f"Size: {size_mb:.2f} MB ({ZIP_OUTPUT.stat().st_size:,} bytes)")
    print(f"SHA-256: {sha256}")
    print("="*70)

if __name__ == "__main__":
    main()
