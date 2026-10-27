# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Đóng gói hồ sơ minh chứng, manifest và tệp nén bàn giao cho `build_compact_final_verification_delivery`.
"""
scripts/build_compact_final_verification_delivery.py
----------------------------------------------------
Packages a compact, self-contained verification evidence ZIP file containing:
- Final verification report
- GT v2 review document
- Unified diff and cryptographic patch proof for oi_2734e597675d8a97
- Raw pytest logs (collection, targeted, environment, summary)
- Multi-policy evaluation summary JSON
- Dataset quality, leakage, and integrity JSON reports
- Core regression test scripts
- Representative GT approval overlay previews for the 6 disputed objects
"""

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")
ZIP_NAME = "ANTI_PHASE3_FINAL_VERIFICATION_EVIDENCE.zip"
ZIP_WORKSPACE = ROOT / ZIP_NAME


def package_evidence():
    files_to_pack = [
        # 1. Final reports
        (ROOT / "reports/ANTI_PHASE3_FINAL_VERIFICATION.md", "00_ANTI_PHASE3_FINAL_VERIFICATION.md"),
        (ROOT / "reports/GT_V2_REVIEW_FOR_APPROVAL.md", "01_GT_V2_REVIEW_FOR_APPROVAL.md"),

        # 2. Patch & Diff Evidence
        (ROOT / "reports/patches/patch_oi_2734e597675d8a97.diff", "diffs/patch_oi_2734e597675d8a97.diff"),
        (ROOT / "reports/patches/patch_oi_2734e597675d8a97_evidence.json", "diffs/patch_oi_2734e597675d8a97_evidence.json"),

        # 3. Raw Pytest Logs & Environment
        (ROOT / "reports/test_evidence/pytest_collection.log", "test_evidence/pytest_collection.log"),
        (ROOT / "reports/test_evidence/pytest_targeted.log", "test_evidence/pytest_targeted.log"),
        (ROOT / "reports/test_evidence/python_environment.json", "test_evidence/python_environment.json"),
        (ROOT / "reports/test_evidence/tests_summary.json", "test_evidence/tests_summary.json"),

        # 4. Evaluation Artifacts
        (ROOT / "reports/qa_acceptance/verified_evaluation_summary_from_raw.json", "evaluation/verified_evaluation_summary_from_raw.json"),
        (ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json", "evaluation/ground_truth_hinh_anh_rac_v2_proposed.json"),
        (ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json", "evaluation/ground_truth_hinh_anh_rac_v1_frozen.json"),

        # 5. Dataset Quality & Leakage Audit Reports
        (ROOT / "reports/DATASET_QUALITY_AND_LEAKAGE_REPORT.json", "data_audit/DATASET_QUALITY_AND_LEAKAGE_REPORT.json"),
        (ROOT / "reports/DATASET_INTEGRITY_AUDIT_REPORT.json", "data_audit/DATASET_INTEGRITY_AUDIT_REPORT.json"),

        # 6. Core Regression Test Code
        (ROOT / "tests/qa/test_compare_eval_v1_vs_v2.py", "tests/test_compare_eval_v1_vs_v2.py"),
        (ROOT / "tests/detection/test_taco_clipping_geometry.py", "tests/test_taco_clipping_geometry.py"),
        (ROOT / "tests/qa/test_ground_truth_v2_integrity.py", "tests/test_ground_truth_v2_integrity.py"),
        (ROOT / "tests/test_environment_integrity.py", "tests/test_environment_integrity.py"),
        (ROOT / "scripts/compare_eval_v1_vs_v2.py", "scripts/compare_eval_v1_vs_v2.py"),

        # 7. Targeted GT Approval Overlays for the 6 Disputed Objects
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_images (12).jpg.jpg", "gt_previews/review_images_12_nested_carton_paper.jpg"),
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_images (10).jpg.jpg", "gt_previews/review_images_10_nested_clothes.jpg"),
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_OIP (1).webp.jpg", "gt_previews/review_OIP_1_overlapping_clothes.jpg"),
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_Phan-Loai-Rac-Tai-Ng-01.jpg.jpg", "gt_previews/review_Phan_Loai_Rac_tetrapak_and_bags.jpg"),
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_images (18).jpg.jpg", "gt_previews/review_images_18_wheeled_dumpster_negative.jpg"),
        (ROOT / "reports/qa_acceptance/gt_review_overlays/review_images (8).jpg.jpg", "gt_previews/review_images_8_office_basket_negative.jpg"),
    ]

    print(f"Building compact verification evidence archive at: {ZIP_WORKSPACE}")
    with zipfile.ZipFile(ZIP_WORKSPACE, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_path, arcname in files_to_pack:
            if src_path.exists():
                zf.write(src_path, arcname)
                print(f"  + Added: {arcname} ({src_path.stat().st_size:,} bytes)")
            else:
                print(f"  ! Missing file: {src_path}")

    print(f"\nArchive created: {ZIP_WORKSPACE} ({ZIP_WORKSPACE.stat().st_size:,} bytes)")

    if DOWNLOADS_DIR.exists():
        dest = DOWNLOADS_DIR / ZIP_NAME
        shutil.copy2(ZIP_WORKSPACE, dest)
        print(f"SUCCESS: Copied deliverable package to: {dest} ({dest.stat().st_size:,} bytes)")
    else:
        print(f"WARNING: Downloads directory not found at {DOWNLOADS_DIR}")


if __name__ == "__main__":
    package_evidence()
