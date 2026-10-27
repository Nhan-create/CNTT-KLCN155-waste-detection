import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOWNLOADS_DIR = (Path.home() / "Downloads")

def build_zip():
    zip_name = "Bao_Cao_Audit_Va_Nang_Cap_Mo_Hinh_CNTT-KLCN155.zip"
    zip_path_workspace = PROJECT_ROOT / zip_name
    
    files_to_add = [
        # Antigravity Phase 3 Core Deliverables
        (PROJECT_ROOT / "reports/ANTI_PHASE3_CONTINUATION.md", "00_ANTI_PHASE3_CONTINUATION.md"),
        (PROJECT_ROOT / "reports/GT_V2_REVIEW_FOR_APPROVAL.md", "01_GT_V2_REVIEW_FOR_APPROVAL.md"),
        (PROJECT_ROOT / "reports/DATASET_AUDIT_AND_TRAINING_READINESS.md", "02_DATASET_AUDIT_AND_TRAINING_READINESS.md"),
        (PROJECT_ROOT / "GLOSSARY.md", "03_GLOSSARY_DOMAIN_MODELING.md"),
        (PROJECT_ROOT / "docs/spec-dataset-standardization.md", "04_spec-dataset-standardization.md"),
        (PROJECT_ROOT / "docs/tickets-dataset-improvement.md", "05_tickets-dataset-improvement.md"),
        
        # Historical & Previous Audit Reports
        (PROJECT_ROOT / "reports/RE_AUDIT_VERIFICATION.md", "reports_history/RE_AUDIT_VERIFICATION.md"),
        (PROJECT_ROOT / "reports/BAO_CAO_KIEM_TOAN_CHUYEN_SAU_VA_DANH_GIA_DO_D.md", "reports_history/BAO_CAO_KIEM_TOAN_CHUYEN_SAU_VA_DANH_GIA_DO_D.md"),
        (PROJECT_ROOT / "docs/BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md", "docs_baseline/01_BAO_CAO_TONG_HOP_GUI_PM_GOC.md"),
        (PROJECT_ROOT / "docs/model-audit.md", "docs_baseline/02_model-audit.md"),
        (PROJECT_ROOT / "docs/error-analysis.md", "docs_baseline/03_error-analysis.md"),
        (PROJECT_ROOT / "docs/dataset-improvement.md", "docs_baseline/04_dataset-improvement.md"),
        (PROJECT_ROOT / "docs/model-experiments.md", "docs_baseline/05_model-experiments.md"),
        (PROJECT_ROOT / "docs/model-comparison.md", "docs_baseline/06_model-comparison.md"),
        (PROJECT_ROOT / "reports/qa_acceptance/QA_ACCEPTANCE_REPORT.md", "docs_baseline/07_QA_ACCEPTANCE_REPORT.md"),
        (PROJECT_ROOT / "reports/qa_acceptance/PRACTICAL_METRICS_REPORT.md", "docs_baseline/08_PRACTICAL_METRICS_REPORT.md"),
        
        # Ground Truth, Audit Reports & Manifests
        (PROJECT_ROOT / "reports/DATASET_INTEGRITY_AUDIT_REPORT.json", "data_manifest/DATASET_INTEGRITY_AUDIT_REPORT.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json", "data_manifest/ground_truth_hinh_anh_rac_v1_frozen.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json", "data_manifest/ground_truth_hinh_anh_rac_v2_proposed.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/suspicious_gt_audit_table.json", "data_manifest/suspicious_gt_audit_table.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/manifest_hinh_anh_rac.json", "data_manifest/manifest_hinh_anh_rac.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/manifest_20_anh_kho.json", "data_manifest/manifest_20_anh_kho.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/manifest_100_anh_online.json", "data_manifest/manifest_100_anh_online.json"),
        
        # Raw Predictions JSON (for independent verification)
        (PROJECT_ROOT / "reports/qa_acceptance/raw_predictions_before_quick_fix.json", "raw_predictions/raw_predictions_before_quick_fix.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/raw_predictions_after_quick_fix.json", "raw_predictions/raw_predictions_after_quick_fix.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/raw_predictions_yolov8n_e1.json", "raw_predictions/raw_predictions_yolov8n_e1.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/verified_evaluation_summary_from_raw.json", "raw_predictions/verified_evaluation_summary_from_raw.json"),
        
        # Historical & Reconstructed Metrics JSON
        (PROJECT_ROOT / "reports/qa_acceptance/practical_metrics_baseline.json", "metrics/practical_metrics_baseline.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/practical_metrics_E1_yolov8n_10cls.json", "metrics/practical_metrics_E1_yolov8n_10cls.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/baseline_results_ssdlite.json", "metrics/baseline_results_ssdlite.json"),
        (PROJECT_ROOT / "reports/qa_acceptance/baseline_results_yolov8n.json", "metrics/baseline_results_yolov8n.json"),
        (PROJECT_ROOT / "artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json", "metrics/eval_val_test_E1_yolov8n_10cls.json"),
        
        # Reproducibility Scripts & Tests
        (PROJECT_ROOT / "scripts/recompile_clean_bytecode.py", "scripts/recompile_clean_bytecode.py"),
        (PROJECT_ROOT / "scripts/audit_taco_coco_exact.py", "scripts/audit_taco_coco_exact.py"),
        (PROJECT_ROOT / "scripts/audit_exact_openimages_provenance.py", "scripts/audit_exact_openimages_provenance.py"),
        (PROJECT_ROOT / "scripts/audit_dataset_integrity_full.py", "scripts/audit_dataset_integrity_full.py"),
        (PROJECT_ROOT / "scripts/audit_gt_exact_counts.py", "scripts/audit_gt_exact_counts.py"),
        (PROJECT_ROOT / "scripts/audit_splits_real_v2_distribution.py", "scripts/audit_splits_real_v2_distribution.py"),
        (PROJECT_ROOT / "scripts/generate_gt_review_overlays.py", "scripts/generate_gt_review_overlays.py"),
        (PROJECT_ROOT / "scripts/build_ground_truth_v2_proposed.py", "scripts/build_ground_truth_v2_proposed.py"),
        (PROJECT_ROOT / "scripts/compare_eval_v1_vs_v2.py", "scripts/compare_eval_v1_vs_v2.py"),
        (PROJECT_ROOT / "scripts/re_evaluate_from_raw_json.py", "scripts/re_evaluate_from_raw_json.py"),
        (PROJECT_ROOT / "tests/test_environment_integrity.py", "tests/test_environment_integrity.py"),
        (PROJECT_ROOT / "tests/test_detection_annotations.py", "tests/test_detection_annotations.py"),
        (PROJECT_ROOT / "tests/qa/test_ground_truth_v2_integrity.py", "tests/test_ground_truth_v2_integrity.py"),
        (PROJECT_ROOT / "tests/qa/test_compare_eval_v1_vs_v2.py", "tests/test_compare_eval_v1_vs_v2.py"),
    ]
    
    # Add all 27 GT review overlays
    review_overlay_dir = PROJECT_ROOT / "reports/qa_acceptance/gt_review_overlays"
    if review_overlay_dir.exists():
        for img_p in review_overlay_dir.glob("*.jpg"):
            files_to_add.append((img_p, f"gt_review_overlays/{img_p.name}"))
            
    print(f"Creating ZIP archive at {zip_path_workspace}...")
    with zipfile.ZipFile(zip_path_workspace, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_add:
            if file_path.exists():
                zf.write(file_path, arcname)
                print(f"  + Added: {arcname} ({file_path.stat().st_size:,} bytes)")
            else:
                print(f"  ! Missing: {file_path}")
                
    print(f"\nArchive created successfully: {zip_path_workspace} ({zip_path_workspace.stat().st_size:,} bytes)")
    
    # Copy to Downloads
    if DOWNLOADS_DIR.exists():
        dest_downloads = DOWNLOADS_DIR / zip_name
        shutil.copy2(zip_path_workspace, dest_downloads)
        print(f"SUCCESS: Copied package to Downloads directory: {dest_downloads} ({dest_downloads.stat().st_size:,} bytes)")
    else:
        print(f"WARNING: Downloads directory not found at {DOWNLOADS_DIR}")

if __name__ == "__main__":
    build_zip()
