# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Xuất báo cáo, kết quả dự đoán thô và bằng chứng kiểm thử cho `export_all_key_reports_to_downloads`.
"""
Export all important project reports, manifests, audit evidence, and specifications
into ~/Downloads/CNTT_KLCN155_Bao_Cao_Va_Bang_Chung_AI_Upgrade.zip
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    repo_root = Path(__file__).resolve().parents[3]
    downloads_root = (Path.home() / "Downloads")
    if not downloads_root.exists():
        print(f"ERROR: {downloads_root} does not exist!")
        sys.exit(1)

    export_base_name = "CNTT_KLCN155_Bao_Cao_Va_Bang_Chung_AI_Upgrade"
    export_dir = downloads_root / export_base_name
    zip_path = downloads_root / f"{export_base_name}.zip"

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Organized categories
    categories = {
        "01_Sprint_AI_Upgrade_01": [
            "reports/archive/task_reports/FAILURE_ANALYSIS.md",
            "reports/archive/task_reports/DATASET_QA_REPORT.md",
            "reports/archive/task_reports/AI_UPGRADE_BASELINE.md",
            "reports/source_verification/CANDIDATE_SOURCE_VERIFICATION_REPORT.md",
            "reports/source_verification/wikimedia_raw_api_responses.json",
            "reports/source_verification/task3_source_verification_delivery.zip",
            "data/acquisition_staging/manifests/batch_a_candidate_manifest.json",
        ],
        "02_Bao_Cao_Tong_Hop_Va_Audit_Chuyen_Sau": [
            "docs/BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md",
            "docs/error-analysis.md",
            "docs/model-audit.md",
            "docs/model-comparison.md",
            "docs/model-experiments.md",
            "docs/dataset-improvement.md",
            "docs/spec-dataset-standardization.md",
            "docs/tickets-dataset-improvement.md",
            "docs/GIAI_THICH_THUAT_NGU_VA_KIEN_TRUC.md",
            "reports/archive/task_reports/BAO_CAO_PHAN_1_MAPPING_10_LOP.md",
        ],
        "03_Manifest_Va_Kiem_Toan_Du_Lieu": [
            "reports/qa_acceptance/manifest_hinh_anh_rac.json",
            "reports/qa_acceptance/manifest_20_anh_kho.json",
            "reports/qa_acceptance/manifest_100_anh_online.json",
            "reports/qa_acceptance/suspicious_gt_audit_table.json",
            "reports/data_v2/split_contamination_report.md",
            "reports/data_v2/split_contamination_report.json",
            "reports/data_v2/phash_leakage_audit_report.json",
            "data/audit/leakage_audit_report.json",
            "data/audit/authoritative_hashes_cache.json",
            "data/acquisition_staging/manifests/mini_pilot_qa_manifest.json",
            "data/acquisition_staging/manifests/train_expansion_candidate_manifest.json",
            "data/acquisition_staging/manifests/phase4e_real_waste_catalog.json",
        ],
        "04_Dac_Ta_Va_Quy_Chuan_Du_An": [
            "AGENTS.md",
            "docs/plan/DATA_COLLECTION_PLAN.md",
            "docs/plan/REAL_WORLD_DEPLOYMENT_SPEC.md",
            "docs/plan/ANNOTATION_POLICY.md",
            "GLOSSARY.md",
            "docs/plan/SPRINT_TASKS.md",
            "docs/plan/TRAINING_AND_EVALUATION_PLAN.md",
            "configs/detection_dataset_real_v2.yaml",
            "configs/detection_source_mapping.yaml",
            "configs/detection_dataset_real_v3_candidate.yaml",
            "configs/detection_thresholds_calibrated.json",
        ],
        "05_Kiem_Thu_Va_Kich_Ban_Tu_Dong": [
            "tests/test_candidate_manifest_qa_lifecycle.py",
            "tests/detection/test_real_mapping_integration.py",
            "tests/detection/test_split_real_v2.py",
            "scripts/archive/phase4_delivery/verify_wikimedia_candidates.py",
        ]
    }

    manifest_records = []
    total_copied = 0

    for cat_name, file_rel_list in categories.items():
        cat_dir = export_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in file_rel_list:
            src_file = repo_root / rel_path
            if src_file.exists():
                dst_file = cat_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                total_copied += 1
                f_size = dst_file.stat().st_size
                f_sha = compute_sha256(dst_file)
                manifest_records.append({
                    "category": cat_name,
                    "filename": src_file.name,
                    "relative_source": rel_path,
                    "size_bytes": f_size,
                    "sha256": f_sha
                })
                print(f"  [OK] {cat_name}/{src_file.name} ({f_size} bytes)")
            else:
                print(f"  [SKIP] Not found: {rel_path}")

    # Generate README in export directory
    readme_content = f"""# GÓI TỔNG HỢP BÁO CÁO & BẰNG CHỨNG DỰ ÁN (CNTT-KLCN155)
**Thời điểm xuất:** {Path(zip_path).name}
**Tổng số tài liệu và bằng chứng:** {total_copied} tệp
**Mục đích:** Bàn giao đầy đủ báo cáo phân tích lỗi, hồ sơ baseline, kiểm kê dataset, đặc tả nguồn dữ liệu và bằng chứng audit cho PM và Người dùng.

## CẤU TRÚC THƯ MỤC
1. **`01_Sprint_AI_Upgrade_01/`**: Bộ báo cáo P0 Sprint AI-UPGRADE-01:
   - `FAILURE_ANALYSIS.md`: Phân tích chuyên sâu 7 ca bệnh demo và 6 khía cạnh lỗi (A-F).
   - `DATASET_QA_REPORT.md`: Kiểm kê bit-for-bit tập dữ liệu v2, v3, holdout 27 ảnh, 100 ảnh online, 20 ảnh kho.
   - `AI_UPGRADE_BASELINE.md`: Đóng băng thông số checkpoint YOLOv8n & SSDLite và preflight GPU RTX 2050.
   - `CANDIDATE_SOURCE_VERIFICATION_REPORT.md`: Báo cáo xác minh Wikimedia API và danh mục đề xuất 10 ảnh Mini-Batch 01.
   - `batch_a_candidate_manifest.json` & `wikimedia_raw_api_responses.json`: Manifest 25 ứng viên và payload API thô.
   - `task3_source_verification_delivery.zip`: Gói bàn giao Task 3.

2. **`02_Bao_Cao_Tong_Hop_Va_Audit_Chuyen_Sau/`**: Các báo cáo tổng kết toàn diện, audit kiến trúc mô hình và cứu hộ hệ thống:
   - `BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md`
   - `error-analysis.md`, `model-audit.md`, `model-comparison.md`, `model-experiments.md`
   - `dataset-improvement.md`, `spec-dataset-standardization.md`

3. **`03_Manifest_Va_Kiem_Toan_Du_Lieu/`**: Hồ sơ băm, manifest nghiệm thu và báo cáo chống rò rỉ:
   - `manifest_hinh_anh_rac.json`: 27 ảnh holdout đóng băng gốc (57 GT).
   - `manifest_20_anh_kho.json` & `manifest_100_anh_online.json`: 120 ảnh kiểm định ngoài.
   - `split_contamination_report.md` & `leakage_audit_report.json`: Bằng chứng phòng chống rò rỉ dữ liệu.

4. **`04_Dac_Ta_Va_Quy_Chuan_Du_An/`**: Các tài liệu hiến pháp, quy chế làm việc multi-agent và cấu hình:
   - `AGENTS.md`, `DATA_COLLECTION_PLAN.md`, `REAL_WORLD_DEPLOYMENT_SPEC.md`
   - `ANNOTATION_POLICY.md`, `TRAINING_AND_EVALUATION_PLAN.md`, các file yaml dataset.

5. **`05_Kiem_Thu_Va_Kich_Ban_Tu_Dong/`**:
   - `test_candidate_manifest_qa_lifecycle.py`: Bộ test hồi quy tự động cho QA Lifecycle.
   - `test_real_mapping_integration.py`, `test_split_real_v2.py`.
   - `verify_wikimedia_candidates.py`: Kịch bản truy vấn MediaWiki API.
"""
    readme_path = export_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")

    # Generate Manifest JSON
    manifest_path = export_dir / "EXPORT_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "export_name": export_base_name,
            "total_files": total_copied,
            "files": manifest_records
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Creating ZIP file: {zip_path} ...")
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in export_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(export_dir)
                zf.write(f, arcname)

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Export completed successfully!")
    print(f"  -> Folder: {export_dir}")
    print(f"  -> ZIP:    {zip_path} ({zip_size_mb:.2f} MiB)")

if __name__ == "__main__":
    main()
