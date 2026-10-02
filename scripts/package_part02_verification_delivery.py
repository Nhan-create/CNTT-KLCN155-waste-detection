"""
scripts/package_part02_verification_delivery.py
------------------------------------------------
Packages code, verification checkpoints, official 1-epoch training runs, raw logs,
pixel audit table & contact sheets, web testing proofs, and formal documentation into:
C:\\Users\\ad\\Downloads\\Bao_Cao_Kiem_Chung_CNTT_KLCN155.zip
"""

import os
import sys
import shutil
import hashlib
import zipfile
from pathlib import Path
import json
import pandas as pd


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    project_root = Path(r"D:\CNTT-KLCN155-waste-detection")
    downloads_root = Path(r"C:\Users\ad\Downloads")
    export_dir = downloads_root / "Bao_Cao_Kiem_Chung_CNTT_KLCN155"
    zip_output_path = downloads_root / "Bao_Cao_Kiem_Chung_CNTT_KLCN155.zip"

    print(f"[INFO] Export Directory: {export_dir}")
    print(f"[INFO] Target ZIP:       {zip_output_path}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    dir_docs = export_dir / "01_Bao_Cao_Va_Ke_Hoach"
    dir_aug = export_dir / "02_Bang_Chung_Augmentation_Va_Loaders"
    dir_backbone = export_dir / "03_Bang_Chung_Backbone_Transfer"
    dir_smoke = export_dir / "04_Checkpoints_Va_Logs_Kiem_Chung"
    dir_audit = export_dir / "05_Kiem_Toan_Pixel_Va_Split"
    dir_src = export_dir / "06_Ma_Nguon_Detection_Va_Test"
    dir_web = export_dir / "07_Kiem_Thu_Web_Va_Tai_Tep"

    for d in [dir_docs, dir_aug, dir_backbone, dir_smoke, dir_audit, dir_src, dir_web]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Docs
    shutil.copy2(project_root / "docs" / "plan" / "PART_01_ACCEPTANCE.md", dir_docs)
    shutil.copy2(project_root / "docs" / "plan" / "PART_02_10CLASS_VERIFICATION.md", dir_docs)
    shutil.copy2(project_root / "docs" / "plan" / "PART_03_EXPERIMENT_MATRIX.md", dir_docs)
    shutil.copy2(project_root / "docs" / "proposal" / "TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md", dir_docs)

    # 2. Augmentation & Loaders
    shutil.copy2(project_root / "artifacts" / "part02" / "augmentation_verification_report.json", dir_aug)
    shutil.copy2(project_root / "scripts" / "verify_augmentation_and_loaders.py", dir_aug)
    shutil.copy2(project_root / "tests" / "detection" / "test_copy_paste_rigorous.py", dir_aug)
    aug_samples_dir = dir_aug / "samples"
    aug_samples_dir.mkdir(parents=True, exist_ok=True)
    for p in (project_root / "artifacts" / "part02" / "augmentation_samples").glob("*.png"):
        shutil.copy2(p, aug_samples_dir / p.name)

    # 3. Backbone transfer
    shutil.copy2(project_root / "data" / "audit" / "backbone_transfer_audit.json", dir_backbone)
    shutil.copy2(project_root / "artifacts" / "part02" / "backbone_transfer_verification.json", dir_backbone)
    shutil.copy2(project_root / "scripts" / "audit_backbone_transfer.py", dir_backbone)

    # 4. Checkpoints & Raw Logs
    # 4.1 Verification run checkpoints (combined_training_verification)
    combined_src = project_root / "artifacts" / "part02" / "combined_training_verification"
    if combined_src.exists():
        comb_ssd_target = dir_smoke / "verification_run" / "combined-ssdlite320" / "weights"
        comb_ssd_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(combined_src / "combined-ssdlite320" / "weights" / "best.pt", comb_ssd_target / "best.pt")

        comb_yolo_target = dir_smoke / "verification_run" / "combined-yolov8n" / "weights"
        comb_yolo_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(combined_src / "combined-yolov8n" / "weights" / "best.pt", comb_yolo_target / "best.pt")

        shutil.copy2(combined_src / "combined_training_verification_report.json", dir_smoke / "verification_run")
        shutil.copy2(combined_src / "smoke_ssdlite_combined_real_val_inference.png", dir_smoke / "verification_run")
        shutil.copy2(combined_src / "smoke_yolov8n_combined_real_val_inference.png", dir_smoke / "verification_run")

    # 4.2 Official pipeline 1-epoch runs
    official_ssd_src = project_root / "artifacts" / "detection" / "combined-ssdlite320-combined-seed42"
    if official_ssd_src.exists():
        official_ssd_dst = dir_smoke / "official_1epoch_run" / "combined-ssdlite320-combined-seed42"
        official_ssd_dst.mkdir(parents=True, exist_ok=True)
        (official_ssd_dst / "weights").mkdir(exist_ok=True)
        if (official_ssd_src / "weights" / "best.pt").exists():
            shutil.copy2(official_ssd_src / "weights" / "best.pt", official_ssd_dst / "weights" / "best.pt")
        for f in ["training_summary.json", "history.json", "run_metadata.json", "augmentation_telemetry.json"]:
            if (official_ssd_src / f).exists():
                shutil.copy2(official_ssd_src / f, official_ssd_dst / f)

    official_yolo_src = project_root / "artifacts" / "detection" / "combined-yolov8n-combined-seed42"
    if official_yolo_src.exists():
        official_yolo_dst = dir_smoke / "official_1epoch_run" / "combined-yolov8n-combined-seed42"
        official_yolo_dst.mkdir(parents=True, exist_ok=True)
        (official_yolo_dst / "weights").mkdir(exist_ok=True)
        if (official_yolo_src / "weights" / "best.pt").exists():
            shutil.copy2(official_yolo_src / "weights" / "best.pt", official_yolo_dst / "weights" / "best.pt")
        for f in ["training_summary.json", "results.csv", "run_metadata.json", "effective_train_args.json"]:
            if (official_yolo_src / f).exists():
                shutil.copy2(official_yolo_src / f, official_yolo_dst / f)

    # 5. Data Audit & Splits & Pixel Audit
    pixel_audit_src = project_root / "artifacts" / "part02" / "pixel_audit"
    if pixel_audit_src.exists():
        shutil.copy2(pixel_audit_src / "pixel_audit_81_candidates.csv", dir_audit)
        shutil.copy2(pixel_audit_src / "pixel_audit_summary.json", dir_audit)
        contact_sheets_dst = dir_audit / "contact_sheets"
        contact_sheets_dst.mkdir(parents=True, exist_ok=True)
        for cs in (pixel_audit_src / "contact_sheets").glob("*.png"):
            shutil.copy2(cs, contact_sheets_dst / cs.name)

    shutil.copy2(project_root / "artifacts" / "part02" / "data_audit_r3" / "real_candidates_systematic_audit_table.csv", dir_audit)
    shutil.copy2(project_root / "artifacts" / "part02" / "data_audit_r3" / "audit_summary_r3.json", dir_audit)
    shutil.copy2(project_root / "data" / "detection" / "manifest_detection_v1.csv", dir_audit)
    shutil.copy2(project_root / "data" / "audit" / "real_detection_source_manifest.csv", dir_audit)

    # 6. Source code & Tests
    shutil.copy2(project_root / "src" / "detection" / "augmentation.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "copy_paste.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "ssdlite.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "ssdlite_train.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "yolo.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "train.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "dataset.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "training_common.py", dir_src)
    shutil.copy2(project_root / "tests" / "detection" / "test_augmentation.py", dir_src)
    shutil.copy2(project_root / "tests" / "detection" / "test_yolo_adapter.py", dir_src)
    shutil.copy2(project_root / "tests" / "detection" / "test_copy_paste_rigorous.py", dir_src)
    shutil.copy2(project_root / "scripts" / "verify_real_training_combined.py", dir_src)
    shutil.copy2(project_root / "scripts" / "pixel_audit_81_candidates.py", dir_src)
    shutil.copy2(project_root / "configs" / "detection_dataset.yaml", dir_src)
    shutil.copy2(project_root / "configs" / "detection_source_mapping.yaml", dir_src)
    shutil.copy2(project_root / "configs" / "detection_training_combined_ssdlite.yaml", dir_src)
    shutil.copy2(project_root / "configs" / "detection_training_combined_yolov8n.yaml", dir_src)
    shutil.copy2(project_root / "pytest.ini", dir_src)

    # 7. Web & MCP verification proofs
    web_src = project_root / "artifacts" / "part02" / "web_verification"
    if web_src.exists():
        for wf in web_src.glob("*.png"):
            shutil.copy2(wf, dir_web / wf.name)

    # Readme
    readme_text = f"""# GÓI BÀN GIAO BẰNG CHỨNG KIỂM CHỨNG KỸ THUẬT & MÃ NGUỒN (PHIÊN BẢN R4)
## ĐỀ TÀI CNTT-KLCN155 — PHÁT HIỆN VÀ PHÂN LOẠI ĐA ĐỐI TƯỢNG RÁC THẢI SINH HOẠT
**Người nhận:** PM Ngô Thanh Nhân (MSSV: 2001230595 - HUIT)  
**Thời gian lập gói:** 02/10/2026  
**Mục tiêu bàn giao:**
1. Thu hồi và đánh dấu INVALID_RETRACTED báo cáo PASS cũ của model 80 lớp COCO; thay thế bằng kiến trúc YOLOv8n 10 lớp thực sự (head.nc=10, cv3 channels=10, names đúng 10 lớp rác).
2. Kiểm chứng 1 epoch huấn luyện thật qua pipeline chính thức `src.detection.train` cho cả SSDLite và YOLOv8n (variant combined).
3. Bằng chứng kiểm thử giao diện Web Streamlit qua Chrome DevTools MCP: chuyển đổi 2 detector, upload ảnh, điều chỉnh threshold, tải tệp và kiểm tra tệp 538 KB trên đĩa cứng.
4. Thẩm định pixel thực tế 81 ảnh ứng viên, xuất bảng CSV, 4 tấm contact sheets, phân định rõ AI_REVIEWED vs HUMAN_CONFIRMED, nêu rõ blocker quản trị dữ liệu (0 ảnh mới vào split, tập test giữ nguyên).
5. Khớp mã băm SHA-256 byte-exact trực tiếp với mọi file trong ZIP.
"""
    with open(export_dir / "00_DANH_MUC_GIAO_NHAN.md", "w", encoding="utf-8") as f:
        f.write(readme_text)

    # Manifest SHA-256
    exported_files = sorted([p for p in export_dir.rglob("*") if p.is_file()])
    manifest_rows = []
    for f in exported_files:
        rel = f.relative_to(export_dir)
        sha = compute_sha256(f)
        size = f.stat().st_size
        manifest_rows.append({"relative_path": str(rel).replace("\\", "/"), "size_bytes": size, "sha256": sha})

    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(export_dir / "MANIFEST_SHA256.csv", index=False, encoding="utf-8")
    print(f"[INFO] Generated MANIFEST_SHA256.csv with {len(df_manifest)} files.")

    # Create target ZIP
    if zip_output_path.exists():
        zip_output_path.unlink()

    print(f"[INFO] Compressing {export_dir} into {zip_output_path}...")
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in export_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(export_dir)
                zf.write(f, arcname)

    zip_size_mb = zip_output_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Packaged delivery ZIP: {zip_output_path} ({zip_size_mb:.2f} MB)")
    print(f"[INFO] ZIP SHA-256: {compute_sha256(zip_output_path)}")


if __name__ == "__main__":
    main()
