"""
scripts/package_part02_verification_delivery.py
------------------------------------------------
Packages code, smoke checkpoints, raw logs, verification telemetry,
sample images, audit data, and formal documentation into a single delivery ZIP:
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
    dir_smoke = export_dir / "04_Smoke_Checkpoints_Va_Logs"
    dir_audit = export_dir / "05_Kiem_Toan_Du_Lieu_Va_Split"
    dir_src = export_dir / "06_Ma_Nguon_Detection_Va_Test"

    for d in [dir_docs, dir_aug, dir_backbone, dir_smoke, dir_audit, dir_src]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Docs
    shutil.copy2(project_root / "docs" / "plan" / "PART_01_ACCEPTANCE.md", dir_docs)
    shutil.copy2(project_root / "docs" / "plan" / "PART_02_10CLASS_VERIFICATION.md", dir_docs)
    shutil.copy2(project_root / "docs" / "plan" / "PART_03_EXPERIMENT_MATRIX.md", dir_docs)
    shutil.copy2(project_root / "docs" / "proposal" / "TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md", dir_docs)

    # 2. Augmentation & Loaders
    shutil.copy2(project_root / "artifacts" / "part02" / "augmentation_verification_report.json", dir_aug)
    shutil.copy2(project_root / "scripts" / "verify_augmentation_and_loaders.py", dir_aug)
    aug_samples_dir = dir_aug / "samples"
    aug_samples_dir.mkdir(parents=True, exist_ok=True)
    for p in (project_root / "artifacts" / "part02" / "augmentation_samples").glob("*.png"):
        shutil.copy2(p, aug_samples_dir / p.name)

    # 3. Backbone transfer
    shutil.copy2(project_root / "data" / "audit" / "backbone_transfer_audit.json", dir_backbone)
    shutil.copy2(project_root / "artifacts" / "part02" / "backbone_transfer_verification.json", dir_backbone)
    shutil.copy2(project_root / "scripts" / "audit_backbone_transfer.py", dir_backbone)

    # 4. Smoke Checkpoints & Raw Logs
    ssdlite_target = dir_smoke / "smoke-ssdlite320"
    ssdlite_target.mkdir(parents=True, exist_ok=True)
    ssdlite_src = project_root / "artifacts" / "detection_smoke" / "smoke-ssdlite320"
    if (ssdlite_src / "weights" / "best.pt").exists():
        (ssdlite_target / "weights").mkdir(exist_ok=True)
        shutil.copy2(ssdlite_src / "weights" / "best.pt", ssdlite_target / "weights" / "best.pt")
    for f in ["training_summary.json", "history.json", "run_metadata.json", "train-val-resolved-dataset.yaml"]:
        if (ssdlite_src / f).exists():
            shutil.copy2(ssdlite_src / f, ssdlite_target / f)

    yolo_target = dir_smoke / "smoke-yolov8n"
    yolo_target.mkdir(parents=True, exist_ok=True)
    yolo_src = project_root / "artifacts" / "detection_smoke" / "smoke-yolov8n"
    if (yolo_src / "weights" / "best.pt").exists():
        (yolo_target / "weights").mkdir(exist_ok=True)
        shutil.copy2(yolo_src / "weights" / "best.pt", yolo_target / "weights" / "best.pt")
    for f in ["training_summary.json", "results.csv", "effective_train_args.json", "run_metadata.json", 
              "BoxPR_curve.png", "confusion_matrix.png", "results.png"]:
        if (yolo_src / f).exists():
            shutil.copy2(yolo_src / f, yolo_target / f)

    # Visual smoke inference & downloaded sample
    for f in ["smoke_ssdlite_real_val_inference.png", "smoke_yolov8n_real_val_inference.png", "downloaded_annotated_sample.png"]:
        src_f = project_root / "artifacts" / "part02" / f
        if src_f.exists():
            shutil.copy2(src_f, dir_smoke / f)

    # 5. Data Audit & Splits
    shutil.copy2(project_root / "artifacts" / "part02" / "real_data_audit" / "real_images_audit_table.csv", dir_audit)
    shutil.copy2(project_root / "artifacts" / "part02" / "real_data_audit" / "real_data_audit_summary.json", dir_audit)
    shutil.copy2(project_root / "data" / "detection" / "manifest_detection_v1.csv", dir_audit)
    shutil.copy2(project_root / "artifacts" / "part02" / "detection_split_audit.json", dir_audit)
    shutil.copy2(project_root / "artifacts" / "part02" / "real_detection_readiness.json", dir_audit)
    shutil.copy2(project_root / "artifacts" / "part02" / "taco_approved_contact_sheet.jpg", dir_audit)

    # 6. Source code & Tests
    shutil.copy2(project_root / "src" / "detection" / "augmentation.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "copy_paste.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "ssdlite_train.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "train.py", dir_src)
    shutil.copy2(project_root / "src" / "detection" / "dataset.py", dir_src)
    shutil.copy2(project_root / "tests" / "detection" / "test_augmentation.py", dir_src)
    shutil.copy2(project_root / "configs" / "detection_dataset.yaml", dir_src)
    shutil.copy2(project_root / "configs" / "detection_source_mapping.yaml", dir_src)
    shutil.copy2(project_root / "pytest.ini", dir_src)

    # Readme
    readme_text = f"""# GÓI BÀN GIAO BẰNG CHỨNG KIỂM CHỨNG KỸ THUẬT & MÃ NGUỒN
## ĐỀ TÀI CNTT-KLCN155 — PHÁT HIỆN VÀ PHÂN LOẠI ĐA ĐỐI TƯỢNG RÁC THẢI SINH HOẠT
**Người nhận:** PM Ngô Thanh Nhân (HUIT)  
**Thời gian lập gói:** 02/10/2026  
**Mục tiêu bàn giao:** Cung cấp đầy đủ mã nguồn đã chỉnh sửa, bằng chứng kiểm chứng augmentation/loaders, checkpoints smoke, log huấn luyện thô, và hồ sơ kiểm toán dữ liệu.

### DANH MỤC THƯ MỤC:
1. `01_Bao_Cao_Va_Ke_Hoach/`:
   - `PART_01_ACCEPTANCE.md`: Báo cáo nghiệm thu phân loại đơn rác MobileNetV3 (Macro-F1 95,59%, Acc 96,18%).
   - `PART_02_10CLASS_VERIFICATION.md`: Báo cáo đối soát kỹ thuật toàn diện, giải đáp 8 câu hỏi, bảng kiểm chứng PASS/FAIL/BLOCKED.
   - `PART_03_EXPERIMENT_MATRIX.md`: Ma trận 8 thí nghiệm (2 kiến trúc x 4 chiến lược tăng cường), điều kiện chặn và giao thức thực nghiệm.
   - `TO_TRINH_XIN_DIEU_CHINH_DE_CUONG_10_LOP.md`: Tờ trình chính thức gửi GVHD ThS. Huỳnh Thị Châu Lan.

2. `02_Bang_Chung_Augmentation_Va_Loaders/`:
   - `augmentation_verification_report.json`: Nhật ký kiểm thử tự động, telemetry tham số, tỷ lệ co giãn, số box trước/sau.
   - `verify_augmentation_and_loaders.py`: Kịch bản kiểm thử độc lập luồng tăng cường và loader của cả 2 detector.
   - `samples/`: 28 ảnh mẫu PNG đối chứng (Normal Mode vs Forced Verification Mode) trên các ảnh TACO thật (`taco_0081`, `taco_0082`, `taco_0853`).

3. `03_Bang_Chung_Backbone_Transfer`:
   - `backbone_transfer_audit.json`: Kết quả đối soát 296 keys khớp, 253 weight/bias tensors (max_abs_diff = 0.0), 43 buffers.
   - `audit_backbone_transfer.py`: Mã nguồn kiểm toán tensor PyTorch trực tiếp.

4. `04_Smoke_Checkpoints_Va_Logs/`:
   - `smoke-ssdlite320/weights/best.pt`: Checkpoint PyTorch SSDLite320 sau 1 epoch smoke (28,4 MB).
   - `smoke-yolov8n/weights/best.pt`: Checkpoint Ultralytics YOLOv8n sau 1 epoch smoke (6,2 MB).
   - Log thô, metadata, và đồ thị huấn luyện/xác minh.
   - `downloaded_annotated_sample.png`: Bằng chứng tải tệp ảnh trực tiếp từ nút xuất kết quả giao diện Web/Review Tool.

5. `05_Kiem_Toan_Du_Lieu_Va_Split`:
   - `real_images_audit_table.csv`: Kết quả rà soát chi tiết từng ảnh OpenImages/TACO.
   - `real_data_audit_summary.json` & `detection_split_audit.json`: Tổng hợp hiện trạng phân bổ nhãn và cảnh.
   - `taco_approved_contact_sheet.jpg`: Bảng ảnh trực quan các mẫu TACO đã duyệt.

6. `06_Ma_Nguon_Detection_Va_Test`:
   - Toàn bộ mã nguồn cốt lõi đã cập nhật (`augmentation.py`, `copy_paste.py`, `ssdlite_train.py`, `train.py`, v.v.).
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
        manifest_rows.append({
            "relative_path": str(rel).replace("\\", "/"),
            "size_bytes": size,
            "sha256": sha
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(export_dir / "MANIFEST_SHA256.csv", index=False, encoding="utf-8")

    # Zip
    print(f"[INFO] Creating delivery ZIP ({len(manifest_rows)} files)...")
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in export_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(export_dir)
                zipf.write(f, arcname)

    zip_sha = compute_sha256(zip_output_path)
    zip_size_mb = zip_output_path.stat().st_size / (1024 * 1024)

    print(f"[SUCCESS] Packaging complete!")
    print(f"Export Folder: {export_dir}")
    print(f"ZIP Archive:   {zip_output_path}")
    print(f"ZIP Size:      {zip_size_mb:.2f} MB")
    print(f"ZIP SHA-256:   {zip_sha}")

    # Save summary json
    summary = {
        "zip_path": str(zip_output_path),
        "zip_size_mb": round(zip_size_mb, 2),
        "zip_sha256": zip_sha,
        "total_files": len(manifest_rows),
        "checkpoints": {
            "ssdlite320_best_pt_sha256": compute_sha256(ssdlite_src / "weights" / "best.pt") if (ssdlite_src / "weights" / "best.pt").exists() else None,
            "yolov8n_best_pt_sha256": compute_sha256(yolo_src / "weights" / "best.pt") if (yolo_src / "weights" / "best.pt").exists() else None
        }
    }
    with open(project_root / "artifacts" / "part02" / "delivery_package_info.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
