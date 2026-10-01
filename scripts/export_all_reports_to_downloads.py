"""
scripts/export_all_reports_to_downloads.py
-------------------------------------------
Exports all project reports, plans, metrics, logs, test suites, and visual evidence
to C:\\Users\\ad\\Downloads\\CNTT-KLCN155_Bao_Cao_Va_Bang_Chung
and archives it into a ZIP file in Downloads for PM Ngô Thanh Nhân.
"""

import os
import sys
import shutil
import hashlib
import zipfile
from pathlib import Path
import pandas as pd

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    project_root = Path(__file__).resolve().parent.parent
    downloads_root = Path(r"C:\Users\ad\Downloads")
    export_dir = downloads_root / "CNTT-KLCN155_Bao_Cao_Va_Bang_Chung"
    zip_output_path = downloads_root / "CNTT-KLCN155_Bao_Cao_Va_Bang_Chung.zip"

    print(f"[INFO] Project Root:   {project_root}")
    print(f"[INFO] Export Folder:  {export_dir}")
    print(f"[INFO] ZIP Output:     {zip_output_path}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    dir_reports = export_dir / "01_Bao_Cao_Nghiem_Thu_Va_Ke_Hoach"
    dir_gate_a_runtime = export_dir / "02_Toi_Uu_Hoa_CPU_Va_Runtime_Gate_A"
    dir_part01 = export_dir / "03_Ket_Qua_Thuc_Nghiem_Part01_MobileNetV3"
    dir_audit_real = export_dir / "04_Kiem_Toan_Du_Lieu_That_Va_Detection_Data"
    dir_tool_evidence = export_dir / "05_Bang_Chung_Kiem_Thu_Cong_Cu_Review_Tool"
    dir_src = export_dir / "06_Ma_Nguon_Cong_Cu_Va_Kiem_Thu"

    for d in [dir_reports, dir_gate_a_runtime, dir_part01, dir_audit_real, dir_tool_evidence, dir_src]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy Docs & Plans & Formal Word Report
    docs_plan = project_root / "docs" / "plan"
    if docs_plan.exists():
        for f in docs_plan.glob("*.md"):
            shutil.copy2(f, dir_reports / f.name)
    pm_scope = project_root / "docs" / "pm" / "P0_SCOPE_AUDIT.md"
    if pm_scope.exists():
        shutil.copy2(pm_scope, dir_reports / "P0_SCOPE_AUDIT.md")
    docx_report = project_root / "artifacts" / "part02" / "Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx"
    if docx_report.exists():
        shutil.copy2(docx_report, dir_reports / docx_report.name)

    # 2. Copy Gate A & CPU Runtime Optimization Artifacts
    part02_artifacts = project_root / "artifacts" / "part02"
    if part02_artifacts.exists():
        for f in (part02_artifacts / "gate_a").glob("*.*"):
            shutil.copy2(f, dir_gate_a_runtime / f.name)
        opt_dir = part02_artifacts / "runtime_optimization"
        if opt_dir.exists():
            for f in opt_dir.glob("*.*"):
                shutil.copy2(f, dir_gate_a_runtime / f.name)

    # 3. Copy Part 1 Artifacts
    official_run = project_root / "artifacts" / "official_run"
    if official_run.exists():
        for f in official_run.glob("*.*"):
            if not f.name.endswith(".pt"):
                shutil.copy2(f, dir_part01 / f.name)

    # 4. Copy Audit & Real Detection Data
    real_audit_dir = part02_artifacts / "real_data_audit"
    if real_audit_dir.exists():
        for item in real_audit_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dir_audit_real / item.name)
            elif item.is_dir():
                shutil.copytree(item, dir_audit_real / item.name, dirs_exist_ok=True)

    data_audit = project_root / "data" / "audit"
    if data_audit.exists():
        for f in data_audit.glob("*.*"):
            shutil.copy2(f, dir_audit_real / f.name)

    data_det = project_root / "data" / "detection"
    if data_det.exists():
        for f in data_det.glob("*.*"):
            shutil.copy2(f, dir_audit_real / f.name)

    # 5. Copy Tool Verification Evidence (Screenshots & Logs)
    mcp_evidence_dir = part02_artifacts / "mcp_test_evidence"
    if mcp_evidence_dir.exists():
        for f in mcp_evidence_dir.glob("*.png"):
            shutil.copy2(f, dir_tool_evidence / f.name)
    ui_png = part02_artifacts / "ui_evidence" / "review_tool_verified.png"
    if ui_png.exists():
        shutil.copy2(ui_png, dir_tool_evidence / ui_png.name)
    val_json = part02_artifacts / "detection_annotation_validation.json"
    if val_json.exists():
        shutil.copy2(val_json, dir_tool_evidence / val_json.name)

    # 6. Copy Core Scripts and Test Files
    scripts_to_copy = [
        project_root / "scripts" / "evaluate_final_test.py",
        project_root / "scripts" / "benchmark_classifier_runtime.py",
        project_root / "scripts" / "prepare_detection_dataset.py",
        project_root / "scripts" / "audit_real_detection_images.py",
        project_root / "scripts" / "cluster_detection_groups.py",
        project_root / "scripts" / "validate_detection_annotations.py",
        project_root / "scripts" / "verify_review_tool_ui.py",
        project_root / "scripts" / "test_review_tool_functional.py",
        project_root / "scripts" / "check_split_leakage.py",
        project_root / "scripts" / "reproduce_official_validation.py",
        project_root / "src" / "ui" / "review_tool.py",
        project_root / "tests" / "conftest.py",
        project_root / "tests" / "test_verification_gates.py",
        project_root / "tests" / "test_detection_annotations.py"
    ]
    for s in scripts_to_copy:
        if s.exists():
            shutil.copy2(s, dir_src / s.name)

    # Generate Readme Index
    readme_content = """# TỔNG MỤC TÀI LIỆU BÁO CÁO VÀ BẰNG CHỨNG THỰC NGHIỆM
## DỰ ÁN CNTT-KLCN155 — HỆ THỐNG PHÁT HIỆN VÀ PHÂN LOẠI RÁC THẢI ĐA ĐỐI TƯỢNG

**Người nhận:** PM Ngô Thanh Nhân (HUIT)  
**Ngày xuất gói:** 02/10/2026  
**Thư mục nguồn:** `D:\\CNTT-KLCN155-waste-detection`  
**Gói nén đính kèm:** `CNTT-KLCN155_Bao_Cao_Va_Bang_Chung.zip`

---

### CẤU TRÚC THƯ MỤC XUẤT RA:

#### 1. `01_Bao_Cao_Nghiem_Thu_Va_Ke_Hoach/`
- **`Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx`**: Báo cáo Word chính thức kiểm thử 9 ca chức năng của công cụ rà soát nhãn (kèm ảnh chụp giao diện và giải trình trạng thái MCP).
- **`PART_02_CPU_OPTIMIZATION.md`**: Báo cáo chi tiết tối ưu hóa thời gian thực thi CPU (PyTorch vs ONNX Runtime 6 luồng đạt 7,40 ms, đối chứng tương đương 100% trên 2.223 ảnh validation).
- **`PART_02_REAL_DATA_AUDIT.md`**: Báo cáo kiểm toán 114 ảnh thật OpenImages (phát hiện 88,6% là giày đang mang trên chân người sống, bị xóa 239 nhãn quần áo).
- **`PART_02_COLLECTION_AND_LABELING_PLAN.md`**: Kế hoạch thu thập 500+ ảnh rác thực tế TP.HCM và quy chuẩn gán nhãn đa rác (10 lớp, quy trình QA/QC 2 vòng).
- **`PART_02_ACCEPTANCE.md`**: Báo cáo tổng hợp Task 2 (Gate A + chuẩn bị dữ liệu đa rác).
- **`PART_02_EVALUATION_PROTOCOL.md`**: Quy chuẩn đánh giá đóng băng của Gate A.
- **`PART_02_TASKS_AND_GATES.md`**: Phân rã nhiệm vụ và tiêu chí cổng kiểm soát Task 2.
- **`PART_01_ACCEPTANCE.md`**: Báo cáo nghiệm thu phân loại đơn rác MobileNetV3.
- **`PART_01_VERIFICATION_R2.md` & `PART_01_CHANGELOG_R2.md`**: Báo cáo kiểm định và lịch sử sửa đổi Phần 1.
- **`00_MASTER_PLAN.md`**: Kế hoạch tổng thể dự án.
- **`BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md`**: Báo cáo tổng kết kiểm kê và kế hoạch chi tiết.
- **`REPRODUCIBILITY_GUIDE.md`**: Hướng dẫn tái hiện độc lập toàn bộ kết quả.
- **`P0_SCOPE_AUDIT.md`**: Báo cáo kiểm toán phạm vi P0 ban đầu.

#### 2. `02_Toi_Uu_Hoa_CPU_Va_Runtime_Gate_A/`
- **`mobilenetv3_large_waste.onnx`**: Model MobileNetV3-Large đã xuất ONNX (16,8 MB, FP32).
- **`validation_pytorch_vs_onnx_comparison.csv`**: Đối chứng dự đoán PyTorch FP32 vs ONNX FP32 trên toàn bộ 2.223 ảnh validation (100,0000% Top-1 match, max diff logit 5.63e-5).
- **`runtime_benchmark_results.json`**: Số liệu đo lường chi tiết PyTorch vs ONNX trên 1, 2, 4, 6, 8, 12 luồng CPU AMD Ryzen 5 6600H.
- **`optimization_protocol_frozen.json`**: Giao thức đóng băng trước khi benchmark tối ưu hóa CPU.
- **`raw_timings.csv`**: Dữ liệu thô 3.000 lần suy luận trên 500 ảnh validation thực tế.
- **`gate_a_metrics.json`**: Kết quả đo Gate A gốc trên final test (Acc: 96,13%, F1: 0,9578, Battery Recall: 95,58%, Latency: 23,86 ms).
- **`gate_a_protocol_frozen.json`**: Giao thức đóng băng Gate A gốc.
- **`gate_a_raw_execution.log`**: Log chạy suy luận thực tế trên 2.223 ảnh test.
- **`test_predictions.csv`**: Bảng dự đoán 2.223 ảnh test kèm hash SHA-256 đọc trực tiếp từ đĩa.
- **`test_confusion_matrix.csv`**: Ma trận nhầm lẫn 10x10 trên tập final test.
- **`test_error_analysis.csv`**: Danh sách chi tiết 86 ca đoán sai trên tập test.

#### 3. `03_Ket_Qua_Thuc_Nghiem_Part01_MobileNetV3/`
- **`official_training_metrics.json`**: Số liệu huấn luyện chính thức 12 epoch mô hình MobileNetV3-Large.
- **`training_history.csv`**: Lịch sử loss/acc từng epoch.
- **`val_predictions.csv` & `val_confusion_matrix.csv`**: Kết quả xác minh trên 2.223 ảnh validation (Acc: 96,18%, F1: 0,9559).
- **`val_error_analysis.csv`**: Danh sách 85 ca đoán sai trên tập validation.
- **`reproduced_validation_summary.json`**: Báo cáo tái hiện độc lập của bên thứ ba.
- **`standalone_package_verification.log`**: Log chạy gói bàn giao độc lập ngoài repo.

#### 4. `04_Kiem_Toan_Du_Lieu_That_Va_Detection_Data/`
- **`real_images_audit_table.csv`**: Kết quả kiểm toán bằng mắt 114/114 ảnh thật OpenImages (103 giày đang mang, 10 ảnh sản phẩm studio, 1 giày ngoài trời).
- **`real_data_audit_summary.json`**: Tổng hợp số liệu kiểm toán ảnh thực tế và số bounding box bị tước bỏ.
- **`contact_sheets/`**: Thư mục chứa 10 contact sheets trực quan (`contact_sheet_01.jpg` đến `contact_sheet_10.jpg`) hiển thị toàn bộ 114 ảnh thực tế kèm nhãn giày và nhãn OpenImages gốc.
- **`thirteen_images_inspection.jpg`**: Ảnh phóng to trực quan 13 ảnh nghi vấn (10 ảnh thương mại, 2 ca hiếm, 1 ca rác ngoài trời).
- **`manifest_detection_v1.csv`**: Bảng kê 1.419 ảnh detection đã bổ sung Group ID phân cụm (pHash Hamming distance <= 4).
- **`leakage_audit_report.json`**: Báo cáo kiểm toán rò rỉ hash (exact SHA-256 = 0, candidate phash = 33, resolved = 33).
- **`phash_decision_table.csv`**: Bảng 33 quyết định kiểm chứng thủ công từng cặp ảnh gần giống nhau.
- **`split_manifest_v2.csv`**: Bảng kê 14.829 ảnh phân loại đơn rác.
- **`dataset_summary.json`**: Thống kê số lượng box từng lớp trên dữ liệu đa rác.

#### 5. `05_Bang_Chung_Kiem_Thu_Cong_Cu_Review_Tool/`
- **`case_01_open_and_display.png` đến `case_09_manifest_audit_log.png`**: 9 ảnh chụp màn hình tương ứng với 9 ca kiểm thử chức năng tự động trên sandbox cô lập.
- **`mcp_live_review_tool.png`**: Ảnh chụp màn hình live stream từ Chrome DevTools MCP trực tiếp trên port 8501.
- **`review_tool_verified.png`**: Ảnh chụp màn hình giao diện review tool hoàn chỉnh.
- **`detection_annotation_validation.json`**: Kết quả kiểm toán 4.602 bounding box (1.419 file nhãn, 0 lỗi hình học hay cú pháp).

#### 6. `06_Ma_Nguon_Cong_Cu_Va_Kiem_Thu/`
- **`evaluate_final_test.py`**: Mã nguồn đánh giá final test Gate A.
- **`benchmark_classifier_runtime.py`**: Mã nguồn benchmark tối ưu hóa CPU runtime (PyTorch vs ONNX đa luồng).
- **`audit_real_detection_images.py`**: Mã nguồn kiểm toán ảnh thực tế OpenImages và đối soát bounding box gốc.
- **`cluster_detection_groups.py`**: Mã nguồn phân cụm Group ID chống rò rỉ phân vùng detection.
- **`prepare_detection_dataset.py`**: Mã nguồn chuẩn bị và phân lập dữ liệu đa rác.
- **`validate_detection_annotations.py`**: Mã nguồn kiểm toán cú pháp và hình học bounding box.
- **`review_tool.py`**: Mã nguồn ứng dụng Streamlit rà soát và chỉnh sửa nhãn đa rác.
- **`test_review_tool_functional.py`**: Bộ kịch bản kiểm thử tự động 9 ca giao diện review tool.
- **`verify_review_tool_ui.py`**: Script kiểm thử cơ bản giao diện review tool.
- **`check_split_leakage.py`**: Script kiểm tra rò rỉ dữ liệu.
- **`reproduce_official_validation.py`**: Script tái hiện suy luận validation.
- **`test_verification_gates.py` & `test_detection_annotations.py`**: Bộ kiểm thử tự động 11 test cases (100% PASS).
"""

    with open(export_dir / "00_HUONG_DAN_DOC_BAO_CAO.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Generate Manifest SHA-256 for all exported files
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

    # Zip the entire folder
    print(f"\n[INFO] Compressing {len(manifest_rows)} files into {zip_output_path}...")
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in export_dir.rglob("*"):
            if f.is_file():
                arcname = f.relative_to(export_dir)
                zipf.write(f, arcname)

    zip_sha = compute_sha256(zip_output_path)
    zip_size_mb = zip_output_path.stat().st_size / (1024 * 1024)

    print(f"\n[SUCCESS] Export Completed!")
    print(f"Directory:    {export_dir}")
    print(f"Total Files:  {len(manifest_rows)}")
    print(f"ZIP Archive:  {zip_output_path}")
    print(f"ZIP Size:     {zip_size_mb:.2f} MB")
    print(f"ZIP SHA-256:  {zip_sha}")

if __name__ == "__main__":
    main()
