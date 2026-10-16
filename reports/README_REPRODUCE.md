# HƯỚNG DẪN TÁI LẬP NHANH (QUICK REPRODUCE GUIDE) - TASK 1 R2

Gói giao nộp: `Phan1_Mapping_10Lop_VERIFIED_R2.zip`  
Thư mục dự án gốc: `.`

---

## 1. Cấu trúc tệp trong gói giao nộp
- `BAO_CAO_XAC_MINH_MAPPING_10_LOP_R2.md`: Báo cáo chi tiết toàn bộ quá trình xác minh, bảng đối chiếu, kết luận checkpoint và kiến nghị.
- `annotation_inventory.csv`: Bảng kiểm kê 1,562 file annotation của dataset hiện hành kèm SHA-256, split, số lượng box, trạng thái lỗi/cảnh báo.
- `annotation_inventory_summary.json`: Tóm tắt thống kê tổng hợp số box theo từng class canonical và số cảnh báo.
- `checkpoint_inspection_report.json`: Kết quả kiểm tra sâu 4 checkpoint nhị phân thật (kích thước, SHA-256, số channel đầu ra, class names, kết quả load code sản phẩm).
- `mendeley_conversion_audit.json`: Kết quả đối chiếu máy sinh giữa 1,305 file annotation gốc Mendeley và `dataset-v1` (4,095 boxes khớp hình học 100%).
- `mutation_test_red_green_log.json`: Nhật ký kiểm thử đột biến cô lập chứng minh test suite phát hiện lỗi (RED -> GREEN).
- `junit_test_report.xml`: Báo cáo kết quả chạy 43 test suite bằng PyTest định dạng chuẩn JUnit.
- `environment_and_execution_log.json`: Thông tin chi tiết môi trường thực thi (Python, PyTorch, CUDA, Git baseline commit, danh sách dirty changes tách biệt).
- `task1_mapping_changes.patch`: Bản vá Git UTF-8 chứa toàn bộ các sửa đổi cho mapping, tên tiếng Việt và các file test tích hợp/hồi quy.
- `mendeley_crops/`: Thư mục chứa 72 ảnh crop trực quan từ 8 nhóm class Mendeley cùng `crop_manifest.json` chứng minh sai lệch nhãn thực tế.
- `reproduce_scripts/`: Toàn bộ các script Python dùng để sinh các báo cáo và kết quả kiểm chứng.

---

## 2. Các lệnh tái lập (Reproduction Commands)

Thực hiện trong terminal PowerShell tại thư mục `.`:

### Bước 1: Kích hoạt môi trường và kiểm tra GPU
```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Bước 2: Tái lập đối chiếu nhãn Mendeley và dataset-v1
```powershell
.\.venv\Scripts\python.exe reports\reproduce_scripts\audit_mendeley_mapping.py
```

### Bước 3: Trích xuất các crop trực quan kiểm tra đối tượng thực tế
```powershell
.\.venv\Scripts\python.exe reports\reproduce_scripts\crop_and_audit_mendeley_classes.py
```

### Bước 4: Kiểm tra sâu 4 checkpoint thật
```powershell
.\.venv\Scripts\python.exe reports\reproduce_scripts\inspect_all_checkpoints.py
```

### Bước 5: Kiểm kê 1,562 file annotation dataset
```powershell
.\.venv\Scripts\python.exe reports\reproduce_scripts\generate_annotation_inventory.py
```

### Bước 6: Chạy test suite và kiểm tra 43 test tích hợp/hồi quy
```powershell
.\.venv\Scripts\python.exe -m pytest tests/detection/ tests/test_detection_annotations.py -v --junitxml=reports/junit_test_report.xml
```

### Bước 7: Tái lập kiểm thử đột biến cô lập (Mutation Testing)
```powershell
.\.venv\Scripts\python.exe reports\reproduce_scripts\run_isolated_mutation_test.py
```
