# BÁO CÁO XÁC MINH VÀ SỬA MAPPING 10 LỚP TRÊN DỮ LIỆU THẬT (TASK 1 - R2 REVISED)

**Dự án:** `D:\CNTT-KLCN155-waste-detection`  
**Thời gian cập nhật:** 09/10/2026  
**Môi trường thực thi (Trích xuất từ `environment_and_execution_log.json`):**
- **Hệ điều hành:** Windows-11-10.0.26200-SP0
- **Python:** 3.12.10 (`D:\CNTT-KLCN155-waste-detection\.venv\Scripts\python.exe`)
- **PyTorch:** `2.5.1+cu121` (CUDA Available: True, GPU: NVIDIA GeForce RTX 2050)
- **Torchvision:** `0.20.1+cu121` | **Ultralytics:** `8.4.171`
- **Git Baseline Commit:** `176317b41554a323209bb5403b96f59c43b86566`

---

## 1. TÓM TẮT ĐIỀU HÀNH & KẾT QUẢ SỬA BLOCKER

Báo cáo này được sinh tự động trực tiếp từ các file kết quả máy sinh (`annotation_inventory_summary.json`, `annotation_inventory.csv`, `mutation_test_red_green_log.json`, `mendeley_conversion_audit.json`, `checkpoint_inspection_report.json`, và `crop_manifest.json`). Toàn bộ các phát hiện và xử lý được thực hiện trên dữ liệu, code và checkpoint thật của dự án:

### 1.1. Sửa lỗi `medical -> battery` và kiểm soát chặt chẽ Automatic Mapping
- **Hiện trạng trước sửa đổi:** Trong `src/detection/schema.py`, `SOURCE_LABEL_MAPPING["vn_trash"]` từng chứa ánh xạ `"medical": "battery"` và `"organic": "biological"`. Tuy nhiên dữ liệu thô gốc `D:\HK7\Đồ án khóa luận\Data\VN Trash classification` chỉ có đúng 9 thư mục nhãn (`Alu, Carton, Foam_box, Milk_box, Other, Paper, Paper_cup, PET, Plastic_cup`), hoàn toàn không có `medical`.
- **Hành động khắc phục:**
  1. Loại bỏ triệt để `medical` và `organic` khỏi `SOURCE_LABEL_MAPPING["vn_trash"]`.
  2. Cập nhật hàm `map_source_label`: Khi nhận nhãn `medical` hoặc `organic` mà không có chuyên gia chỉ định, hệ thống **TỪ CHỐI TỰ ĐỘNG (REJECT)** và ném ngoại lệ `DetectionDatasetError("Exclude or manually review vn_trash/medical; a material class and reviewer are required")`.
  3. Chỉ cho phép gán nhãn khi có đầy đủ tham số `manual_class` hợp lệ và định danh `reviewer`.
  4. Đã bổ sung regression test `test_schema_rejects_automatic_medical_and_organic_mapping` và `test_real_source_converter_independent_verification` gọi trực tiếp code sản phẩm và căn cứ theo ground truth độc lập `data/metadata/label_mapping.csv` để khóa chặn vĩnh viễn.

### 1.2. Đánh giá lại bản chất dữ liệu Mendeley: Lớp hỗn hợp (Mixed / Unknown Classes) & Cách ly Quarantine
- **Kiểm chứng thị giác trên 72 crop mẫu có provenance chuẩn:**
  * **Class 2 (từng kết luận là bìa carton):** Bằng chứng thực tế cho thấy chứa cả **Khẩu trang y tế (surgical face mask)** (`mendeley_c2_sample1_syn_000266.jpg`), **túi lưới nilon nhựa** (`mendeley_c2_sample0_syn_000037.jpg`), và các mảnh bìa -> **Lớp hỗn hợp / Không đồng nhất (Mixed / Unknown)**.
  * **Class 4 (từng kết luận là thủy tinh):** Bằng chứng thực tế cho thấy chứa cả **Túi giấy McDonald's** (`mendeley_c4_sample0_syn_003640.jpg`), **Vỏ hộp bánh Pizza** (`mendeley_c4_sample1_syn_003199.jpg`), **Bao bì kẹo KitKat bằng nhựa**, và chai lọ thủy tinh -> **Lớp hỗn hợp / Không đồng nhất (Mixed / Unknown)**.
  * **Class 6:** Chủ yếu là lon nhôm bia/nước ngọt (`mendeley_c6_sample0_syn_001448.jpg`), nhưng có background composite.
  * **Class 8:** Chứa cành cây, cọng lá, vỏ chuối (`mendeley_c8_sample0_syn_004474.jpg`).
  * **Class 9:** Chứa giày da/thể thao đen (`mendeley_c9_sample0_syn_001247.jpg`) và các mảnh rác vụn.
  * **Class 7:** **Phân biệt rõ ràng giữa toàn nguồn và matched subset**: Trong toàn bộ 2,000 file nguồn Mendeley thô có **699 boxes** Class 7. Tuy nhiên, trong 1,305 file synthetic được chọn vào `dataset-v1`, Class 7 nguồn xuất hiện đúng **0 boxes** (không có file nào chứa Class 7 gốc được đưa vào v1). Toàn bộ **472 boxes** mang nhãn đích Class 7 (`plastic`) trong dataset-v1 thực chất xuất phát từ Class 0 của Mendeley được convert sang.
- **Quy tắc quyết định & Hành động Quarantine:**
  1. **TUYỆT ĐỐI KHÔNG remap nguyên lớp 1-1 mù quáng cho Mendeley** dựa trên abstract văn bản. Dữ liệu Mendeley là tập dữ liệu cắt dán nhân tạo có mức độ nhiễu cao.
  2. **Cách ly / Quarantine chính thức Class 2, Class 4**: Đã đánh dấu 640 file synthetic chứa 928 bounding box của Class 2 và Class 4 vào `data/detection/labels/mendeley_quarantine_manifest.json` và cập nhật chính sách `DO_NOT_REMAP_WHOLE_CLASS` trong `configs/detection_source_mapping.yaml`.

### 1.3. Ràng buộc Runtime Checkpoint & Phân định rõ ràng từng Checkpoint từ Logs thật
Ứng dụng web runtime (`src/web/detection_settings.py`) ràng buộc cụ thể với:
- **Mô hình chính (Primary Detector):** `DEFAULT_SSDLITE_PATH = Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt")` (Backend mặc định: `ssdlite`).
- **Mô hình đối chứng (Secondary Detector):** `DEFAULT_YOLO_PATH = Path("artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt")`.
- **Số liệu huấn luyện trích xuất trực tiếp từ logs và checkpoints thật (không chèn số cố định):**
  1. `waste-ssdlite320-mobilenetv3-combined-seed42`: Kích thước 28,456,982 bytes, SHA-256 `584c5980eb4ebe37...`. Head có 6 đầu dò, `out_channels=66` ($6 \times 11$ classes gồm background 0). Huấn luyện hoàn thành **59/120 epochs** budget (dừng sớm ở epoch 59 với patience 30 do best epoch đạt tại epoch **29** với val mAP50-95 = **0.04888**). Nạp code sản phẩm: **PASS**. Nhưng do tập train chứa 1,305 ảnh Mendeley (~90% box train) có nhãn sai lệch và hỗn hợp, mô hình mang **khuyết tật nhãn huấn luyện (Semantic Defect)**. **CẦN RETRAIN**.
  2. `waste-yolov8n-combined-seed42`: Kích thước 6,217,379 bytes, SHA-256 `97f46ccaaa0a63c5...`. Head 10 classes. Huấn luyện hoàn thành **67/120 epochs** budget (dừng sớm ở epoch 67 với patience 30 do best epoch đạt tại epoch **37** với val mAP50-95 = **0.04456**). Nạp code sản phẩm: **PASS**. Cùng chung tập dữ liệu huấn luyện sai lệch, mang **khuyết tật nhãn huấn luyện (Semantic Defect)**. **CẦN RETRAIN**.
  3. `smoke-ssdlite320`: Kích thước 28,456,918 bytes, SHA-256 `c9a079d630bdff79...`. Epoch 2/2 budget. **KHÔNG DÙNG TRONG RUNTIME ỨNG DỤNG**. Đây chỉ là checkpoint kiểm tra hạ tầng CI smoke test.
  4. `smoke-yolov8n`: Kích thước 6,208,931 bytes, SHA-256 `84b677fcb4de6cb0...`. Epoch 2/2 budget. **KHÔNG DÙNG TRONG RUNTIME ỨNG DỤNG**. Đây chỉ là checkpoint kiểm tra hạ tầng CI smoke test.

---

## 2. BẢNG MAPPING CHI TIẾT & ĐÁNH GIÁ TỪ NGUỒN THẬT

| Nguồn dữ liệu | ID/Nhãn gốc | ID Chuẩn | Tên chuẩn Canonical | Nơi xử lý | Bằng chứng đối chiếu | Đánh giá & Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VN-Trash (Raw)** | `Alu` | 5 | `metal` (Kim loại) | `src/detection/schema.py` | 89 ảnh lon nhôm trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Carton` | 2 | `cardboard` (Bìa carton) | `src/detection/schema.py` | 134 ảnh thùng bìa giấy trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Foam_box` | 7 | `plastic` (Nhựa) | `src/detection/schema.py` | 98 ảnh hộp xốp nhựa trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Milk_box` | 2 | `cardboard` (Bìa carton) | `src/detection/schema.py` | 112 ảnh vỏ hộp sữa trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Other` | 9 | `trash` (Rác khác) | `src/detection/schema.py` | 65 ảnh rác vụn hỗn hợp | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Paper` | 6 | `paper` (Giấy) | `src/detection/schema.py` | 105 ảnh giấy vụn trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Paper_cup` | 6 | `paper` (Giấy) | `src/detection/schema.py` | 76 ảnh cốc giấy trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `PET` | 7 | `plastic` (Nhựa) | `src/detection/schema.py` | 142 ảnh chai nhựa PET trong folder raw | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Raw)** | `Plastic_cup` | 7 | `plastic` (Nhựa) | `src/detection/schema.py` | 88 ảnh cốc nhựa trà sữa | **Đã xác minh & ánh xạ tự động** |
| **VN-Trash (Legacy)**| `medical` | N/A | Không ánh xạ tự động | `src/detection/schema.py` | Không có thư mục `medical` trong raw | **BỊ TỪ CHỐI TỰ ĐỘNG (Cần Review)** |
| **VN-Trash (Legacy)**| `organic` | N/A | Không ánh xạ tự động | `src/detection/schema.py` | Không có thư mục `organic` trong raw | **BỊ TỪ CHỐI TỰ ĐỘNG (Cần Review)** |
| **Mendeley Class 0** | Class 0 | 7 | `plastic` (v1 cũ) | Legacy converter (Word 0) | Crop: Khăn giấy, giấy ăn, màng nilon | **Lớp hỗn hợp / Nhiễu (Mixed / Noisy)** |
| **Mendeley Class 1** | Class 1 | 6 | `paper` (v1 cũ) | Legacy converter (Word 1) | Crop: Gói snack Lay's, cốc giấy | **Lớp hỗn hợp / Composite** |
| **Mendeley Class 2** | Class 2 | 2 | `cardboard` (v1 cũ) | Legacy converter (Word 2) | Crop: **Khẩu trang y tế**, túi lưới nilon | **QUARANTINED (Lớp hỗn hợp / Không remap)** |
| **Mendeley Class 3** | Class 3 | 5 | `metal` (v1 cũ) | Legacy converter (Word 3) | Crop: Cuộn dây dù, lưới nilon | **Lớp hỗn hợp / Sai lệch** |
| **Mendeley Class 4** | Class 4 | 4 | `glass` (v1 cũ) | Legacy converter (Word 4) | Crop: **Hộp Pizza, bao bì McDonald's, KitKat** | **QUARANTINED (Lớp hỗn hợp / Không remap)** |
| **Mendeley Class 5** | Class 5 | 1 | `biological` (v1 cũ) | Legacy converter (Word 5) | Crop: Bao thuốc lá THIS, nắp hộp | **Lớp hỗn hợp / Sai lệch** |
| **Mendeley Class 6** | Class 6 | 0 | `battery` (v1 cũ) | Legacy converter (Word 6) | Crop: Lon bia, lon nước ngọt kim loại | **Nhiều lon kim loại (Bị gán pin sai)** |
| **Mendeley Class 7** | Class 7 | N/A | Khuyết trong subset v1 | Converter bỏ qua | 699 boxes toàn nguồn, 0 boxes trong subset | **Khuyết trong matched subset v1** |
| **Mendeley Class 8** | Class 8 | 3 | `clothes` (v1 cũ) | Legacy converter (Word 8) | Crop: Cành lá, cọng cây, vỏ chuối | **Thiên về hữu cơ (Bị gán áo quần sai)** |
| **Mendeley Class 9** | Class 9 | 9 | `trash` (v1 cũ) | Legacy converter (Word 9) | Crop: Giày thể thao đen, rác vụn | **Lớp hỗn hợp (Giày dép + Rác)** |
| **TACO** | Categories | 0-9 | TAXONOMY_10 | `configs/detection_source_mapping.yaml` | 60 categories COCO chuẩn | Đã xác minh |
| **OpenImages** | MIDs | 0-9 | TAXONOMY_10 | `configs/detection_source_mapping.yaml` | Boxable classes chuẩn | Đã xác minh |

---

## 3. THỐNG KÊ ĐỐI CHIẾU NGUỒN MENDELEY & DATASET-V1 (CÓ ĐƠN VỊ ĐO RÕ RÀNG)

Số liệu trích xuất trực tiếp từ `reports/mendeley_conversion_audit.json` và `crop_manifest.json`:
- **Tổng số file ảnh/nhãn tổng hợp trong `dataset-v1`:** 1305 files (train: 918 files, val: 210 files, test: 177 files).
- **Số file đối chiếu khớp giữa Mendeley và dataset-v1:** 1305 files (100.0%).
- **Số file Mendeley không nằm trong dataset-v1:** 564 files (file nguồn Mendeley không được đưa vào tập v1).
- **Số file dataset-v1 không tìm thấy gốc Mendeley:** 0 files.
- **Tổng số bounding box đối chiếu hình học:** 4095 boxes.
- **Số box trùng khớp hoàn toàn:** 4095 / 4095 boxes (100.0% matched, 0 unmatched, 0 duplicate).
- **Số file Mendeley trùng lặp stem giữa 2 split train và val:** 131 duplicate stems (tổng 2,000 files nguồn gồm 1,869 unique stems).

### Chi tiết cặp chuyển đổi Source - Target (Đơn vị: boxes):
| Mendeley Source Class | Dataset-v1 Target Class | Tên chuẩn Canonical | Số lượng Bounding Box | Tỷ lệ (%) |
| :--- | :--- | :--- | :--- | :--- |
| Mendeley Class 0 | Class 7 | `plastic` | 472 boxes | 100.0% |
| Mendeley Class 1 | Class 6 | `paper` | 422 boxes | 100.0% |
| Mendeley Class 2 | Class 2 | `cardboard` | 452 boxes | 100.0% |
| Mendeley Class 3 | Class 5 | `metal` | 448 boxes | 100.0% |
| Mendeley Class 4 | Class 4 | `glass` | 476 boxes | 100.0% |
| Mendeley Class 5 | Class 1 | `biological` | 446 boxes | 100.0% |
| Mendeley Class 6 | Class 0 | `battery` | 462 boxes | 100.0% |
| Mendeley Class 8 | Class 3 | `clothes` | 462 boxes | 100.0% |
| Mendeley Class 9 | Class 9 | `trash` | 455 boxes | 100.0% |

---

## 4. KIỂM KÊ (INVENTORY) TOÀN BỘ ANNOTATION DATASET HIỆN HÀNH

Số liệu trích xuất chính xác 100% từ `reports/annotation_inventory_summary.json` và `reports/annotation_inventory.csv`:
- **Tổng số file annotation đã kiểm tra:** 1562 files.
- **Tổng số bounding box đã kiểm định:** 5750 boxes.
- **Số file lỗi cú pháp / sai format:** 0 files (0 syntax error).
- **Số file có cảnh báo trích xuất từ CSV:** 1 file (`data/detection/labels/real/taco_0804.txt`: [DEGENERATE_BOX] L6: Extremely tiny box: w=0.012663, h=0.001838).
- **Phân bố số lượng Bounding Box theo từng Class Canonical:**
  * **Class 0 (`battery`):** 464 boxes
  * **Class 1 (`biological`):** 454 boxes
  * **Class 2 (`cardboard`):** 521 boxes
  * **Class 3 (`clothes`):** 520 boxes
  * **Class 4 (`glass`):** 557 boxes
  * **Class 5 (`metal`):** 557 boxes
  * **Class 6 (`paper`):** 511 boxes
  * **Class 7 (`plastic`):** 878 boxes
  * **Class 8 (`shoes`):** 514 boxes
  * **Class 9 (`trash`):** 774 boxes
- **Phân bố theo nguồn dữ liệu:**
  * `synthetic_mendeley`: 1305 files, 4095 boxes.
  * `real_taco`: 105 files, 1049 boxes.
  * `real_openimages`: 152 files, 606 boxes.

---

## 5. KẾT QUẢ KIỂM THỬ TRÊN ĐƯỜNG CHẠY THẬT & MUTATION TESTING

### 5.1. Test Suite Sản Phẩm (Pytest 18 passed)
- Lệnh chạy:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/detection/ tests/test_detection_annotations.py -v --junitxml=reports/junit_test_report.xml
  ```
- **Kết quả:** **18 PASSED / 0 FAILED** (35.08 giây). Trích xuất trực tiếp từ `reports/junit_test_report.xml`.
- **Cải tiến trong test suite:**
  * `test_real_dataset_loader_background_shift`: Đọc trực tiếp từng file annotation gốc trên ổ đĩa, kiểm tra từng bounding box để đảm bảo nhãn nạp vào tensor target luôn thỏa mãn `target_id == raw_id + 1` (khắc phục kiểm tra miền giá trị lỏng lẻo trước đây).
  * `test_real_source_converter_independent_verification`: Đọc dữ liệu mẫu/metadata độc lập từ `data/metadata/label_mapping.csv` và `data/metadata/dataset_info.csv` trên ổ đĩa. Tuyệt đối không lấy từ cùng dictionary/config đang kiểm tra. Với dữ liệu classification, kiểm tra material mapping mà không sinh bbox giả. Kiểm tra nghiêm ngặt từ chối tự động `medical`/`organic` và chấp nhận khi có reviewer chỉ định.
  * Phân định rõ ràng trong docstrings: Các fixture tensor trong `test_real_ssdlite_tensor_decoder_shift` và `test_yolo_ssd_fusion_cross_agreement` được ghi chú rõ là **[UNIT FIXTURE]** dùng để kiểm tra giải thuật toán học/WBF, không dùng làm bằng chứng inference của model huấn luyện thật.

### 5.2. Kiểm thử đột biến cô lập (Mutation Testing)
Số liệu trích xuất trực tiếp từ `reports/mutation_test_red_green_log.json`:
- **Thư mục cô lập:** `<scratch>/isolated_mutation_run` (Bảo toàn working tree người dùng).
- **Đột biến RED (Cố ý làm sai cả loader shift và decoder shift):**
  * Sửa `labels.append(int(c) + 1)` $\rightarrow$ `labels.append(int(c))` trong `ssdlite_train.py`.
  * Sửa `class_index = int(label) - 1` $\rightarrow$ `class_index = int(label)` trong `ssdlite.py`.
  * **Kết quả:** Exit Code = **1** (Thất bại: **2 failed, 4 passed**).
  * Lỗi ghi nhận:
    1. `FAILED test_real_dataset_loader_background_shift`: `AssertionError: Loader failed shift on taco_0001.txt: raw 2 was loaded as 2, expected 3`.
    2. `FAILED test_real_ssdlite_tensor_decoder_shift`: `AssertionError: assert 1 == 0`.
- **Khôi phục GREEN (Khôi phục nguyên trạng):**
  * **Kết quả:** Exit Code = **0** (**6 passed**).

---

## 6. PHÂN TÁCH THAY ĐỔI & BẢN VÁ GIT PATCH

### 6.1. Bảo toàn Dirty Changes có trước của User
Các file sau thuộc working tree của user trước task và được giữ nguyên 100%:
- `configs/detection_dataset.yaml`
- `data/audit/leakage_audit_report.json`
- `src/web/detection_logic.py`
- `src/web/detection_settings.py`
- `streamlit_app.py`

### 6.2. Thay đổi Task 1 đã triển khai:
1. `src/detection/schema.py`: 9 nhãn raw VN-Trash và cơ chế từ chối `medical`/`organic` tự động.
2. `configs/detection_source_mapping.yaml`: Đã bổ sung cấu hình nguồn `mendeley` với chính sách cách ly `DO_NOT_REMAP_WHOLE_CLASS` và ghi chú từng class.
3. `data/detection/labels/mendeley_quarantine_manifest.json`: Manifest cách ly 640 file synthetic chứa Class 2 và Class 4 mixed/unknown.
4. `reports/reproduce_scripts/audit_mendeley_mapping.py`: Kiểm toán toàn diện xuất đầy đủ unique/matched/unmatched/duplicate counters và source-target pairs đúng đơn vị (files, boxes), phân biệt Class 7 toàn nguồn vs matched subset.
5. `tests/detection/test_real_mapping_integration.py`: Sửa `test_real_source_converter_independent_verification` đọc ground truth độc lập từ `data/metadata/label_mapping.csv` và `dataset_info.csv`, không sinh bbox giả.
6. `reports/reproduce_scripts/generate_verified_report.py`: Đọc động CSV/JSON/JUnit/train logs thật để sinh báo cáo không chèn số cố định.

---

## 7. HƯỚNG DẪN TÁI LẬP (REPRODUCTION COMMANDS)

1. **Kiểm tra môi trường:**
   ```powershell
   .\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```
2. **Tái lập đối chiếu nhãn Mendeley & sinh Quarantine Manifest:**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\audit_mendeley_mapping.py
   ```
3. **Tái lập sinh 72 crop và manifest có provenance chuẩn:**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\crop_and_audit_mendeley_classes.py
   ```
4. **Tái lập kiểm tra 4 Checkpoint:**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\inspect_all_checkpoints.py
   ```
5. **Tái lập kiểm kê 1,562 Annotation files:**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\generate_annotation_inventory.py
   ```
6. **Chạy toàn bộ PyTest Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/detection/ tests/test_detection_annotations.py -v --junitxml=reports/junit_test_report.xml
   ```
7. **Tái lập kiểm thử đột biến cô lập (RED -> GREEN):**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\run_isolated_mutation_test.py
   ```
8. **Sinh báo cáo xác minh tổng hợp:**
   ```powershell
   .\.venv\Scripts\python.exe reports\reproduce_scripts\generate_verified_report.py
   ```