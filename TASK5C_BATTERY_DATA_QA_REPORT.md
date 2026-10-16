# BÁO CÁO THỰC THI SPRINT AI-UPGRADE-01 — TASK 5C
## TIẾP NHẬN NGUỒN PIN, KIỂM TOÁN DỮ LIỆU & ĐỀ XUẤT PILOT TRAINING

- **Repository:** `D:\CNTT-KLCN155-waste-detection`
- **Mã công việc:** SPRINT AI-UPGRADE-01 — TASK 5C (BATTERY DATA QA)
- **Thời điểm hoàn tất:** 2026-10-13 22:50 UTC+7
- **Phân định vai trò:** Antigravity Integrator + Chuyên trách Data / Model / QA
- **Tệp nguồn xác minh:** `Waste-battery-only.v1i.yolov8.zip` (18,593,303 bytes)
- **URL nguồn chính thức:** [https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1](https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1)
- **Gói bàn giao:** `C:\Users\ad\Downloads\CNTT_KLCN155_Task5C_Delivery.zip`
- **Cam kết Gatekeeper:** `APPROVED_FOR_TRAIN = 0` (Chờ quyết định phê duyệt của PM).

---

## 1. TỔNG QUAN KẾT QUẢ TIẾP NHẬN & GIẢI NÉN AN TOÀN

Người dùng đã tải trực tiếp tệp `.zip` chính thức từ Roboflow Universe và hệ thống đã di chuyển an toàn vào `data/acquisition_staging/roboflow_battery/`.

Quy trình giải nén và kiểm tra an toàn theo quy chuẩn:
1. **Kiểm tra an toàn lưu trữ trước giải nén:**
   - Số lượng tệp: **704 tệp** ($< 2,000$ giới hạn an toàn).
   - Dung lượng giải nén: **~19.8 MB** ($< 300$ MB giới hạn an toàn).
   - Tỷ lệ nén cao nhất: $1.8\times$ ($< 20.0\times$ giới hạn Zip Bomb).
   - Kiểm tra đường dẫn: 0 tệp chứa dấu vết Zip Slip (`..`, đường dẫn tuyệt đối, hay symlink).
2. **Thư mục giải nén độc lập:**
   - Dữ liệu được trích xuất vào thư mục riêng biệt theo timestamp: `data/acquisition_staging/roboflow_battery/extracted_20261013_154819/`.
   - Tuyệt đối không ghi đè vào thư mục dữ liệu `data/detection/` hay các split baseline v2.

---

## 2. PHÂN TÍCH `data.yaml` & MAPPING 10 LỚP CỐ ĐỊNH

Tệp `data.yaml` gốc trong archive chứa metadata:
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 3
names: ['Alkaline', 'Alkaline_jp', 'LR_Alkaline']

roboflow:
  workspace: battery-detection-yuv0z
  project: waste-battery-only
  version: 1
  license: CC BY 4.0
  url: https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1
```

### Kết quả thẩm định domain & Taxonomy Mapping:
- Cả 3 lớp nguồn đều là các phân nhóm của pin kiềm (Alkaline battery tiêu chuẩn, pin chữ Nhật Bản `Alkaline_jp`, và pin cúc áo/trụ ký hiệu chuẩn quốc tế `LR_Alkaline`).
- Trong hệ thống 10 lớp cố định của dự án `[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`:
  - **Class 0 (`Alkaline`)** $\rightarrow$ **Mapped to Class 0: `battery`**
  - **Class 1 (`Alkaline_jp`)** $\rightarrow$ **Mapped to Class 0: `battery`**
  - **Class 2 (`LR_Alkaline`)** $\rightarrow$ **Mapped to Class 0: `battery`**
- Đã cập nhật định nghĩa nguồn chính thức vào `configs/detection_source_mapping.yaml`.

---

## 3. THỐNG KÊ CHI TIẾT SỐ LƯỢNG ẢNH VÀ BOUNDING BOX THẬT

Toàn bộ **350 ảnh** và **505 bounding boxes** đã được kiểm tra hình học và giải mã ảnh:

| Tiêu chí thống kê | Tập Train (Roboflow) | Tập Validation (Roboflow) | Toàn bộ Archive | Đánh giá chất lượng |
|---|:---:|:---:|:---:|:---:|
| **Tổng số file ảnh** | 315 ảnh | 35 ảnh | **350 ảnh** | 100% giải mã PIL thành công |
| **Số ảnh gốc độc bản (Base Images)** | 105 ảnh | 35 ảnh | **140 ảnh** | Zero overlap giữa train và valid |
| **Số ảnh augmentations (3x)** | 210 ảnh | 0 ảnh | 210 ảnh | Tập valid 100% ảnh gốc sạch |
| **Tổng số Bounding Boxes** | 452 boxes | 53 boxes | **505 boxes** | 100% tọa độ hữu hạn trong $[0.0, 1.0]$ |
| - Class 0: `Alkaline` | 90 boxes | 9 boxes | **99 boxes** | Chuẩn pin kiềm |
| - Class 1: `Alkaline_jp` | 317 boxes | 38 boxes | **355 boxes** | Chuẩn pin kiềm |
| - Class 2: `LR_Alkaline` | 45 boxes | 6 boxes | **51 boxes** | Chuẩn pin cúc áo / trụ |
| **Hộp pin hợp lệ (Target Class 0)** | 452 boxes | 53 boxes | **505 boxes (100%)** | 0 hộp sai nhãn |
| **Hộp vi phạm biên (Out of bounds)** | 0 | 0 | **0** | Đã qua kiểm tra biên chặt chẽ |
| **Lỗi NaN / Infinity** | 0 | 0 | **0** | Đã kiểm tra số học |

---

## 4. KIỂM TOÁN RÒ RỈ (LEAKAGE) & TRÙNG LẶP (pHash AUDIT)

1. **Đối chiếu với tập dữ liệu dự án hiện hữu (4,046 references từ train, val, test, external holdout):**
   - **Rò rỉ SHA-256 chính xác:** **0 ảnh** (Không trùng bất kỳ ảnh nào hiện có).
   - **Rò rỉ gần đúng pHash (Hamming distance $\le 8$):** **0 ảnh** (Không có ảnh nào trùng góc chụp hay phiên chụp với tập dữ liệu nội bộ và holdout `D:\Hinh_Anh_Rac`).
2. **Kiểm tra chồng lấn giữa các Split nguồn trong Roboflow:**
   - 105 base images của `train` và 35 base images của `valid` có **độ giao nhau (intersection) = 0**.
   - Không có ảnh validation nào là phiên bản augment của ảnh train.
3. **Trùng lặp nội bộ (Internal Duplicates):**
   - Trùng lặp SHA-256 hoàn toàn: **0 ảnh** (Cả 350 file đều có checksum riêng biệt).
   - Cặp gần đúng nội bộ trong train: 201 cặp pHash $\le 8$ tương ứng chính xác với các biến thể augmentations có chủ đích của 105 base images trong tập train.

---

## 5. KẾT QUẢ XUẤT CONTACT SHEETS & VISUAL QA

Hệ thống đã xuất **13 contact sheets** có vẽ bounding box màu xanh `(0, 220, 0)` kèm nhãn `bat(0)` và metadata độ phân giải, số hộp, phân tách split để người dùng nghiệm thu trực quan:

- **Thư mục lưu trữ:** [`data/acquisition_staging/roboflow_battery/contact_sheets/`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/roboflow_battery/contact_sheets/)
- **Danh mục Contact Sheets:**
  1. `battery_valid_all_01.jpg` (520 KB): 16 ảnh validation đầu tiên.
  2. `battery_valid_all_02.jpg` (530 KB): 16 ảnh validation tiếp theo.
  3. `battery_valid_all_03.jpg` (145 KB): 3 ảnh validation còn lại (Bao phủ đủ 100% 35 ảnh validation gốc).
  4. `battery_train_sample_01.jpg` (680 KB): Mẫu 16 ảnh train tiêu biểu.
  5. `battery_train_sample_02.jpg` (650 KB): Mẫu 16 ảnh train tiêu biểu tiếp theo.
  6. `battery_multi_battery_01.jpg` đến `08.jpg` (4.1 MB): Toàn bộ **114 ảnh chứa từ 2 pin trở lên** (kiểm chứng khả năng phát hiện đa vật thể).
- **Mẫu Overlays phân giải gốc (Full-res Overlays):** 6 ảnh mẫu tại [`data/acquisition_staging/roboflow_battery/overlays/`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/roboflow_battery/overlays/).

### Phân loại trạng thái QA tự động (QA Decisions):
- `QA_PASS_SINGLE_OBJECT`: **226 ảnh** (Chứa đúng 1 pin, bọc khít, nhãn sạch).
- `QA_PASS_MULTI_OBJECT`: **124 ảnh** (Chứa từ 2 đến nhiều pin, ranh giới rõ ràng).
- `APPROVED_FOR_TRAIN`: **0 ảnh** (Tuyệt đối không tự động duyệt, giữ đúng cờ `approved_for_train = False` và `user_train_approval = "PENDING_PM_APPROVAL"`).

---

## 6. ĐỐI CHIẾU SỐ LƯỢNG THIẾU & ĐỀ XUẤT DATASET PILOT V3

### A. Đối chiếu bù đắp thâm hụt (Deficit Fulfillment)
Theo tính toán từ Task 5A/5B, dự án thiếu chính xác **29 boxes train** và **5 boxes val** cho lớp pin để đủ điều kiện chạy pilot kỹ thuật:

| Tiêu chí | Ngưỡng Khởi Đầu Cần Đạt | Baseline v2 Hiện Có | Nguồn Pin Roboflow Đã QA | Sau Khi Bổ Sung (Đề Xuất) | Đánh giá |
|---|:---:|:---:|:---:|:---:|:---:|
| **Box Pin Train (Class 0)** | $\ge 30$ | **1 box** | **+452 boxes** (hoặc +148 base) | **149 – 453 boxes** | 🟢 Vượt ngưỡng yêu cầu |
| **Box Pin Val (Class 0)** | $\ge 5$ | **0 box** | **+53 boxes** (unaugmented) | **53 boxes** | 🟢 Vượt ngưỡng yêu cầu |
| **Rò rỉ với Test/Holdout** | 0 | 0 | **0** | **0** | 🟢 Tuyệt đối an toàn |

### B. Hai phương án cấu hình Dataset Pilot V3
- **Phương án 1 — Pilot Tối Giản (Khuyên dùng - Clean Base Images Only):**
  - Chỉ lấy **105 ảnh base gốc cho Train** (148 boxes pin) và **35 ảnh base gốc cho Val** (53 boxes pin).
  - Ưu điểm: 100% ảnh gốc, không bị trùng lặp pHash do augmentations, tốc độ fine-tuning nhanh, tránh bias phân phối góc chụp.
- **Phương án 2 — Pilot Đầy Đủ (Full Roboflow 3x):**
  - Sử dụng toàn bộ **315 ảnh cho Train** (452 boxes pin) và **35 ảnh cho Val** (53 boxes pin).
  - Ưu điểm: Đa dạng độ sáng và góc xoay đã được augment sẵn.

> **Quy chuẩn bất biến:**
> - Toàn bộ dữ liệu baseline v2 (train, val, test) và checkpoint gốc `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` được **giữ nguyên 100% không chỉnh sửa**.
> - Tập external holdout 27 ảnh (`D:\Hinh_Anh_Rac`) không được dùng trong quá trình training hay tinh chỉnh threshold.

---

## 7. KẾT QUẢ KIỂM THỬ HỆ THỐNG (87/87 TESTS PASS)

```text
tests/qa/test_task5c_battery_qa.py: 5 passed (YAML Subtypes, Manifest Integrity, Contact Sheets, Leakage Audit, Gatekeeper)
tests/qa/test_task5b_hotfixes_and_intake.py: 23 passed (Allowlist, Zip Safety, Full BBox Bounds, Physical Staging, Anti-Bypass)
tests/qa/test_fail_closed_dataset_intake.py: 9 passed (Fail-closed Dataset Builder)
tests/test_candidate_manifest_qa_lifecycle.py: 4 passed (QA State Lifecycle)
tests/test_task4_image_acquisition_and_qa.py: 4 passed (Image Acquisition & Zero Approval DoD)
tests/qa/test_split_contamination_safety.py: 11 passed (Split Contamination & Anti-bypass)
tests/qa/test_phash_leakage_audit.py: 4 passed (pHash Cross-split & Holdout Leakage)
tests/detection/test_real_mapping_integration.py: 14 passed (10-class Taxonomy, SSDLite Shift & Roboflow Source Mapping)
tests/detection/test_mapping_regression.py: 5 passed (Mapping Regression)
tests/detection/test_preflight_guard_integration.py: 8 passed (Preflight Safety Guard)
------------------------------------------------------------------------------------------------------------------------
TỔNG CỘNG: 87 passed in 15.52s (100% PASS, 0 FAILURES)
```

---

## 8. NỘI DUNG GÓI BÀN GIAO `CNTT_KLCN155_Task5C_Delivery.zip`

Gói bàn giao nghiệm thu được lưu trữ tại:
👉 `C:\Users\ad\Downloads\CNTT_KLCN155_Task5C_Delivery.zip`

Bao gồm:
1. `TASK5C_BATTERY_DATA_QA_REPORT.md` (Báo cáo nghiệm thu đầy đủ).
2. `roboflow_battery_staging_manifest.json` (Manifest chi tiết 350 ảnh, 505 boxes kèm quyết định QA).
3. Thư mục `contact_sheets/` (13 ảnh contact sheet kiểm tra trực quan).
4. Thư mục `overlays/` (6 ảnh full-res overlay tiêu biểu).
5. Mã nguồn script kiểm toán: `scripts/import_and_qa_roboflow_battery.py`.
6. Cấu hình mapping nguồn: `configs/detection_source_mapping.yaml`.
7. Bộ kiểm thử tự động: `tests/qa/test_task5c_battery_qa.py`.

---

## 9. TRẠNG THÁI HIỆN TẠI & BƯỚC TIẾP THEO

- **Trạng thái:** Dữ liệu pin thật đã được tiếp nhận, kiểm toán sạch sẽ, xác minh 0 rò rỉ và xuất đầy đủ bằng chứng thị giác.
- **Hành động tiếp theo:** Anti **DỪNG** tại đây theo đúng chỉ thị giao việc. Chờ PM kiểm tra các contact sheet trong gói bàn giao và ra quyết định phê duyệt (số lượng ảnh được train) cho bước Fine-tuning.
