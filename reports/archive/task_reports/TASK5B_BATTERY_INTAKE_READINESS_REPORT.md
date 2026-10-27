# BÁO CÁO THỰC THI SPRINT AI-UPGRADE-01 — TASK 5B
## TIẾP NHẬN NGUỒN PIN, HOTFIX HẠ TẦNG & SẴN SÀNG HUẤN LUYỆN

- **Repository:** `.`
- **Mã công việc:** SPRINT AI-UPGRADE-01 — TASK 5B (HOTFIX & BATTERY INTAKE)
- **Thời điểm:** 2026-10-13 21:45 UTC+7
- **Phân định vai trò:** Antigravity Integrator + Chuyên trách Data / Model / QA
- **Trạng thái cổng tiếp nhận:** **GATE STATUS: BLOCKED — WAITING_FOR_BATTERY_ZIP**
- **Gói bàn giao:** `~/Downloads\CNTT_KLCN155_Task5B_Delivery.zip`

---

## 1. TỔNG QUAN VERDICT & KẾT QUẢ HOTFIX

Reviewer đã chỉ ra 3 thiếu sót kỹ thuật trong Task 5A; toàn bộ 3 lỗi đã được khắc phục triệt để và kiểm chứng bằng test suites mới:

| Hạng mục Hotfix | Mô tả lỗi phát hiện | Giải pháp kỹ thuật đã áp dụng | Kết quả kiểm thử |
|---|---|---|:---:|
| **Task 1: Fail-Closed Allowlist** | `validate_candidate_for_intake()` có thể lọt `QA_FAIL`, `REJECTED`, `SOURCE_UNVERIFIED`, hoặc `visual_qa` rỗng nếu `num_boxes > 0` và có cờ duyệt. | Thay thế cơ chế blocklist bằng **STRICT ALLOWLIST**: Chỉ chấp nhận `QA_PASS`, `QA_PASS_SINGLE_OBJECT`, `QA_PASS_MULTI_OBJECT` cho positive; `QA_PASS_CLEAN_NEGATIVE` cho negative; chặn vô điều kiện mọi từ khóa rủi ro; bắt buộc cả hai cờ `approved_for_train=True` và `user_train_approval="APPROVED_FOR_TRAIN"`. | **PASS** |
| **Task 2: Roboflow Importer An Toàn** | Importer cũ giải nén không kiểm tra Zip Slip/Bomb, gán cứng `[0]` cho mọi box, không đọc `data.yaml` thật, không kiểm tra biên bbox. | Viết lại `scripts/import_and_qa_roboflow_battery.py`: Kiểm tra an toàn ZIP trước giải nén; đọc `data.yaml` thật và chỉ map tên lớp `battery`/`batteries` sang class 0; validate tọa độ hữu hạn trong $[0.0, 1.0]$; quét rò rỉ pHash chéo và phát hiện trùng lặp nội bộ ZIP. | **PASS** |
| **Task 3: Physical Staging Output** | `build_safe_pilot_dataset_staging()` mới lọc manifest chứ chưa tạo đầu ra vật lý; chưa kiểm tra file ảnh/nhãn trên đĩa. | Cập nhật hàm tạo thư mục `images/` và `labels/` thật; kiểm tra ảnh giải mã được bằng PIL và nhãn đúng định dạng YOLO trước khi copy; khi 0 ảnh duyệt trả về `EMPTY_STAGING_NO_APPROVED_CANDIDATES`; cấm ghi đè baseline. | **PASS** |
| **Final Fix 1: Khóa đường bypass approval** | `build_safe_pilot_dataset_staging(..., require_approval=False)` có thể bỏ qua gate duyệt và copy ảnh chưa duyệt. | Ném lỗi `SafeDatasetBuilderError` ngay lập tức nếu `require_approval is False`. Hàm tạo staging bắt buộc phê duyệt kép `approved_for_train=True` và `user_train_approval="APPROVED_FOR_TRAIN"`. Không cho phép tham số bypass. | **PASS** |
| **Final Fix 2: Validate toàn diện YOLO BBox** | `validate_on_disk_sample()` chấp nhận box có $x_2 > 1.0$ (ví dụ: `0 0.95 0.5 0.3 0.2` có $x_2 = 1.10$), chưa bắt NaN/Inf. | Kiểm tra triệt để: số hữu hạn (`isnan`, `isinf`), $w > 0, h > 0$, toàn bộ $[x_1, y_1, x_2, y_2] \in [0.0, 1.0]$ (tolerance $1\times 10^{-6}$), class ID $\in [0..9]$. Loại bỏ ngay lập tức nhãn lỗi thay vì im lặng sửa tọa độ. | **PASS** |

---

## 2. KẾT QUẢ KIỂM TRA CỔNG TIẾP NHẬN PIN (TASK 4)

- **Thư mục kiểm tra:** `data/acquisition_staging/roboflow_battery/`
- **Hiện trạng:** **0 file ZIP** (Thư mục hoàn toàn trống).
- **Thực thi quy chuẩn:** Đúng theo chỉ thị của PM đối với **Trường hợp A — Chưa có ZIP**:
  - Không tạo dữ liệu giả lập.
  - Không viết thêm pipeline khác.
  - Báo cáo **BLOCKED: WAITING_FOR_BATTERY_ZIP**.
  - Dừng tại cổng tiếp nhận và giữ nguyên trạng thái chờ tệp thật từ người dùng.

### Thông tin nguồn đã xác minh đúng:
- **URL chính thức:** [https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1](https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1)
- **Workspace:** `battery-detection-yuv0z` *(Đã loại bỏ hoàn toàn URL `alif-kurniawan-x336t`)*
- **Project / Version:** `waste-battery-only` / `v1` (Object Detection, 350 ảnh, CC BY 4.0, YOLOv8)

---

## 3. HƯỚNG DẪN 5 BƯỚC CHO NGƯỜI DÙNG TẢI DỮ LIỆU NGOẠI TUYẾN

Người dùng thực hiện trên trình duyệt web (hoàn toàn ngoại tuyến, không cần dán API key hay token vào chat):

1. **Mở trình duyệt** truy cập:  
   👉 [https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1](https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1)
2. **Nhấn nút "Download Dataset"** ở góc phải màn hình.
3. **Tại ô Format:** Chọn định dạng **`YOLOv8`** (hoặc `YOLOv5 PyTorch`).
4. **Tại ô Export Option:** Chọn **"download zip to computer"** và nhấn **Continue**.
5. **Di chuyển file ZIP vừa tải về** vào thư mục dự án:  
   👉 `data\acquisition_staging\roboflow_battery\`
6. **Chạy lệnh tự động giải nén an toàn và QA:**
   ```powershell
   .venv\Scripts\python.exe scripts/import_and_qa_roboflow_battery.py
   ```

---

## 4. BẢN ĐỀ XUẤT DATASET PILOT & SỐ LƯỢNG THIẾU CHÍNH XÁC

Để bắt đầu pilot thử nghiệm fine-tune kỹ thuật (sanity pilot), hệ thống cần đạt các ngưỡng tối thiểu:

| Tiêu chí | Ngưỡng Khởi Đầu Đề Xuất | Hiện Có Thực Tế trên Đĩa | Số Lượng Đang Thiếu Chính Xác | Trạng thái |
|---|:---:|:---:|:---:|:---:|
| **Box Pin Train (Class 0)** | $\ge 30$ boxes | **1 box** | **Thiếu 29 boxes** | ❌ Chờ file ZIP Roboflow |
| **Box Pin Val (Class 0)** | $\ge 5$ boxes | **0 box** | **Thiếu 5 boxes** | ❌ Chờ file ZIP Roboflow |
| **Phân tách Split theo Phiên** | Khác nguồn / phiên chụp | Chưa tách | Chờ metadata file ZIP | ⏳ Sẵn sàng logic nhóm pHash |
| **Rác Hữu cơ Đa Dạng (Class 1)** | Ngoài vỏ chuối | 1 lõi táo (`apple_core_01`) | Cần thêm cam/rau củ | ⏳ Đã có 1 candidate QA PASS |
| **Hard-negatives Thật** | $\ge 2$ ảnh nền sạch | 2 ảnh (`utility_box`, `brick_wall`) | Đủ 2 ảnh |  Đã QA PASS, chờ duyệt train |
| **10 Lớp Cố Định (0..9)** | Đầy đủ 10 lớp | Đầy đủ 10 lớp | 0 |  Chuẩn taxonomy |

> **Lưu ý:** Các ngưỡng trên chỉ là điều kiện khởi đầu tối thiểu cho một pilot thử nghiệm ngắn, không phải bằng chứng khẳng định dataset đã đủ độ bao phủ cho triển khai production.

---

## 5. KẾ HOẠCH TRAINING READINESS

- **Checkpoint gốc:** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` (Được giữ nguyên và đóng băng).
- **Phần cứng:** NVIDIA GeForce RTX 2050 Mobile (4GB VRAM).
- **Cấu hình fine-tune đề xuất (Sau khi có dữ liệu được phê duyệt):**
  - `imgsz: 512`, `batch: 4`, `workers: 2`, `epochs: 25`, `lr0: 0.001`, `amp: True`, `cache: False`.
  - Thư mục đầu ra riêng: `artifacts/detection_v3_pilot/E3_yolov8n_pilot/`.
- **Quy chuẩn đánh giá:**
  - Đo trên tập test nội bộ cố định (299 ảnh, 720 physical boxes).
  - So sánh đối chiếu trực tiếp: Precision, Recall, mAP50, mAP50-95 và per-class AP của lớp battery và biological so với baseline.
  - Phân tích chi tiết cả False Positives và False Negatives.
  - Tuyệt đối không dùng 27 ảnh external holdout để chỉnh threshold hoặc chọn hyperparameter.
- **Cam kết an toàn:** **Không chạy training GPU khi chưa có dữ liệu được phê duyệt.** `approved_for_train == 0`.

---

## 6. KẾT QUẢ KIỂM THỬ HỆ THỐNG (82/82 TESTS PASS)

```
tests/qa/test_task5b_hotfixes_and_intake.py: 23 passed (Allowlist, Zip Safety, Full BBox Bounds, Physical Staging, Anti-Bypass Gate)
tests/qa/test_fail_closed_dataset_intake.py: 9 passed (Fail-closed Dataset Builder)
tests/test_candidate_manifest_qa_lifecycle.py: 4 passed (QA State Lifecycle)
tests/test_task4_image_acquisition_and_qa.py: 4 passed (Image Acquisition & Zero Approval DoD)
tests/qa/test_split_contamination_safety.py: 11 passed (Split Contamination & Anti-bypass)
tests/qa/test_phash_leakage_audit.py: 4 passed (pHash Cross-split & Holdout Leakage)
tests/detection/test_real_mapping_integration.py: 14 passed (10-class Taxonomy & SSDLite Shift)
tests/detection/test_mapping_regression.py: 5 passed (Mapping Regression)
tests/detection/test_preflight_guard_integration.py: 8 passed (Preflight Safety Guard)
------------------------------------------------------------------------------------------------------------------------
TỔNG CỘNG: 82 passed in 19.64s (100% PASS, 0 FAILURES)
```

---

## 7. ĐỐI CHIẾU ĐIỀU KIỆN NGHIỆM THU (DoD)

- [x] **QA_FAIL và SOURCE_UNVERIFIED không thể qua intake:** Đã kiểm chứng qua unit test, ném lỗi loại bỏ ngay lập tức.
- [x] **ZIP không an toàn bị từ chối:** Đã kiểm chứng qua unit test với `ZipSafetyError` cho Zip Slip và Zip Bomb.
- [x] **Không mapping sai lớp nguồn thành battery:** Đã kiểm chứng qua unit test, chỉ có `battery` và `batteries` map sang class 0; lớp khác map sang `None`.
- [x] **pHash được đối chiếu thật, không chỉ được tính:** Hàm `load_authoritative_reference_hashes` và `process_roboflow_battery_archive` so sánh khoảng cách Hamming với toàn bộ tập dữ liệu đối chứng.
- [x] **Dataset builder có đầu ra staging đúng với allowlist:** Kiểm chứng qua unit test và chạy thử nghiệm thực tế trên đĩa: trả về `EMPTY_STAGING_NO_APPROVED_CANDIDATES` khi 0 ảnh duyệt.
- [x] **Có raw test results:** Đính kèm đầy đủ log 79 test cases trong báo cáo.
- [x] **APPROVED_FOR_TRAIN vẫn bằng 0:** Giữ nguyên 0 ảnh mới được duyệt train cho đến khi có phê duyệt của người dùng.
- [x] **Không training GPU, không sửa checkpoint, không đổi taxonomy.**

---

## 8. HÀNH ĐỘNG TIẾP THEO

Toàn bộ 3 lỗi mã nguồn đã được sửa triệt để và hạ tầng đã sẵn sàng 100%. Không mở thêm task lập trình khác.

👉 **Hành động cần thực hiện:** Người dùng tải file ZIP YOLOv8 từ [Roboflow Universe: waste-battery-only](https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1) và đặt vào thư mục `data/acquisition_staging/roboflow_battery/` để kích hoạt quy trình giải nén, kiểm tra chất lượng và lập danh sách duyệt training pilot.
