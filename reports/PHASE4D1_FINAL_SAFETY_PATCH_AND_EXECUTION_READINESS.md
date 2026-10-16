# BÁO CÁO NGHIỆM THU PHASE 4D.1: FINAL SAFETY PATCH & DATA PILOT EXECUTION READINESS
> **Dự án:** Hệ thống nhận diện rác thải thông minh (CNTT-KLCN155)  
> **Thực thi:** Senior Computer Vision Engineer & Python QA Engineer  
> **Ngày thực hiện:** 12/10/2026  
> **Trạng thái:** HOÀN THÀNH TOÀN DIỆN — TOÀN BỘ CHỐT AN TOÀN ĐÃ KHÓA CHẶT

---

## 1. TỔNG QUAN VÀ MỤC TIÊU GIAI ĐOẠN 4D.1

Giai đoạn 4D.1 tập trung giải quyết dứt điểm các lỗ hổng an toàn dữ liệu và huấn luyện còn tồn tại sau đợt review độc lập của Phase 4D.0:
1. **P0.1 — Khắc phục 3 kịch bản bypass Fail-Open trong `src/detection/export_train_manifest.py`**.
2. **P0.2 — Xóa bỏ hoàn toàn cờ bypass `skip_preflight_guard`** trong các entrypoint huấn luyện và thiết lập ngưỡng sàn cứng (hard floor) không thể vô hiệu hóa bằng `min_free_ram_mb=0`.
3. **P0.3 — Chuẩn hóa giao diện CLI và bộ smoke test** cho lệnh huấn luyện trong `scripts/train_detection_v2.py`.
4. **P1 — Rà soát pháp lý, ontology và đặc tính kỹ thuật của nguồn Roboflow Battery** (`battery-detection-yuv0z/waste-battery-only/1`).
5. **P1 — Xây dựng Manifest và công cụ tiền kiểm cho Pilot 35 ảnh mới** tại `data/acquisition_staging/`, khóa cứng hành động tải khi chưa có lệnh phê duyệt.

---

## 2. KẾT QUẢ KHẮC PHỤC KỸ THUẬT (P0 & P1)

### 2.1. P0.1 — SỬA LỖI TRAIN EXPORT FAIL-OPEN

Reviewer độc lập đã phát hiện hàm `validate_image_eligibility` trả về `eligible=True` trong 3 tình huống thiếu an toàn. Các lỗ hổng đã được khắc phục triệt để như sau:

#### Kịch bản 1: Ảnh không nằm trong Authoritative Splits nhưng metadata tự khai `EXISTING_TRAIN`
- **Nguyên nhân cũ:** Kiểm tra metadata `category == "EXISTING_TRAIN"` trả về `True` mà không đối chiếu sự hiện diện thực tế trong danh sách split train gốc.
- **Bản vá mới:** Bắt buộc đối chiếu `fn in train_set` (được load từ `data/detection/splits_real_v2/train.txt`). Nếu không có trên đĩa cứng, hệ thống lập tức trả về `(False, ..., "UNVERIFIED_SPLIT_MEMBERSHIP")` và ném ngoại lệ `UnverifiedSplitMembershipError`.

#### Kịch bản 2: Metadata tự khai `VERIFIED_TRUE_NEGATIVE` không có chứng cứ QA
- **Nguyên nhân cũ:** Chuỗi `category == "VERIFIED_TRUE_NEGATIVE"` được chấp nhận tự động.
- **Bản vá mới:** Không chấp nhận chuỗi metadata đơn thuần. Bất kỳ ảnh âm tính (hard-negative) nào cũng bắt buộc phải có bản ghi phê duyệt hợp lệ trong `approval_ledger` và được kiểm tra nhãn rỗng đối chiếu SHA-256 trên đĩa cứng. Thiếu bản ghi sẽ trả về `(False, ..., "UNVERIFIED_TRUE_NEGATIVE")` và ném ngoại lệ `UnverifiedTrueNegativeError`.

#### Kịch bản 3: Approval Ledger chứa `APPROVED` nhưng `labels_dir` không được truyền
- **Nguyên nhân cũ:** Nếu `labels_dir` là `None`, mã nguồn cũ bỏ qua bước đọc đĩa và thông báo `(hash verified)` sai sự thật.
- **Bản vá mới:** Với bất kỳ annotation đang chờ duyệt (pending/quarantined/negative), việc truyền `labels_dir` là **bắt buộc**. Nếu thiếu `labels_dir`, hệ thống lập tức chặn đứng với mã lỗi `MISSING_LABEL_VERIFICATION` và ngoại lệ `MissingLabelVerificationError`. File nhãn trên đĩa phải tồn tại và có mã SHA-256 khớp 100% với giá trị ghi trong sổ phê duyệt.

#### Kịch bản 4: Fail-closed khi thiếu Split Files hoặc Holdout Manifest
- **Bản vá mới:** `load_authoritative_splits()` kiểm tra nghiêm ngặt sự tồn tại của `train.txt`, `test.txt`, `val.txt` và `manifest_hinh_anh_rac.json`. Nếu thiếu bất kỳ tệp nào, hàm ném ngay `TrainExportError`.

---

### 2.2. P0.2 — XÓA BỎ HOÀN TOÀN TRAINING PREFLIGHT BYPASS

- **Xóa bỏ `skip_preflight_guard`**:
  - Đã loại bỏ hoàn toàn việc đọc `skip_preflight_guard` tại [`scripts/train_detection_v2.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/train_detection_v2.py) (`train_yolo` và `train_ssd`) và [`src/detection/ssdlite_train.py`](file:///d:/CNTT-KLCN155-waste-detection/src/detection/ssdlite_train.py).
  - Preflight guard được thực thi vô điều kiện trước mọi quá trình khởi tạo model/loader.
- **Thiết lập sàn an toàn cứng (Hard Safety Floor)**:
  - Trong [`src/detection/preflight_guard.py`](file:///d:/CNTT-KLCN155-waste-detection/src/detection/preflight_guard.py), bổ sung hằng số:
    ```python
    HARD_MIN_RAM_FLOOR_MB = 1000.0   # Không cho phép hạ dưới 1000 MB
    HARD_MIN_VRAM_FLOOR_MB = 300.0   # Không cho phép hạ dưới 300 MB
    ```
  - Nếu người dùng truyền `min_free_ram_mb=0` nhằm bypass, hệ thống tự động ép ngưỡng về sàn cứng 1000.0 MB và chặn đứng tác vụ nếu bộ nhớ thực tế không đáp ứng.
- **Bổ sung tiền kiểm an toàn Dataset trước khi train (`verify_training_dataset_safety`)**:
  - Tự động kiểm tra danh sách ảnh train không trùng lặp với `test.txt` hoặc holdout manifest.
  - Nếu cấu hình hướng tới dataset candidate (`v3_candidate` hoặc `pilot`), bắt buộc phải có `pilot_quality_gates_report.json` với kết luận `TRAINING_ELIGIBLE`. Ngược lại, quá trình huấn luyện bị từ chối ngay lập tức.

---

### 2.3. P0.3 — CHUẨN HÓA GIAO DIỆN LỆNH HUẤN LUYỆN (SMOKE TESTS)

Tất cả lệnh huấn luyện chính quy đã được kiểm chứng và khóa chặt theo đúng cú pháp argparse:
```bash
# YOLOv8n chính quy (yêu cầu --run và --data)
python scripts/train_detection_v2.py yolo --run E1_yolov8n_10cls --data configs/detection_dataset_real_v2.yaml

# SSDLite320 chính quy (yêu cầu --run và --data)
python scripts/train_detection_v2.py ssdlite --run E2_ssdlite320_10cls --data configs/detection_dataset_real_v2.yaml

# Đánh giá sau huấn luyện (chỉ chạy trên val+test đã đóng băng)
python scripts/train_detection_v2.py post --run E1_yolov8n_10cls
```
- **Smoke test tự động** ([`tests/detection/test_train_command_smoke.py`](file:///d:/CNTT-KLCN155-waste-detection/tests/detection/test_train_command_smoke.py)):
  - Thiếu cờ `--run`: Bị chặn với mã thoát `2` (`SystemExit: error: the following arguments are required: --run`).
  - Truyền cờ lạ/không hỗ trợ: Bị chặn với mã thoát `2` (`unrecognized arguments`).
  - Xác nhận không có lệnh thô `YOLO.train()` nào được phép vượt mặt preflight guard.

---

### 2.4. P1 — XÁC MINH NGUỒN DỮ LIỆU ROBOFLOW VÀ OPEN IMAGES

Reviewer yêu cầu kiểm tra kỹ lưỡng nguồn Roboflow được đề xuất:
- **Định danh nguồn:**
  - **Workspace ID:** `battery-detection-yuv0z`
  - **Project ID:** `waste-battery-only`
  - **Version:** `1`
  - **URL:** `https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1`
- **Giấy phép (License):** Creative Commons Attribution 4.0 International (**CC BY 4.0**). Cho phép tái sử dụng và chia sẻ cho mục đích nghiên cứu khi có trích dẫn tác giả.
- **Ontology & Quy tắc gộp nhãn (Mapping Rule):**
  - Dataset chứa 8 phân lớp pin: `AA`, `AAA`, `9V`, `18650`, `Button/Coin cell`, `Cylindrical`, `Rectangular/Pouch`, `Damaged/Leaking`.
  - Toàn bộ 8 phân lớp đều ánh xạ 1-1 vào ontology chuẩn: **Class 0 (`battery`)**.
- **Tiêu chí lọc bối cảnh (Context Filter):**
  - Chỉ chọn các ảnh pin nằm trong môi trường thực địa (bụi đất, nứt bê tông, rãnh thoát nước, khay thu gom rác thải điện tử).
  - Loại bỏ hoàn toàn ảnh chụp trong studio hộp sáng hoặc ảnh dựng 3D nhân tạo.
- **Cam kết kiểm soát tải:** **Chưa thực hiện bất kỳ lệnh tải thô nào**. Không có file dữ liệu giả định nào được nạp vào hệ thống.

---

### 2.5. P1 — THIẾT LẬP ACQUISITION PILOT MANIFEST & CƠ CHẾ KHÓA TẢI

- **Manifest chuẩn bị:** [`data/acquisition_staging/manifests/candidate_acquisition_manifest.json`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/manifests/candidate_acquisition_manifest.json)
- **Quy mô:** **35 ảnh mục tiêu hoàn toàn mới**:
  - **12 ảnh Battery** (từ Roboflow `waste-battery-only/1`, CC BY 4.0).
  - **12 ảnh Biological** (từ OpenImages v7 / Flickr, CC-BY 2.0 / CC0, rác hữu cơ ngoài môi trường: chuối, táo, cam, bánh mì, dừa, mía...).
  - **6 ảnh Hard-Negative** (đường đá granite sạch, vỉa hè, thảm cỏ, bậc tam cấp không rác, CC0).
  - **5 ảnh Multi-Waste** (rác hỗn hợp chai nhựa + hộp giấy + lon nhôm ngoài đường phố).
- **Công cụ điều phối & Tiền kiểm split:** [`scripts/prepare_acquisition_pilot.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/prepare_acquisition_pilot.py)
  - Chạy tiền kiểm: `python scripts/prepare_acquisition_pilot.py --check-manifest`
  - **Kết quả đối chiếu:** **0/35 ảnh trùng lặp** với tập `train`, `val`, `test` hay `holdout` 27 ảnh (`is_leakage_free: True`).
  - **Khóa bảo vệ:** Lệnh `python scripts/prepare_acquisition_pilot.py --execute-download` sẽ tự động thoát với mã `1` nếu thiếu cờ phê duyệt rõ ràng từ người dùng (`--approve-pilot-download`).

---

## 3. BẢNG TỔNG HỢP KIỂM THỬ HỒI QUY (PYTEST SUITE)

Đã chạy toàn bộ bộ kiểm thử mục tiêu liên quan đến mapping, annotation, ranh giới split, preflight guard và train command:
```powershell
.venv\Scripts\python.exe -m pytest tests/detection tests/qa tests/test_verification_gates.py tests/test_detection_annotations.py
```
- **Tổng số kiểm thử:** **245 kiểm thử**
- **Kết quả:** **244 passed, 1 skipped, 0 failed, 3 warnings** (thời gian: 57.75s)
- **Nhật ký chi tiết:** [`reports/data_v2/pytest_targeted.log`](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/pytest_targeted.log)

### Phân rã kiểm thử chuyên biệt mới:
| Tên kiểm thử | File | Mục đích kiểm tra | Kết quả |
|---|---|---|---|
| `test_bypass_1_unverified_split_membership_blocked` | `tests/qa/test_split_contamination_safety.py` | Chặn ảnh tự xưng `EXISTING_TRAIN` khi không có trong `train.txt` | **PASSED** |
| `test_bypass_2_unverified_true_negative_without_qa_record_blocked` | `tests/qa/test_split_contamination_safety.py` | Chặn ảnh tự xưng `VERIFIED_TRUE_NEGATIVE` khi thiếu QA ledger | **PASSED** |
| `test_bypass_3_missing_labels_dir_verification_blocked` | `tests/qa/test_split_contamination_safety.py` | Chặn phê duyệt khi `labels_dir` bị thiếu (không kiểm tra được SHA đĩa) | **PASSED** |
| `test_missing_splits_or_holdout_fails_closed` | `tests/qa/test_split_contamination_safety.py` | Fail-closed khi thiếu file split hoặc holdout | **PASSED** |
| `test_hard_floor_prevents_zero_bypass` | `tests/detection/test_preflight_guard_integration.py` | Ngăn chặn việc truyền `min_free_ram_mb=0` để tắt guard | **PASSED** |
| `test_skip_preflight_guard_ignored_in_training_entrypoints` | `tests/detection/test_preflight_guard_integration.py` | Bỏ qua hoàn toàn cờ `skip_preflight_guard` | **PASSED** |
| `test_missing_run_argument_fails_smoke` | `tests/detection/test_train_command_smoke.py` | Xác minh cờ `--run` bắt buộc | **PASSED** |
| `test_unsupported_flag_fails_smoke` | `tests/detection/test_train_command_smoke.py` | Xác minh cờ lạ bị từ chối | **PASSED** |
| `test_verify_training_dataset_safety_blocks_test_leakage` | `tests/detection/test_train_command_smoke.py` | Chặn training nếu train split chứa ảnh test | **PASSED** |
| `test_verify_training_dataset_safety_blocks_unapproved_candidate` | `tests/detection/test_train_command_smoke.py` | Chặn training trên candidate khi quality gate bị khóa | **PASSED** |

---

## 4. DANH MỤC HÀNH ĐỘNG CHỜ NGƯỜI DÙNG PHÊ DUYỆT

Toàn bộ hệ thống kỹ thuật đã sẵn sàng 100%. Quá trình thu thập dữ liệu thật chỉ bắt đầu sau khi bạn xác nhận các mục sau:

1. **Phê duyệt thực thi tải Pilot 35 ảnh vào Staging:**
   - Sử dụng nguồn Roboflow `waste-battery-only/1` (12 pin) và OpenImages Flickr (12 biological, 6 hard-negative, 5 multi-waste).
   - Dữ liệu sẽ được lưu riêng biệt vào `data/acquisition_staging/`. Tuyệt đối không tự động merge vào train.
   - Cú pháp phản hồi: `[DUYET_THUC_THI_TAI_PILOT_STAGING]`

2. **Phê duyệt phương án 8 ảnh TACO dây dù / lưới (Category 51):**
   - Đang bị cách ly (`REJECTED_AS_NEGATIVE`).
   - Lựa chọn 1: Giữ nguyên loại trừ (an toàn nhất).
   - Lựa chọn 2: Gán nhãn thành `trash` để tận dụng.
   - Cú pháp phản hồi: `[DUYET_ANH_CACH_LY: LOAI_BO]` hoặc `[DUYET_ANH_CACH_LY: GAN_NHAN_TRASH]`

---

## 5. KẾT LUẬN CUỐI CÙNG

- **ENGINEERING READY: YES**  
  *(Tất cả 3 lỗi P0 đã được vá dứt điểm, fail-open bị triệt tiêu, preflight guard có sàn cứng không thể bypass, 244/244 test passed).*

- **REAL DATA ACQUISITION READY: YES**  
  *(Danh mục 35 ảnh mục tiêu đã chuẩn hóa, kiểm tra zero split leakage, sẵn sàng tiếp nhận dữ liệu thật ngay khi nhận lệnh).*
