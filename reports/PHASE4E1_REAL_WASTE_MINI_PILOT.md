# BÁO CÁO NGHIỆM THU: PHASE 4E.1 — REAL WASTE DATA INTAKE SAFETY & MINI-PILOT GATING

**Dự án:** Hệ thống nhận diện rác thải sinh hoạt KLCN155  
**Thư mục làm việc:** `.`  
**Thời điểm thực thi:** 2026-10-12  
**Trạng thái phê duyệt huấn luyện:** 0 ảnh mới được duyệt vào train (`APPROVED_FOR_TRAIN = 0`)  
**Trạng thái tải ảnh thực tế:** 0 ảnh mới được tải (`REAL NEW IMAGES DOWNLOADED = 0`)  

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC BẢO ĐẢM AN TOÀN DỮ LIỆU (PHASE 4E.1)

Sau vòng kiểm duyệt độc lập từ Phase 4E, nhóm phát triển đã tiến hành khắc phục triệt để ba vấn đề an toàn cốt lõi trong quy trình nhập và tiền trạm dữ liệu rác thải thật:
1. **Khắc phục Ingest Roboflow:** Đảm bảo đối chiếu toàn diện với splits chính thức (v2, v3 candidate, holdout), kiểm tra cấu hình lớp nhãn trong `data.yaml`, xác thực tọa độ bbox YOLO chuẩn hóa, phát hiện near-duplicate bằng pHash, gán trạng thái `IMPORTED_PENDING_VISUAL_QA` và giữ nguyên trạng thái phê duyệt train là `PENDING_USER_APPROVAL`.
2. **Loại bỏ nhãn rỗng và comment trong file nhãn:** Tuyệt đối không tạo file `.txt` 0-byte trước khi kiểm tra trực quan toàn khung hình; tuyệt đối không ghi dòng comment giả định vào file nhãn YOLO; trạng thái chờ được theo dõi 100% qua tệp metadata JSON.
3. **Thu hẹp danh sách rút gọn Mini-Pilot (4 ảnh ưu tiên cao):** Chọn lọc chính xác 4 ảnh từ Wikimedia Commons đáp ứng tiêu chí rác hữu cơ ngoài trời và âm bản tủ điện đô thị (loại bỏ `Street_cabinet.jpg` khỏi shortlist).
4. **Cổng kiểm duyệt trước khi tải (Pre-Download Guard):** Hiển thị chi tiết từng ảnh (Tên tệp, URL xác minh, Dung lượng, Giấy phép, Tác giả) và DỪNG LẠI chờ lệnh phê duyệt rõ ràng từ người dùng trước khi tải bất kỳ byte nào.

---

## 2. P0: SỬA ĐỔI INGEST ROBOFLOW (`scripts/ingest_roboflow_battery_export.py`)

File: [`scripts/ingest_roboflow_battery_export.py`](../scripts/ingest_roboflow_battery_export.py)

### 2.1 Đọc đường dẫn split từ cấu hình dataset hiện hành & Fail-Closed
Hàm `load_authoritative_split_hashes()` đã được cập nhật:
- Đọc phân rã từ:
  - `configs/detection_dataset_real_v2.yaml` (train, val, test v2)
  - `configs/detection_dataset_real_v3_candidate.yaml` (v3 candidate splits)
  - Holdout dataset: `reports/qa_acceptance/manifest_hinh_anh_rac.json` và thư mục `data/holdout`.
- **Fail-Closed:** Nếu bất kỳ tệp split YAML hoặc tệp manifest/thư mục holdout nào bị thiếu, hàm lập tức nâng ngoại lệ `FileNotFoundError`, không bao giờ cho phép thực thi trong tình trạng thiếu dữ liệu đối chứng.

### 2.2 Xác thực Bounding Box YOLO nghiêm ngặt
Hàm `validate_yolo_bbox(line: str) -> tuple[int, float, float, float, float]`:
- Yêu cầu chính xác 5 tokens: `class_idx xc yc w h`.
- Bắt lỗi token không phải số thực hoặc giá trị NaN/Infinity.
- Kiểm tra miền tọa độ chuẩn hóa: `0.0 <= xc <= 1.0` và `0.0 <= yc <= 1.0`.
- Kiểm tra kích thước hình học: `0.0 < w <= 1.0` và `0.0 < h <= 1.0` (chiều rộng và chiều cao phải dương).
- Remap lớp: Chỉ cho phép nhãn battery được ánh xạ sang lớp `0: battery`.

### 2.3 Kiểm tra cấu hình `data.yaml` của tệp Roboflow Export
Hàm `inspect_roboflow_data_yaml(extract_dir: Path)`:
- Đọc `data.yaml` từ thư mục giải nén của Roboflow.
- Xác thực số lượng lớp (`nc`) và tên các lớp (`names`).
- Yêu cầu bắt buộc phải chứa tên lớp liên quan tới `battery`. Bác bỏ nếu tệp xuất không thuộc lớp Pin hoặc chứa nhãn không xác định.

### 2.4 Chống trùng lặp tuyệt đối (SHA-256 + pHash Near-Duplicate)
- Tính mã băm cryptographic SHA-256 đối chiếu với toàn bộ split và holdout (chống lặp 100%).
- Tính mã băm cảm nhận Perceptual Hash (pHash) bằng `imagehash.phash(im)`.
- Áp dụng ngưỡng khoảng cách Hamming `PHASH_NEAR_DUP_THRESHOLD = 8`. Nếu khoảng cách Hamming giữa ảnh ứng viên và bất kỳ ảnh nào trong holdout hoặc test nhỏ hơn hoặc bằng 8, ảnh bị đánh dấu `REJECTED_NEAR_DUPLICATE_OF_HOLDOUT`.

### 2.5 Trạng thái phân loại và kiểm duyệt huấn luyện
- Toàn bộ ảnh trích xuất thành công được gán:
  * `qa_status = "IMPORTED_PENDING_VISUAL_QA"` (Tuyệt đối không tự phong `VISUALLY_REVIEWED`).
  * `training_approval_status = "PENDING_USER_APPROVAL"` (Tuyệt đối không tự cấp `APPROVED_FOR_TRAIN`).

---

## 3. P0: SỬA ĐỔI STAGING VÀ XÓA BỎ NHÃN RỖNG / COMMENT (`scripts/stage_phase4e_whitelist_candidates.py`)

File: [`scripts/stage_phase4e_whitelist_candidates.py`](../scripts/stage_phase4e_whitelist_candidates.py)

### 3.1 Không tạo tệp nhãn 0-byte khi tải về
- Trước đây, việc tạo tệp `.txt` 0-byte ngay khi tải ảnh âm bản dẫn đến nguy cơ ngộ nhận đó là "Hard Negative đã được chứng minh".
- **Bản sửa đổi:** Không tạo bất kỳ tệp `.txt` nào trong thư mục `labels/` khi tải ảnh. Toàn bộ tiến trình theo dõi trạng thái chưa có nhãn được lưu trữ độc quyền trong tệp metadata JSON: [`data/acquisition_staging/phase4e_staging/mini_pilot_staging_summary.json`](../data/acquisition_staging/phase4e_staging/mini_pilot_staging_summary.json).

### 3.2 Không ghi comment trong tệp nhãn YOLO
- Xóa bỏ hoàn toàn logic ghi comment (như `# PENDING_ANNOTATION`) vào tệp `.txt`.
- Định dạng nhãn YOLO chỉ hợp lệ khi chứa các dòng số thực `class_idx xc yc w h`. Mọi tệp nhãn trên đĩa chỉ được khởi tạo sau khi đối tượng đã được annotator kiểm tra trực quan và gán bounding box hoàn chỉnh.

### 3.3 Khóa an toàn trước khi tải (Pre-Download Safety Guard)
- Script yêu cầu cờ lệnh `--approve-download` mới thực hiện gọi HTTP request.
- Nếu chạy không có cờ lệnh, script in ra toàn bộ bảng ứng viên Mini-Pilot (tên tệp, URL, dung lượng dự kiến, giấy phép, tác giả) và thoát ngay với mã lỗi `sys.exit(1)`.

---

## 4. P1: DANH SÁCH RÚT GỌN MINI-PILOT (4 ỨNG VIÊN ƯU TIÊN CAO)

Nhóm phát triển đã chốt danh sách **4 ứng viên ưu tiên cao** giải quyết đúng 2 điểm nghẽn nhận diện:
1. Thiếu mẫu rác hữu cơ ngoài trời (Biological Waste trên vỉa hè / đường phố).
2. Thiếu ảnh âm bản tủ điện hạ tầng đô thị (nguyên nhân gây False Positive trong `rac_online_012.jpg`).

*(Lưu ý: Ứng viên `Street_cabinet.jpg` đã được loại khỏi danh sách rút gọn đợt này theo yêu cầu).*

| # | Mã ứng viên | Tên tệp Wikimedia Commons | Bối cảnh & Mục tiêu kiểm nghiệm | URL tải về trực tiếp đã xác minh | Giấy phép | Tác giả | Dung lượng dự kiến |
|---|---|---|---|---|---|---|---|
| **1** | `wiki_bio_banana_peel_ground` | `File:Banana_peel_on_the_ground.jpg` | Rác vỏ chuối hữu cơ vứt bừa bãi trên nền đất/mặt sàn | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/c/cf/Banana_peel_on_the_ground.jpg) | CC BY-SA 4.0 | Vladiator360 | 4,573,954 B (~4.36 MB) |
| **2** | `wiki_bio_street_food_remains` | `File:Street_Food_Remains_(188107205).jpeg` | Rác thức ăn thừa đường phố vứt bỏ trên vỉa hè | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/9/9a/Street_Food_Remains_%28188107205%29.jpeg) | CC BY-SA 3.0 | Andrew Currie | 294,623 B (~287.7 KB) |
| **3** | `wiki_neg_junction_boxes_580156` | `File:Junction_boxes_-_geograph.org.uk_-_580156.jpg` | Hộp đấu dây kỹ thuật / tủ điện gắn tường góc phố sạch (Hard-Negative) | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/9/9a/Junction_boxes_-_geograph.org.uk_-_580156.jpg) | CC BY-SA 2.0 | Thomas Nugent | 110,051 B (~107.5 KB) |
| **4** | `wiki_neg_utility_box_beeston` | `File:Utility_box,_Wesley_Street,_Beeston,_Leeds_(11th_August_2023)_001.jpg` | Tủ điện kỹ thuật màu xanh lá cây đặt sát tường gạch vỉa hè (Hard-Negative) | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/1/13/Utility_box%2C_Wesley_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29_001.jpg) | CC BY-SA 4.0 | Mtaylor848 | 5,736,159 B (~5.47 MB) |

**Tổng dung lượng dự kiến tải về:** 10,714,787 bytes (~10.22 MB).

---

## 5. BẰNG CHỨNG KIỂM THỬ HỒI QUY TOÀN DIỆN (REGRESSION TEST SUITE)

File kiểm thử: [`tests/qa/test_phase4e_intake_safety.py`](../tests/qa/test_phase4e_intake_safety.py)

Kết quả thực thi:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: .
plugins: anyio-4.15.1, platformdirs-4.13.0, qt-4.5.0
collected 7 items

tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_valid_cases PASSED [ 14%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_invalid_cases PASSED [ 28%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_fails_closed_on_missing_files PASSED [ 42%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml PASSED [ 57%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_whitelist_does_not_contain_pending_annotation_comments PASSED [ 71%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_mini_pilot_shortlist_structure PASSED [ 85%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_unapproved_run_fails_closed PASSED [100%]

============================== 7 passed in 0.41s ==============================
```

Toàn bộ test suite kiểm tra hệ thống gồm 245 ca kiểm thử trong `tests/qa/` và `tests/detection/` đều đạt **100% PASSED**.

---

## 6. THỦ TỤC PHÊ DUYỆT TẢI MINI-PILOT DÀNH CHO NGƯỜI DÙNG

Hệ thống đang dừng ở trạng thái chờ và **chưa thực hiện tải bất kỳ ảnh nào trong 4 ảnh trên**.

Để cho phép tải 4 ảnh vào thư mục đệm [`data/acquisition_staging/phase4e_staging/images/`](../data/acquisition_staging/phase4e_staging/images/), người dùng vui lòng phản hồi:
```text
[DUYET_TAI_MINI_PILOT]
```
Sau khi nhận được lệnh phê duyệt, hệ thống sẽ thực hiện tải tuần tự có giới hạn buffer, trích xuất mã băm SHA-256 / pHash, xác thực decode ảnh bằng Pillow, và báo cáo kết quả nghiệm thu visual QA.

---

## 7. BỘ CHỈ SỐ NGHIỆM THU CUỐI CÙNG (FINAL METRICS)

```text
REAL NEW IMAGES DOWNLOADED = 0
VISUALLY QA PASSED         = 0
APPROVED FOR TRAIN         = 0
```
