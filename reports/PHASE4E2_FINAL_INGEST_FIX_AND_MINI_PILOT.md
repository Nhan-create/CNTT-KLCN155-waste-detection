# BÁO CÁO NGHIỆM THU: PHASE 4E.2 — FINAL INGEST FIX & REAL MINI-PILOT GATING

**Dự án:** Hệ thống nhận diện rác thải sinh hoạt KLCN155  
**Thư mục làm việc:** `.`  
**Thời điểm thực thi:** 2026-10-12  
**Trạng thái phê duyệt huấn luyện:** 0 ảnh mới được duyệt vào train (`APPROVED_FOR_TRAIN = 0`)  
**Trạng thái tải ảnh thực tế:** 0 ảnh mới được tải (`REAL NEW IMAGES DOWNLOADED = 0`)  

---

## 1. KHẮC PHỤC CÁC LỖI TÁI HIỆN BỞI REVIEWER (P0 INGEST & LEAKAGE)

File: [`scripts/ingest_roboflow_battery_export.py`](../scripts/ingest_roboflow_battery_export.py)

### 1.1 Kiểm tra chính xác 5 tokens YOLO
- Trước đây hàm chỉ kiểm tra `len(parts) < 5`, dẫn đến trường hợp dòng có hơn 5 tokens (ví dụ 6 tokens) vẫn lọt qua.
- **Bản sửa đổi:** Kiểm tra chính xác `len(parts) != 5`. Bất kỳ dòng nào có ít hơn hoặc nhiều hơn 5 tokens đều bị bác bỏ ngay lập tức với mã lỗi `EXACTLY_5_TOKENS_REQUIRED`.

### 1.2 Kiểm tra Class ID hợp lệ thuộc `data.yaml`
- Trước đây Class ID 999 hoặc -1 không bị chặn.
- **Bản sửa đổi:**
  * Bắt buộc `class_id >= 0` (bác bỏ số âm với lỗi `NEGATIVE_CLASS_ID`).
  * Bắt buộc `class_id` là số nguyên nằm trong danh sách các ID hợp lệ được định nghĩa từ `data.yaml` (`valid_class_ids = set(range(nc))`). Class ID 999 bị bác bỏ với lỗi `CLASS_ID_NOT_IN_DATA_YAML`.

### 1.3 Yêu cầu Bounding Box thực sự nằm trong ảnh (Không vượt biên)
- Trước đây hàm cho phép dung sai biên lên tới $[-0.05, 1.05]$.
- **Bản sửa đổi:** Tọa độ hộp bao phải nằm trọn vẹn trong khung hình chuẩn hóa:
  $$x_1 = xc - w/2 \ge 0.0, \quad x_2 = xc + w/2 \le 1.0$$
  $$y_1 = yc - h/2 \ge 0.0, \quad y_2 = yc + h/2 \le 1.0$$
  Nếu bất kỳ cạnh nào tràn ra ngoài biên ảnh quá sai số số học siêu nhỏ ($10^{-6}$), hộp bao bị từ chối với lỗi `BBOX_EXCEEDS_IMAGE_BOUNDS`.

### 1.4 Kiểm tra đẳng thức số lượng lớp: `nc == len(names)`
- Trước đây file `data.yaml` có `nc=1` nhưng `names` chứa 2 lớp vẫn hợp lệ.
- **Bản sửa đổi:** Hàm [`inspect_roboflow_data_yaml()`](../scripts/ingest_roboflow_battery_export.py) nâng ngoại lệ `ValueError("Roboflow data.yaml class count mismatch: nc={nc} != len(names)={len(names)}")` nếu có sự bất nhất giữa `nc` và số lượng tên lớp.

### 1.5 Phân loại và cách ly (Quarantine) các lớp nguồn không xác minh
- Trước đây toàn bộ các lớp trong file nhãn bị chuyển thành `0: battery`.
- **Bản sửa đổi:**
  * Chỉ các lớp nguồn có tên chứa từ khóa `battery` hoặc `batteries` mới được ánh xạ sang lớp `0: battery`.
  * Các đối tượng có Class ID thuộc các lớp khác hoặc không nhận diện được sẽ khiến ảnh bị **CÁCH LY (QUARANTINED)**: `QUARANTINED_CONTAINS_NON_BATTERY_OR_UNRECOGNIZED_CLASS_ID_{id}`, tuyệt đối không tự ý gộp vào lớp Pin.

### 1.6 Xử lý ZIP an toàn (Zip Slip & Zip Bomb Protection)
- Hàm [`safe_extract_untrusted_zip()`](../scripts/ingest_roboflow_battery_export.py):
  * Giới hạn tối đa 2,000 tệp và 300 MiB dung lượng giải nén (ngăn chặn Zip Bomb).
  * Kiểm tra đường dẫn từng tệp giải nén bằng `path.is_relative_to(target_dir)` (ngăn chặn tấn công Zip Slip ghi đè ra ngoài thư mục đích).

### 1.7 Kiểm tra Leakage toàn diện (V2, V3, Holdout Fail-Closed)
- Bắt buộc kiểm tra đồng thời v2, v3 candidate và holdout. Thiếu cấu hình YAML, file split `.txt` hoặc bất kỳ ảnh vật lý nào trên đĩa đều nâng `FileNotFoundError`.
- Holdout đọc toàn bộ 27 ảnh bao gồm cả `.jpg`, `.jpeg`, và `.png` dựa trên [`reports/qa_acceptance/manifest_hinh_anh_rac.json`](../reports/qa_acceptance/manifest_hinh_anh_rac.json).
- Kiểm tra SHA-256 (chống trùng lặp tuyệt đối) và pHash Hamming distance ($\le 8$) với toàn bộ train, val, test và holdout.
- Bổ sung kiểm tra **Near-Duplicate giữa các ảnh bên trong nội bộ tệp xuất Roboflow**.
- Thiết lập bộ đệm băm [`data/audit/authoritative_hashes_cache.json`](../data/audit/authoritative_hashes_cache.json) giúp tốc độ kiểm tra hồi quy giảm từ 82 giây xuống 0.7 giây mà vẫn bảo đảm 100% độ chính xác mật mã.
- Toàn bộ ảnh trích xuất thành công được gán trạng thái `IMPORTED_PENDING_VISUAL_QA` và `PENDING_USER_APPROVAL` (không tự gán `VISUALLY_REVIEWED` hoặc `APPROVED_FOR_TRAIN`).

---

## 2. KẾT QUẢ KIỂM THỬ HỒI QUY (TDD EVIDENCE)

Tệp kiểm thử: [`tests/qa/test_phase4e_intake_safety.py`](../tests/qa/test_phase4e_intake_safety.py)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: .
collected 14 items

tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_valid_cases PASSED [  7%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_token_count_strictly_five PASSED [ 14%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_class_id_validation PASSED [ 21%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_strictly_within_image_bounds PASSED [ 28%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_nc_mismatch PASSED [ 35%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_identifies_battery_classes PASSED [ 42%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_zip_slip_protection PASSED [ 50%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_bomb_protection PASSED [ 57%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_fails_closed_on_missing_files PASSED [ 64%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_verifies_holdout_jpg_jpeg_png PASSED [ 71%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_whitelist_does_not_contain_pending_annotation_comments PASSED [ 78%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_mini_pilot_shortlist_structure PASSED [ 85%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_unapproved_run_fails_closed PASSED [ 92%]
tests/qa/test_phase4e_intake_safety.py::test_quarantine_non_battery_source_class PASSED [100%]

============================= 14 passed in 0.71s ==============================
```
*Toàn bộ hệ thống test suite: **252/252 tests PASSED**.*

---

## 3. DANH SÁCH RÚT GỌN MINI-PILOT WIKIMEDIA (4 ẢNH XÁC MINH NGUỒN)

*(Ứng viên `Street_cabinet.jpg` đã được loại bỏ hoàn toàn khỏi danh sách rút gọn).*

| # | Mã ứng viên | Tên tệp Wikimedia gốc | Phân loại & Bối cảnh kiểm thử | URL tải về trực tiếp đã xác minh | Giấy phép | Tác giả | Dung lượng dự kiến |
|---|---|---|---|---|---|---|---|
| **1** | `wiki_bio_banana_peel_ground` | `File:Banana_peel_on_the_ground.jpg` | **Biological**<br>Rác vỏ chuối hữu cơ vứt trên nền đất/sàn | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/c/cf/Banana_peel_on_the_ground.jpg) | CC BY-SA 4.0 | Vladiator360 | 4,573,954 B (~4.36 MB) |
| **2** | `wiki_bio_street_food_remains` | `File:Street_Food_Remains_(188107205).jpeg` | **Biological**<br>Rác thức ăn thừa đường phố trên vỉa hè | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/9/9a/Street_Food_Remains_%28188107205%29.jpeg) | CC BY-SA 3.0 | Andrew Currie | 294,623 B (~287.7 KB) |
| **3** | `wiki_neg_junction_boxes` | `File:Junction_boxes_-_geograph.org.uk_-_580156.jpg` | **Hard-Negative**<br>Hộp kỹ thuật / tủ điện gắn tường góc phố sạch | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/9/9a/Junction_boxes_-_geograph.org.uk_-_580156.jpg) | CC BY-SA 2.0 | Thomas Nugent | 110,051 B (~107.5 KB) |
| **4** | `wiki_neg_utility_box_leeds` | `File:Utility_box,_Wesley_Street,_Beeston,_Leeds_(11th_August_2023)_001.jpg` | **Hard-Negative**<br>Tủ điện màu xanh lá sát tường gạch vỉa hè sạch | [Wikimedia CDN](https://upload.wikimedia.org/wikipedia/commons/1/13/Utility_box%2C_Wesley_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29_001.jpg) | CC BY-SA 4.0 | Mtaylor848 | 5,736,159 B (~5.47 MB) |

- **Tổng dung lượng 4 ảnh:** 10,714,787 bytes (~10.22 MiB), tuân thủ trần giới hạn tối đa 20 MiB.
- **Quy trình khi được phê duyệt:**
  1. Tải ảnh có giới hạn chunk buffer qua script [`scripts/stage_phase4e_whitelist_candidates.py`](../scripts/stage_phase4e_whitelist_candidates.py).
  2. Tính toán mã băm SHA-256 và pHash, kiểm tra toàn vẹn định dạng bằng Pillow.
  3. Tự động kết xuất ảnh Contact Sheet tổng hợp 2x2 lưu tại [`data/acquisition_staging/phase4e_staging/contact_sheet/mini_pilot_contact_sheet.jpg`](../data/acquisition_staging/phase4e_staging/contact_sheet/).
  4. Visual QA: Đề xuất bounding box riêng biệt cho `biological` và kiểm tra 100% khung hình đối với ảnh tủ điện âm bản.
  5. Đối chiếu leakage chống trùng lặp trước khi đệ trình người dùng xem xét đưa vào huấn luyện.

---

## 4. KHÓA AN TOÀN TRƯỚC KHI TẢI (PRE-DOWNLOAD GATE)

Hệ thống đang tuân thủ nghiêm ngặt nguyên tắc fail-closed: **CHƯA TẢI BẤT KỲ ẢNH NÀO VỀ ĐĨA**.

Khi người dùng gửi lệnh phê duyệt rõ ràng:
```text
[DUYET_TAI_MINI_PILOT]
```
Hệ thống sẽ kích hoạt tải 4 ảnh, kết xuất contact sheet và lập bảng đánh giá Visual QA chi tiết.

---

## 5. BỘ CHỈ SỐ NGHIỆM THU CUỐI CÙNG (FINAL METRICS)

```text
REAL NEW IMAGES DOWNLOADED = 0
VISUALLY QA PASSED         = 0
APPROVED FOR TRAIN         = 0
```
