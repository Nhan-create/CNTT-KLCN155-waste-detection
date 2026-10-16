# BÁO CÁO NGHIỆM THU: PHASE 4E.3 — FINAL DATA INTAKE PATCH & WIKIMEDIA MINI-PILOT GATING

**Dự án:** Hệ thống nhận diện rác thải sinh hoạt KLCN155  
**Thư mục làm việc:** `.`  
**Thời điểm thực thi:** 2026-10-12  
**Trạng thái phê duyệt huấn luyện:** 0 ảnh mới được duyệt vào train (`APPROVED_FOR_TRAIN = 0`)  
**Trạng thái tải ảnh thực tế:** 0 ảnh mới được tải (`REAL NEW IMAGES DOWNLOADED = 0`)  

---

## 1. TASK A: BẢN VÁ PHÂN LOẠI LỚP PIN & BỘ ĐỆM BĂM (BATTERY TAXONOMY & CACHE)

Tệp sửa đổi: [`scripts/ingest_roboflow_battery_export.py`](../scripts/ingest_roboflow_battery_export.py)

### 1.1 Khắc phục nhận diện loại pin không chứa từ khóa "battery"
- **Không tự nhận định cố định:** Hệ thống không mặc định áp đặt 8 lớp cụ thể nào trong Roboflow v1, mà đọc động cấu hình `data.yaml` từ tệp xuất thực tế.
- **Tập từ khóa pin đã kiểm nghiệm (`VETTED_BATTERY_TOKENS`):**
  * Các dạng hình học: `aa`, `aaa`, `c`, `d`, `9v`, `18650`, `21700`, `26650`, `button`, `coin`, `cr2032`, `cylindrical`, `rectangular`, `pouch`, `prismatic`.
  * Hóa tính pin: `li_ion`, `lithium`, `nimh`, `nicd`, `alkaline`, `lead_acid`, `dry_cell`, `accumulator`.
  * Từ khóa chuẩn: `battery`, `batteries`, `waste_battery`, `car_battery`.

### 1.2 Loại trừ phụ kiện và thiết bị sạc (`ACCESSORY_NEGATIVE_TOKENS`)
- Bổ sung bộ lọc phủ định nghiêm ngặt:
  `{"charger", "case", "holder", "cable", "adapter", "accessory", "accessories", "dock", "station", "compartment", "cover", "plug", "wire"}`
- Các lớp như `battery charger`, `battery case`, `battery holder` bị gán nhãn `QUARANTINE_ACCESSORY` và loại trừ tuyệt đối, không bao giờ được coi là pin.

### 1.3 Bảng ánh xạ phân loại động (Source Class ID → Target Class)
Hàm [`inspect_roboflow_data_yaml()`](../scripts/ingest_roboflow_battery_export.py) xây dựng bảng ánh xạ minh bạch và in ra log:
- Lớp pin hợp lệ $\longrightarrow$ `0: battery` (`PROPOSED_ACCEPTED`).
- Lớp phụ kiện sạc/vỏ $\longrightarrow$ `QUARANTINE_ACCESSORY` (`EXCLUDED_ACCESSORY`).
- Lớp mơ hồ hoặc chưa xác minh (ví dụ `Damaged`, `box`) $\longrightarrow$ `QUARANTINE_UNVERIFIED` (`REQUIRES_APPROVAL`).
- Chỉ những ảnh chứa $100\%$ đối tượng thuộc lớp pin đã duyệt mới được chấp nhận; ảnh chứa lớp chưa xác minh sẽ bị **cách ly**.

### 1.4 Khắc phục hoàn toàn xung đột khóa Cache SHA/pHash
- Thay thế khóa đệm cũ `p.name` bằng đường dẫn tuyệt đối đã chuẩn hóa:
  ```python
  resolved = str(p.resolve()).replace("\\", "/")
  ckey = f"{resolved}_{st.st_size}_{st.st_mtime_ns}"
  ```
- Loại bỏ hoàn toàn nguy cơ trùng khóa giữa hai tệp cùng tên nằm ở các thư mục khác nhau.

### 1.5 Cam kết liêm chính đối với Roboflow Export
- Nếu chưa có tệp ZIP thực tế trong [`data/acquisition_staging/roboflow_battery/`](../data/acquisition_staging/roboflow_battery/), script thông báo hướng dẫn người dùng tải từ trang web chính thức và thoát an toàn với mã `0`. Tuyệt đối không giả lập dữ liệu hay sinh kết quả ảo.

---

## 2. KẾT QUẢ KIỂM THỬ HỒI QUY (TDD EVIDENCE)

Tệp kiểm thử: [`tests/qa/test_phase4e_intake_safety.py`](../tests/qa/test_phase4e_intake_safety.py)

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: .
collected 19 items

tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_valid_cases PASSED [  5%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_token_count_strictly_five PASSED [ 10%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_class_id_validation PASSED [ 15%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_strictly_within_image_bounds PASSED [ 21%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_nc_mismatch PASSED [ 26%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_identifies_battery_classes PASSED [ 31%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_zip_slip_protection PASSED [ 36%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_bomb_protection PASSED [ 42%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_fails_closed_on_missing_files PASSED [ 47%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_verifies_holdout_jpg_jpeg_png PASSED [ 52%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_whitelist_does_not_contain_pending_annotation_comments PASSED [ 57%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_mini_pilot_shortlist_structure PASSED [ 63%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_unapproved_run_fails_closed PASSED [ 68%]
tests/qa/test_phase4e_intake_safety.py::test_quarantine_non_battery_source_class PASSED [ 73%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_vetted_battery_types PASSED [ 78%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_accessories_excluded PASSED [ 84%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_ambiguous_quarantined PASSED [ 89%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_with_vetted_battery_form_factors PASSED [ 94%]
tests/qa/test_phase4e_intake_safety.py::test_cache_key_uniqueness_for_different_paths PASSED [100%]

============================= 19 passed in 1.23s ==============================
```

---

## 3. TASK B: TIỀN TRẠM VÀ KIỂM TRA MẠNG 4 ẢNH WIKIMEDIA MINI-PILOT

Hệ thống đã thực hiện kiểm tra mạng độc lập (HTTP HEAD request) đối với 4 URL nguồn:

| # | Mã ứng viên | Tên tệp Wikimedia gốc | HTTP Status | Content-Length | Giấy phép | Tác giả | Trạng thái tiền trạm |
|---|---|---|---|---|---|---|---|
| **1** | `wiki_bio_banana_peel_ground` | `File:Banana_peel_on_the_ground.jpg` | **200 OK** | 4,573,954 B | CC BY-SA 4.0 | Vladiator360 | URL & Bản quyền hợp lệ |
| **2** | `wiki_bio_street_food_remains` | `File:Street_Food_Remains_(188107205).jpeg` | **200 OK** | 294,623 B | CC BY-SA 3.0 | Andrew Currie | URL & Bản quyền hợp lệ |
| **3** | `wiki_neg_junction_boxes` | `File:Junction_boxes_-_geograph.org.uk_-_580156.jpg` | **200 OK** | 110,051 B | CC BY-SA 2.0 | Thomas Nugent | URL & Bản quyền hợp lệ |
| **4** | `wiki_neg_utility_box_leeds` | `File:Utility_box,_Wesley_Street,_Beeston,_Leeds_(11th_August_2023)_001.jpg` | **200 OK** | 5,736,159 B | CC BY-SA 4.0 | Mtaylor848 | URL & Bản quyền hợp lệ |

- **Tổng dung lượng tải thực tế:** 10,714,787 bytes (~10.22 MiB), thấp hơn đáng kể so với ngưỡng trần 20 MiB.
- **Tệp QA Manifest & Bbox sơ bộ:** Đã chuẩn bị sẵn tại [`data/acquisition_staging/manifests/mini_pilot_qa_manifest.json`](../data/acquisition_staging/manifests/mini_pilot_qa_manifest.json).

---

## 4. BẢNG THỐNG KÊ TRẠNG THÁI QA ỨNG VIÊN MINI-PILOT

| Trạng thái | Số lượng ảnh | Chi tiết |
|---|---|---|
| **PASS** | **0** | Chưa có ảnh nào được duyệt chính thức vì chưa tải vật lý về đĩa |
| **REJECT** | **0** | Chưa có ứng viên nào bị loại trong đợt shortlist 4 ảnh này |
| **PENDING** | **4** | Cả 4 ảnh đang chờ lệnh phê duyệt tải và kiểm duyệt trực quan toàn khung hình |

---

## 5. KHÓA AN TOÀN TRƯỚC KHI TẢI (PRE-DOWNLOAD GATE)

Hệ thống đang tuân thủ nghiêm ngặt chỉ thị: **CHỈ THỰC HIỆN TẢI NẾU NGƯỜI DÙNG ĐÃ PHÊ DUYỆT RÕ PHẠM VI NÀY**.

Hiện tại:
- Chưa tải bất kỳ byte ảnh nào trong 4 ảnh trên về đĩa.
- Chưa tạo file nhãn `.txt` 0-byte hoặc nhãn train.
- Chưa đưa bất kỳ dữ liệu nào vào tập huấn luyện.

Để phê duyệt tải 4 ảnh về thư mục đệm và tạo Contact Sheet 2x2 cùng Visual QA thật, người dùng vui lòng gửi lệnh:
```text
[DUYET_TAI_MINI_PILOT]
```

---

## 6. BỘ CHỈ SỐ NGHIỆM THU CUỐI CÙNG (FINAL METRICS)

```text
REAL NEW IMAGES DOWNLOADED = 0
VISUALLY QA PASSED         = 0
APPROVED FOR TRAIN         = 0
```
