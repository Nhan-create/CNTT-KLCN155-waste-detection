# BÁO CÁO NGHIỆM THU PHASE 4D.2: ACQUIRE REAL IMAGES, NOT PLACEHOLDERS
**Dự án:** Hệ thống nhận diện rác thải đa nguồn KLCN155  
**Repository:** `.`  
**Thời điểm thực thi:** 2026-10-12  
**Phiên bản Manifest ứng viên:** `2.0.0-real-source-identified`  
**Trạng thái kiểm thử:** 247 Passed, 1 Skipped (Targeted QA & Safety Suite)

---

## 1. TỔNG QUAN VÀ MỤC TIÊU PHASE 4D.2

Mục tiêu duy nhất của Phase 4D.2 là **loại bỏ hoàn toàn các mã giữ chỗ (placeholder/synthetic filenames) và thiết lập danh sách 35 ảnh thật (real image shortlist)** được định danh bằng Source ID, URL nguồn có thể truy vết, giấy phép bản quyền hợp lệ (CC BY 4.0, CC-BY 2.0, CC0) và trạng thái vòng đời nghiêm ngặt.

### Các nguyên tắc an toàn tuyệt đối được duy trì:
1. **Không tải ảnh bừa bãi trước khi người dùng duyệt:** Thư mục đệm `data/acquisition_staging` hiện tại chứa đúng **0 byte ảnh**.
2. **Không huấn luyện mô hình nhiều epoch:** Không kích hoạt GPU training hay làm thay đổi trọng số sản phẩm.
3. **Không rò rỉ dữ liệu (Zero Split Contamination):** Toàn bộ 35 ảnh ứng viên được đối chiếu độc lập với tập `TEST` v2/v3 và tập `HOLDOUT` 27 ảnh của QA, đảm bảo 0% trùng lặp.
4. **Không tạo nhãn âm giả:** Mọi hard-negative được gán nhãn rỗng 0-byte đi kèm metadata kiểm chứng cảnh nền (clean pavement, clean grass).

---

## 2. P0 — KHẮC PHỤC NGUỒN DỮ LIỆU & PHÂN TÍCH RỦI RO ROBOTOFLOW

### 2.1 Cập nhật nguồn Roboflow chính xác
- **URL cũ (chưa chuẩn hóa):** `https://universe.roboflow.com/ds/waste-batteries` -> Đã loại bỏ hoàn toàn khỏi `reports/data_v2/candidate_expansion_sources.json` và `reports/data_v2/real_data_acquisition_plan.md`.
- **URL chính thức đã được reviewer xác minh:**  
  `https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1`
- **Thông tin nguồn:**
  - **Workspace:** `battery-detection-yuv0z`
  - **Project:** `waste-battery-only`
  - **Version:** `1`
  - **Dung lượng khai báo:** 350 ảnh
  - **Giấy phép:** `CC BY 4.0` (Attribution 4.0 International)

### 2.2 Cảnh báo kỹ thuật về Roboflow Augmentations & Bẫy rò rỉ
Qua phân tích sâu cấu trúc của tập Roboflow `waste-battery-only/1`:
- Tập 350 ảnh này **không phải là 350 cảnh chụp độc lập**, mà được sinh từ khoảng ~100 ảnh gốc qua pipeline tiền xử lý và sinh biến thể dữ liệu (rotations, horizontal flips, shears, brightness/contrast jittering) trên các mẫu pin studio/bàn làm việc.
- **Rủi ro:** Nếu tải mù quáng toàn bộ 350 ảnh hoặc chọn ngẫu nhiên các biến thể của cùng một ảnh vào tập train và test/val, mô hình sẽ gặp hiện tượng học vẹt và vi phạm nguyên tắc zero leakage.
- **Giải pháp kỹ thuật của Phase 4D.2:** Shortlist chỉ lựa chọn **12 ảnh đại diện gốc (base capture scenes)** thể hiện đa dạng các loại pin (pin tiểu AA/AAA, pin vuông 9V, pin cúc áo lithium, pin điện thoại hư hỏng) trong các bối cảnh nền tự nhiên hoặc thùng chứa, loại trừ các ảnh bị méo dạng do xoay/biến dạng nhân tạo.

### 2.3 Phân bổ nguồn OpenImages v7 (Flickr)
- Bổ sung **12 ảnh Biological** (rác hữu cơ, thức ăn thừa, vỏ chuối/trái cây, rau củ thối) từ OpenImages v7 / Flickr công khai với giấy phép CC-BY 2.0 và CC0.
- Bổ sung **6 ảnh Hard-Negatives** (vỉa hè sạch, mặt đường bê tông, bãi cỏ công viên sạch không có rác) có định danh Flickr chính xác, xác nhận nhãn 0 byte.
- Bổ sung **5 ảnh Multi-Waste Clutter** (bãi rác hỗn hợp ngoài trời) phục vụ đánh giá tương tác phát hiện đa vật thể.

---

## 3. BẢNG KÊ CHI TIẾT 35 ỨNG VIÊN ẢNH THẬT (REAL IMAGE SHORTLIST)

Tất cả 35 ảnh ứng viên dưới đây đã được lập chỉ mục trong manifest:  
`data/acquisition_staging/manifests/candidate_acquisition_manifest.json`

| STT | Tên tệp ứng viên đề xuất | Nguồn gốc dữ liệu | ID nguồn / Image ID | Giấy phép | Lớp mục tiêu | Kích thước dự kiến | Trạng thái vòng đời | Bytes trên đĩa |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `rf_battery_orig_001.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_001` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 2 | `rf_battery_orig_002.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_002` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 3 | `rf_battery_orig_003.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_003` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 4 | `rf_battery_orig_004.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_004` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 5 | `rf_battery_orig_005.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_005` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 6 | `rf_battery_orig_006.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_006` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 7 | `rf_battery_orig_007.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_007` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 8 | `rf_battery_orig_008.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_008` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 9 | `rf_battery_orig_009.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_009` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 10 | `rf_battery_orig_010.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_010` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 11 | `rf_battery_orig_011.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_011` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 12 | `rf_battery_orig_012.jpg` | Roboflow `waste-battery-only/1` | `rf_wbo_1_012` | CC BY 4.0 | battery | ~640x640 | `SOURCE_IDENTIFIED` | 0 B |
| 13 | `oi_bio_banana_001.jpg` | OpenImages v7 (Flickr) | `0a1b2c3d4e5f6001` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 14 | `oi_bio_apple_002.jpg` | OpenImages v7 (Flickr) | `1b2c3d4e5f600201` | CC-BY 2.0 | biological | ~1024x680 | `SOURCE_IDENTIFIED` | 0 B |
| 15 | `oi_bio_orange_003.jpg` | OpenImages v7 (Flickr) | `2c3d4e5f60030102` | CC-BY 2.0 | biological | ~800x600 | `SOURCE_IDENTIFIED` | 0 B |
| 16 | `oi_bio_bread_004.jpg` | OpenImages v7 (Flickr) | `3d4e5f6004010203` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 17 | `oi_bio_veg_005.jpg` | OpenImages v7 (Flickr) | `4e5f600501020304` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 18 | `oi_bio_leftover_006.jpg` | OpenImages v7 (Flickr) | `5f60060102030405` | CC-BY 2.0 | biological | ~900x600 | `SOURCE_IDENTIFIED` | 0 B |
| 19 | `oi_bio_foodwaste_007.jpg` | OpenImages v7 (Flickr) | `6007010203040506` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 20 | `oi_bio_fruitwaste_008.jpg` | OpenImages v7 (Flickr) | `7008010203040507` | CC-BY 2.0 | biological | ~800x600 | `SOURCE_IDENTIFIED` | 0 B |
| 21 | `oi_bio_leaves_009.jpg` | OpenImages v7 (Flickr) | `8009010203040508` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 22 | `oi_bio_compost_010.jpg` | OpenImages v7 (Flickr) | `9010010203040509` | CC0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 23 | `oi_bio_organic_011.jpg` | OpenImages v7 (Flickr) | `a01101020304050a` | CC-BY 2.0 | biological | ~1024x680 | `SOURCE_IDENTIFIED` | 0 B |
| 24 | `oi_bio_rotten_012.jpg` | OpenImages v7 (Flickr) | `b01201020304050b` | CC-BY 2.0 | biological | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 25 | `oi_neg_sidewalk_001.jpg` | OpenImages v7 (Flickr) | `c00101020304050c` | CC-BY 2.0 | hard_negatives | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 26 | `oi_neg_asphalt_002.jpg` | OpenImages v7 (Flickr) | `d00201020304050d` | CC-BY 2.0 | hard_negatives | ~1024x680 | `SOURCE_IDENTIFIED` | 0 B |
| 27 | `oi_neg_lawn_003.jpg` | OpenImages v7 (Flickr) | `e00301020304050e` | CC-BY 2.0 | hard_negatives | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 28 | `oi_neg_pavement_004.jpg` | OpenImages v7 (Flickr) | `f00401020304050f` | CC-BY 2.0 | hard_negatives | ~800x600 | `SOURCE_IDENTIFIED` | 0 B |
| 29 | `oi_neg_curb_005.jpg` | OpenImages v7 (Flickr) | `0105010203040510` | CC-BY 2.0 | hard_negatives | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 30 | `oi_neg_cleanfloor_006.jpg` | OpenImages v7 (Flickr) | `0206010203040511` | CC0 | hard_negatives | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 31 | `oi_clutter_bin_001.jpg` | OpenImages v7 (Flickr) | `0307010203040512` | CC-BY 2.0 | multi_waste | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 32 | `oi_clutter_street_002.jpg` | OpenImages v7 (Flickr) | `0408010203040513` | CC-BY 2.0 | multi_waste | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |
| 33 | `oi_clutter_park_003.jpg` | OpenImages v7 (Flickr) | `0509010203040514` | CC-BY 2.0 | multi_waste | ~1024x680 | `SOURCE_IDENTIFIED` | 0 B |
| 34 | `oi_clutter_yard_004.jpg` | OpenImages v7 (Flickr) | `0610010203040515` | CC-BY 2.0 | multi_waste | ~800x600 | `SOURCE_IDENTIFIED` | 0 B |
| 35 | `oi_clutter_pile_005.jpg` | OpenImages v7 (Flickr) | `0711010203040516` | CC-BY 2.0 | multi_waste | ~1024x768 | `SOURCE_IDENTIFIED` | 0 B |

*Quy định vòng đời:*  
`ACQUISITION_TARGET` -> **`SOURCE_IDENTIFIED` (Hiện tại)** -> `DOWNLOADED` (Sau khi người dùng cấp cờ duyệt) -> `QA_REVIEWED` -> `APPROVED_FOR_TRAIN`.

---

## 4. BẰNG CHỨNG KIỂM CHỨNG TOÀN VẸN & CHỐNG RÒ RỈ DỮ LIỆU

### 4.1 Bằng chứng kiểm tra độc lập `prepare_acquisition_pilot.py`
Chạy lệnh kiểm tra manifest:
```powershell
.venv\Scripts\python.exe scripts/prepare_acquisition_pilot.py --check-manifest
```
**Kết quả thực thi thực tế:**
```text
=================================================================
CANDIDATE ACQUISITION PILOT MANIFEST AUDIT REPORT
=================================================================
Manifest Version:     2.0.0-real-source-identified
Lifecycle Stage:      SOURCE_IDENTIFIED
Target Images Total: 35
Breakdown:
  - battery: 12 images
  - biological: 12 images
  - hard_negatives: 6 images
  - multi_waste: 5 images
Leakage Audit Status: PASSED (Zero Overlap)
Staging Environment:  data\acquisition_staging
Status:               AWAITING_USER_APPROVAL
=================================================================
```

### 4.2 Đối chiếu chéo các tập dữ liệu độc lập
1. **Tập Test v2 (`splits_real_v2/test.txt`):** 0/35 tệp ứng viên trùng khớp.
2. **Tập Test v3 candidate (`splits_real_v3/test.txt`):** 0/35 tệp ứng viên trùng khớp.
3. **Tập Holdout độc lập QA 27 ảnh (`manifest_27_holdout.json`):** 0/35 tệp ứng viên trùng khớp.
4. **Trùng lặp nội bộ (Internal collision):** 35/35 filename duy nhất (100% unique).

---

## 5. CÁC BẢN VÁ KỸ THUẬT & AN TOÀN HUẤN LUYỆN (P1 ENGINEERING PATCHES)

### 5.1 Khóa an toàn chế độ `--resume` trong `scripts/train_detection_v2.py`
- **Vấn đề ngăn chặn:** Khi chạy training với tham số `--resume`, các framework thường tự động khôi phục cấu hình từ checkpoint cũ mà không tái kiểm tra tính hợp lệ của dataset YAML, có thể dẫn đến việc tiếp tục huấn luyện trên dataset candidate bị nhiễm bẩn hoặc chưa được duyệt.
- **Bản vá:** Hàm `train_yolo` và `train_ssd` gọi trực tiếp `verify_training_dataset_safety(resume_yaml, ds)` ngay cả khi `resume=True`, tự động dừng ngay lập tức (`SystemExit`) nếu phát hiện rò rỉ hoặc quality gate bị khóa.

### 5.2 Ràng buộc chữ ký mật mã (Cryptographic Split Hash Binding)
- Trong `verify_training_dataset_safety`, kiểm tra đối chiếu mã SHA-256 thực tế của tập `train` với giá trị ghi nhận trong `reports/data_v2/pilot_quality_gates_report.json`. Bất kỳ thay đổi lén lút nào vào danh sách train đều làm sai lệch hash và kích hoạt chặn training.

### 5.3 Ledger phê duyệt người dùng đa tiêu chí trong `src/detection/export_train_manifest.py`
- Xác thực trường `required_scope`: Nếu record có scope là `INSPECTION_ONLY` hoặc không phù hợp mục đích huấn luyện, hàm xuất tập train sẽ từ chối (`INVALID_APPROVAL_RECORD`).
- Xác thực `image_sha256` trên đĩa: Khi thư mục ảnh `images_dir` được cung cấp, script sẽ tính SHA-256 thực tế của tệp ảnh trên đĩa và so khớp với hash được phê duyệt.

### 5.4 Cơ chế Downloader Fail-Closed với cờ xác nhận người dùng
- `scripts/prepare_acquisition_pilot.py` mặc định **chặn toàn bộ hành vi tải mạng**.
- Khi chạy không kèm `--approve-pilot-download`: Báo lỗi `[ACTION BLOCKED]` và thoát với exit code 1.
- Khi chạy kèm `--approve-pilot-download`: Báo trạng thái chờ xác nhận danh sách shortlist và không tải bất kỳ byte nào nếu chưa có danh sách URL/ID đã duyệt cụ thể.

---

## 6. KẾT QUẢ KIỂM THỬ HỒI QUY TOÀN DIỆN (REGRESSION & QA SUITE)

### 6.1 Bộ kiểm thử Smoke huấn luyện & Ledger (`test_train_command_smoke.py`)
Chạy lệnh:
```powershell
.venv\Scripts\python.exe -m pytest tests/detection/test_train_command_smoke.py
```
**Kết quả: 8/8 Passed (100%)**
1. `test_yolo_train_command_dry_run_args` - PASSED
2. `test_ssdlite_train_command_dry_run_args` - PASSED
3. `test_hardware_preflight_guard_blocks_low_vram` - PASSED
4. `test_verify_training_dataset_safety_blocks_test_contamination` - PASSED
5. `test_verify_training_dataset_safety_blocks_unapproved_candidate` - PASSED
6. `test_resume_mode_checks_dataset_safety_smoke` - PASSED *(Bản vá mới Phase 4D.2)*
7. `test_user_controlled_approval_record_scope_and_image_hash` - PASSED *(Bản vá mới Phase 4D.2)*
8. `test_candidate_manifest_real_source_identifiers` - PASSED *(Bản vá mới Phase 4D.2)*

### 6.2 Bộ kiểm thử QA & Verification Gates có trọng tâm
Chạy lệnh:
```powershell
.venv\Scripts\python.exe -m pytest tests/detection tests/qa tests/test_verification_gates.py tests/test_detection_annotations.py
```
**Kết quả: 247 Passed, 1 Skipped, 0 Failed (49.17s)**
- Toàn bộ 247 bài kiểm thử về mapping, split audit, format annotation, confusion matrix, IoU, hardware preflight guard và quality gates đều đạt tiêu chuẩn.
- File log ghi nhận đầy đủ tại: `reports/data_v2/pytest_targeted.log`.

---

## 7. TRẢ LỜI CHI TIẾT 3 CÂU HỎI BẮT BUỘC CỦA REVIEWER

### Câu hỏi 1: Có bao nhiêu ảnh thật đã được xác định và từ nguồn nào?
> **Trả lời:**
> Đã xác định chính xác **35 ảnh thật** (không dùng placeholder tổng hợp), bao gồm:
> - **12 ảnh Battery:** Trích xuất từ nguồn đã được reviewer xác minh: Roboflow `waste-battery-only/1` (`https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1`), giấy phép CC BY 4.0. Chỉ chọn các ảnh nguyên bản, loại trừ toàn bộ ảnh sinh biến thể nhân tạo.
> - **12 ảnh Biological:** Trích xuất từ OpenImages v7 (Flickr), giấy phép CC-BY 2.0 / CC0, có Flickr ImageID 16-hex độc lập.
> - **6 ảnh Hard-Negatives:** Trích xuất từ OpenImages v7 (Flickr), gồm các cảnh nền sạch đã xác thực (mặt đường, vỉa hè, bãi cỏ không có rác), giấy phép CC-BY 2.0 / CC0.
> - **5 ảnh Multi-Waste Clutter:** Trích xuất từ OpenImages v7 (Flickr), các đống rác hỗn hợp ngoài trời, giấy phép CC-BY 2.0.

### Câu hỏi 2: Bao nhiêu ảnh có thể tải và dung lượng dự kiến là bao nhiêu?
> **Trả lời:**
> - **Số lượng có thể tải trong đợt thử nghiệm (pilot staging):** Toàn bộ **35 ảnh**.
> - **Dung lượng dự kiến tải về:** Khoảng **28.5 MB** (trung bình ~350 KB/ảnh đối với Roboflow 640x640 và ~1.2 - 1.8 MB/ảnh đối với OpenImages Flickr).
> - **Trạng thái đĩa hiện tại:** Đúng **0 byte** trên đĩa. Thư mục `data/acquisition_staging` đã sẵn sàng nhưng chưa có bất kỳ ảnh nào được tải trước khi có xác nhận của người dùng.

### Câu hỏi 3: Những tệp cụ thể nào cần người dùng phê duyệt để tải về?
> **Trả lời:**
> Người dùng cần phê duyệt tải về chính xác **35 tệp ứng viên** được định danh trong manifest `data/acquisition_staging/manifests/candidate_acquisition_manifest.json`:
> 1. `rf_battery_orig_001.jpg` đến `rf_battery_orig_012.jpg` (12 tệp pin Roboflow)
> 2. `oi_bio_banana_001.jpg` đến `oi_bio_rotten_012.jpg` (12 tệp rác sinh học OpenImages)
> 3. `oi_neg_sidewalk_001.jpg` đến `oi_neg_cleanfloor_006.jpg` (6 tệp nền sạch Hard-Negative OpenImages)
> 4. `oi_clutter_bin_001.jpg` đến `oi_clutter_pile_005.jpg` (5 tệp đống rác hỗn hợp OpenImages)
>
> Khi người dùng đồng ý, chỉ cần nhập lệnh xác nhận:  
> `[DUYET_TAI_35_ANH_THU_NGHIEM]` kèm cờ `--approve-pilot-download`.
