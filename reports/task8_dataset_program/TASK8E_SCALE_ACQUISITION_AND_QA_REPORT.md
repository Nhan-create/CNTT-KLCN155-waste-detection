# BÁO CÁO NGHIỆM THU TASK 8E — SCALE REAL-WASTE DATA ACQUISITION & TASK 8D FOCUSED VERIFICATION

- **Repository:** `D:\CNTT-KLCN155-waste-detection`
- **Phiên bản báo cáo:** `TASK8E_SCALE_ACQUISITION_AND_QA_V1`
- **Trạng thái phê duyệt huấn luyện (`APPROVED_FOR_TRAIN`):** `False` *(100% ảnh mới giữ ở staging chờ PM/User nghiệm thu độc lập; tuyệt đối chưa chạy huấn luyện)*
- **Tính toàn vẹn tập Test/Holdout:** `100%` *(Không đụng chạm tập frozen test `299` ảnh, candidate test `274` ảnh, hay `D:\Hinh_Anh_Rac` `27` ảnh)*

---

## 1. XÁC MINH NHANH & KHẮC PHỤC CÓ BẰNG CHỨNG DỮ LIỆU TASK 8D (MỤC 1)

Thực hiện kiểm tra tập trung toàn bộ `443` bản ghi của Task 8D (`data/acquisition_staging/task8d_high_yield_real_waste/task8d_qa_manifest.json`) bằng script `scripts/qa_and_annotate_task8d_batch.py`:

1. **Khớp 1–1 giữa ảnh, nhãn YOLO, ảnh overlay và manifest:**
   - Đã kiểm tra tự động: `100%` ảnh `REAL_WASTE_PASS` (`147` ảnh, `261` bounding boxes) có tệp `.jpg`, tệp nhãn `.txt` tại `labels_real_waste_pass/`, ảnh `.jpg` vẽ viền dày tại `overlays/`, và bản ghi khớp hoàn toàn trong `task8d_qa_manifest.json` & `.csv`.
2. **Bảo toàn tuyệt đối phân tách gốc (`source_split`) và chống rò rỉ `test`/`valid` vào `train`:**
   - Toàn bộ ảnh lấy từ `Shynuaa/Street-garbage/yolo/valid` giữ nguyên `source_split = "valid"` và được gán vào `VAL_REAL_WASTE_W8D` (`74` ảnh đạt chuẩn).
   - Toàn bộ ảnh lấy từ `Open Images V7 validation` giữ nguyên `source_split = "validation"` (`VAL_REAL_WASTE_W8D`), và ảnh từ `Open Images V7 test` (`task8d_0396`) được tách riêng vào `HOLD_SOURCE_TEST_W8D` (**`0` ảnh `test` hay `valid` lọt vào `TRAIN_REAL_WASTE_W8D`**).
3. **Xác minh nguồn gốc ảnh (`provenance_tier`) và điều kiện giấy phép của `Shynuaa/Street-garbage`:**
   - Không tin mù quáng vào nhãn `MIT` của cả kho `Shynuaa/Street-garbage`: kiểm tra tên tệp và siêu dữ liệu cho thấy kho này trộn ảnh tự chụp bằng điện thoại năm 2025 (`2025MMDD_HHMMSS.jpg`, `IMG_2025MMDD_HHMMSS.jpg`) với các ảnh tải từ web (`001.jpg`, `P1126-*.jpg`, `R-C*.jpg`, `OIP-C*.jpg`).
   - **Xử lý có bằng chứng:** Loại bỏ (`REJECT`) toàn bộ `12` ảnh nguồn web không rõ tác giả (`task8d_0001, 0070, 0071, 0072, 0073, 0074, 0085, 0086, 0087, 0088, 0089, 0090`), `3` ảnh chụp lặp cùng mẫu vật (`task8d_0077, 0079, 0080`), và `1` ảnh chụp bóng phản chiếu toà nhà (`task8d_0084`). Chỉ giữ `74` ảnh chụp thực địa chính chủ (`VERIFIED_FIRST_PARTY_SMARTPHONE_CAPTURE_MIT`).
4. **Tinh chỉnh bounding boxes đa lớp / đa vật thể và tách riêng thùng rác đóng kín:**
   - Đã gán nhãn lại từng instance cho `task8d_0002` (tách 2 thùng carton + 2 túi rác), `task8d_0031` (tách 4 chai nhựa + 2 hộp giấy trên bãi cỏ), `task8d_0093` (cắt sát viên pin kiềm rỉ hóa chất), `task8d_0318` (tách quả cam mốc khỏi bát sứ).
   - Chuyển toàn bộ các ảnh thùng rác đóng kín không thấy rác bên trong hoặc vật dụng nguyên vẹn từ Open Images / Wikimedia sang `KEEP_AUXILIARY` (`68` ảnh).
5. **Sửa số lượng bounding boxes của Roboflow Battery trong sổ cái 5.000 ảnh/lớp:**
   - Đếm trực tiếp từ `data/detection_upgrade_v3_pilot/splits/train.txt` (`111` ảnh gốc) và `val.txt` (`29` ảnh gốc) đối chiếu với `labels/roboflow_battery/*.txt`: xác nhận chính xác **`111` ảnh train = `111` boxes (`1.0` box/ảnh)** và **`29` ảnh val = `29` boxes (`1.0` box/ảnh)**. Đã xoá bỏ hoàn toàn con số giả định `250 (RF)` trước đây.
6. **Gói bằng chứng kiểm định độc lập gọn nhẹ cho Task 8D (`task8d_independent_qa_mini_pack`):**
   - Đóng gói kèm trong ZIP bàn giao tại `data/acquisition_staging/task8d_independent_qa_mini_pack/` gồm `30` ảnh đại diện phân tầng + nhãn YOLO + ảnh overlay viền dày + 2 contact sheets + toàn bộ `task8d_qa_manifest.json` & `.csv` đã kiểm định.

---

## 2. ĐÁNH GIÁ PILOT 5 NGUỒN & KẾT QUẢ THU THẬP MỞ RỘNG WAVE 8E (MỤC 2 & 3)

### 2.1. Kết quả kiểm tra Pilot (`~30` ảnh/nguồn) trước khi mở rộng

| Mã Nguồn | Tên Nguồn & Phân Tập (`source_split`) | Cỡ Mẫu Pilot | Tỷ Lệ Rác Thực (`ACTUAL_WASTE`) | Quyết Định | Ghi Chú Kiểm Định Nguồn & Giấy Phép |
|---|---|---:|---:|---|---|
| `PILOT_8E_01` | Shynuaa/Street-garbage (HuggingFace `yolo/train` First-Party 2025 Captures) | 30 | **96.7%** | **EXPAND_APPROVED** | High-yield real street litter across cardboard, paper, metal cans, glass bottles, plastic bags/bottles/cups, biological food scraps, and mixed trash. |
| `PILOT_8E_02` | hariharan944212005/indoor-waste-detection-merged (HuggingFace) | 30 | **33.3%** | **STOPPED_AFTER_PILOT** | Rejected per Task 8E rules: heavy offline augmentation clones and unverified upstream license provenance. |
| `PILOT_8E_03` | ninditya/aquasense-waste-detection-dataset (HuggingFace) | 30 | **60.0%** | **STOPPED_AFTER_PILOT** | Rejected per Task 8E rules: missing license provenance and single-class plastic bottle video frames. |
| `PILOT_8E_04` | Wikimedia Commons Explicit Real-Waste Categories & Bitmap Queries | 30 | **63.3%** | **EXPAND_APPROVED_WITH_MANUAL_BOX_ANNOTATION** | Essential source for scarce classes (`0: battery`, `1: biological`, `3: clothes`, `4: glass`, `5: metal`, `8: shoes`). |
| `PILOT_8E_05` | Open Images V7 (`train` and `validation` splits only, waste-context & multi-object filtered) | 30 | **46.7%** | **EXPAND_APPROVED_FILTERED** | Standalone closed trash containers (`/m/0bjyj5`) are routed to `KEEP_AUXILIARY`; multi-class/visible-waste scenes pass to `REAL_WASTE_PASS`. |

### 2.2. Thống kê tổng hợp `1.246` ảnh ứng viên độc lập mới của Wave 8E

| Mã Nguồn | Tải Về (Đã Khử Trùng) | `REAL_WASTE_PASS` (Tổng) | `TRAIN_REAL_WASTE_W8E` | `VAL_REAL_WASTE_W8E` | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` | Hiệu Suất `REAL_WASTE_PASS` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SRC_8E_01_SHYNUAA_STREET_GARBAGE_TRAIN` | 449 | **275** | **275** | **0** | 0 | 6 | 168 | **61.25%** |
| `SRC_8E_02_WIKIMEDIA_REAL_WASTE_CATEGORIES` | 467 | **114** | **104** | **10** | 91 | 12 | 250 | **24.41%** |
| `SRC_8E_03_OPENIMAGES_TRAIN_VAL_REAL_WASTE` | 330 | **14** | **13** | **1** | 298 | 16 | 2 | **4.24%** |
| **TỔNG CỘNG WAVE 8E** | **1246** | **403** | **392** | **11** | **389** | **34** | **420** | **32.34%** |

- **Tổng số bounding boxes chặt theo từng instance (`REAL_WASTE_PASS`):** `672` boxes trên `403` ảnh (`392` ảnh `TRAIN_REAL_WASTE_W8E` + `11` ảnh `VAL_REAL_WASTE_W8E`).
- **Khử trùng chuỗi chụp liên tiếp (`Shynuaa/Street-garbage` `yolo/train`):** Toàn bộ các cú bấm máy liên tiếp cách nhau `< 9 giây` (`168` ảnh góc chụp phụ) đã được loại bỏ (`REJECT`) để đảm bảo mỗi ảnh `REAL_WASTE_PASS` là một mẫu vật/điểm chụp độc lập.

---

## 3. BẢNG BAO PHỦ 10 LỚP THEO 10 CHIỀU BẮT BUỘC (MỤC 4)

Bảng dưới đây thống kê trực tiếp từ `task8e_qa_manifest.json` cho `403` ảnh `REAL_WASTE_PASS` của đợt Wave 8E (ghi rõ `UNKNOWN` cho thông tin góc máy khi nguồn ảnh cộng đồng không có metadata lắp đặt camera):

| Lớp (`ID: Tên`) | (1) Ảnh Train Độc Lập (`W8E`) | (1b) Ảnh Val Độc Lập (`W8E`) | (2) Tổng Ảnh `ACTUAL_WASTE` (`W8E`) | (2b) Tổng Boxes (`W8E`) | (3) Số Specimen / Session Độc Lập | (4) Ngoài Trời (`OUTDOOR`) | (5) Trong Nhà (`INDOOR`) | (6) Trong Thùng Rác (`in_bin`) | (7) Đa Vật Thể / Đa Lớp | (8) Nhỏ / Nghiêng / Bẩn / Che Khuất | (9) Camera Cố Định (`FIXED`) | (10) Camera Di Động / Đa Góc | (10b) Camera `UNKNOWN` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | **21** | 2 | **23** | 34 | 18 | 20 | 3 | 1 | 7 | 7 | 0 | 0 | 23 |
| `1: biological` | **23** | 2 | **25** | 27 | 21 | 25 | 0 | 2 | 6 | 8 | 0 | 0 | 25 |
| `2: cardboard` | **59** | 0 | **59** | 70 | 42 | 59 | 0 | 22 | 38 | 43 | 0 | 53 | 6 |
| `3: clothes` | **18** | 0 | **18** | 18 | 10 | 18 | 0 | 0 | 0 | 15 | 0 | 1 | 17 |
| `4: glass` | **10** | 0 | **10** | 10 | 9 | 9 | 1 | 0 | 0 | 10 | 0 | 3 | 7 |
| `5: metal` | **39** | 2 | **41** | 45 | 39 | 41 | 0 | 13 | 16 | 29 | 0 | 33 | 8 |
| `6: paper` | **41** | 0 | **41** | 49 | 33 | 41 | 0 | 13 | 30 | 34 | 0 | 37 | 4 |
| `7: plastic` | **162** | 0 | **162** | 178 | 106 | 162 | 0 | 50 | 80 | 134 | 0 | 152 | 10 |
| `8: shoes` | **23** | 4 | **27** | 29 | 23 | 27 | 0 | 1 | 3 | 9 | 0 | 0 | 27 |
| `9: trash` | **125** | 1 | **126** | 212 | 90 | 125 | 1 | 100 | 99 | 112 | 0 | 110 | 16 |

---

## 4. KIỂM ĐỊNH CHỐNG RÒ RỈ SESSION / SPECIMEN GIỮA CÁC SPLIT (MỤC 5)

- `train_val_session_overlap_count`: **`0`**
- `train_val_specimen_overlap_count`: **`0`**
- `source_test_in_train_count`: **`0`**
- `source_valid_in_train_count`: **`0`**
- `cross_wave_capture_minute_overlap (W8D valid vs W8E train)`: **`0`**

---

## 5. SỔ CÁI TIẾN ĐỘ 5.000 ẢNH/LỚP CẬP NHẬT TỪ IMAGE IDs THỰC TẾ (MỤC 6)

Sổ cái được tính tự động bởi `scripts/build_canonical_5000_ledger.py` từ tập hợp `set[str]` mã ảnh thực tế (không cộng trùng, không trộn validation/test, không tính augmentation):

| Lớp (`ID: Tên`) | Train V2 Frozen | Train V3 Clean (`ACTUAL_WASTE`) | Pre-Task8 (`Task7+Mini01`) | W1 Pass (`ACTUAL_WASTE`) | W2 Salvage (`ACTUAL_WASTE`) | W8D Train (`ACTUAL_WASTE`) | W8E Train (`ACTUAL_WASTE`) | Tổng Staged Mới (`Pending`) | **Tổng Train `ACTUAL_WASTE` Sạch Hiện Có** | Auxiliary & V3 RF (Tách Riêng) | Thiếu Tới M1 (`500`) | Thiếu Tới M2 (`1.500`) | Thiếu Tới M4 (`5.000`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | 1 | **1** | 16 | 0 | 4 | 11 | **21** | **52** | **53** | 235 | 447 | 1447 | 4947 |
| `1: biological` | 3 | **3** | 6 | 20 | 9 | 14 | **23** | **72** | **75** | 19 | 425 | 1425 | 4925 |
| `2: cardboard` | 126 | **108** | 0 | 1 | 3 | 3 | **59** | **66** | **174** | 48 | 326 | 1326 | 4826 |
| `3: clothes` | 212 | **169** | 0 | 1 | 1 | 9 | **18** | **29** | **198** | 12 | 302 | 1302 | 4802 |
| `4: glass` | 101 | **89** | 0 | 6 | 2 | 3 | **10** | **21** | **110** | 37 | 390 | 1390 | 4890 |
| `5: metal` | 214 | **189** | 0 | 0 | 6 | 4 | **39** | **49** | **238** | 71 | 262 | 1262 | 4762 |
| `6: paper` | 123 | **111** | 1 | 3 | 0 | 5 | **41** | **50** | **161** | 55 | 339 | 1339 | 4839 |
| `7: plastic` | 791 | **732** | 0 | 5 | 2 | 14 | **162** | **183** | **915** | 4 | 0 | 585 | 4085 |
| `8: shoes` | 117 | **111** | 0 | 7 | 4 | 5 | **23** | **39** | **150** | 24 | 350 | 1350 | 4850 |
| `9: trash` | 294 | **233** | 0 | 2 | 5 | 11 | **125** | **143** | **376** | 45 | 124 | 1124 | 4624 |

---

## 6. ĐÓNG GÓI BÀN GIAO `< 500 MB` TẠI `C:\Users\ad\Downloads\` (MỤC 7)

- **Tệp đóng gói:** `C:\Users\ad\Downloads\TASK8E_SCALE_REAL_WASTE_DELIVERY.zip` (nhỏ hơn giới hạn `500 MB`).
- **Thành phần bên trong gói bàn giao:**
  1. Manifest & provenance đầy đủ của Wave 8E (`data/acquisition_staging/task8e_scale_real_waste/task8e_qa_manifest.json`, `.csv`, `task8e_raw_candidates_manifest.json`).
  2. Toàn bộ ảnh đã qua QA của Wave 8E (`images/`), nhãn YOLO (`labels_real_waste_pass/`, `labels_auxiliary/`, `labels_needs_manual_review/`), ảnh overlay (`overlays/`) và contact sheets (`contact_sheets/`).
  3. Gói bằng chứng QA độc lập gọn nhẹ cho Task 8D (`data/acquisition_staging/task8d_independent_qa_mini_pack/` + `task8d_qa_manifest.json` & `.csv`).
  4. Báo cáo nghiệm thu Task 8E, báo cáo đánh giá Pilot 5 nguồn, và Sổ cái chuẩn 5.000 ảnh/lớp (`reports/task8_dataset_program/`).
  5. Toàn bộ scripts tái lập và bộ kiểm thử tự động (`scripts/`, `tests/detection/test_task8_manifest_report_consistency.py`).
