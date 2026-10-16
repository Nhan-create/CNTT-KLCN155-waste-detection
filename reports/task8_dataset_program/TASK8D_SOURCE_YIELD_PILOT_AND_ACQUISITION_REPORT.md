# BÁO CÁO NGHIỆM THU TASK 8D — HIGH-YIELD REAL-WASTE ACQUISITION & BASELINE RECONCILIATION

- **Repository:** `D:\CNTT-KLCN155-waste-detection`
- **Trạng thái cổng huấn luyện:** `APPROVED_FOR_TRAIN = False` trên 100% ảnh mới (Tuyệt đối chưa chạy E4/E5 training, chưa thay đổi E1 trên Streamlit, chưa chạm vào tập test frozen `299` / `274` / `D:\Hinh_Anh_Rac`).
- **Sổ cái chuẩn hóa 5.000 ảnh/lớp:** [`CANONICAL_5000_DATASET_LEDGER.md`](file:///d:/CNTT-KLCN155-waste-detection/reports/task8_dataset_program/CANONICAL_5000_DATASET_LEDGER.md) | [`.json`](file:///d:/CNTT-KLCN155-waste-detection/reports/task8_dataset_program/CANONICAL_5000_DATASET_LEDGER.json) | [`.csv`](file:///d:/CNTT-KLCN155-waste-detection/reports/task8_dataset_program/CANONICAL_5000_DATASET_LEDGER.csv)
- **Manifest QA Task 8D:** [`task8d_qa_manifest.json`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/task8d_qa_manifest.json) | [`task8d_qa_manifest.csv`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/task8d_qa_manifest.csv)
- **Bộ mẫu đại diện Task 8C đính kèm để PM kiểm tra trực tiếp:** [`task8c_pm_review_representative_pack/`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8c_pm_review_representative_pack/task8c_representative_manifest.json) (`76` ảnh + nhãn `.txt` + overlays + contact sheets).

---

## NHIỆM VỤ 1 — CHUẨN HÓA BẢNG BASELINE & SỔ CÁI 5.000 ẢNH/LỚP

Đã đối chiếu trực tiếp từ các file vật lý trên đĩa bằng [`scripts/build_canonical_5000_ledger.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/build_canonical_5000_ledger.py) và bãi bỏ hoàn toàn các dict hardcode cũ:

1. **`Train V2 Frozen` (`1.398` ảnh unique, `3.963` bbox):** Tập train đóng băng của V2 ([`data/detection/splits_real_v2/train.txt`](file:///d:/CNTT-KLCN155-waste-detection/data/detection/splits_real_v2/train.txt)) gồm `1.029` ảnh TACO + `369` ảnh Open Images V7 dùng để huấn luyện mô hình E1 hiện hành.
2. **`Train Baseline Sạch Thực Tế (V3 Candidate Sanitized)` (`1.277` ảnh unique, `3.052` bbox):** Tập `Train V2 Frozen` sau khi loại bỏ `121` ảnh train nằm trong danh sách `176` ảnh quarantine/near-duplicate theo chuẩn V3 ([`data/detection_v3_candidate/yolo/labels/train`](file:///d:/CNTT-KLCN155-waste-detection/data/detection_v3_candidate/yolo/labels/train)). **Đây là Baseline Train Sạch chuẩn xác nhất đã được duyệt (`Approved Clean Baseline`).**
3. **Giải thích sai lệch bảng `Train Độc Lập Trước Task 8` của Wave 2 cũ (`2.586` lượt lớp):** Do dict hardcode `PRIOR_TRAIN_COUNTS` trong script Wave 2 cũ cộng lẫn cả `train + val` của `real_v2` cùng `140` ảnh `roboflow_battery` (gồm cả studio + val) và gán nhầm cột lớp giữa `clothes` và `paper`.
4. **Giải thích sai lệch bảng `Train Baseline Sạch` của Task 8C cũ (`2.428` ảnh / `2.469` lượt lớp):** Do dict hardcode `BASELINE_TRAIN_PER_CLASS` trong script Task 8C cũ đếm từ thư mục thô trước V2 (`data/detection/labels/real` — `2.130` ảnh) cộng với các ảnh Mendeley đơn nhãn đã bị quarantine khi sang V2/V3.
5. **`Dữ liệu V3 Pin (Roboflow) & Task 7`:** `140` ảnh gốc `roboflow_battery` (`111` train + `29` val) từng được nhân bản 3x bằng offline augmentation thành `333` file trong E3. Trong sổ cái chuẩn hóa, **loại bỏ 100% `333` bản augmentation**, tách `111` ảnh gốc Roboflow Battery (chứa nhiều ảnh studio) sang cột `Aux & V3-RF Train`, và chỉ cộng **`16` ảnh dương tính `TRAIN_V4` của Task 7 (`41` bbox)** + **`10` ảnh `batch_a_mini01`** vào cột Staged Real-Waste Train.
6. **Các tập Staging Wave 1 / Wave 2 / Wave 8D:** Tách bạch hoàn toàn `REAL_WASTE_PASS` (`TRAIN_REAL_WASTE_*` vs `VAL_REAL_WASTE_*`) và `KEEP_AUXILIARY` (`AUXILIARY_*`). Không tính bất kỳ ảnh Validation/Test hay Auxiliary nào vào tiến độ `5.000` ảnh train rác thực tế/lớp.

---

## NHIỆM VỤ 2 & 3 — THAY ĐỔI CHIẾN LƯỢC TÌM DỮ LIỆU & PILOT 30 ẢNH/NGUỒN TRƯỚC KHI TẢI LỚN

- **Chấm dứt tìm kiếm từ khóa danh từ chung (`apple`, `book`, `glass`, `shoes`).** Chuyển sang khảo sát các bộ dữ liệu detection chuyên về rác thải và truy vấn theo ngữ cảnh rác thực tế (`litter`, `dumpster`, `garbage bin`, `food waste`, `rotten fruit`, `crushed can`, `broken glass`, `corroded/leaked battery`, `discarded shoes/clothes`).
- **Kiểm tra mẫu Pilot 30 ảnh trên 7 nguồn ứng viên ([`pilot_30_yield_summary.json`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_30_yield_summary.json)):** Đã **DỪNG NGAY (`STOPPED_AFTER_PILOT`, không nhập bất kỳ ảnh nào)** đối với **4 nguồn** trả về ảnh sản phẩm sạch, đồ gia dụng đang dùng, nhãn sai hoặc trùng lặp TrashNet/specimen; chỉ mở rộng **3 nguồn có tỷ lệ rác thực tế cao**:

| Mã nguồn | Loại nguồn | Giấy phép | Mẫu Pilot | Tỷ lệ Rác Thực Tế ở Pilot | Quyết định sau Pilot | Contact Sheet 30 ảnh Pilot | Phát hiện & Lý do giữ / dừng |
|---|---|---|---:|---:|---|---|---|
| `SRC_8D_01_KEREMBERKE_GARBAGE_OD` | `HF Object Detection Dataset` | `CC BY 4.0 (contaminated with watermarked stock photos)` | `30` | **6/30 (`20.0%`)** | **`STOPPED_AFTER_PILOT`** | [`pilot_src01_keremberke_garbage_od_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src01_keremberke_garbage_od_30.jpg) | 20.0% real-waste yield (6/30). High duplicate overlap with TrashNet studio images already in data/processed_v2, severe watermark contamination (Shutterstock, Getty, iStock, 123RF), and potted flowers in a vase mislabeled as biodegradable waste -> STOPPED immediately. |
| `SRC_8D_02_ALEXNOVA_MRS_TRASH_OD` | `HF Object Detection Dataset` | `Unspecified / Mixed` | `30` | **2/30 (`6.7%`)** | **`STOPPED_AFTER_PILOT`** | [`pilot_src02_alexnova_mrs_trash_od_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src02_alexnova_mrs_trash_od_30.jpg) | 6.7% real-waste yield (2/30). 93% of pilot samples are clean household objects in normal use (kitchen cleavers, spatulas, spoons, notebooks, toy excavators, Rubik's cubes) labeled by material -> STOPPED immediately. |
| `SRC_8D_03_DMEDHI_GARBAGE_OD` | `HF Object Detection Dataset` | `Unspecified / Mixed` | `30` | **7/30 (`23.3%`)** | **`STOPPED_AFTER_PILOT`** | [`pilot_src03_dmedhi_garbage_od_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src03_dmedhi_garbage_od_30.jpg) | 23.3% real-waste yield (7/30). TrashNet derivative overlapping data/processed_v2, severe taxonomy corruption (glass wine bottles labeled as Metal, metal foil pans labeled as Trash), and VectorStock watermarks -> STOPPED immediately. |
| `SRC_8D_04_SHYNUAA_STREET_GARBAGE` | `HF Object Detection Dataset (CVAT YOLO bboxes)` | `MIT License` | `30` | **27/30 (`90.0%`)** | **`APPROVED_FOR_EXPANSION`** | [`pilot_src04_shynuaa_street_garbage_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src04_shynuaa_street_garbage_30.jpg) | 90.0% real-waste yield (27/30). Authentic outdoor street litter on pavement, grass, curbs, and bins with tight CVAT bounding boxes (EXIF orientation normalized) -> APPROVED & refined to 10-class material taxonomy. |
| `SRC_8D_05_VOXEL51_FOOD_WASTE` | `HF Object Detection Dataset` | `CC BY 4.0` | `30` | **2/30 (`6.7%`)** | **`STOPPED_AFTER_PILOT`** | [`pilot_src05_voxel51_food_waste_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src05_voxel51_food_waste_30.jpg) | 6.7% unique real-waste specimen yield (2/30). 18 consecutive images are multi-angle bursts of the exact same meal plate (bonid=497808) with noisy auto-generated yoloe boxes covering the ceramic plate -> STOPPED immediately. |
| `SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED` | `Public OD Benchmark (Human Verified Boxes)` | `CC BY 2.0 (Individual Flickr Authors Verified)` | `30` | **11/30 (`36.7%`)** | **`APPROVED_FOR_EXPANSION`** | [`pilot_src06_oi_waste_context_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src06_oi_waste_context_30.jpg) | 36.7% pilot real-waste yield (11/30) after replacing generic class pulls with strict waste-context Flickr title filters (trash, garbage, litter, dump, scrap, broken, crushed, rotten) or Waste container (/m/01y9k5) co-occurrence -> APPROVED with manual per-image visual gate. |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `Curated Real-Waste Category & Bitmap Search` | `CC0 / Public Domain / CC BY / CC BY-SA` | `30` | **13/30 (`43.3%`)** | **`APPROVED_FOR_EXPANSION`** | [`pilot_src07_wikimedia_waste_context_30.jpg`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src07_wikimedia_waste_context_30.jpg) | 43.3% pilot real-waste yield (13/30) using explicit waste/decay categories (Category:Leaked_electric_batteries, Category:Food_waste, Category:Broken_glass, Category:Aluminium_cans, Category:Litter) + tight manual instance bounding boxes -> APPROVED. |

---

## NHIỆM VỤ 4 & 5 — KẾT QUẢ THU THẬP LÔ RÁC THỰC TẾ TASK 8D & THỐNG KÊ NGHIỆM THU

### 5.1. Thống kê tổng hợp & Tỷ lệ chấp nhận theo từng nguồn mở rộng

- **Tổng số ảnh ứng viên tải về (đã khử trùng lặp SHA-256 & pHash so với toàn bộ repo):** **`443` ảnh**
- **Tổng số ảnh đạt `REAL_WASTE_PASS` (`OBJECT_ANNOTATION_VALID=True`, `REAL_WASTE_CONTEXT_VERIFIED=True`, `domain_tag=ACTUAL_WASTE`):** **`200` ảnh độc lập** (Tỷ lệ chấp nhận toàn đợt: **`45.15%`** — tăng gấp **7.3 lần** so với tỷ lệ `6.17%` của Wave 2).
  - Tập `TRAIN_REAL_WASTE_W8D`: **`175` ảnh độc lập** (`314` bbox).
  - Tập `VAL_REAL_WASTE_W8D`: **`25` ảnh độc lập** (`54` bbox).
  - **Kiểm tra rò rỉ Train–Val (`same-session` / `same-specimen`):** **`0` trùng lặp** (`train_val_session_overlap_count = 0`, `train_val_specimen_overlap_count = 0`).
- **Tổng số Bounding Box trong `200` ảnh `REAL_WASTE_PASS`:** **`368` instance bboxes** (trung bình `1.84` bbox/ảnh; **`21` ảnh đa lớp**).
- **Số ảnh giữ ở `KEEP_AUXILIARY` (`AUXILIARY_NON_WASTE_TRAIN_HOLD`):** **`36` ảnh** (tọa độ bbox đúng nhưng thuộc ngữ cảnh `IN_USE` / `PRODUCT_STUDIO` / `AMBIGUOUS_CONTEXT`, tách riêng không tính vào rác thực tế).
- **Số ảnh chuyển `NEEDS_REVIEW` (`MANUAL_REVIEW_QUEUE`):** **`28` ảnh** (các kiện rác ép/đống rác mật độ quá dày không thể tách từng vật thể độc lập, giữ lại chờ duyệt tay để tuân thủ quy tắc *không vẽ 1 box trùm cả đống rác*).
- **Số ảnh `REJECT`:** **`179` ảnh** (ảnh có người/vật đang sử dụng không phải rác, trang sách scan, 4 ảnh ghép collage nhiều khung, watermark, hoặc cùng mẫu vật).

| Nguồn mở rộng (3 nguồn vượt qua Pilot) | Số ảnh tải về | Số ảnh `REAL_WASTE_PASS` | Tỷ lệ chấp nhận (`Yield %`) | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` |
|---|---:|---:|---:|---:|---:|---:|
| `SRC_8D_04_SHYNUAA_STREET_GARBAGE` | **90** | **81** | **`90.00%`** | 0 | 5 | 4 |
| `SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED` | **81** | **27** | **`33.33%`** | 15 | 1 | 38 |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | **272** | **92** | **`33.82%`** | 21 | 22 | 137 |
| **TỔNG CỘNG TOÀN ĐỢT TASK 8D** | **`443`** | **`200`** | **`45.15%`** | **`36`** | **`28`** | **`179`** |

### 5.2. Phân bố theo 10 lớp của lô Task 8D & Khoảng cách còn lại tới Mốc 500 / 1.500 / 3.000 / 5.000

| Lớp (`0..9`) | Task 8D `REAL_WASTE_PASS` (All: `200`) | **Task 8D `TRAIN_REAL_WASTE` (`175`)** | Task 8D `VAL_REAL_WASTE` (`25`) | Task 8D BBoxes (`368`) | Baseline Sạch V3 (`1.277`) | Tổng Staged RW Train (T7+Mini+W1+W2+W8D) | **Tổng Train Rác Thực Tế Tích Lũy** | Thiếu tới **M1 (`500`)** | Thiếu tới **M2 (`1.500`)** | Thiếu tới **M3 (`3.000`)** | Thiếu tới **M4 (`5.000`)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | **16** | **12** | 4 | **27** | 1 | +32 | **33** (69 bx) | **467** | **1,467** | **2,967** | **4,967** |
| `1: biological` | **22** | **18** | 4 | **27** | 3 | +53 | **56** (67 bx) | **444** | **1,444** | **2,944** | **4,944** |
| `2: cardboard` | **20** | **19** | 1 | **33** | 108 | +23 | **131** (181 bx) | **369** | **1,369** | **2,869** | **4,869** |
| `3: clothes` | **10** | **10** | 0 | **11** | 169 | +12 | **181** (312 bx) | **319** | **1,319** | **2,819** | **4,819** |
| `4: glass` | **11** | **10** | 1 | **19** | 89 | +18 | **107** (180 bx) | **393** | **1,393** | **2,893** | **4,893** |
| `5: metal` | **12** | **11** | 1 | **17** | 189 | +17 | **206** (352 bx) | **294** | **1,294** | **2,794** | **4,794** |
| `6: paper` | **33** | **27** | 6 | **52** | 111 | +31 | **142** (187 bx) | **358** | **1,358** | **2,858** | **4,858** |
| `7: plastic` | **31** | **28** | 3 | **56** | 732 | +35 | **767** (1373 bx) | **0** | **733** | **2,233** | **4,233** |
| `8: shoes` | **9** | **9** | 0 | **12** | 111 | +20 | **131** (242 bx) | **369** | **1,369** | **2,869** | **4,869** |
| `9: trash` | **63** | **57** | 6 | **114** | 233 | +64 | **297** (576 bx) | **203** | **1,203** | **2,703** | **4,703** |

### 5.3. Nguồn gốc & Giấy phép (`Provenance & License`) của 200 ảnh `REAL_WASTE_PASS`

| Mã nguồn | Giấy phép xác minh | Số ảnh `REAL_WASTE_PASS` | Bằng chứng Provenance lưu trong `task8d_qa_manifest.json` |
|---|---|---:|---|
| `SRC_8D_04_SHYNUAA_STREET_GARBAGE` | `MIT License` | **81** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED` | `https://creativecommons.org/licenses/by/2.0/` | **27** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY 2.0` | **15** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY 3.0` | **1** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY-SA 2.0` | **15** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY-SA 3.0` | **11** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY-SA 3.0 de` | **1** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC BY-SA 4.0` | **31** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `CC0` | **13** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |
| `SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES` | `Public domain` | **5** | 100% có URL gốc, SHA-256, dHash & tác giả/attribution xác minh |

### 5.4. Danh mục Contact Sheets & Overlays nghiệm thu trực quan

1. **Contact Sheets của 7 mẫu Pilot 30 ảnh/nguồn (`pilot_30_per_source/`):**
   - [`pilot_src01_keremberke_garbage_od_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src01_keremberke_garbage_od_30.jpg) (`STOPPED_AFTER_PILOT`)
   - [`pilot_src02_alexnova_mrs_trash_od_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src02_alexnova_mrs_trash_od_30.jpg) (`STOPPED_AFTER_PILOT`)
   - [`pilot_src03_dmedhi_garbage_od_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src03_dmedhi_garbage_od_30.jpg) (`STOPPED_AFTER_PILOT`)
   - [`pilot_src04_shynuaa_street_garbage_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src04_shynuaa_street_garbage_30.jpg) (`APPROVED_FOR_EXPANSION` — 90% real street litter)
   - [`pilot_src05_voxel51_food_waste_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src05_voxel51_food_waste_30.jpg) (`STOPPED_AFTER_PILOT`)
   - [`pilot_src06_oi_waste_context_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src06_oi_waste_context_30.jpg) (`APPROVED_FOR_EXPANSION`)
   - [`pilot_src07_wikimedia_waste_context_30.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/pilot_30_per_source/pilot_src07_wikimedia_waste_context_30.jpg) (`APPROVED_FOR_EXPANSION`)
2. **Contact Sheets của 200 ảnh `REAL_WASTE_PASS` & `KEEP_AUXILIARY` / `NEEDS_REVIEW` (`contact_sheets/`):**
   - [`task8d_real_waste_pass_shynuaa.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_real_waste_pass_shynuaa.jpg) (`81` ảnh Street Garbage)
   - [`task8d_real_waste_pass_openimages.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_real_waste_pass_openimages.jpg) (`27` ảnh Open Images Waste-Filtered)
   - [`task8d_real_waste_pass_wikimedia_p1.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_real_waste_pass_wikimedia_p1.jpg) (`48` ảnh Wikimedia Real-Waste p1)
   - [`task8d_real_waste_pass_wikimedia_p2.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_real_waste_pass_wikimedia_p2.jpg) (`44` ảnh Wikimedia Real-Waste p2)
   - [`task8d_real_waste_pass_multiclass.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_real_waste_pass_multiclass.jpg) (`21` ảnh Đa Lớp)
   - [`task8d_keep_auxiliary.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_keep_auxiliary.jpg) (`36` ảnh `KEEP_AUXILIARY`)
   - [`task8d_needs_review.jpg`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8d_high_yield_real_waste/contact_sheets/task8d_needs_review.jpg) (`28` ảnh `NEEDS_REVIEW`)
3. **Bộ ảnh & nhãn đại diện của Task 8C để PM kiểm tra trực tiếp (`task8c_pm_review_representative_pack/`):**
   - Gồm `76` mẫu đại diện (`41` ảnh Wave 2 `REAL_WASTE_PASS` + `20` ảnh Wave 1 `REAL_WASTE_PASS` + `15` ảnh Wave 2 `KEEP_AUXILIARY` bao gồm `task8w2_0022, 0116, 0484, 0588, 0731`), kèm đủ ảnh gốc (`images/`), file nhãn YOLO (`labels/`), ảnh vẽ bbox (`overlays/`), và 3 contact sheets (`contact_sheets/`).
