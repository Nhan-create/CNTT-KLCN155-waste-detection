# BÁO CÁO BÀN GIAO ĐỢT 1 — TASK 8: CHƯƠNG TRÌNH XÂY DỰNG DATASET 10 LỚP CÂN BẰNG QUY MÔ LỚP (5.000+ ẢNH TRAIN ĐỘC LẬP/LỚP)

- **Thời điểm xuất báo cáo:** `2026-10-14T07:46:13.941377+00:00`
- **Repository:** `.`
- **Trạng thái cổng phê duyệt:** **DỪNG TẠI CỔNG PHÊ DUYỆT TRƯỚC KHI HUẤN LUYỆN (`approved_for_train = False`)**
- **Tuân thủ tuyệt đối các giới hạn thực thi:**
  - Chưa chạy huấn luyện E4/E5 (`0` GPU training jobs).
  - Chưa tự ý phê duyệt dữ liệu vào tập train chính thức (`approved_for_train = False`).
  - Giữ nguyên tuyệt đối checkpoint E1/E3 và cấu hình mô hình trong `streamlit_app.py`.
  - Giữ nguyên tuyệt đối tập `frozen test` (`299` ảnh `real_v2` / `274` ảnh `v3_candidate`) và `holdout` (`27` ảnh tại `data/holdout`).

---

## 1. TASK 1 — ĐÁNH GIÁ CÁC NGUỒN DATASET QUY MÔ LỚN (KHÔNG CHỈ WIKIMEDIA COMMONS)

| Mã nguồn | Tên nguồn dữ liệu | Ảnh thật? | Có sẵn BBox? | Khớp 10-Class Taxonomy? | License & Provenance | Độ trùng lặp với hiện tại | Tỷ lệ ảnh thực tế (In-the-wild) | Ước lượng ảnh độc lập khả dụng | Dung lượng (GB) | Mức độ phù hợp |
|---|---|---|---|---|---|---|---|---:|---:|---|
| `SRC_01_OPEN_IMAGES_V7` | **Open Images Dataset V7 (Full Train + Validation + Test Detection Pool)** | Yes (100% real Flickr photographs) | Yes (Official human-verified bounding boxes) | High (8/10 classes map via verified MIDs: Box->2, Clothing/Glove/Towel/Footwear->3/8, Bottle/Jug/Wine glass->4/7, Tin can->5, Book/Paper cup->6, Plastic bag/Straw->7; weak on rotten biological & spent batteries) | Valid (Annotations CC BY 4.0 by Google; Images CC BY 2.0 with per-image URL, author & SHA-256) | Partial on validation split (525 val images already in real_v2; 42 new val images acquired in Wave 1; 1.74M train split is 100% unused) | High real-world indoor/outdoor diversity (~75% real-world context, ~25% studio/product which must be filtered) | **18,500** | **9.8 GB** | `TIER_1_PRIMARY_DETECTION` |
| `SRC_02_TACO_EXTENDED_COCO` | **TACO (Trash Annotations in Context) Official + Extended Community Pools** | Yes (100% in-the-wild litter photos) | Yes (COCO instance segmentation & tight bounding boxes across 60 subcategories) | Exact 9/10 classes (1:biological, 2:cardboard, 3:clothes, 4:glass, 5:metal, 6:paper, 7:plastic, 8:shoes, 9:trash; 0:battery has <5 images) | Valid (CC BY 4.0) | High on official v0.1 1,500 pool (already in real_v2); zero overlap on extended community/Roboflow unaugmented derivative splits | 100% outdoor/indoor real-world litter on streets, beaches, grass, soil | **2,200** | **1.6 GB** | `TIER_1_PRIMARY_DETECTION` |
| `SRC_03_ROBOFLOW_CC_BY_4_DETECTION` | **Roboflow Universe Curated CC-BY-4.0 Waste Detection Datasets (Base Unaugmented Originals Only)** | Yes (Real camera/phone photos; synthetic/augmented copies strictly stripped via filename stem & pHash<=8) | Yes (YOLO/COCO bounding boxes) | High when combining class-specialized projects (e.g. waste-battery-only, organic-food-waste, cardboard-recycling, scrap-metal, footwear-litter) | Valid ONLY for projects explicitly marked CC BY 4.0 or Public Domain (exclude custom/unspecified licenses) | Low (140 base battery images already staged in roboflow_battery; other CC-BY-4.0 projects are untouched) | Good (Mix of conveyor, household, bins, street litter) | **16,000** | **6.4 GB** | `TIER_1_PRIMARY_DETECTION` |
| `SRC_04_ZEROWASTE_DATASET` | **ZeroWaste Dataset (Boston University / CVPR — Conveyor & Real Material Sorting)** | Yes (High-resolution industrial & real waste sorting camera frames) | Yes (COCO bounding boxes & polygons) | Partial (Covers 2:cardboard, 5:metal, 6:paper, 7:plastic in dense multi-object scenes) | Valid (CC BY-NC 4.0 / Academic Research License) | Zero (0% overlap with real_v2) | 100% real-world crushed, dirty, deformed post-consumer waste (conveyor domain) | **4,500** | **3.2 GB** | `TIER_1_PRIMARY_DETECTION` |
| `SRC_05_UAVVASTE_PLASTOPOL_WADE` | **Outdoor Litter Detection Benchmarks (UAVVaste, PlastOPol, Cigarette Butt / Litter Detection)** | Yes (100% outdoor grass, beach, pavement, street litter photos) | Yes (COCO/YOLO bounding boxes) | Partial (Strong on 7:plastic, 5:metal, 4:glass, 6:paper, 9:trash; coarse litter labels require 10-class subclass verification) | Valid (CC BY 4.0 / Academic Open Access) | Zero (0% overlap) | 100% outdoor in-the-wild complex backgrounds | **4,800** | **2.5 GB** | `TIER_2_REMAP_DETECTION` |
| `SRC_06_MENDELEY_GARBAGE_10_12_CLASS` | **Mendeley / Kaggle 10-Class & 12-Class Garbage Classification Pools (Single & Multi-Object Real Photos)** | Yes (Real photos, though ~30% are clean e-commerce product shots that must be filtered out) | No (Classification-only folder labels; requires Detector-Assisted Pre-Box + Human QA) | Exact 10-class taxonomy (`battery`, `biological`, `cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`) | Mixed (Mendeley Data CC BY 4.0 valid; scraped retail subsets must be filtered for copyright/product bias) | Zero with real_v2 (previously quarantined when pseudo-labeled with full-image boxes; only usable after true tight physical bbox annotation) | Moderate (~70% real household/discarded items, ~30% white-background catalog images to reject) | **8,500** | **2.1 GB** | `TIER_2_NEEDS_BBOX_ANNOTATION` |
| `SRC_07_WIKIMEDIA_FLICKR_CC_COMMONS` | **Wikimedia Commons & Flickr Creative Commons API (Targeted Real-World Waste & Hard Negatives)** | Yes (Requires filtering out diagrams, posters, museum specimens, historical B&W) | No (Requires manual tight YOLO annotation) | High (Targeted category/search queries per class, especially corroded batteries, rotting food/compost, dirty packaging, abandoned shoes) | Valid (CC0, PD, CC BY, CC BY-SA with full API extmetadata, author & URL) | Controlled via SHA-256 + 4-rotation pHash<=8 pre-filter (Wave 1 + Task 7 already staged) | Very High (Rich in dirty, decayed, corroded, outdoor complex scenes) | **4,000** | **2.4 GB** | `TIER_2_HIGH_DIVERSITY_SUPPLEMENT` |
| `SRC_08_FIELD_SMARTPHONE_CAPTURE_VN` | **Controlled Field & Household Smartphone Capture Campaign (Vietnam Local Waste Context)** | Yes (100% native real-world smartphone photos matching deployment camera distribution) | No (Requires manual CVAT/LabelStudio annotation) | Exact 10-class taxonomy + intentional multi-class combinations + targeted hard negatives | 100% First-Party Project Owned (CC BY 4.0 / Full commercial & academic rights) | Zero (Strictly separated from frozen 27-image holdout `data/holdout` by photographer session & specimen ID) | 100% real-world Vietnamese household, campus, street, wet-market, and bin scenes | **5,000** | **4.5 GB** | `TIER_1_DOMAIN_ANCHOR` |

> **Bài học thực nghiệm quan trọng từ Wave 1 khi kiểm tra mẫu Open Images V7 (`task8a_155`..`task8a_196`):**
> - Lớp `Coffee cup` (`/m/02p5f1q`) trong Open Images V7 chứa phần lớn là **cốc sứ/tách trà đang có đồ uống** (`task8a_161`..`166` -> bị loại `REJECT`), không phải cốc giấy rác (`6: paper`).
> - Lớp `Box` (`/m/025dyy`) chứa cả rương gỗ/kim loại (`task8a_158` -> bị loại `REJECT`).
> - Lớp `Bottle` (`/m/04dr76w`) chứa cả chai thủy tinh (`4: glass`, ví dụ `task8a_169`, `171`, `173`, `174`) và bình nhựa PET (`7: plastic`, ví dụ `task8a_170` đã được remap sang `7: plastic`).
> - Do đó, **bắt buộc phải có bước kiểm tra mẫu & phân tách vật liệu (material verification)** trước khi nhập hàng loạt từ Open Images V7 hoặc các nguồn mở.

---

## 2. TASK 2 & TASK 3 — KẾT QUẢ THU THẬP ĐỢT ĐẦU (WAVE 1 MULTI-SOURCE ACQUISITION & STANDARDIZATION)

- **Thư mục lưu trữ Wave 1:** `data/acquisition_staging/task8a_multiclass_batch/`
- **Tổng số ảnh ứng viên thật đã tải và kiểm định thị giác từng ảnh:** **221 ảnh** (`61.47 MB`), kết hợp từ **2 họ nguồn độc lập**:
  1. **Open Images V7 Validation (AWS S3 Official Bounding Boxes, chưa từng xuất hiện trong `real_v2`):** `42` ảnh ứng viên -> **`29` ảnh đạt `QA_PASS`** (`1` ảnh giữ tại `NEEDS_REVIEW`, `12` ảnh `REJECT` do là cốc sứ/rương gỗ/ảnh ghép sản phẩm).
  2. **Wikimedia Commons API (Multi-Query & Category Search toàn bộ 10 lớp):** `179` ảnh ứng viên -> **`81` ảnh đạt `QA_PASS`** (`24` ảnh giữ tại `NEEDS_REVIEW`, `74` ảnh `REJECT` do có người đang sử dụng, ảnh lịch sử, kính hiển vi, tranh vẽ hoặc trùng phiên chụp).
- **Tổng hợp phán quyết QA Đợt 1:**
  - **`QA_PASS`:** **`110` ảnh độc lập** (`TRAIN_WAVE1 = 85`, `VAL_REAL_WAVE1 = 25`), gồm **`107` ảnh dương tính** (`190` bounding boxes chuẩn YOLO 10 lớp) và **`3` ảnh Hard Negative (`0` boxes)** là các thùng rác ngoài trời sạch không có rác lộ thiên (`task8a_194`, `195`, `196`).
  - **`NEEDS_REVIEW` (`MANUAL_REVIEW_HOLD`):** **`25` ảnh** (các đống rác/bãi chai lọ/giày dép quá dày đặc >20 vật thể hoặc còn vật thể phụ chưa chắc chắn — được tách riêng tại `labels_needs_manual_review/`, tuyệt đối không trộn vào `QA_PASS`).
  - **`REJECT` (`EXCLUDED`):** **`86` ảnh** (loại bỏ hoàn toàn, không dùng làm negative).
- **Kiểm toán chống rò rỉ dữ liệu (Leakage & Deduplication Audit):**
  - `external_sha256_collisions`: `0` (Đã phát hiện và loại bỏ `task8a_209`, `210`, `211` do trùng phiên pin rỉ sét với `task7_diverse_battery_batch`).
  - `external_phash_le8_collisions`: `0` (`zero_external_leakage = True`).
  - `cross_split_shared_specimens`: `0` (`zero_cross_split_leakage = True`).

---

## 3. TASK 4 & MỤC BÀN GIAO 5, 6 — BẢNG THỐNG KÊ CHẤT LƯỢNG, ĐỘ CÂN BẰNG 10 LỚP VÀ SỐ ẢNH CÒN THIẾU ĐẾN 4 MỐC (500 / 1.500 / 3.000 / 5.000)

### 3.1. Số lượng ảnh độc lập hợp lệ hiện có và khoảng cách (Gap) tới 4 mốc mục tiêu Train

| Class ID & Tên lớp | Train Baseline (`v3_candidate`) | Staged Trước Wave 1 (Train) | Wave 1 Mới (`TRAIN_WAVE1`) | **Tổng Ảnh Train Độc Lập Hợp Lệ Hiện Có** | Tổng Ảnh Val Độc Lập Hợp Lệ | Frozen Test | Tổng Số Boxes Hiện Có | **Thiếu tới Mốc 1 (500)** | **Thiếu tới Mốc 2 (1.500)** | **Thiếu tới Mốc 3 (3.000)** | **Thiếu tới Mốc 4 (5.000)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0: battery** | 1 | 128 | +2 | **131** | 40 | 1 | 383 | **369** | **1,369** | **2,869** | **4,869** |
| **1: biological** | 3 | 10 | +23 | **36** | 8 | 1 | 62 | **464** | **1,464** | **2,964** | **4,964** |
| **2: cardboard** | 108 | 0 | +13 | **121** | 28 | 25 | 235 | **379** | **1,379** | **2,879** | **4,879** |
| **3: clothes** | 169 | 0 | +5 | **174** | 35 | 37 | 446 | **326** | **1,326** | **2,826** | **4,826** |
| **4: glass** | 89 | 0 | +15 | **104** | 24 | 19 | 242 | **396** | **1,396** | **2,896** | **4,896** |
| **5: metal** | 189 | 1 | +1 | **191** | 43 | 40 | 477 | **309** | **1,309** | **2,809** | **4,809** |
| **6: paper** | 111 | 1 | +9 | **121** | 23 | 25 | 212 | **379** | **1,379** | **2,879** | **4,879** |
| **7: plastic** | 732 | 1 | +10 | **743** | 164 | 150 | 1,913 | **0** | **757** | **2,257** | **4,257** |
| **8: shoes** | 111 | 0 | +13 | **124** | 27 | 23 | 318 | **376** | **1,376** | **2,876** | **4,876** |
| **9: trash** | 233 | 0 | +2 | **235** | 47 | 48 | 633 | **265** | **1,265** | **2,765** | **4,765** |

### 3.2. Bảng kiểm soát chất lượng và độ đa dạng (Quality & Balance Control) trong Đợt 1 (Wave 1)

| Class ID & Tên lớp | Wave 1 `QA_PASS` (Ảnh) | Wave 1 Boxes | Số nguồn dữ liệu tích lũy | Số Specimen/Session (Wave 1) | Ảnh Đa Lớp (Multi-class) | Ảnh Nền Phức Tạp | Ảnh Ngoài Trời | Ảnh Bẩn / Hư Hỏng | Hard Negatives Liên Quan | `QA_PASS` / `NEEDS_REVIEW` / `REJECT` (Wave 1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0: battery** | **3** (`2` Tr / `1` Val) | 5 | 3 (Roboflow waste-battery-only/1, TACO v0.1, Wikimedia Commons) | 2 | 0 | 0 | 0 | 3 | 6 | `3` / `0` / `3` |
| **1: biological** | **30** (`23` Tr / `7` Val) | 37 | 2 (TACO v0.1, Wikimedia Commons) | 29 | 4 | 22 | 13 | 30 | 1 | `30` / `3` / `26` |
| **2: cardboard** | **17** (`13` Tr / `4` Val) | 27 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 17 | 6 | 17 | 2 | 8 | 1 | `17` / `6` / `7` |
| **3: clothes** | **6** (`5` Tr / `1` Val) | 9 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 6 | 0 | 5 | 0 | 2 | 1 | `6` / `1` / `3` |
| **4: glass** | **20** (`15` Tr / `5` Val) | 37 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 20 | 3 | 19 | 8 | 10 | 1 | `20` / `5` / `11` |
| **5: metal** | **2** (`1` Tr / `1` Val) | 6 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 2 | 1 | 1 | 0 | 1 | 1 | `2` / `2` / `2` |
| **6: paper** | **13** (`9` Tr / `4` Val) | 16 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 12 | 8 | 11 | 5 | 6 | 1 | `13` / `3` / `17` |
| **7: plastic** | **15** (`10` Tr / `5` Val) | 20 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 14 | 10 | 15 | 6 | 10 | 1 | `15` / `1` / `3` |
| **8: shoes** | **17** (`13` Tr / `4` Val) | 31 | 3 (Open Images V7, TACO v0.1, Wikimedia Commons) | 17 | 0 | 16 | 6 | 12 | 1 | `17` / `5` / `12` |
| **9: trash** | **2** (`2` Tr / `0` Val) | 2 | 2 (TACO v0.1, Wikimedia Commons) | 2 | 0 | 2 | 2 | 2 | 3 | `2` / `1` / `2` |

---

## 4. TASK 5 & MỤC BÀN GIAO 7 — KIỂM SOÁT CHI PHÍ, DUNG LƯỢNG, THỜI GIAN QA VÀ KẾ HOẠCH ACQUISITION TIẾP THEO

### 4.1. Ước lượng tài nguyên theo 4 mốc (Milestone 1 -> Milestone 4)

| Mốc mục tiêu | Tổng lượt ảnh-lớp Train mục tiêu | Lượt ảnh-lớp còn thiếu | Số ảnh độc lập mới cần thu thập (hệ số đa lớp ~1.25x) | Tỷ lệ tái sử dụng BBox có sẵn | Số ảnh cần gán nhãn mới / chỉnh BBox | Dung lượng đĩa tăng thêm | Thời gian QA & Gán nhãn (giờ) | Chi phí GPU (1 lần train kiểm chứng tại mốc) | Thời gian thực hiện dự kiến |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Milestone 1 (500 independent train images/class)** | 5,000 | 3,263 | **2,610 ảnh** | 72% | 1,100 ảnh | **2.4 GB** | ~18h | ~1.5h GPU | Sprint Wave 2A–2B (5–7 days) |
| **Milestone 2 (1,500 independent train images/class)** | 15,000 | 13,020 | **10,416 ảnh** | 75% | 2,900 ảnh | **7.2 GB** | ~45h | ~3.0h GPU | Waves 3–4 (2–3 weeks) |
| **Milestone 3 (3,000 independent train images/class)** | 30,000 | 28,020 | **22,416 ảnh** | 70% | 6,800 ảnh | **14.8 GB** | ~95h | ~5.5h GPU | Waves 5–7 (4–6 weeks) |
| **Milestone 4 (5,000+ independent train images/class)** | 50,000 | 48,020 | **38,416 ảnh** | 68% | 12,200 ảnh | **25.5 GB** | ~165h | ~8.5h GPU | Waves 8–10 (8–10 weeks cumulative) |

### 4.2. Kế hoạch Acquisition Đợt 2 (Wave 2 — Hướng tới Milestone 1: 500 ảnh train độc lập/lớp)

1. **Nhóm khẩn cấp số 1 (`1: biological` thiếu `464` ảnh train, `0: battery` thiếu `369` ảnh train):**
   - **Nguồn cho `1: biological`:** (a) Nhập các dự án Roboflow Universe `CC BY 4.0` chuyên về *organic food waste / rotten fruit & vegetable / kitchen compost detection* (chỉ giữ ảnh gốc base image, loại bỏ 100% bản xoay/lật augmentation qua stem + `pHash <= 8`); (b) Khai thác các lớp thực phẩm phân hủy/thừa trong Open Images V7 Train split (`Banana`, `Apple`, `Orange`, `Bread`, `Tomato`, `Food` trong ngữ cảnh rác/bỏ đi); (c) Chiến dịch chụp thực địa hộ gia đình/nhà ăn (vỏ trái cây, cơm thừa, lá rau úa, bã trà/cà phê, vỏ trứng).
   - **Nguồn cho `0: battery`:** (a) Mở rộng thêm 2–3 bộ Roboflow Universe `CC BY 4.0` về *e-waste / spent household batteries / lithium pack / button cell* (lọc bỏ augmentation); (b) Chụp thực tế theo phiên độc lập (AA/AAA, 9V, pin cúc áo, pin điện thoại phồng, pin sạc dự phòng cũ trên nền sàn nhà, thùng rác, bàn làm việc).
2. **Nhóm thiếu hụt trung bình (`4: glass` thiếu `396`, `2: cardboard` thiếu `379`, `6: paper` thiếu `379`, `8: shoes` thiếu `376`, `3: clothes` thiếu `326`, `5: metal` thiếu `309`, `9: trash` thiếu `265`; riêng `7: plastic` đã đạt `743/500` cho Mốc 1):**
   - **Khai thác tự động có kiểm duyệt từ Open Images V7 Train Split (1.74M ảnh):** Tải theo danh sách MID đã kiểm chứng (`Box`, `Paper towel`, `Book`, `Bottle`, `Wine glass`, `Boot`, `Sandal`, `High heels`, `Glove`, `Clothing`, `Tin can`, `Plastic bag`), chạy script lọc bỏ ảnh có người (`Person`/`Human body`) và kiểm tra mẫu theo lô 100 ảnh bằng contact sheet trước khi đưa vào `QA_PASS`.
   - **Khai thác ZeroWaste & Extended TACO:** Bổ sung mạnh cho `2: cardboard`, `6: paper`, `5: metal`, `7: plastic` trong bối cảnh rác nhàu nát, chồng lấp phức tạp.

---

## 5. TASK 6 — CHIẾN LƯỢC ĐÁNH GIÁ ĐA TẬP KIỂM THỬ (MULTI-BENCHMARK EVALUATION STRATEGY)

Khi đạt từng mốc Milestone (bắt đầu từ Milestone 1 = 500 ảnh train/lớp và sau khi có phê duyệt của PM/User), mọi mô hình (**YOLOv8n đối chứng** và **SSDLite320-MobileNetV3 chính**) bắt buộc phải vượt qua **4 cổng đánh giá độc lập**:

1. **Frozen Test (`data/detection_v3_candidate/images/test`, 274 ảnh):** Đo `Precision`, `Recall`, `AP@50`, `AP@50:95` của **từng lớp trong 10 lớp**. Không chấp nhận chỉ nhìn `mAP@50` trung bình.
2. **Independent Real Validation (`VAL_REAL` mở rộng theo nguồn chụp độc lập, gồm `VAL_REAL_BATTERY` + `VAL_REAL_WAVE1`):** Kiểm tra khả năng tổng quát hóa trên các nguồn ảnh hoàn toàn tách biệt theo `specimen_group`.
3. **External Real Holdout (`data/holdout`, 27 ảnh thực tế độc lập):** Kiểm tra one-to-one đối chiếu Ground Truth độc lập của QA Agent D, theo dõi tỷ lệ phát hiện đúng trên `0: battery`, `1: biological`, `2: cardboard`, `6: paper` và tỷ lệ báo động giả (False Positives) trên nền phức tạp.
4. **Cổng chống suy giảm chéo (Anti-Forgetting / Regression Gate):** Tự động so sánh bảng `AP@50` và `Recall` của từng lớp so với E1/E3 baseline; nếu bất kỳ lớp nào trong 8 lớp mạnh (`cardboard`, `clothes`, `glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`) sụt giảm quá `> 0.03 AP@50` khi nâng `battery`/`biological`, hệ thống cảnh báo suy giảm (catastrophic forgetting / class imbalance shift) và chặn xuất bản checkpoint.
