# TASK 8H — SCARCE-CLASS DATASET SCALING REPORT

## 1. Báo Cáo Trực Tiếp Các Chỉ Số Thực Thi (Section 6 Deliverables)
- **Số ảnh mới tải thật (Newly Downloaded Real Images):** **`419`** ảnh độc lập (`0` trùng SHA-256 / dHash với toàn bộ Baseline V2/V3 và Waves 1, 2, 8D, 8E, 8F, 8G).
- **Số ảnh được QA thực tế (QA'd Real Images):** **`419`** (`100%` qua kiểm tra vật thể, bối cảnh rác thật, chồng lấp box và phân tách domain).
- **Số ảnh đạt chuẩn `REAL_WASTE_PASS` (`ACTUAL_WASTE`):** **`215`** ảnh (**`176`** ảnh `TRAIN_REAL_WASTE_W8H` + **`39`** ảnh `VAL_REAL_WASTE_W8H`, tổng cộng **`303`** bounding boxes).
- **Độ phủ đa vật thể / đa lớp (`REAL_WASTE_PASS`):** **`59`** ảnh đa vật thể (`>= 2` boxes, `27.4%`) và **`8`** ảnh đa lớp (`>= 2` classes, `3.7%`).
- **Số truy vấn thu thập riêng cho `7: plastic`:** **`0`** (ngừng hoàn toàn việc thu thập đại trà `plastic` do đã vượt mốc `M1=500` ở mức `985` ảnh; `plastic` chỉ được gán nhãn khi đồng xuất hiện trong ảnh đa lớp của 5 lớp khan hiếm).
- **Trạng thái `APPROVED_FOR_TRAIN`:** **`False` trên 100% bản ghi** (`0` tự cấp quyền train).

## 2. Tăng Trưởng Số Ảnh Train Độc Lập & Khoảng Cách Đến 500 (`M1`) / 5.000 (`M4`)

| Class ID | Tên Lớp | Nhóm | Trước 8H (`Train`) | Wave 8H `PASS` Tổng | **Wave 8H `Train` Tăng Thêm (`+`)** | Wave 8H `Val` (`+`) | **Tổng `Train` Độc Lập Sau 8H** | Tổng `Val` Sau 8H | Còn Thiếu Đến `500` (`M1`) | Còn Thiếu Đến `5.000` (`M4`) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | **`battery`** | **KHAN HIẾM (Target)** | `52` | `42` | **`+34`** (`61` bx) | `+8` | **`86`** | `31` | **`414`** | `4914` |
| `1` | **`biological`** | **KHAN HIẾM (Target)** | `90` | `96` | **`+76`** (`123` bx) | `+20` | **`166`** | `40` | **`334`** | `4834` |
| `2` | **`cardboard`** | Phổ biến / Đồng xuất hiện | `185` | `0` | **`+0`** (`0` bx) | `+0` | **`185`** | `44` | **`315`** | `4815` |
| `3` | **`clothes`** | **KHAN HIẾM (Target)** | `34` | `24` | **`+21`** (`43` bx) | `+3` | **`55`** | `4` | **`445`** | `4945` |
| `4` | **`glass`** | **KHAN HIẾM (Target)** | `90` | `28` | **`+23`** (`32` bx) | `+5` | **`113`** | `32` | **`387`** | `4887` |
| `5` | **`metal`** | Phổ biến / Đồng xuất hiện | `301` | `5` | **`+4`** (`8` bx) | `+1` | **`305`** | `57` | **`195`** | `4695` |
| `6` | **`paper`** | Phổ biến / Đồng xuất hiện | `201` | `0` | **`+0`** (`0` bx) | `+0` | **`201`** | `47` | **`299`** | `4799` |
| `7` | **`plastic`** | Phổ biến / Đồng xuất hiện | `985` | `2` | **`+2`** (`2` bx) | `+0` | **`987`** | `177` | **`0`** | `4013` |
| `8` | **`shoes`** | **KHAN HIẾM (Target)** | `55` | `27` | **`+23`** (`34` bx) | `+4` | **`78`** | `15` | **`422`** | `4922` |
| `9` | **`trash`** | Phổ biến / Đồng xuất hiện | `361` | `0` | **`+0`** (`0` bx) | `+0` | **`361`** | `72` | **`139`** | `4639` |

## 3. Tỷ Lệ Tuyển Chọn Mỗi Nguồn Trong Task 8H

| Source ID | Tải Thật (`Downloaded`) | `REAL_WASTE_PASS` | `Train` (`W8H`) | `Val` (`W8H`) | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` | Tỷ Lệ Tuyển Chọn (`Pass Rate`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `SRC_8H_01_OPENVERSE_SCARCE_WASTE` | `130` | **`60`** | **`51`** | `9` | `10` | `1` | `59` | **`46.2%`** |
| `SRC_8H_02_ZENODO_ACADEMIC_BATTERY_AND_TEXTILE_WASTE` | `26` | **`12`** | **`8`** | `4` | `14` | `0` | `0` | **`46.2%`** |
| `SRC_8H_03_WIKIMEDIA_DEEP_SCARCE_SUBCATEGORIES` | `156` | **`53`** | **`44`** | `9` | `28` | `2` | `73` | **`34.0%`** |
| `SRC_8H_04_INATURALIST_BIOLOGICAL_DECAY_FIELD` | `107` | **`90`** | **`73`** | `17` | `2` | `0` | `15` | **`84.1%`** |

## 4. Nghiên Cứu & Pilot 18 Nguồn Tiềm Năng Cho 5 Lớp Khan Hiếm (`>= 3` Nguồn/Lớp)

| Eval ID | Lớp Mục Tiêu | Tên Nguồn / Bộ Dữ Liệu | Giấy Phép | Mẫu Pilot | Ước Tính Ảnh Độc Lập Khai Thác Được | Quyết Định | Lý Do Cụ Thể |
|---|---|---|---|---:|---|---|---|
| `EVAL_8H_BAT_01` | `0:battery` | Openverse Creative Commons Field Photography (Flickr & Geograph UK — Battery Waste Queries) | `CC0 / PDM / CC-BY / CC-BY-SA (per-` | `23` | 15-30 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Pilot confirmed authentic outdoor discarded alkaline cells, corroded AA/9V batteries, and abandoned lead-acid car batteries with explicit CC licenses. |
| `EVAL_8H_BAT_02` | `0:battery` | Zenodo 19818270 — Battery Image Dataset for EV Circularity Research (Centria Univ / EU RECIRCULATE 2026) | `CC-BY-4.0 (Ville Pitkäkangas, Cent` | `19` | 19 independent physical specimens (strictly 1 full-unit dismantle photo per battery type; 727 multi-angle/label close-ups rejected) | **`EXPAND_APPROVED_SPECIMEN_DEDUP`** | Authentic end-of-life dismantled batteries on workshop pallets/tables with academic CC-BY-4.0 license; strict 1-image-per-specimen rule prevents multi-angle leakage. |
| `EVAL_8H_BAT_03` | `0:battery` | Wikimedia Commons Deep Battery Corrosion & E-Waste Subcategories | `CC0 / CC-BY / CC-BY-SA (per-file e` | `25` | 15-25 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Provides macro/bench and bin photos of leaking alkaline/zinc-carbon batteries with visible potassium carbonate crust. |
| `EVAL_8H_BAT_04` | `0:battery` | HuggingFace afinnn/waste-detection-dataset (dataset_hscn.zip) & viola77data/recycling-dataset | `Unverified upstream web scrapes (v` | `20` | 0 (Blocked due to unverified copyright scrapes and offline rotation clones) | **`STOPPED_AFTER_PILOT`** | Contains DuckDuckGo scrapes without author attribution, no bounding boxes, and synthetic 90-degree rotation clones. |
| `EVAL_8H_CLO_01` | `3:clothes` | Openverse Creative Commons Field Photography (Flickr & Geograph UK — Discarded Garments & Lost Gloves/Socks) | `CC0 / PDM / CC-BY / CC-BY-SA` | `25` | 20-35 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | High proportion of genuine street/roadside discarded single garments (gloves, socks, shirts, hats) on pavement/mud. |
| `EVAL_8H_CLO_02` | `3:clothes` | Wikimedia Commons Deep Textile Litter & Abandoned Clothing Subcategories | `CC0 / CC-BY / CC-BY-SA` | `25` | 18-30 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Contains real outdoor discarded clothing and multi-class litter scenes with garments. |
| `EVAL_8H_CLO_03` | `3:clothes` | Zenodo 19044186 — Norway Post-Consumer Household Textiles & TrondheimSocks (NTNU 2025) | `CC-BY-4.0 (Johan Berg Pettersen & ` | `16` | 0 REAL_WASTE_PASS (uniform white lab sorting sheet); 10 holed/damaged specimens sampled strictly for KEEP_AUXILIARY | **`STOPPED_FOR_REAL_WASTE_PASS_AUX_SAMPLE_ONLY`** | Although items are genuine post-consumer waste from a recycling station bin, they are photographed on a uniform white lab sorting mat (`PRODUCT_STUDIO`), so they cannot be counted as `REAL_WASTE_PASS`. |
| `EVAL_8H_CLO_04` | `3:clothes` | Zenodo 7643417 — Medical-Waste-4.0-Dataset (CNR Italy) & HuggingFace khoaliamle/Garbage_Classification_YOLO | `CC-BY-4.0 (Zenodo 7643417) / Unver` | `16` | 0 REAL_WASTE_PASS | **`STOPPED_AFTER_PILOT`** | Medical-Waste-4.0 items are photographed isolated on a clean beige laboratory table (`PRODUCT_STUDIO`); khoaliamle lacks per-image license provenance. |
| `EVAL_8H_SHO_01` | `8:shoes` | Openverse Creative Commons Field Photography (Flickr & Geograph UK — Abandoned & Discarded Footwear) | `CC0 / PDM / CC-BY / CC-BY-SA` | `25` | 25-40 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Strongest public real-waste source for `8: shoes`: abundant single lost/abandoned shoes photographed in mud, beaches, gutters, and grass. |
| `EVAL_8H_SHO_02` | `8:shoes` | Wikimedia Commons Deep Footwear Litter & Marine Debris Subcategories | `CC0 / CC-BY / CC-BY-SA` | `25` | 20-35 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | High yield of discrete worn-out/discarded shoes and boots in outdoor environments. |
| `EVAL_8H_SHO_03` | `8:shoes` | Zenodo 7643417 (`shoe_cover_single`/`pair`) & HuggingFace khoaliamle/Garbage_Classification_YOLO (`shoes`) | `CC-BY-4.0 (Zenodo) / Unverified sc` | `12` | 0 REAL_WASTE_PASS | **`STOPPED_AFTER_PILOT`** | Fails real-waste outdoor/disposal context gate and lacks bounding box annotations. |
| `EVAL_8H_BIO_01` | `1:biological` | Openverse Creative Commons Field Photography (Flickr & Geograph UK — Rotten Fruit, Peels & Food Waste) | `CC0 / PDM / CC-BY / CC-BY-SA` | `25` | 25-45 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Rich source of discrete rotting fruit, banana peels on asphalt, apple cores, and moldy food with clear CC attribution. |
| `EVAL_8H_BIO_02` | `1:biological` | Wikimedia Commons Deep Organic Decay, Moldy Food & Kitchen Waste Subcategories | `CC0 / CC-BY / CC-BY-SA` | `25` | 25-40 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | High visual clarity of biological decay (mold, brown rot, discarded peels/cores) in non-studio settings. |
| `EVAL_8H_BIO_03` | `1:biological` | iNaturalist Citizen-Science Field Observations (`Monilinia` Brown Rot on Fruit & Outdoor Litter) | `CC0 / CC-BY / CC-BY-SA (per-photo ` | `20` | 8-15 verified REAL_WASTE_PASS images (broad keyword search stopped due to 85% macro insect/fungus rate) | **`LIMITED_TARGETED_SLICE_ONLY`** | Pilot showed general keyword search returns macro beetles/wild mushrooms; restricted strictly to verified whole rotting fruit and discarded bottle observations. |
| `EVAL_8H_BIO_04` | `1:biological` | HuggingFace ninoid/food-waste-dataset (MIT, 297 JPEGs) & nflechas/recycling_app (GARBAGE CLASSIFICATION 3) | `MIT (ninoid) / Unverified Roboflow` | `16` | 0 REAL_WASTE_PASS | **`STOPPED_AFTER_PILOT`** | Stopped after visual pilot (`grid_ninoid.jpg`) proved all 297 images are repeated incremental weighing shots of ~7 food portions on the same plate in 1 session. |
| `EVAL_8H_GLA_01` | `4:glass` | Openverse Creative Commons Field Photography (Flickr & Geograph UK — Discarded & Broken Glass Bottles/Jars) | `CC0 / PDM / CC-BY / CC-BY-SA` | `25` | 25-40 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | High yield of real outdoor littered beer/wine bottles and broken glass shards with unambiguous disposal context. |
| `EVAL_8H_GLA_02` | `4:glass` | Wikimedia Commons Deep Glass Litter & Broken Bottle Subcategories | `CC0 / CC-BY / CC-BY-SA` | `25` | 20-35 verified REAL_WASTE_PASS images | **`EXPAND_APPROVED`** | Provides clean outdoor single and multi-object glass bottle litter scenes. |
| `EVAL_8H_GLA_03` | `4:glass` | HuggingFace sharktide/recycling-in-common-context (Glass subset) | `Marked CC-BY-4.0 on HF but lacks p` | `16` | 0 REAL_WASTE_PASS | **`STOPPED_AFTER_PILOT`** | Stopped after visual pilot (`grid_sharktide.jpg`) revealed handheld full bottles (`IN_USE`) and unattributed Roboflow compilation images. |
| `EVAL_8H_GLA_04` | `4:glass` | Zenodo 6412647 (ZeroWaste-F), Zenodo 22043339 (Bangladesh Waste OD), Zenodo 12517610 (RealDirt) | `CC-BY-4.0` | `30` | 0 for scarce classes (Taxonomy mismatch) | **`STOPPED_AFTER_PILOT`** | None of these three Zenodo object-detection datasets annotate `glass`, `battery`, `clothes`, `shoes`, or `biological`. |

## 5. Danh Sách Các Nguồn Đã Hết Khả Năng Khai Thác (`EXHAUSTED` — Không Tải Lại)

| Source ID | Tên Nguồn | Các Đợt Đã Khai Thác | Tổng Đã Quét | Trạng Thái | Lý Do Đánh Dấu `EXHAUSTED` |
|---|---|---|---:|---|---|
| `SRC_EXHAUSTED_01_SHYNUAA_STREET_GARBAGE` | Shynuaa/Street-garbage (HuggingFace First-Party 2025 Street Captures) | `Task 8F (85 images), Task 8G (430 images)` | `515` | **`EXHAUSTED`** | All 916 valid images across train/valid were scanned in Waves 8F and 8G; remaining unscanned images are predominantly single-class plastic/paper street litter. |
| `SRC_EXHAUSTED_02_TACO_OFFICIAL_BASELINE` | TACO Official Dataset (1,500 images) | `Baseline V3 (958 train, 243 val, 299 frozen test)` | `1500` | **`EXHAUSTED`** | 100% of TACO images are already partitioned in the canonical baseline and frozen test split. |
| `SRC_EXHAUSTED_03_OPENIMAGES_GENERIC_KEYWORD_SLICES` | Open Images V7 Generic Waste Keyword Slices (Tin can / Bottle / Clothing / Footwear / Food) | `Task 8E (169 images), Task 8G (370 images)` | `539` | **`EXHAUSTED`** | All high-confidence waste-captioned Open Images V7 train/val items for target classes were already scanned and filtered in Tasks 8E and 8G; remaining items are active-use portraits, fashion, and studio meals. |
| `SRC_EXHAUSTED_04_WIKIMEDIA_TOP_LEVEL_GENERAL_CATEGORIES` | Wikimedia Commons Top-Level General Waste Categories (Plastic, Metal, Cardboard, Paper, General Litter) | `Wave 1, Wave 2, Task 8D, Task 8E, Task 8F, Task 8G` | `6673` | **`EXHAUSTED`** | Top-level Wikimedia categories and all 6,673 previously seen pageids/URLs are exhausted; plastic (985 train) already exceeds M1=500. |

## 6. Những Lớp Bắt Buộc Cần Tự Thu Thập Ảnh Tại Hiện Trường
- Cả **5 lớp khan hiếm (`0: battery`, `3: clothes`, `8: shoes`, `1: biological`, `4: glass`)** đều bắt buộc phải triển khai thu thập thực địa theo kế hoạch chi tiết tại `reports/task8_dataset_program/SCARCE_CLASS_FIELD_COLLECTION_PLAN.md` để đạt mốc `M1 = 500` và tiến tới `M4 = 5.000` ảnh train độc lập mỗi lớp.
