# BÁO CÁO NGHIỆM THU TASK 8A — THU THẬP & KIỂM ĐỊNH DỮ LIỆU ĐA LỚP THỰC TẾ

- **Repository:** `D:\CNTT-KLCN155-waste-detection`
- **Thời điểm tạo báo cáo (UTC):** `2026-10-14T07:32:30.615626+00:00`
- **Trạng thái tuân thủ Governance:**
  - `APPROVED_FOR_TRAIN = 0` (`approved_for_train: false` trên toàn bộ 154/154 ứng viên)
  - **KHÔNG** chạy Task 8B, **KHÔNG** huấn luyện E4/E5, **KHÔNG** thay đổi checkpoint Streamlit hay bộ Frozen Test (`299` ảnh) / External Holdout (`27` ảnh `D:\Hinh_Anh_Rac`).

---

## 1. TỔNG QUAN KẾT QUẢ ĐỐI CHIẾU CHỈ TIÊU TASK 8A

| Lớp ưu tiên Task 8A | Chỉ tiêu PM (`QA_PASS`) | Kết quả đạt (`QA_PASS` ảnh) | Tổng Bounding Boxes (`QA_PASS`) | Đề xuất `TRAIN_V4_PROPOSED` (ảnh / boxes) | Đề xuất `VAL_REAL_MULTICLASS_PROPOSED` (ảnh / boxes) | `NEEDS_MANUAL_REVIEW` (ảnh) | `REJECT` (ảnh) | Đánh giá DoD |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `1: biological` | `25–35` | **31** | **57** | `22` ảnh (`43` boxes) | `9` ảnh (`14` boxes) | `2` | `24` | **ĐẠT (IN QUOTA)** |
| `6: paper` | `8–12` | **11** | **16** | `8` ảnh (`12` boxes) | `3` ảnh (`4` boxes) | `4` | `9` | **ĐẠT (IN QUOTA)** |
| `2: cardboard` | `8–12` | **10** | **35** | `7` ảnh (`26` boxes) | `3` ảnh (`9` boxes) | `8` | `3` | **ĐẠT (IN QUOTA)** |
| `4: glass` | `8–12` | **12** | **27** | `8` ảnh (`17` boxes) | `4` ảnh (`10` boxes) | `6` | `6` | **ĐẠT (IN QUOTA)** |
| `8: shoes` | `8–12` | **11** | **24** | `7` ảnh (`13` boxes) | `4` ảnh (`11` boxes) | `6` | `9` | **ĐẠT (IN QUOTA)** |
| **TỔNG CỘNG (5 lớp ưu tiên + 2 ứng viên kiểm tra chéo)** | `57–83` | **75** | **159** | **`52` ảnh (`111` boxes)** | **`23` ảnh (`48` boxes)** | **`27`** | **`52`** | **ĐẠT 100% DoD** |

- **Số ảnh `QA_PASS` có từ 2 vật thể trở lên (`boxes >= 2`):** `31 / 75` ảnh (`41.3%`).
- **Số ảnh `QA_PASS` đa lớp (Multi-class co-occurrence trong cùng khung hình):** `5` ảnh (`task8a_044`, `task8a_078`, `task8a_091`, `task8a_115`, `task8a_126`).
- **Số ảnh `QA_PASS` có nền thực tế phức tạp (đường phố, vỉa hè, bãi cỏ, thùng rác mở, sàn nhà, bãi biển):** `58 / 75` ảnh (`77.3%`).

---

## 2. KIỂM TRA TRÙNG LẶP, RÒ RỈ DỮ LIỆU (LEAKAGE AUDIT) & NHÓM PHIÊN CHỤP

- **Độ phủ các tập dữ liệu tham chiếu (100% đầy đủ):**
  - `dataset_v2_and_cache` (Train V2 + Val V2 + Cache): `4046` bản ghi
  - `frozen_test_299` (`data/detection/splits_real_v2/test.txt`): `299` ảnh
  - `roboflow_v3_battery_140_base`: `140` ảnh
  - `task7_diverse_battery_56`: `56` ảnh
  - `batch_a_mini01_10`: `10` ảnh
  - `holdout_27_hinh_anh_rac` (`D:\Hinh_Anh_Rac` SHA-256 + pHash): `27` ảnh
- **Tổng số mã băm đối chiếu:** `2088` SHA-256 và `4112` pHash.
- **Trùng lặp SHA-256 với dữ liệu hiện hữu:** `0` (`zero_external_sha256_leakage = True`).
- **Gần trùng 4-rotation pHash (`Hamming distance <= 8`) với dữ liệu hiện hữu:** `0` (`zero_external_phash_le8_leakage = True`).
- **Rò rỉ chéo giữa `TRAIN_V4_PROPOSED` và `VAL_REAL_MULTICLASS_PROPOSED`:** `0` (`zero_cross_split_leakage = True`).
- **Xử lý chống trùng lặp nội bộ và khóa phiên chụp (`specimen_group`):**
  - Các ảnh cắt cúp / góc chụp trùng lặp sát nhau (`task8a_087` là bản crop của `task8a_088`, `task8a_111` trùng góc chụp với `task8a_110`, `task8a_140` trùng góc với `task8a_139`, `task8a_153` là bản crop của `task8a_152`) đã bị loại bỏ (`REJECT`) để không làm phình ảo số lượng `QA_PASS`.
  - Các cặp ảnh cùng phiên chụp nhưng khác bố cục vật thể (`task8a_086` + `088` thuộc `SPEC_BIO_BANANE_A_SESSION`, `task8a_095` + `096` thuộc `SPEC_BIO_ORANGE_PEEL_FAMARTIN`, `task8a_109` + `110` thuộc `SPEC_PAPER_CANADA150_CUP`) được gom chung vào `TRAIN_V4_PROPOSED`, tuyệt đối không tách sang tập Validation.

---

## 3. DANH SÁCH 6 CONTACT SHEETS PHỤC VỤ PM NGHIỆM THU TRỰC QUAN

1. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_01_qa_pass_biological_part1.jpg` — `1: biological` `QA_PASS` (Ảnh 1..16)
2. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_02_qa_pass_biological_part2.jpg` — `1: biological` `QA_PASS` (Ảnh 17..31)
3. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_03_qa_pass_paper_and_cardboard.jpg` — `6: paper` (11 ảnh) & `2: cardboard` (10 ảnh) `QA_PASS`
4. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_04_qa_pass_glass_and_shoes.jpg` — `4: glass` (12 ảnh) & `8: shoes` (11 ảnh) `QA_PASS`
5. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_05_needs_manual_review.jpg` — `27` ảnh `NEEDS_MANUAL_REVIEW` tách riêng chờ PM quyết định
6. `data/acquisition_staging/task8a_multiclass_batch/contact_sheets/contact_sheet_06_rejected_candidates.jpg` — `52` ảnh `REJECT` (ảnh lịch sử, bản vẽ, tiêu bản hiển vi, ảnh trùng lặp)

---

## 4. CHI TIẾT 75 ẢNH `QA_PASS` (ĐÃ KIỂM ĐỊNH BBOX, LỚP & BẢN QUYỀN)

| Mã ảnh | Lớp chính | Split đề xuất | Specimen Group | Số BBox | Các lớp trong ảnh | Giấy phép | Tác giả | Ghi chú kiểm định trực quan |
|---|---|---|---|---:|---|---|---|---|
| `task8a_001` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_001_ROTTEN_APPLE_HALF` | 1 | `1: biological` | `CC BY-SA 4.0` | Anouar bg | Rotten decaying half-apple with dark fungal core; full fruit body and stem enclosed. |
| `task8a_002` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_002_MUMMIFIED_APPLE` | 1 | `1: biological` | `CC BY 4.0` | Natalka Ukraine (baby-bear.o | Shriveled brown rotten apple against natural foliage background; full fruit body boxed. |
| `task8a_003` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_003_ROTTEN_APPLE_GROUND` | 1 | `1: biological` | `CC BY 4.0` | Natalka Ukraine (baby-bear.o | Rotten apple decaying on soil/dry grass outdoors; full apple body boxed. |
| `task8a_004` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_004_ROTTEN_CASHEWS_STEPS` | 3 | `1: biological` | `CC BY-SA 4.0` | Bukky658 | Three rotten yellow cashew fruits discarded on outdoor concrete steps; each fruit boxed individually. |
| `task8a_009` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_009_MOLDY_LEMON` | 1 | `1: biological` | `CC BY-SA 4.0` | Ali Alaswad | Moldy citrus lemon fruit with white fungal growth; visible fruit body boxed. |
| `task8a_013` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_013_CUT_APPLE_CORE` | 1 | `1: biological` | `CC BY-SA 3.0` | RTG | Knife-cut apple core discarded on printed sheet; full apple core boxed. |
| `task8a_014` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_014_APPLE_CORE_CURB` | 1 | `1: biological` | `CC BY-SA 4.0` | Ildar Sagdejev (Specious) | Half-eaten red apple core discarded on outdoor concrete curb beside wrought-iron fence. |
| `task8a_015` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_015_CODLING_ROTTEN_APPLES` | 6 | `1: biological` | `Public domain` | Richard Wilde | Six worm-damaged/rotting Gravenstein apples and cut halves; each apple boxed individually. |
| `task8a_016` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_016_MOLDY_BREAD` | 1 | `1: biological` | `CC BY-SA 4.0` | Jan Myšák | Heavily moldy slice of bread spoiled with green mold; full slice boxed. |
| `task8a_017` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_017_ROTTEN_TOMATOES_ROPE` | 2 | `1: biological` | `CC BY 2.0` | cogdogblog | Two rotten moldy red tomatoes resting on coiled blue rope; both tomatoes boxed individually. |
| `task8a_018` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_018_ROTTING_STRAWBERRY` | 1 | `1: biological` | `CC BY 3.0 us` | Clemson University | Strawberry with brown fungal fruit rot at base; full fruit boxed. |
| `task8a_019` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_019_BOTRYTIS_STRAWBERRY` | 1 | `1: biological` | `CC BY-SA 3.0` | Rasbak | Rotting strawberry covered in gray Botrytis mold on wooden surface; full fruit boxed. |
| `task8a_023` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_023_CRUMPLED_BALL` | 1 | `6: paper` | `Copyrighted free use` | Nikhilesh Haval | Crumpled white paper ball; full crumpled paper geometry boxed. |
| `task8a_024` | `6: paper` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_PAPER_024_CRUMPLED_BALL` | 1 | `6: paper` | `CC BY-SA 3.0` | Army1987 | Crumpled lined paper ball; full paper ball boxed. |
| `task8a_026` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_026_BIODEGRADABLE_CUP_SOIL` | 1 | `6: paper` | `CC BY 2.0` | Meanwell Packaging | Discarded biodegradable kraft paper coffee cup lying on dark soil/compost outdoors. |
| `task8a_027` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_027_YELLOW_COFFEE_CUP` | 1 | `6: paper` | `CC BY 2.0` | Henry Söderlund | Disposable yellow paper coffee cup with lid on wet dark outdoor surface. |
| `task8a_028` | `6: paper` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_PAPER_028_ALEPA_COFFEE_CUP` | 1 | `6: paper` | `CC BY 2.0` | Henry Söderlund | Discarded Alepa paper coffee cup standing on mossy log/littered ground outdoors. |
| `task8a_038` | `2: cardboard` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_CARDBOARD_038_DAMAGED_FRAGILE_BOX` | 1 | `2: cardboard` | `CC BY 2.0` | Meanwell Packaging | Dented/damaged corrugated cardboard box with red FRAGILE sticker sitting on outdoor stone steps; full box enclosed (not sticker). |
| `task8a_040` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_040_WET_TORN_CARDBOARD` | 1 | `2: cardboard` | `CC0` | Jidanni | Wet, chewed, weathered corrugated cardboard box flap outdoors with visible corrugated inner flutes. |
| `task8a_041` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_041_FIVE_CARTONS_STACK` | 5 | `2: cardboard` | `CC BY-SA 4.0` | Peachyeung316 | Stack of five corrugated cardboard cartons on wooden pallet; each carton boxed individually. |
| `task8a_043` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_043_THREE_BOXES_CART` | 3 | `2: cardboard` | `CC0` | LYUHOI Tima LUNGP | Three corrugated cardboard shipping boxes of different sizes stacked on a blue hand cart; each box annotated separately. |
| `task8a_044` | `2: cardboard` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_CARDBOARD_044_OPEN_BOX_SIDEWALK` | 2 | `2: cardboard, 7: plastic` | `CC BY 4.0` | Ka23 13 | Open corrugated cardboard box on sidewalk in Matsudo with plastic bottle inside; multi-class annotation (2: cardboard + 7: plastic). |
| `task8a_050` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_050_BROKEN_CLEAR_BOTTLE` | 2 | `4: glass` | `CC BY-SA 4.0` | Ildar Sagdejev (Specious) | Broken clear glass bottle shattered into two main pieces on outdoor textured brick pavement; both pieces boxed. |
| `task8a_053` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_053_PRESERVING_JAR` | 1 | `4: glass` | `CC BY-SA 4.0` | Franz van Duns | Empty clear glass preserving jar below wooden opener tool; full glass jar body boxed. |
| `task8a_059` | `8: shoes` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_SHOES_059_NEWPORT_LITTER_BIN` | 5 | `8: shoes` | `CC BY 3.0` | Editor5807 | Discarded boots on top of a street litter bin and two pairs of discarded shoes on the pavement around the bin in Newport High Street. |
| `task8a_061` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_061_FOUR_BLACK_SHOES_CURB` | 4 | `8: shoes` | `CC BY-SA 2.0` | Michael Coghlan from Adelaid | Four discarded black leather shoes lined up on outdoor pavement/grass border; each shoe boxed individually. |
| `task8a_062` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_062_MUDDY_SNOW_BOOTS` | 2 | `8: shoes` | `CC BY-SA 3.0` | Andrei.computer | Pair of muddy worn snow boots lying on tiled floor; both boots boxed individually. |
| `task8a_065` | `8: shoes` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_SHOES_065_ABANDONED_SHOE_MUD` | 1 | `8: shoes` | `CC BY-SA 2.0` | Lee Haywood from Wollaton, N | Single old dirty leather shoe abandoned on muddy ground and twigs with dirt inside. |
| `task8a_069` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_069_BROWN_BANANA_PEEL` | 1 | `1: biological` | `CC BY-SA 3.0` | Anitamahotra | Spotted brown banana peel strips; full peel structure boxed. |
| `task8a_070` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_070_BANANA_PEEL_MAISELS` | 1 | `1: biological` | `CC BY 4.0` | Kenneth C. Zirkel | Small discarded banana peel lying on concrete sidewalk mosaic ('MAISEL'S'); tests small-scale outdoor biological detection. |
| `task8a_071` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_071_BANANA_PEEL_ASPHALT` | 1 | `1: biological` | `CC BY-SA 4.0` | Vladiator360 | Discarded yellow banana peel lying on wet asphalt sidewalk with pedestrians in background. |
| `task8a_072` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_072_YELAKKI_BANANA_PEEL` | 1 | `1: biological` | `CC BY 4.0` | Mohanramus | Yelakki banana peel discarded on dark speckled granite counter; full peel including stem boxed. |
| `task8a_077` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_077_ROTTEN_APPLE_LEAVES` | 1 | `1: biological` | `CC BY-SA 4.0` | Sally V | Brown decaying rotten apple partially eaten on dead leaves under tree. |
| `task8a_078` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_078_ROTTEN_PEACHES_PUNNET` | 5 | `1: biological, 7: plastic` | `CC BY-SA 4.0` | Springnuts | Four moldy rotting peaches inside a clear plastic punnet container; multi-class annotation (4x 1:biological + 1x 7:plastic). |
| `task8a_081` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_081_ROTTEN_PEAR_GRASS` | 1 | `1: biological` | `CC BY-SA 4.0` | Jerzy Opioła | Rotten browning half-pear lying in green grass outdoors; full pear body and stem boxed. |
| `task8a_082` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_082_MOLDY_TOMATOES` | 5 | `1: biological` | `CC BY-SA 4.0` | JIP | Moldy spoiled red tomatoes with white fungal spots inside box; 5 distinct moldy tomatoes boxed. |
| `task8a_083` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_083_DISPOSABLE_CUP_VS_MUG` | 1 | `6: paper` | `CC BY-SA 4.0` | JIP | Disposable paper coffee cup (left, boxed as 6:paper) next to reusable red ceramic mug (right, intentionally unboxed hard negative context). |
| `task8a_084` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_084_BLACKENED_BANANA_SKIN` | 1 | `1: biological` | `CC0` | Simon Speed | Overripe blackened banana peel skin; full peel and stem boxed. |
| `task8a_086` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_BANANE_A_SESSION` | 1 | `1: biological` | `Public domain` | Priwo | Single peeled yellow banana skin; grouped in SPEC_BIO_BANANE_A_SESSION with task8a_088 in TRAIN_V4_PROPOSED. |
| `task8a_088` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_BANANE_A_SESSION` | 2 | `1: biological` | `Public domain` | Priwo | Two overlapping banana peels from same session as task8a_086; both peels boxed individually and grouped in TRAIN_V4_PROPOSED. |
| `task8a_089` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_089_RED_APPLE_CORE` | 1 | `1: biological` | `CC BY-SA 4.0` | Famartin | Eaten red apple core standing upright; full stem, core, and base boxed. |
| `task8a_090` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_090_APFELGRIEBSCH` | 1 | `1: biological` | `CC BY 3.0` | Z thomas | Eaten yellow-red apple core; full stem, core, and base boxed. |
| `task8a_091` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_091_MULTI_TRASH_APPLE_CORES` | 6 | `1: biological, 6: paper, 7: plastic` | `CC BY 2.0` | Weldon Kennedy from London,  | Real open waste bin containing multiple eaten apple cores (1:biological), a white disposable paper cup (6:paper), and a plastic bag wrapper (7:plastic). |
| `task8a_092` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_092_STARK_APPLE_CORE` | 1 | `1: biological` | `CC BY-SA 3.0` | Roberta F. | Eaten dark-red Stark apple core; full core and stem boxed. |
| `task8a_095` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_ORANGE_PEEL_FAMARTIN` | 2 | `1: biological` | `CC BY-SA 4.0` | Famartin | Two discarded orange peel shells on table; both peel halves boxed and grouped in SPEC_BIO_ORANGE_PEEL_FAMARTIN with task8a_096 in TRAIN_V4_PROPOSED. |
| `task8a_096` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_ORANGE_PEEL_FAMARTIN` | 1 | `1: biological` | `CC BY-SA 4.0` | Famartin | Torn orange peel pieces showing white pith from same session as task8a_095; grouped in TRAIN_V4_PROPOSED. |
| `task8a_097` | `1: biological` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_BIO_097_EGGSHELL_FOREST` | 1 | `1: biological` | `CC BY-SA 4.0` | Roy Zuo | Broken eggshell lying on outdoor moss and forest litter in Finkenpark Berlin. |
| `task8a_104` | `1: biological` | `TRAIN_V4_PROPOSED` | `SPEC_BIO_104_BIOWASTE_BIN_MIX` | 4 | `1: biological` | `CC BY-SA 4.0` | Achim Raschka | Real household biowaste bin containing banana peel, lemon rind, tomato top, and eggshell. |
| `task8a_106` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_106_CRUMPLED_OLIVE_SHEET` | 1 | `6: paper` | `Public domain` | Photos Public Domain | Unfolded crumpled olive-green paper sheet with dense crease texture. |
| `task8a_109` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_CANADA150_CUP` | 1 | `6: paper` | `CC BY-SA 2.0` | Coastal Elite from Halifax,  | Disposable Tim Hortons paper coffee cup with plastic lid; grouped with task8a_110 in TRAIN_V4_PROPOSED. |
| `task8a_110` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_CANADA150_CUP` | 2 | `6: paper` | `CC BY-SA 2.0` | Coastal Elite from Halifax,  | Disposable paper coffee cup and detached paper sleeve on table; grouped with task8a_109 in TRAIN_V4_PROPOSED. |
| `task8a_112` | `6: paper` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_PAPER_112_TWO_SECONDCUP_CUPS` | 2 | `6: paper` | `CC BY-SA 2.0` | Coastal Elite from Halifax,  | Two disposable Second Cup paper coffee cups on wooden table; both cups boxed individually. |
| `task8a_113` | `6: paper` | `TRAIN_V4_PROPOSED` | `SPEC_PAPER_113_AIRFRANCE_CUPS_NAPKIN` | 4 | `6: paper` | `CC BY-SA 4.0` | Simply Aviation | Two disposable paper cups, white paper napkin, and paper sugar/cutlery packet on tray. |
| `task8a_115` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_115_PIZZA_BOX_PLASTIC_TUBS` | 3 | `2: cardboard, 7: plastic` | `CC BY 2.0` | Government of Thailand | Corrugated cardboard pizza takeaway box beneath two empty clear plastic food containers; multi-class annotation (2:cardboard + 2x 7:plastic). |
| `task8a_117` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_117_PARK_SLOPE_PIZZA_BOXES` | 5 | `2: cardboard` | `CC BY-SA 4.0` | Wil540 | Stack of five used corrugated cardboard pizza boxes on table; each box annotated individually. |
| `task8a_118` | `2: cardboard` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_CARDBOARD_118_SIX_PIZZA_BOXES` | 6 | `2: cardboard` | `CC BY-SA 4.0` | Martin Rulsch, Wikimedia Com | Stack of six corrugated cardboard pizza boxes on counter; each box annotated individually. |
| `task8a_120` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_120_BANANA_BOXES_STACK` | 8 | `2: cardboard` | `CC0` | Stefan Kühn | Two stacks of heavy-duty corrugated cardboard banana cartons (8 boxes boxed individually). |
| `task8a_121` | `2: cardboard` | `TRAIN_V4_PROPOSED` | `SPEC_CARDBOARD_121_ALLEY_BANANA_BOX` | 1 | `2: cardboard` | `CC BY 3.0` | B Konti | Discarded Bonita corrugated cardboard box stacked atop wooden pallets in outdoor stone alley. |
| `task8a_124` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_124_MYUNA_CREEK_SHARD` | 1 | `4: glass` | `CC BY 3.0` | Sardaka (talk) 07:57, 15 May | Thick broken glass bottle shard with embossed lettering discarded at Myuna Creek. |
| `task8a_126` | `4: glass` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_GLASS_126_FIVE_BOTTLES_TABLE` | 6 | `4: glass, 6: paper` | `CC BY 3.0` | dubrovka.911 | Five empty green glass bottles (1 standing, 4 lying) and a red paper cigarette pack left on an outdoor wooden picnic table in forest. |
| `task8a_127` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_127_BROKEN_BEER_BOTTLES_PAVEMENT` | 4 | `4: glass` | `CC BY-SA 4.0` | Raimond Spekking | Two smashed brown glass beer bottles ('Sonnen-Hopfen') on cobblestone pavement with weeds; 4 major bottle bodies/necks boxed. |
| `task8a_130` | `4: glass` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_GLASS_130_BROKEN_GREEN_BOTTLE_BEACH` | 1 | `4: glass` | `CC BY-SA 2.0` | まじぷり | Broken green glass bottle neck/shoulder half-buried in coastal beach sand. |
| `task8a_131` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_131_BROKEN_FLENSBURGER_BOTTLE` | 1 | `4: glass` | `CC BY-SA 2.0` | marsupium photography | Broken brown glass beer bottle top with metal/porcelain swing-top cap lying on soil. |
| `task8a_132` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_132_BROKEN_JARRITOS_FLOOR` | 2 | `4: glass` | `CC BY-SA 4.0` | JIP | Broken Jarritos glass soda bottle pieces in spilled liquid on tiled floor. |
| `task8a_133` | `4: glass` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_GLASS_133_BROKEN_VANA_TALLINN` | 2 | `4: glass` | `CC BY-SA 4.0` | JIP | Smashed brown glass liqueur bottle and broken neck piece in spilled puddle on terrazzo floor. |
| `task8a_139` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_139_ABANDONED_PINK_PAIR` | 2 | `8: shoes` | `CC BY-SA 4.0` | Illustratedjc | Pair of abandoned worn pink peep-toe shoes on outdoor concrete wall ledge; both shoes boxed individually. |
| `task8a_141` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_141_BEACH_ABANDONED_SANDAL` | 1 | `8: shoes` | `CC BY-SA 4.0` | TheArchivistMan | Weathered, sand-encrusted abandoned foam sandal washed up on beach sand. |
| `task8a_144` | `8: shoes` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_SHOES_144_LONDON_PAVEMENT_SHOE` | 1 | `8: shoes` | `CC BY-SA 2.0` | Lloyd Davis | Single dirty blue flat shoe abandoned on stone pavement slabs among fallen leaves in London. |
| `task8a_145` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_145_WORN_BROWN_WINGTIPS` | 2 | `8: shoes` | `CC BY 2.0` | cogdogblog | Pair of heavily scuffed worn-out brown suede/leather shoes; both shoes boxed individually. |
| `task8a_146` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_146_WADI_RUM_TORN_BOOT` | 1 | `8: shoes` | `CC BY-SA 2.0` | Grand Parc - Bordeaux, Franc | Torn, sun-bleached old ankle boot abandoned on desert sand and dry grass in Wadi Rum. |
| `task8a_149` | `8: shoes` | `TRAIN_V4_PROPOSED` | `SPEC_SHOES_149_DIRTY_BLUNDSTONE_BOOT` | 1 | `8: shoes` | `CC BY 3.0` | Best For My Feet | Dirty mud-splattered worn Blundstone work boot; full boot profile boxed. |
| `task8a_150` | `8: shoes` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_SHOES_150_FOUR_DIRTY_CONVERSE` | 4 | `8: shoes` | `CC BY 2.0` | Anathea Utley from lisbon, m | Two pairs (4 shoes) of dirty worn yellow and green Converse canvas sneakers on coarse doormat; all 4 shoes boxed individually. |
| `task8a_151` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_151_CLEAR_EMPTY_BOTTLE_RAIL` | 1 | `4: glass` | `CC BY-SA 4.0` | Santeri Viinamäki | Empty clear glass bottle standing on weathered outdoor wooden rail with grass background. |
| `task8a_152` | `4: glass` | `TRAIN_V4_PROPOSED` | `SPEC_GLASS_152_FIVE_PAOLYTA_BOTTLES` | 5 | `4: glass` | `CC BY-SA 4.0` | Solomon203 | Five empty brown Paolyta-B glass bottles lined up outdoors in front of shutter door; all 5 glass bottles boxed individually. |
| `task8a_154` | `4: glass` | `VAL_REAL_MULTICLASS_PROPOSED` | `SPEC_GLASS_154_WHISBIH_BOTTLE_STEPS` | 1 | `4: glass` | `CC BY-SA 4.0` | Solomon203 | Empty brown Whisbih 600cc glass bottle discarded horizontally on rough pebble-concrete steps. |

---

## 5. CHI TIẾT 27 ẢNH `NEEDS_MANUAL_REVIEW` (GIỮ RIÊNG CHỜ PM PHÁN QUYẾT)

| Mã ảnh | Lớp dự kiến | Số BBox gợi ý | Lý do giữ tại `NEEDS_MANUAL_REVIEW` |
|---|---|---:|---|
| `task8a_029` | `6: paper` | 1 | Wicker wastebasket overflowing with tangled coiled paper strips; held for PM review on whether to box whole tangled mass or individual coils. |
| `task8a_034` | `2: cardboard` | 1 | Abandoned room filled with dozens of overlapping torn cardboard boxes and debris; requires PM crowd/pile segmentation review. |
| `task8a_035` | `2: cardboard` | 2 | Two large tied bales of hundreds of flattened cardboard sheets on Jerusalem street; held for PM review on bale vs individual sheet boxing. |
| `task8a_036` | `2: cardboard` | 1 | Compressed industrial bale of flattened corrugated cardboard; held for PM review on bale-level annotation policy. |
| `task8a_037` | `2: cardboard` | 1 | Warehouse worker operating industrial cardboard baler with blurred foreground cardboard stack; held for PM review. |
| `task8a_039` | `2: cardboard` | 1 | Massive outdoor heap of hundreds of overlapping cardboard boxes ('Cimetiere de carton'); requires crowd/pile policy review. |
| `task8a_042` | `2: cardboard` | 1 | Molded-pulp cardboard egg carton filled with 12 intact fresh uncooked eggs; held for PM review so intact fresh food is not mixed with discarded waste. |
| `task8a_045` | `2: cardboard` | 1 | Worker pulling compressed cardboard bale on pallet jack; held with task8a_037 for bale policy review. |
| `task8a_051` | `4: glass` | 1 | Dense ground layer of thousands of tiny broken glass cullet fragments; cannot isolate individual bottles. |
| `task8a_052` | `4: glass` | 1 | Broken glass shards embedded into mortar on top of security wall; held for PM review as architectural fixture vs waste. |
| `task8a_064` | `8: shoes` | 3 | Muddy boots actively worn on people's feet standing in festival mud; held for PM review as worn footwear vs discarded waste. |
| `task8a_075` | `1: biological` | 1 | Small apple core piece on soil next to large Burchell's starling bird; held for PM review due to bird dominating frame. |
| `task8a_079` | `1: biological` | 1 | Dense overlapping heap of 60+ rotten oranges filling entire frame; held for PM crowd/pile annotation review. |
| `task8a_085` | `6: paper` | 4 | Wet marathon street littered with 40+ crushed paper cups and banana peels among runners; held for PM dense street-litter annotation review. |
| `task8a_105` | `6: paper` | 1 | Dense heap of 50+ overlapping crumpled white paper balls with Arabic watermark text in top-left corner; held for PM review. |
| `task8a_114` | `6: paper` | 1 | Paper cup filled with fresh rolled ice cream held by vendor; held for PM review as food-in-hand vs discarded waste. |
| `task8a_116` | `2: cardboard` | 3 | Three open cardboard pizza boxes containing fresh edible pizzas at event; held in NEEDS_MANUAL_REVIEW per PM rule not to label clean served food as biological waste. |
| `task8a_125` | `4: glass` | 2 | Wide plaza view after parade with tiny distant broken bottles/cans (<15px); held for PM review due to sub-resolution object scale. |
| `task8a_128` | `4: glass` | 1 | Hundreds of broken glass bottle necks and shards washed up on beach shoreline; held for PM crowd/pile annotation review. |
| `task8a_129` | `4: glass` | 1 | Second angle of dense broken glass bottle deposit on beach shoreline; held in NEEDS_MANUAL_REVIEW with task8a_128. |
| `task8a_134` | `4: glass` | 1 | Over 100 glass wine/beer bottles and plastic bags piled around street bottle-bank container; held for PM dense crowd review. |
| `task8a_136` | `7: plastic` | 1 | Fetched from Wikimedia Category:Empty bottles, but visual QA verified it is a clear PET plastic bottle (7:plastic), NOT glass (4:glass). Held in NEEDS_MANUAL_REVIEW per PM Section 4 glass-vs-plastic rule. |
| `task8a_138` | `8: shoes` | 1 | Wide night street scene with very small sandal in foreground and pedestrians wearing shoes in background; held for PM review. |
| `task8a_142` | `8: shoes` | 1 | Cluster of 15+ shoes tied together and flung over overhead utility wires against sky; held for PM review. |
| `task8a_143` | `8: shoes` | 2 | Over 25 overlapping abandoned shoes piled along brick wall on London sidewalk; held for PM dense-pile annotation review. |
| `task8a_147` | `8: shoes` | 1 | Extreme close-up crop of frayed sneaker shoelaces only, cutting off toe and heel geometry; held in NEEDS_MANUAL_REVIEW. |
| `task8a_148` | `8: shoes` | 1 | Roadside rack displaying 30+ stacked second-hand shoes in Baliuag; held for PM dense-rack annotation review. |
