# BÁO CÁO XÁC MINH NGUỒN DỮ LIỆU & ĐỀ XUẤT MINI-BATCH 01 (TASK 3)
**Sprint:** SPRINT AI-UPGRADE-01 — Task 3: Real Training Data Preparation  
**Dự án:** Hệ thống Nhận diện Rác thải 10 Lớp (CNTT-KLCN155)  
**Thời điểm thực hiện:** 2026-10-12T15:55:00Z  
**Tác giả:** Anti (AI Engineer) | **Kính gửi:** ChatGPT (PM / Reviewer) & Người dùng  

---

## I. TỔNG HỢP KẾT QUẢ XÁC MINH QUA WIKIMEDIA COMMONS API

Toàn bộ 25 ứng viên trong manifest [batch_a_candidate_manifest.json](../../data/acquisition_staging/manifests/batch_a_candidate_manifest.json) đã được truy vấn trực tiếp qua API chính thức của Wikimedia Commons (`https://commons.wikimedia.org/w/api.php?action=query&prop=imageinfo&iiprop=url|size|mime|extmetadata`). Bằng chứng JSON thô được lưu tại [wikimedia_raw_api_responses.json](../../reports/source_verification/wikimedia_raw_api_responses.json).

### 1. Số liệu tổng quan:
- **TỔNG SỐ ỨNG VIÊN QUÉT:** 25
- **SOURCES VERIFIED (Hợp lệ, có metadata thật):** **16**
- **SOURCES UNVERIFIED (Không tìm thấy / 404):** **9**
- **DOMAIN REVIEW REQUIRED (Cần rà soát nghiệp vụ):** **1** (`wiki_neg_street_cabinet_01` — ảnh chụp thiết bị khuếch đại bên trong tủ, không phải ngoại thất tủ điện hè phố)
- **NEW IMAGES DOWNLOADED TRONG LƯỢT NÀY:** **0**
- **TRAIN APPROVED:** **0** (Tuyệt đối không tự ý nạp vào train)

---

## II. DANH SÁCH 10 ẢNH ĐƯỢC ĐỀ XUẤT TẢI (MINI-BATCH 01)
Được chọn lọc nghiêm ngặt từ các nguồn **VERIFIED**, đại diện cho các ca bệnh quan sát khi demo (Pin, Rác hữu cơ, Hard-Negative sạch, Đa rác hỗn hợp):

| STT | Candidate ID | Target Class | Trang nguồn Canonical (Wikimedia) | Original Image Download URL | Author | License | File Size (Bytes) | Kích thước (MiB) | Resolution | Verification Status |
|:---:|---|:---:|---|---|---|---|---:|---:|:---:|:---:|
| 1 | `wiki_bat_corroded_screwdriver` | `battery` | [Commons Page](https://commons.wikimedia.org/wiki/File:Corroded_Screwdriver_Battery.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/3/38/Corroded_Screwdriver_Battery.jpg) | Eoin | CC BY-SA 4.0 | 536.994 | 0.51 MiB | 949x1600 | **VERIFIED** |
| 2 | `wiki_bio_apple_core_01` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:Apple_core.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/4/48/Apple_core.jpg) | Adrian Zion | Copyrighted free use | 130.620 | 0.12 MiB | 960x1280 | **VERIFIED** |
| 3 | `wiki_bio_banana_peel_bucharest_02` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:Banana_peel.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/1/1d/Banana_peel.jpg) | Tiia Monto | CC BY-SA 4.0 | 4.604.017 | 4.39 MiB | 3152x2856 | **VERIFIED** |
| 4 | `wiki_bio_banana_peel_sidewalk_06` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:2020-03-15_01_55_05_A_single_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill,_Fairfax_County,_Virginia.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/4/4f/2020-03-15_01_55_05_A_single_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg) | Famartin | CC BY-SA 4.0 | 1.208.913 | 1.15 MiB | 4032x3024 | **VERIFIED** |
| 5 | `wiki_bio_banana_peel_bus_stop_03` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:Banana_peel_on_seat.JPG) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/9/9f/Banana_peel_on_seat.JPG) | Lemur12 | CC BY-SA 3.0 | 2.293.236 | 2.19 MiB | 3264x2448 | **VERIFIED** |
| 6 | `wiki_bio_banana_peel_oak_hill_05` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:2019-02-07_13_32_24_A_single_Chiquita_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill,_Fairfax_County,_Virginia.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/4/40/2019-02-07_13_32_24_A_single_Chiquita_banana_peel_in_the_Franklin_Farm_section_of_Oak_Hill%2C_Fairfax_County%2C_Virginia.jpg) | Famartin | CC BY-SA 4.0 | 2.933.165 | 2.80 MiB | 4032x3024 | **VERIFIED** |
| 7 | `wiki_bio_food_scraps_bin_09` | `biological` | [Commons Page](https://commons.wikimedia.org/wiki/File:Food_Scraps_and_Yard_Debris_Collection_in_Portland_2010_by_Tim_Jewett.JPG) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/6/6a/Food_Scraps_and_Yard_Debris_Collection_in_Portland_2010_by_Tim_Jewett.JPG) | Tim Jewett | CC BY-SA 2.5 | 4.057.650 | 3.87 MiB | 2000x3000 | **VERIFIED** |
| 8 | `wiki_neg_utility_box_corner_05` | `hard_negatives` | [Commons Page](https://commons.wikimedia.org/wiki/File:Utility_box_on_the_corner_of_Wesley_Street_and_Town_Street,_Beeston,_Leeds_(11th_August_2023).jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/b/b7/Utility_box_on_the_corner_of_Wesley_Street_and_Town_Street%2C_Beeston%2C_Leeds_%2811th_August_2023%29.jpg) | Mtaylor848 | CC BY-SA 4.0 | 6.131.636 | 5.85 MiB | 5152x3864 | **VERIFIED** |
| 9 | `wiki_neg_brick_wall_clean` | `hard_negatives` | [Commons Page](https://commons.wikimedia.org/wiki/File:Brick_wall_texture.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/6/63/Brick_wall_texture.jpg) | Georges Grondin | Public domain | 1.668.359 | 1.59 MiB | 3264x2448 | **VERIFIED** |
| 10 | `wiki_multi_banana_and_cup_01` | `multi_waste` | [Commons Page](https://commons.wikimedia.org/wiki/File:Unbinned_litter_at_Sainsbury%27s,_Chingford,_London,_England_1.jpg) | [Direct URL](https://upload.wikimedia.org/wikipedia/commons/a/a3/Unbinned_litter_at_Sainsbury%27s%2C_Chingford%2C_London%2C_England_1.jpg) | Acabashi | CC BY-SA 4.0 | 14.942.093 | 14.25 MiB | 4923x3692 | **VERIFIED** |

### Dự toán dung lượng Mini-Batch 01:
- **Tổng số ảnh:** 10 ảnh
- **Tổng dung lượng chính xác:** **38.506.683 bytes** ($\approx \mathbf{36.72\text{ MiB}}$)
- **Giới hạn an toàn:** Dưới ngưỡng $50\text{ MiB}$ theo yêu cầu của PM.

---

## III. TÌNH TRẠNG 4 ẢNH ĐÃ CÓ TRÊN ĐĨA (PHASE 4E)

| Candidate ID | Tệp đĩa | Dung lượng | QA Status | Ghi chú & Phán quyết |
|---|---|---:|:---:|---|
| `wiki_bio_banana_peel_ground` | `data/acquisition_staging/phase4e_staging/images/` | 4.573.954 B | `VISUALLY_REVIEWED` | Bbox snug `[2100, 1280, 5184, 2300]`, chạm mép phải. Chờ duyệt. |
| `wiki_neg_junction_boxes` | `data/acquisition_staging/phase4e_staging/images/` | 110.051 B | `VISUALLY_REVIEWED` | 100% full-frame sạch rác. Tệp nhãn 0-byte. Chờ duyệt. |
| `wiki_neg_utility_box_leeds` | `data/acquisition_staging/phase4e_staging/images/` | 5.736.159 B | `PENDING_OBJECT_IDENTITY_REVIEW` | Nghi vấn dị vật chân tủ. **QUARANTINE, không tự coi là sạch.** |
| `wiki_bio_street_food_remains` | `data/acquisition_staging/phase4e_staging/images/` | 294.623 B | `REJECTED_AS_BIOLOGICAL_NEEDS_PER_OBJECT_BBOX` | Hộp xốp chiếm đa số. **Không gán biological trùm lên hộp xốp.** |

---

## IV. BẢNG 9 NGUỒN CHƯA XÁC MINH (SOURCES UNVERIFIED / 404)
Các tệp này do tên file phỏng đoán không tồn tại trên Wikimedia Commons API (Page ID = -1). Đã đánh dấu `SOURCE_UNVERIFIED` và `BLOCKED`, **tuyệt đối không tự ý phát minh URL hay thay thế ngoài whitelist**:

1. `wiki_neg_clean_asphalt_pavement`: `File:Asphalt_road_surface.jpg` (404)
2. `wiki_multi_cans_and_bottles_pavement`: `File:Litter_in_the_street_-_geograph.org.uk_-_1746200.jpg` (404)
3. `wiki_multi_cardboard_can_litter`: `File:Litter_on_grass_verge.jpg` (404)
4. `wiki_multi_bottles_and_cans_curb`: `File:Litter_on_street_corner.jpg` (404)
5. `wiki_bat_leaking_alkaline`: `File:Leaking_alkaline_battery.jpg` (404)
6. `wiki_bat_swollen_lipo`: `File:Puffed_Lithium_Polymer_Battery.jpg` (404)
7. `wiki_bat_old_car_battery_dumped`: `File:Old_car_battery_dumped.jpg` (404)
8. `wiki_bat_assorted_recycle_bin`: `File:Battery_recycling_box.jpg` (404)
9. `wiki_bat_discarded_dry_cells_ground`: `File:Discarded_batteries_on_ground.jpg` (404)

---

## V. ĐIỀU KIỆN TIẾP TỤC & YÊU CẦU PHÊ DUYỆT (TASK 5)

Hệ thống **CHƯA TẢI BẤT KỲ ẢNH NÀO** trong lượt này. Kính trình PM và Người dùng xem xét phê duyệt:
- **Phê duyệt phạm vi tải Mini-Batch 01 (10 ảnh, dung lượng 36.72 MiB)** theo đúng danh sách Mục II.
- Sau khi được phê duyệt, hệ thống sẽ thực thi tải về, tính SHA-256/pHash, quét rò rỉ và xuất contact sheet kèm bbox để nghiệm thu trực quan.
