# BÁO CÁO TASK 7 — THU THẬP DỮ LIỆU PIN ĐA DẠNG NGOÀI ROBOFLOW, QA TRỰC QUAN & ĐỀ XUẤT DATASET V4 / E4 (DO NOT RUN)

- **Dự án:** `D:\CNTT-KLCN155-waste-detection`
- **Sprint:** `AI-UPGRADE-01 — TASK 6B & TASK 7`
- **Thư mục Staging:** `data/acquisition_staging/task7_diverse_battery_batch/`
- **Trạng thái phê duyệt huấn luyện:** **`APPROVED_FOR_TRAIN = 0`** *(Tất cả dữ liệu và cấu hình ở trạng thái đề xuất chờ PM nghiệm thu; tuyệt đối chưa chạy huấn luyện E4)*

---

## 1. TỔNG QUAN CHIẾN LƯỢC THU THẬP DỮ LIỆU (PHẦN B1)

Từ kết quả thực nghiệm của Pilot E3 ở Task 6:
- Tập 140 ảnh Roboflow (`111 train + 29 val`) giúp E3 đạt `mAP50 = 0.99235` trên cùng domain bàn gỗ đơn chiếc (`1 bbox/ảnh`).
- Tuy nhiên, trên tập Holdout thực tế (`D:\Hinh_Anh_Rac`, ảnh `IMG_2608.JPG` gồm cụm 4 viên pin chụp góc nghiêng), E3 chỉ phát hiện được `1/4` viên pin ở ngưỡng thấp và tăng False Positive trên các lớp khác.
- Nguyên nhân cốt lõi: **Domain shift nghiêm trọng** (100% ảnh pin bổ sung ở V3 là ảnh 1 viên pin đơn lẻ trên nền bàn phẳng, thiếu hoàn toàn cụm nhiều viên pin đứng/nằm chồng lấn, pin cúc áo, pin 9V vuông, pin Li-ion/18650/pin điện thoại, pin rỉ sét/chảy hóa chất và mẫu âm tính khó dạng trụ/thiết bị điện).

### Nguồn thu thập độc lập & Bản quyền mở
- Toàn bộ **56 ảnh ứng viên (`task7_01` .. `task7_56`)** được truy vấn và tải trực tiếp từ **Wikimedia Commons API** với giấy phép mở hợp lệ (`CC0`, `Public Domain`, `CC BY 3.0/4.0`, `CC BY-SA 2.5/3.0/4.0`).
- Mỗi tệp ảnh đều được ghi nhận đầy đủ: URL tải gốc (`image_url`), trang mô tả (`description_url`), tác giả (`artist`), giấy phép (`license_short`), kích thước ảnh gốc, mã băm `SHA-256` và `pHash` 4 góc xoay (`0°, 90°, 180°, 270°`) trong `raw_download_manifest.json` và `task7_qa_manifest.json`.

| Chỉ số tổng hợp Batch Task 7 | Giá trị |
|---|---|
| Tổng số ảnh tải về và kiểm tra (`task7_01`..`task7_56`) | **56** ảnh |
| Số ảnh đạt chuẩn **`QA_PASS`** | **30** ảnh (**26** ảnh có pin + **4** ảnh hard negative) |
| Tổng số bounding box `0: battery` đạt chuẩn **`QA_PASS`** | **61** full-object boxes |
| Số ảnh `QA_PASS` chứa **cụm nhiều viên pin (`>= 2 boxes/ảnh`)** | **14** / 26 ảnh có pin (**53.8%**) |
| Đề xuất đưa vào **`TRAIN_V4`** | **19** ảnh (**40** battery boxes, gồm 16 ảnh dương tính + 3 ảnh hard negative) |
| Đề xuất đưa vào **`VAL_REAL_BATTERY`** (tập Val độc lập ngoài Roboflow) | **11** ảnh (**21** battery boxes, gồm 10 ảnh dương tính + 1 ảnh hard negative) |
| Số ảnh **`NEEDS_MANUAL_REVIEW`** (giữ riêng chờ PM quyết định) | **10** ảnh (20 candidate boxes trong `labels_needs_manual_review/`) |
| Số ảnh **`REJECT`** (loại bỏ hoàn toàn) | **16** ảnh (0 boxes) |
| **`APPROVED_FOR_TRAIN`** | **`0`** |

---

## 2. KẾT QUẢ KIỂM TRA TRỰC QUAN & FULL-OBJECT RE-ANNOTATION (PHẦN B2)

### Quy tắc gán nhãn nghiêm ngặt đã áp dụng
1. **100% Full-Object Bounding Box:** Mỗi bounding box bao trọn toàn bộ thân vật lý của từng viên pin (từ cực dương đến đáy âm, bao gồm cả vỏ kim loại/nhựa và phần muối rỉ sét bám quanh thân). Tuyệt đối không khoanh chữ, logo hay nhãn hiệu.
2. **Tách rời từng viên pin trong cụm (`multi-battery cluster`):** Trong các ảnh có 2, 3, 4 hoặc 6 viên pin đứng/nằm cạnh nhau hoặc bắt chéo nhau, mỗi viên pin được gán 1 bounding box riêng biệt.
3. **Không gộp cả đống hoặc cả hộp bán lẻ:** Các ảnh vỉ nhựa phản quang, hộp 48 viên, hoặc đống >40 viên pin chồng lấp nặng được phân loại vào `NEEDS_MANUAL_REVIEW`, không đưa vào `QA_PASS`.

### 2.1. Danh sách 30 ảnh đạt chuẩn `QA_PASS` (61 Full-Battery Boxes)

| ID | Tệp ảnh | Nhóm đa dạng | Kích thước | Số Box | Split đề xuất | Specimen Group | Giấy phép | Nhận xét Visual QA |
|---|---|---|---|---:|---|---|---|---|
| `task7_08` | `task7_08_button_coin_cell_coin_cell_cr2032_h_by_hitachi_maxell.jpg` | `button_coin_cell` | 1280x1280 | **1** | `TRAIN_V4` | `SPEC_08_MAXELL_CR2032` | `CC BY-SA 4.0` | Clear Hitachi Maxell CR2032 3V coin cell; full circular body boxed. |
| `task7_09` | `task7_09_button_coin_cell_ml2032_rechargeable_button_cell_batt.jpg` | `button_coin_cell` | 1280x1280 | **1** | `VAL_REAL_BATTERY` | `SPEC_09_MAXELL_ML2032` | `CC BY-SA 4.0` | Clear Maxell ML2032 rechargeable button cell; full circular body boxed. |
| `task7_11` | `task7_11_nine_volt_battery_9v_bateri_ujian_short_circuit_png.jpg` | `nine_volt_battery` | 635x308 | **1** | `TRAIN_V4` | `SPEC_11_9V_SHORT_TEST` | `CC BY-SA 4.0` | Prismatic 9V battery lying horizontally with yellow wire; full 9V body & terminals boxed. |
| `task7_12` | `task7_12_nine_volt_battery_9v_pp3_battery_1530632_jpg.jpg` | `nine_volt_battery` | 1280x960 | **1** | `TRAIN_V4` | `SPEC_12_13_MOTOMA_9V_WOOD` | `CC BY-SA 3.0` | Motoma 9V 6F22 battery side view on wooden surface; grouped in TRAIN_V4 with task7_13. |
| `task7_13` | `task7_13_nine_volt_battery_9v_pp3_battery_1530633_jpg.jpg` | `nine_volt_battery` | 1280x1707 | **1** | `TRAIN_V4` | `SPEC_12_13_MOTOMA_9V_WOOD` | `CC BY-SA 3.0` | Top-down terminal view of the same Motoma 9V battery on wooden surface; grouped in TRAIN_V4 with task7_12. |
| `task7_14` | `task7_14_nine_volt_battery_lithium_9v_battery_6f22_jpg.jpg` | `nine_volt_battery` | 1576x2520 | **1** | `VAL_REAL_BATTERY` | `SPEC_14_ULTRALIFE_9V` | `CC BY-SA 4.0` | Ultralife 9V lithium battery; full rectangular body and snap terminals boxed. |
| `task7_15` | `task7_15_cylindrical_18650_old_18650_and_21700_lithium_ion_battery_.jpg` | `cylindrical_18650_old` | 2048x1536 | **2** | `TRAIN_V4` | `SPEC_15_18650_21700_PAIR` | `CC0` | Two blue cylindrical lithium-ion cells (18650 and 21700) on dark surface; each cell boxed individually. |
| `task7_16` | `task7_16_cylindrical_18650_old_18650_li_ion_panasonic_cr123a_201211.jpg` | `cylindrical_18650_old` | 3298x2474 | **3** | `VAL_REAL_BATTERY` | `SPEC_16_THREE_LIION_CELLS` | `CC BY-SA 3.0` | Three distinct lithium cells (standing blue 18650, lying silver Panasonic 18650, standing white CR123A); all 3 boxed individually. |
| `task7_21` | `task7_21_hard_neg_capacitor_cylindrical_electrolytic_capacitors_lg_jpg.jpg` | `hard_neg_capacitor_cylindrical` | 450x348 | **0** | `TRAIN_V4` | `SPEC_21_HARD_NEG_CAPACITORS` | `CC BY 2.5` | Hard negative: two cylindrical electrolytic capacitors with wire leads on textured surface (0 battery boxes). |
| `task7_25` | `task7_25_hard_neg_electrical_box_electrical_box03_jpg.jpg` | `hard_neg_electrical_box` | 2087x1186 | **0** | `TRAIN_V4` | `SPEC_25_HARD_NEG_ORANGE_BOX` | `CC0` | Hard negative: two round orange electrical wall boxes (0 battery boxes). |
| `task7_26` | `task7_26_hard_neg_electrical_box_accidental_art_electrical_conduits_m.jpg` | `hard_neg_electrical_box` | 1280x853 | **0** | `TRAIN_V4` | `SPEC_26_HARD_NEG_CONDUITS` | `CC BY-SA 4.0` | Hard negative: outdoor wall with electrical conduit boxes and pipes (0 battery boxes). |
| `task7_27` | `task7_27_hard_neg_electrical_box_conduit_junction_boxes2_jpg.jpg` | `hard_neg_electrical_box` | 1920x2560 | **0** | `VAL_REAL_BATTERY` | `SPEC_27_HARD_NEG_JUNCTION_BOXES` | `CC BY-SA 3.0` | Hard negative: industrial electrical junction boxes and metal conduits (0 battery boxes). |
| `task7_28` | `task7_28_multi_battery_assorted_18650_aa_and_aaa_battery_with_penn.jpg` | `multi_battery_assorted` | 1239x1964 | **3** | `TRAIN_V4` | `SPEC_28_18650_AA_AAA_PENNY` | `CC BY-SA 4.0` | Three cylindrical batteries (green 18650, Kirkland AA, Kirkland AAA) standing side-by-side behind a US penny;penny excluded, 3 batteries boxed. |
| `task7_29` | `task7_29_multi_battery_assorted_three_aaa_batteries_and_one_aa_bat.jpg` | `multi_battery_assorted` | 920x1044 | **4** | `VAL_REAL_BATTERY` | `SPEC_29_THREE_AAA_ONE_AA` | `CC BY-SA 4.0` | Four batteries in one frame (3 vertical AAA cells + 1 horizontal AA cell); directly tests multi-battery cluster detection. |
| `task7_30` | `task7_30_multi_battery_assorted_kodak_aa_and_aaa_alkaline_and_zinc.jpg` | `multi_battery_assorted` | 4239x2826 | **6** | `TRAIN_V4` | `SPEC_30_SIX_KODAK_BATTERIES` | `CC BY-SA 4.0` | Six Kodak AA and AAA batteries (3 standing, 3 lying touching each other); patched in Task 7 Final QA Patch so all 3 standing positive terminals and 3 angled horizontal bodies are fully and tightly enclosed. |
| `task7_31` | `task7_31_multi_battery_assorted_auchan_lr6_aa_and_lr03_aaa_batteri.jpg` | `multi_battery_assorted` | 3072x2304 | **3** | `VAL_REAL_BATTERY` | `SPEC_31_THREE_AUCHAN_BATTERIES` | `CC BY-SA 4.0` | Three Auchan LR6/LR03 batteries (2 standing, 1 lying) on wooden desk. |
| `task7_32` | `task7_32_multi_battery_assorted_aa_aaa_aaaa_a23_battery_comparison.jpg` | `multi_battery_assorted` | 510x458 | **4** | `TRAIN_V4` | `SPEC_32_FOUR_FORM_FACTORS` | `CC BY-SA 3.0` | Four cylindrical batteries of different diameters/heights (AA, AAA, AAAA, A23) side-by-side. |
| `task7_35` | `task7_35_multi_battery_pile_electric_batteries_lp_e6_jpg.jpg` | `multi_battery_pile` | 4130x3096 | **2** | `TRAIN_V4` | `SPEC_35_TWO_LPE6_PACKS` | `CC BY-SA 4.0` | Two rectangular LP-E6 lithium-ion camera battery packs (front and back view). |
| `task7_39` | `task7_39_corroded_leaking_battery_alkaline_battery_leaked_2_jpg.jpg` | `corroded_leaking_battery` | 2448x1928 | **1** | `TRAIN_V4` | `SPEC_39_40_NANFU_LEAKED_AA` | `CC BY-SA 4.0` | Severely corroded leaking AA alkaline battery with white potassium carbonate crust; grouped in TRAIN_V4 with task7_40. |
| `task7_40` | `task7_40_corroded_leaking_battery_alkaline_battery_leaked_1_jpg.jpg` | `corroded_leaking_battery` | 3400x1888 | **1** | `TRAIN_V4` | `SPEC_39_40_NANFU_LEAKED_AA` | `CC BY-SA 4.0` | Second angle of the same leaking AA alkaline battery on tile; grouped in TRAIN_V4 with task7_39 to prevent split leakage. |
| `task7_41` | `task7_41_corroded_leaking_battery_leaking_nopopo_battery_2nd_generat.jpg` | `corroded_leaking_battery` | 3264x2448 | **3** | `VAL_REAL_BATTERY` | `SPEC_41_THREE_LEAKING_NOPOPO` | `CC BY-SA 3.0` | Three corroded/leaking cylindrical batteries lying at perspective angles on tabletop. |
| `task7_43` | `task7_43_multi_battery_assorted_duracell_9_volt_0849_jpg.jpg` | `multi_battery_assorted` | 1406x2091 | **1** | `VAL_REAL_BATTERY` | `SPEC_43_DURACELL_9V` | `CC BY 3.0` | Duracell Plus 9V rectangular battery angled view; full 3D body and top snap terminals boxed. |
| `task7_45` | `task7_45_multi_battery_assorted_duracell_batteries_2184022784_jpg.jpg` | `multi_battery_assorted` | 1280x853 | **4** | `TRAIN_V4` | `SPEC_45_FOUR_DURACELL_DARK` | `CC BY 2.0` | Four Duracell Plus AA batteries (2 upright, 2 crossed in foreground) on dark background. |
| `task7_46` | `task7_46_multi_battery_assorted_energizer_nimh_rechargeable_aa_bat.jpg` | `multi_battery_assorted` | 4192x2794 | **4** | `TRAIN_V4` | `SPEC_46_48_52_ENERGIZER_NIMH` | `CC BY-SA 4.0` | Four Energizer NiMH 2000mAh AA batteries standing in a row; grouped in TRAIN_V4 with task7_48 and task7_52. |
| `task7_48` | `task7_48_multi_battery_assorted_energizer_nimh_rechargeable_aa_bat.jpg` | `multi_battery_assorted` | 4050x2700 | **4** | `TRAIN_V4` | `SPEC_46_48_52_ENERGIZER_NIMH` | `CC BY-SA 4.0` | Four Energizer NiMH 2000mAh AA batteries (2 standing, 2 lying angled); grouped in TRAIN_V4 with task7_46 and task7_52. |
| `task7_50` | `task7_50_multi_battery_assorted_aa_varta_battery_side_pnr_0782_jpg.jpg` | `multi_battery_assorted` | 1467x3876 | **1** | `VAL_REAL_BATTERY` | `SPEC_50_VARTA_ENERGY_AA` | `CC BY-SA 4.0` | Varta Energy Alkaline AA battery upright; full cylinder boxed. |
| `task7_52` | `task7_52_multi_battery_assorted_energizer_nimh_rechargeable_aa_bat.jpg` | `multi_battery_assorted` | 1352x2952 | **1** | `TRAIN_V4` | `SPEC_46_48_52_ENERGIZER_NIMH` | `CC BY-SA 4.0` | Single Energizer NiMH 2000mAh AA battery from the same photographer session as task7_46/48; grouped in TRAIN_V4. |
| `task7_53` | `task7_53_multi_battery_assorted_ikea_ladda_rechargeable_aa_batteri.jpg` | `multi_battery_assorted` | 1280x960 | **4** | `TRAIN_V4` | `SPEC_53_FOUR_IKEA_LADDA` | `CC0` | Four IKEA LADDA NiMH rechargeable AA batteries (2 standing, 2 lying in front); patched in Task 7 Final QA Patch so Boxes #3 and #4 enclose the entire angled bodies and upper-right ends. |
| `task7_54` | `task7_54_multi_battery_assorted_nimh_various_jpg.jpg` | `multi_battery_assorted` | 800x520 | **3** | `VAL_REAL_BATTERY` | `SPEC_54_THREE_NIMH_TYPES` | `CC BY-SA 3.0` | Three NiMH batteries of distinct form factors (GP 1450 prismatic gumstick, Uniross AA, Uniross AAA). |
| `task7_56` | `task7_56_phone_liion_swollen_bl_5c_nokia_battery_jpg.jpg` | `phone_liion_swollen` | 1280x916 | **1** | `VAL_REAL_BATTERY` | `SPEC_56_NOKIA_BL5C_STANDALONE` | `CC BY-SA 3.0` | Standalone Nokia BL-5C Li-ion mobile phone battery; full pouch/prismatic body boxed. |

### 2.2. Danh sách 10 ảnh `NEEDS_MANUAL_REVIEW` (Giữ riêng chờ PM quyết định)

Các ảnh dưới đây có pin thật nhưng nằm trong tình huống biên (trong vỉ nhựa blister pack, gắn trong đế giữ CMOS trên bo mạch, cell 18650 tháo rời vỏ, 4 cell nằm trong bộ sạc, ắc quy cắt bổ lộ bản cực, đống >40 viên pin chồng lấp dày đặc, hộp 48 viên Duracell, hoặc pin điện thoại nằm trong khoang máy). Chúng được tách riêng nhãn vào `labels_needs_manual_review/` và không trộn vào `labels/`:

| ID | Tệp ảnh | Nhóm | Kích thước | Candidate Boxes | Giấy phép | Lý do cần PM quyết định (`NEEDS_MANUAL_REVIEW`) |
|---|---|---|---|---:|---|---|
| `task7_07` | `task7_07_button_coin_cell_button_batterys_cr2032_3v_jpg.jpg` | `button_coin_cell` | 1280x1707 | 2 | `CC BY-SA 4.0` | Two CR2032 coin cells enclosed inside reflective plastic/cardboard retail blister pack held by hand; requires PM policy decision on whether outer blister pack should also be labeled plastic/cardboard. |
| `task7_10` | `task7_10_button_coin_cell_asus_p5pl2_newsun_cr2032_0428_jpg.jpg` | `button_coin_cell` | 1280x1280 | 1 | `CC BY-SA 4.0` | CR2032 coin cell seated inside a black CMOS holder on a motherboard PCB; requires review of socket rim inclusion. |
| `task7_17` | `task7_17_cylindrical_18650_old_lithium_ion_cell_cylindric_jpg.jpg` | `cylindrical_18650_old` | 1280x1184 | 2 | `CC BY-SA 3.0` | Disassembled 18650 lithium-ion cell separated into green internal jelly-roll and empty steel cylinder shell. |
| `task7_18` | `task7_18_cylindrical_18650_old_four_18650_lithium_cells_in_a_liitok.jpg` | `cylindrical_18650_old` | 1200x1680 | 4 | `CC0` | Four Robiton 18650 cells partially occluded inside a 4-bay LiitoKala charger. |
| `task7_36` | `task7_36_multi_battery_pile_lead_acid_batteries_components_2_j.jpg` | `multi_battery_pile` | 2000x1328 | 1 | `CC BY-SA 4.0` | Open-top cutaway lead-acid automotive battery exposing internal lead plates. |
| `task7_37` | `task7_37_multi_battery_pile_used_batteries_jpg.jpg` | `multi_battery_pile` | 1280x622 | 6 | `CC BY-SA 4.0` | Dense overlapping heap of 40+ spent AA/AAA batteries with severe occlusion; requires high-density crowd annotation review before training. |
| `task7_38` | `task7_38_corroded_leaking_battery_eta_esa_955_112_battery_leak_1_png.jpg` | `corroded_leaking_battery` | 1280x1156 | 1 | `CC0` | Internal quartz watch movement with corroded button-cell compartment. |
| `task7_42` | `task7_42_corroded_leaking_battery_polysix_battery_corroded_klm367a_b.jpg` | `corroded_leaking_battery` | 3024x4032 | 1 | `CC BY-SA 4.0` | Corroded cylindrical NiCd backup battery soldered onto synthesizer circuit board. |
| `task7_44` | `task7_44_multi_battery_assorted_duracell_battery_arrangement_defec.jpg` | `multi_battery_assorted` | 700x899 | 1 | `CC BY-SA 4.0` | Retail 48-pack cardboard box containing 48 tightly stacked Duracell AA batteries behind plastic window. |
| `task7_55` | `task7_55_phone_liion_swollen_nokia_bl_5c_battery_jpg.jpg` | `phone_liion_swollen` | 4032x2016 | 1 | `CC BY-SA 4.0` | Nokia BL-5C Li-ion battery seated inside open phone housing with holographic sticker. |

### 2.3. Danh sách 16 ảnh `REJECT` (Loại bỏ hoàn toàn)

Các ảnh do tìm kiếm từ khóa trả về nhưng chỉ chụp vỏ thùng thu gom từ bên ngoài (không thấy viên pin), biểu tượng đồ họa, trang văn bản bằng sáng chế năm 1889, mặt cắt hiển vi tụ điện, hoặc ảnh ngoài mục tiêu:

| ID | Tệp ảnh | Nhóm tìm kiếm | Kích thước | Giấy phép | Lý do loại bỏ (`REJECT`) |
|---|---|---|---|---|---|
| `task7_01` | `task7_01_multi_battery_pile_raccoglitore_pile_usate_presso_centr.jpg` | `multi_battery_pile` | 1280x960 | `CC BY-SA 4.0` | Yellow outdoor battery collection bins exterior only; zero visible battery objects. |
| `task7_02` | `task7_02_multi_battery_pile_recycle_12_png.jpg` | `multi_battery_pile` | 1475x1475 | `CC BY-SA 4.0` | Non-photographic recycling icon graphic with transparent/black alpha background. |
| `task7_03` | `task7_03_multi_battery_pile_80_g_44208_30500917234_jpg.jpg` | `multi_battery_pile` | 2868x2388 | `Public domain` | Off-target historical B&W photograph of wooden timber pile. |
| `task7_04` | `task7_04_multi_battery_collection_e_avfall_retur_av_elektriske_produkt.jpg` | `multi_battery_collection` | 1280x1707 | `CC BY-SA 4.0` | Supermarket e-waste collection bin exterior only; no individual batteries visible. |
| `task7_05` | `task7_05_multi_battery_collection_pant_flaskepant_automat_bottle_depos.jpg` | `multi_battery_collection` | 1280x960 | `CC BY-SA 4.0` | Off-target reverse vending machine (PANT) in supermarket. |
| `task7_06` | `task7_06_multi_battery_collection_library_economy_heavy_items_delivery.jpg` | `multi_battery_collection` | 1280x890 | `CC BY 4.0` | Off-target outdoor delivery truck photograph. |
| `task7_19` | `task7_19_hard_neg_capacitor_cylindrical_electrolytic_capacitor_cross_section.jpg` | `hard_neg_capacitor_cylindrical` | 1280x960 | `CC BY-SA 4.0` | Macro internal cross-section of cut-open capacitor, not an external object. |
| `task7_20` | `task7_20_hard_neg_capacitor_cylindrical_electrolytic_capacitor_cross_section.jpg` | `hard_neg_capacitor_cylindrical` | 1280x960 | `CC BY-SA 4.0` | Longitudinal internal cross-section of cut-open capacitor, not an external object. |
| `task7_22` | `task7_22_hard_neg_cartridge_fuse_blasting_cartridge_dpla_691ca9e52681.jpg` | `hard_neg_cartridge_fuse` | 1500x2250 | `Public domain` | Scanned 1889 US Patent text document. |
| `task7_23` | `task7_23_hard_neg_cartridge_fuse_blasting_cartridge_dpla_691ca9e52681.jpg` | `hard_neg_cartridge_fuse` | 1500x2250 | `Public domain` | Scanned 1889 US Patent specification page. |
| `task7_24` | `task7_24_hard_neg_cartridge_fuse_blasting_cartridge_dpla_691ca9e52681.jpg` | `hard_neg_cartridge_fuse` | 1500x2250 | `Public domain` | Scanned 1889 US Patent line drawing. |
| `task7_33` | `task7_33_multi_battery_pile_six_lead_acid_batteries_used_as_pa.jpg` | `multi_battery_pile` | 5080x4761 | `Public domain` | Historical B&W laboratory glass-jar stationary battery cells. |
| `task7_34` | `task7_34_multi_battery_pile_lead_acid_batteries_components_jpg.jpg` | `multi_battery_pile` | 2000x1328 | `CC BY-SA 4.0` | Covered table with printed Russian sheets; no visible battery objects. |
| `task7_47` | `task7_47_multi_battery_assorted_energizer_bunny_hot_hare_balloon_5.jpg` | `multi_battery_assorted` | 1672x941 | `CC BY 2.0` | Off-target Energizer Bunny hot air balloon in sky. |
| `task7_49` | `task7_49_multi_battery_assorted_03_2024_varta_ag_ellwangen_img_858.jpg` | `multi_battery_assorted` | 1280x690 | `CC BY-SA 4.0` | Off-target Varta corporate building exterior. |
| `task7_51` | `task7_51_multi_battery_assorted_ellwangen_varta_1_jpg.jpg` | `multi_battery_assorted` | 1280x805 | `CC BY-SA 3.0` | Off-target aerial photograph of industrial facility. |

---

## 3. KIỂM ĐỊNH CHỐNG RÒ RỈ DỮ LIỆU — ZERO LEAKAGE AUDIT (PHẦN B2/B3)

Báo cáo chi tiết lưu tại: `data/acquisition_staging/task7_diverse_battery_batch/task7_leakage_report.json`.

1. **Đối chiếu với toàn bộ các tập dữ liệu hiện hữu (`Train V2: 1398`, `Val V2: 299`, `Test: 299`, `Holdout: 27`, `Roboflow 140 Base: 140`):**
   - Trùng khớp `SHA-256`: **`0`**
   - Cặp ảnh gần trùng `4-rotation pHash <= 8` với bất kỳ tập hiện hữu nào: **`0`** (`zero_external_leakage = True`)
2. **Đối chiếu nội bộ giữa 30 ảnh `QA_PASS` và phân nhóm theo Specimen/Photographer:**
   - Phát hiện **3 nhóm ảnh chụp cùng viên pin / cùng buổi chụp** (`pHash <= 8` hoặc cùng mẫu vật):
     - `SPEC_12_13_MOTOMA_9V_WOOD`: `task7_12` (mặt bên pin vuông Motoma 9V trên mặt gỗ) và `task7_13` (mặt cực của cùng viên pin Motoma 9V trên mặt gỗ) -> **Gom cả hai vào `TRAIN_V4`**.
     - `SPEC_39_40_NANFU_LEAKED_AA`: `task7_39` và `task7_40` (hai góc chụp của cùng viên pin AA Nanfu bị chảy hóa chất trắng) -> **Gom cả hai vào `TRAIN_V4`**.
     - `SPEC_46_48_52_ENERGIZER_NIMH`: `task7_46` (4 viên Energizer NiMH đứng), `task7_48` (4 viên Energizer NiMH 2 đứng 2 nằm) và `task7_52` (1 viên Energizer NiMH cùng buổi chụp) -> **Gom cả ba vào `TRAIN_V4`**.
   - Nhờ gom theo `specimen_group`, số cặp rò rỉ giữa `TRAIN_V4` (`19` ảnh) và `VAL_REAL_BATTERY` (`11` ảnh) ở ngưỡng `4-rotation pHash <= 8` là: **`0`** (`zero_cross_split_leakage = True`).

---

## 4. ĐỀ XUẤT DATASET V4, TẬP `VAL_REAL_BATTERY` & CẤU HÌNH E4 (PHẦN B3 — DO NOT RUN)

Báo cáo đề xuất chi tiết: `data/acquisition_staging/task7_diverse_battery_batch/task7_dataset_v4_and_e4_proposal.json`  
File cấu hình đề xuất (chưa chạy): `configs/training_yolov8n_v4_proposed_DO_NOT_RUN.yaml`

### 4.1. Kiến trúc đánh giá đa tầng cho V4 (Multi-Benchmark Evaluation Gate)
Để tránh lặp lại tình trạng điểm Validation cao ảo trên ảnh đơn chiếc Roboflow (`Val V3 = 0.992 mAP50` nhưng `Holdout = 0.008 mAP50`), chúng tôi đề xuất tách bạch 5 cổng đánh giá:
1. **`VAL_V2` (`299` ảnh, `901` boxes, `0` battery):** Cổng kiểm soát chống quên kiến thức (`Catastrophic Forgetting Gate`) trên 9 lớp rác hiện hữu.
2. **`VAL_V3_TABLETOP` (`328` ảnh, `930` boxes, `29` tabletop batteries):** Cổng kiểm tra hồi quy trong miền Roboflow.
3. **`VAL_REAL_BATTERY` (`11` ảnh độc lập từ Task 7: `10` ảnh đa dạng gồm `21` battery boxes + `1` ảnh hard negative):** Đánh giá độc lập trên cụm nhiều pin (`task7_16`, `task7_29`, `task7_31`, `task7_41`, `task7_54`), pin cúc áo (`task7_09`), pin 9V (`task7_14`, `task7_43`), pin điện thoại BL-5C (`task7_56`) và hard negative tủ điện/ống kim loại (`task7_27`).
4. **`TEST` (`299` ảnh, `720` boxes):** Tập kiểm thử nội bộ đóng băng (`FROZEN`).
5. **`HOLDOUT` (`27` ảnh `D:\Hinh_Anh_Rac`, `57` GT boxes):** Tập nghiệm thu thực tế bên ngoài đóng băng (`FROZEN`).

### 4.2. Số lượng ảnh và box dự kiến của `TRAIN_V4`
- `Train V2 baseline`: `1.398` ảnh (`3.963` boxes, trong đó `1` battery box).
- `Roboflow V3 Clean Train`: `+111` ảnh (`+111` battery boxes).
- `Task 7 Diverse Battery Train`: `+19` ảnh (`+40` battery boxes từ 16 ảnh dương tính + `3` ảnh hard negative 0 box).
- **Tổng `TRAIN_V4` dự kiến:** **`1.528` ảnh (`4.114` boxes, trong đó `152` battery boxes)**.

### 4.3. Phân tích nguyên nhân E3 suy giảm ở Task 6 & Đề xuất cấu hình E4 bảo thủ (`DO NOT RUN`)
Tại sao E3 (`imgsz=512, freeze=0, lr0=0.0005, mosaic=0.5`) làm giảm Holdout mAP50 từ `0.05914` xuống `0.00823`?
1. **`freeze=0` (Unfrozen Backbone) + Learning Rate `0.0005`:** Khi mở khóa toàn bộ backbone trên tập dữ liệu mất cân bằng và thêm 111 ảnh pin có cùng một kiểu nền bàn gỗ, các tầng trích xuất đặc trưng của YOLOv8n bị dịch chuyển theo hướng nhạy quá mức với vật thể nhỏ trên nền phẳng, gây ra thêm 15 False Positives trên Holdout (`FP` tăng từ `37` lên `52`) và làm mất các phát hiện đúng ở các lớp `metal`, `plastic`, `paper`.
2. **`imgsz=512` thay vì `640`:** Checkpoint gốc E1 được huấn luyện ở độ phân giải `imgsz=640`. Khi hạ xuống `512` trong quá trình fine-tune E3, tỷ lệ scale của các anchor/feature map thay đổi so với trọng số gốc E1.

**Đề xuất cấu hình E4 (chỉ chạy sau khi bổ sung đủ dữ liệu và có lệnh phê duyệt mới):**
- **Khởi tạo từ:** `E1` (`artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` — không khởi tạo nối tiếp từ E3 đã bị trôi đặc trưng).
- **`imgsz`: `640`** (khớp 100% độ phân giải gốc của E1; ở Task 6 `imgsz=512, batch=4` chỉ tốn `400.1 MB` VRAM nên `imgsz=640` hoàn toàn an toàn dưới `1 GB` VRAM trên RTX 2050 4GB).
- **`freeze`: `10`** (đóng băng toàn bộ 10 block backbone C2f/SPPF của YOLOv8n để bảo toàn tuyệt đối đặc trưng 9 lớp rác hiện hữu; chỉ cập nhật neck/head).
- **`lr0`: `0.0002`** (giảm 2.5 lần so với E3), `lrf: 0.01`, `epochs: 20`, `patience: 5`, `batch: 4`, `workers: 0`.

---

## 5. BÁO CÁO TRUNG THỰC VỀ KHOẢNG TRỐNG DỮ LIỆU CÒN LẠI (REMAINING DATA GAPS)

Mặc dù lô 30 ảnh `QA_PASS` (`61` boxes) của Task 7 đã giải quyết tốt sự thiếu hụt về:
- Cụm nhiều viên pin (`14/26` ảnh dương tính có từ 2 đến 6 viên pin),
- Đa dạng hình thái (pin cúc áo CR2032/ML2032, pin vuông 9V, pin sạc 18650/21700, pin máy ảnh LP-E6, pin điện thoại Nokia BL-5C, pin rỉ sét/chảy hóa chất),
- Mẫu âm tính khó (tụ điện trụ, hộp điện âm tường, ống luồn dây kim loại),

**Vẫn còn 3 khoảng trống dữ liệu thực tế cần bổ sung tiếp trước khi kỳ vọng cải thiện mạnh trên `D:\Hinh_Anh_Rac`:**
1. **Ảnh hỗn hợp nhiều loại rác (`Mixed-Class Waste Context`):** Toàn bộ 30 ảnh `QA_PASS` từ Wikimedia Commons chỉ chứa lớp `0: battery` (hoặc hard negative không có rác), chưa có ảnh chụp **pin nằm lẫn cùng vỏ chai nhựa, giấy vụn, lon kim loại, bìa carton trên cùng một khung hình** như trong thùng rác sinh hoạt thực tế.
2. **Nền hiện trường phức tạp & khoảng cách chụp xa (`Small Object Scale`):** Phần lớn ảnh Wikimedia là ảnh chụp cận cảnh/trung cảnh (`bbox area` chiếm 10%–70% diện tích ảnh), trong khi ảnh thực tế ngoài đời hoặc camera giám sát có thể chụp từ khoảng cách 1–1.5 mét trên nền gạch đá hoa, nền bê tông hoặc đất cát khiến viên pin chỉ chiếm `< 3%` diện tích ảnh.
3. **Chính sách gán nhãn đống pin dày đặc (`Dense Pile Policy`):** `task7_37` (đống >40 viên pin chồng lấp) và `task7_44` (hộp 48 viên Duracell) đang được giữ trong `NEEDS_MANUAL_REVIEW` chờ PM quyết định có gán nhãn từng viên bị che khuất hay loại bỏ.

---

## 6. BẢNG NGUỒN GỐC & MÃ BĂM TOÀN BỘ 56 ẢNH ỨNG VIÊN TASK 7

| ID | QA Verdict | Giấy phép | Tác giả | SHA-256 (16 ký tự đầu) | pHash (0°) | Nguồn Wikimedia Commons |
|---|---|---|---|---|---|---|
| `task7_01` | `REJECT` | `CC BY-SA 4.0` | `Ensahequ` | `8cd7f950ef604f85...` | `f0226c7c42497cbf` | [Link](https://commons.wikimedia.org/wiki/File%3ARaccoglitore%20pile%20usate%20presso%20Centro%20commerciale%20Le%20Colline%2C%20Reggio%20Emilia.jpg) |
| `task7_02` | `REJECT` | `CC BY-SA 4.0` | `Dylan'sLabatory` | `0836f7eed6d5d58c...` | `0000000000000000` | [Link](https://commons.wikimedia.org/wiki/File%3ARecycle-12.png) |
| `task7_03` | `REJECT` | `Public domain` | `National Museum of the U.S. Navy` | `10fae197e21faa2e...` | `98958e46e3e3f170` | [Link](https://commons.wikimedia.org/wiki/File%3A80-G-44208%20%2830500917234%29.jpg) |
| `task7_04` | `REJECT` | `CC BY-SA 4.0` | `Wolfmann` | `a3a5deede9eb78ea...` | `a297df689633d244` | [Link](https://commons.wikimedia.org/wiki/File%3AE-AVFALL%20retur%20av%20elektriske%20produkter%20elektronikk%20lysp%C3%A6rer%20batterier%20lysstoffr%C3%B8r%20%28E-waste%20return%20of%20electrical%20products%20batteries%29%20Vi%20bryr%20oss%20om%20milj%C3%B8et%20Avfallsdunk%20%28waste%20bin%29%20Spar%20%28grocery%20store%29%20Tj%C3%B8me%20NORWAY%202026-08-15%20IMG%20994.jpg) |
| `task7_05` | `REJECT` | `CC BY-SA 4.0` | `Wolfmann` | `9b7c9e3ecdd2d750...` | `d934de69b263b260` | [Link](https://commons.wikimedia.org/wiki/File%3APANT%20flaskepant%20automat%20%28bottle%20deposit-return%20machines%29%20Tomra%20T-63HCp%20T-820%20R%C3%B8de%20Kors%20Br%C3%B8dskj%C3%A6remaskin%20%28bread%20slicer%29%20vask%20%28sink%29%20avfall%20%28bin%29%20Retur%20lysr%C3%B8r%20batterier%20lysp%C3%A6rer%20sm%C3%A5el.%20%28E-waste%29%20EXTRA%20Coop%20grocery%20store%20Tj%C3%B8me%20NORW.jpg) |
| `task7_06` | `REJECT` | `CC BY 4.0` | `Jacob coffin` | `05fa0860d94d1a61...` | `d1d15f55cb28148f` | [Link](https://commons.wikimedia.org/wiki/File%3ALibrary%20Economy%20-%20Heavy%20Items%20Delivery%20or%20Collections.png) |
| `task7_07` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `KKPCW（Kyu3）` | `d5feec9e90c61a59...` | `a3f8890fc037d978` | [Link](https://commons.wikimedia.org/wiki/File%3AButton%20batterys%20CR2032%203V.jpg) |
| `task7_08` | `QA_PASS` | `CC BY-SA 4.0` | `Raimond Spekking` | `7c2c397f99785312...` | `fb27f0508593c719` | [Link](https://commons.wikimedia.org/wiki/File%3ACoin%20cell%20CR2032%20H%20by%20Hitachi%20Maxell-7730.jpg) |
| `task7_09` | `QA_PASS` | `CC BY-SA 4.0` | `Raimond Spekking` | `e2432666cd576652...` | `fe7aa0a19585d385` | [Link](https://commons.wikimedia.org/wiki/File%3AML2032%20Rechargeable%20button%20cell%20battery%2C%20Hitachi%20Maxell-3922.jpg) |
| `task7_10` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Raimond Spekking` | `e3d8d1444aae1c7e...` | `eae0d48b91bc8fc8` | [Link](https://commons.wikimedia.org/wiki/File%3AAsus%20P5PL2%20-%20Newsun%20CR2032-0428.jpg) |
| `task7_11` | `QA_PASS` | `CC BY-SA 4.0` | `Encik Tekateki` | `b43b6818342a220c...` | `bddbc226c2253d25` | [Link](https://commons.wikimedia.org/wiki/File%3A9V%20Bateri%20Ujian%20Short%20Circuit.png) |
| `task7_12` | `QA_PASS` | `CC BY-SA 3.0` | `Nevit Dilmen (talk)` | `fe86031b64d03f9b...` | `e992256c967133d3` | [Link](https://commons.wikimedia.org/wiki/File%3A9V%20PP3%20Battery%201530632.JPG) |
| `task7_13` | `QA_PASS` | `CC BY-SA 3.0` | `Nevit Dilmen (talk)` | `9304c3dcf1806941...` | `e0538fac7253295b` | [Link](https://commons.wikimedia.org/wiki/File%3A9V%20PP3%20Battery%201530633.JPG) |
| `task7_14` | `QA_PASS` | `CC BY-SA 4.0` | `Cave2596` | `10a67f485aea30e9...` | `ae9591d8915ec46e` | [Link](https://commons.wikimedia.org/wiki/File%3ALithium%209V%20Battery%206F22.jpg) |
| `task7_15` | `QA_PASS` | `CC0` | `Sevenethics` | `51f83927ccd06e13...` | `d4a77b98902769d2` | [Link](https://commons.wikimedia.org/wiki/File%3A18650%20and%2021700%20lithium%20ion%20battery%20cell.jpg) |
| `task7_16` | `QA_PASS` | `CC BY-SA 3.0` | `Mk2010` | `e2e7d0d30bb6835a...` | `f229ed96062dcd16` | [Link](https://commons.wikimedia.org/wiki/File%3A18650%20Li-ion%20%26%20Panasonic%20CR123A%2020121116.jpg) |
| `task7_17` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 3.0` | `RudolfSimon` | `221924d4a807df61...` | `854b1c3873e347ae` | [Link](https://commons.wikimedia.org/wiki/File%3ALithium-Ion%20Cell%20cylindric.JPG) |
| `task7_18` | `NEEDS_MANUAL_REVIEW` | `CC0` | `Retired electrician` | `e917ef2997d52552...` | `fa9090b8906fc36f` | [Link](https://commons.wikimedia.org/wiki/File%3AFour%2018650%20lithium%20cells%20in%20a%20Liitokala%20PL4%20charger.jpg) |
| `task7_19` | `REJECT` | `CC BY-SA 4.0` | `TubeTimeUS` | `d31459559269551f...` | `e419ebc7a6ae1815` | [Link](https://commons.wikimedia.org/wiki/File%3AElectrolytic%20capacitor%20cross%20section.jpg) |
| `task7_20` | `REJECT` | `CC BY-SA 4.0` | `TubeTimeUS` | `f0d007cbec304630...` | `81aa3e9ac5ced568` | [Link](https://commons.wikimedia.org/wiki/File%3AElectrolytic%20capacitor%20cross%20section%20lengthwise.jpg) |
| `task7_21` | `QA_PASS` | `CC BY 2.5` | `Lgreen` | `90ecccd8c74e650e...` | `f241cd373036cf38` | [Link](https://commons.wikimedia.org/wiki/File%3AElectrolytic%20Capacitors%20LG.JPG) |
| `task7_22` | `REJECT` | `Public domain` | `Todd, James Polk; Welch, John Steel` | `22db042b4cf6a992...` | `bd3d3fc2c2c2c294` | [Link](https://commons.wikimedia.org/wiki/File%3ABlasting%20Cartridge.%20-%20DPLA%20-%20691ca9e52681044b07b4c2997df071e8%20%28page%203%29.jpg) |
| `task7_23` | `REJECT` | `Public domain` | `Todd, James Polk; Welch, John Steel` | `fda548415e021716...` | `ee94916bc53d94c4` | [Link](https://commons.wikimedia.org/wiki/File%3ABlasting%20Cartridge.%20-%20DPLA%20-%20691ca9e52681044b07b4c2997df071e8%20%28page%202%29.jpg) |
| `task7_24` | `REJECT` | `Public domain` | `Todd, James Polk; Welch, John Steel` | `5770f13f76d3c89a...` | `e49999669319e669` | [Link](https://commons.wikimedia.org/wiki/File%3ABlasting%20Cartridge.%20-%20DPLA%20-%20691ca9e52681044b07b4c2997df071e8%20%28page%201%29.jpg) |
| `task7_25` | `QA_PASS` | `CC0` | `Leotard` | `da4e68461cbd7158...` | `bbb598c4c449c63d` | [Link](https://commons.wikimedia.org/wiki/File%3AElectrical%20box03.jpg) |
| `task7_26` | `QA_PASS` | `CC BY-SA 4.0` | `Marc Merlin` | `6003526b5ae5781e...` | `fd49d23cc09a70e3` | [Link](https://commons.wikimedia.org/wiki/File%3AAccidental%20art-%20electrical%20conduits%2C%20meters%20and%20junction%20boxes%20on%20the%20backside%20of%20the%20Briarcliff%20Plaza%20shopping%20Center.jpg) |
| `task7_27` | `QA_PASS` | `CC BY-SA 3.0` | `Achim Hering` | `9fbddcb52e8619c2...` | `8b115952993fcae5` | [Link](https://commons.wikimedia.org/wiki/File%3AConduit%20junction%20boxes2.jpg) |
| `task7_28` | `QA_PASS` | `CC BY-SA 4.0` | `JayCubby` | `01f72718a9944b69...` | `a3029abf9d1908fe` | [Link](https://commons.wikimedia.org/wiki/File%3A18650%2C%20AA%2C%20and%20AAA%20battery%20with%20penny.jpg) |
| `task7_29` | `QA_PASS` | `CC BY-SA 4.0` | `Jonatan Svensson Glad` | `609dedab002b29cb...` | `f9e11e80791ec639` | [Link](https://commons.wikimedia.org/wiki/File%3AThree%20AAA%20Batteries%20and%20one%20AA%20Battery.JPG) |
| `task7_30` | `QA_PASS` | `CC BY-SA 4.0` | `Multicherry` | `f556d586a7f9394b...` | `af99c26c99e25a29` | [Link](https://commons.wikimedia.org/wiki/File%3AKodak%20AA%20and%20AAA%20alkaline%20and%20zinc%20batteries.jpg) |
| `task7_31` | `QA_PASS` | `CC BY-SA 4.0` | `Antonio Kless` | `df8dfb56d22c3447...` | `c12d9c7e07fa5123` | [Link](https://commons.wikimedia.org/wiki/File%3AAuchan%20LR6%20%28AA%29%20and%20LR03%20%28AAA%29%20batteries.jpg) |
| `task7_32` | `QA_PASS` | `CC BY-SA 3.0` | `Lead holder` | `d1365fa3a6801595...` | `a89593cad26cb55a` | [Link](https://commons.wikimedia.org/wiki/File%3AAA%20AAA%20AAAA%20A23%20battery%20comparison-1.jpg) |
| `task7_33` | `REJECT` | `Public domain` | `National Institute of Standards and` | `d7e7c71e29e09ae9...` | `bfb9924690f4c638` | [Link](https://commons.wikimedia.org/wiki/File%3ASix%20lead-acid%20batteries%20used%20as%20part%20of%20the%20DC%20power%20supply%20at%20NBS%20included%20in%20a%20test%20of%20battery%20additive.jpg) |
| `task7_34` | `REJECT` | `CC BY-SA 4.0` | `Mike-fiesta` | `e09bf28e8d8a2590...` | `ec91354f33308d6e` | [Link](https://commons.wikimedia.org/wiki/File%3ALead-acid%20batteries%20components.jpg) |
| `task7_35` | `QA_PASS` | `CC BY-SA 4.0` | `Surprizi` | `807b0b28ac34d7ba...` | `b8cfc79092612d6d` | [Link](https://commons.wikimedia.org/wiki/File%3AElectric%20batteries%20LP-E6.jpg) |
| `task7_36` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Mike-fiesta` | `fcbc78ebdeb1b2fe...` | `ed188f8bd4632273` | [Link](https://commons.wikimedia.org/wiki/File%3ALead-acid%20batteries%20components%202.jpg) |
| `task7_37` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Peter Fiskerstrand` | `a4443c59a417f1f4...` | `89cf161ff1831ae4` | [Link](https://commons.wikimedia.org/wiki/File%3AUsed%20batteries.jpg) |
| `task7_38` | `NEEDS_MANUAL_REVIEW` | `CC0` | `Kaffe42` | `c622353d0da8aea8...` | `bba4c4c39a8a939b` | [Link](https://commons.wikimedia.org/wiki/File%3AETA-ESA%20955.112%2C%20battery%20leak%201.png) |
| `task7_39` | `QA_PASS` | `CC BY-SA 4.0` | `Leiem` | `9a84790bd3822ead...` | `e8c79f1830674d6c` | [Link](https://commons.wikimedia.org/wiki/File%3AAlkaline%20battery%20%28leaked%29%202.jpg) |
| `task7_40` | `QA_PASS` | `CC BY-SA 4.0` | `Leiem` | `311c824a2cdd0755...` | `f9c796383c29e1c2` | [Link](https://commons.wikimedia.org/wiki/File%3AAlkaline%20battery%20%28leaked%29%201.jpg) |
| `task7_41` | `QA_PASS` | `CC BY-SA 3.0` | `Wiliam Davies` | `6a7dfe3677bec056...` | `dbd4d44b3b94c02b` | [Link](https://commons.wikimedia.org/wiki/File%3ALeaking%20Nopopo%20battery%202nd%20generation.JPG) |
| `task7_42` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Untergeek` | `aa2ddfa28563112c...` | `a8b657900a5b5be9` | [Link](https://commons.wikimedia.org/wiki/File%3APolysix%20Battery%20corroded%20KLM367A%20board%20n.jpg) |
| `task7_43` | `QA_PASS` | `CC BY 3.0` | `Ashley Pomeroy` | `1c58ef8e718787d8...` | `fa8dc56a85cc8695` | [Link](https://commons.wikimedia.org/wiki/File%3ADuracell%209%20Volt%200849.jpg) |
| `task7_44` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Encik Tekateki` | `005e592beec85014...` | `c3046e6b25b4b637` | [Link](https://commons.wikimedia.org/wiki/File%3ADuracell%20Battery%20Arrangement%20Defects.jpg) |
| `task7_45` | `QA_PASS` | `CC BY 2.0` | `Demion` | `bd70619128db7d32...` | `c07e63871cc53f32` | [Link](https://commons.wikimedia.org/wiki/File%3ADuracell%20Batteries%20%282184022784%29.jpg) |
| `task7_46` | `QA_PASS` | `CC BY-SA 4.0` | `Multicherry` | `8c0f71af7b9ac980...` | `ba87c53885f07e85` | [Link](https://commons.wikimedia.org/wiki/File%3AEnergizer%20NiMH%20rechargeable%20AA%20batteries%20%28four%29%20%28vertical%29.jpg) |
| `task7_47` | `REJECT` | `CC BY 2.0` | `Mark Turnauckas from Akron, Ohio, U` | `32881ca2a451c4c9...` | `9fe043bd6ed32418` | [Link](https://commons.wikimedia.org/wiki/File%3AEnergizer%20Bunny%C2%AE%20Hot%20%E2%80%9CHare%E2%80%9D%20Balloon%20%285820020310%29.jpg) |
| `task7_48` | `QA_PASS` | `CC BY-SA 4.0` | `Multicherry` | `541ce5902b927762...` | `fb529c2d87846dc2` | [Link](https://commons.wikimedia.org/wiki/File%3AEnergizer%20NiMH%20rechargeable%20AA%20batteries%20%28four%29%20%28various%20positions%29.jpg) |
| `task7_49` | `REJECT` | `CC BY-SA 4.0` | `Alexander-93` | `e71b8ebc042de588...` | `d2ac2153df70ac2d` | [Link](https://commons.wikimedia.org/wiki/File%3A03%202024%20Varta%20AG%20Ellwangen%20IMG%208586.jpg) |
| `task7_50` | `QA_PASS` | `CC BY-SA 4.0` | `D-Kuru` | `826a206459ed49cb...` | `b9d496c3c3399c64` | [Link](https://commons.wikimedia.org/wiki/File%3AAA%20VARTA%20battery-side%20PNr%C2%B00782.jpg) |
| `task7_51` | `REJECT` | `CC BY-SA 3.0` | `Kabelnet at German Wikipedia` | `998c995ed0684dc3...` | `d06f3fb112d9590c` | [Link](https://commons.wikimedia.org/wiki/File%3AEllwangen%20Varta%201.jpg) |
| `task7_52` | `QA_PASS` | `CC BY-SA 4.0` | `Multicherry` | `50fe42bc9874b2c0...` | `b0c7c79c9b748760` | [Link](https://commons.wikimedia.org/wiki/File%3AEnergizer%20NiMH%20rechargeable%20AA%20battery%20%28vertical%29.jpg) |
| `task7_53` | `QA_PASS` | `CC0` | `RAIL P (RAIL.PHOTOGRAPHY)` | `f0a2a90c522cc7aa...` | `e3178c69973449a7` | [Link](https://commons.wikimedia.org/wiki/File%3AIKEA%20LADDA%20rechargeable%20AA-batteries.jpg) |
| `task7_54` | `QA_PASS` | `CC BY-SA 3.0` | `Wikimedia Contributor` | `82cb16fbccd07245...` | `ef53959e2ce0e0e0` | [Link](https://commons.wikimedia.org/wiki/File%3ANiMH%20various.jpg) |
| `task7_55` | `NEEDS_MANUAL_REVIEW` | `CC BY-SA 4.0` | `Kaga tau` | `c68405354e0fce42...` | `fa80054e1f3a3f3a` | [Link](https://commons.wikimedia.org/wiki/File%3ANokia%20BL-5C%20battery.jpg) |
| `task7_56` | `QA_PASS` | `CC BY-SA 3.0` | `Adam Brookes (Adambro)` | `c5c09fa127dbd63e...` | `ba95e19a8c739d44` | [Link](https://commons.wikimedia.org/wiki/File%3ABL-5C%20Nokia%20battery.jpg) |
