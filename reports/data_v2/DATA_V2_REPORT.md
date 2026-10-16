# DATA V2 REPORT — Split chỉ ảnh thật (không Mendeley) — Agent A (Data/Mapping)

Ngày: 2026-10-11 · seed 42 · Mọi số liệu dưới đây đều lấy từ lần chạy thật (log nằm trong `reports/data_v2/`).

## 1. Kết quả QA nhãn trên 257 ảnh thật ban đầu
Script: `reports/reproduce_scripts/data_v2_label_qa.py` → `reports/data_v2/label_qa.json`, overlay trong `reports/data_v2/overlays/` (20 ảnh, seed 42).

| Kiểm tra | Kết quả |
|---|---|
| Ảnh / file nhãn `.txt` | 257 / 257 (thêm 6 file `.txt.bak` mồ côi, giữ nguyên) |
| Thiếu nhãn / nhãn mồ côi / file nhãn rỗng / class ngoài 0..9 / tọa độ hình học lỗi | 0 / 0 / 0 / 0 / 0 |
| **Box giữ chỗ `0.5 0.5 0.4 0.4`** (giá trị fallback của `collect_real_detection_data.py`) | **4 ảnh** (`oi_00a36f96…`, `oi_02deba01…`, `oi_04e310af…`, `oi_07d2a7e5…`). Overlay `oi_07d2a7e5` cho thấy box "cardboard" nằm trên một chiếc mặt nạ phòng độc → nhãn sai |
| Box rất nhỏ (diện tích < 1e-4) | 32 ảnh (chủ yếu tàn thuốc / nắp chai trong TACO, hợp lệ) |
| Phân bố box theo nguồn | TACO (105 ảnh): plastic 405, trash 319, metal 92, paper 89, cardboard 67, glass 60, biological 8, shoes 7, battery 2. OI (152 ảnh): **shoes 507**, clothes 58, glass 21, metal 17, cardboard 2, plastic 1 |

**Lỗi nghiêm trọng phát hiện qua việc xem overlay:** 102/152 ảnh OpenImages là giày/quần áo **đang được người mặc** (ví dụ `oi_09d7e1de` là cầu thủ floorball, `oi_208fcde7` là người chơi roller derby, `oi_93c70947` là người tập xà đơn). Điều này vi phạm `acceptance_criteria.exclude_in_use_person_clothing` và khiến lớp `shoes` học thành "bàn chân người". Các ảnh này đã bị quarantine (lọc theo nhãn người trong CSV box của OI, mục 3).

**Lỗi hướng ảnh (EXIF):** 10 ảnh TACO có sẵn (`taco_0241, 0298, 0329, 0330, 0332, 0336, 0359, 0460, 0608, 0611`) được lưu ở độ phân giải gốc kèm EXIF Orientation=6, trong khi box TACO lại theo khung ảnh *đã xoay theo EXIF*. Ultralytics có áp EXIF, còn `YoloBoxDataset` (`Image.open().convert`) **thì không** → với SSDLite các box sẽ bị lệch. Vì vậy chúng bị quarantine; ngoài ra còn 1 ảnh mới tải có EXIF=3 (`taco_0559`).

## 2. Mở rộng dữ liệu (`scripts/collect_real_detection_v2.py`, file mới, không ghi đè)
- **TACO đầy đủ** (dùng `annotations.json` chính thức, 1500 ảnh, cache tại `data/audit/taco_annotations_raw.json`), map theo `category_mapping`: tải mới 1253 ảnh bản Flickr 640px, giữ nguyên 95 ảnh đã có. 142 ảnh bản 640px bị lệch hướng (đã mất EXIF hoặc bị chuyển vị) nên được **phục hồi** bằng cách tải ảnh gốc → `exif_transpose` → kiểm tra tỉ lệ khung khớp với width/height trong annotation → lưu JPEG đã xoay thẳng (cạnh dài tối đa 1280, không còn tag EXIF). Đã kiểm tra trực quan overlay (`reports/data_v2/overlays_v2/*_recovered_overlay.jpg`). Box thuộc nhóm REQUIRES_REVIEW hoặc chưa map (Carded blister pack, Plastified paper bag, Rope…) **không** ghi vào nhãn 10 lớp; chúng được lưu ở `data/detection/real_v2_ambiguous_boxes.json`.
- **OpenImages V7 validation** (CSV box có sẵn trên máy): chỉ dùng các nhãn có `class_id` khác null trong `class_mapping`. Loại các ảnh có bất kỳ nhãn người nào (Person/Man/Woman/Boy/Girl/Human *), và loại ảnh có box map được mà mang cờ IsDepiction hoặc IsGroupOf. Giới hạn 300 ảnh/lớp. Sau lọc chỉ còn 527 ảnh: clothes 295, shoes 153, glass 54, metal 19, paper 3, plastic 3. Tải mới 491 ảnh, 36 ảnh đã có.
- **Không bổ sung được battery**: OpenImages không có lớp boxable "Battery", và toàn bộ TACO chỉ có 2 box battery, 8 box Food waste, 7 box Shoe. Muốn bổ sung các lớp này cần tập train của OI (~2 GB CSV, mà cũng không có battery) hoặc một nguồn khác → ngoài phạm vi thời gian lần này.
- Thời gian tải: TACO 139 s, OI khoảng 60 s, phục hồi hướng ảnh khoảng 50 s.

## 3. Split `splits_real_v2` (`scripts/build_real_v2_splits.py`)
Trong 2143 ảnh real → **1996 ảnh dùng được**, **147 ảnh bị quarantine** (`data/detection/quarantine_real_v2.csv`; file thô không bị xóa):

| Lý do | Số ảnh |
|---|---|
| oi_human_present_in_use_object | 101 |
| review_status_REJECTED / NEEDS_RELABEL (manifest_detection_v1) | 14 / 4 |
| empty_label (ảnh TACO chỉ có box ambiguous) | 13 |
| taco_orientation_mismatch (EXIF=6, ảnh có sẵn) | 10 |
| oi_groupof_or_depiction_box | 4 |
| exif_orientation_3 | 1 |

Cách chia: nhóm ảnh gần trùng theo pHash (DCT 64-bit, Hamming ≤ 8, union-find; có 15 cặp, cả hai ảnh của mỗi cặp luôn nằm cùng một split). Phân tầng theo **nguồn + lớp hiếm nhất có mặt trong ảnh**. Lưu ý: tôi chọn "lớp hiếm nhất" thay vì "lớp chiếm ưu thế" như đề bài, vì nếu dùng lớp chiếm ưu thế thì battery/biological sẽ bị gộp vào stratum plastic và có thể không xuất hiện ở val/test. Tỉ lệ 70/15/15. Ngoài ra loại các ảnh trùng SHA-256 tuyệt đối và ảnh trùng SHA với holdout (kết quả: 0 ảnh trùng holdout).

| Split | Ảnh | Box | TACO / OI |
|---|---|---|---|
| train | 1398 | 3963 | 1029 / 369 |
| val | 299 | 902 | 221 / 78 |
| test | 299 | 720 | 221 / 78 |

Box (số ảnh có lớp) theo lớp:

| Lớp | train | val | test |
|---|---|---|---|
| 0 battery | **1** (1) | 0 | 1 (1) |
| 1 biological | **3** (3) | 2 (1) | 1 (1) |
| 2 cardboard | 169 (126) | 39 (31) | 36 (26) |
| 3 clothes | 368 (212) | 69 (44) | 91 (45) |
| 4 glass | 213 (101) | 82 (22) | 32 (22) |
| 5 metal | 420 (214) | 74 (45) | 73 (43) |
| 6 paper | 160 (123) | 43 (27) | 34 (28) |
| 7 plastic | 1560 (791) | 382 (173) | 274 (161) |
| 8 shoes | 215 (117) | 41 (23) | 39 (23) |
| 9 trash | 854 (294) | 170 (64) | 139 (63) |

**Lớp có < 150 instance ở train:** `battery` (1) và `biological` (3), gần như không thể học được. `paper` (160) và `cardboard` (169) chỉ vừa vượt ngưỡng. Lưu ý thêm: clothes/shoes đến từ OI, là ảnh sản phẩm hoặc trong nhà chứ không phải bối cảnh rác.

Fingerprint SHA-256 của các file split (cũng ghi trong YAML):
- train `ad5992df37de58140e36a513c6e712b6b701af340b62566d0928750cabfc265c`
- val `df316653b5fca6a762f71a94f412c9aeb67e17c3767da6b8cc2a21427ecebba9`
- test `d4218618d8ef0db43ba319a757bc35a9d7a7be4e3d768fd89d55505ed017bd02`

### Cấu hình
- `configs/detection_dataset_real_v2.yaml`: 10 lớp, `path ../data/detection`, danh sách `splits_real_v2/{train,val,test}.txt` (dòng dạng `images/real/x.jpg`). Đã kiểm chứng `split_images` / `_label_path` của dự án đọc đúng.
- `configs/detection_dataset_real_v2_ultralytics.yaml`: cùng tập ảnh, nhưng dùng `*_ultra.txt` với dòng dạng `./../images/real/x.jpg`. **Lý do:** Ultralytics 8.4.171 coi dòng tương đối không có `./` là tương đối với CWD, và chỉ thay `./` bằng thư mục chứa file txt (`ultralytics/data/base.py:212`). Vì vậy nếu dùng split cũ dạng `images/...` cho YOLO thì chỉ chạy đúng khi CWD = data/detection. Đã kiểm chứng `check_det_dataset` + `img2label_paths`: toàn bộ ảnh và nhãn đều tồn tại.
- `configs/detection_dataset_real_v2_agnostic.yaml`: 1 lớp `waste`, `path ../data/detection_agnostic`. Cây `images/real` là hardlink tới `data/detection/images/real`. `labels/real` là bản sao của `data/detection/labels_agnostic/real` (mọi class → 0, cộng thêm 16 box TACO ambiguous có IoU < 0.7 so với box sẵn có). Số box: train 3972 / val 906 / test 723.
  - ⚠️ `src/detection/dataset.py::load_detection_dataset_yaml` bắt buộc nc=10 và đúng 10 tên lớp, nên SSDLite Stage‑1 1 lớp cần sửa loader (thuộc phạm vi Agent B). YAML agnostic hiện chỉ dùng được với Ultralytics.

## 4. Kiểm thử
`tests/detection/test_split_real_v2.py` (14 test): không rò rỉ theo tên file, SHA-256 hay nhóm pHash; không có ảnh synthetic/Mendeley; mọi ảnh đều có nhãn hợp lệ (class 0..9, hình học đúng, không có box giữ chỗ); không trùng holdout (so với tập SHA-256 **duy nhất** trong `manifest_hinh_anh_rac.json`: 27 file nhưng chỉ 26 hash vì `images (7).jpg` và `images (13).jpg` giống hệt nhau); ảnh quarantine không lọt vào split; fingerprint YAML khớp; danh sách ultra khớp danh sách chuẩn; loader của dự án đọc được; nhãn agnostic chỉ có class 0 và khớp với bản mirror.

```
.venv\Scripts\python.exe -m pytest tests/detection/test_split_real_v2.py -q
14 passed in 14.54s
```

## 5. Các lệnh đã chạy
```
.venv\Scripts\python.exe reports/reproduce_scripts/data_v2_label_qa.py --overlays 20
.venv\Scripts\python.exe scripts/collect_real_detection_v2.py --workers 16 --oi-cap 300      # log: reports/data_v2/collect_v2_log.txt
.venv\Scripts\python.exe scripts/collect_real_detection_v2.py --recover-orientation          # log: reports/data_v2/collect_v2_recover_log.txt
.venv\Scripts\python.exe scripts/build_real_v2_splits.py                                     # log: reports/data_v2/build_v2_log.txt
.venv\Scripts\python.exe -m pytest tests/detection/test_split_real_v2.py -q                 # reports/data_v2/pytest_split_real_v2.txt
```

## 6. File tạo ra (không sửa split/config cũ)
- Scripts: `scripts/collect_real_detection_v2.py`, `scripts/build_real_v2_splits.py`, `reports/reproduce_scripts/data_v2_label_qa.py`
- Dữ liệu: `data/detection/images/real/*` + `labels/real/*` (chỉ thêm file mới), `data/detection/splits_real_v2/*`, `data/detection/labels_agnostic/real/*`, `data/detection_agnostic/`, `data/detection/quarantine_real_v2.csv`, `data/detection/real_v2_split_manifest.csv`, `data/detection/real_v2_source_manifest.csv`, `data/detection/real_v2_ambiguous_boxes.json`
- Config: `configs/detection_dataset_real_v2.yaml`, `configs/detection_dataset_real_v2_ultralytics.yaml`, `configs/detection_dataset_real_v2_agnostic.yaml`
- Báo cáo: `reports/data_v2/` (label_qa.json, split_real_v2_summary.json, near_duplicate_pairs.json, overlays/, overlays_v2/, log)
- Test: `tests/detection/test_split_real_v2.py`

## 7. Việc còn tồn đọng / khuyến nghị
1. Battery (2 box) và biological (6 box) cần nguồn dữ liệu thật mới, hoặc tạm đánh dấu là "không đánh giá được" trong báo cáo kết quả.
2. 1886 ảnh mới tải (TACO 1253 + 142 ảnh phục hồi hướng, OI 491) mang nhãn chuyển đổi tự động từ annotation chính thức và **chưa được người review** (review_status UNREVIEWED). Mới chỉ xem mẫu khoảng 35 overlay.
3. Clothes/shoes của OI là ảnh không có người nhưng vẫn ở bối cảnh sản phẩm hoặc trong nhà (domain gap so với ảnh rác).
4. Không có ảnh nào trong `D:\Hinh_Anh_Rac` được đọc hay dùng; chỉ dùng SHA-256 từ manifest QA.
