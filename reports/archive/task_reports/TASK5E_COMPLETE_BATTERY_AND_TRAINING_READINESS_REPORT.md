# BÁO CÁO NGHIỆM THU TASK 5E (FINAL DATASET SIGN-OFF FIX) — BẢNG ACCOUNTING CHUẨN & DRY-RUN PILOT V3

**Dự án:** `.`  
**Giai đoạn:** Sprint AI-UPGRADE-01 — Task 5E Final Dataset Sign-Off Fix  
**Trạng thái cổng phê duyệt:** `APPROVED_FOR_TRAIN = 0` (`USER_TRAIN_APPROVAL = PENDING_PM_APPROVAL`)  
**Phần cứng xác minh trực tiếp (`torch.cuda` / `nvidia-smi`):** `NVIDIA GeForce RTX 2050` (`4095.5 MB` / `4 GB VRAM`)  
**Huấn luyện GPU / Sửa đổi Streamlit / Gán nhãn lại 140 ảnh pin:** Chưa thực hiện (Giữ nguyên 100% 140 nhãn pin đã được PM nghiệm thu ĐẠT)

---

## 1. TASK 1 — BẢNG NGUỒN SỰ THẬT DUY NHẤT (SINGLE SOURCE OF TRUTH) TỪ NHÃN THẬT TRÊN ĐĨA

Toàn bộ số liệu dưới đây được đếm trực tiếp bằng hàm `count_yolo_split_labels_directly()` từ các tệp split `data/detection/splits_real_v2/{train,val,test}.txt` và các tệp nhãn thực tế được trỏ tới trong `data/detection/labels/real/*.txt` (đồng bộ 100% với `configs/detection_dataset_real_v3_pilot.yaml` và kết quả thực thi dry-run):

### 1.1. Bảng đối chiếu quy mô Ảnh & Bounding Boxes (`v2` Frozen Baseline vs. `Pilot V3` Battery-Only Extension)

| Tập dữ liệu (Split) | `v2` Frozen Baseline (Số ảnh) | `v2` Frozen Baseline (Tổng số box) | Pin mới bổ sung (`reannotated_v2_full140`) | `batch_a_mini01` (8 ảnh chưa duyệt) | `Pilot V3` Sau Phê Duyệt (Số ảnh) | `Pilot V3` Sau Phê Duyệt (Tổng số box) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | **`1,398` ảnh** *(1 ảnh có pin)* | **`3,963` boxes** | **`+111` ảnh** *(+111 box `battery`)* | `+0` *(Loại khỏi Pilot V3 đầu)* | **`1,509` ảnh** *(112 ảnh có pin)* | **`4,074` boxes** |
| **Validation** | **`299` ảnh** *(0 ảnh có pin)* | **`901` boxes** | **`+29` ảnh** *(+29 box `battery`)* | `+0` | **`328` ảnh** *(29 ảnh có pin)* | **`930` boxes** |
| **Test (Frozen)** | **`299` ảnh** *(1 ảnh có pin)* | **`720` boxes** | `+0` ảnh | `+0` | **`299` ảnh** *(1 ảnh có pin)* | **`720` boxes** |
| **Tổng (`train+val+test`)** | **`1,996` ảnh** | **`5,584` boxes** | **`+140` ảnh (`+140` box)** | **`+0`** | **`2,136` ảnh** | **`5,724` boxes** |

### 1.2. Bảng phân bố Bounding Boxes theo từng lớp (10 Classes) — Đếm trực tiếp từ Label Files

| Class ID | Tên lớp | `v2` Train (Boxes / Ảnh có lớp) | `Pilot V3` Train (Boxes / Ảnh có lớp) | `v2` Val (Boxes / Ảnh có lớp) | `Pilot V3` Val (Boxes / Ảnh có lớp) | `v2` / `Pilot V3` Test Frozen (Boxes / Ảnh có lớp) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **`battery`** | `1` / `1` | **`112` / `112`** *(+111)* | `0` / `0` | **`29` / `29`** *(+29)* | `1` / `1` |
| **1** | `biological` | `3` / `3` | `3` / `3` | `2` / `1` | `2` / `1` | `1` / `1` |
| **2** | `cardboard` | `169` / `126` | `169` / `126` | `39` / `31` | `39` / `31` | `36` / `26` |
| **3** | `clothes` | `368` / `212` | `368` / `212` | `68` / `44` | `68` / `44` | `91` / `45` |
| **4** | `glass` | `213` / `101` | `213` / `101` | `82` / `22` | `82` / `22` | `32` / `22` |
| **5** | `metal` | `420` / `214` | `420` / `214` | `74` / `45` | `74` / `45` | `73` / `43` |
| **6** | `paper` | `160` / `123` | `160` / `123` | `43` / `27` | `43` / `27` | `34` / `28` |
| **7** | `plastic` | `1,560` / `791` | `1,560` / `791` | `382` / `173` | `382` / `173` | `274` / `161` |
| **8** | `shoes` | `215` / `117` | `215` / `117` | `41` / `23` | `41` / `23` | `39` / `23` |
| **9** | `trash` | `854` / `294` | `854` / `294` | `170` / `64` | `170` / `64` | `139` / `63` |
| **Tổng** | **10 lớp** | **`3,963` boxes (`1,398` ảnh)** | **`4,074` boxes (`1,509` ảnh)** | **`901` boxes (`299` ảnh)** | **`930` boxes (`328` ảnh)** | **`720` boxes (`299` ảnh)** |

### 1.3. Đối chiếu Ground Truth tập Holdout 27 ảnh (`data/holdout`) — Giải trình con số `33` vs `57`

- **Manifest gốc đã đóng băng (`reports/qa_acceptance/ground_truth_hinh_anh_rac.json`, `version: 1.0_FROZEN`):**
  - Tổng số ảnh: **`27` ảnh**.
  - Tổng số đối tượng Ground Truth: **`57` objects** (`battery: 2, biological: 3, cardboard: 3, clothes: 8, glass: 11, metal: 4, paper: 2, plastic: 17, shoes: 6, trash: 1`).
- **Manifest đề xuất chuẩn hóa (`reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json`, `version: v2.0-proposed-standardized`, trạng thái `PROPOSED_ONLY_NOT_FROZEN`):**
  - Giữ nguyên toàn bộ **`57` bản ghi gốc**, trong đó phân loại **`45` đối tượng active cho đánh giá discrete-instance** (`39` đối tượng `ACCEPTED` rời rạc + `6` đối tượng đang tranh chấp) và `12` đối tượng inactive (`8` `FLAGGED_UNSEGREGATED_PILE` + `4` `QUARANTINED_*`).
- **Nguyên nhân xuất hiện số `33` trước đây:** Con số `33` chỉ là một hằng số hardcode nhầm trong phiên bản cũ của script `validate_pilot_v3_pretraining_readiness.py` (dòng 315), không tồn tại trong bất kỳ manifest Ground Truth nào của `data/holdout`. Đã xóa bỏ hằng số hardcode này và đối chiếu trực tiếp từ tệp JSON gốc mà **không sửa đổi bất kỳ tệp Ground Truth nào**.

---

## 2. TASK 2 — XÁC MINH PHÁP Y 8 ẢNH BỔ SUNG (`batch_a_mini01`) & QUYẾT ĐỊNH LOẠI KHỎI PILOT V3 ĐẦU TIÊN

Đã kiểm tra trực tiếp từng tệp ảnh và nhãn trong `data/acquisition_staging/batch_a_mini01/manifest.json` (`10` ảnh tổng cộng, trong đó có `8` ảnh mang trạng thái pass QA và `2` ảnh bị khóa) và xuất báo cáo chi tiết tại `data/acquisition_staging/additional_8_images_forensic_verification.json`:

| STT | Candidate ID | SHA-256 (16 ký tự đầu) | Phân loại thực chất | Số Box & Lớp thực tế trên đĩa | Trạng thái `visual_qa` | Phê duyệt (`approved_for_train`) | Quyết định cho Pilot V3 |
| :---: | :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| 1 | `wiki_neg_brick_wall_clean` | `25899ccf6521d09a...` | **`TRUE_CLEAN_NEGATIVE`** | `0` box (`0` byte) | `QA_PASS_CLEAN_NEGATIVE` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 2 | `wiki_neg_utility_box_corner_05` | `1848b7ca08ab4099...` | **`TRUE_CLEAN_NEGATIVE`** | `0` box (`0` byte) | `QA_PASS_CLEAN_NEGATIVE` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 3 | `wiki_bio_apple_core_01` | `168dac8d2a142a13...` | **`POSITIVE_WASTE` (In-the-wild)** | `1` box (`1: biological`) | `QA_PASS_SINGLE_OBJECT` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 4 | `wiki_bio_banana_peel_bucharest_02` | `8a36952731ef28ea...` | **`POSITIVE_WASTE` (In-the-wild)** | `1` box (`1: biological`) | `QA_PASS` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 5 | `wiki_bio_banana_peel_bus_stop_03` | `5a130eaef7f65f99...` | **`POSITIVE_WASTE` (In-the-wild)** | `1` box (`1: biological`) | `QA_PASS` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 6 | `wiki_multi_banana_and_cup_01` | `3a5492661f8f024b...` | **`POSITIVE_WASTE` (In-the-wild)** | `2` boxes (`1: biological`, `6: paper`) | `QA_PASS` | `False` | **Loại khỏi Pilot V3 đầu** *(Chờ duyệt riêng)* |
| 7 | `wiki_bio_banana_peel_oak_hill_05` | `a820b05e365424c0...` | **`POSITIVE_WASTE` (`OBJECT_ONLY`)** | `1` box (`1: biological`) | `QA_PASS_OBJECT_ONLY_WHITE_BG` | `False` | **Bị chặn mặc định (`OBJECT_ONLY` slice)** |
| 8 | `wiki_bio_banana_peel_sidewalk_06` | `366e0ff19bdfdabe...` | **`POSITIVE_WASTE` (`OBJECT_ONLY`)** | `1` box (`1: biological`) | `QA_PASS_OBJECT_ONLY_WHITE_BG` | `False` | **Bị chặn mặc định (`OBJECT_ONLY` slice)** |
| 9 | `wiki_bat_corroded_screwdriver` | `94871903adebb340...` | **`QUARANTINED_BLOCKED`** | `0` box *(Nhãn rỗng do quarantine)* | `QUARANTINED_AMBIGUOUS_BATTERY_ASSEMBLY` | `False` | **BỊ KHÓA TUYỆT ĐỐI (Cấm làm negative)** |
| 10 | `wiki_bio_food_scraps_bin_09` | `5e4f022f9a490ae3...` | **`PENDING_BLOCKED`** | `0` box *(Nhãn rỗng do pending)* | `PENDING_ANNOTATION` | `False` | **BỊ KHÓA TUYỆT ĐỐI (Cấm làm negative)** |

- **Giải trình sai lệch trước đây:** Báo cáo cũ ghi nhầm cả 8 ảnh `visual_qa_pass` trong `batch_a_mini01` là clean negative (0 box), trong khi YAML cũ giữ dự phóng lỗi thời `+6 biological, +3 glass`. Thực tế trên đĩa chỉ có `2` ảnh là `TRUE_CLEAN_NEGATIVE`, `4` ảnh là `POSITIVE_WASTE` thực tế ngoài trời (`4 biological + 1 paper`), và `2` ảnh là `OBJECT_ONLY` trên nền giấy trắng.
- **Quyết định theo đúng chỉ thị PM:** Do cả 8 ảnh này đều chưa có phê duyệt (`approved_for_train = False`) và thuộc luồng thu thập khác, **đã loại toàn bộ 8 ảnh này khỏi Pilot V3 đầu tiên**, giúp Pilot V3 trở thành bản mở rộng **`Battery-Only Extension` thuần khiết trên nền `v2`** (`v2 + 140 ảnh pin đã nghiệm thu`).

---

## 3. TASK 3 — KIỂM TRA DATASET PILOT THỰC THI (EXECUTABLE BUILDER & CONTROLLED DRY-RUN)

Đã bổ sung hàm tạo dataset thực thi `build_pilot_v3_battery_extension_dataset()` trong `src/detection/safe_dataset_builder.py` và chạy kiểm chứng Dry-Run trong thư mục tạm có kiểm soát (`data/acquisition_staging/roboflow_battery/pilot_v3_dryrun_execution_log.json`):

1. **Trước khi có phê duyệt (`approved_for_train = 0` trên đĩa thật):**
   - Hàm builder trả về `PROPOSAL_ONLY_NO_APPROVED_CANDIDATES`, giữ nguyên `data/detection_upgrade_v3_pilot/splits/{train,val,test}.txt` ở mức `1,398 / 299 / 299` ảnh của `v2` (`0` ảnh chưa duyệt lọt vào staging).
2. **Dry-Run có kiểm soát (Controlled Temporary Fixture — Không chạm vào cờ approval thật):**
   - Sao chép manifest sang `tempfile.TemporaryDirectory`, mô phỏng trạng thái sau khi PM duyệt 140 ảnh pin, chạy `build_pilot_v3_battery_extension_dataset()` và kiểm tra trực tiếp bằng `ultralytics.data.utils.check_det_dataset()` + `img2label_paths()`:
     - **Train (`splits/train.txt`):** Resolve thành công `1,509 / 1,509` ảnh và `1,509 / 1,509` tệp nhãn (`0` missing), đếm trực tiếp từ các tệp nhãn thu được **`4,074` boxes** (`battery = 112`).
     - **Val (`splits/val.txt`):** Resolve thành công `328 / 328` ảnh và `328 / 328` tệp nhãn (`0` missing), đếm trực tiếp từ các tệp nhãn thu được **`930` boxes** (`battery = 29`).
     - **Test (`splits/test.txt`):** Resolve thành công `299 / 299` ảnh và `299 / 299` tệp nhãn (`0` missing), đếm trực tiếp từ các tệp nhãn thu được **`720` boxes** (`battery = 1`).
     - **Tách biệt Specimen giữa Train và Val:** `7` viên pin tập Train và `3` viên pin tập Val tách biệt tuyệt đối (`overlapping_specimens = []`).
     - **Bảo toàn trạng thái thật:** SHA-256 của `full140_reannotation_manifest.json` trước và sau dry-run khớp 100% (`92be49d6...`), `APPROVED_FOR_TRAIN = 0`.

---

## 4. TASK 4 — CẤU HÌNH HUẤN LUYỆN (`configs/training_yolov8n_v3_pilot.yaml`)

- **Thiết bị GPU thực tế đã kiểm chứng:** `NVIDIA GeForce RTX 2050` (`4095.5 MB` VRAM).
- **Checkpoint khởi tạo (Bất biến):** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`.
- **Dataset YAML:** `configs/detection_dataset_real_v3_pilot.yaml` (`1,509` train / `328` val / `299` test).
- **Siêu tham số an toàn VRAM:** `imgsz=512`, `batch=4`, `epochs=25`, `patience=5` (early stopping chỉ dựa trên tập `val`, tuyệt đối không dùng `test` hay `holdout`), `cache=False`, `workers=2`, `amp=True`, `optimizer=AdamW`, `lr0=0.0005`, `lrf=0.05`.
- **Thư mục đầu ra cách ly:** `project: artifacts/detection_v3_pilot`, `name: E3_yolov8n_pilot`.
- **Danh sách xin phê duyệt:** Đúng **`140` ảnh pin đại diện gốc đã re-annotate (`111` ảnh Train + `29` ảnh Validation)** trong `data/acquisition_staging/roboflow_battery/training_approval_candidate_list.json`.
