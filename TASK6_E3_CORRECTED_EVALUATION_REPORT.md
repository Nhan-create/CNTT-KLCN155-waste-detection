# BÁO CÁO NGHIỆM THU TASK 6 (ĐÃ CHUẨN HÓA SỐ LIỆU TASK 6B): YOLOV8N V3 PILOT TRAINING & MODEL EVALUATION

**Dự án:** CNTT-KLCN155 — Hệ thống nhận diện và phân loại rác thải  
**Sprint:** AI-UPGRADE-01 — TASK 6B (Evidence Correction & Reconciliation)  
**Repository:** `D:\CNTT-KLCN155-waste-detection`  
**Ngày cập nhật:** 14/10/2026  
**Nguồn sự thật duy nhất (Single Source of Truth):**
- [`artifacts/detection_v3_pilot/evaluation_e1_vs_e3/e1_vs_e3_comprehensive_evaluation.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/evaluation_e1_vs_e3/e1_vs_e3_comprehensive_evaluation.json)
- [`artifacts/detection_v3_pilot/dataset_v3_materialization_and_preflight_report.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/dataset_v3_materialization_and_preflight_report.json)
- [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/epoch_telemetry.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/E3_yolov8n_pilot/epoch_telemetry.json)
- [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv)

---

## 0. BẢNG ĐÍNH CHÍNH & ĐỒNG BỘ SỐ LIỆU (TASK 6B ERRATA SUMMARY)

Báo cáo này khắc phục triệt để 4 điểm sai lệch giữa văn bản báo cáo cũ và các tệp JSON/CSV thực thi thật (mà **không** sửa đổi bất kỳ tệp raw evaluation JSON, CSV, log lịch sử hay checkpoint nào):

1. **A1 — Holdout Metrics (`evaluation_D_holdout_27` trong `e1_vs_e3_comprehensive_evaluation.json`):**
   - **E1 @512:** `mAP50 = 0.05913945` (`0.05914`), `mAP50-95 = 0.03790507` (`0.03791`), tại `conf=0.25, IoU=0.50`: **`TP=7, FP=37, FN=50`** (`Precision = 0.15909091`, `Recall = 0.12280702`, `F1 = 0.13861386`). *(Báo cáo cũ ghi nhầm sang cấu hình khác 0.04340 / TP=5, FP=41).*
   - **E1 @640:** `mAP50 = 0.05710398` (`0.05710`), `mAP50-95 = 0.02890579` (`0.02891`), tại `conf=0.25, IoU=0.50`: **`TP=5, FP=27, FN=52`** (`Precision = 0.15625000`, `Recall = 0.08771930`, `F1 = 0.11235955`).
   - **E3 @512:** `mAP50 = 0.00822891` (`0.00823`), `mAP50-95 = 0.00288265` (`0.00288`), tại `conf=0.25, IoU=0.50`: **`TP=1, FP=52, FN=56`** (`Precision = 0.01886792`, `Recall = 0.01754386`, `F1 = 0.01818182`).
   - **Chênh lệch chuẩn (`E3 @512 - E1 @512`):** `Delta mAP50 = -0.05091054`, `Delta mAP50-95 = -0.03502242`, `Delta TP = -6`, `Delta FP = +15`, `Delta FN = +6`, `Delta F1 = -0.12043204`.
2. **A2 — Bảng phân bố Per-Class Dataset (`dataset_v3_materialization_and_preflight_report.json`):**
   - Toàn bộ số lượng bounding boxes từng lớp trong Mục 3.2 dưới đây được trích xuất trực tiếp từ `measured_on_disk_splits` và `ultralytics_yolodataset_loaded_verification`, khớp 100% với tổng `Train: 1.509 ảnh / 4.074 boxes`, `Val: 328 ảnh / 930 boxes`, `Test: 299 ảnh / 720 boxes` (loại bỏ hoàn toàn bảng per-class cũ bị lệch phân phối lớp).
3. **A3 — Đối chiếu Nhật ký Huấn luyện (`results.csv`, `epoch_telemetry.json`, `training_log.txt`):**
   - `results.csv` có đúng **22 dòng tương ứng 22 epochs huấn luyện** (`Epoch 1..22`).
   - `epoch_telemetry.json` ghi nhận **Best Epoch = 17** (`val_mAP50 = 0.33459`, `val_mAP50_95 = 0.24153`) và **EarlyStopping kích hoạt tại cuối Epoch 22** (sau `patience = 5` epochs từ 18 đến 22 không vượt qua fitness của Epoch 17).
   - Dòng `Epoch 23/25` trong `training_log.txt` là lượt **validation cuối cùng tự động của Ultralytics trên `best.pt`** sau khi dừng vòng lặp train (trong `epoch_telemetry.json` đã tách riêng vào trường `final_eval_on_best_pt`), **không phải** epoch huấn luyện thứ 23.
   - Mã băm SHA-256 trong `training_log.txt` (`best.pt`: `91efc77df60344a648a4090f384d588e8d27ecf1edc01c11ebdef1dba3ec03e5`, `last.pt`: `e401a1f58a9b4d0de230bbfb693ae64b46c3bd16ded609013f04a377f2b2a295`) là mã băm **trước khi đóng dấu `phase2_metadata`**. Ngay sau dòng log đó, script gọi `_inject_phase2_metadata()` để nhúng thông tin 10 lớp cho `WasteDetector`, tạo ra artifact cuối cùng trên đĩa có SHA-256 chuẩn là `best.pt = e451cb8d0a0faab9374f5dac90811bba681b2827d51aba07f6e6a120b5364169` và `last.pt = bed7b909b3b9b9829c73a2fcce47d482bbb1d2fde77d557c65e2f3e6fc3030e4`.
4. **A4 — Phân biệt rõ Recall theo ngưỡng (PR-Curve Max-F1 Recall vs Recall tại `conf=0.25`):**
   - Trên tập **Validation V3 (`29` GT `battery`)**:
     - **Recall theo đường cong PR của Ultralytics (tại ngưỡng tối ưu F1):** `E1 @512 = 0.00000`, **`E3 @512 = 0.96139`**.
     - **Recall tại `conf=0.25` trong Ma trận nhầm lẫn đa lớp Ultralytics (`confusion_matrix.matrix`):** `E1 @512 = 0/29 = 0.00000`, **`E3 @512 = 25/29 = 0.86207`** (`TP = 25`, `FN = 4` do 4 viên pin có điểm dự đoán lớp `metal` cao hơn khi ghép đa lớp).
     - **Recall tại `conf=0.25, IoU=0.50` khi khớp độc lập từng lớp (`operating_point_conf025_E3_512`):** `E3 @512` đạt `TP = 29/29` (`Recall = 1.00000`), nhưng có `FP = 3` (`Precision = 29/32 = 0.90625`, gồm `1` FP trên ảnh `taco_1405.jpg` của Val V2 và `2` box phụ confidence thấp `0.3751, 0.3385` ở mép bàn trên ảnh `image-2024-08-03T183548-365`).

---

## 1. TASK 1 — BẰNG CHỨNG GIẢI PHÓNG GPU & KIỂM TRA CUDA

File log gốc: [`artifacts/detection_v3_pilot/gpu_preflight_log.txt`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/gpu_preflight_log.txt)

- **Trạng thái `HD-Player.exe` (BlueStacks):** Đã tắt hoàn toàn (`STOPPED_CLEANLY`, `0` tiến trình còn chạy).
- **Trạng thái `nvidia-smi` trước khi huấn luyện:** `0 MiB / 4096 MiB` (`No running processes found`).
- **Xác minh môi trường `.venv`:** `Torch: 2.5.1+cu121`, `CUDA: 12.1`, `Available: True`, `GPU: NVIDIA GeForce RTX 2050`.

---

## 2. TASK 2 — ĐÓNG DẤU PHÊ DUYỆT 140 ẢNH PIN & BẢNG PER-CLASS DATASET V3 CHUẨN XÁC (A2)

Nguồn sự thật: [`artifacts/detection_v3_pilot/dataset_v3_materialization_and_preflight_report.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/dataset_v3_materialization_and_preflight_report.json)

### 2.1. Thống kê phê duyệt dữ liệu pin

| Nguồn dữ liệu | Tổng số ảnh | Số ảnh phê duyệt (`USER_APPROVED_TASK6_PILOT`) | Split đích trong Pilot V3 | Trạng thái |
|---|---:|---:|---|---|
| **Roboflow Battery Canonical Base Images** | `140` | **`140`** | `111` Train + `29` Val | Đã phê duyệt & đưa vào V3 |
| **Roboflow Augmented Variants** | `210` | **`0`** | Không sử dụng | Loại bỏ hoàn toàn (chống rò rỉ) |
| **Batch A Mini-01 Candidates** | `8` (`10` tổng) | **`0`** | Chỉ dùng kiểm thử định tính | Giữ ngoài tập Train/Val V3 |

### 2.2. Bảng phân bố Bounding Boxes từng lớp (Khớp 100% `measured_on_disk_splits` & `YOLODataset`)

| Lớp (`class_id: name`) | Train V2 (Disk) | Train V3 (Disk) | Train V3 (`YOLODataset`) | Val V2 (Disk) | Val V3 (Disk) | Val V3 (`YOLODataset`) | Test Frozen (Disk) | Test Frozen (`YOLODataset`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`0: battery`** | `1` | **`112`** (`+111`) | `112` | `0` | **`29`** (`+29`) | `29` | `1` | `1` |
| **`1: biological`** | `3` | `3` | `3` | `2` | `2` | `2` | `1` | `1` |
| **`2: cardboard`** | `169` | `169` | `169` | `39` | `39` | `39` | `36` | `36` |
| **`3: clothes`** | `368` | `368` | `368` | `68` | `68` | `68` | `91` | `91` |
| **`4: glass`** | `213` | `213` | `213` | `82` | `82` | `82` | `32` | `32` |
| **`5: metal`** | `420` | `420` | `420` | `74` | `74` | `74` | `73` | `73` |
| **`6: paper`** | `160` | `160` | `160` | `43` | `43` | `43` | `34` | `34` |
| **`7: plastic`** | `1560` | `1560` | `1560` | `382` | `382` | `382` | `274` | `274` |
| **`8: shoes`** | `215` | `215` | `215` | `41` | `41` | `41` | `39` | `39` |
| **`9: trash`** | `854` | `854` | `854` | `170` | `170` | `170` | `139` | `139` |
| **Tổng số Bounding Boxes** | **`3963`** | **`4074`** (`+111`) | **`4074`** | **`901`** | **`930`** (`+29`) | **`930`** | **`720`** | **`720`** |
| **Tổng số Ảnh** | **`1398`** | **`1509`** (`+111`) | **`1509`** | **`299`** | **`328`** (`+29`) | **`328`** | **`299`** | **`299`** |
| **Missing / Corrupt / Invalid** | `0` | **`0`** | **`0`** | `0` | **`0`** | **`0`** | **`0`** | **`0`** |

---

## 3. TASK 3 — KẾT QUẢ HUẤN LUYỆN E3 & ĐỐI CHIẾU LOG / CHECKPOINT SHA-256 (A3)

- **Bản ghi đối chiếu Errata JSON:** [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/training_log_errata_and_reconciliation.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/E3_yolov8n_pilot/training_log_errata_and_reconciliation.json)
- **Telemetry JSON:** [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/epoch_telemetry.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/E3_yolov8n_pilot/epoch_telemetry.json)
- **Results CSV (`22` dòng):** [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv)

### 3.1. Bảng đối chiếu SHA-256 trước và sau khi nhúng `phase2_metadata`

| Tệp Checkpoint | Đường dẫn | SHA-256 Raw Ultralytics (trước `phase2_metadata`, ghi trong `training_log.txt`) | SHA-256 Artifact Cuối Cùng trên đĩa (sau `phase2_metadata`, ghi trong `epoch_telemetry.json`) |
|---|---|---|---|
| **E1 Baseline (`best.pt`)** | `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` | `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8` | `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8` *(Bất biến 100%)* |
| **E3 Best (`best.pt` — Epoch 17)** | `artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/best.pt` | `91efc77df60344a648a4090f384d588e8d27ecf1edc01c11ebdef1dba3ec03e5` | **`e451cb8d0a0faab9374f5dac90811bba681b2827d51aba07f6e6a120b5364169`** |
| **E3 Last (`last.pt` — Epoch 22)** | `artifacts/detection_v3_pilot/E3_yolov8n_pilot/weights/last.pt` | `e401a1f58a9b4d0de230bbfb693ae64b46c3bd16ded609013f04a377f2b2a295` | **`bed7b909b3b9b9829c73a2fcce47d482bbb1d2fde77d557c65e2f3e6fc3030e4`** |

### 3.2. Bảng toàn bộ 22 Epochs Huấn luyện (`results.csv` & `epoch_telemetry.json`) + Lượt Final Validation

| Epoch | `box_loss` | `cls_loss` | `dfl_loss` | Val Precision | Val Recall | Val `mAP50` | Val `mAP50-95` | `lr/pg0` | Peak VRAM (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **1** | `1.35048` | `1.99898` | `1.25567` | `0.36283` | `0.29770` | `0.26875` | `0.17653` | `0.000332` | `384.5` |
| **2** | `1.32810` | `1.88484` | `1.24137` | `0.61837` | `0.25756` | `0.28611` | `0.19736` | `0.000663` | `384.5` |
| **3** | `1.35038` | `1.86295` | `1.25957` | `0.58330` | `0.26727` | `0.28306` | `0.17983` | `0.000984` | `393.3` |
| **4** | `1.36734` | `1.87191` | `1.26588` | `0.48408` | `0.31130` | `0.30864` | `0.20760` | `0.000965` | `393.3` |
| **5** | `1.37040` | `1.88122` | `1.27045` | `0.52138` | `0.29808` | `0.29089` | `0.19578` | `0.000939` | `395.7` |
| **6** | `1.33383` | `1.84770` | `1.25170` | `0.46767` | `0.31514` | `0.29209` | `0.19229` | `0.000905` | `395.7` |
| **7** | `1.34970` | `1.81439` | `1.25768` | `0.49608` | `0.31058` | `0.31669` | `0.20500` | `0.000866` | `395.7` |
| **8** | `1.32774` | `1.79440` | `1.25000` | `0.54032` | `0.29276` | `0.30172` | `0.21151` | `0.000821` | `400.1` |
| **9** | `1.30014` | `1.74317` | `1.24009` | `0.49602` | `0.29237` | `0.30721` | `0.21235` | `0.000770` | `400.1` |
| **10** | `1.28628` | `1.71219` | `1.23669` | `0.51987` | `0.30354` | `0.31374` | `0.21711` | `0.000716` | `400.1` |
| **11** | `1.28636` | `1.67549` | `1.22999` | `0.45359` | `0.34090` | `0.30492` | `0.19522` | `0.000658` | `400.1` |
| **12** | `1.29050` | `1.66630` | `1.22511` | `0.51892` | `0.30544` | `0.30803` | `0.20886` | `0.000598` | `400.1` |
| **13** | `1.25276` | `1.61894` | `1.20685` | `0.45354` | `0.33543` | `0.30652` | `0.21470` | `0.000536` | `400.1` |
| **14** | `1.24459` | `1.57683` | `1.19829` | `0.55908` | `0.32696` | `0.32927` | `0.22888` | `0.000474` | `400.1` |
| **15** | `1.24309` | `1.54716` | `1.19446` | `0.51973` | `0.34833` | `0.32380` | `0.22609` | `0.000412` | `400.1` |
| **16** | `1.15642` | `1.40254` | `1.12378` | `0.56835` | `0.34407` | `0.32753` | `0.22430` | `0.000352` | `400.1` |
| **17 **(BEST)**** | `1.14635` | `1.32736` | `1.11290` | `0.56056` | `0.34935` | `0.33459` | `0.24153` | `0.000294` | `400.1` |
| **18** | `1.13030` | `1.32679` | `1.10382` | `0.53956` | `0.36096` | `0.32892` | `0.23190` | `0.000240` | `400.1` |
| **19** | `1.14139` | `1.27675` | `1.11083` | `0.53433` | `0.35691` | `0.32788` | `0.23086` | `0.000189` | `400.1` |
| **20** | `1.11426` | `1.26502` | `1.09405` | `0.51815` | `0.35150` | `0.32439` | `0.23153` | `0.000144` | `400.1` |
| **21** | `1.10565` | `1.22532` | `1.08958` | `0.52686` | `0.35361` | `0.33052` | `0.23446` | `0.000105` | `400.1` |
| **22 **(EARLY STOP)**** | `1.09853` | `1.21707` | `1.09557` | `0.60354` | `0.33034` | `0.32943` | `0.23529` | `0.000071` | `400.1` |
| *Post-Train Final Eval on `best.pt`* *(ghi là `Epoch 23/25` trong `training_log.txt`)* | `1.09853` | `1.21707` | `1.09557` | `0.56357` | `0.34941` | `0.33437` | `0.24263` | `0.000071` | `400.1` |

---

## 4. TASK 4 — ĐÁNH GIÁ CÔNG BẰNG E1 VS E3 TRÊN 4 BENCHMARKS (ĐÃ CHUẨN HÓA A1 & A4)

Nguồn sự thật: [`artifacts/detection_v3_pilot/evaluation_e1_vs_e3/e1_vs_e3_comprehensive_evaluation.json`](file:///d:/CNTT-KLCN155-waste-detection/artifacts/detection_v3_pilot/evaluation_e1_vs_e3/e1_vs_e3_comprehensive_evaluation.json)

### 4.1. Bảng tổng hợp 4 Benchmarks (`E1 @512`, `E1 @640`, `E3 @512` và Chênh lệch `E3@512 - E1@512`)

| Benchmark | Chỉ số đánh giá | E1 (`imgsz=512`) | E1 (`imgsz=640` gốc) | E3 (`imgsz=512`) | Delta (`E3@512 - E1@512`) |
|---|---|---:|---:|---:|---:|
| **A. Validation V3**<br>(`328` ảnh, `930` boxes, `29` GT `battery`) | `mAP50` (10 lớp)<br>`mAP50-95` (10 lớp)<br>`battery` `AP50`<br>`battery` `AP50-95`<br>**`battery` Recall (Ultralytics PR-curve @ max-F1)**<br>**`battery` Recall @ `conf=0.25` (Multi-class Confusion Matrix)**<br>**`battery` Operating Point @ `conf=0.25, IoU=0.50` (Single-class match)** | `0.24775`<br>`0.17152`<br>`0.01287`<br>`0.00935`<br>`0.00000`<br>`0.00000` (`0/29 TP, 29 FN`)<br>`0.00000` (`TP=0, FP=0, FN=29`) | `0.23415`<br>`0.16300`<br>`0.00135`<br>`0.00135`<br>`0.00000`<br>`0.03448` (`1/29 TP, 28 FN`)<br>*(không chạy tách riêng @640)* | **`0.33446`**<br>**`0.24118`**<br>**`0.99500`**<br>**`0.78917`**<br>**`0.96139`**<br>**`0.86207` (`25/29 TP, 4 FN`)**<br>**`1.00000` (`TP=29, FP=3, FN=0, P=0.90625`)** | **`+0.08671`**<br>**`+0.06966`**<br>**`+0.98213`**<br>**`+0.77982`**<br>**`+0.96139`**<br>**`+0.86207` (`+25 TP, -25 FN`)**<br>**`+1.00000` (`+29 TP, +3 FP, -29 FN`)** |
| **B. Validation V2**<br>(`299` ảnh, `901` boxes, `0` GT `battery`) | `mAP50` (9 lớp có GT)<br>`mAP50-95` (9 lớp có GT)<br>FP `battery` @ `conf=0.25` (Single-class `predict`)<br>FP `battery` trong Confusion Matrix (`row 0`) | `0.28671`<br>`0.19880`<br>`0`<br>`5` (`1 card, 1 glass, 1 metal, 2 plastic`) | `0.27492`<br>`0.19324`<br>`0`<br>`5` (`2 card, 1 metal, 2 plastic`) | `0.26204`<br>`0.18332`<br>`1` (`taco_1405.jpg`, conf `0.5943`)<br>`8` (`4 metal, 4 plastic`) | `-0.02467`<br>`-0.01548`<br>`+1`<br>`+3` |
| **C. Internal Test Frozen**<br>(`299` ảnh, `720` boxes, `1` GT `battery`) | `mAP50` (10 lớp)<br>`mAP50-95` (10 lớp)<br>`battery` `AP50` (`taco_0082.jpg`)<br>`battery` `AP50-95`<br>`battery` Recall (PR-curve & `@conf=0.25`) | `0.24131`<br>`0.16918`<br>`0.00000`<br>`0.00000`<br>`0.00000` (`0/1`) | `0.24677`<br>`0.16742`<br>`0.00000`<br>`0.00000`<br>`0.00000` (`0/1`) | `0.21539`<br>`0.14910`<br>`0.01508`<br>`0.01206`<br>`0.00000` (`0/1`) | `-0.02592`<br>`-0.02008`<br>`+0.01508`<br>`+0.01206`<br>`0.00000` |
| **D. External Holdout**<br>(`27` ảnh, `57` GT, `2` GT `battery`) | `mAP50` (10 lớp)<br>`mAP50-95` (10 lớp)<br>Class-agnostic `AP50`<br>**TP / FP / FN @ `conf=0.25, IoU=0.50`**<br>**Precision @ `conf=0.25`**<br>**Recall @ `conf=0.25`**<br>**F1-score @ `conf=0.25`**<br>`battery` TP @ `conf=0.25` (`2` GT) | **`0.05913945`**<br>**`0.03790507`**<br>`0.21501218`<br>**`7 / 37 / 50`**<br>`0.15909091`<br>`0.12280702`<br>`0.13861386`<br>`0 / 2` | **`0.05710398`**<br>**`0.02890579`**<br>`0.19590173`<br>**`5 / 27 / 52`**<br>`0.15625000`<br>`0.08771930`<br>`0.11235955`<br>`0 / 2` | **`0.00822891`**<br>**`0.00288265`**<br>`0.11259657`<br>**`1 / 52 / 56`**<br>`0.01886792`<br>`0.01754386`<br>`0.01818182`<br>`0 / 2` | **`-0.05091054`**<br>**`-0.03502242`**<br>`-0.10241560`<br>**`-6 TP / +15 FP / +6 FN`**<br>`-0.14022298`<br>`-0.10526316`<br>`-0.12043204`<br>`0` |

### 4.2. Bảng chi tiết từng lớp trên Validation V2 (Kiểm tra Catastrophic Forgetting + Phân biệt 2 loại Recall)

| Lớp (`class_name`) | `E1@512` AP50 | `E1@640` AP50 | `E3@512` AP50 | `Delta AP50 (512)` | `E1@512` AP50-95 | `E3@512` AP50-95 | `Delta AP50-95 (512)` | `E1@512` PR-Recall | `E3@512` PR-Recall | `E1@512` Recall@0.25 (`TP/GT`) | `E3@512` Recall@0.25 (`TP/GT`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`biological`** | `0.00000` | `0.00000` | `0.00000` | `+0.00000` | `0.00000` | `0.00000` | `+0.00000` | `0.00000` | `0.00000` | `0.00000` (`0/2`) | `0.00000` (`0/2`) |
| **`cardboard`** | `0.29250` | `0.22413` | `0.21205` | `-0.08045` | `0.23519` | `0.16472` | `-0.07047` | `0.33333` | `0.25774` | `0.30769` (`12/39`) | `0.17949` (`7/39`) |
| **`clothes`** | `0.41877` | `0.42739` | `0.41705` | `-0.00172` | `0.30374` | `0.29291` | `-0.01083` | `0.38235` | `0.39706` | `0.36765` (`25/68`) | `0.36765` (`25/68`) |
| **`glass`** | `0.13217` | `0.14268` | `0.14769` | `+0.01552` | `0.09603` | `0.11035` | `+0.01432` | `0.10976` | `0.14634` | `0.10976` (`9/82`) | `0.17073` (`14/82`) |
| **`metal`** | `0.41332` | `0.37993` | `0.35814` | `-0.05518` | `0.30376` | `0.24119` | `-0.06257` | `0.47297` | `0.49246` | `0.43243` (`32/74`) | `0.48649` (`36/74`) |
| **`paper`** | `0.06276` | `0.07561` | `0.09041` | `+0.02765` | `0.04700` | `0.07426` | `+0.02726` | `0.06977` | `0.11628` | `0.04651` (`2/43`) | `0.00000` (`0/43`) |
| **`plastic`** | `0.39764` | `0.39701` | `0.36468` | `-0.03296` | `0.25568` | `0.23334` | `-0.02234` | `0.41078` | `0.37696` | `0.36387` (`139/382`) | `0.35864` (`137/382`) |
| **`shoes`** | `0.79357` | `0.73782` | `0.72201` | `-0.07156` | `0.51794` | `0.51244` | `-0.00550` | `0.68293` | `0.68293` | `0.60976` (`25/41`) | `0.63415` (`26/41`) |
| **`trash`** | `0.06966` | `0.08973` | `0.04636` | `-0.02330` | `0.02984` | `0.02065` | `-0.00919` | `0.06471` | `0.03529` | `0.05882` (`10/170`) | `0.04118` (`7/170`) |

---

## 5. KẾT LUẬN NGHIỆM THU TASK 6B (PHẦN A)

1. **Tính toàn vẹn dữ liệu & thực thi:** Toàn bộ checkpoint `E1`, dữ liệu `v2`, `ground_truth_hinh_anh_rac.json`, các tệp JSON đánh giá gốc và nhật ký `training_log.txt` được giữ nguyên 100%.
2. **Quyết định triển khai:** **GIỮ NGUYÊN E1 TRÊN STREAMLIT — KHÔNG DEPLOY E3.**
3. **Chuyển tiếp sang Task 7 (Phần B):** Tập trung thu thập, kiểm định visual QA và gán nhãn full-object cho bộ ảnh pin đa dạng ngoài domain Roboflow mặt bàn (nhiều viên pin trong một ảnh, pin cúc áo / 9V / Li-ion, pin rỉ sét/bẩn, pin lẫn rác ngoài trời và hard negatives hình trụ) để giải quyết triệt để giới hạn khái quát hóa của E3.
