# BÁO CÁO KỸ THUẬT & BÀN GIAO: SPRINT AI-UPGRADE-01 — TASK 5D FIX (BATTERY OBJECT ANNOTATION & QA STATE MACHINE)

**Repository:** `D:\CNTT-KLCN155-waste-detection`  
**Trạng thái Gatekeeper:** `APPROVED_FOR_TRAIN = 0` (`HALTED_PENDING_PM_REVIEW`)  
**Huấn luyện GPU:** `0` (Tuyệt đối chưa chạy fine-tuning YOLOv8n)

---

## 1. TASK 1 — CHÍNH SÁCH ANNOTATION LỚP BATTERY (`0: battery`)

### 1.1. Phân biệt bài toán nguồn Roboflow và bài toán của dự án
- **Bài toán của bộ Roboflow gốc (`Waste-battery-only v1`):** Người tạo dataset gốc đặt 3 lớp (`0: Alkaline`, `1: Alkaline_jp`, `2: LR_Alkaline`) để **nhận diện vùng chữ/ký hiệu in trên vỏ pin** (như chữ `"ALKALINE"`, chữ katakana `"アルカリ"`, hoặc mã `"LR6"` / `"LR03"`). Do đó:
  - Các bounding box gốc chỉ bao quanh cụm chữ (chiếm từ **2,4% đến 22,5% diện tích thực của viên pin**).
  - Trên **48 nhóm ảnh gốc (`base_id`)** (tương ứng **124/350 file**), cùng **1 viên pin vật lý duy nhất** bị gắn từ **2 đến 4 bounding boxes rời rạc** trên các cụm chữ khác nhau.
- **Bài toán của hệ thống CNTT-KLCN155:** **Battery Object Detection (`class 0: battery`)** — phát hiện **toàn bộ vật thể viên pin vật lý** trong ảnh rác thải. Các bounding box chỉ ôm chữ của Roboflow **không phải Ground Truth hợp lệ** và tuyệt đối không được đưa trực tiếp vào huấn luyện.

### 1.2. Quy tắc Ground Truth chuẩn hóa cho lớp `0: battery`
1. **1 Viên Pin Vật Lý = 1 Bounding Box:** Mỗi viên pin vật lý hoàn chỉnh chỉ được gán **duy nhất 1 bounding box** (`class_id = 0`) ôm sát toàn bộ thân trụ/khối pin và hai đầu cực kim loại (cực dương nhô lên và đáy cực âm) nhìn thấy được.
2. **Không tách box theo chữ/logo/rỉ sét:** Chữ in trên thân pin (`ALKALINE`, `アルカリ`, `LR6`), logo hãng, hoặc vết rỉ sét trên vỏ không phải là vật thể độc lập.
3. **Không bao gồm bóng đổ và bụi rỉ rời trên mặt bàn:** Ranh giới box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ dừng đúng tại mép vật lý của vỏ pin và đầu cực, loại trừ vùng bóng đổ xám trên mặt bàn hoặc các hạt bụi rỉ sét rơi tách rời khỏi thân pin (ví dụ: bụi rỉ phía trên đầu cực ở ảnh `171736-085`).
4. **Nhiều box chỉ khi có nhiều viên pin thật:** Chỉ tạo $>1$ box trong một ảnh khi trong ảnh thực sự xuất hiện nhiều viên pin vật lý tách biệt. Toàn bộ **140/140 nhóm `base_id`** của bộ Roboflow này đều chỉ chứa **1 viên pin vật lý duy nhất/ảnh**.
5. **Cấm nhân hệ số cố định:** Mọi nhãn re-annotation phải được đo trực tiếp theo biên pixel vật lý của từng viên pin trong từng ảnh.

---

## 2. TASK 2 — KẾT QUẢ RE-ANNOTATE MINI-PILOT 20 ẢNH ĐẠI DIỆN

- **Thư mục lưu trữ có versioning:** [`data/acquisition_staging/roboflow_battery/reannotated_v1_minipilot20/`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/roboflow_battery/reannotated_v1_minipilot20/)
  - `images/`: 20 ảnh gốc (`768x768`) đại diện cho cả `train` (9 ảnh) và `valid` (11 ảnh).
  - `labels/`: 20 file nhãn YOLO `.txt` mới (`class 0: battery`), mỗi file chứa đúng **1 full-battery bounding box**.
  - `overlays_before_after/`: **20 ảnh đối chiếu BEFORE / AFTER đặt cạnh nhau** (`before_after_01_*.jpg` .. `before_after_20_*.jpg`) kèm 2 bảng tổng hợp (`summary_before_after_sheet_01_of_02.jpg`, `02_of_02.jpg`).
  - `minipilot20_reannotation_manifest.json`: Bản ghi review trực quan từng ảnh (`reviewed_by`, `review_timestamp`, `review_method`, `review_note`, tọa độ cũ vs. mới).
- **Script tái lập:** [`scripts/reannotate_battery_minipilot_20.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/reannotate_battery_minipilot_20.py)

### Bảng đối chiếu 20 ảnh Mini-Pilot (Từ 41 box chữ cũ $\rightarrow$ 20 box toàn bộ viên pin):

| # | `base_id` | Split gốc | Số box cũ (chữ) | Lớp nguồn cũ | Tỷ lệ diện tích box cũ / pin thật | Box mới `[x1, y1, x2, y2]` (px) | YOLO `0 xc yc w h` mới | Đặc điểm vật lý & Ghi chú Review |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **01** | `image-100-_png` *(Ví dụ PM #1)* | `train` | 1 | `[1]` | 4,7% | `[275, 170, 482, 628]` | `0 0.4928 0.5195 0.2695 0.5964` | Pin vàng nghiêng đứng; box cũ chỉ ôm chữ katakana giữa thân. |
| **02** | `image-2024-08-03T171736-085_png` *(Ví dụ PM #2)* | `valid` | 1 | `[1]` | 5,3% | `[298, 170, 450, 648]` | `0 0.4870 0.5326 0.1979 0.6224` | Pin vàng đứng bị rỉ sét nặng; loại trừ bụi rỉ rời trên bàn ($y < 170$). |
| **03** | `image-2024-08-03T182055-514_png` *(Ví dụ PM #3)* | `train` | **2** | `[1, 2]` | 5,4% | `[85, 310, 552, 444]` | `0 0.4147 0.4909 0.6081 0.1745` | Pin đỏ nằm ngang; gộp 2 box chữ rời (`アルカリ` + `LR6`) thành 1 box ôm trọn thân + cực trắng. |
| **04** | `image-58-_png` | `train` | **4** | `[1, 2, 1, 1]` | 2,7% | `[172, 268, 585, 498]` | `0 0.4928 0.4987 0.5378 0.2995` | Pin Mitsubishi đỏ/vàng nghiêng ngang; gộp 4 box chữ nhỏ thành 1 box toàn thân. |
| **05** | `image-62-_png` | `train` | **4** | `[1, 1, 0, 1]` | 2,2% | `[252, 195, 536, 590]` | `0 0.5130 0.5111 0.3698 0.5143` | Pin Mitsubishi đặt chéo; gộp 4 box chữ nhỏ thành 1 box ôm hai đầu cực. |
| **06** | `image-60-_png` | `valid` | **3** | `[1, 1, 2]` | 2,5% | `[270, 182, 506, 594]` | `0 0.5052 0.5052 0.3073 0.5365` | Pin đỏ/vàng nghiêng đứng; gộp 3 box chữ thành 1 box. |
| **07** | `image-59-_png` | `train` | **3** | `[1, 1, 1]` | 2,0% | `[278, 176, 504, 604]` | `0 0.5091 0.5078 0.2943 0.5573` | Pin đỏ/vàng phơi sáng cao (+20% brightness); gộp 3 box chữ thành 1 box. |
| **08** | `image-64-_png` | `train` | **3** | `[1, 1, 2]` | 2,4% | `[174, 256, 642, 426]` | `0 0.5312 0.4440 0.6094 0.2214` | Pin đỏ/vàng nằm ngang; gộp 3 box chữ, loại trừ bóng đổ dưới $y=426$. |
| **09** | `image-2024-08-03T183134-801_png` | `train` | **3** | `[1, 1, 2]` | 4,1% | `[314, 114, 502, 694]` | `0 0.5312 0.5260 0.2448 0.7552` | Pin Panasonic Evolta xanh/vàng đứng; gộp 3 box chữ, bao gồm núm cực dương $y=114$. |
| **10** | `image-2024-08-03T183432-418_png` | `valid` | **2** | `[2, 1]` | 2,7% | `[298, 166, 456, 682]` | `0 0.4909 0.5521 0.2057 0.6719` | Pin Panasonic Evolta đứng (`valid`); gộp 2 box chữ thành 1 box. |
| **11** | `image-2024-08-03T184517-673_png` | `valid` | **2** | `[1, 0]` | 19,4% | `[115, 252, 616, 528]` | `0 0.4759 0.5078 0.6523 0.3594` | Pin vàng/đỏ/xanh nằm nghiêng ngang; gộp 2 box chữ chồng lấp thành 1 box. |
| **12** | `image-88-_png` | `valid` | **2** | `[0, 1]` | 20,4% | `[252, 220, 520, 654]` | `0 0.5026 0.5690 0.3490 0.5651` | Pin đại (D-cell) màu đỏ đứng; gộp 2 box chữ, cắt bỏ bóng đổ bên trái ($x < 252$). |
| **13** | `image-92-_png` | `valid` | **2** | `[0, 1]` | 19,3% | `[252, 205, 500, 636]` | `0 0.4896 0.5475 0.3229 0.5612` | Pin đại (D-cell) màu đỏ đứng bị tróc vỏ; gộp 2 box chữ thành 1 box. |
| **14** | `image-95-_png` | `valid` | **2** | `[0, 1]` | 22,5% | `[188, 166, 574, 596]` | `0 0.4961 0.4961 0.5026 0.5599` | Pin đại (D-cell) màu đỏ đặt chéo; gộp 2 box chữ thành 1 box. |
| **15** | `image-2024-08-03T185333-198_png` | `valid` | **2** | `[1, 0]` | 22,2% | `[304, 140, 476, 688]` | `0 0.5078 0.5391 0.2240 0.7135` | Pin vàng/đỏ/xanh đứng có đầu cực dưới rỉ sét; gộp 2 box chữ dọc thành 1 box. |
| **16** | `image-2024-08-03T171506-124_png` | `train` | 1 | `[1]` | 3,2% | `[164, 186, 592, 656]` | `0 0.4922 0.5482 0.5573 0.6120` | Pin đại màu đỏ bị ăn mòn/rỉ sét rất nặng đặt chéo; mở rộng từ 1 box chữ nhỏ ra toàn thân. |
| **17** | `image-2024-08-03T181349-281_png` | `valid` | 1 | `[0]` | 20,9% | `[330, 196, 448, 656]` | `0 0.5065 0.5547 0.1536 0.5990` | Pin tiểu màu đỏ (`4 ALKALINE`) đứng; ôm trọn thân trụ và hai đầu. |
| **18** | `image-2024-08-03T182313-819_png` | `valid` | 1 | `[0]` | 2,4% | `[295, 140, 478, 638]` | `0 0.5033 0.5065 0.2383 0.6484` | Pin Duracell đen/đồng đứng; box cũ chỉ chiếm 2,4% diện tích thân pin. |
| **19** | `image-87-_png` | `train` | 1 | `[0]` | 19,0% | `[222, 212, 508, 654]` | `0 0.4753 0.5638 0.3724 0.5755` | Pin đại màu đỏ đứng (`train`); ôm trọn thân trụ và núm cực dương. |
| **20** | `image-2024-08-03T183506-087_png` | `valid` | 1 | `[1]` | 2,8% | `[314, 152, 628, 716]` | `0 0.6133 0.5651 0.4089 0.7344` | Pin xanh/vàng nghiêng đứng; mở rộng từ góc chữ nhỏ sang toàn bộ viên pin. |

---

## 3. TASK 3 — PHÂN TÍCH BIẾN THỂ AUGMENTATION, `ORIGINAL_UNKNOWN` & CHỐNG RÒ RỈ PHIÊN CHỤP

File phân tích định lượng: [`data/acquisition_staging/roboflow_battery/augmentation_and_session_leakage_analysis.json`](file:///d:/CNTT-KLCN155-waste-detection/data/acquisition_staging/roboflow_battery/augmentation_and_session_leakage_analysis.json) (tạo bởi [`scripts/analyze_battery_variants_and_sessions.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/analyze_battery_variants_and_sessions.py)).

1. **Tại sao kiểm tra `".rf." in filename` không hợp lệ và phải gán `ORIGINAL_UNKNOWN`:**
   - Toàn bộ **350/350 file** trong bản xuất Roboflow đều có định dạng `<base_id>.rf.<md5_hash>.jpg`.
   - Theo file `README.roboflow.txt` bên trong gói ZIP, Roboflow đã áp dụng:
     - Tiền xử lý cho mọi ảnh: *Auto-orientation of pixel data (with EXIF-orientation stripping)*.
     - Augmentation tạo **3 phiên bản trên mỗi ảnh nguồn trong tập `train`** (`105 base_id × 3 = 315` file): xoay 90° ngẫu nhiên (`none`, `clockwise`, `counter-clockwise`), chỉnh độ sáng ngẫu nhiên `-20%..+20%`, và **rắc nhiễu muối tiêu (salt-and-pepper noise) lên 1,05% tổng số pixel**.
   - Vì Roboflow không xuất kèm ảnh gốc trước khi rắc nhiễu muối tiêu cho tập `train` và đã xóa toàn bộ EXIF, không có bất kỳ file nào có bằng chứng metadata là ảnh nguyên bản chưa can thiệp. Do đó, toàn bộ 350 file (và 140 nhóm `base_id`) được ghi nhận chính thức là:
     - `original_status = "ORIGINAL_UNKNOWN"`
     - Trong mỗi nhóm `base_id` (140 nhóm), chọn đúng **1 ảnh đại diện (`variant_role = "BASE_REPRESENTATIVE"`)** có độ sáng gần mức trung tính của mặt bàn nhất, và loại bỏ hoàn toàn **210 biến thể sinh thêm (`variant_role = "AUGMENTED_VARIANT_EXCLUDED"`)**.

2. **Giải trình chính xác con số 201 cờ trùng gần (`internal_duplicates = 201`):**
   - Trong 350 file, có **0 cặp trùng tuyệt đối SHA-256**, nhưng có **201 file** có khoảng cách `pHash Hamming <= 8` với một file đứng trước nó:
     - **61 trường hợp** là trùng gần **nội bộ trong cùng một `base_id`** (khi 2 hoặc 3 biến thể augmentation của cùng một ảnh gốc trùng góc xoay 90°).
     - **140 trường hợp** là trùng gần **giữa các `base_id` khác nhau** (do người tạo bộ dữ liệu chụp liên tiếp nhiều khung hình của cùng một viên pin trên cùng một mặt bàn xám chiếm >75% diện tích ảnh).

3. **Phát hiện và xử lý rò rỉ phiên chụp / mẫu vật pin giữa `train` và `valid`:**
   - Khi gom 140 `base_id` theo chuỗi thời gian chụp (`2024-08-03T17:15:06` $\rightarrow$ `19:01:15`) và số thứ tự khung hình (`image-58-` $\rightarrow$ `image-100-`), bộ dữ liệu gồm **12 cụm phiên chụp / mẫu viên pin vật lý (`SPECIMEN_01` .. `SPECIMEN_12`)**.
   - Trong bản chia ngẫu nhiên của Roboflow, **11/12 mẫu viên pin vật lý bị chia cắt vào cả `train` lẫn `valid`** (gây ra **65 cặp** trùng gần pHash chuẩn $\le 8$ và **160 cặp** trùng gần bất biến góc xoay $\le 10$ giữa `train` và `valid`).
   - **Giải pháp Split chống rò rỉ (`recommended_leak_free_split`):** Phân tách theo **cụm mẫu vật (Specimen-Isolated Group Split)**:
     - **Train (`111` `base_id` đại diện):** Gồm 9 cụm mẫu vật (`SPECIMEN_01`, `02`, `03`, `05`, `07`, `08`, `09`, `11`, `12`).
     - **Valid (`29` `base_id` đại diện):** Giữ riêng 3 cụm mẫu vật độc lập (`SPECIMEN_04`, `SPECIMEN_06`, `SPECIMEN_10`).
     - **Độ chồng lấp mẫu vật / phiên chụp giữa `train` và `valid`:** **0% (0/12 cụm bị rò rỉ)**.

---

## 4. TASK 4 — SỬA QA STATE MACHINE & REGRESSION TESTS

Đã sửa đổi [`src/detection/safe_dataset_builder.py`](file:///d:/CNTT-KLCN155-waste-detection/src/detection/safe_dataset_builder.py), [`scripts/import_and_qa_roboflow_battery.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/import_and_qa_roboflow_battery.py) và [`scripts/generate_all_140_base_contact_sheets.py`](file:///d:/CNTT-KLCN155-waste-detection/scripts/generate_all_140_base_contact_sheets.py):

1. **Tách biệt 5 trạng thái QA độc lập:**
   - `TECHNICAL_VALID`: Ảnh giải mã hợp lệ, tọa độ YOLO hợp lệ về mặt toán học (`140/140` base images). Không được dùng trạng thái này để đưa vào huấn luyện.
   - `VISUAL_QA_PASS`: Chỉ được gán khi đã kiểm tra trực quan, bbox bao trọn viên pin vật lý và **bắt buộc có `review_evidence` hợp lệ** (`reviewed_by`, `review_timestamp`, `review_note`) — hiện chỉ có đúng **`20/140` ảnh mini-pilot đã re-annotate**.
   - `NEEDS_REANNOTATION`: Trạng thái mặc định cho toàn bộ ảnh Roboflow chưa sửa nhãn (**`120/140` base images**, `330/350` tổng số file). Luôn bị chặn bởi `validate_candidate_for_intake()`.
   - `REJECT`: Ảnh hỏng, lỗi định dạng hoặc rò rỉ (`0` ảnh).
   - `APPROVED_FOR_TRAIN`: Chỉ được bật sau khi PM phê duyệt (`0` ảnh).
2. **Xóa bỏ hoàn toàn việc gán `QA_PASS` mặc định:** Khi chạy `process_roboflow_battery_archive()` trên một gói ZIP mới chưa có bản ghi review độc lập, **100% ảnh hợp lệ kỹ thuật đều nhận `visual_qa = "NEEDS_REANNOTATION"` (`VISUAL_QA_PASS = 0`)**.
3. **Chốt chặn `verify_visual_qa_pass_evidence()`:** Mọi ứng viên khai báo `visual_qa = "VISUAL_QA_PASS"` nhưng thiếu `review_evidence` hoặc thiếu bất kỳ trường nào trong (`reviewed_by`, `review_timestamp`, `review_note`) đều bị `validate_candidate_for_intake()` từ chối ngay lập tức.

---

## 5. TASK 5 — ĐỐI CHIẾU SỐ LIỆU THỐNG KÊ CHÍNH XÁC

### 5.1. Đính chính số lượng Bounding Boxes của tập `test` đóng băng
Trong [`configs/detection_dataset_real_v3_pilot.yaml`](file:///d:/CNTT-KLCN155-waste-detection/configs/detection_dataset_real_v3_pilot.yaml), số lượng box của tập `test` cố định (299 ảnh) có lớp `trash = 139` (đã sửa lỗi đánh máy `130` trong bảng báo cáo cũ), khớp tuyệt đối tổng **720 boxes**:

| Lớp | `0:bat` | `1:bio` | `2:card` | `3:cloth` | `4:glass` | `5:metal` | `6:paper` | `7:plast` | `8:shoes` | `9:trash` | **Tổng cộng** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train v2 (1.398 ảnh)** | 1 | 3 | 169 | 368 | 213 | 420 | 160 | 1.560 | 215 | 854 | **3.963** |
| **Val v2 (299 ảnh)** | 0 | 2 | 39 | 68 | 82 | 74 | 43 | 382 | 41 | 170 | **901** |
| **Test v2 (299 ảnh)** | 1 | 1 | 36 | 91 | 32 | 73 | 34 | 274 | 39 | **139** | **720** |

### 5.2. Thống kê trạng thái thực tế của nguồn dữ liệu Battery Roboflow

| Hạng mục kiểm kê | Số lượng ảnh | Số lượng Box | Trạng thái QA / Huấn luyện |
|---|:---:|:---:|---|
| Tổng số file trong ZIP Roboflow | 350 | 505 *(box chữ cũ)* | `ORIGINAL_UNKNOWN` |
| Biến thể Augmentation bị loại trừ | 210 | 306 *(box chữ cũ)* | `AUGMENTED_VARIANT_EXCLUDED` |
| Tổng số ảnh đại diện theo `base_id` | 140 | 199 *(box chữ cũ)* | `TECHNICAL_VALID` (100% ảnh chỉ có 1 viên pin thật) |
| **Mini-Pilot đã Re-annotate chuẩn (Task 2)** | **20** | **20** *(1 box/pin)* | **`VISUAL_QA_PASS`** *(Có đầy đủ `review_evidence` & BEFORE/AFTER)* |
| **Ảnh còn lại chưa Re-annotate** | **120** | *158 (box chữ cũ)* | **`NEEDS_REANNOTATION`** *(Bị khóa, không đưa vào train)* |
| **Đã phê duyệt đưa vào Huấn luyện** | **0** | **0** | **`APPROVED_FOR_TRAIN = 0`** |

*(Lưu ý: Không tính 199 box chữ cũ hay 140 ảnh chưa sửa nhãn là dữ liệu sẵn sàng huấn luyện. Hiện tại chỉ có **20 ảnh / 20 full-battery boxes** của Mini-Pilot đạt `VISUAL_QA_PASS`, và **120 ảnh base còn lại** đang bị khóa ở trạng thái `NEEDS_REANNOTATION` chờ PM nghiệm thu mẫu 20 ảnh trước khi mở rộng).*

---

## 6. BẰNG CHỨNG KIỂM THỬ PYTEST

Toàn bộ bộ kiểm thử QA và Detection đều đạt **100% PASS** (log chi tiết tại [`reports/task5d_fix_pytest_log.txt`](file:///d:/CNTT-KLCN155-waste-detection/reports/task5d_fix_pytest_log.txt)):
- `tests/qa/test_task5d_fix_reannotation_and_qa.py`: **5/5 PASSED**
- `tests/qa/test_task5d_approval_package.py`: **5/5 PASSED**
- `tests/qa/test_task5c_battery_qa.py`: **5/5 PASSED**
- `tests/qa/test_task5b_hotfixes_and_intake.py`: **22/22 PASSED**
- `tests/detection/`: **62/62 PASSED**
