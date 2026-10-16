# BÁO CÁO NGHIỆM THU BẢN VÁ TASK 7 FINAL QA PATCH (CHỈ SỬA 2 ẢNH: `task7_30` & `task7_53`)

- **Repository:** `.`
- **Reviewer & Patch Author:** `Antigravity Visual QA + PM Audit Patch (Task 7 Final QA Patch)`
- **Thời gian cập nhật (UTC):** `2026-10-14T06:30:58.493745+00:00`
- **Trạng thái phê duyệt huấn luyện:** **`APPROVED_FOR_TRAIN = 0`** (`False` trên toàn bộ 56 ứng viên)

---

## 1. CHI TIẾT SỬA TỌA ĐỘ BOUNDING BOX TRÊN ẢNH ĐỘ PHÂN GIẢI GỐC

### 1.1. Ảnh `task7_30` — Kodak AA/AAA (`4239x2826`)
- **File ảnh gốc:** `data/acquisition_staging/task7_diverse_battery_batch/images/task7_30_multi_battery_assorted_kodak_aa_and_aaa_alkaline_and_zinc.jpg` (`SHA-256: f556d586a7f9394b2ed0ae22b0fc80cdd167c20cb2d3c2277ba789841cba0e11`)
- **File YOLO label đã cập nhật:** `data/acquisition_staging/task7_diverse_battery_batch/labels/task7_30_multi_battery_assorted_kodak_aa_and_aaa_alkaline_and_zinc.txt` (đúng **6** boxes)
- **Lý do sửa:** Ba viên pin đứng (`#1, #2, #3`) trước đó bị cắt phần cực dương nhô lên ở đỉnh (`ymin` cũ `0.12, 0.19, 0.20`); 3 viên pin nằm nghiêng (`#4, #5, #6`) được tinh chỉnh lại trên lưới tọa độ độ phân giải gốc `4239x2826` để ôm khít toàn bộ thân trụ và đầu cực mà không lấy dư nền trắng.

| Box | Tọa độ `xyxy_norm` TRƯỚC khi sửa | Tọa độ `xyxy_norm` SAU khi sửa | Dòng nhãn YOLO (`class xc yc w h`) SAU khi sửa |
|---:|---|---|---|
| **#1** *(Pin đứng AA trái)* | `[0.080, 0.120, 0.210, 0.720]` | **`[0.073, 0.075, 0.232, 0.710]`** | `0 0.152500 0.392500 0.159000 0.635000` |
| **#2** *(Pin đứng AAA giữa)* | `[0.240, 0.190, 0.340, 0.710]` | **`[0.242, 0.158, 0.343, 0.706]`** | `0 0.292500 0.432000 0.101000 0.548000` |
| **#3** *(Pin đứng AAA phải)* | `[0.350, 0.200, 0.450, 0.710]` | **`[0.360, 0.162, 0.453, 0.704]`** | `0 0.406500 0.433000 0.093000 0.542000` |
| **#4** *(Pin nằm AA trái)* | `[0.380, 0.510, 0.710, 0.900]` | **`[0.383, 0.492, 0.747, 0.912]`** | `0 0.565000 0.702000 0.364000 0.420000` |
| **#5** *(Pin nằm AAA giữa)* | `[0.540, 0.560, 0.850, 0.910]` | **`[0.546, 0.563, 0.838, 0.918]`** | `0 0.692000 0.740500 0.292000 0.355000` |
| **#6** *(Pin nằm AAA phải)* | `[0.660, 0.600, 0.950, 0.920]` | **`[0.675, 0.570, 0.948, 0.918]`** | `0 0.811500 0.744000 0.273000 0.348000` |

---

### 1.2. Ảnh `task7_53` — IKEA LADDA (`1280x960`)
- **File ảnh gốc:** `data/acquisition_staging/task7_diverse_battery_batch/images/task7_53_multi_battery_assorted_ikea_ladda_rechargeable_aa_batteri.jpg` (`SHA-256: f0a2a90c522cc7aa179be13e2d6a357c7066ec38519a6ea8e4c4d975909b47e0`)
- **File YOLO label đã cập nhật:** `data/acquisition_staging/task7_diverse_battery_batch/labels/task7_53_multi_battery_assorted_ikea_ladda_rechargeable_aa_batteri.txt` (đúng **4** boxes)
- **Lý do sửa:** Hai viên pin nằm nghiêng phía trước (`Box #3` và `Box #4`) trước đó bị cắt phần đầu/thân phía trên bên phải (`ymin` cũ `0.63` và `0.66`). Đã đo lại trên ảnh gốc `1280x960` để `Box #3` (`[0.258, 0.530, 0.678, 0.842]`) và `Box #4` (`[0.450, 0.560, 0.868, 0.882]`) bao trọn từ núm cực dương phía dưới bên trái đến hết đầu thân pin phía trên bên phải, đồng thời thu gọn 2 box pin đứng (`#1, #2`) vừa khít phần thân nhìn thấy được.

| Box | Tọa độ `xyxy_norm` TRƯỚC khi sửa | Tọa độ `xyxy_norm` SAU khi sửa | Dòng nhãn YOLO (`class xc yc w h`) SAU khi sửa |
|---:|---|---|---|
| **#1** *(Pin đứng trái)* | `[0.350, 0.100, 0.480, 0.720]` | **`[0.353, 0.122, 0.478, 0.638]`** | `0 0.415500 0.380000 0.125000 0.516000` |
| **#2** *(Pin đứng phải)* | `[0.480, 0.090, 0.590, 0.650]` | **`[0.476, 0.095, 0.592, 0.592]`** | `0 0.534000 0.343500 0.116000 0.497000` |
| **#3** *(Pin nằm trước trái)* | `[0.260, 0.630, 0.660, 0.870]` | **`[0.258, 0.530, 0.678, 0.842]`** | `0 0.468000 0.686000 0.420000 0.312000` |
| **#4** *(Pin nằm trước phải)* | `[0.470, 0.660, 0.860, 0.930]` | **`[0.450, 0.560, 0.868, 0.882]`** | `0 0.659000 0.721000 0.418000 0.322000` |

---

## 2. XÁC NHẬN SỐ LƯỢNG ẢNH / BOX CUỐI CÙNG & KIỂM TRA HỒI QUY

| Chỉ tiêu kiểm tra hồi quy | Kết quả xác nhận |
|---|---|
| `task7_30` số lượng full-object boxes | **6 boxes** (`PASS`) |
| `task7_53` số lượng full-object boxes | **4 boxes** (`PASS`) |
| Tổng số ứng viên `QA_PASS` | **30 ảnh** (**26** ảnh có pin + **4** ảnh hard negative `0` box) |
| Tổng số bounding box `QA_PASS` toàn batch | **61 full-object battery boxes** (`TRAIN_V4`: 19 ảnh / 40 boxes; `VAL_REAL_BATTERY`: 11 ảnh / 21 boxes) |
| Cách ly `NEEDS_MANUAL_REVIEW` & `REJECT` | **10 ảnh `NEEDS_MANUAL_REVIEW`** (`labels_needs_manual_review/`) & **16 ảnh `REJECT`** (không có label trong `labels/`) |
| Trạng thái `approved_for_train` | **`False` (`0`)** trên toàn bộ 56 mục |
| Kiểm định chống rò rỉ (`task7_leakage_report.json`) | **`0` trùng SHA-256**, **`0` cặp `pHash <= 8` ngoại lai**, **`0` cặp rò rỉ giữa `TRAIN_V4` và `VAL_REAL_BATTERY`** |
| Bảo toàn source, author, license & tính tái lập khi chạy lại script | **100% PASS** (`pytest tests/qa/test_task6b_report_and_task7_candidates.py` -> **`4 passed in 0.25s`**) |
