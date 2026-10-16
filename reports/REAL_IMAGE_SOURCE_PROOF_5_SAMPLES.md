# BÁO CÁO XÁC MINH NGUỒN ẢNH THẬT: 5 MẪU CHỨNG MINH ĐỘC LẬP
**Dự án:** Hệ thống nhận diện rác thải đa nguồn KLCN155  
**Repository:** `.`  
**Thời điểm thực thi:** 2026-10-12  
**Mục tiêu duy nhất:** Chứng minh 5 bản ghi nguồn THẬT, hạ cấp toàn bộ 35 ID cũ về `SOURCE_UNVERIFIED`, và siết chặt toàn diện logic phê duyệt train.

---

## 1. THỰC HIỆN YÊU CẦU 1 & 2: HẠ CẤP 35 ID CŨ VỀ SOURCE_UNVERIFIED

Thực hiện nghiêm túc kết luận của reviewer độc lập:
1. **Hạ cấp toàn bộ 35 mục cũ:** Toàn bộ 35 mục trong `data/acquisition_staging/manifests/candidate_acquisition_manifest.json` đã được chuyển trạng thái từ `SOURCE_IDENTIFIED` xuống **`SOURCE_UNVERIFIED`** (thuộc giai đoạn **`ACQUISITION_TARGET`**).
2. **Loại bỏ tính chất suy đoán / patterned:** Ghi nhận lý do hạ cấp rõ ràng: các ID cũ được sinh theo mẫu quy ước thay vì được trích xuất từ bản ghi metadata nguồn thực tế; toàn bộ bị khóa không cho phép tải cho đến khi có bản ghi chứng minh độc lập.
3. **Tuyệt đối không bịa đặt:** Không tiếp tục sáng tạo `source_image_id`, filename, URL ảnh, kích thước, license hay nội dung ảnh suy diễn.

### Bảng đối chiếu 35 mục cũ đã bị hạ cấp

| STT | Mã tệp cũ | ID cũ đã bị từ chối | Nguồn | Trạng thái mới | Lý do hạ cấp |
|:---:|:---|:---|:---|:---:|:---|
| 1 | `rf_battery_orig_001.jpg` | `rf_wbo_1_001` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 2 | `rf_battery_orig_002.jpg` | `rf_wbo_1_002` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 3 | `rf_battery_orig_003.jpg` | `rf_wbo_1_003` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 4 | `rf_battery_orig_004.jpg` | `rf_wbo_1_004` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 5 | `rf_battery_orig_005.jpg` | `rf_wbo_1_005` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 6 | `rf_battery_orig_006.jpg` | `rf_wbo_1_006` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 7 | `rf_battery_orig_007.jpg` | `rf_wbo_1_007` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 8 | `rf_battery_orig_008.jpg` | `rf_wbo_1_008` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 9 | `rf_battery_orig_009.jpg` | `rf_wbo_1_009` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 10 | `rf_battery_orig_010.jpg` | `rf_wbo_1_010` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 11 | `rf_battery_orig_011.jpg` | `rf_wbo_1_011` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 12 | `rf_battery_orig_012.jpg` | `rf_wbo_1_012` | Roboflow | `SOURCE_UNVERIFIED` | ID tổng hợp; không trích từ bản ghi file Roboflow thật |
| 13 | `oi_bio_banana_001.jpg` | `0a1b2c3d4e5f6001` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 14 | `oi_bio_apple_002.jpg` | `1b2c3d4e5f600201` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 15 | `oi_bio_orange_003.jpg` | `2c3d4e5f60030102` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 16 | `oi_bio_bread_004.jpg` | `3d4e5f6004010203` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 17 | `oi_bio_veg_005.jpg` | `4e5f600501020304` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 18 | `oi_bio_leftover_006.jpg` | `5f60060102030405` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 19 | `oi_bio_foodwaste_007.jpg` | `6007010203040506` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 20 | `oi_bio_fruitwaste_008.jpg` | `7008010203040507` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 21 | `oi_bio_leaves_009.jpg` | `8009010203040508` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 22 | `oi_bio_compost_010.jpg` | `9010010203040509` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 23 | `oi_bio_organic_011.jpg` | `a01101020304050a` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 24 | `oi_bio_rotten_012.jpg` | `b01201020304050b` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 25 | `oi_neg_sidewalk_001.jpg` | `c00101020304050c` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 26 | `oi_neg_asphalt_002.jpg` | `d00201020304050d` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 27 | `oi_neg_lawn_003.jpg` | `e00301020304050e` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 28 | `oi_neg_pavement_004.jpg` | `f00401020304050f` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 29 | `oi_neg_curb_005.jpg` | `0105010203040510` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 30 | `oi_neg_cleanfloor_006.jpg` | `0206010203040511` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 31 | `oi_clutter_bin_001.jpg` | `0307010203040512` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 32 | `oi_clutter_street_002.jpg` | `0408010203040513` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 33 | `oi_clutter_park_003.jpg` | `0509010203040514` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 34 | `oi_clutter_yard_004.jpg` | `0610010203040515` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |
| 35 | `oi_clutter_pile_005.jpg` | `0711010203040516` | OpenImages | `SOURCE_UNVERIFIED` | Hex tuần tự nhân tạo; không có trong OpenImages validation CSV |

---

## 2. NĂM BẢN GHI NGUỒN THẬT XÁC THỰC (5 AUTHENTIC SOURCE SAMPLES)

Toàn bộ 5 bản ghi dưới đây được lưu trong file cấu trúc máy đọc:  
`data/acquisition_staging/manifests/verified_source_proof_5_samples.json`

---

### MẪU 1: PIN THẢI 1 (ROBOFLOW) — BLOCKED_REQUIRES_ACCESS
- **Lớp mục tiêu:** `battery` (Pin)
- **Bộ dữ liệu nguồn:** Roboflow Universe `waste-battery-only/1`
- **URL dự án nguồn:** `https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1`
- **Workspace:** `battery-detection-yuv0z` | **Project:** `waste-battery-only` | **Version:** `1`
- **Exact Source ID:** **`BLOCKED_REQUIRES_ACCESS`**
- **Filename / URL ảnh:** **`BLOCKED_REQUIRES_ACCESS`**
- **Giấy phép công bố:** `CC BY 4.0` (khai báo tại trang phát hành dataset Roboflow)
- **Trạng thái truy cập:** **`BLOCKED_REQUIRES_ACCESS`**
- **Bằng chứng kỹ thuật thực tế (Raw Evidence):**
  Lệnh kiểm tra:
  ```python
  urllib.request.urlopen("https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1")
  ```
  **Kết quả:**
  ```text
  urllib.error.HTTPError: HTTP Error 403: Forbidden
  ```
- **Nguyên nhân kỹ thuật:** Hệ thống Roboflow Universe sử dụng Cloudflare WAF / Anti-Bot Protection. Việc tải dataset hoặc truy cập danh sách từng ảnh yêu cầu:
  1. Người dùng đăng nhập qua trình duyệt và xuất tệp ZIP tải về thủ công, HOẶC
  2. Cung cấp mã `ROBOFLOW_API_KEY` hợp lệ qua API chính thức.
- **Tuân thủ quy tắc:** **KHÔNG tạo ID thay thế.** Ghi nhận trung thực `BLOCKED_REQUIRES_ACCESS`.
- **Trạng thái ảnh trên đĩa:** `CHƯA TẢI` (0 byte trên đĩa).

---

### MẪU 2: PIN THẢI 2 (ROBOFLOW) — BLOCKED_REQUIRES_ACCESS
- **Lớp mục tiêu:** `battery` (Pin)
- **Bộ dữ liệu nguồn:** Roboflow Universe `waste-battery-only/1`
- **URL dự án nguồn:** `https://universe.roboflow.com/battery-detection-yuv0z/waste-battery-only/dataset/1`
- **Exact Source ID:** **`BLOCKED_REQUIRES_ACCESS`**
- **Filename / URL ảnh:** **`BLOCKED_REQUIRES_ACCESS`**
- **Giấy phép:** `CC BY 4.0`
- **Trạng thái truy cập:** **`BLOCKED_REQUIRES_ACCESS`**
- **Bằng chứng kỹ thuật:** Cùng bị chặn bởi Cloudflare HTTP 403 Forbidden.
- **Tuân thủ quy tắc:** **KHÔNG tạo ID thay thế.**
- **Trạng thái ảnh trên đĩa:** `CHƯA TẢI` (0 byte trên đĩa).

---

### MẪU 3: RÁC SINH HỌC 1 (OPEN IMAGES V7 VALIDATION) — VERIFIED
- **Lớp mục tiêu:** `biological` (Rác sinh học / Táo)
- **Bộ dữ liệu nguồn:** Open Images V7 (Tập Validation)
- **Exact Source ID:** **`078821d86db99fc9`** (16 ký tự hex thực tế)
- **Tên tệp chuẩn hóa đề xuất:** `oi_078821d86db99fc9.jpg`
- **File CSV bằng chứng nguồn trên đĩa:** `data/external/waste-training/openimages-validation-boxes.csv`
- **Dòng raw CSV trong `openimages-validation-boxes.csv`:**
  ```csv
  078821d86db99fc9,xclick,/m/014j1m,1,0.244792,0.542847,0.092857,0.539823,0,0,0,0,0
  078821d86db99fc9,xclick,/m/014j1m,1,0.383545,0.697571,0.475664,0.942478,0,0,0,0,0
  078821d86db99fc9,xclick,/m/07j7r,1,0,0.998535,0,0.995575,0,0,0,0,0
  ```
- **Chi tiết Annotation thực tế:**
  - Box 1: Nhãn `/m/014j1m` (Apple), Conf: 1.0, Bbox: `[XMin: 0.244792, XMax: 0.542847, YMin: 0.092857, YMax: 0.539823]`, IsGroupOf: 0, IsDepiction: 0 -> **Target Class: 1 (biological)**
  - Box 2: Nhãn `/m/014j1m` (Apple), Conf: 1.0, Bbox: `[XMin: 0.383545, XMax: 0.697571, YMin: 0.475664, YMax: 0.942478]`, IsGroupOf: 0, IsDepiction: 0 -> **Target Class: 1 (biological)**
  - Box 3: Nhãn `/m/07j7r` (Tree), Conf: 1.0, Bbox nền -> Background
  - Số nhãn con người (Human/Person/Clothing): **0** (không có người, tay hay trang phục).
- **Metadata chính thức trích từ `validation-images-with-rotation.csv` (Open Images):**
  - **Tiêu đề gốc:** `Omenapuu Åkerö`
  - **Tác giả:** `Matleena Laakso`
  - **Trang Flickr gốc:** `https://www.flickr.com/photos/matleenalaakso/8154112499`
  - **URL ảnh Flickr gốc:** `https://c2.staticflickr.com/8/7261/8154112499_cd0d5a31d0_o.jpg`
  - **Giấy phép:** `https://creativecommons.org/licenses/by/2.0/` (CC-BY 2.0)
  - **Dung lượng khai báo:** 6,184,508 bytes (~6.18 MB)
  - **Mã MD5 gốc:** `Me+e2gfBRtmot86R+3g6jw==`
  - **Rotation:** `0.0`
- **URL phân phối AWS S3 chính thức:**  
  `https://open-images-dataset.s3.amazonaws.com/validation/078821d86db99fc9.jpg`
- **Kiểm chứng URL:** HTTP HEAD trả về **`200 OK`**, Content-Length: **`189,451`** bytes, Content-Type: `image/jpeg`.
- **Trạng thái ảnh trên đĩa:** `CHƯA TẢI` (0 byte trên đĩa).
- **Cam kết không suy đoán nội dung:** Nội dung được ghi nhận trung thực và duy nhất dựa trên annotation: gồm 2 hộp đối tượng Apple độc lập và 1 hộp Tree nền, tiêu đề gốc tiếng Phần Lan: "Omenapuu Åkerö" (Cây táo giống Åkerö).

---

### MẪU 4: RÁC SINH HỌC 2 (OPEN IMAGES V7 VALIDATION) — VERIFIED
- **Lớp mục tiêu:** `biological` (Rác sinh học / Chuối)
- **Bộ dữ liệu nguồn:** Open Images V7 (Tập Validation)
- **Exact Source ID:** **`74a44994e3d6b962`** (16 ký tự hex thực tế)
- **Tên tệp chuẩn hóa đề xuất:** `oi_74a44994e3d6b962.jpg`
- **File CSV bằng chứng nguồn trên đĩa:** `data/external/waste-training/openimages-validation-boxes.csv`
- **Dòng raw CSV trong `openimages-validation-boxes.csv`:**
  ```csv
  74a44994e3d6b962,xclick,/m/01dwwc,1,0,0.870833,0.246875,0.74375,0,0,0,0,0
  74a44994e3d6b962,xclick,/m/0270h,1,0,0.85,0.2,0.742188,0,0,0,0,0
  74a44994e3d6b962,xclick,/m/09qck,1,0.108333,0.425,0.290625,0.448438,0,0,0,0,0
  74a44994e3d6b962,xclick,/m/09qck,1,0.23125,0.466667,0.201562,0.310937,0,0,0,0,0
  74a44994e3d6b962,xclick,/m/09qck,1,0.322917,0.716667,0.232812,0.41875,0,0,0,0,0
  74a44994e3d6b962,xclick,/m/09qck,1,0.535417,0.74375,0.220312,0.367188,0,0,0,0,0
  ```
- **Chi tiết Annotation thực tế:**
  - Box 1: Nhãn `/m/09qck` (Banana), Conf: 1.0, Bbox: `[XMin: 0.108333, XMax: 0.425000, YMin: 0.290625, YMax: 0.448438]`, Grp: 0, Dep: 0 -> **Target Class: 1 (biological)**
  - Box 2: Nhãn `/m/09qck` (Banana), Conf: 1.0, Bbox: `[XMin: 0.231250, XMax: 0.466667, YMin: 0.201562, YMax: 0.310937]`, Grp: 0, Dep: 0 -> **Target Class: 1 (biological)**
  - Box 3: Nhãn `/m/09qck` (Banana), Conf: 1.0, Bbox: `[XMin: 0.322917, XMax: 0.716667, YMin: 0.232812, YMax: 0.418750]`, Grp: 0, Dep: 0 -> **Target Class: 1 (biological)**
  - Box 4: Nhãn `/m/09qck` (Banana), Conf: 1.0, Bbox: `[XMin: 0.535417, XMax: 0.743750, YMin: 0.220312, YMax: 0.367188]`, Grp: 0, Dep: 0 -> **Target Class: 1 (biological)**
  - Các nhãn khác: `/m/01dwwc` (Pancake), `/m/0270h` (Dessert).
  - Số nhãn con người: **0**.
- **Metadata chính thức trích từ `validation-images-with-rotation.csv` (Open Images):**
  - **Tiêu đề gốc:** `Vegan Lemon Cornmeal Pancakes`
  - **Tác giả:** `tomatoes and friends`
  - **Trang Flickr gốc:** `https://www.flickr.com/photos/tomatoesandfriends/7206585280`
  - **URL ảnh Flickr gốc:** `https://farm3.staticflickr.com/7242/7206585280_2bb38d9e8d_o.jpg`
  - **Giấy phép:** `https://creativecommons.org/licenses/by/2.0/` (CC-BY 2.0)
  - **Dung lượng khai báo:** 6,081,404 bytes (~6.08 MB)
  - **Mã MD5 gốc:** `lVqUp8fHLbr6xnruF+y43Q==`
  - **Rotation:** `0.0`
- **URL phân phối AWS S3 chính thức:**  
  `https://open-images-dataset.s3.amazonaws.com/validation/74a44994e3d6b962.jpg`
- **Kiểm chứng URL:** HTTP HEAD trả về **`200 OK`**, Content-Length: **`139,446`** bytes, Content-Type: `image/jpeg`.
- **Trạng thái ảnh trên đĩa:** `CHƯA TẢI` (0 byte trên đĩa).
- **Cam kết không suy đoán nội dung:** Báo cáo căn cứ chuẩn xác theo 4 bounding box quả chuối riêng biệt và tiêu đề gốc của tác giả.

---

### MẪU 5: CẢNH NỀN ÂM TÍNH 1 (OPEN IMAGES V7 VALIDATION) — VERIFIED
- **Lớp mục tiêu:** `hard_negatives` (Nền kiến trúc không chứa rác)
- **Bộ dữ liệu nguồn:** Open Images V7 (Tập Validation)
- **Exact Source ID:** **`e29db1c1963628e7`** (16 ký tự hex thực tế)
- **Tên tệp chuẩn hóa đề xuất:** `oi_e29db1c1963628e7.jpg`
- **File CSV bằng chứng nguồn trên đĩa:** `data/external/waste-training/openimages-validation-boxes.csv`
- **Dòng raw CSV trong `openimages-validation-boxes.csv`:**
  ```csv
  e29db1c1963628e7,xclick,/m/0cgh4,1,0,0.999219,0,0.998958,0,0,0,0,0
  ```
- **Chi tiết Annotation thực tế:**
  - Box 1: Nhãn `/m/0cgh4` (Building), Conf: 1.0, Bbox: `[XMin: 0.000000, XMax: 0.999219, YMin: 0.000000, YMax: 0.998958]` -> Background
  - Số nhãn thuộc 10 lớp rác thải (Waste objects): **0** (không có chai, lon, túi nilon, giấy, pin, rác sinh học).
  - Số nhãn con người: **0**.
  - Quy ước nhãn file train: File text rỗng đúng 0 byte (`empty .txt`), phục vụ làm mẫu âm thật để huấn luyện ức chế False Positives trên nền gạch/tường/nhà cửa.
- **Metadata chính thức trích từ `validation-images-with-rotation.csv` (Open Images):**
  - **Tiêu đề gốc:** `Ronda`
  - **Tác giả:** `Manuela Aguilera`
  - **Trang Flickr gốc:** `https://www.flickr.com/photos/94570646@N03/8621828727`
  - **URL ảnh Flickr gốc:** `https://farm8.staticflickr.com/8406/8621828727_f570169d39_o.jpg`
  - **Giấy phép:** `https://creativecommons.org/licenses/by/2.0/` (CC-BY 2.0)
  - **Dung lượng khai báo:** 71,473 bytes (~71.5 KB)
  - **Mã MD5 gốc:** `+1TNHBRxqTXkbr+sh20/5w==`
  - **Rotation:** `0.0`
- **URL phân phối AWS S3 chính thức:**  
  `https://open-images-dataset.s3.amazonaws.com/validation/e29db1c1963628e7.jpg`
- **Kiểm chứng URL:** HTTP HEAD trả về **`200 OK`**, Content-Length: **`71,473`** bytes, Content-Type: `image/jpeg`.
- **Trạng thái ảnh trên đĩa:** `CHƯA TẢI` (0 byte trên đĩa; đang chờ QA/Người dùng kiểm duyệt).

---

## 3. THỰC HIỆN YÊU CẦU 7: QUY TẮC BẮT BUỘC ĐỐI VỚI ẢNH MULTI-WASTE

Đối với các ảnh chứa nhiều loại rác (Multi-Waste Clutter):
1. **Tuyệt đối cấm gộp toàn khung hình thành một box to mang nhãn `trash`:** Hành vi này gây nhiễu nghiêm trọng bộ dò đối tượng, làm sai lệch phân phối kích thước và khiến mô hình mất khả năng nhận diện các vật thể nhỏ cụ thể.
2. **Quy tắc dán nhãn bắt buộc:** Mọi ảnh multi-waste đưa vào ứng viên bổ sung phải có các bounding box độc lập, tách rời cho từng vật thể thành phần (`metal`, `plastic`, `paper`, `glass`, v.v.), hoặc phải qua công cụ gán nhãn thủ công (như CVAT / LabelImg) để tạo nhãn chuẩn YOLO/SSDLite trước khi được phê duyệt vào tập train.

---

## 4. THỰC HIỆN YÊU CẦU 8: THẮT CHẶT TOÀN DIỆN LOGIC PHÊ DUYỆT TRONG EXPORT_TRAIN_MANIFEST.PY

Đã rà soát và cập nhật trực tiếp tại [`src/detection/export_train_manifest.py`](../src/detection/export_train_manifest.py):

### 4.1 Phạm vi (Scope) và Image SHA-256 là BẮT BUỘC đối với ảnh mới
- Đối với mọi ảnh mới bổ sung (`category in ("NEW_CANDIDATE", "ACQUISITION_TARGET", "PILOT_IMAGE_NEW")`):
  - **Trường `scope` là bắt buộc:** Record phê duyệt phải có trường `scope` mang giá trị hợp lệ (`"TRAIN_EXPANSION"`, `"TRAIN_EXPORT"`, hoặc `"ALL"`). Nếu thiếu hoặc scope là `INSPECTION_ONLY`, hàm sẽ lập tức từ chối xuất với mã lỗi `INVALID_APPROVAL_RECORD`.
  - **Trường `image_sha256` là bắt buộc:** Phải có mã băm 64 ký tự hex chuẩn của tệp ảnh. Khi thư mục `images_dir` được truyền vào, SHA-256 thực tế của tệp ảnh trên đĩa phải khớp tuyệt đối với giá trị đã duyệt.

### 4.2 Loại bỏ triệt để danh tính tự xưng (Self-declared Approver)
- Danh sách định danh cấm tự xưng (`FORBIDDEN_APPROVERS`):
  `{"unspecified", "anonymous", "none", "unknown", "system", "agent", "ai", "bot", "auto", "autogenerated", "assistant", "self", "script"}`.
- Nếu trường `approved_by` thuộc các giá trị máy móc trên, hàm sẽ từ chối với thông báo:
  `"...approval record lacks valid authoritative approver identity (got '...')"` và trả lỗi `INVALID_APPROVAL_RECORD`.

### 4.3 Cam kết không tự tạo Approval Record
- Sub-agent và hệ thống tuyệt đối không tự động tạo các bản ghi `APPROVED` giả định thay cho người dùng. Mọi bản ghi trong approval ledger bắt buộc phải xuất phát từ chỉ thị phê duyệt thật của người dùng.

---

## 5. KẾT QUẢ KIỂM THỬ THỰC TẾ

Không chạy vòng lặp audit lại toàn dự án; chỉ chạy các bộ kiểm thử trực tiếp liên quan đến logic an toàn và bản vá mới:

1. **Bộ kiểm thử Smoke & Approval Tightening (`test_train_command_smoke.py`):**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/detection/test_train_command_smoke.py -v
   ```
   **Kết quả: 9 passed in 4.75s**
   - `test_yolo_train_command_dry_run_args` - PASSED
   - `test_ssdlite_train_command_dry_run_args` - PASSED
   - `test_hardware_preflight_guard_blocks_low_vram` - PASSED
   - `test_verify_training_dataset_safety_blocks_test_leakage` - PASSED
   - `test_verify_training_dataset_safety_blocks_unapproved_candidate` - PASSED
   - `test_resume_mode_checks_dataset_safety_smoke` - PASSED
   - `test_user_controlled_approval_record_scope_and_image_hash` - PASSED *(Đã bổ sung kiểm thử từ chối approver tự xưng, thiếu scope, thiếu image_sha256)*
   - `test_candidate_manifest_unverified_demotion` - PASSED *(Kiểm thử xác nhận 35 mục cũ đã bị hạ cấp về SOURCE_UNVERIFIED)*
   - `test_verified_source_proof_5_samples` - PASSED *(Kiểm thử xác minh 5 bản ghi thật với đầy đủ cấu trúc và không sinh ID giả)*

2. **Bộ kiểm thử Split Contamination & Gate Isolation (`test_split_contamination_safety.py`):**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/qa/test_split_contamination_safety.py
   ```
   **Kết quả: 11 passed in 0.75s**

3. **Bộ kiểm thử Pilot Dataset v3 (`test_pilot_dataset_v3.py`):**
   ```powershell
   .venv\Scripts\python.exe -m pytest tests/qa/test_pilot_dataset_v3.py
   ```
   **Kết quả: 7 passed in 2.22s**

---

## 6. ĐỀ XUẤT XIN QUYỀN TẢI PREVIEW NHỎ (NẾU ĐƯỢC DUYỆT)

Hiện tại, thư mục `data/acquisition_staging` vẫn duy trì **đúng 0 byte ảnh tải về**.

Nếu bạn muốn kiểm tra trực quan bằng mắt các ảnh đã được chứng minh nguồn:
- **Nguồn:** AWS S3 Open Images Dataset chính thức (`https://open-images-dataset.s3.amazonaws.com/validation/`).
- **Số lượng:** Đúng **3 ảnh** (2 ảnh Biological `078821d86db99fc9`, `74a44994e3d6b962` và 1 ảnh Hard-Negative `e29db1c1963628e7`).
- **Tổng dung lượng dự kiến:** Xấp xỉ **400 KB** (189 KB + 139 KB + 71 KB).
- **Hành động hiện tại:** **DỪNG VÀ CHỜ PHÊ DUYỆT.** Chưa có bất kỳ byte nào được tải về.
