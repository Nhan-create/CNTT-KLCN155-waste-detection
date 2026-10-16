# BÁO CÁO NGHIỆM THU VÀ HIỆU CHỈNH TASK 4: REAL IMAGE ACQUISITION & QA
**Sprint:** SPRINT AI-UPGRADE-01 — Task 4 (Đã hiệu chỉnh theo ý kiến thẩm định của PM)  
**Repository:** `.`  
**Ngày thực hiện:** 13/10/2026  
**Kỹ sư thực hiện:** Antigravity AI Engineer  
**Thẩm định nghiệm thu:** ChatGPT PM / Reviewer  

---

## I. TỔNG HỢP KẾT QUẢ VÀ 4 BIẾN ĐẾM BẮT BUỘC (DoD METRIC COUNTERS)

```
NEW DOWNLOADED       = 10
VISUAL QA REVIEWED   = 10
ANNOTATIONS REVIEWED = 10 (Tổng 7 active boxes, 4 empty label files)
APPROVED FOR TRAIN   = 0 (Tuân thủ tuyệt đối DoD)
```

| Chỉ số bắt buộc | Giá trị nghiệm thu | Ghi chú & Trạng thái thẩm định |
|---|---|---|
| **NEW DOWNLOADED** | **10** | Tải stream 10/10 ảnh VERIFIED, tổng **38.506.683 bytes (~36.72 MiB)** $\le$ 50 MiB hard limit |
| **VISUAL QA REVIEWED** | **10** | 10/10 ảnh đã rà soát: 6 QA_PASS, 1 QUARANTINED, 1 PENDING_ANNOTATION, 2 CLEAN_NEGATIVE |
| **ANNOTATIONS REVIEWED** | **10** | 10/10 file nhãn YOLO chuẩn hóa tại `labels/`: 7 bounding boxes chuẩn pixel, 4 file 0-byte |
| **APPROVED FOR TRAIN** | **0** | **Tuân thủ tuyệt đối DoD:** Toàn bộ mẫu giữ ở `PENDING_PM_APPROVAL`, chưa đưa vào split |

---

## II. KẾT QUẢ TASK 1: HIỆU CHỈNH TOÀN DIỆN ANNOTATION TỪNG ẢNH

### 1. `wiki_bat_corroded_screwdriver.jpg` (Pin máy vặn vít)
- **Vấn đề phát hiện:** Box cũ gán bao trùm công tắc, dây điện và motor; cell pin bị sùi hóa chất che khuất nặng, không thể phân định biên giới chắc chắn.
- **Phán quyết QA:** Chuyển sang **`QUARANTINED_AMBIGUOUS_BATTERY_ASSEMBLY`** theo đúng chỉ thị PM.
- **Xử lý nhãn:** Làm rỗng file nhãn `wiki_bat_corroded_screwdriver.txt` (0 boxes), không để nhãn không chắc chắn gây nhiễu detector.

### 2. `wiki_bio_banana_peel_bucharest_02.jpg` (Vỏ chuối đường phố Bucharest)
- **Vấn đề phát hiện:** Box cũ `[318, 493, 2093, 2315]` bị cắt mất chóp cuống phía trên (`y < 493`) và dải vỏ nằm ngang trải dài sang phải (`x > 2093`).
- **Đo đạc pixel gốc:**
  - Đỉnh chóp trên cùng: `y = 246, x = 1891`
  - Đỉnh mút bên phải: `x = 3148, y = 2025`
  - Đáy dải vỏ bên dưới: `y = 2851, x = 1696`
  - Rìa trái: `x = 318, y = 1607`
- **Tọa độ YOLO mới:** `1 0.548858 0.541842 0.901015 0.915616` (Pixel: `[310, 240, 3150, 2855]`).
- **Phán quyết QA:** **`QA_PASS`** (Bao trọn 100% vỏ chuối, ôm khít ranh giới thực tế).

### 3. `wiki_bio_food_scraps_bin_09.jpg` (Thùng rác hữu cơ SSO)
- **Vấn đề phát hiện:** Box cũ bao trùm cả xô cầm tay, hộp pizza carton bên cạnh và thùng rác xanh đô thị, vi phạm nguyên tắc phân đoạn vật thể đơn lẻ.
- **Phán quyết QA:** Chuyển sang **`PENDING_ANNOTATION`** theo đúng chỉ đạo PM.
- **Xử lý nhãn:** File nhãn `wiki_bio_food_scraps_bin_09.txt` được chuyển thành 0-byte, không cấp phép train khi chưa có box riêng biệt cho từng loại rác.

### 4. `wiki_multi_banana_and_cup_01.jpg` (Đa rác: Vỏ chuối + Cốc cà phê)
- **Kiểm tra vật liệu cốc:** Cốc giấy dùng một lần thương hiệu siêu thị (rolled-rim paper cup) có bọc lớp carton sóng cách nhiệt (corrugated ripple sleeve). Theo quy chuẩn [src/detection/schema.py](src/detection/schema.py#L55), vật thể này thuộc lớp `6: paper`.
- **Hiệu chỉnh box vỏ chuối:** Box cũ cắt mất phần cuống xanh/nâu phía trên (`y = 340`). Đã mở rộng box lên `[200, 335, 2155, 2460]` bao trọn cuống và mép vỏ.
- **Tọa độ YOLO mới:**
  - Vỏ chuối (biological): `1 0.239183 0.378521 0.397116 0.575569`
  - Cốc giấy (paper): `6 0.744566 0.661972 0.332521 0.386782`
- **Phán quyết QA:** **`QA_PASS`** (Hai box riêng biệt, tách bạch hoàn toàn).

---

## III. KẾT QUẢ TASK 2: RÀ SOÁT NGỮ CẢNH VÀ PHÂN ĐỊNH DOMAIN

### 1. Phân biệt `OBJECT_ONLY` vs `REAL_DISCARDED_WASTE`
Soi pixel gốc hai ảnh vỏ chuối:
- `wiki_bio_banana_peel_sidewalk_06.jpg`: Chụp trong nhà trên nền giấy trắng phẳng (`STUDIO_INDOOR`, `OBJECT_ONLY_STUDIO_WHITE_BG`).
- `wiki_bio_banana_peel_oak_hill_05.jpg`: Chụp trong nhà trên bề mặt khăn giấy ăn dập vân (`STUDIO_INDOOR`, `OBJECT_ONLY_STUDIO_WHITE_BG`).
- **Xử lý:** Gỡ bỏ nhãn gán sai `MOBILE ngoài trời` trong manifest. Đánh dấu rõ ràng ngữ cảnh `OBJECT_ONLY_STUDIO_WHITE_BG` (thích hợp cho copy-paste augmentation hoặc pretraining, không phải rác đường phố).

### 2. Rà soát Full-frame hai ảnh Hard-Negative sạch
- `wiki_neg_utility_box_corner_05.jpg`: Soi toàn khung hình 5152x3864. Tủ điện góc phố, mặt đường nhựa và chân vỉa hè sạch 100% rác thải $\rightarrow$ **`QA_PASS_CLEAN_NEGATIVE`**.
- `wiki_neg_brick_wall_clean.jpg`: Soi toàn khung hình 3264x2448. Bề mặt tường đá/gạch sạch 100% $\rightarrow$ **`QA_PASS_CLEAN_NEGATIVE`**.

---

## IV. KẾT QUẢ TASK 3: ĐỒNG BỘ MANIFEST & MINH BẠCH LEAKAGE

- **Đồng bộ số lượng box:** Khớp chính xác 7 boxes trên toàn bộ 10 file nhãn.
- **Phân tách rành mạch 3 cột trạng thái trong manifest:**
  1. `technical_valid`: `true` (file thật, decode RGB).
  2. `visual_qa`: Tình trạng thẩm định ảnh (`QA_PASS`, `QA_PASS_OBJECT_ONLY_WHITE_BG`, `QA_PASS_CLEAN_NEGATIVE`, `QUARANTINED_AMBIGUOUS_BATTERY_ASSEMBLY`, `PENDING_ANNOTATION`).
  3. `user_train_approval`: `PENDING_PM_APPROVAL` (duy nhất người dùng và PM có quyền đổi thành APPROVED).
- **Giải trình minh bạch về kết quả Leakage Check:**
  - File cache [authoritative_hashes_cache.json](data/audit/authoritative_hashes_cache.json) chứa **8.913 checksums** của toàn bộ tập dữ liệu (train, val, test và holdout).
  - Giá trị `sha256_match: null` có nghĩa là **KHÔNG TÌM THẤY TRÙNG LẶP NÀO** trong 8.913 ảnh đã biết.
  - pHash Hamming Distance đối chiếu toàn bộ dataset đều $> 8$.

---

## V. KẾT QUẢ TASK 4: KẾ HOẠCH BỔ SUNG BATTERY THẬT

Do ảnh pin tự do trên Wikimedia rất hạn chế và ứng viên duy nhất đã bị `QUARANTINE`:
1. **Nguồn ưu tiên hàng đầu:** Bộ dữ liệu `waste-battery-only v1` trên Roboflow Universe (254 ảnh pin thực tế gán nhãn YOLOv8, CC BY 4.0).
2. **Quy trình tiếp nhận an toàn:**
   - Người dùng tải trực tiếp file `.zip` từ trình duyệt:  
     `https://universe.roboflow.com/alif-kurniawan-x336t/waste-battery-only/dataset/1`
   - Đặt file `.zip` vào thư mục `data/acquisition_staging/roboflow_battery/`.
   - Chạy pipeline offline [scripts/import_and_qa_roboflow_battery.py](scripts/import_and_qa_roboflow_battery.py).
   - Tuyệt đối không cần dán API key, không tạo ảnh pin AI, không xâm phạm tập test/holdout.

---

## VI. KẾT QUẢ TASK 5: CHUẨN BỊ CHO BƯỚC FINE-TUNING TIẾP THEO

1. **Danh sách ảnh đủ điều kiện đề xuất đưa vào pilot training (sau khi được duyệt):**
   - `wiki_bio_banana_peel_bucharest_02` (biological - rác đường phố)
   - `wiki_bio_banana_peel_bus_stop_03` (biological - rác ghế chờ xe buýt)
   - `wiki_bio_apple_core_01` (biological - cuống táo)
   - `wiki_multi_banana_and_cup_01` (biological & paper - rác vỉa hè)
   - `wiki_neg_utility_box_corner_05` (hard-negative sạch triệt tiêu FP plastic)
   - `wiki_neg_brick_wall_clean` (hard-negative sạch triệt tiêu FP cardboard)
2. **Tách biệt Dataset:** Giữ nguyên `configs/detection_dataset_real_v2.yaml`, thiết lập file cấu hình mới `configs/detection_dataset_real_v3_pilot.yaml`.
3. **Đánh giá phần cứng & Kế hoạch Pilot:**
   - GPU: NVIDIA GeForce RTX 2050 Mobile (4GB VRAM).
   - Cấu hình đề xuất: `imgsz=512, batch=4, workers=2, amp=true, cache=false, epochs=25`.
   - Dự toán thời gian chạy: $\approx 25 - 35$ phút.

---

## VII. TỔNG HỢP DANH MỤC GIAO NỘP & BẢO CHỨNG KIỂM THỬ

1. **Manifest đã hiệu chỉnh:** [manifest.json](data/acquisition_staging/batch_a_mini01/manifest.json)
2. **Thư mục 10 file nhãn chuẩn:** [data/acquisition_staging/batch_a_mini01/labels/](data/acquisition_staging/batch_a_mini01/labels/)
3. **Contact Sheet trực quan cập nhật:**
   - [minibatch01_annotated_contact_sheet.jpg](reports/source_verification/contact_sheets/minibatch01_annotated_contact_sheet.jpg)
   - [minibatch01_contact_sheet.jpg](reports/source_verification/contact_sheets/minibatch01_contact_sheet.jpg)
4. **Thư mục 10 Overlays chi tiết:** [reports/source_verification/overlays/](reports/source_verification/overlays/)
5. **Bộ kiểm thử hồi quy tự động:** **18 passed, 1 skipped** (100% tests PASS).
6. **Gói ZIP bàn giao cập nhật hoàn chỉnh:**  
   `~/Downloads\CNTT_KLCN155_Task4_Acquisition_Va_QA.zip`
