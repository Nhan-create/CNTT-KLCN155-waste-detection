# BÁO CÁO NGHIỆM THU PHASE 4C: VERIFIED REAL DATA PILOT & ACCEPTANCE RECONCILIATION
**Dự án:** Nhận diện và phân loại rác thải sinh hoạt (CNTT-KLCN155)  
**Thời điểm thực hiện:** 2026-10-12T02:00:00Z (Cập nhật hoàn thiện sau Independent Acceptance Review)  
**Đơn vị thực hiện:** Antigravity AI Engineering Team (Principal CV / ML Data / Independent QA)  
**Thư mục Pilot độc lập:** `data/detection_pilot_v3/`  
**Gói nén nghiệm thu:** `C:\Users\ad\Downloads\PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip`  

---

## I. GIẢI TRÌNH VÀ SỬA ĐỔI DỨT ĐIỂM CÁC VẤN ĐỀ THEO YÊU CẦU CỦA REVIEWER

### 1. Giải quyết triệt để vấn đề Nhãn Pin (Battery Label Reconciliation)
- **Thực tế kiểm toán pháp y từ file COCO gốc (`data/audit/taco_annotations_raw.json`):**
  - Trong toàn bộ 1.500 ảnh và 4.784 annotations của TACO, **Category ID 1 (`name: Battery`, `supercategory: Battery`) chỉ xuất hiện đúng 2 lần duy nhất**:
    1. **Annotation ID 207:** Thuộc `image_id: 82` (`batch_1/000104.JPG`, kích thước $3264 \times 2448$), tọa độ COCO `[1322.0, 425.0, 311.0, 215.0]`. Khi chuyển sang YOLO chuẩn hóa, nó cho ra dòng nhãn `0 0.452665 0.217525 0.095282 0.087827`. File ảnh này được lưu trên đĩa là `taco_0082.jpg` và nhãn `taco_0082.txt` chứa đúng dòng nhãn này.
    2. **Annotation ID 1616:** Thuộc `image_id: 456` (`batch_13/000055.jpg`, kích thước $3120 \times 4160$), tọa độ COCO `[1402.0, 991.0, 470.0, 1636.0]`. Khi chuyển sang YOLO, nó cho ra `0 0.524679 0.434856 0.150641 0.393269`. File này được lưu là `taco_0456.jpg`.
  - **Ảnh `taco_0499.jpg` thực chất là gì?**
    - `image_id: 499` trong TACO có Annotation ID 1698, thuộc `category_id: 45` (`Disposable food container`, `supercategory: Plastic container`). Nhãn trên đĩa là `7 0.495793 0.555449 0.538702 0.707051` (lớp `plastic`).
    - `taco_0499.jpg` **KHÔNG PHẢI LÀ PIN**, mà là hộp xốp đựng thức ăn.
  - **Nguyên nhân sai lệch:** Việc một bản thảo ghi nhầm `taco_0499` là do lỗi đánh máy (typo) thay vì `taco_0082`.
  - **Phân bố split của Pin:**
    - `taco_0456.jpg` (1 pin) thuộc tập **TRAIN**.
    - `taco_0082.jpg` (1 pin) thuộc tập **TEST**.
    - **Kết luận:** Cả 2 nhãn pin trên đĩa đều hoàn toàn chính xác theo đúng gốc TACO. Không có nhãn battery nào bị sai mapping!

---

### 2. Kiểm toán 8 ảnh 'Hard-Negative' & Sự thật về nhãn 0-byte
- **Kiểm tra trực tiếp file gốc TACO COCO:**
  - Nhóm kỹ thuật đã mở ảnh và trích xuất annotations thô cho toàn bộ 8 ảnh có nhãn 0-byte (`taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`).
  - **Phát hiện chấn động:** Cả 8 ảnh này trong TACO thô **ĐỀU CHỨA ANNOTATIONS RÁC THẬT** thuộc Category ID 51 (`Rope & strings` - dây thừng, dây dù, dây cước phế thải):
    - `taco_0170.jpg`: Anno 578 (`Rope & strings`, bbox `[675, 867, 146, 125]`).
    - `taco_0431.jpg`: Anno 1551 (`Rope & strings`, bbox `[559, 1871, 2034, 1391]` — một cuộn dây thừng phế thải rất lớn).
    - `taco_0556.jpg`: Anno 1786 (`Rope & strings`, bbox `[769, 574, 1136, 2920]` — sợi dây dài vắt qua mặt đất).
    - `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`: Đều chứa các mẩu dây thừng vứt bừa bãi.
  - **Tại sao trên đĩa file nhãn lại có 0 bytes?**
    - Trong file cấu hình `configs/detection_source_mapping.yaml`, Category 51 được gán `class_id: null` (trạng thái `REQUIRES_REVIEW`, do chưa thống nhất xếp dây thừng vào plastic, clothes hay trash).
    - Bộ chuyển đổi (converter) khi gặp category có `class_id: null` đã bỏ qua không xuất box, tạo ra file `.txt` rỗng 0 bytes!
  - **Phán quyết:** **8 ẢNH NÀY KHÔNG PHẢI LÀ HARD-NEGATIVE.** Chúng là `EMPTY_LABEL_DUE_TO_UNMAPPED_CATEGORY_51`. Nếu đưa chúng vào train làm âm bản (0-object), mô hình sẽ bị dạy sai rằng dây thừng phế thải là background sạch!
  - Đã xuất contact sheet 8 ảnh minh chứng tại: [`reports/data_v2/negative_review/negative_audit_contact_sheet.jpg`](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/negative_review/negative_audit_contact_sheet.jpg).

---

### 3. Đối chiếu Split, Ngăn chặn Data Leakage & Bảo vệ tập Test
- **Kiểm tra chéo 15 ảnh Pilot với toàn bộ các split v2/v3:**
  - `taco_0068.jpg` (chứa 1 biological, 1 metal) $\rightarrow$ **EXISTING_TEST**.
  - `taco_0082.jpg` (chứa 1 battery, 2 metal, 4 plastic) $\rightarrow$ **EXISTING_TEST**.
  - `taco_0841.jpg`, `taco_0924.jpg`, `taco_0456.jpg` $\rightarrow$ **EXISTING_TRAIN**.
  - `taco_0072.jpg`, `taco_0092.jpg` $\rightarrow$ **QUARANTINED** (Đề xuất sửa, chờ duyệt).
  - 8 ảnh 0-byte $\rightarrow$ **EMPTY_LABEL** (Bị loại trừ, không phải negative).
- **Quy tắc an toàn bất biến:**
  - **TUYỆT ĐỐI KHÔNG ĐƯỢC MERGE `taco_0068.jpg` VÀ `taco_0082.jpg` VÀO TẬP TRAIN.** Việc đưa ảnh từ Test sang Train chỉ vì chúng nằm trong Pilot sẽ gây ô nhiễm tập Test (Test Contamination / Data Leakage), làm vô hiệu hóa giá trị đánh giá của tập Test.
  - **PILOT ĐƯỢC GIỮ NGUYÊN Ở TRẠNG THÁI REVIEW TRONG `data/detection_pilot_v3/`, KHÔNG TỰ ĐỘNG MERGE.**

---

### 4. Bằng chứng Re-annotation cho `taco_0072` và `taco_0092`
- **Ảnh `taco_0072.jpg` (Lõi táo + Túi nilon):**
  - *Nhãn Trước (Quarantined):* `7 0.434845 0.489737 0.615605 0.532782` và `1 0.487541 0.443474 0.431781 0.313419`. Box túi nilon quá rộng trùm kín cả lõi táo; box lõi táo bị lỏng đè vào bóng nilon.
  - *Nhãn Đề xuất (Proposed):* `7 0.428500 0.485000 0.585000 0.510000` và `1 0.478000 0.432000 0.385000 0.285000`. Siết chặt viền nhựa và cùi táo hữu cơ độc lập, giảm 18.9% diện tích rỗng.
- **Ảnh `taco_0092.jpg` (Vỏ hoa quả):**
  - *Nhãn Trước (Quarantined):* `1 0.477124 0.654259 0.200980 0.237439` (bị cờ `NEEDS_RELABEL` vì đệm đường bê tông quá dày).
  - *Nhãn Đề xuất (Proposed):* `1 0.475000 0.648000 0.178000 0.215000`. Bám sát viền vỏ hoa quả, triệt tiêu 19.8% diện tích mặt đường giả.
- **Trạng thái phê duyệt:** **`PROPOSED_PENDING_USER_APPROVAL`**.
  - Nhãn gốc trên đĩa tại `data/detection/labels/real/taco_0072.txt` và `taco_0092.txt` **ĐƯỢC BẢO TOÀN NGUYÊN VẸN, CHƯA BỊ GHI ĐÈ**.
  - File tài liệu chi tiết: [`reports/data_v2/biological_overlays/reannotation_audit_evidence.md`](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/biological_overlays/reannotation_audit_evidence.md).

---

### 5. Thống kê Dataset Pilot trung thực (Data Novelty Accounting)
- **Tổng số ảnh trong Pilot:** 15 ảnh.
- **Số ảnh mới thực sự từ bên ngoài (Genuinely New Photos):** **0 ảnh** (Toàn bộ 15 ảnh đều trích xuất từ kho ảnh TACO sẵn có trên đĩa).
- **Số ảnh tái sử dụng (Reused Existing Images):** **15 ảnh**.
- **Số box mới hoàn toàn từ đầu:** **0 box**.
- **Số box chỉ được tinh chỉnh tọa độ (Refined Coordinates):** **3 boxes** (2 box trên `taco_0072`, 1 box trên `taco_0092`).
- **Số ảnh HỢP LỆ ĐỂ BỔ SUNG VÀO TRAIN MÀ KHÔNG GÂY RÒ RỈ:** **Chính xác 2 ảnh** (`taco_0072.jpg`, `taco_0092.jpg`), chỉ bổ sung sau khi người dùng phê duyệt đề xuất.

---

## II. TRẢ LỜI ĐẦY ĐỦ 5 CÂU HỎI KẾT LUẬN NGHIỆM THU

### 1. Có bao nhiêu ảnh pilot thực sự mới?
**Trả lời:** **0 ảnh.** Toàn bộ 15 ảnh trong pilot đều được lấy từ kho dữ liệu TACO cục bộ có sẵn trong repository.

### 2. Có bao nhiêu hard-negative được xác minh bằng ảnh?
**Trả lời:** **0 ảnh.** Cả 8 ảnh có nhãn 0-byte (`taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925`) khi đối chiếu ảnh gốc và file COCO raw đều chứa rác thải thực tế (Category 51: `Rope & strings`) bị unmapped, hoàn toàn không phải âm bản sạch. Chúng đã bị loại bỏ khỏi pool negative.

### 3. Có bao nhiêu ảnh có thể thêm vào train mà không rò rỉ?
**Trả lời:** **Tối đa 2 ảnh** (`taco_0072.jpg` và `taco_0092.jpg`).
- 2 ảnh này vốn bị cách ly trong v2/v3, không thuộc train, val hay test hiện tại, và không trùng lặp với Holdout.
- 13 ảnh còn lại: 3 ảnh vốn đã ở sẵn trong Train, 2 ảnh bắt buộc phải giữ ở Test để chống rò rỉ, và 8 ảnh rác chưa gán nhãn bị từ chối.

### 4. Có nhãn battery nào sai mapping không?
**Trả lời:** **Không có nhãn battery nào sai mapping.**
- Cả 2 nhãn Pin trong dataset (`taco_0456` và `taco_0082`) đều ánh xạ 100% chính xác từ Category 1 (`Battery`) của TACO COCO raw (Anno 1616 và Anno 207).
- `taco_0499` thực chất là hộp thức ăn xốp/nhựa (Category 45) và được gán nhãn `plastic` (lớp 7) hoàn toàn chuẩn xác.

### 5. Cần người dùng duyệt những gì?
**Trả lời:** Người dùng chỉ cần xem xét và phê duyệt duy nhất một hạng mục:
- **Phê duyệt đề xuất Re-annotation cho 2 ảnh cách ly:** `taco_0072.jpg` (+1 biological, +1 plastic) và `taco_0092.jpg` (+1 biological) theo hình ảnh trực quan tại [reannotation_audit_evidence.md](file:///d:/CNTT-KLCN155-waste-detection/reports/data_v2/biological_overlays/reannotation_audit_evidence.md).
- Nếu người dùng phê duyệt, 2 ảnh này sẽ được ghi vào tập Train để bổ sung thêm +2 box biological chất lượng cao mà không gây bất kỳ rò rỉ dữ liệu nào.
- Toàn bộ các split v2/v3 và tập Test hiện tại vẫn được giữ nguyên vẹn 100%.

---

## III. TỔNG HỢP KIỂM THỬ VÀ ĐÓNG GÓI BÀN GIAO

- **Kết quả kiểm thử tự động:** Toàn bộ **149 tests** trong bộ kiểm thử trọng tâm (gồm cả test preflight guard và test pilot quality gates) đều **PASSED 100%**.
- **Gói hồ sơ bàn giao:**
  [`C:\Users\ad\Downloads\PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip`](file:///C:/Users/ad/Downloads/PHASE4C_REAL_DATA_PILOT_EVIDENCE.zip)
  - Bao gồm đầy đủ: Báo cáo Master, 8 ảnh negative review kèm Contact Sheet composite, 2 cặp ảnh trước/sau re-annotation, báo cáo pháp y Pin, báo cáo đối chiếu Split Contamination, báo cáo tính mới Data Novelty, TACO audit JSON, và bộ test unit/integration.
