# TASK 8I — BỘ HƯỚNG DẪN THU THẬP ẢNH THỰC ĐỊA (FIRST-PARTY FIELD CAPTURE KIT)

> **Trạng thái trung thực tuyệt đối:** Chưa có ảnh thực địa mới nào được chụp trong phiên này (`real_field_photos_captured_to_date = 0`, `ai_or_synthetic_images_generated = 0`). Bộ hướng dẫn, biểu mẫu manifest và công cụ CLI dưới đây sẵn sàng để tiếp nhận lô Pilot `50–100` ảnh chụp điện thoại thật hợp pháp đầu tiên.

## 1. Thứ tự ưu tiên 5 lớp khan hiếm
1. **Quần áo bỏ đi (`3: clothes`):** hiện có `55` ảnh train (`gap_to_500 = 445`).
2. **Giày dép bỏ đi (`8: shoes`):** hiện có `78` ảnh train (`gap_to_500 = 422`).
3. **Pin đã qua sử dụng (`0: battery`):** hiện có `86` ảnh train (`gap_to_500 = 414`).
4. **Thủy tinh thải bỏ (`4: glass`):** hiện có `113` ảnh train (`gap_to_500 = 387`).
5. **Rác hữu cơ thực tế (`1: biological`):** hiện có `119` ảnh train sau xác minh iNaturalist Task 8I (`gap_to_500 = 381`).

## 2. Danh sách bối cảnh cần chụp (Scene Checklist)
- **Bối cảnh B1 (Hộ gia đình / Nhà bếp):** Thùng rác mở nắp, góc tập kết rác bếp trên nền gạch/xi măng.
- **Bối cảnh B2 (Hẻm dân cư / Điểm tập kết rác phố):** Cạnh thùng rác công cộng, vỉa hè, mặt nhựa đường.
- **Bối cảnh B3 (Ký túc xá / Khuôn viên trường):** Điểm phân loại rác tái chế, hộp thu gom pin cũ.
- **Bối cảnh B4 (Ngoài trời / Công viên / Bãi cỏ):** Rác vứt trên đất, cỏ, rãnh thoát nước, gốc cây.
- **Bối cảnh B5 (Nhà kho / Xưởng sửa chữa):** Khay/hộp chứa pin cũ, chai lọ thủy tinh vỡ/bỏ đi, giẻ lau dầu mỡ.

## 3. Checklist từng loại vật thể & Quy chuẩn Ảnh đơn / Đa vật thể / Đa lớp
- **Tỷ lệ bắt buộc trong mỗi buổi chụp (Session):**
  - **30% Ảnh đơn (`Single-object`):** 1 mẫu vật thuộc lớp ưu tiên trên nền bối cảnh rác thật.
  - **25% Ảnh đa vật thể cùng lớp (`Multi-object same-class`):** 2–5 mẫu vật độc lập cùng lớp đặt gần nhau.
  - **45% Ảnh đa lớp (`Multi-class co-occurrence`):** Mẫu vật lớp khan hiếm chụp chung với 1–3 lớp rác khác (`plastic`, `metal`, `paper`, `cardboard`, `trash`).

## 4. Quy tắc định danh `specimen_id` & `session_id` (Độc lập dữ liệu Task 3)
- **`specimen_id`:** `SPEC_<3LETTER_CLASS>_<YYYYMM>_<4DIGIT_SEQ>` (ví dụ `SPEC_CLO_202610_0001`).
- **`session_id`:** `SESS_<YYYYMMDD>_<LOCATION_CODE>` (ví dụ `SESS_20261016_HCM_Q7_ALLEY01`).
- **Quy tắc đa góc chụp:** Tối đa `3` góc chụp (`top-down 90°`, `oblique 45°`, `low-angle 20°`) cho cùng 1 mẫu vật; cả 3 ảnh phải mang **cùng 1 `specimen_id`**, chỉ được tính là **`1` mẫu vật độc lập (`unique_specimen = 1`)** và bắt buộc nằm trong cùng 1 split (`train` hoặc `val`).

## 5. Quy chuẩn khoảng cách, ánh sáng, góc chụp & Gán nhãn đầy đủ 10 lớp
- **Khoảng cách:** `0.5m` (cận cảnh có nền), `1.2m` (tiêu chuẩn cầm tay), `2.5m` (toàn cảnh miệng thùng/góc phố).
- **Gán nhãn đầy đủ 10 lớp:** Mọi vật thể rác nhìn thấy trong khung hình thuộc bất kỳ lớp nào trong 10 lớp (`0: battery .. 9: trash`) đều phải được vẽ bbox riêng, không bỏ sót vật thể phụ làm false-negative.
- **Biểu mẫu import:** `data/acquisition_staging/task8i_source_signoff_and_field_pilot/field_pilot_templates/field_capture_import_template.csv`
- **Công cụ CLI Import & QA tự động:** `python scripts/field_capture_import_and_qa.py --manifest <csv/json> --images-dir <dir> --labels-dir <dir> --output-dir <dir>`
