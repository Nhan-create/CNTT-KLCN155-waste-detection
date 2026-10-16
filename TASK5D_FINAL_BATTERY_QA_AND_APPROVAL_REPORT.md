# BÁO CÁO KỸ THUẬT & BÀN GIAO: SPRINT AI-UPGRADE-01 — TASK 5D FIX (BATTERY OBJECT ANNOTATION & QA STATE MACHINE)

*(Bản cập nhật hợp nhất theo kết luận nghiệm thu Task 5D Fix — Chi tiết xem tại [`TASK5D_FIX_BATTERY_ANNOTATION_REPORT.md`](file:///d:/CNTT-KLCN155-waste-detection/TASK5D_FIX_BATTERY_ANNOTATION_REPORT.md))*

- **Trạng thái QA 140 nhóm ảnh `base_id`:**
  - `TECHNICAL_VALID`: **140/140** ảnh (`768x768`, giải mã hợp lệ, tọa độ toán học hợp lệ).
  - `ORIGINAL_UNKNOWN`: **140/140** nhóm (350/350 file có `.rf.<hash>`, 210 biến thể sinh thêm bị loại trừ `AUGMENTED_VARIANT_EXCLUDED`).
  - `VISUAL_QA_PASS`: **20/140** ảnh đại diện trong Mini-Pilot đã được **re-annotate** sang **1 full-battery bounding box/viên pin** (`20` boxes mới thay thế cho `41` box chữ cũ), kèm đầy đủ `review_evidence` và **20 ảnh đối chiếu BEFORE / AFTER**.
  - `NEEDS_REANNOTATION`: **120/140** ảnh đại diện còn lại đang mang box chữ gốc của Roboflow, bị khóa chặt bởi Gatekeeper (không đưa vào huấn luyện).
  - `APPROVED_FOR_TRAIN`: **0** (`approved_for_train: False`).
- **Đính chính bảng số liệu tập `test` đóng băng (299 ảnh):** Lớp `9: trash = 139` boxes, tổng cộng $1 + 1 + 36 + 91 + 32 + 73 + 34 + 274 + 39 + 139 = \mathbf{720}$ boxes.
