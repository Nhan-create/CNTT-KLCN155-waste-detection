# BÁO CÁO NGHIỆM THU TASK 8I (FINAL HOTFIX) — FIELD CAPTURE QA SAFETY & INATURALIST SIGN-OFF

## 1. Task 1 (Hotfix P0) — Tách biệt Kiểm tra Kỹ thuật & Xác nhận Bối cảnh Rác thật trong `scripts/field_capture_import_and_qa.py`
- Đã sửa triệt để lỗi P0 trong [`scripts/field_capture_import_and_qa.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/field_capture_import_and_qa.py):
  1. **Tách riêng `FILE_AND_BBOX_FORMAT_VALID` và `REAL_WASTE_CONTEXT_VERIFIED`:** Việc ảnh hợp lệ và tọa độ YOLO `.txt` đúng chuẩn chỉ thiết lập `FILE_AND_BBOX_FORMAT_VALID = True`, tuyệt đối không tự suy ra `REAL_WASTE_CONTEXT_VERIFIED = True`.
  2. **Mặc định `qa_status = NEEDS_REVIEW`:** Mọi ảnh mới nhập chưa có bản ghi `visual_qa_review` của người duyệt đều giữ ở `NEEDS_REVIEW` (`target_split = MANUAL_REVIEW_QUEUE`).
  3. **Chặn nâng `PRODUCT_STUDIO` và `IN_USE` lên `REAL_WASTE_PASS`:** Bất kể nhãn YOLO hợp lệ hay người nhập khai báo sai, `context_type in (PRODUCT_STUDIO, IN_USE)` luôn có `REAL_WASTE_CONTEXT_VERIFIED = False` và không bao giờ đạt `REAL_WASTE_PASS`.
  4. **Yêu cầu bằng chứng người duyệt thật (`reviewer_id`, `review_timestamp`, `context_evidence >= 10 ký tự`, `bbox_tightness_verified = True`):** Chỉ khi có đủ các trường này trên mẫu `ACTUAL_WASTE` đã xác nhận quyền sử dụng thì mới đạt `REAL_WASTE_CONTEXT_VERIFIED = True`.
  5. **Giữ `APPROVED_FOR_TRAIN = False` trên 100% bản ghi:** Kể cả khi `qa_status == REAL_WASTE_PASS`.
  6. **Không tự gán `CC-BY-4.0`:** Nếu thiếu giấy phép rõ ràng hoặc `rights_confirmed = False`, gán `UNCONFIRMED_USAGE_RIGHTS` và giữ ở `NEEDS_REVIEW`.
- **Ví dụ kiểm thử giả lập `PRODUCT_STUDIO` bị giữ ở `NEEDS_REVIEW`:** [`data/acquisition_staging/task8i_source_signoff_and_field_pilot/regression_examples/product_studio_held_at_needs_review_example.json`](file:///D:/CNTT-KLCN155-waste-detection/data/acquisition_staging/task8i_source_signoff_and_field_pilot/regression_examples/product_studio_held_at_needs_review_example.json)

## 2. Task 2 (Hotfix) — Đồng bộ Báo cáo iNaturalist & Xử lý 3 Ca Trái cây Mốc Cầm tay (`task8h_0085`, `0092`, `0137`)
- **Số liệu thực trong manifest trước Hotfix:** `33` `REAL_WASTE_PASS` (`27` train + `6` validation), `29` `KEEP_AUXILIARY`, `45` `REJECT`.
- **Kiểm tra lại `task8h_0085`, `task8h_0092`, `task8h_0137`:** Cả 3 ảnh đều chụp trái cây có mốc được cầm trên tay trong nhà, `description` trên iNaturalist trống (`''` / `None`), không có thùng rác hay bằng chứng đã thải bỏ $\rightarrow$ chuyển cả 3 ảnh từ `REAL_WASTE_PASS` (`train`) sang **`KEEP_AUXILIARY` (`HANDHELD_SPOILED_FRUIT_UNVERIFIED_DISPOSAL`)**.
- **Số liệu thực sau Hotfix:** **`30` `REAL_WASTE_PASS`** (**`24` train** + **`6` validation**), **`37` `KEEP_AUXILIARY`**, **`40` `REJECT`**.

## 3. Task 3 (Hotfix) — Chính sách Giấy phép Kép (`Photo License` vs `Observation License`) trên 11 Ca Hạn chế
- Đã tách riêng trường `inat_photo_license_usable` (quyền bản quyền bức ảnh) và `inat_observation_license_open` (quyền đối với bản ghi dữ liệu quan sát).
- Kiểm tra lại toàn bộ 11 ca có `observation_license_code` là `cc-by-nc` (`9` ca) hoặc `all-rights-reserved` (`2` ca) nhưng `photo_license_code` mở (`cc0`/`cc-by`/`cc-by-sa`):
  - **`5` ca (`task8h_0072, 0102, 0114, 0141, 0145`)** có vật thể rõ và giấy phép ảnh hợp lệ nhưng ngữ cảnh chưa đủ bằng chứng vứt bỏ $\rightarrow$ chuyển từ `REJECT` sang **`KEEP_AUXILIARY`**.
  - **`6` ca (`task8h_0087, 0118, 0149, 0151, 0152, 0155`)** giữ ở **`REJECT`** do lỗi hình ảnh hiển vi (`0087, 0118`), chụp côn trùng/động vật (`0149, 0155`), hoặc chai nhựa bị lọt vào truy vấn thủy tinh (`0151, 0152`).
  - **`0` ca** được nâng lên `REAL_WASTE_PASS`.
