# TASK 8F FINAL SAFETY CHECK — CLOSE SEMANTIC QA SIGN-OFF REPORT

- **Repository:** `.`
- **Manifest Version:** `TASK8F_SCARCE_CLASSES_V2_SEMANTIC_QA_CORRECTED`
- **Manifest JSON SHA-256 (Idempotent Run 1 == Run 2):** `7a24e9344c8e410b70269ce42b8408d18016ce1f664f0cee36f5f91d8529bb85`
- **Manifest CSV SHA-256 (Idempotent Run 1 == Run 2):** `8e11b1068791b551a726ae363c19be2197a82f84f4f3f87fca8f9fbed396b83d`
- **Training / Deployment Gate:** `APPROVED_FOR_TRAIN = False` trên toàn bộ `748/748` bản ghi (KHÔNG train E4/E5, KHÔNG deploy, giữ nguyên E1/E3 và các tập test/holdout).

---

## 1. TASK 1 — KIỂM TRA & GIA CỐ NMS VÀ BBOX (`deduplicate_same_class_boxes_iou` & `agnostic_nms`)

### 1.1. Phát hiện rủi ro trong code trước khi gia cố & Cách xử lý
1. **Rủi ro 1 — `deduplicate_same_class_boxes_iou` và `deduplicate_boxes_struct` dùng ngưỡng phẳng `IoU > 0.70` / `0.65`:**
   - Trước khi gia cố, hàm deduplicate xóa mọi cặp cùng lớp có `IoU > 0.70` (hoặc `> 0.65` ở tập phụ) mà không kiểm tra xem hai box có cùng tâm/cạnh của **một vật thể duy nhất** hay là **hai vật thể thực sự chồng lấp mạnh**.
   - Hệ quả thực tế phát hiện được: Ở các bản ghi `KEEP_AUXILIARY` / `NEEDS_REVIEW` từ Open Images V7 và Wikimedia (`task8f_0052`, `task8f_0409`, `task8f_0440`, `task8f_0462`, `task8f_0472` x2, `task8f_0492`, `task8f_0493`, `task8f_0497`), có **9 cặp ground-truth cùng lớp chồng lấp hợp lệ** (`IoU = 0.7042 .. 0.9002`, `max_edge_delta = 0.0596 .. 0.2242`) từng bị xóa tự động để ép số cặp `IoU > 0.70` về `0`.
   - **Biện pháp khắc phục:** Bổ sung hàm `classify_box_pair_overlap(box_a, box_b)` phân biệt rõ 3 trường hợp vật lý:
     - `DUPLICATE_ANNOTATION`: Cùng lớp (`cid_a == cid_b`), `IoU >= 0.90` **và** các cạnh trùng khớp (`max_edge_delta <= 0.045`) — đây là lỗi sinh 2 box trùng một vật thể do ánh xạ lớp detector -> **Gộp còn 1 box**.
     - `VALID_SAME_CLASS_OVERLAP`: Cùng lớp (`cid_a == cid_b`), `IoU > 0.70` nhưng khác vị trí cạnh/tâm (`IoU < 0.90` hoặc `max_edge_delta > 0.045`) — đây là hai vật thể cùng lớp chồng lấp mạnh -> **Giữ nguyên cả 2 box (`keep_both=True`) và gắn cờ `requires_visual_review=True`** (tuyệt đối không tự động xóa!).
     - `VALID_CROSS_CLASS_OVERLAP`: Khác lớp (`cid_a != cid_b`, `IoU > 0.0`) -> **Giữ nguyên cả 2 box và cả 2 lớp (`keep_both=True`)**.
   - Khi `is_ground_truth=True` (nhãn từ Open Images GT hoặc nhãn đã qua người kiểm duyệt), `norm_boxes_to_struct` không bao giờ chạy NMS cắt tỉa các vật thể chồng lấp.
2. **Rủi ro 2 — `agnostic_nms=True` trong `extract_tight_per_instance_boxes`:**
   - Trước khi gia cố, `detector.predict(..., agnostic_nms=True, iou=0.65)` kết hợp gán đè `out_c = primary_cid` có thể triệt tiêu nhãn của vật thể thứ hai khác lớp nằm chồng lên vật thể thứ nhất (ví dụ thức ăn `1:biological` nằm trong hộp `2:cardboard` như ở `task8f_0694`).
   - **Biện pháp khắc phục:** Chuyển sang `agnostic_nms=False` (NMS phân biệt theo lớp) và giữ nguyên class dự đoán khi `conf >= 0.45` khác `primary_cid`.

### 1.2. Thống kê cặp BBox chồng lấp trên toàn bộ 748 ứng viên sau khi gia cố
- **Số cặp `DUPLICATE_ANNOTATION` (trùng 1 vật thể) trên 748 ảnh:** `0` cặp (toàn bộ 33 cặp lỗi ở 23 ảnh trước sửa đã được xử lý triệt để).
- **Số cặp `DUPLICATE_ANNOTATION` trong 86 ảnh `REAL_WASTE_PASS`:** `0` cặp.
- **Số cặp `VALID_SAME_CLASS_OVERLAP` (`IoU > 0.70` hợp lệ giữa 2 vật thể cùng lớp) được khôi phục & bảo toàn ở `KEEP_AUXILIARY` / `NEEDS_REVIEW`:** `9` cặp (`task8f_0052`, `task8f_0409`, `task8f_0440`, `task8f_0462`, `task8f_0472` [2 cặp], `task8f_0492`, `task8f_0493`, `task8f_0497`) — không ép về 0.
- **Các ảnh `REAL_WASTE_PASS` có nhiều vật thể chồng lấp hợp lệ được bảo toàn đầy đủ:**
  - `task8f_0285` (`0:battery`): `4` viên pin AA rỉ sét nằm sát nhau (3 cặp giao nhau với `IoU = 0.3668, 0.3547, 0.4388`).
  - `task8f_0694` (`2:cardboard` + `1:biological`): `2` vật thể khác lớp lồng nhau (`VALID_CROSS_CLASS_OVERLAP`, `IoU = 0.0422`, bánh kem `1:biological` nằm hoàn toàn bên trong hộp bánh `2:cardboard` bị vứt trên mặt nhựa đường).
  - `task8f_0716` (`8:shoes`): `4` chiếc giày cũ bỏ rơi xếp chồng cạnh nhau trên đường mòn (`IoU = 0.0952`).
  - `task8f_0039` (`8:shoes`): `2` chiếc giày rách chồng lên nhau (`IoU = 0.1398`).
  - `task8f_0131` (`8:shoes`): `2` chiếc giày Converse chồng mũi (`IoU = 0.1441`).
  - `task8f_0095` (`1:biological`): `3` miếng vỏ cam bỏ đi chồng cạnh (`IoU = 0.1602`).

### 1.3. Kết quả 3 Regression Cases bắt buộc (`test_task8f_final_safety_nms_and_overlap_three_regression_cases`)
| Regression Case | Đầu vào `(cid, x1, y1, x2, y2)` | `IoU` | `max_edge_delta` | Phân loại `classify_box_pair_overlap` | Số box sau `deduplicate` | Trạng thái |
|---|---|---:|---:|---|---:|---|
| **1. Hai annotation trùng 1 vật thể** | `(6, 0.190, 0.250, 0.610, 0.670)` & `(6, 0.194, 0.252, 0.608, 0.668)` | `0.976262` | `0.0040` | `DUPLICATE_ANNOTATION` (`keep_both=False`) | `1` | **PASS** |
| **2. Hai vật thể cùng lớp chồng lấp mạnh** | `(2, 0.200, 0.200, 0.700, 0.700)` & `(2, 0.260, 0.200, 0.760, 0.700)` | `0.785714` | `0.0600` | `VALID_SAME_CLASS_OVERLAP` (`keep_both=True`) | `2` | **PASS** |
| **3a. Hai vật thể khác lớp lồng nhau (`task8f_0694`)** | `(2, 0.210, 0.030, 0.830, 0.940)` & `(1, 0.390, 0.560, 0.540, 0.720)` | `0.042538` | `0.5300` | `VALID_CROSS_CLASS_OVERLAP` (`keep_both=True`) | `2` (`[2, 1]`) | **PASS** |
| **3b. Hai vật thể khác lớp chồng lấp mạnh (`IoU > 0.90`)** | `(4, 0.200, 0.200, 0.700, 0.700)` & `(6, 0.210, 0.210, 0.690, 0.690)` | `0.921600` | `0.0100` | `VALID_CROSS_CLASS_OVERLAP` (`keep_both=True`) | `2` (`[4, 6]`) | **PASS** |

---

## 2. TASK 2 — KIỂM TRA & GIA CỐ HÀM `infer_material_class_from_title`

1. **Chỉ đề xuất nhãn ứng viên (`propose_candidate_class_from_title`):**
   - Hàm trả về cấu trúc đề xuất với `is_candidate_only = True`, `can_auto_promote_qa_pass = False`, `can_auto_assert_actual_waste = False`.
2. **Không tự nâng `QA_PASS` (`REAL_WASTE_PASS`) và không tự khẳng định `ACTUAL_WASTE` từ tên tệp:**
   - Trong nhánh xử lý ứng viên chưa qua kiểm duyệt thủ công (`acquire_and_qa_task8f_scarce_classes.py`), mọi ứng viên có `requires_visual_qa = True` (bất kỳ khi nào tên tệp đề xuất đổi class hoặc thuộc lớp hiếm) đều bị chặn không cho tự động vào `REAL_WASTE_PASS` và không được tự động gán `domain_tag = 'ACTUAL_WASTE'`, mà bắt buộc chuyển vào `NEEDS_REVIEW` (`domain_tag = 'AMBIGUOUS_CONTEXT'`, `REAL_WASTE_CONTEXT_VERIFIED = False`) chờ người kiểm duyệt bằng ảnh gốc.
3. **Bảo toàn tuyệt đối các annotation đã được người kiểm duyệt:**
   - Khi `human_verified = True`, `propose_candidate_class_from_title` và `infer_material_class_from_title` trả về đúng `verified_cid` của người kiểm duyệt bất kể tên tệp có chứa từ khóa gây nhiễu (ví dụ tên tệp có cả `Paper cup` và `Pepsi can`).
   - Ngoài ra, vòng lặp QA trong `acquire_and_qa_task8f_scarce_classes.py` nạp trực tiếp 4 bảng quyết định thủ công (`VERIFIED_REAL_WASTE_PASS_SPECS`, `DEMOTE_TO_AUXILIARY_SPECS`, `MOVE_TO_NEEDS_REVIEW_SPECS`, `REJECT_FROM_PASS_SPECS`) ở đầu tiên để đảm bảo không bao giờ ghi đè nhãn đã duyệt.

---

## 3. TASK 3 — XÁC NHẬN DỮ LIỆU CUỐI & TÍNH IDEMPOTENT

| Chỉ số kiểm tra | Giá trị xác nhận từ Manifest & File vật lý | Trạng thái |
|---|---:|---|
| Tổng số ứng viên Task 8F | `748` ảnh | **ĐẠT** |
| Số ảnh `REAL_WASTE_PASS` | `86` ảnh (`71` `TRAIN_REAL_WASTE_W8F` / `15` `VAL_REAL_WASTE_W8F`) | **ĐẠT** |
| Tổng số bounding boxes `REAL_WASTE_PASS` | `120` boxes (`97` train boxes / `23` val boxes) | **ĐẠT** |
| Số ảnh `KEEP_AUXILIARY` / `NEEDS_REVIEW` / `REJECT` | `379` / `126` / `157` | **ĐẠT** |
| Khớp 1-1 giữa `images/`, `labels_real_waste_pass/`, `overlays/` và manifest | `86/86` ảnh, `86/86` file `.txt` (`120` dòng YOLO), `86/86` ảnh overlay | **ĐẠT** |
| Bảo toàn `source_split` (`0` ảnh source `test` trong train/val) | `0` vi phạm | **ĐẠT** |
| Rò rỉ `session_group` hoặc `specimen_group` giữa train và val | `0` overlap | **ĐẠT** |
| Kiểm tra Idempotent (chạy lại `correct_task8f_semantic_qa.py` 2 lần liên tiếp) | SHA-256 khớp tuyệt đối: `7a24e9344c8e410b...` | **ĐẠT** |
| Cờ `APPROVED_FOR_TRAIN` | `False` (`0/748` ảnh được bật) | **ĐẠT** |
| Bộ kiểm thử tự động `pytest tests/detection/test_task8_manifest_report_consistency.py` | `9/9 PASS` | **ĐẠT** |

---

## 4. DANH SÁCH CÁC CA KHÓ / GÂY TRANH CÃI ĐANG GIỮ Ở `NEEDS_REVIEW` (TIÊU BIỂU)

- `task8f_0712` (`3:clothes`, `2` boxes): Đôi găng tay đấm bốc màu đỏ treo trên dây điện (`Boxing gloves hanging from wires.jpg`) — đã sửa class `8:shoes -> 3:clothes` và tách 2 bbox khít từng chiếc găng, giữ ở `NEEDS_REVIEW` do bối cảnh treo trên dây.
- `task8f_0289` (`0:battery`, `6` boxes): 6 viên pin 9V Duracell đặt cạnh thiết bị báo khói (`9V batteries with smoke detector`) — đã tách 6 bbox từng viên, giữ ở `NEEDS_REVIEW` vì bối cảnh bảo trì trong nhà.
- `task8f_0010` (`8:shoes`, `6` boxes): Giày trượt băng cũ treo trên tường gạch — giữ ở `NEEDS_REVIEW` do có thể là vật trang trí.
- `task8f_0037` (`5:metal`, `4` boxes): 4 lon nước ngọt bẹp có gắn mắt nhựa trang trí — giữ ở `NEEDS_REVIEW` (tái chế thủ công / đồ chơi).
- `task8f_0065` (`8:shoes`, `4` boxes): Nhiều đôi giày treo trên cột đèn đường (`Sneakers on a power line`) — cụm 3 chiếc phía trên chồng lấp phức tạp, giữ ở `NEEDS_REVIEW`.
- `task8f_0076`, `task8f_0166`, `task8f_0231`, `task8f_0696`: Các đống rác hỗn hợp / bãi phế liệu nhiều hơn 6–10 vật thể chồng lấp dày đặc không thể tách hết từng instance độc lập — giữ ở `NEEDS_REVIEW` theo đúng quy tắc instance-detection.
