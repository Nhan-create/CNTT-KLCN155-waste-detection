# NHẬT KÝ THAY ĐỔI VÀ CỦNG CỐ CỔNG KIỂM CHỨNG (CHANGELOG R2) — PHẦN 1

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3  
**Phiên:** Task P1-R2 — Khắc phục 3 lỗ hổng cổng kiểm chứng và bảo đảm tính tái hiện tuyệt đối  
**Ngày thực hiện:** 02/10/2026  
**Người thực hiện:** Tech Lead ML (Antigravity Engineering Team)  

---

## 1. Khắc phục 3 Lỗ hổng Cổng Kiểm chứng (Gate Vulnerability Fixes)

- [x] **Lỗ hổng 1: Cổng kiểm tra đĩa vật lý không chặn lỗi khi thiếu thư mục ảnh hoặc hỏng file**
  - **Script sửa đổi:** [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py)
  - **Thay đổi:**
    - Kiểm tra đĩa vật lý là bước bắt buộc (`disk_performed = not args.skip_disk_hash`).
    - Nếu thư mục `processed_dir` không tồn tại, đánh dấu toàn bộ mẫu là thiếu (`missing_count = len(manifest)`), đặt `disk_verified = False`.
    - Phán quyết cuối cùng đưa `disk_verified` thành điều kiện tiên quyết: Nếu thất bại, trả về trạng thái `FAILED_DISK_IMAGE_VERIFICATION` và mã thoát `exit 1`.
- [x] **Lỗ hổng 2: Đếm sai số cặp ứng viên `UNRESOLVED` khi đã có trong bảng quyết định**
  - **Script sửa đổi:** [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py)
  - **Thay đổi:**
    - Thêm biến cờ `is_unresolved = False`.
    - Khi tra cứu trong `decision_map`, nếu `audit_status != 'VERIFIED'` hoặc `verdict.startswith('UNRESOLVED')`, kích hoạt `is_unresolved = True`.
    - Nếu `is_unresolved = True`, luôn tăng `unresolved_count += 1`.
    - Khi `unresolved_count > 0`, phán quyết cuối cùng trả về `UNRESOLVED_CANDIDATES_PRESENT` và mã thoát `exit 1`.
- [x] **Lỗ hổng 3: Bảng dự đoán validation chưa tính mã băm trực tiếp từ từng byte ảnh đọc từ đĩa**
  - **Script sửa đổi:** [scripts/reproduce_validation.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/reproduce_validation.py)
  - **Thay đổi:**
    - Tính mã băm SHA-256 trực tiếp từ file ảnh vật lý trên đĩa (`actual_disk_sha256 = compute_sha256(sample_p)`).
    - Đối chiếu mã băm tính từ đĩa với manifest; nếu có bất kỳ sự sai lệch nào (`disk_hash_mismatches > 0`), dừng script ngay lập tức (`exit 1`).
    - Lưu mã băm tính từ đĩa vào cột `sha256` của [artifacts/official_run/val_predictions.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_predictions.csv).
    - Thêm khối `disk_hash_verification` vào [artifacts/official_run/reproduced_validation_summary.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/reproduced_validation_summary.json).

---

## 2. Xây dựng Bộ Kiểm thử Tự động Kiểm chứng Cổng (Verification Gates Test Suite)

- [x] **Viết bộ kiểm thử độc lập:** [tests/test_verification_gates.py](file:///D:/CNTT-KLCN155-waste-detection/tests/test_verification_gates.py)
  - **Test 1 (Thiếu thư mục ảnh):** Chạy `check_split_leakage.py` với đường dẫn không tồn tại $\to$ Báo lỗi `FAILED_DISK_IMAGE_VERIFICATION`, mã thoát `exit 1` (**PASSED**).
  - **Test 2 (Cặp ứng viên mang trạng thái `UNRESOLVED`):** Chạy `check_split_leakage.py` với bảng quyết định có 1 hàng `UNRESOLVED` $\to$ Báo lỗi `UNRESOLVED_CANDIDATES_PRESENT`, phát hiện 1 cặp chưa giải quyết, mã thoát `exit 1` (**PASSED**).
  - **Test 3 (Kiểm tra dữ liệu chính thức):** Quét toàn bộ 14.829 ảnh vật lý $\to$ Đạt 100% không rò rỉ, 0 va chạm, 0 lỗi đĩa, mã thoát `exit 0` (**PASSED**).
  - **Test 4 (Tái hiện validation với băm trực tiếp từ đĩa):** Quét 2.223 ảnh $\to$ 0 sai lệch mã băm, khớp chính xác 2.138 đúng / 85 sai, Accuracy 96,18%, Macro-F1 0,9559, mã thoát `exit 0` (**PASSED**).

---

## 3. Cập nhật Đóng gói và Bàn giao (Packaging Updates)

- [x] **Cập nhật script đóng gói:** [scripts/package_part01_delivery.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/package_part01_delivery.py)
  - Bổ sung thư mục `tests` vào danh mục đóng gói.
  - Tạo gói chính thức: `CNTT-KLCN155_PART01_VERIFIED_R2.zip` (192 tệp, 94.04 MB, SHA-256: `27ac768a528fd587b74733e88c056132b1b4869e75f4a1bbe73fa0300832307d`).
  - Cập nhật gói tương thích: `CNTT-KLCN155_PART01_VERIFIED_R1.zip` (192 tệp, 94.04 MB, SHA-256: `345a7219b36989f5fbd64fcdd02150fdf93618dffcaf5dc64cf2e980b56e1c66`).
  - Xuất bảng kê chi tiết từng file: `CNTT-KLCN155_PART01_VERIFIED_R2_manifest.csv` và `CNTT-KLCN155_PART01_VERIFIED_R1_manifest.csv`.
