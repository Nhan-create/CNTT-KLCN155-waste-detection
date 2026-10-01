# NHẬT KÝ THAY ĐỔI VÀ HOÀN THIỆN KỸ THUẬT (CHANGELOG R1) — PHẦN 1

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3  
**Phiên:** Task P1-R1 — Hoàn thiện bằng chứng và khả năng tái hiện để nghiệm thu  
**Ngày thực hiện:** 02/10/2026  
**Người thực hiện:** Tech Lead ML (Antigravity Engineering Team)  

---

## 1. Sửa Kiểm toán Rò rỉ Dữ liệu và Thẩm định Trực quan (Leakage Audit Rectification)

- [x] **Xóa bỏ giả định tự động:** Loại bỏ hoàn toàn nhánh gán tự động `c1 != c2 -> CROSS_CLASS_COINCIDENCE` trong [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py).
- [x] **Quét toàn bộ dữ liệu (All-Class Pairwise Scan):**
  - Viết và thực thi [scripts/scan_all_phash_candidates.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/scan_all_phash_candidates.py) quét toàn bộ ma trận $14.829 \times 14.829$ ảnh sạch với ngưỡng khoảng cách Hamming $\le 4$.
  - Phát hiện tổng cộng **33 cặp ứng viên**: 29 cặp cùng lớp (đều nằm cùng split) và 4 cặp khác lớp (3 cặp xuyên split, 1 cặp nội bộ train).
- [x] **Thẩm định trực quan và lập Bảng Quyết định:**
  - Viết [scripts/build_comprehensive_visual_audit.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/build_comprehensive_visual_audit.py) tạo bảng quyết định [data/audit/visual_audit_decision_table.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/visual_audit_decision_table.csv) gồm 36 bản ghi thẩm định chi tiết.
  - Tạo đầy đủ 36 ảnh đối chiếu song song tại [data/audit/visual_phash_inspection/](file:///D:/CNTT-KLCN155-waste-detection/data/audit/visual_phash_inspection/).
  - Xác nhận 4 cặp khác lớp đều là các vật thể vật lý khác nhau hoàn toàn trên nền trắng (trùng ngẫu nhiên hình học), không có rò rỉ hay xung đột nhãn mới.
- [x] **Cập nhật script kiểm toán liên thông:** Sửa [scripts/find_all_transitive_clusters.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/find_all_transitive_clusters.py) xây dựng đồ thị NetworkX toàn cục, xác nhận 0 cụm vắt qua các split.
- [x] **Xác thực mã băm tệp trực tiếp trên đĩa:** Kiểm tra byte stream của 14.829 tệp vật lý trong `data/processed_v2/`, xác nhận 100% khớp mã băm SHA-256 với `split_manifest_v2.csv` (0 file thiếu, 0 file hỏng).

---

## 2. Đối chứng Lịch sử Huấn luyện và Chuỗi Chứng cứ (Training Provenance)

- [x] **Bảo toàn log gốc của lần huấn luyện chính thức:**
  - Sao chép log gốc của tác vụ huấn luyện P1.6 sang [artifacts/official_run/official_training_raw_execution.log](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_raw_execution.log) (12.574 bytes, SHA-256: `a1c6ca0c5beca94b7b8e22f3bdfbc9645b8579df981602d4198211cf9355f7ed`).
- [x] **Xóa bỏ hardcoding trong script xuất báo cáo:**
  - Viết lại [scripts/export_official_results.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/export_official_results.py) để phân tích cú pháp trực tiếp 12 epochs từ log thô bằng regular expression.
  - Tính toán động mã băm SHA-256 của checkpoint và manifest tại thời điểm chạy.
  - Phân tách rõ ràng các trường đo trực tiếp (`peak_vram_mb`, `epoch_time_sec`), trường đọc cấu hình và trường nguồn gốc dữ liệu trong [artifacts/official_run/official_training_metrics.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_metrics.json).
- [x] **Tăng cường độ bền vững của script train:**
  - Cập nhật [scripts/train_official_mobilenetv3.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/train_official_mobilenetv3.py) tự động ghi lịch sử epoch vào `training_history.csv` ngay khi mỗi epoch hoàn tất.

---

## 3. Tái hiện Suy luận Validation Độc lập (Validation Reproduction)

- [x] **Xây dựng script tái hiện độc lập:**
  - Viết [scripts/reproduce_validation.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/reproduce_validation.py) nhận các tham số dòng lệnh cấu hình linh hoạt.
  - Tải checkpoint `best_model.pt` và chạy suy luận trên 2.223 ảnh validation.
- [x] **Xuất Bảng Dự đoán Chi tiết Từng Mẫu:**
  - Tạo tệp [artifacts/official_run/val_predictions.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_predictions.csv) (2.223 dòng), ghi nhận đầy đủ `filename, true_label, predicted_label, confidence, is_correct, sha256`.
- [x] **Kiểm chứng Khớp 100% Số liệu Baseline:**
  - Accuracy: **`96.18%`** (2.138 / 2.223 mẫu, chênh lệch $= 0.00e+00$).
  - Macro-F1: **`0.9559`** (chênh lệch $= 0.00e+00$).
  - Tổng số lỗi: **`85`** lỗi (chênh lệch $= 0$).
  - Battery Recall: **`94.69%`** (Precision: 100.00%, F1: 0.9727).
  - Tóm tắt xuất bản tại [artifacts/official_run/reproduced_validation_summary.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/reproduced_validation_summary.json).

---

## 4. Hoàn thiện Gói Bàn giao Độc lập (Self-Contained Packaging)

- [x] **Đưa metadata vào repository:**
  - Sao chép [data/metadata/dataset_info.csv](file:///D:/CNTT-KLCN155-waste-detection/data/metadata/dataset_info.csv) (2.298.135 bytes, 15.755 dòng) vào repo; gỡ bỏ mọi phụ thuộc vào thư mục bên ngoài.
- [x] **Biên soạn tài liệu hướng dẫn tái hiện:**
  - Viết [docs/plan/REPRODUCIBILITY_GUIDE.md](file:///D:/CNTT-KLCN155-waste-detection/docs/plan/REPRODUCIBILITY_GUIDE.md) hướng dẫn chi tiết các bước thiết lập môi trường, kiểm tra mã băm và chạy lại các script trên máy khác.
- [x] **Đóng gói phiên bản kiểm chứng mới:**
  - Cập nhật script đóng gói [scripts/package_part01_delivery.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/package_part01_delivery.py) tạo gói `CNTT-KLCN155_PART01_VERIFIED_R1.zip` kèm tệp manifest mã băm SHA-256 từng tệp tin.
