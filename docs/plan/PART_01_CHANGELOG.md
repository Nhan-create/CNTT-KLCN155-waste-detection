# NHẬT KÝ THAY ĐỔI KỸ THUẬT (CHANGELOG) — PHẦN 1

**Dự án:** Hệ thống Phân loại và Phát hiện Rác thải Sinh hoạt (`CNTT-KLCN155`)  
**Học phần:** PHẦN 1 — Tiền xử lý dữ liệu và huấn luyện MobileNetV3 phân loại rác  
**Ngày thực hiện:** 01/10/2026  
**Người thực hiện:** Tech Lead & Machine Learning Engineer  

---

## 1. Kiểm toán Dữ liệu và Sửa lỗi Báo cáo (Audit & Bugfixes)
- [x] **Chạy lại và cập nhật `audit_summary.json`:** Khắc phục tình trạng script cũ chưa được chạy lại trước khi nén zip; chạy lại `scripts/audit_and_reconcile_data.py`, ghi nhận đúng `cross_source_duplicates = 890` (879 exact MD5) và cập nhật timestamp mới.
- [x] **Thẩm định chi tiết 18 cặp pHash ứng viên:**
  - Viết [scripts/inspect_plate_details.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/inspect_plate_details.py) phân tích trực quan từng ảnh, tỷ lệ sáng/tối, độ tương phản và hình thái vật thể.
  - Tái đánh giá Cặp 10 (`train_cardboard438.jpg` vs `paper_582.jpg`): Phát hiện xung đột nhãn giữa hai nguồn (`cardboard` vs `paper`) cho cùng một vật thể cốc giấy/bìa. Đưa cả 2 mẫu vào diện cách ly [data/audit/quarantined_samples.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/quarantined_samples.csv).
  - Tái đánh giá Cặp 12 (`metal_435.jpg` vs `metal_158.jpg`): Xác nhận là hai lon kim loại khác nhau, bối cảnh khác nhau (lon có bóng tối bên phải vs lon đứng giữa) $\to$ Trùng ngẫu nhiên hình học.
  - Tái đánh giá Cặp 18 (`Foam_box/train_00000024.jpg` vs `Foam_box/train_00000012.jpg`): Hai hộp xốp khác hình dạng (hộp kín góc thấp vs hộp mở hai ngăn) $\to$ Trùng ngẫu nhiên hình học.
- [x] **Kiểm toán chuỗi liên kết bắc cầu (Transitive Chains):**
  - Viết [scripts/find_all_transitive_clusters.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/find_all_transitive_clusters.py) sử dụng NetworkX phân tích thành phần liên thông trên toàn bộ 14.831 ảnh sạch.
  - Xác nhận 29 cụm gần trùng cùng lớp ($\text{size} = 2$), không có chuỗi bắc cầu bậc 3 trở lên.
  - Xuất báo cáo [data/audit/transitive_chains_report.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/transitive_chains_report.json).

## 2. Tạo Phân chia Mới và Kiểm chứng Triệt tiêu Rò rỉ (Split V2 Generation)
- [x] **Tạo script phân chia theo nhóm:** Viết [scripts/create_split_v2.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/create_split_v2.py) áp dụng giải thuật Disjoint Set Union (DSU) gom toàn bộ các cặp gần trùng thành các đơn vị nguyên tử.
- [x] **Phân bổ Stratified Group Split (Seed 42):**
  - Train: **10.383** ảnh ($70,02\%$)
  - Val: **2.223** ảnh ($14,99\%$)
  - Test: **2.223** ảnh ($14,99\%$)
  - Tổng cộng sạch: **14.829** ảnh (đã cách ly 2 ảnh xung đột nhãn).
- [x] **Vật lý hóa dữ liệu:** Sử dụng cơ chế NTFS Hardlink tạo thư mục vật lý [data/processed_v2/](file:///D:/CNTT-KLCN155-waste-detection/data/processed_v2/) tức thì không tốn dung lượng ổ đĩa.
- [x] **Xuất Split Manifest V2:** Lưu tại [data/audit/split_manifest_v2.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/split_manifest_v2.csv) (mã băm SHA-256: `1429a22ebbf95cc464881f903dae5ac6dcc5e6e64152e1dd3353e9365429ec96`).
- [x] **Kiểm toán rò rỉ độc lập trên Split V2:** Chạy [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py) xác nhận `ZERO_LEAKAGE_VERIFIED` (0 cặp rò rỉ burst-shot, 0 cặp rò rỉ SHA-256).

## 3. Pipeline Tiền xử lý và Kiểm định (Preprocessing Verification)
- [x] **Kiểm định Pipeline và DataLoader:** Viết và chạy [scripts/verify_preprocessing_pipeline.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/verify_preprocessing_pipeline.py).
- [x] **Đối soát số lượng tệp vật lý:** 10.383 ảnh Train và 2.223 ảnh Val khớp chính xác 100% với manifest.
- [x] **Đồng nhất `class_to_idx`:** Khớp chuẩn 10 lớp ImageFolder.
- [x] **Kiểm tra tensor batch thực tế:** Dải giá trị chuẩn hóa ImageNet `[-2.118, 2.640]`, 0 giá trị NaN/Inf.
- [x] **Xuất báo cáo kiểm định:** Lưu tại [data/audit/preprocessing_pipeline_verification.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/preprocessing_pipeline_verification.json).

## 4. Thực nghiệm Chạy thử (Smoke Test V2)
- [x] **Viết script chạy thử trên Split V2:** Tạo [scripts/run_smoke_test_v2.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/run_smoke_test_v2.py) chạy 2 epochs trên GPU RTX 2050 (AMP fp16, batch size 64).
- [x] **Kết quả thực thi:**
  - Epoch 1: Train Loss 0.8714, Val Loss 0.6963, Val Acc 93.57%, Val Macro-F1 0.9308.
  - Epoch 2: Train Loss 0.6538, Val Loss 0.6573, Val Acc 95.19%, Val Macro-F1 0.9504.
- [x] **Kiểm tra nạp lại Checkpoint nghiêm ngặt:** Đánh giá lại mô hình nạp trên toàn bộ tập Validation, khớp chính xác từng giá trị: sai lệch Loss diff = 0.0, Acc diff = 0.0, F1 diff = 0.0.
- [x] **Lưu trữ artifact:** Checkpoint lưu tại [artifacts/smoke_test_v2/best_smoke_checkpoint.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/smoke_test_v2/best_smoke_checkpoint.pt).

## 5. Huấn luyện Chính thức MobileNetV3 (Official Fine-Tuning)
- [x] **Chiến lược Two-Phase Fine-Tuning:**
  - Phase 1: 3 epochs Warmup đóng băng backbone, chỉ huấn luyện classifier head ($lr=10^{-3}$).
  - Phase 2: 9 epochs Fine-Tuning toàn bộ mạng ($lr_{backbone}=10^{-4}, lr_{head}=3 \times 10^{-4}$) kèm CosineAnnealingLR.
  - Early stopping: Dừng sau 4 epochs không cải thiện Val Macro-F1.
  - Checkpoint selection: Chọn mô hình có Val Macro-F1 cao nhất.
- [x] **Kết quả thực nghiệm 12 epochs trên GPU RTX 2050:**
  - Hoàn thành đầy đủ 12 epochs trong ~25 phút, không gặp lỗi bộ nhớ (peak VRAM: 1.514,5 MB).
  - Checkpoint tốt nhất lưu tại **Epoch 12** ([artifacts/official_run/best_model.pt](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/best_model.pt), 48,57 MB).
  - Validation Accuracy: **96.18%** (2.138 / 2.223 mẫu chính xác).
  - Validation Macro-F1: **0.9559**.
  - Battery Recall: **94.69%** (Precision: 100.00%, F1: 0.9727) — vượt xa ngưỡng cổng $\ge 88.0\%$.
  - 10/10 lớp đều có Recall $\ge 85.33\%$ (Clothes và Shoes đạt 100.00%).
- [x] **Kiểm tra nạp lại Checkpoint (Reload Test):**
  - Đánh giá lại mô hình nạp từ checkpoint trên toàn bộ 2.223 ảnh Validation.
  - Sai lệch Loss diff = 0.0, Acc diff = 0.0, Macro-F1 diff = 0.0 (khớp bitwise/float tuyệt đối).
- [x] **Xuất bản đầy đủ các tạo tác (Artifacts):**
  - [artifacts/official_run/training_history.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/training_history.csv)
  - [artifacts/official_run/val_confusion_matrix.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_confusion_matrix.csv)
  - [artifacts/official_run/val_error_analysis.csv](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/val_error_analysis.csv) (85 lỗi / 2.223 mẫu)
  - [artifacts/official_run/official_training_metrics.json](file:///D:/CNTT-KLCN155-waste-detection/artifacts/official_run/official_training_metrics.json)
- [x] **Cam kết Khóa tập Test:**
  - 100% không truy cập tập Test (2.223 ảnh tại `data/processed_v2/test/`). Zero snooping.

