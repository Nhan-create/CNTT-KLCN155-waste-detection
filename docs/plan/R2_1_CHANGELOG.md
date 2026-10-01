# NHẬT KÝ THAY ĐỔI KỸ THUẬT (CHANGELOG) — TASK R2.1

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Phiên bản:** Task R2.1  
**Ngày thực hiện:** 01/10/2026  
**Người thực hiện:** Tech Lead ML  

---

## 1. Môi trường và Hạ tầng (Environment & Hardware)
- [x] **Cài đặt PyTorch CUDA:** Cài đặt thành công `torch==2.5.1+cu121` và `torchvision==0.20.1+cu121` vào virtualenv `.venv` tại `D:\CNTT-KLCN155-waste-detection\.venv`.
- [x] **Xác thực GPU:** Nhận diện và kích hoạt card đồ họa **NVIDIA GeForce RTX 2050 Laptop GPU (4.095,5 MB VRAM)**; CUDA driver 591.86.
- [x] **Cài đặt thư viện bổ trợ:** Đã cài đặt `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `tqdm==4.70.1`.

## 2. Kiểm toán Dữ liệu và Sửa lỗi Giải thuật (Data Audit & Algorithmic Bugfixes)
- [x] **Sửa lỗi gán cứng số liệu trong audit:** Đã xóa bỏ dòng gán cứng `"cross_source_duplicates": 0` tại `scripts/audit_and_reconcile_data.py`. Số liệu đếm đúng thực tế trong `duplicate_groups.csv` là **890 cặp trùng liên nguồn** giữa Garbage V2 và VN Trash (trong đó **879 cặp trùng tuyệt đối MD5**).
- [x] **Tạo script kiểm tra rò rỉ phân chia tập độc lập:** Tạo [scripts/check_split_leakage.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/check_split_leakage.py) tính toán ma trận khoảng cách bitwise pHash trên toàn bộ 14.831 ảnh qua các split Train, Val, Test.
- [x] **Tái hiện 100% các cặp rò rỉ của PM:** Tái hiện chính xác 18 cặp ứng viên pHash $\le 4$ xuyên split; kiểm định trực quan phát hiện **15 cặp rò rỉ dữ liệu thật** (10 cặp burst-shot cùng vật thể, 5 cặp cùng vật thể xoay góc) và 3 cặp trùng ngẫu nhiên khác lớp.
- [x] **Xuất bản trực quan hóa rò rỉ:** Tạo script [scripts/generate_phash_visual_audit.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/generate_phash_visual_audit.py) và kết xuất 18 ảnh so sánh trực quan song song (side-by-side) tại `data/audit/visual_phash_inspection/pair_01_*.jpg` đến `pair_18_*.jpg`.
- [x] **Đặt trạng thái Cổng Dữ liệu:** Đặt trạng thái `DATA_GATE_FLAGGED_WITH_LEAKAGE` cho split vật lý cũ; yêu cầu gom cụm 15 cặp này trước khi chạy huấn luyện chính thức báo cáo khóa luận.

## 3. Kiến trúc và Chuyển giao Trọng số (Architecture & Transferability)
- [x] **Kiểm toán chuyển giao Backbone sang SSDLite320:** Tạo script [scripts/audit_backbone_transfer.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/audit_backbone_transfer.py); đối soát từng tầng giữa `mobilenet_v3_large` và `ssdlite320_mobilenet_v3_large`.
- [x] **Xuất bảng đối soát chi tiết:** Xuất [data/audit/backbone_transfer_keys.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_keys.csv) (312 dòng) và [data/audit/backbone_transfer_audit.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json).
- [x] **Thực nghiệm nạp trọng số và suy luận:** Xác nhận 258/308 keys trích xuất đặc trưng (83,8%) khớp hoàn hảo; nạp thử nghiệm `load_state_dict(strict=False)` không gặp lỗi; chạy forward pass thành công trên tensor `torch.randn(1, 3, 320, 320)`, trích xuất đúng 300 boxes dự đoán.

## 4. Thực nghiệm Kỹ thuật Chạy thử (Smoke Test Execution)
- [x] **Xây dựng script chạy thử chuẩn:** Viết [scripts/run_smoke_test.py](file:///D:/CNTT-KLCN155-waste-detection/scripts/run_smoke_test.py) tối ưu hóa cho phần cứng RTX 2050, kích hoạt AMP fp16 và xử lý mã hóa UTF-8 trên Windows.
- [x] **Huấn luyện 3 Epochs trên Train/Val:**
  - Train: 10.381 ảnh | Val: 2.225 ảnh | Test: 2.225 ảnh (**Khóa chặt 100% — Không nạp**)
  - Epoch 1: Train Loss 0.8835, Val Acc 92.99%, Val Macro-F1 0.9303, VRAM 1.511,2 MB.
  - Epoch 2: Train Loss 0.6509, Val Acc 94.34%, Val Macro-F1 0.9421, VRAM 1.511,2 MB.
  - Epoch 3: Train Loss 0.5889, Val Acc 94.56%, Val Macro-F1 0.9430, VRAM 1.511,2 MB.
- [x] **Kiểm tra nạp lại Checkpoint (Reload Test):** Lưu checkpoint 48,58 MB tại `artifacts/smoke_test/best_smoke_checkpoint.pt`. Nạp vào mô hình mới và kiểm tra dự đoán: sai lệch $\text{Max Diff} = 0.00000000\text{e+}00$.

## 5. Quy trình và Chuẩn hóa Tài liệu (Process & Protocol Alignment)
- [x] **Xóa bỏ vi phạm quy trình Test Set:** Loại bỏ hoàn toàn vòng lặp "nếu fail test thì điều chỉnh loss rồi train lại" trong `06_SINGLE_OBJECT_EVALUATION_GATE.md`. Quy định Test Set chỉ mở đúng 1 lần duy nhất sau khi mô hình đóng băng.
- [x] **Bác bỏ nhận định VN Trash là tập test ngoại bộ:** Khẳng định VN Trash đã nằm trong Train (1.801 ảnh) và trùng với Garbage V2 (879 ảnh). Tập test ngoại bộ bắt buộc phải là ảnh chụp thực tế TP.HCM mới.
- [x] **Đồng bộ toàn bộ tài liệu:** Cập nhật `03_SPLITS_AND_PREPROCESSING.md`, `04_GITHUB_REUSE_AND_DEPENDENCIES.md`, `05_SINGLE_OBJECT_TRAINING.md`, `06_SINGLE_OBJECT_EVALUATION_GATE.md`, `BAO_CAO_TONG_KET_KIEM_KE_VA_KE_HOACH_CHI_TIET.md`.
