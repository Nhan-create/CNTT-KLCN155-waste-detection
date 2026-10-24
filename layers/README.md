# HỆ THỐNG PHÂN TẦNG VÀ QUẢN LÝ SCRIPTS (CNTT-KLCN155)

Thư mục `layers/` được tạo lập nhằm cấu trúc hóa toàn bộ **210 scripts thực thi** theo 5 tầng chức năng rõ ràng, giúp dễ dàng tra cứu, kiểm tra luồng và gọi chạy theo đúng thứ tự logic mà **không làm xáo trộn đường dẫn gốc `scripts/`**.

Các tệp bên trong mỗi thư mục chức năng là **symlink tượng trưng tương đối** trỏ trực tiếp về script gốc trong `scripts/`. Người dùng có thể chạy trực tiếp từ `layers/...` hoặc từ `scripts/...`.

---

## CẤU TRÚC DANH MỤC CÁC TẦNG

```text
layers/
├── layer_1_data_acquisition/                     # TẦNG 1: Thu thập Dữ liệu
│   ├── 01_download_datasets_chuan_va_pilot/      # 7 files: Tải dữ liệu Kaggle, OpenImages, pilot
│   ├── 02_cao_va_tim_kiem_theo_dot/              # 17 files: Task 7, Task 8 (Wave 1, 2, Task 8a..8h)
│   └── 03_ingestion_va_thu_thap_thuc_dia/        # 8 files: Ingest Roboflow, Field capture thực địa
│
├── layer_2_data_processing/                      # TẦNG 2: Xử lý Dữ liệu, Gán nhãn & Chia tập
│   ├── 01_tien_xu_ly_va_cau_truc_dataset/        # 8 files: Parse YOLO format, ledger 5000, dataset v3
│   ├── 02_kiem_tra_khu_trung_phash_va_leakage/   # 8 files: Quét trùng pHash, BK-tree, kiểm tra rò rỉ split
│   ├── 03_gan_nhan_va_kiem_dinh_qa/              # 13 files: Gán nhãn BBox, QA gates Task 7 & Task 8
│   ├── 04_hieu_chinh_ngu_nghia_va_reannotation/  # 7 files: Sửa nhãn sai ngữ nghĩa, re-annotate pin 140
│   ├── 05_tinh_chinh_bounding_box_va_pixel/      # 17 files: Khớp pixel contour (chuối, vỏ lon, junk)
│   └── 06_chia_tap_va_kiem_toan_phan_phoi/       # 22 files: Tạo split v1/v2, kiểm toán phân phối 10 lớp
│
├── layer_3a_model_training/                      # TẦNG 3A: Huấn luyện Mô hình
│   ├── 01_huan_luyen_phan_loai_stage_a/          # 4 files: Train MobileNetV3-Large 10 lớp, chuyển giao backbone
│   └── 02_huan_luyen_detector_stage_b/           # 8 files: Train SSDLite320 & YOLOv8n, pilot training
│
├── layer_3b_evaluation_and_benchmarks/           # TẦNG 3B: Đánh giá & Kiểm thử
│   ├── 01_danh_gia_chuan_va_baseline/            # 9 files: Test set evaluation, baseline holdout 27
│   ├── 02_do_thoi_gian_latency_va_chan_doan/     # 4 files: Benchmark FPS/Latency RTX 2050 (Batch=1)
│   └── 03_smoke_test_va_tai_hien_loi/            # 6 files: Smoke test, reproduce demo image 17, lỗi box
│
└── layer_4_reporting_and_delivery/               # TẦNG 4: Báo cáo & Bàn giao
    ├── 01_xuat_ket_qua_va_bang_chung/            # 8 files: Export test evidence, predictions raw
    ├── 02_bieu_do_bao_cao_va_contact_sheets/     # 33 files: Biểu đồ PR, contact sheets, forensic reports
    ├── 03_dong_goi_ban_giao_va_artifacts/        # 29 files: Đóng gói ZIP Phase 4, Part 1, Task 6/7/8
    └── 04_kiem_thu_giao_dien_review_tool/        # 2 files: Test kiểm thử công cụ Review Tool UI
```

---

## HƯỚNG DẪN THỰC THI THEO THỨ TỰ (EXECUTION WORKFLOW)

1. **Thu thập dữ liệu thô (Tầng 1):**
   ```bash
   python layers/layer_1_data_acquisition/01_download_datasets_chuan_va_pilot/download_datasets.py
   ```
2. **Tiền xử lý, lọc trùng và gán nhãn (Tầng 2):**
   ```bash
   python layers/layer_2_data_processing/01_tien_xu_ly_va_cau_truc_dataset/prepare_detection_dataset.py
   python layers/layer_2_data_processing/02_kiem_tra_khu_trung_phash_va_leakage/scan_all_phash_candidates.py
   python layers/layer_2_data_processing/06_chia_tap_va_kiem_toan_phan_phoi/build_real_v2_splits.py
   ```
3. **Huấn luyện mô hình (Tầng 3A):**
   ```bash
   python layers/layer_3a_model_training/01_huan_luyen_phan_loai_stage_a/train_official_mobilenetv3.py
   python layers/layer_3a_model_training/02_huan_luyen_detector_stage_b/train_detection_v2.py yolo --run E1_yolov8n_10cls --data configs/detection_dataset_real_v2.yaml
   ```
4. **Đánh giá và đo tốc độ (Tầng 3B):**
   ```bash
   python layers/layer_3b_evaluation_and_benchmarks/01_danh_gia_chuan_va_baseline/run_current_baseline_eval.py
   python layers/layer_3b_evaluation_and_benchmarks/02_do_thoi_gian_latency_va_chan_doan/benchmark_inference.py
   ```
5. **Xuất báo cáo và đóng gói (Tầng 4):**
   ```bash
   python layers/layer_4_reporting_and_delivery/01_xuat_ket_qua_va_bang_chung/export_official_results.py
   python layers/layer_4_reporting_and_delivery/03_dong_goi_ban_giao_va_artifacts/build_final_delivery.py
   ```
