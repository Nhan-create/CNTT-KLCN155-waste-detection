# HƯỚNG DẪN TÁI HIỆN ĐỘC LẬP TOÀN BỘ KẾT QUẢ KIỂM TOÁN (PHIÊN BẢN V3)
**Dự án:** Kiểm toán độc lập mô hình nhận diện đa loại rác (`CNTT-KLCN155-waste-detection`)  
**Run ID:** `20261008_192500_v3` | **Thời điểm cập nhật:** 2026-10-08T22:45:00+07:00  

Tài liệu này cung cấp quy trình từng bước được tham số hóa đầy đủ để bất kỳ kỹ sư QA hoặc kiểm toán viên độc lập nào cũng có thể chạy lại và xác thực 100% các kết luận, số liệu và bằng chứng kỹ thuật trong gói kiểm toán V3.

---

## 1. Yêu Cầu Môi Trường & Xác Minh Checkpoint

* **Hệ điều hành:** Windows 10/11 hoặc Linux x86_64.
* **Môi trường Python:** Python 3.12 (khuyến nghị dùng virtualenv đã thiết lập sẵn: `.\.venv\Scripts\python.exe`).
* **Thư viện cốt lõi:** `torch>=2.5.1`, `torchvision>=0.20.1`, `ultralytics>=8.4.171`, `ensemble_boxes>=1.0.9`, `pytest>=8.0.0`, `psutil`.

### Xác minh tính toàn vẹn của 2 Checkpoint mô hình trước khi chạy:
```powershell
# Kiểm tra mã băm SHA-256 của Checkpoint SSDLite320 chính thức
Get-FileHash artifacts\detection\waste-ssdlite320-mobilenetv3-combined-seed42\weights\best.pt -Algorithm SHA256
# Kết quả mong đợi: 584c5980eb4ebe3718da7b3e87a3c95edfee3d78f186def3e5feba32cf03dff1

# Kiểm tra mã băm SHA-256 của Checkpoint YOLOv8n đối chứng
Get-FileHash artifacts\detection\waste-yolov8n-combined-seed42\weights\best.pt -Algorithm SHA256
# Kết quả mong đợi: 97f46ccaaa0a63c5934225d51ccd399c1c18eb3e1af8a63b9f14bec1aaa085c9
```

---

## 2. Quy Trình Tái Hiện Từng Bước

### Bước 1: Tái hiện Trace kỹ thuật E2E & Đo đạc Tensor PyTorch Forward
Script `scripts/generate_all_traces.py` được trang bị giao diện dòng lệnh `argparse`, cơ chế PyTorch forward hook thật và kiểm tra preflight nghiêm ngặt (`raise FileNotFoundError` nếu thiếu bất kỳ tệp ảnh nào).

```powershell
# Chạy với tham số mặc định
.\.venv\Scripts\python reports\multi_waste_detection_audit\20261008_192500\scripts\generate_all_traces.py

# Hoặc tùy biến đường dẫn nếu cần:
.\.venv\Scripts\python reports\multi_waste_detection_audit\20261008_192500\scripts\generate_all_traces.py `
  --case-images-dir reports\multi_waste_detection_audit\20261008_192500\images `
  --output-dir reports\multi_waste_detection_audit\20261008_192500
```
* **Đầu ra sinh ra:**
  - 4 tệp JSON Trace chi tiết tại `traces/`:
    - `traces/case_001_trace.json`: Ảnh bãi chai nhựa/thủy tinh người dùng báo lỗi (`case_001_raw.webp`).
    - `traces/case_002_trace.json`: Ảnh rác hữu cơ thực tế (`case_002_raw.jpg`).
    - `traces/case_003_trace.json`: Ảnh rác quần áo phế thải ngoài trời (`case_003_raw.jpg`).
    - `traces/case_004_trace.json`: Ảnh đống túi rác nilon sinh hoạt ven đường đất (`case_004_raw.jpg`).
  - Bảng tổng hợp: `cases.csv`.
  - Các ảnh overlay kiểm toán tại các mức ngưỡng `conf=0.20` và `conf=0.45` được lưu tại `images/`.

---

### Bước 2: Tái hiện Thống kê Bounding Box & Dữ liệu Nguồn Mendeley Data
Script `scripts/generate_mendeley_provenance.py` duyệt qua toàn bộ 1.305 tệp nhãn synthetic, thống kê 4.095 boxes và trích xuất crop chứng thực:

```powershell
.\.venv\Scripts\python reports\multi_waste_detection_audit\20261008_192500\scripts\generate_mendeley_provenance.py
```
* **Đầu ra sinh ra:**
  - `data_provenance/synthetic_dataset_box_counts.json`: Bảng thống kê chính xác số lượng box từng lớp, xác thực 2.712 boxes bị đảo lộn trực tiếp và 3.167 boxes bị lệch khái niệm.
  - `data_provenance/mendeley_crops_provenance.json`: Metadata, tọa độ bbox và mã băm SHA-256 của 19 crops mẫu.
  - `data_provenance/samples/`: 4 ảnh và nhãn mẫu gốc từ kho lưu trữ synthetic.

---

### Bước 3: Chạy bộ Regression & Characterization Test Suite (`/test-driven-development`)
Bộ test thực thi các kiểm thử hồi quy và đặc tính hóa:
1. `test_taxonomy_pipeline_real_script`: Kiểm tra AST của `prepare_detection_dataset.py` và chạy parser trên dòng annotation thật để chứng minh lỗi đảo lộn nhãn.
2. `test_bbox_known_box_roundtrip_all_sizes`: Kiểm tra khả nghịch tọa độ trên 20 known boxes và kiểm thử đột biến (mutation test).
3. `test_wbf_singleton_penalty_math`: Chứng minh toán học hình phạt chia đôi điểm số của WBF (`avg` và `box_and_model_avg`).
4. `test_ui_default_slider_exceeds_ssdlite_confidence_range`: Kiểm chứng ngưỡng slider mặc định 0.45 dập tắt hoàn toàn dự đoán của SSDLite.

```powershell
.\.venv\Scripts\pytest reports\multi_waste_detection_audit\20261008_192500\scripts\test_pipeline_regressions.py -v --junitxml=reports\multi_waste_detection_audit\20261008_192500\metrics\junit_report.xml
```
* **Đầu ra mong đợi:**
  - `4 passed, 0 failed, 0 skipped` trong thời gian ~5-7 giây.
  - Báo cáo JUnit XML: `metrics/junit_report.xml` với `errors="0" failures="0" skipped="0" tests="4"`.
  - Nhật ký thực thi: `raw_logs/pytest_stdout.log`.

---

### Bước 4: Đo đạc Độ trễ & Bộ nhớ trên Ảnh Thực Tế (`/performance-optimization`)
Script `scripts/benchmark_latency.py` hỗ trợ tham số `--image-path`, `--device` (cpu/cuda), `--runs`, và `--output-json`. Script đo đạc trên ảnh thực tế người dùng gửi (`case_001_raw.webp`), thu thập 30 mẫu thô và ghi nhận dung lượng bộ nhớ Resident Set Size sau khi chạy (`cpu_rss_end_mb`).

```powershell
.\.venv\Scripts\python reports\multi_waste_detection_audit\20261008_192500\scripts\benchmark_latency.py --runs 30 --device cpu
```
* **Đầu ra mong đợi:**
  - Tệp số liệu: `metrics/benchmark_latency.json` (chứa 30 mẫu thô `raw_samples_ms`, `cpu_rss_end_mb`, số luồng PyTorch và mã băm ảnh).
  - Kết quả trên CPU tiêu chuẩn (8 luồng):
    - YOLOv8n: Median ~48.6 ms, P95 ~58.0 ms, FPS ~20.0, RSS ~549 MB.
    - SSDLite320: Median ~47.9 ms, P95 ~69.1 ms, FPS ~20.3, RSS ~577 MB.
    - WBF Ensemble: Median ~95.6 ms, P95 ~119.3 ms, FPS ~10.2, RSS ~619 MB.

---

### Bước 5: Kiểm tra Giao diện Streamlit bằng Chrome DevTools MCP (`/impeccable`)
1. Khởi chạy ứng dụng Streamlit:
   ```powershell
   .\.venv\Scripts\streamlit run streamlit_app.py --server.port 8501 --server.headless true
   ```
2. Mở trình duyệt đến `http://localhost:8501/`.
3. Tải lên tệp ảnh `reports/multi_waste_detection_audit/20261008_192500/images/case_001_raw.webp`.
4. Quan sát và đối chiếu với ma trận 6 ảnh chụp màn hình kiểm toán trong `images/`:
   - `browser_yolo_conf045.png`: YOLO @ conf `0.45` $\rightarrow$ 2 boxes (Quần áo 76%, Thủy tinh 54%).
   - `browser_yolo_conf020.png`: YOLO @ conf `0.20` $\rightarrow$ 4 boxes (Quần áo 76%, Thủy tinh 54%, Nhựa 34%, Thủy tinh 22%).
   - `browser_ssdlite_conf045.png`: SSDLite @ conf `0.45` $\rightarrow$ 0 box ("Không phát hiện thấy rác").
   - `browser_ssdlite_conf020.png`: SSDLite @ conf `0.20` $\rightarrow$ 7 boxes (Nhựa 38%, Nhựa 30%, Pin 28%...).
   - `browser_wbf_conf045.png`: WBF @ conf `0.45` $\rightarrow$ 0 box ("Không phát hiện thấy rác").
   - `browser_wbf_conf020.png`: WBF @ conf `0.20` $\rightarrow$ 2 boxes (Quần áo 38%, Thủy tinh 27%).

---

## 3. Phân Biệt Giữa Kiểm Tra Hash Gói Gốc và Kết Quả Benchmark

* **Mã băm SHA-256 cố định (Deterministic):**
  - Checkpoint models, ảnh gốc, mã nguồn trong `source_snapshot/` và cấu hình YAML phải khớp 100% với danh sách trong `manifest.sha256`.
* **Kết quả đo lường có biến thiên (Empirical Latency & Memory):**
  - Thời gian suy luận (Latency P50, P95, FPS) và bộ nhớ RAM phụ thuộc vào tài nguyên phần cứng CPU/GPU và tiến trình nền của máy tính tại thời điểm chạy lại. Sự sai lệch $\pm 10-15\%$ là hoàn toàn bình thường trong kỹ thuật đo lường hiệu năng thực tế.
