# Multi-Object Waste Detection: End-to-End Implementation and Acceptance Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development with Antigravity Integrator coordinating Agents A, B, C, D. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai, khắc phục toàn diện lỗi mô hình và đường ống nhận diện đa loại rác thải, kiểm định độc lập trên toàn bộ 27 ảnh `data/holdout`, đạt 100% tiêu chí DoD bắt buộc.

**Architecture:** Kiến trúc 2 giai đoạn: Giai đoạn 1 MobileNetV3 trích xuất đặc trưng; Giai đoạn 2 SSDLite320-MobileNetV3 (mô hình chính) và YOLOv8n (đối chứng), tiền xử lý chuẩn hóa tọa độ ảnh gốc, hậu xử lý NMS và tích hợp giao diện Streamlit.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.3.163, OpenCV, Albumentations, Streamlit, Pytest.

**Spec:** `~/Downloads\PROMPT_ANTI_HOAN_TAT_WASTE_DETECTION.md`

## Global Constraints
- SSDLite320-MobileNetV3 là mô hình chính. YOLOv8n là mô hình đối chứng; đánh giá riêng.
- Thứ tự 10 lớp cố định: `[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`.
- SSDLite: Background = 0, Foreground = 1..10. Target load cộng 1, decode trừ 1. YOLO dùng 0..9.
- Không git reset --hard, không xóa dữ liệu thô, giữ nguyên dirty changes của người dùng.
- Mọi kết quả phải từ thực thi thật; không hardcode nhãn/tọa độ theo filename hay checksum.
- GPU chỉ chạy một job train/eval nặng tại một thời điểm.

## Review Focus
1. Bounding box transform lệch do EXIF orientation hoặc letterbox scale/offset sai.
2. SSDLite confidence score phân bổ thấp khiến NMS/filter threshold 0.40 lọc sạch vật thể thật.
3. Nhãn bị gán sai do tập synthetic Mendeley bị nhiễu (khẩu trang y tế gán carton, bao bì thực phẩm gán thủy tinh).
4. Khung bao bị rỗng hoặc vi phạm giới hạn ảnh gốc: $x_1 < 0$ hoặc $x_2 > W$ hoặc $y_2 > H$.
5. Streamlit UI fallback âm thầm sang checkpoint hoặc backend khác mà không báo lỗi.

---

### Task 1: Hoàn tất các điểm còn thiếu của Task 1 (Owner: Agent A & Integrator)
**Files:**
- Modify: `tests/detection/test_real_mapping_integration.py`
- Modify: `reports/reproduce_scripts/generate_verified_report.py`
- Modify: `reports/reproduce_scripts/audit_mendeley_mapping.py`
- Create/Update: `reports/BAO_CAO_XAC_MINH_MAPPING_10_LOP_R2.md`

- [ ] **Step 1.1:** Cập nhật `test_real_source_converter_independent_verification` đọc metadata thật từ sample disk độc lập, không đọc từ `configs/detection_source_mapping.yaml`.
- [ ] **Step 1.2:** Đảm bảo xuất đầy đủ bộ đếm unique/matched/unmatched/duplicate counters và source-target pairs đúng đơn vị.
- [ ] **Step 1.3:** Đánh dấu/quarantine các mẫu Mendeley mixed/unknown (Class 2, 4, v.v.), không remap nguyên lớp theo text abstract.
- [ ] **Step 1.4:** Cập nhật `generate_verified_report.py` đọc CSV/JSON/JUnit/train logs thật để lấy warning từ CSV (taco_0804 L6), phân biệt ID7 toàn nguồn với matched subset, best epoch với completed epochs/budget.
- [ ] **Step 1.5:** Chạy kiểm thử pytest xác nhận suite passed.
Lệnh: `.\.venv\Scripts\python.exe -m pytest tests/detection/test_real_mapping_integration.py -v`

---

### Task 2: Kiểm kê và Xây dựng Ground Truth Độc lập tập `data/holdout` (Owner: Agent D)
**Files:**
- Create: `reports/qa_acceptance/manifest_hinh_anh_rac.json`
- Create: `reports/qa_acceptance/ground_truth_hinh_anh_rac.json`
- Create: `reports/qa_acceptance/inventory_hinh_anh_rac.py`

- [ ] **Step 2.1:** Quét đệ quy toàn bộ 27 ảnh trong `data/holdout`, tính SHA-256, kích thước WxH, kiểm tra decode PIL/OpenCV, ghi nhận trạng thái.
- [ ] **Step 2.2:** Quan sát từng ảnh ở độ phân giải gốc, xác định tất cả instance rác có trong ảnh, gán `object_id`, `class_name`, `class_id` (0..9) và bounding box chuẩn `[x1, y1, x2, y2]` theo tọa độ pixel ảnh gốc.
- [ ] **Step 2.3:** Ghi nhận rõ căn cứ review (reviewer: Agent D, reasoning, visibility, occlusion). Không bỏ sót vật thể nhỏ, sát mép.
- [ ] **Step 2.4:** Đóng băng (Freeze) manifest và ground truth tại `reports/qa_acceptance/ground_truth_hinh_anh_rac.json`.

---

### Task 3: Kiểm định và Sửa lỗi Tọa độ, Letterbox, EXIF và Rendering (Owner: Agent C)
**Files:**
- Modify: `src/detection/visualization.py`
- Modify: `src/web/detection_logic.py`
- Test: `tests/detection/test_visualization.py`
- Test: `tests/web/test_detection_image_outputs.py`

- [ ] **Step 3.1:** Kiểm tra toàn bộ luồng xử lý ảnh từ ảnh gốc $\rightarrow$ PIL $\rightarrow$ EXIF transpose $\rightarrow$ resize/letterbox $\rightarrow$ tensor $\rightarrow$ raw box $\rightarrow$ inverse transform về pixel ảnh gốc.
- [ ] **Step 3.2:** Khắc phục lỗi clipping $0 \le x_1 < x_2 \le W$ và $0 \le y_1 < y_2 \le H$, loại bỏ box vô hiệu (rỗng, NaN, Inf).
- [ ] **Step 3.3:** Kiểm tra vẽ bounding box và nhãn tiếng Việt trên ảnh gốc, đảm bảo khung bao sát vật thể, màu sắc chuẩn xác theo 10 lớp.
- [ ] **Step 3.4:** Chạy kiểm thử tọa độ và visualization.
Lệnh: `.\.venv\Scripts\python.exe -m pytest tests/detection/test_visualization.py tests/web/test_detection_image_outputs.py -v`

---

### Task 4: Chạy Đánh giá Baseline trên tập `data/holdout` (Owner: Agent B & Agent D)
**Files:**
- Create: `reports/qa_acceptance/run_baseline_evaluation.py`
- Create: `reports/qa_acceptance/baseline_results_ssdlite.json`
- Create: `reports/qa_acceptance/baseline_results_yolo.json`

- [ ] **Step 4.1:** Nạp checkpoint thật SSDLite320 và YOLOv8n qua mã nguồn sản phẩm (`SSDLiteWasteDetector`, `WasteDetector`).
- [ ] **Step 4.2:** Thực hiện suy luận trên toàn bộ 27 ảnh, lưu raw predictions và filtered predictions.
- [ ] **Step 4.3:** Thực hiện thuật toán ghép 1-1 giữa Prediction và Ground Truth theo chuẩn IoU (Hungarian matching hoặc Greedy IoU).
- [ ] **Step 4.4:** Tính toán chi tiết: IoU, intersection/GT coverage, TP, FP, FN, wrong-class cho từng ảnh và từng vật thể.
- [ ] **Step 4.5:** Xuất bảng phân tích lỗi chi tiết (FN, FP, wrong-class, bbox lệch) và kiểm tra đối chiếu với DoD.

---

### Task 5: Tinh chỉnh / Huấn luyện Lại Mô hình nếu Baseline chưa đạt DoD (Owner: Agent B, hỗ trợ bởi Agent A)
**Files:**
- Modify: `src/detection/ssdlite_train.py`
- Modify: `configs/detection_dataset.yaml`
- Modify: `scripts/run_experiment_matrix.py`

- [ ] **Step 5.1:** Phân tích nguyên nhân lỗi từ baseline: mô hình không phát hiện (low recall/confidence) hay do dữ liệu train bị nhiễu.
- [ ] **Step 5.2:** Sử dụng tập dữ liệu đã cách ly mẫu lỗi (quarantined noisy synthetic) từ Agent A, bổ sung augmentation phù hợp.
- [ ] **Step 5.3:** Tiến hành train/fine-tune SSDLite320-MobileNetV3 (và YOLOv8n nếu cần), giám sát GPU, lưu logs và checkpoints.
- [ ] **Step 5.4:** Chọn best checkpoint theo validation metrics, ghi nhận SHA-256 và provenance rõ ràng.
- [ ] **Step 5.5:** Chạy lại đánh giá nghiệm thu trên tập `data/holdout`.

---

### Task 6: Tích hợp và Kiểm thử Giao diện Streamlit (Owner: Agent C & Integrator)
**Files:**
- Modify: `streamlit_app.py`
- Modify: `src/web/detection_logic.py`
- Modify: `src/web/detection_settings.py`
- Test: `tests/web/test_streamlit_contract.py`

- [ ] **Step 6.1:** Kiểm tra giao diện Streamlit kết nối đúng checkpoint và runtime cấu hình.
- [ ] **Step 6.2:** Kiểm tra hiển thị ảnh test, bounding box, nhãn tiếng Việt, bảng thống kê số lượng và chỉ dẫn thùng rác.
- [ ] **Step 6.3:** Chụp screenshot bằng chứng hoạt động thực tế của app.

---

### Task 7: Nghiệm thu Toàn diện DoD và Đóng gói Bàn giao (Owner: Agent D & Integrator)
**Files:**
- Create: `reports/final_acceptance/final_acceptance_report.md`
- Create: `reports/final_acceptance/case_results_all_objects.json`
- Create: `reports/final_acceptance/overlays/` (toàn bộ 27 ảnh overlay)
- Create: ZIP package trong thư mục `Downloads` của người dùng

- [ ] **Step 7.1:** Đối chiếu từng tiêu chí trong bảng DoD: Coverage 100%, Đủ vật thể (FN=0), Đúng nhãn (wrong-class=0), Không rác giả (FP=0), Khung toàn vẹn (IoU $\ge$ 0.75, Inter/GT $\ge$ 0.99), Khung hợp lệ, Kết quả sản phẩm, Bằng chứng đầy đủ.
- [ ] **Step 7.2:** Tạo báo cáo tổng hợp `reports/final_acceptance/final_acceptance_report.md` tự động từ artifacts máy sinh.
- [ ] **Step 7.3:** Lưu toàn bộ ảnh overlay có vẽ box GT và Prediction.
- [ ] **Step 7.4:** Đóng gói ZIP bàn giao có run_id vào `~/Downloads\`, tính và lưu mã băm SHA-256.
