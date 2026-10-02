# KẾ HOẠCH MA TRẬN THỰC NGHIỆM CHÍNH THỨC (PHASE 2: OBJECT DETECTION)
## DỰ ÁN CNTT-KLCN155: WASTE DETECTION & RECYCLING CLASSIFICATION

- **Cơ quan:** Trường Đại học Công Thương TP. Hồ Chí Minh (HUIT) - Khoa Công nghệ Thông tin
- **Người thực hiện:** Sinh viên Ngô Thanh Nhân (PM kiêm Lead)
- **Kiến trúc sư kiểm tra:** Tech Lead & ML Engineer
- **Phiên bản:** v1.1-verified
- **Mục tiêu:** Xây dựng ma trận thực nghiệm chuẩn tắc, đảm bảo tính công bằng khoa học, khả năng tái hiện và bám sát 100% đề cương khóa luận.

---

## 1. THIẾT KẾ MA TRẬN THỰC NGHIỆM 2 × 4

Ma trận bao gồm **2 cấu trúc mô hình** kết hợp với **4 chiến lược tăng cường dữ liệu**, tạo thành **8 lượt chạy thực nghiệm độc lập**:

| Mã thí nghiệm (Exp ID) | Mô hình (Detector) | Vai trò trong nghiên cứu | Chiến lược Tăng cường (Augmentation) | Chi tiết kỹ thuật tăng cường |
|:---:|:---|:---|:---|:---|
| **EXP-SSD-01** | SSDLite320-MobileNetV3 | Mô hình chính (Primary) | Strategy 1: None (Baseline) | Không tăng cường, chỉ chuẩn hóa ảnh RGB tensor [0, 1]. |
| **EXP-SSD-02** | SSDLite320-MobileNetV3 | Mô hình chính (Primary) | Strategy 2: Geometric | HorizontalFlip, ShiftScaleRotate (Affine) đồng bộ Bbox và nhãn. |
| **EXP-SSD-03** | SSDLite320-MobileNetV3 | Mô hình chính (Primary) | Strategy 3: Photometric | ColorJitter (brightness, contrast, saturation, hue), Gaussian/Motion Blur. |
| **EXP-SSD-04** | SSDLite320-MobileNetV3 | Mô hình chính (Primary) | Strategy 4: Combined | Phối hợp Geometric + Photometric + Simple Copy-Paste (Ghiasi et al., 2021) từ DonorBank tập Train. |
| **EXP-YOLO-01** | YOLOv8n | Mô hình đối chứng (Benchmark) | Strategy 1: None (Baseline) | Không tăng cường, tắt toàn bộ mosaic, mixup, autoaugment mặc định. |
| **EXP-YOLO-02** | YOLOv8n | Mô hình đối chứng (Benchmark) | Strategy 2: Geometric | HorizontalFlip, ShiftScaleRotate qua pipeline Albumentations đồng nhất. |
| **EXP-YOLO-03** | YOLOv8n | Mô hình đối chứng (Benchmark) | Strategy 3: Photometric | ColorJitter và Blur quang học mô phỏng camera thực tế. |
| **EXP-YOLO-04** | YOLOv8n | Mô hình đối chứng (Benchmark) | Strategy 4: Combined | Phối hợp Geometric + Photometric + Simple Copy-Paste từ DonorBank tập Train. |

> [!NOTE]
> **Về kỹ thuật Simple Copy-Paste [5] (Ghiasi et al., 2021):**
> Nhánh Copy-Paste đã được tích hợp thành công bằng cách khai thác 185 polygon segmentation masks chuẩn COCO từ 12 ảnh TACO thuộc tập `train`. Bộ nhớ `DonorBank` chứa 68 đối tượng foreground có mask phân đoạn chính xác. Quá trình dán đối tượng có cơ chế xử lý che khuất (Occlusion Handling): loại bỏ các box nền cũ nếu bị che khuất $\ge 80\%$ diện tích. Toàn bộ donor và background được cách ly 100% trong tập `train`, tuyệt đối không dùng mẫu từ `val` hoặc `test`.

---

## 2. NGUYÊN TẮC SO SÁNH CÔNG BẰNG (FAIR COMPARISON PROTOCOL)

Để đảm bảo kết luận khoa học có giá trị học thuật cao, hai mô hình được đặt trong điều kiện đánh giá nghiêm ngặt và đồng nhất:

1. **Đồng nhất dữ liệu và phân chia tập chống rò rỉ (Zero-Leakage Split):**
   - Cả hai mô hình sử dụng chung tệp chỉ mục dữ liệu đã kiểm toán: [`configs/detection_dataset.yaml`](file:///D:/CNTT-KLCN155-waste-detection/configs/detection_dataset.yaml) và [`data/detection/manifest.jsonl`](file:///D:/CNTT-KLCN155-waste-detection/data/detection/manifest.jsonl).
   - Tỷ lệ dữ liệu thật trong Validation và Test là 100% (không dùng ảnh synthetic trong tập đánh giá).
2. **Cố định hạt giống ngẫu nhiên (Reproducibility Seed):**
   - Thiết lập `seed = 42` trên toàn bộ hệ thống (PyTorch, NumPy, Python standard random, CUDA deterministic ops).
3. **Ngân sách huấn luyện và dừng sớm (Training Budget):**
   - Số epoch tối đa: **120 epochs**.
   - Cơ chế dừng sớm (Early Stopping): **Patience = 30 epochs** dựa trên chỉ số Validation mAP@0.5:0.95.
   - Batch size: 8 hoặc 16 (tùy thuộc giới hạn 4GB VRAM của card RTX 2050).
   - **Lưu ý học thuật:** Không khẳng định chắc chắn mô hình sẽ đạt chất lượng cao chỉ vì huấn luyện 120 epochs; chất lượng thực tế phải được đo đạc khách quan và chứng minh bằng số liệu mAP, loss hội tụ, và ma trận nhầm lẫn.
4. **Tiêu chí lựa chọn Checkpoint (Checkpoint Selection Rule):**
   - Checkpoint tối ưu (`best.pt`) được chọn duy nhất dựa trên giá trị **mAP@0.5:0.95 cao nhất trên tập VALIDATION**.
   - **CẤM KỴ:** Tuyệt đối không dùng tập Test để chọn checkpoint, chọn augmentation, chọn threshold hay chọn siêu tham số WBF.

---

## 3. QUY TRÌNH HỢP NHẤT VÀ TINH CHỈNH WBF (WBF ENSEMBLE PROTOCOL)

1. **Khởi tạo cặp mô hình hợp nhất:**
   - Lấy checkpoint `best.pt` của SSDLite320 (chạy tốt nhất trong các nhánh tăng cường) và checkpoint `best.pt` của YOLOv8n (chạy tốt nhất).
2. **Quy tắc tinh chỉnh siêu tham số WBF (Hyperparameter Tuning):**
   - Thuật toán Weighted Boxes Fusion (Solovyev et al.) có 3 siêu tham số quan trọng:
     + Trọng số tương quan mô hình: $(w_{ssd}, w_{yolo}) \in \{(1, 1), (1, 2), (2, 1), (1, 1.5), (1.5, 1)\}$.
     + Ngưỡng IoU hợp nhất: $\text{iou\_thr} \in [0.45, 0.65]$ (bước nhảy $0.05$).
     + Ngưỡng lọc confidence ban đầu: $\text{skip\_box\_thr} \in [0.01, 0.10]$.
   - **NGUYÊN TẮC BẤT DI BẤT DỊCH:** **Toàn bộ quá trình quét lưới (Grid Search) siêu tham số WBF PHẢI ĐƯỢC THỰC HIỆN TRÊN TẬP VALIDATION** (`selected_on == 'val'`).
   - Kết quả tinh chỉnh được niêm phong vào tệp JSON đóng băng cấu hình (ví dụ: `artifacts/detection/wbf_frozen_selection.json`).

---

## 4. GIAO THỨC ĐÁNH GIÁ VÀ BÁO CÁO KẾT QUẢ CUỐI CÙNG (FINAL EVALUATION)

Sau khi hoàn thành huấn luyện và đóng băng tham số, tiến hành đánh giá một lần duy nhất trên **TẬP TEST (Blind Test Set)** bằng module [`src/detection/evaluate.py`](file:///D:/CNTT-KLCN155-waste-detection/src/detection/evaluate.py):

### A. Độ chính xác nhận diện (Accuracy Metrics)
- **Chuẩn giao thức quốc tế COCO:** Báo cáo AP từng lớp, **mAP@0.5** và **mAP@0.5:0.95** thông qua thư viện chuẩn `pycocotools.COCOeval`.
- **Chỉ số tại điểm vận hành (Operating Thresholds):** Báo cáo Precision, Recall, F1-score tổng thể và chi tiết cho từng lớp tại ngưỡng confidence công bố.

### B. Đánh giá trên các điều kiện thực tế thách thức (Subsets Evaluation)
Phân tích độ bền vững của mô hình trên các điều kiện môi trường chụp thực tế:
1. Thiếu sáng / Ánh sáng yếu ban đêm.
2. Ngược sáng / Chói sáng ngoài trời.
3. Bối cảnh nền phức tạp (lẫn cát, cỏ, sỏi, gạch đá).
4. Vật thể kích thước nhỏ ($area < 32^2\text{ pixels}$).
5. Vật thể bị che khuất một phần hoặc chồng lấn lên nhau.

### C. Đo đạc hiệu năng và tài nguyên phần cứng (System Benchmark)
Đo đạc trên cùng một hệ thống máy tính trang bị GPU NVIDIA GeForce RTX 2050:
- **Độ trễ suy luận (Inference Latency):** Đo đạc với Batch size = 1 (thực hiện warmup 5 ảnh đầu, tính trung vị p50 và phân vị p95 tính bằng mili-giây ms).
- **Tốc độ xử lý (Frames Per Second - FPS):** Đo đạc từ khâu ảnh đầu vào PIL đến sau khi hậu xử lý NMS/WBF.
- **Kích thước mô hình trên đĩa:** Kích thước tệp trọng số (Megabytes).
- **Tài nguyên tiêu thụ:** Dung lượng bộ nhớ đồ họa VRAM đỉnh (Peak CUDA Memory MB) và RAM CPU.

---

## 5. ĐIỀU KIỆN TIÊN QUYẾT TRƯỚC KHI CHẠY TOÀN BỘ 8 THÍ NGHIỆM

Để tránh lãng phí thời gian tính toán và đảm bảo kết quả có giá trị khoa học, 8 thí nghiệm chính thức CHỈ ĐƯỢC BẤM NÚT CHẠY khi thỏa mãn đầy đủ các điều kiện tiên quyết sau:
1. **Dữ liệu thật được hoàn thiện:** Thẩm định xong 235 ảnh ngoài split (xử lý dứt điểm các ảnh UNREVIEWED và NEEDS_RELABEL), và tổ chức thu thập bổ sung ảnh thực địa TP.HCM cho 3 lớp còn thiếu (`battery`, `clothes`, `shoes`).
2. **Data Loader & Augmentation được kiểm chứng:** Đã tích hợp và kiểm chứng qua backward pass với loss hữu hạn trên cả 2 kiến trúc (đã HOÀN THÀNH tại commit hiện tại).
3. **Đóng băng cấu hình và seed:** Toàn bộ configs huấn luyện cho 8 runs được đóng băng và lưu trữ có mã băm SHA-256.

| Thứ tự ưu tiên | Hạng mục công việc | Trạng thái hiện tại | Phụ thuộc |
|:---:|:---|:---:|:---|
| 1 | Pipeline Augmentation 4 chiến lược + Copy-Paste | **PASS** | Đã kiểm chứng xong |
| 2 | SSDLite & YOLO Data Loader backward verification | **PASS** | Đã kiểm chứng xong |
| 3 | Khảo sát & trích xuất 68 polygon masks train TACO | **PASS** | Đã kiểm chứng xong |
| 4 | Thẩm định 235 ảnh real ngoài split qua Review Tool | **PENDING** | Chờ PM phê duyệt phân loại |
| 5 | Thu thập ảnh thực địa TP.HCM cho 3 lớp thiếu | **PENDING** | Chờ đợt chụp thực địa Tuần 6–7 |
| 6 | Khởi chạy toàn bộ ma trận 8 thí nghiệm (120 epochs) | **READY ONCE DATA CLOSED** | Phụ thuộc mục 4 & 5 |
