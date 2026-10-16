# KẾ HOẠCH HUẤN LUYỆN VÀ ĐÁNH GIÁ THỰC ĐỊA (TRAINING & EVALUATION PLAN)
**Quy trình Huấn luyện Kiểm soát GPU và Đánh giá Độc lập Hai Chế độ Camera**  
**Dự án:** `CNTT-KLCN155-waste-detection`  
**Giai đoạn:** Sprint 0 — Thiết lập Kế hoạch Huấn luyện Pilot & Giao thức Đánh giá Thực tế  
**Tác giả:** Antigravity Integrator (Multi-Agent Team — Chuyên trách: Agent B Model & Agent D QA)  
**Ngày lập:** 2026-10-12  

---

## 1. RÀNG BUỘC PHẦN CỨNG VÀ BẢO VỆ PREFLIGHT

### 1.1 Khảo sát phần cứng thực tế (Host Telemetry)
Kiểm tra trực tiếp tại máy trạm thông qua môi trường ảo `.venv`:
- **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4.0 GB VRAM tổng, CUDA 12.1).
- **Host RAM:** 15.31 GB Tổng. Lưu ý: RAM khả dụng biến động xung quanh $2.3\text{--}3.5\text{ GB}$.
- **Cơ chế an toàn Preflight:** Trước mọi lần nạp mô hình hoặc huấn luyện, hệ thống bắt buộc phải kích hoạt hàm `check_training_safety()` từ [preflight_guard.py](file:///d:/CNTT-KLCN155-waste-detection/src/detection/preflight_guard.py):
  - Yêu cầu sàn RAM trống: $\ge 1.200\text{ MB}$.
  - Yêu cầu sàn VRAM trống: $\ge 500\text{ MB}$.
  - Nếu không thỏa mãn, tiến trình lập tức kích hoạt lỗi `PreflightSafetyError` và dừng lại an toàn, ngăn chặn đứng máy hoặc tràn RAM hệ điều hành.

### 1.2 Nguyên tắc kiểm soát thực thi GPU
1. **Duy nhất một tác vụ nặng tại một thời điểm:** Tuyệt đối không chạy song song 2 lệnh huấn luyện hoặc 1 lệnh train + 1 lệnh eval nặng.
2. **Quyền phê duyệt của Người dùng:** Mọi lần chạy huấn luyện tốn GPU $\ge 5$ phút bắt buộc phải xin phép và được người dùng đồng ý trước khi thực thi.

---

## 2. KẾ HOẠCH HUẤN LUYỆN PILOT (PILOT TRAINING PIPELINE)

### 2.1 Lựa chọn mô hình
- **YOLOv8n:** Được chọn làm **Detector chính** cho ứng dụng thực tế nhờ tốc độ suy luận nhanh ($10\text{--}20\text{ ms}$ trên RTX 2050), kiến trúc nhẹ, hỗ trợ xuất khẩu ONNX/TensorRT dễ dàng cho camera.
- **SSDLite320-MobileNetV3:** Giữ làm **Mô hình đối chứng học thuật** và so sánh theo yêu cầu khóa luận.

### 2.2 Cấu hình huấn luyện tối ưu cho RTX 2050 4GB
Dựa trên các bài học kinh nghiệm từ cấu hình `configs/detection_training_rtx2050.yaml`:

```yaml
# configs/detection_training_pilot_v3.yaml (Dự thảo)
model: artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt  # Fine-tune từ checkpoint v2 tốt nhất
data: configs/detection_dataset_real_v3_pilot.yaml
epochs: 30                  # Pilot ngắn 20-40 epochs để đo cải thiện thực chất
imgsz: 512                  # Giảm từ 640 xuống 512: tiết kiệm ~35% VRAM, tăng tốc độ 1.8x
batch: 4                    # Thử nghiệm với batch 4; tự động backoff về batch 2 nếu VRAM chạm ngưỡng
workers: 2                  # Giới hạn 2 worker để bảo toàn RAM hệ thống (tránh OOM host)
optimizer: AdamW
lr0: 0.0005                 # Learning rate nhỏ cho fine-tuning để bảo toàn tri thức cũ
lrf: 0.01
patience: 10                # Early stopping sau 10 epochs không cải thiện val loss
amp: true                   # Automatic Mixed Precision (FP16) giảm 50% dung lượng tensor
seed: 42                    # Deterministic reproducibility
deterministic: true
save_period: 5
```

### 2.3 Cơ chế xử lý Out-Of-Memory (OOM Backoff)
- Nếu xảy ra lỗi `torch.cuda.OutOfMemoryError`:
  - Bắt lỗi ngoại lệ và dọn dẹp cache `torch.cuda.empty_cache()`.
  - Giảm batch size tuần tự: $\text{batch}=4 \to \text{batch}=2 \to \text{batch}=1$.
  - Ghi nhận thông tin thay đổi vào log huấn luyện chính thức; **không âm thầm thay đổi siêu tham số mà không ghi nhật ký**.

### 2.4 Quản lý mã băm và khả năng tái lập (Artifact Provenance)
Mọi lượt chạy huấn luyện đều phải sinh file nhật ký `train_provenance.json` ghi rõ:
1. Mã băm SHA-256 của Checkpoint khởi tạo và Checkpoint `best.pt` sau huấn luyện.
2. Mã băm SHA-256 của file cấu hình `.yaml` và các file split (`train.txt`, `val.txt`).
3. Dấu vân tay pHash của tập dữ liệu huấn luyện.
4. Lịch sử log epoch, train/val box loss, cls loss, dfl loss, mAP50, mAP50-95.

---

## 3. THIẾT LẬP HAI TẬP KIỂM THỬ ĐỘC LẬP MỚI (DUAL REAL-WORLD TEST PROTOCOL)

Bảo toàn nguyên vẹn tập nghiệm thu hiện tại (`D:\Hinh_Anh_Rac`, 27 ảnh, 57 vật thể Ground Truth đã đóng băng). Xây dựng thêm **2 tập kiểm thử thực địa chuyên biệt**:

```
+-----------------------------------------------------------------------------------+
|                        HỆ THỐNG ĐÁNH GIÁ 3 TẬP ĐỘC LẬP                            |
|                                                                                   |
|  [TẬP 1: HOLDOUT ĐÓNG BĂNG HIỆN HỮU]                                              |
|  - 27 ảnh D:\Hinh_Anh_Rac, 57 Ground Truth objects (Đã đóng băng từ Sprint 0)     |
|  - Thước đo so sánh độ tiến bộ trực tiếp với Baseline SSDLite và YOLOv8n E1       |
|                                                                                   |
|  [TẬP 2: FIXED CAMERA HOLDOUT (MỚI)]                                              |
|  - Góc máy cố định (Top-down, góc 45 độ giám sát thùng rác / bàn gom)             |
|  - Rác ở nhiều cự ly (gần, trung bình, xa), kích thước vật thể nhỏ/vừa            |
|  - Cảnh nền sạch xen kẽ cảnh có rác để đo False Positive theo thời gian            |
|  - Video liên tục >= 1 giờ để đo độ trôi dạt (drift) và độ ổn định                |
|                                                                                   |
|  [TẬP 3: MOBILE CAMERA HOLDOUT (MỚI)]                                             |
|  - Quay quét bằng camera điện thoại thông minh và webcam cầm tay                  |
|  - Chuyển động lia máy, rung tay, xoay ngang/dọc (Portrait vs Landscape)          |
|  - Thay đổi đột ngột điều kiện ánh sáng (trong phòng, ngoài trời, bóng râm)       |
|  - Cụm đa rác dày đặc, che khuất nhau từ 30% đến 70%                              |
+-----------------------------------------------------------------------------------+
```

---

## 4. BỘ CHỈ SỐ NGHIỆM THU THỰC TẾ (PRACTICAL METRICS & KPI)

Tuân thủ nguyên tắc: **Không bao giờ tuyên bố triển khai thực tế thành công chỉ vì mAP train/val tăng**. Mọi kết luận phải dựa trên bộ chỉ số đo lường thực địa:

### 4.1 Bảng chỉ số cốt lõi
1. **mAP50 và mAP50-95 (COCO Standard):** Đánh giá tổng thể độ chính xác định vị và phân loại.
2. **Class-Agnostic AP50:** Đo năng lực phát hiện vị trí vật thể thuần túy (loại bỏ yếu tố nhầm lớp).
3. **Precision / Recall / F1 tại ngưỡng vận hành 0.25 (IoU $\ge 0.5$):**
   - **Tỉ lệ thu hồi Pin (Battery Recall):** Mục tiêu $\ge 50\%$ (Baseline hiện tại: 0.00%).
   - **Tỉ lệ thu hồi Rác sinh học (Biological Recall):** Mục tiêu $\ge 50\%$ (Baseline hiện tại: 0.00%).
4. **Tỉ lệ báo động giả trên cảnh sạch (False Positive Rate - FPR):**
   - Số khung bao rác ảo trên mỗi 100 khung hình cảnh sạch (Mục tiêu: $< 3$ box/100 frames).
   - Tần suất False Positive trên 1 giờ camera cố định (Mục tiêu: $\le 5$ cảnh báo sai/giờ).
5. **Hiệu năng thời gian thực:**
   - Độ trễ trung bình (Mean Latency) và phân vị 95 (P95 Latency).
   - FPS đo được trên luồng video camera thật (Mục tiêu: $\ge 15\text{ FPS}$).
   - Mức tiêu thụ VRAM (Mục tiêu: $\le 2.5\text{ GB}$ trên RTX 2050).

### 4.2 Báo cáo phân tích lỗi trực quan (Visual Failure Catalog)
Sau mỗi lần đánh giá, hệ thống tự động xuất thư mục ảnh overlay lỗi:
- Các trường hợp **Bỏ sót (False Negatives):** Viền màu đỏ nét đứt quanh rác bị lọt lưới.
- Các trường hợp **Báo ảo (False Positives):** Viền màu vàng quanh vật thể ngoại cảnh bị nhận nhầm thành rác.
- Các trường hợp **Sai lớp (Class Confusion):** Ghi rõ nhãn Ground Truth và nhãn Mô hình dự đoán.
