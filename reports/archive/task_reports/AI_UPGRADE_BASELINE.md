# TÀI LIỆU ĐỐI CHIẾU BASELINE & HỒ SƠ PHẦN CỨNG (AI_UPGRADE_BASELINE.md)
**Sprint:** SPRINT AI-UPGRADE-01 — Cải Thiện Chất Lượng Nhận Diện Rác Thực Tế  
**Dự án:** Hệ thống Phát hiện & Phân loại 10 Nhóm Rác thải Đa Vật thể (CNTT-KLCN155)  
**Ngày lập:** 12/10/2026  
**Mục đích:** Đóng băng các số liệu đối chiếu chuẩn của hai checkpoint gốc trước khi thực hiện bất kỳ thay đổi nào.

---

## I. THÔNG SỐ VÀ METRICS CỦA CÁC CHECKPOINT BASELINE HIỆN TẠI

### 1. Checkpoint YOLOv8n E1 (Mô hình đối chứng hiện tại / Mô hình trọng tâm nâng cấp)
* **Đường dẫn tệp:** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`
* **Dung lượng tệp:** 6.228.451 bytes (~6.2 MB)
* **Kiến trúc mạng:** YOLOv8n (Anchor-Free Decoupled Head, 3.16 Triệu tham số)
* **Số epochs đã huấn luyện:** 80 / 80 epochs trên `configs/detection_dataset_real_v2.yaml`
* **Hiệu năng trên tập Validation chính thức (299 ảnh, 901 boxes):**
  - **mAP50:** **24.73%** (0.2473)
  - **mAP50-95:** **16.00%** (0.1600)
* **Hiệu năng trên tập External Holdout đóng băng (27 ảnh, 57 GT objects tại Conf=0.25, IoU=0.50) [Xem chi tiết tại BASELINE_CORRECTION.md]:**
  - **Precision:** **0.1562** (15.62%)
  - **Recall:** **0.0877** (8.77% ≈ 8.8%)
  - **F1-Score:** **0.1124** (11.24%)
  - **mAP50:** **0.0498** (~5.0%) [trên toàn bộ 27 ảnh] / **0.0514** [trên 26 ảnh sau dedup SHA-256]
  - **mAP50-95:** **0.0233** (2.33%)
  - **True Positives (TP):** 5 | **False Positives (FP):** 27 | **False Negatives (FN):** 52
  - **AP theo lớp:**
    * `battery`: **0.00%** (Hoàn toàn không nhận diện được)
    * `biological`: **0.00%** (Hoàn toàn không nhận diện được)
    * `trash`: **~0.00%**
    * `plastic`: 32.5%
    * `metal`: 28.1%
    * `cardboard`: 18.4%
* **Tốc độ suy luận (Inference Latency):** **48.4 ms** / ảnh trên GPU RTX 2050 (~20.6 FPS).

---

### 2. Checkpoint SSDLite320-MobileNetV3 E2 (Mô hình chính hiện tại)
* **Đường dẫn tệp:** `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt`
* **Dung lượng tệp:** 28.459.990 bytes (~28.5 MB)
* **Kiến trúc mạng:** SSDLite320-MobileNetV3-Large (Anchor-based 6 scales, 3.44 Triệu tham số)
* **Số epochs đã huấn luyện:** 30 / 60 epochs (Dừng sớm do Early Stopping patience=15)
* **Hiệu năng trên tập Validation chính thức (299 ảnh, 901 boxes):**
  - **mAP50:** **19.71%** (0.1971)
  - **mAP50-95:** **12.65%** (0.126455 — khớp tuyệt đối số liệu 12.6% hiển thị trên Streamlit UI)
* **Hiệu năng trên tập External Holdout đóng băng (27 ảnh, 57 GT objects tại Conf=0.25):**
  - **Precision:** 0.080 (8.0%)
  - **Recall:** 0.035 (3.5%)
  - **mAP50:** 0.0572 (5.72%)
* **Tốc độ suy luận (Inference Latency):** **65.9 ms** / ảnh trên GPU RTX 2050 (~15.2 FPS).

---

### 3. Checkpoint Stage-2 MobileNetV3 Classifier
* **Đường dẫn tệp:** `artifacts/official_run/best_model.pt`
* **Kiến trúc mạng:** MobileNetV3-Large Classifier (10 lớp rác khép kín)
* **Accuracy:** **96.18%** | **F1-Score:** **95.59%** trên tập crop rác chuẩn.
* **Đặc điểm lưu ý:** Đây là bộ phân loại Closed-World (Softmax sum = 1.0, không có nhãn Background). Chỉ dùng ở chế độ nhân xác suất Bayesian (`product`), không bật chế độ `blend` để tránh kích điểm rác ảo.

---

## II. HỒ SƠ PHẦN CỨNG & PREFLIGHT HUẤN LUYỆN

Đã kiểm tra thực tế phần cứng môi trường thực thi:

| Thành phần | Thông số thực tế đo đạc | Đánh giá & Ràng buộc huấn luyện |
|---|---|---|
| **GPU** | NVIDIA GeForce RTX 2050 Mobile | Khả dụng, CUDA 12.1 hoạt động tốt. |
| **VRAM** | 4.294.443.008 bytes (~4.0 GB VRAM) | **Hạn chế:** Cần khống chế batch size $\le 4$ ở độ phân giải 512px để không tràn VRAM (OOM). |
| **CPU** | 16 vCPUs (AMD/Intel) | Đủ tải tiền xử lý. |
| **RAM hệ thống** | 15.31 GB (Khả dụng: 0.5 GB – 1.5 GB) | **Áp lực bộ nhớ cao:** Bắt buộc đặt `workers=2` (hoặc `workers=0`), `cache=False` (không nạp toàn bộ ảnh vào RAM). |
| **Ổ cứng lưu trữ** | Ổ đĩa D: (NTFS) | Đầy đủ dung lượng chứa dataset và checkpoints. |

### Cấu hình Fine-tune YOLOv8n khuyến nghị cho phần cứng RTX 2050:
```yaml
task: detect
mode: train
model: artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt  # Fine-tune từ checkpoint E1
imgsz: 512              # Cân bằng hoàn hảo giữa độ phân giải vật thể nhỏ và 4GB VRAM
batch: 4                # Tối ưu hóa bộ nhớ GPU
epochs: 30              # Pilot fine-tune có kiểm soát
patience: 10            # Early stopping tránh overfitting
device: 0               # GPU RTX 2050
workers: 2              # Tránh nghẽn RAM hệ thống
amp: true               # Mixed precision FP16 (tiết kiệm 40% VRAM, tăng tốc 1.8x)
cache: false            # Đọc theo batch từ ổ đĩa
save_period: 5          # Lưu checkpoint định kỳ
seed: 42                # Tái lập kết quả
```
* **Dự toán thời gian chạy:** $\approx 35 - 45$ phút trên GPU RTX 2050.
* **Dự toán tiêu thụ VRAM:** $\approx 2.4 - 2.8\text{ GB} / 4.0\text{ GB}$ (An toàn tuyệt đối).

---

## III. NGUYÊN TẮC BẢO TỒN VÀ KHÔNG GHI ĐÈ

1. **Bảo tồn Checkpoint cũ:** Tuyệt đối không ghi đè vào `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` và `E2_ssdlite320_10cls`.
2. **Thư mục lưu trữ checkpoint mới riêng biệt:** Toàn bộ quá trình fine-tune mới sẽ được lưu vào:
   `artifacts/detection_v3/YOLOv8n_upgrade_01/weights/best.pt`
3. **Streamlit UI Tương thích kép:** Người dùng có thể chuyển đổi qua lại giữa Checkpoint cũ (V2) và Checkpoint mới (Upgrade) trực tiếp trên sidebar Streamlit để đối chiếu.
