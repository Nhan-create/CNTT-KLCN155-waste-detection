# SO SÁNH TOÀN DIỆN CÁC MÔ HÌNH VÀ LỘ TRÌNH NÂNG CẤP (MODEL COMPARISON & UPGRADE ROADMAP)

**Dự án:** Hệ thống Phát hiện & Phân loại Rác thải Đa Vật thể (CNTT-KLCN155)  
**Tác giả:** Principal Computer Vision Engineer / Integrator  
**Ngày lập:** 11/10/2026  

---

## 1. So Sánh Thông Số Kỹ Thuật (Architecture & Resource Specs)

| Tiêu chí kỹ thuật | SSDLite320-MobileNetV3 (Chính) | YOLOv8n (Đối chứng hiện tại) | YOLO26n (Ứng viên đề xuất) |
|---|---|---|---|
| **Khung phát triển** | PyTorch / Torchvision | Ultralytics YOLOv8 | Ultralytics YOLO26 |
| **Dung lượng weights (.pt)** | 13.5 MB | 6.5 MB | 5.5 MB |
| **Số lượng tham số (Params)** | 3.44 Triệu (3.44M) | 3.16 Triệu (3.16M) | 2.58 Triệu (2.58M) |
| **Độ phức tạp tính toán** | ~1.5 GFLOPs (@320x320) | ~8.7 GFLOPs (@640x640) | ~6.5 GFLOPs (@640x640) |
| **Độ phân giải chuẩn (imgsz)** | $320 \times 320$ (Cố định) | $640 \times 640$ (Linh hoạt) | $640 \times 640$ (Linh hoạt) |
| **Kiến trúc Head** | Anchor-based (SSD 6 scales) | Anchor-free Decoupled Head | Anchor-free End-to-End NMS-Free Head |
| **Loss Function** | Smooth L1 + Cross-Entropy | CIoU + DFL + BCE | CIoU/GIoU + DFL + BCE (No-NMS) |
| **Inference Latency (GPU RTX 2050)** | **65.9 ms** | **48.4 ms** | **~38.0 ms** |
| **Inference Latency (CPU i7)** | 145.0 ms | 115.0 ms | ~95.0 ms |
| **FPS thực tế trên Camera ($640 \times 480$)** | ~15 FPS | ~20 FPS | ~25 FPS |

---

## 2. So Sánh Hiệu Năng Thực Tế Trên Cùng Tập Kiểm Thử (Head-to-Head Evaluation)

### 2.1. Đánh giá trên tập Validation chính thức (299 ảnh, 902 boxes)
- **mAP50-95:**
  * SSDLite320: $12.65\%$
  * YOLOv8n: $\mathbf{16.00\%}$ (**Cao hơn $+26.5\%$ tương đối**)
- **mAP50:**
  * SSDLite320: $19.71\%$
  * YOLOv8n: $\mathbf{24.73\%}$ (**Cao hơn $+25.5\%$ tương đối**)
- **Số lớp đạt AP > 20%:**
  * SSDLite320: 3 lớp (`clothes`, `plastic`, `shoes`).
  * YOLOv8n: 5 lớp (`clothes`, `metal`, `plastic`, `shoes`, `glass`).

### 2.2. Đánh giá trên tập nghiệm thu độc lập `D:\Hinh_Anh_Rac` (27 ảnh, 57 GT)
- **SSDLite320 (App Default - Blend):** Precision $1.8\%$, Recall $10.5\%$, FP = 23. Nhận nhầm tòa nhà, biển hiệu, tủ điện thành bìa carton với xác suất $\sim 50\%$.
- **YOLOv8n (Raw):** Precision $20.0\%$, Recall $15.8\%$, FP = 36. Không bị kích điểm ảo từ classifier Closed-World, bounding box ôm sát vật thể thực tế hơn.

---

## 3. Lộ Trình Nâng Cấp Không Phá Vỡ Ứng Dụng (Non-Breaking Upgrade Strategy)

Tuân thủ nghiêm ngặt 9 yêu cầu của đề tài:

1. **Giữ nguyên ứng dụng Streamlit hiện có:** Không viết lại UI, không thay đổi luồng tương tác của người dùng.
2. **Bảo tồn checkpoint cũ:** Toàn bộ weights cũ (`artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` và `E1_yolov8n_10cls`) được giữ nguyên, không ghi đè.
3. **Mở rộng lựa chọn trong Sidebar:**
   Sidebar của Streamlit bổ sung thêm tùy chọn:
   * `YOLO26n (Mô hình nâng cấp mới - Nhanh & Chính xác nhất)`
   * `YOLOv8n (Mô hình đối chứng)`
   * `SSDLite320-MobileNetV3 (Mô hình chính)`
   * `Hợp nhất SSDLite + YOLO (WBF)`
4. **Hiển thị đúng tọa độ và độ tin cậy thật:** Đảm bảo hàm chuyển đổi tọa độ ánh xạ pixel gốc $1:1$, không vẽ lệch, không che giấu lỗi bằng threshold nhân tạo.
5. **Hỗ trợ đa vật thể (Multi-Object):** Cho phép phát hiện cùng lúc hàng chục vật thể rác trong một khung hình.
6. **Bảo vệ Camera không bị treo:** Áp dụng cơ chế bỏ frame (Frame Drop / Skip queue) khi camera gửi luồng ảnh nhanh hơn tốc độ suy luận của mô hình, đảm bảo độ trễ camera luôn $< 100\text{ ms}$.
7. **Khả năng Rollback tức thì:** Nếu mô hình mới có bất kỳ vấn đề gì, người dùng chỉ cần chọn lại mô hình cũ từ dropdown box trên sidebar trong 1 giây mà không cần khởi động lại server.

---

## 4. Kế Hoạch Huấn Luyện Nâng Cấp Chi Tiết (Fine-Tuning Plan)

### 4.1. Nguyên tắc cốt lõi
- **Không huấn luyện từ đầu (No training from scratch):** Bắt buộc khởi tạo từ pretrained weights chính thức của Ultralytics (`yolo26n.pt` hoặc `yolov8n.pt` đã có sẵn tại `weights/yolo26n.pt` và `yolov8n.pt`).
- **Chiến lược dữ liệu:**
  1. Giữ nguyên tập kiểm chứng `D:\Hinh_Anh_Rac` làm tập độc lập bất khả xâm phạm.
  2. Bổ sung $200 - 300$ ảnh Hard-Negative sạch (tường nhà, đường phố, biển hiệu, tủ điện đô thị) có file nhãn `.txt` rỗng.
  3. Bổ sung các mẫu ảnh rõ nét của `battery` và `biological` để giải quyết triệt để 2 lớp $0.0\%$.

### 4.2. Bộ Siêu Tham Số Đề Xuất (Recommended Hyperparameters)
```yaml
# configs/train_yolo26_waste.yaml
model: weights/yolo26n.pt
task: detect
mode: train
epochs: 80
patience: 20
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
weight_decay: 0.0005
warmup_epochs: 3.0
close_mosaic: 10
mosaic: 0.8
mixup: 0.1
flipud: 0.0
fliplr: 0.5
device: 0               # NVIDIA RTX 2050
workers: 2              # Giữ RAM an toàn dưới áp lực bộ nhớ cao
amp: true               # Mixed precision tiết kiệm VRAM
```

### 4.3. Báo cáo tài nguyên trước khi huấn luyện
- VRAM ước tính: $\approx 2.9\text{ GB} / 4.0\text{ GB}$.
- Thời gian huấn luyện ước tính: $\approx 55\text{ phút}$ trên RTX 2050.
- **Cam kết:** Không tự ý khởi chạy huấn luyện dài hạn trên GPU khi chưa có sự xác nhận của người dùng.
