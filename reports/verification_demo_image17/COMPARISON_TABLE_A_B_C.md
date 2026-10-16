# BẢNG SO SÁNH THỰC NGHIỆM ĐỘC LẬP TRÊN ẢNH `images (17).jpg`
**Kích thước ảnh:** 547x365 px | **SHA-256:** `c1a77f0cbbf36c400669d7af088578803f533eaaff8f64cd7b35e68e0a71072a` | **Số vật thể GT:** 11 vật thể
**Phần cứng thực thi:** NVIDIA GeForce RTX 2050 (4.0 GB VRAM) | **Ngưỡng cài đặt:** 0.25 (25%)

## 1. Bảng đối đầu A / B / C

| Tiêu chí | Cấu hình A (SSDLite + Tiling + TwoStage) | Cấu hình B (SSDLite + Tiling, No TwoStage) | Cấu hình C (YOLOv8n + Tiling, No TwoStage) |
|---|---|---|---|
| **Checkpoint** | `E2_ssdlite320_10cls` (`c9eb52...`) | `E2_ssdlite320_10cls` (`c9eb52...`) | `E1_yolov8n_10cls` (`7e986d...`) |
| **Tiling (800)** | Bị bỏ qua (ảnh 547x365 <= 800) | Bị bỏ qua (ảnh 547x365 <= 800) | Bị bỏ qua (ảnh 547x365 <= 800) |
| **Two-Stage** | BẬT (MobileNetV3 Product) | TẮT | TẮT |
| **Số box hiển thị trên UI** | **1 box** (`plastic` 23.8%) | **1 box** (`plastic` 29.7%) | **5 boxes** (1 shoes, 3 plastic, 1 metal) |
| **Số box nếu ngưỡng phẳng 0.25** | **0 box** (bị lọc sạch) | **1 box** | **5 boxes** |
| **Thời gian xử lý** | **72.1 ms** | **430.7 ms** | **585.8 ms** |
| **Độ bao phủ rác thật (Recall trên 11 GT)** | **1 / 11** (9.1%) | **1 / 11** (9.1%) | **4 / 11** (36.4%) |
| **Box ảo (False Positive)** | 0 FP | 0 FP | 1 FP (`shoes` 65.4% do texture thùng rác) |
| **Nguyên nhân bỏ sót rác** | Raw SSDLite scores của 10 vật thể khác đều < 0.25 (chỉ 0.04-0.21). Two-stage hạ điểm từ 29.7% xuống 23.8%. | Raw SSDLite score quá yếu, không bắt được chi tiết lon nhôm, chai thủy tinh nhỏ. | Khá hơn nhiều nhưng vẫn sót rác khuất/nhỏ do độ phân giải vào 320x320. |

## 2. Chi tiết từng Bounding Box phát hiện

### Cấu hình A (SSDLite + Tiling + TwoStage - UI Thực tế)
- **[plastic]** Conf: `23.8%` (0.2385) | Box: `(393.1, 53.8, 477.9, 182.4)` -> Khớp GT chai nước khoáng nhựa trong suốt (IoU = 0.71)

*Lý do hiển thị 23.8% khi slider là 25%:*
Detector raw score là 29.71%, classifier thẩm định ra 80.28% plastic -> Tích số Bayesian = 29.71% * 80.28% = 23.85%. Giao diện ngầm nạp `configs/detection_thresholds_calibrated.json` có ngưỡng plastic = 22% nên box được giữ lại.

### Cấu hình B (SSDLite + Tiling, Không TwoStage)
- **[plastic]** Conf: `29.7%` (0.2971) | Box: `(393.1, 53.8, 477.9, 182.4)`

### Cấu hình C (YOLOv8n + Tiling, Không TwoStage)
- **[shoes]** Conf: `65.4%` (0.6543) | Box: `(91.5, 292.1, 160.5, 353.8)`
- **[plastic]** Conf: `56.2%` (0.5620) | Box: `(243.1, 17.7, 337.0, 210.0)`
- **[plastic]** Conf: `37.5%` (0.3745) | Box: `(403.5, 46.3, 493.0, 206.7)`
- **[metal]** Conf: `30.7%` (0.3074) | Box: `(351.9, 173.0, 426.5, 209.2)`
- **[plastic]** Conf: `27.7%` (0.2768) | Box: `(0.1, 105.5, 274.5, 365.0)`
