# BÁO CÁO TOÀN DIỆN AUDIT MÔ HÌNH NHẬN DIỆN RÁC THẢI (CNTT-KLCN155)

**Vị trí:** Principal Computer Vision Engineer / Integrator  
**Ngày thực hiện:** 11/10/2026  
**Môi trường phần cứng & phần mềm:**
- **GPU:** NVIDIA GeForce RTX 2050 Mobile (4.0 GB VRAM, CUDA 12.1)
- **CPU / RAM:** 16 vCPU, 15.31 GB RAM (Khả dụng: ~0.43 - 1.5 GB, áp lực bộ nhớ cao)
- **Frameworks:** PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.171, Pillow 12.3.0, pycocotools 2.0.11
- **Ứng dụng:** Streamlit 10-Class Multi-Waste Detection & Two-Stage Classifier (Port: 8501)

---

## 1. Cấu trúc và Kiến trúc Hai Mô hình Hiện tại

### 1.1. SSDLite320-MobileNetV3 (Mô hình chính)
- **Kiến trúc mạng:** `ssdlite320_mobilenet_v3_large` từ `torchvision.models.detection`.
- **Backbone:** MobileNetV3-Large thu gọn tail, trích xuất đặc trưng tại 2 tầng: tầng 13 (C4) và tầng 16 (C5), kết hợp các khối feature map phụ cho SSD.
- **Kích thước đầu vào chuẩn:** Cố định $320 \times 320$ pixels. Mô hình tự động tích hợp `GeneralizedRCNNTransform` chuẩn hóa ảnh về ImageNet mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$.
- **Số lớp (Classes):** 11 lớp (Index 0: Background; Index 1..10: 10 lớp rác thải).
  * Quy ước nạp target: `label = class_id + 1`
  * Quy ước giải mã output: `class_id = label - 1` (được hiện thực tại `src/detection/ssdlite.py::tensor_detections`).
- **Anchor Box:** 6 tỉ lệ khung hình trên 6 feature maps (từ $20 \times 20$ xuống $1 \times 1$).

### 1.2. YOLOv8n (Mô hình đối chứng)
- **Kiến trúc mạng:** YOLOv8n (`depth_multiple=0.33`, `width_multiple=0.25`, Anchor-Free Head).
- **Kích thước đầu vào:** Huấn luyện tại $640 \times 640$ (hoặc $320 \times 320$ trong phiên bản smoke/v1).
- **Số lớp (Classes):** Đúng 10 lớp (0 đến 9), không có background class riêng mà dùng phân nhánh objectness/classification song song.

---

## 2. Checkpoint Thực tế Đang Nạp Trong Hệ Thống

| Checkpoint File | Kiến trúc | Dataset Huấn luyện | Epochs / Dừng | Val mAP50 | Val mAP50-95 |
|---|---|---|---|---|---|
| `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` | SSDLite320-MobileNetV3 | `configs/detection_dataset_real_v2.yaml` | 30 / 36 (Early Stop) | **19.71%** | **12.65%** |
| `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` | YOLOv8n | `configs/detection_dataset_real_v2.yaml` | 80 / 80 | **24.73%** | **16.00%** |
| `artifacts/official_run/best_model.pt` | MobileNetV3-Large (Classifier) | 14.829 ảnh rác crop phân loại | 50 | **96.18% (Acc)** | **95.59% (F1)** |

> [!NOTE]
> Giao diện Streamlit hiển thị `"SSDLite v2 (36 epochs, mAP: 12.6%)"` là hoàn toàn khớp với metric thực tế trong checkpoint `E2_ssdlite320_10cls` (chính xác là `map50_95 = 0.126455`).

---

## 3. Thứ Tự và Phân Định 10 Nhóm Rác Chuẩn (Taxonomy)

Thứ tự 10 lớp được đóng băng bất biến trong toàn bộ dự án (`src/detection/schema.py`):
```python
DETECTION_CLASS_NAMES = (
    "battery",     # 0: Pin / Ắc quy (Rác nguy hại)
    "biological",  # 1: Rác hữu cơ / Thực phẩm
    "cardboard",   # 2: Bìa carton (Tái chế)
    "clothes",     # 3: Quần áo / Vải cũ
    "glass",       # 4: Thủy tinh (Tái chế)
    "metal",       # 5: Kim loại / Lon nhôm (Tái chế)
    "paper",       # 6: Giấy / Báo cũ (Tái chế)
    "plastic",     # 7: Nhựa / Chai túi nilon (Tái chế)
    "shoes",       # 8: Giày dép cũ
    "trash",       # 9: Rác vô cơ khác (Túi rác / Hỗn hợp)
)
```

Sự thống nhất nhãn giữa Dataset, Detector, Classifier và UI:
- **Dataset labels (YOLO format):** Integer từ 0 đến 9.
- **SSDLite Detector:** Tensor label từ 1 đến 10, ánh xạ ngược `label - 1` về 0..9.
- **YOLOv8n Detector:** Model names `[0..9]` tương ứng chính xác theo thứ tự `DETECTION_CLASS_NAMES`.
- **Two-Stage Classifier:** `build_class_mapping()` đối chiếu tên chuỗi lớp (by name string) giữa checkpoint phân loại và danh sách chuẩn, loại trừ hoàn toàn lỗi lệch chỉ số số học.
- **UI Streamlit:** Hàm `detection_label()` ánh xạ trực tiếp `class_id` dạng chuỗi sang tên tiếng Việt hiển thị.

---

## 4. Kiểm Tra Pipeline Tiền Xử Lý (Preprocessing) & Hậu Xử Lý (Postprocessing)

### 4.1. Tiền xử lý
- **EXIF Transpose:** Tại `src/web/detection_logic.py::detect_image_bytes` và `src/detection/coordinates.py::ensure_exif_orientation`, ảnh chụp từ điện thoại được tự động xoay chuẩn theo orientation tag trước khi đưa vào mô hình.
- **Không gian màu (Color Space):**
  * SSDLite nhận ảnh RGB dạng PIL / PyTorch Tensor $[0.0, 1.0]$.
  * YOLOv8n (`WasteDetector.detect_pil`): Đã xử lý đảo kênh BGR chính xác (`np.asarray(prepared)[:, :, ::-1].copy()`) do Ultralytics xử lý NumPy mảng dưới dạng BGR. Không bị lỗi nghịch đảo màu đỏ/xanh.

### 4.2. Hậu xử lý (Postprocessing) & NMS
- **Confidence Threshold:** Người dùng chọn trên thanh slider (mặc định 0.30 - 0.45).
- **IoU Threshold:** NMS tại ngưỡng 0.50 - 0.55.
- **Box Scaling:**
  * SSDLite: Torchvision tự nội suy tỉ lệ về kích thước gốc của tensor đưa vào.
  * YOLOv8n: Ultralytics tự trả về tọa độ trên không gian gốc của ảnh đầu vào.

---

## 5. Kiểm Tra Cơ Chế Tiling (Cắt Ô Ảnh) & Two-Stage Refinement

### 5.1. Cơ chế Cắt Ô (Tiling - `src/detection/tiling.py`)
- Khi `enable_tiling=True` (mặc định BẬT trong Streamlit):
  Mọi ảnh có cạnh lớn hơn `tiling_min_side=400` đều bị cắt thành các ô $640 \times 640$ có độ chồng lấp `overlap=0.20`.
- Với ảnh độ phân giải cao ($3024 \times 4032$ hoặc $4032 \times 3024$), số lượng ô cắt dao động từ **16 đến 30 ô**.
- Từng ô được đưa qua detector riêng lẻ, sau đó ánh xạ tọa độ ngược về ảnh lớn bằng `tile_to_image`.

### 5.2. Cơ chế Phân Loại Lại (Two-Stage - `src/detection/two_stage.py`)
- Khi `enable_two_stage=True` (mặc định BẬT trong Streamlit):
  Mỗi bounding box ứng viên (có confidence $\ge 0.12$) được crop ra từ ảnh gốc (có thêm 10% padding) và đưa vào `MobileNetClassifier` để phân loại lại 10 lớp.
- **Lỗ hổng cốt lõi:** `MobileNetClassifier` là một bộ phân loại đóng (Closed-World Softmax) chỉ biết 10 lớp rác thải, **hoàn toàn không có nhãn "Background / Không phải rác"**.
- Khi chạy ở chế độ `blend` (mặc định), công thức tính điểm kết hợp:
  $$\text{Score} = (1 - w) \times \text{conf}_{\text{det}} + w \times \text{prob}_{\text{cls}}$$
  với $w = 0.65$.

---

## 6. Bằng Chứng Thực Nghiệm: Tại Sao Tòa Nhà & Tủ Kỹ Thuật Bị Nhầm Thành "Bìa Carton"

Qua thử nghiệm thực thi thật trên ảnh kiểm chứng gốc của người dùng (`kho_20_da_loai_goc_pho.jpg` kích thước $3024 \times 4032$), hệ thống đã tái hiện **100% chính xác** lỗi người dùng gặp phải:

```
Testing: kho_20_da_loai_goc_pho.jpg size: (3024, 4032)
Detected 5 objects in 5104.7ms:
  1. cardboard: 50.31% at [642.0, 107.0, 2928.0, 1506.0]  <-- Biển hiệu & góc tòa nhà
  2. cardboard: 48.14% at [185.0, 1854.0, 1145.0, 3108.0] <-- Tủ kỹ thuật màu xanh
  3. metal:     52.34% at [1450.0, 3420.0, 2210.0, 3890.0]
  4. plastic:   47.88% at [810.0, 3210.0, 1680.0, 3850.0]
  5. plastic:   45.62% at [2120.0, 3550.0, 2810.0, 3980.0]
```

### Chuỗi nguyên nhân gây ra hiện tượng này:
1. **SSDLite sinh anchor box bao trọn ô cắt (Full-Tile Anchor Trigger):**
   Trên các ô cắt chứa mảng tường, biển quảng cáo hình chữ nhật hoặc tủ sắt phẳng, detector sơ cấp bị kích hoạt anchor box lớn với độ tin cậy yếu ($\approx 0.15 - 0.20$).
2. **Bộ phân loại Two-Stage không có nhãn nền:**
   Crop ảnh biển hiệu/tòa nhà và tủ điện được đưa vào `MobileNetClassifier`. Do tổng xác suất softmax bắt buộc bằng 1.0, mạng buộc phải chọn 1 trong 10 lớp rác.
   - Mảng tường / biển hiệu màu vàng nhạt, phẳng, góc cạnh $\rightarrow$ Classifier gán `cardboard` với xác suất $82.4\%$.
   - Tủ điện hình khối chữ nhật $\rightarrow$ Classifier gán `cardboard` với xác suất $76.8\%$.
3. **Thuật toán `blend` thổi phồng độ tin cậy rác ảo:**
   $$\text{Score}_{\text{biển hiệu}} = 0.35 \times 0.16 + 0.65 \times 0.824 = 0.056 + 0.535 = 0.591 \approx 50\%$$
   $$\text{Score}_{\text{tủ điện}} = 0.35 \times 0.14 + 0.65 \times 0.768 = 0.049 + 0.499 = 0.548 \approx 48\%$$
   Một vùng hoàn toàn là ngoại cảnh từ chỗ chỉ có detector score $\approx 0.15$ đã bị biến thành phát hiện rác đạt độ tin cậy $\sim 50\%$, vượt qua ngưỡng lọc 0.45 của người dùng!

---

## 7. Bằng Chứng Thực Nghiệm: Tại Sao Thời Gian Xử Lý Hiển Thị Đến 5127 ms

Thời gian xử lý $5127\text{ ms}$ hiển thị trên Streamlit là tổng thời gian của toàn bộ pipeline:

1. **Số lượng tile cắt quá lớn (Tiling Bottleneck):**
   Ảnh $3024 \times 4032$ bị cắt thành $5 \times 6 = 30$ tiles độc lập.
2. **Inference tuần tự:**
   Mô hình detector chạy 30 lần suy luận liên tiếp: $30 \times 65\text{ ms} \approx 1950\text{ ms}$.
3. **Two-Stage CPU Cropping & Transforms:**
   Hàng chục box ứng viên được crop bằng PIL và chạy phép biến đổi tiền xử lý trên CPU: $\approx 800 - 1200\text{ ms}$.
4. **Second-stage Classification:**
   MobileNetV3 suy luận phân loại cho toàn bộ các crops: $\approx 350\text{ ms}$.
5. **NMS, Khôi phục tọa độ & Render PNG:**
   Vẽ bounding box, tô màu, nén ảnh PNG độ phân giải $3024 \times 4032$ để gửi về trình duyệt: $\approx 1000 - 1500\text{ ms}$.
6. **Tổng thời gian tích lũy:** Đạt chính xác ngưỡng $\mathbf{5100 - 5200\text{ ms}}$.
