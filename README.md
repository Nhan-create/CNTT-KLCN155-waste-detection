# Hệ Thống Nhận Diện và Phân Loại 10 Nhóm Rác Thải (CNTT-KLCN155)

Ứng dụng nhận diện và phân loại **10 nhóm rác thải sinh hoạt** kết hợp hai giai đoạn:
- **Giai đoạn 1 — Phân loại ảnh đơn vật thể:** Mô hình **MobileNetV3-Large** phân loại chính xác từng nhóm rác và gợi ý thùng phân loại tại nguồn.
- **Giai đoạn 2 — Phát hiện đa vật thể (Object Detection):** Xác định tọa độ khung bao (Bounding Box) và đếm số lượng từng loại rác bằng **SSDLite320-MobileNetV3** (mô hình chính), **YOLOv8n** (mô hình đối chứng) và thuật toán hợp nhất **Weighted Boxes Fusion (WBF)**, kết hợp cắt ô ảnh (**SAHI Tiling**) và phân loại lại hai giai đoạn (**Two-Stage Refinement**).

> Xem hướng dẫn cài đặt chi tiết từng bước tại: [`HUONG_DAN_CAI_DAT.md`](HUONG_DAN_CAI_DAT.md)

---

## 1. 10 Nhóm Rác Thải Hỗ Trợ

| STT | Mã lớp (`class_id`) | Tên tiếng Việt | Nhóm thùng phân loại |
| :--- | :--- | :--- | :--- |
| 1 | `battery` | Pin / Ắc quy | Thùng Đỏ / Cam (Rác nguy hại) |
| 2 | `biological` | Rác hữu cơ / Thực phẩm | Thùng Xanh lá (Rác hữu cơ) |
| 3 | `cardboard` | Bìa carton | Thùng Vàng / Trắng (Rác tái chế) |
| 4 | `clothes` | Quần áo / Vải cũ | Thùng Xám / Tái sử dụng |
| 5 | `glass` | Thủy tinh | Thùng Vàng / Trắng (Rác tái chế) |
| 6 | `metal` | Kim loại / Lon nhôm | Thùng Vàng / Trắng (Rác tái chế) |
| 7 | `paper` | Giấy / Báo cũ | Thùng Vàng / Trắng (Rác tái chế) |
| 8 | `plastic` | Nhựa / Chai túi nilon | Thùng Vàng / Trắng (Rác tái chế) |
| 9 | `shoes` | Giày dép cũ | Thùng Xám / Tái sử dụng |
| 10 | `trash` | Rác vô cơ khác | Thùng Xám (Rác vô cơ khác) |

---

## 2. Trọng Số Mô Hình Đi Kèm Sẵn Khi Clone

Repository đã tích hợp sẵn bộ trọng số đã huấn luyện trong thư mục `artifacts/`, người dùng **chỉ cần `git clone` về là chạy được ngay** mà không cần huấn luyện lại:

- `artifacts/official_run/best_model.pt` — Trọng số phân loại **MobileNetV3-Large** (10 lớp).
- `artifacts/detection_v4_expanded/E4_ssdlite320_10cls/weights/best.pt` — Trọng số phát hiện chính **SSDLite320-MobileNetV3** (E4).
- `artifacts/detection_v4_expanded/E4_yolov8n_10cls/weights/best.pt` — Trọng số phát hiện đối chứng **YOLOv8n** (đã kiểm định trên tập ảnh thực tế).
- `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` & `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` — Trọng số dự phòng phiên bản E2/E1.

---

## 3. Hướng Dẫn Nhanh: Clone Về Máy và Chạy Mô Hình

### Bước 1: Clone mã nguồn từ GitHub

```bash
git clone https://github.com/Nhan-create/CNTT-KLCN155-waste-detection.git
cd CNTT-KLCN155-waste-detection
```

### Bước 2: Tạo môi trường ảo Python (khuyến nghị Python 3.10 – 3.12)

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(Nếu PowerShell báo lỗi quyền thực thi script, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi kích hoạt lại).*

**Trên macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài đặt thư viện phụ thuộc

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Khởi chạy ứng dụng Web Streamlit (Khuyên dùng)

```bash
streamlit run streamlit_app.py
```

Sau khi chạy lệnh, trình duyệt sẽ tự động mở tại địa chỉ **`http://localhost:8501`**:
- **Tab 1 (`Phân loại rác đơn vật thể`):** Tải ảnh cận cảnh một vật thể rác để phân loại bằng MobileNetV3-Large, xem xác suất Top-5 và hướng dẫn bỏ vào đúng thùng rác.
- **Tab 2 (`Phát hiện vị trí đa vật thể`):** Tải một hoặc nhiều ảnh chứa nhiều vật thể rác, chọn mô hình ở thanh bên trái (`SSDLite320-MobileNetV3`, `YOLOv8n`, hoặc `Hợp nhất SSDLite + YOLOv8n (WBF)`), bấm **Phát hiện vị trí rác** để vẽ khung bao, thống kê số lượng và xuất kết quả ra file `.PNG`, `.CSV` hoặc `.ZIP`.

---

## 4. Các Chế Độ Chạy Khác

### Chạy ứng dụng Desktop (PyQt5)
```bash
python app.py --checkpoint artifacts/official_run/best_model.pt --device auto
```

### Chạy phân loại ảnh trực tiếp bằng CLI
```bash
python -m src.detection.predict --checkpoint artifacts/official_run/best_model.pt --input duong_dan_anh.jpg
```

### Chạy bộ kiểm thử tự động (Unit & Integration Tests)
```bash
pytest -q
```

---

## 5. Cấu Trúc Thư Mục Chính

```text
CNTT-KLCN155-waste-detection/
├── artifacts/                  # Bộ trọng số mô hình (.pt) có sẵn khi clone
├── configs/                    # Cấu hình ngưỡng hiệu chuẩn theo từng lớp
├── scripts/                    # Script tải mô hình, chuẩn bị dữ liệu, huấn luyện & đánh giá
├── src/
│   ├── app/                    # Dịch vụ giao diện Streamlit, PyQt5 và điều phối pipeline
│   ├── data/                   # Schema dữ liệu 10 lớp và kiểm tra manifest
│   └── detection/              # Mô hình MobileNetV3, SSDLite320, YOLOv8n, WBF, Tiling, Two-Stage
├── tests/                      # Bộ kiểm thử đơn vị và tích hợp (pytest)
├── app.py                      # Điểm vào ứng dụng Desktop PyQt5
├── streamlit_app.py            # Điểm vào ứng dụng Web Streamlit
├── requirements.txt            # Danh sách thư viện cài đặt
└── HUONG_DAN_CAI_DAT.md        # Tài liệu hướng dẫn cài đặt & vận hành chi tiết
```
