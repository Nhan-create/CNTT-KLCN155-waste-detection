# Hướng Dẫn Clone và Chạy Mô Hình Nhận Diện Rác Thải

Tài liệu này hướng dẫn từng bước cách tải mã nguồn (`git clone`) về máy tính cá nhân và khởi chạy các mô hình phân loại / phát hiện rác thải trên **Windows**, **macOS** và **Linux**.

---

## 1. Yêu Cầu Hệ Thống

- **Hệ điều hành:** Windows 10/11 (64-bit), macOS, hoặc Ubuntu/Linux.
- **Python:** Phiên bản **3.10**, **3.11** hoặc **3.12** (kiểm tra bằng lệnh `python --version`).
- **Git:** Đã cài đặt Git (kiểm tra bằng lệnh `git --version`).
- **Phần cứng:**
  - **CPU:** Chạy tốt trực tiếp trên mọi máy tính phổ thông (RAM tối thiểu 8 GB).
  - **GPU (Tùy chọn):** Nếu máy có card đồ họa NVIDIA hỗ trợ CUDA, hệ thống sẽ tự động nhận diện (`device="auto"`) để tăng tốc độ suy luận.

---

## 2. Các Bước Cài Đặt Từ Đầu

### Bước 1: Clone repository về máy

Mở **Terminal** (hoặc **PowerShell** / **Command Prompt** trên Windows), di chuyển tới thư mục bạn muốn lưu dự án và chạy:

```bash
git clone https://github.com/Nhan-create/CNTT-KLCN155-waste-detection.git
cd CNTT-KLCN155-waste-detection
```

### Bước 2: Tạo và kích hoạt môi trường ảo (`.venv`)

Việc dùng môi trường ảo giúp cài đặt đúng phiên bản thư viện mà không ảnh hưởng tới các dự án Python khác trên máy.

- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```

- **Trên Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **Trên macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

*(Khi kích hoạt thành công, đầu dòng lệnh sẽ xuất hiện chữ `(.venv)`).*

### Bước 3: Cài đặt các thư viện cần thiết

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Tùy chọn cho máy có GPU NVIDIA:** Nếu muốn cài PyTorch bản hỗ trợ CUDA 12.1 thay vì bản mặc định, bạn có thể chạy thêm:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 3. Kiểm Tra Trọng Số Mô Hình Có Sẵn

Ngay sau khi `git clone`, toàn bộ các file trọng số (`.pt`) cần thiết đã nằm sẵn trong thư mục `artifacts/`:

1. **Mô hình phân loại đơn vật thể (MobileNetV3-Large):**
   - Đường dẫn: `artifacts/official_run/best_model.pt`
2. **Mô hình phát hiện đa vật thể chính (SSDLite320-MobileNetV3):**
   - Đường dẫn: `artifacts/detection_v4_expanded/E4_ssdlite320_10cls/weights/best.pt`
3. **Mô hình phát hiện đối chứng (YOLOv8n):**
   - Đường dẫn: `artifacts/detection_v4_expanded/E4_yolov8n_10cls/weights/best.pt`

Nếu vì lý do nào đó file trọng số bị thiếu hoặc bạn muốn tải lại bản gốc từ GitHub Release `v1.0.0-models`, hãy chạy lệnh:

```bash
python scripts/download_models.py
```

---

## 4. Cách Chạy Các Mô Hình

### Cách 1: Chạy ứng dụng Web Streamlit (Khuyên dùng)

Đây là giao diện đầy đủ nhất, tích hợp cả phân loại đơn vật thể và phát hiện vị trí đa vật thể:

```bash
streamlit run streamlit_app.py
```

Mở trình duyệt tại địa chỉ **`http://localhost:8501`**:

1. **Sử dụng Tab 1 — Phân loại rác đơn vật thể (MobileNetV3-Large):**
   - Nhấn **Browse files** để tải lên một ảnh rác (định dạng `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
   - Hệ thống tự động trả về tên loại rác dự đoán, độ tin cậy, biểu đồ xác suất Top-5 lớp và khuyến nghị thùng phân loại rác tại nguồn.

2. **Sử dụng Tab 2 — Phát hiện vị trí đa vật thể (SSDLite320 / YOLOv8n / WBF):**
   - Ở thanh công cụ bên trái (**TÙY CHỌN MÔ HÌNH PHÁT HIỆN**):
     - **Mô hình:** Chọn `SSDLite320-MobileNetV3 (Mô hình chính)`, `YOLOv8n (Mô hình đối chứng)` hoặc `Hợp nhất SSDLite + YOLOv8n (WBF)`.
     - **Ngưỡng tin cậy:** Mặc định `0.22` (có thể kéo tăng để lọc bớt khung điểm thấp hoặc giảm để bắt thêm vật thể khó).
     - **Chia ảnh để tìm thêm vật thể nhỏ (Tiling):** Cắt ảnh thành các ô chồng lấp giúp phát hiện vật thể rác kích thước nhỏ.
     - **Phân loại lại từng khung bằng MobileNetV3 (Two-Stage):** Dùng MobileNetV3-Large thẩm định lại từng vùng ảnh được khoanh.
   - Tải lên một hoặc nhiều ảnh ở khung chính rồi bấm nút **Phát hiện vị trí rác**.
   - Xem ảnh đã vẽ khung bao, bảng tọa độ chi tiết, bảng đếm số lượng từng loại rác và tải kết quả về máy (`.PNG`, `.CSV`, hoặc gói `.ZIP` khi tải nhiều ảnh).

---

### Cách 2: Chạy ứng dụng Desktop PyQt5

Ứng dụng cửa sổ trực tiếp trên máy tính, hỗ trợ chọn ảnh từ ổ đĩa, mở webcam trực tiếp và lưu lịch sử phân loại vào cơ sở dữ liệu SQLite (`outputs/history.sqlite3`):

```bash
python app.py --checkpoint artifacts/official_run/best_model.pt --device auto --confidence-threshold 0.55
```

---

### Cách 3: Chạy suy luận nhanh qua dòng lệnh (CLI)

Phân loại một ảnh hoặc cả một thư mục ảnh và xuất kết quả dạng JSON ra màn hình:

```bash
python -m src.detection.predict --checkpoint artifacts/official_run/best_model.pt --input duong_dan_anh_hoac_thu_muc --device auto
```

---

## 5. Kiểm Tra Hệ Thống Sau Khi Cài Đặt

Để xác nhận môi trường và toàn bộ mô hình hoạt động ổn định, chạy lệnh kiểm thử:

```bash
pytest -q
```

Nếu tất cả bài kiểm thử hiển thị `passed`, môi trường của bạn đã sẵn sàng 100%.

---

## 6. Xử Lý Một Số Lỗi Thường Gặp

1. **Lỗi `streamlit: The term 'streamlit' is not recognized...`:**
   - Nguyên nhân: Chưa kích hoạt môi trường ảo `.venv`.
   - Cách khắc phục: Kích hoạt `.venv` theo Bước 2, hoặc chạy trực tiếp qua Python trong `.venv`:
     ```powershell
     .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
     ```

2. **Lỗi cổng `8501` đang được sử dụng:**
   - Cách khắc phục: Chạy Streamlit trên một cổng khác, ví dụ cổng `8502`:
     ```bash
     streamlit run streamlit_app.py --server.port 8502
     ```

3. **Lỗi thiếu `DLL` của OpenCV hoặc PyTorch trên Windows mới cài:**
   - Cách khắc phục: Cài đặt gói [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) của Microsoft rồi khởi động lại Terminal.
