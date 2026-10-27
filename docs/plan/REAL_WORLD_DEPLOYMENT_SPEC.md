# ĐẶC TẢ KỸ THUẬT TRIỂN KHAI THỰC TẾ (REAL-WORLD DEPLOYMENT SPECIFICATION)
**Hệ thống Nhận diện 10 Loại Rác Thải Đa Chế độ (Fixed & Mobile Camera)**  
**Dự án:** `CNTT-KLCN155-waste-detection`  
**Giai đoạn:** Sprint 0 — Thiết kế Kiến trúc & Chuẩn hóa Kỹ thuật  
**Tác giả:** Antigravity Integrator (Multi-Agent Team)  
**Ngày lập:** 2026-10-12  

---

## 1. TỔNG QUAN & MỤC TIÊU HỆ THỐNG

### 1.1 Bối cảnh
Hệ thống hiện tại đã sở hữu:
- Pipeline phát hiện đa vật thể 10 lớp (`src/detection/`) với hai backend chính: **SSDLite320-MobileNetV3** (mô hình nghiên cứu chính) và **YOLOv8n** (mô hình đối chứng thực tế cao).
- Module phân loại thứ cấp **MobileNetV3-Large** (96.18% độ chính xác) phục vụ kiến trúc Two-Stage.
- Giao diện Streamlit xử lý ảnh tĩnh (`streamlit_app.py`), cùng các nguyên mẫu live video thô (`src/web/detection_live.py`, `src/web/detection_video.py`).

### 1.2 Mục tiêu chuyển đổi thực tế
Nâng cấp toàn diện hệ thống từ chế độ "xử lý ảnh đơn lẻ" sang **hệ thống thị giác máy tính thời gian thực ngoài thực địa** phục vụ 2 kịch bản vận hành cốt lõi:
1. **Camera cố định (Fixed Camera Mode):** Điểm thu gom rác thông minh, băng chuyền phân loại, thùng rác công cộng tự động, hoặc camera an ninh (IP/RTSP/USB Webcam).
2. **Camera di động (Mobile Camera Mode):** Nhân viên tuần tra môi trường, người dân phân loại rác tại nguồn thông qua điện thoại thông minh (smartphone browser) hoặc thiết bị cầm tay.

### 1.3 Ràng buộc hệ thống bất biến
1. **Taxonomy 10 lớp cố định:** `[0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash]`.
2. **Khung phần cứng khắt khe:** Laptop Host NVIDIA GeForce RTX 2050 (4.0 GB VRAM, 15.31 GB RAM). VRAM khả dụng cho mô hình: $\le 2.8\text{ GB}$ (giữ an toàn hệ thống $\ge 1.2\text{ GB}$).
3. **Độ trễ vận hành:** End-to-end latency $\le 120\text{ ms}$ trên luồng camera thời gian thực, FPS ổn định $\ge 12\text{ fps}$ (khuyên dùng $15\text{--}20\text{ fps}$).
4. **Không làm hỏng luồng chạy ảnh hiện tại:** Bảo toàn nguyên vẹn Tab Phân loại và Tab Phát hiện ảnh tĩnh trong `streamlit_app.py`.

---

## 2. KIẾN TRÚC LÕI DÙNG CHUNG (SHARED INFERENCE CORE)

Cả hai chế độ Fixed Camera và Mobile Camera đều sử dụng chung một engine suy luận đa luồng an toàn (`SharedInferenceCore`) nhằm tránh trùng lặp mã nguồn và tiết kiệm tối đa VRAM.

```
+-----------------------------------------------------------------------------------+
|                           SHARED INFERENCE CORE ENGINE                            |
|                                                                                   |
|  +--------------------+    +---------------------+    +------------------------+  |
|  | Input Ingestion    |    | Coordinate Pipeline |    | Detector Backend       |  |
|  | - USB Webcam (cv2) |    | - EXIF Transpose    |    | - YOLOv8n (Primary)    |  |
|  | - RTSP/IP Stream   |--->| - Aspect Letterbox  |--->| - SSDLite320 (Compare) |  |
|  | - WebRTC/HTTP Push |    | - Bound Clipping    |    | - FP16 / TensorRT/ONNX |  |
|  +--------------------+    +---------------------+    +------------------------+  |
|             |                         |                            |              |
|             v                         v                            v              |
|  +--------------------+    +---------------------+    +------------------------+  |
|  | Bounded Frame Drop |    | Scale Back Coords   |    | Post-processing        |  |
|  | Queue (Max size=1) |    | (src/detection/     |    | - Class Thresholds     |  |
|  | Zero Lag Guarantee |    |  coordinates.py)    |    | - Two-Stage Classifier |  |
|  +--------------------+    +---------------------+    | - Temporal Tracking    |  |
|                                                       +------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 2.1 Đặc tả thành phần Shared Core
1. **Thread-safe Inference Worker:**
   - Quản lý vòng đời nạp model duy nhất trong bộ nhớ (Singleton Pattern qua `@st.cache_resource` hoặc dedicated daemon process).
   - Tự động kiểm tra an toàn phần cứng thông qua `src/detection/preflight_guard.py` trước khi cấp phát bộ nhớ GPU.
2. **Xử lý hình học và tọa độ (`src/detection/coordinates.py`):**
   - Mọi khung hình nhận vào (bất kể độ phân giải gốc $640\times 480$, $1280\times 720$, $1920\times 1080$, hay $4000\times 3000$) đều đi qua hàm `letterbox_image(image, target_size=512)`.
   - Giữ nguyên tỉ lệ khung hình (aspect ratio), chèn viền màu trung tính `(114, 114, 114)`.
   - Kết quả dự đoán trên không gian letterbox được chiếu ngược về không gian ảnh gốc bằng `scale_coords_back()`, đảm bảo bounding box khớp từng pixel.
3. **Cơ chế chống tích lũy độ trễ (Zero-Lag Bounded Queue):**
   - Stream video gửi về hàng đợi có kích thước giới hạn `queue_size = 1`.
   - Nếu worker inference đang bận xử lý frame $N$, các frame $N+1, N+2$ đến sau sẽ tự động bị drop. Khi worker hoàn tất, nó luôn lấy frame mới nhất trên camera. Điều này triệt tiêu hoàn toàn hiện tượng "video đi chậm hơn thời gian thực".
4. **Hậu xử lý và khử nhiễu:**
   - Áp dụng bảng ngưỡng độ tin cậy theo từng lớp (`configs/detection_thresholds_calibrated.json`).
   - Tùy chọn Two-Stage (`src/detection/two_stage.py`) sử dụng MobileNetV3 để xác thực lại nhãn các vùng cắt rác nhỏ trước khi hiển thị.

---

## 3. CHẾ ĐỘ 1: CAMERA CỐ ĐỊNH (FIXED CAMERA MODE)

### 3.1 Nguồn cấp dữ liệu (Input Sources)
- **USB Webcam:** Truy cập qua chỉ số thiết bị phần cứng (Index 0, 1, 2) thông qua OpenCV Backend (`cv2.CAP_DSHOW` trên Windows để khởi động nhanh). Đã kiểm thử thực tế trên máy trạm: `Camera index 0 opened: True, Shape: (480, 640, 3)`.
- **IP Camera / RTSP Stream:** Nhận URI luồng: `rtsp://<user>:<pass>@<ip>:<port>/h264Preview_01_main` hoặc `http://<ip>:<port>/video.mjpg`.

### 3.2 Cơ chế kết nối bền bỉ (Auto-Reconnect Worker)
- Luồng camera được đọc trong một background thread chuyên biệt (`VideoStreamReader`).
- Khi phát hiện mất frame liên tục 3 giây hoặc `cap.read()` trả về `False`:
  - Đóng và giải phóng tài nguyên stream cũ.
  - Kích hoạt thuật toán phục hồi với cơ chế Exponential Backoff: thử lại sau $1\text{s} \to 2\text{s} \to 4\text{s} \to 8\text{s} \dots \max 30\text{s}$.
  - Ghi nhận nhật ký trạng thái kết nối và hiển thị cảnh báo trực quan trên giao diện mà không làm crash ứng dụng.

### 3.3 Vùng quan sát trọng tâm (Region of Interest - ROI)
- Cho phép người dùng thiết lập đa giác hoặc hình chữ nhật ROI trực tiếp trên giao diện.
- Chỉ thực hiện nhận diện hoặc đếm rác bên trong vùng quy định (ví dụ: bề mặt thùng rác, lòng băng tải).
- Bỏ qua các chuyển động ngoại cảnh ở rìa (người đi lại, xe cộ, tường gạch) nhằm triệt tiêu False Positives.

### 3.4 Theo dõi và chống đếm trùng theo thời gian (Tracking & Anti-Duplicate)
- Trong bài toán camera giám sát cố định $\ge 1$ giờ: rác nằm yên trong khung hình có thể bị đếm đi đếm lại hàng ngàn lần nếu chỉ phát hiện theo từng frame.
- **Giải pháp:** Sử dụng `TemporalDetectionSmoother` (`src/detection/tracking.py`) mở rộng:
  - Gán định danh duy nhất (`track_id`) cho từng vật thể dựa trên Hungarian Algorithm kết hợp ma trận IoU.
  - Chỉ tăng bộ đếm số lượng rác (`cumulative_waste_count`) khi một `track_id` mới xuất hiện ổn định $\ge 5$ khung hình liên tiếp.
  - Khi rác được lấy đi (vắng mặt quá `max_missed = 30` khung hình $\approx 2$ giây), track tương ứng tự động kết thúc.

---

## 4. CHẾ ĐỘ 2: CAMERA DI ĐỘNG (MOBILE CAMERA MODE)

### 4.1 Mô hình kết nối và giao thức
Người dùng mở ứng dụng từ trình duyệt điện thoại (Chrome/Safari) qua mạng nội bộ Wi-Fi hoặc kết nối Internet bảo mật.

```
+-----------------------------------+               +-----------------------------------+
|      MOBILE CLIENT (PHONE)        |               |      INFERENCE HOST SERVER        |
|                                   |               |                                   |
| [Back Camera] -> HTML5 Canvas     |    WebRTC     | WebRTC PeerConnection             |
| - Auto Orientation Handling       |  (or WS/HTTP) | - Thread-safe Frame Decode        |
| - Downscale to 720p/512px         |-------------->| - SharedInferenceCore             |
| - Low-latency frame streaming     |               | - Bounding Box Annotation         |
|                                   |<--------------|                                   |
| [Display] Canvas Overlay (BBox)   |  JSON / Video | Return: Annotated Frame or BBoxes |
+-----------------------------------+               +-----------------------------------+
```

1. **Giao thức chính (WebRTC Peer-to-Peer):**
   - Hỗ trợ truyền video 2 chiều độ trễ cực thấp ($< 150\text{ ms}$).
   - Yêu cầu HTTPS (tự sinh chứng chỉ SSL/TLS cục bộ hoặc sử dụng Cloudflare Tunnel/ngrok khi chạy kiểm thử từ xa).
2. **Giao thức dự phòng (Lightweight WebSocket / HTTP Stream):**
   - Nếu môi trường mạng cấm UDP hoặc không cấu hình được WebRTC ICE: Client điện thoại chụp frame từ thẻ `<video>` và gửi qua WebSocket nhị phân hoặc HTTP POST `/api/detect_frame` với tần suất $5\text{--}10\text{ fps}$.
3. **Chế độ Chụp & Phân tích cận cảnh (Snapshot Fallback):**
   - Khi mạng yếu hoặc rác ở quá xa/nhỏ: Người dùng bấm nút **"Chụp ảnh phân tích"**. Điện thoại gửi ảnh tĩnh độ phân giải cao lên server, kích hoạt đầy đủ Tiling (`src/detection/tiling.py`) và Two-Stage (`src/detection/two_stage.py`) để phân tích chi tiết.

### 4.2 Xử lý xoay máy và thay đổi độ phân giải (Orientation & Resolution)
- **Xoay máy (Portrait vs Landscape):**
  - Điện thoại thường chụp dọc ($9:16$). Nếu mô hình nhận diện được huấn luyện trên ảnh vuông hoặc ngang, việc xoay sai hướng sẽ làm giảm nghiêm trọng độ chính xác.
  - Sử dụng `ensure_exif_orientation()` (`src/detection/coordinates.py`) và đọc metadata hướng xoay thiết bị (`window.orientation` / CSS transform).
  - Tự động đệm letterbox chính xác để bảo toàn hình dạng rác (không làm lon nước bị bẹp hay chai nhựa bị kéo dẹt).
- **Điều chỉnh độ phân giải phía Client (Dynamic Client Downscaling):**
  - Camera điện thoại có thể lên tới $4K$ hoặc $12\text{ MP}$. Nếu truyền trực tiếp luồng $4K$ qua Wi-Fi sẽ gây nghẽn băng thông và tràn bộ nhớ.
  - Client-side JavaScript tự động resize frame về độ phân giải chuẩn ($720\text{p}$ hoặc $640\times 480$) trước khi truyền lên inference server.

---

## 5. THIẾT KẾ TÍCH HỢP GIAO DIỆN (UI/UX STREAMLIT)

Giao diện `streamlit_app.py` được mở rộng thêm các Tab chuyên biệt mà không can thiệp vào các Tab đang chạy ổn định:

| Tab | Tên chức năng | Mục đích sử dụng | Nguồn dữ liệu |
|---|---|---|---|
| **Tab 1** *(Hiện hữu)* | Phân loại MobileNetV3 96.2% | Phân loại rác cận cảnh một vật thể | Ảnh tải lên (`st.file_uploader`) |
| **Tab 2** *(Hiện hữu)* | Phát hiện vị trí đa vật thể | Bounding box nhiều rác, Tiling, Two-Stage | Ảnh tải lên (hàng loạt) |
| **Tab 3** *(Mới)* | 📹 Camera cố định (Webcam / RTSP) | Giám sát điểm thu gom, băng chuyền, thùng rác | `cv2.VideoCapture`, RTSP URI |
| **Tab 4** *(Mới)* | 📱 Camera di động (Mobile Phone) | Quét rác bằng điện thoại, tạo QR Code kết nối | WebRTC / Mobile Canvas Stream |

### 5.1 Kiểm soát tài nguyên chuyển Tab (Resource Cleanup)
- Khi người dùng chuyển từ Tab Camera sang Tab Ảnh hoặc ngược lại:
  - Tự động ngắt kết nối `VideoCapture` và đóng WebRTC connection.
  - Giải phóng hàng đợi frame đệm.
  - Gọi `torch.cuda.empty_cache()` để dọn dẹp VRAM rác, đảm bảo VRAM không bị rò rỉ (memory leak).

---

## 6. TIÊU CHÍ NGHIỆM THU KỸ THUẬT (ACCEPTANCE CRITERIA)

| Hạng mục | Tiêu chí Định lượng / Định tính | Phương pháp Kiểm chứng |
|---|---|---|
| **Độ ổn định Camera cố định** | Chạy liên tục $\ge 60$ phút không crash, không rò rỉ RAM/VRAM ($< 10\%$ biến động). | Log giám sát tài nguyên bằng `psutil` và `torch.cuda.memory_allocated()`. |
| **Tự phục hồi stream** | Tự động kết nối lại trong vòng $\le 5$ giây sau khi giả lập rút cáp USB hoặc ngắt mạng RTSP. | Kịch bản kiểm thử ngắt kết nối có chủ đích (`tests/web/test_camera_recovery.py`). |
| **Độ trễ Camera di động** | End-to-end latency $< 150\text{ ms}$ trên mạng Wi-Fi tiêu chuẩn, hiển thị khung bao mượt mà. | Đo đạc timestamp round-trip (client send $\to$ server return). |
| **Chống đếm trùng (Tracking)** | Một vật thể rác cố định trong 5 phút chỉ được tính là 1 vật thể duy nhất. | Test case video tĩnh và đối soát ID vật thể. |
| **Bảo toàn chức năng cũ** | 100% test suites hiện hữu (`tests/detection`, `tests/qa`, `tests/web`) vượt qua đầy đủ (**179+ tests PASSED**). | Chạy hồi quy tự động với `pytest`. |
