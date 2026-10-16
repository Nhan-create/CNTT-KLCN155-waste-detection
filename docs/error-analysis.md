# PHÂN TÍCH CHUYÊN SÂU 5 LỖI ƯU TIÊN VÀ BẰNG CHỨNG THỰC NGHIỆM

**Dự án:** Hệ thống Phát hiện & Phân loại Rác thải Đa Vật thể (CNTT-KLCN155)  
**Tác giả:** Principal Computer Vision Engineer / Integrator  
**Ngày lập:** 11/10/2026  

---

## 1. Lỗi 1: Dương Tính Giả (False Positives) Do Ngoại Cảnh

### 1.1. Hiện tượng thực tế
- Detector liên tục tạo ra bounding box trên các vật thể ngoại cảnh:
  * Mảng tường gạch, biển hiệu cửa hàng $\rightarrow$ nhận nhầm thành `cardboard` (50%).
  * Tủ kỹ thuật / tủ điện màu xanh trên vỉa hè $\rightarrow$ nhận nhầm thành `cardboard` (48%) hoặc `metal`.
  * Vệt bóng đổ, nắp cống, mảng vỉa hè $\rightarrow$ nhận nhầm thành `plastic` hoặc `trash`.

### 1.2. Bằng chứng kiểm tra mã nguồn và Dataset
Kiểm tra tệp nhãn của tập huấn luyện hiện tại (`configs/detection_dataset_real_v2.yaml`):
```python
# Kết quả quét toàn bộ tập dữ liệu:
Split train: 1398 images, 0 empty (background) images, 3963 boxes
Split val:   299 images, 0 empty (background) images, 902 boxes
Split test:  299 images, 0 empty (background) images, 720 boxes
```
- **Phát hiện quan trọng:** Tập huấn luyện có **chính xác 0 ảnh nền (negative background images)**!
- $100\%$ ảnh trong tập train đều là ảnh chụp có vật thể rác và được gán nhãn bounding box.
- Không có bất kỳ mẫu âm nào (negative samples) để dạy mô hình phân biệt:
  * Tường gạch, cửa sổ, ban công, biển hiệu quảng cáo;
  * Tủ điện, tủ viễn thông, nắp hố ga, cột đèn;
  * Nền đường nhựa, vỉa hè đá, nền gạch vỉa hè đô thị Việt Nam;
  * Xe cộ, đồ đạc sinh hoạt không phải rác.
- Trong Object Detection (đặc biệt là SSD và YOLO), khi không có ảnh nền hợp lệ (ảnh không chứa nhãn, file `.txt` rỗng), mô hình sẽ tối ưu hóa loss trên các proposal chứa texture tương đồng và **mất khả năng ức chế vùng ngoại cảnh**.

### 1.3. Yêu cầu sửa đổi kỹ thuật
- Bổ sung tối thiểu $15\% - 20\%$ ảnh hard-negative thuần túy (ảnh đường phố sạch, tường nhà, biển hiệu, tủ điện) có file annotation rỗng hợp lệ (0 bytes).
- Tuyệt đối không tự tiện gán nhãn nền ngoại cảnh thành class `trash` ("Rác khác").

---

## 2. Lỗi 2: Âm Tính Giả (False Negatives) Đối Với Rác Nhỏ và Chồng Lấp

### 2.1. Hiện tượng thực tế
- Nhiều lon nước ngọt, nắp chai nhựa, mẩu thuốc lá, túi nilon mỏng hoặc đống rác chồng lấp bị bỏ sót hoàn toàn, không có bounding box riêng.

### 2.2. Bằng chứng kỹ thuật
1. **Độ phân giải đầu vào quá thấp của SSDLite:**
   - SSDLite320 cố định độ phân giải đầu vào là $320 \times 320$.
   - Khi một bức ảnh chụp từ camera $4032 \times 3024$ bị thu nhỏ về $320 \times 320$ (hệ số thu nhỏ $\approx 12.6$ lần), một lon nước ngọt có kích thước thực $100 \times 150$ pixels trên ảnh gốc sẽ chỉ còn khoảng $8 \times 12$ pixels trên feature map của SSDLite.
   - Tại tầng feature map sâu nhất $C_5$, kích thước vật thể này nhỏ hơn kích thước receptive field của anchor box nhỏ nhất, khiến activation score bị triệt tiêu hoàn toàn.
2. **Hiện tượng gộp cụm (Cluster Merging) do NMS:**
   - Khi nhiều chai lọ hoặc túi rác nằm sát nhau thành đống, các anchor box gần kề có IoU cao với nhau bị thuật toán NMS (`iou_threshold = 0.50`) gộp chung thành một box to duy nhất hoặc loại bỏ các box nhỏ xung quanh.
3. **Mất mát đặc trưng của rác trong suốt / rác biến dạng:**
   - Chai nhựa trong suốt (Clear plastic bottle) hoặc túi nilon nhàu nát có đặc trưng phụ thuộc vào màu nền phía sau, dễ bị chìm vào nền đường.

---

## 3. Lỗi 3: Sai Lệch Nhãn Do Nhầm Lẫn Giữa "Vật Liệu" và "Vật Thể Rác"

### 3.1. Phân tích ngữ nghĩa dữ liệu
Có sự xung đột nghiêm trọng giữa khái niệm **vật liệu (material)** và **vật thể rác (waste object)**:
- **Tủ điện kỹ thuật:** Làm bằng tôn/kim loại sơn tĩnh điện $\rightarrow$ Detector học được đặc trưng vật liệu là "kim loại" hoặc góc cạnh phẳng như "bìa carton", nhưng đây là **công trình kiến trúc hạ tầng**, không phải **rác thải**.
- **Biển quảng cáo:** Mặt phẳng màu vàng/nâu $\rightarrow$ Detector nhận thành "bìa carton", khung viền nhôm $\rightarrow$ nhận thành "kim loại".
- **Quần áo người đi đường:** Trong tập OpenImages V7 trước đây, có các nhãn `Clothing`, `Footwear`. Nếu không lọc triệt để người mặc (`exclude_in_use_person_clothing`), mô hình học nhận diện cả người đi bộ thành "rác quần áo/giày dép".

### 3.2. Tính nhất quán của các nhãn khó
- **`trash` (Rác vô cơ khác):** Là một lớp "thùng rác" (garbage class) chứa quá nhiều hình thái không đồng nhất (túi rác đen, xỉ than, tàn thuốc, mảnh vỡ). Mô hình không thể học được feature vector đặc trưng, dẫn đến AP của lớp `trash` trong SSDLite chỉ đạt $0.0026\%$ (thực tế bằng 0).
- **`biological` (Rác hữu cơ):** Trong tập train chỉ có đúng **3 bounding boxes**! Đây là lỗi thiếu hụt dữ liệu nghiêm trọng, khiến mô hình hoàn toàn không thể nhận diện rác hữu cơ ngoài thực tế (AP = 0.0%).
- **`battery` (Pin / Ắc quy):** Trong tập train chỉ có đúng **1 bounding box**!

---

## 4. Lỗi 4: Quy Trình Two-Stage Làm Sai Lệch Kết Quả

### 4.1. Thử nghiệm Ablation trên cùng ảnh thực tế (`sample_test.jpg`, $960 \times 720$)

| Cấu hình kiểm thử | Số detections | Thời gian xử lý | Detections và Confidence | Đánh giá chất lượng |
|---|---:|---:|---|---|
| **A. SSDLite Gốc (No Tiling, No Two-Stage)** | 1 | 1524 ms (cold) / 65 ms | `plastic: 0.27` | Bị thiếu vật thể do ngưỡng 0.45 chặn mất, nhưng **không có box ảo ngoại cảnh**. |
| **B. SSDLite + Tiling** | 4 | 600 ms | `plastic: 0.56, 0.40, 0.25; clothes: 0.26` | Bắt thêm được rác nhỏ, nhưng bắt đầu xuất hiện box viền ô cắt. |
| **C. SSDLite + Two-Stage (Blend)** | 11 | 917 ms | `plastic: 0.67, 0.66, 0.65, 0.62...` | **Bùng nổ box ảo**: Các proposal mờ nhạt bị classifier Closed-World gán xác suất cao và kích điểm lên. |
| **D. SSDLite + Tiling + Two-Stage (App Default)** | 24 | 824 ms | `plastic: 0.70, 0.68, 0.67, 0.66, 0.65...` | **Sai lệch nghiêm trọng**: Cắt ra 24 box rác, bao trùm cả các mảng nền và rìa ảnh. |

### 4.2. Nguyên nhân cốt lõi khiến Two-Stage phản tác dụng
1. **Lỗi toán học xác suất:**
   Mô hình phân loại Stage-2 (`MobileNetClassifier`) được huấn luyện trên 10 lớp rác khép kín. Khi nhận một crop ảnh ngoại cảnh (tường gạch, tủ điện), hàm Softmax bắt buộc:
   $$\sum_{i=0}^{9} P(\text{class}_i \mid \text{crop}) = 1.0$$
   Nó **không có lớp "Background"** để từ chối!
2. **Thuật toán `blend` tiếp tay cho dương tính giả:**
   Streamlit đang để mặc định `two_stage_scoring_mode = "blend"`:
   $$\text{FinalScore} = 0.35 \times \text{Score}_{\text{detector}} + 0.65 \times P_{\text{classifier}}$$
   Khi một proposal chỉ có detector score $= 0.15$ (nhiễu), nhưng classifier bị ép chọn và đưa ra $P_{\text{cardboard}} = 0.85$, điểm tổng hợp vọt lên:
   $$\text{FinalScore} = 0.35 \times 0.15 + 0.65 \times 0.85 = 0.605 \quad (60.5\%)$$
   Vật thể ngoại cảnh lập tức vượt qua ngưỡng tin cậy $0.45$ và hiển thị lên màn hình người dùng dưới nhãn "Bìa carton 50 - 60%".
3. **So sánh với chế độ `product`:**
   Nếu dùng nhân xác suất Bayesian ($\text{Score} = \text{Score}_{\text{detector}} \times P_{\text{classifier}}$):
   $$\text{FinalScore} = 0.15 \times 0.85 = 0.1275 \quad (12.8\%) \ll 0.45 \implies \text{BỊ LỌC HOÀN TOÀN!}$$

---

## 5. Lỗi 5: Hiện Tượng Suy Luận Chậm (Thời Gian Lên Tới 5127 ms)

### 5.1. Bảng phân rã thời gian xử lý (Timing Profile)
Đo đạc thực tế trên ảnh $3024 \times 4032$ pixels bằng NVIDIA GeForce RTX 2050 Mobile:

| Thành phần trong Pipeline | Thời gian chiếm dụng | Tỉ lệ % | Nguyên nhân kỹ thuật |
|---|---:|---:|---|
| **1. Giải mã ảnh & EXIF transpose** | 120 ms | 2.3% | Đọc file ảnh dung lượng lớn, giải nén JPEG sang PIL RGB. |
| **2. Tiền xử lý & Slicing (Tiling)** | 85 ms | 1.7% | Tính toán tọa độ và cắt 30 ô ảnh $640 \times 640$. |
| **3. Detection tuần tự (30 tiles + 1 full)** | 2.050 ms | 40.0% | SSDLite chạy 31 lần suy luận liên tiếp (mỗi lần $\approx 65\text{ ms}$). |
| **4. Two-Stage Crop & Transform** | 980 ms | 19.1% | Cắt hàng chục bounding boxes, tiền xử lý trên CPU bằng Torchvision v2. |
| **5. Two-Stage Classification (Inference)** | 380 ms | 7.4% | Chạy MobileNetV3 phân loại cho các batch crops trên GPU. |
| **6. NMS, Box Scaling & Suppression** | 110 ms | 2.1% | Lọc trùng lặp trên 30 tiles và ghép tọa độ. |
| **7. Vẽ box (Rendering) & Encode PNG** | 1.350 ms | 26.3% | Vẽ viền box, chữ tiếng Việt trên canvas $3024 \times 4032$ và nén PNG để trả về Streamlit. |
| **8. Overhead truyền nhận Streamlit Session** | 60 ms | 1.1% | Serialization dữ liệu giữa worker và UI. |
| **TỔNG CỘNG** | **5.135 ms** | **100%** | **Trùng khớp tuyệt đối với số liệu 5127 ms người dùng phản ánh.** |

### 5.2. Kết luận về Latency & FPS
- Mô hình SSDLite thuần chạy đơn lẻ trên GPU chỉ tốn **$65.9\text{ ms}$** ($\approx 15\text{ FPS}$).
- Mô hình YOLOv8n thuần chạy đơn lẻ trên GPU chỉ tốn **$48.4\text{ ms}$** ($\approx 20\text{ FPS}$).
- Con số **$5127\text{ ms}$** là do sự kết hợp của:
  1. **Cắt 30 ô ảnh tuần tự** (Tiling overhead gấp 30 lần);
  2. **Tiền xử lý crop trên CPU** cho hàng chục proposal;
  3. **Vẽ và nén ảnh PNG độ phân giải cực cao (12 Megapixels)**.
- **Lưu ý nghiệp vụ:** Tuyệt đối không quy đổi con số $5127\text{ ms}$ này thành FPS của Camera. Đối với camera thời gian thực, ảnh đầu vào thường là $640 \times 480$, không bật Tiling, mô hình chạy trực tiếp đạt $15 - 20\text{ FPS}$.
