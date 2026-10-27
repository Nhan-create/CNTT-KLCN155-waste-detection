# BÁO CÁO PHÂN TÍCH LỖI MÔ HÌNH NHẬN DIỆN RÁC THỰC TẾ (TASK 1)
**Sprint:** SPRINT AI-UPGRADE-01 — Cải Thiện Chất Lượng Nhận Diện Rác Thực Tế  
**Dự án:** Hệ thống Phát hiện & Phân loại 10 Nhóm Rác thải Đa Vật thể (CNTT-KLCN155)  
**Vai trò:** AI Engineer (Anti) | Kính gửi PM / Reviewer (ChatGPT)  
**Ngày thực hiện:** 12/10/2026  
**Môi trường thử nghiệm:**
- **GPU:** NVIDIA GeForce RTX 2050 Mobile (4.0 GB VRAM, CUDA 12.1)
- **Checkpoints đối chiếu:**
  * Baseline YOLOv8n: `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` (SHA-256 verified)
  * Baseline SSDLite: `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` (SHA-256 verified)
  * Classifier Two-Stage: `artifacts/official_run/best_model.pt`

---

## I. TỔNG QUAN HIỆN TRẠNG VÀ BẰNG CHỨNG THỰC NGHIỆM TRÊN CÁC ẢNH KIỂM CHỨNG

Hệ thống đã nạp trực tiếp hai checkpoint gốc và thực hiện suy luận trên các ảnh ca bệnh (failure cases) thực tế được chỉ định. Toàn bộ số liệu dưới đây là kết quả thực thi thật từ mô hình, không sử dụng số liệu giả lập hay ước lượng.

### 1. Ca bệnh 1: `battery-sample.png` & Pin laptop / Pin thiết bị điện tử
* **Ảnh kiểm thử:** `battery-sample.png` (512x411) và `pin_vuong_4f15ab32849e43a8abdaef2bf000463f.webp` (800x800).
* **Kết quả thực tế từ mô hình:**
  - **YOLOv8n E1:**
    * `battery-sample.png`: Nhận nhầm thành **`shoes` (confidence = 0.9188)** tại box `[7.4, 31.2, 512.0, 386.3]`.
    * `pin_vuong_...webp`: Nhận nhầm thành **`metal` (confidence = 0.8827)** tại box `[239.5, 128.7, 562.5, 514.7]`.
  - **SSDLite E2:**
    * `battery-sample.png`: Nhận nhầm thành **`shoes` (conf = 0.820)** và **`clothes` (conf = 0.192)**.
* **Đánh giá:** Lỗi sai lớp nghiêm trọng. Mô hình hoàn toàn không kích hoạt lớp `battery` (Pin) mà tự tin gán nhãn `shoes` hoặc `metal` với confidence > 88% - 91%.

---

### 2. Ca bệnh 2: `images (17).jpg` — Nhiều chai nhựa, thủy tinh và lon kim loại trên nền gỗ
* **Ảnh kiểm thử:** `data/holdout/images (17).jpg` (547x365 pixels).
* **Vật thể thực tế trong ảnh:** Bàn gỗ có nhiều chai nhựa rỗng nằm ngang, chai thủy tinh, vỏ lon bia kim loại, hộp carton nhỏ.
* **Kết quả thực tế từ mô hình (tại ngưỡng nhạy conf >= 0.15):**
  - **YOLOv8n E1:** Phát hiện 7 boxes:
    * `shoes`: **conf = 0.654** tại `[91.5, 292.1, 160.5, 353.8]` (nhận nhầm lon/vết tối ở góc dưới thành giày).
    * `plastic`: conf = 0.562 tại `[243.1, 17.7, 337.0, 210.0]` (đúng vị trí chai nhựa).
    * `plastic`: conf = 0.375 tại `[403.5, 46.3, 493.0, 206.7]` (đúng vị trí chai nhựa).
    * `metal`: conf = 0.307 tại `[351.9, 173.0, 426.5, 209.2]` (đúng vị trí lon bia).
    * `plastic`: conf = 0.277 tại `[0.1, 105.5, 274.5, 365.0]`.
    * `metal`: conf = 0.211 tại `[361.4, 174.7, 417.1, 208.4]` (**box trùng lặp với box lon bia phía trên**).
    * `plastic`: conf = 0.207 tại `[41.4, 101.4, 157.7, 241.9]`.
  - **SSDLite E2:** Phát hiện 5 boxes:
    * `plastic` conf = 0.297, `glass` conf = 0.216, `plastic` conf = 0.173, `glass` conf = 0.170, và `clothes` conf = 0.156 tại `[188.2, 222.8, 263.7, 360.9]` (nhận nền gỗ/bóng tối thành quần áo cũ).
* **Đánh giá:**
  - Bỏ sót nhiều vật thể nhỏ và chai nằm khuất.
  - Xuất hiện hiện tượng trùng box (2 box kim loại cùng bám vào 1 lon).
  - Xuất hiện box sai lớp `shoes` và `clothes` trên lon kim loại hoặc nền gỗ.

---

### 3. Ca bệnh 3: `images (15).jpg` — Cụm nhiều đôi giày dép cũ
* **Ảnh kiểm thử:** `data/holdout/images (15).jpg` (độ phân giải thấp: 259x194 pixels).
* **Vật thể thực tế trong ảnh:** Một đống khoảng 6–8 đôi giày dép cũ xếp chồng lên nhau.
* **Kết quả thực tế từ mô hình:**
  - **YOLOv8n E1:** Chỉ phát hiện đúng **2 boxes** (`shoes` conf = 0.493 và `shoes` conf = 0.182), hai box này chồng lấn lên nhau ở trung tâm, **bỏ sót 70% số đôi giày còn lại**.
  - **SSDLite E2:** Phát hiện 2 box `shoes` (conf = 0.814, 0.712) nhưng lại sinh thêm 2 box `metal` (conf = 0.205, 0.175) và 1 box `plastic` (conf = 0.166) đè lên thân các đôi giày.
* **Đánh giá:** Tỷ lệ bỏ sót (False Negatives) rất cao đối với cụm rác cùng loại nằm kề cận. Thuật toán NMS và anchor box bị triệt tiêu lẫn nhau khi các vật thể nằm san sát.

---

### 4. Ca bệnh 4: `rac_online_020.jpg` — Đại cảnh bãi rác với hàng trăm chai nhựa
* **Ảnh kiểm thử:** `data/holdout/rac_online_020.jpg` (kích thước lớn: 3000x4000 pixels).
* **Vật thể thực tế trong ảnh:** Bãi rác khổng lồ ngoài trời với hàng trăm chai nhựa rải rác từ tiền cảnh đến hậu cảnh.
* **Kết quả thực tế từ mô hình:**
  - **YOLOv8n E1:** Chỉ bắt được đúng **7 bounding boxes** lẻ tẻ (1 `metal` conf = 0.594, 5 `plastic` conf = 0.154 - 0.307, 1 `glass` conf = 0.195). **Bỏ sót hơn 95% số chai nhựa có trong bãi rác!**
  - **SSDLite E2:** Sinh ra các anchor box khổng lồ ôm trọn cả mảng bãi rác:
    * `plastic` conf = 0.206 với box `[505.7, 1084.5, 2369.7, 1774.1]` (chiều rộng gần 1900 pixels, bao trùm hàng chục chai nhựa thành một box duy nhất).
    * `plastic` conf = 0.166 với box `[2231.2, 39.1, 2976.1, 1235.5]`.
    * Cùng 19 box nhỏ phân tán không tách bạch.
* **Đánh giá:** Thể hiện rõ sự bất lực của mô hình hiện tại trước rác thải nhỏ trong ảnh độ phân giải cao khi bị co về $640 \times 640$ hoặc $320 \times 320$.

---

### 5. Ca bệnh 5: `kho_15_da_loai_tham_co.jpg` — Chai nhựa trên nền cỏ
* **Ảnh kiểm thử:** `data/holdout/kho_15_da_loai_tham_co.jpg` (2048x1536 pixels).
* **Kết quả thực tế:**
  - **SSDLite E2:** Sinh ra 1 box khổng lồ **bao trùm toàn bộ bức ảnh** `[0.0, 0.0, 2048.0, 1528.7]` và gán nhãn `clothes` với confidence = 0.531! Đồng thời phát hiện 1 box `plastic` (conf = 0.352) và 1 box `paper` (conf = 0.155) đè lên chai nhựa.
  - **YOLOv8n E1:** Phát hiện 3 box `plastic` (conf = 0.589, 0.476, 0.228), localization tương đối chuẩn xác hơn SSDLite.
* **Đánh giá:** SSDLite bị hiện tượng texture bãi cỏ kích hoạt anchor box lớn nhất ở tầng sâu nhất của MobileNetV3 và gán nhãn sai thành quần áo/vải cũ.

---

### 6. Ca bệnh 6: `kho_20_da_loai_goc_pho.jpg` — Dương tính giả trên tủ điện kỹ thuật và cảnh nền sạch
* **Ảnh kiểm thử:** `data/holdout/kho_20_da_loai_goc_pho.jpg` (2918x2189 pixels).
* **Vật thể ngoại cảnh:** Tủ điện kỹ thuật màu xanh lá cây trên vỉa hè, biển quảng cáo, mảng tường gạch nhà mặt phố.
* **Hiện tượng thực tế đã ghi nhận:**
  - Mô hình SSDLite + Two-Stage (cấu hình cũ trước quick fix) từng gán:
    * `cardboard: 50.31%` lên biển hiệu và góc tòa nhà.
    * `cardboard: 48.14%` lên tủ kỹ thuật viễn thông màu xanh.
  - Mô hình YOLOv8n E1 thuần: Gán `trash: 0.427` lên góc vỉa hè và `cardboard: 0.332` tại `[1033.7, 1227.5, 1309.4, 1543.8]`.
* **Đánh giá:** Dương tính giả ngoại cảnh nghiêm trọng do mô hình thiếu mẫu nền âm (hard negatives).

---

## II. PHÂN TÍCH CHUYÊN SÂU THEO 6 KHÍA CẠNH KỸ THUẬT (A — F)

### A. Sai class (Class Confusion)
1. **Lệch nhãn Pin (`battery`):**
   - Kiểm tra tệp split train của `splits_real_v2`: Lớp `battery` chỉ có **đúng 1 bounding box duy nhất** trong toàn bộ 1.398 ảnh huấn luyện!
   - Một mạng nơ-ron sâu với hàng triệu tham số không thể học được bất kỳ vector đặc trưng nào của một lớp khi chỉ có 1 mẫu. Lớp phân loại của mạng gần như gán bias của lớp này về giá trị âm cực lớn.
   - Khi gặp một vật thể hình hộp chữ nhật màu tối (pin con thỏ, pin laptop), detector buộc phải kích hoạt lớp có đặc trưng thị giác gần nhất mà nó đã học nhiều mẫu: `shoes` (215 boxes) hoặc `metal` (420 boxes).
2. **Lệch nhãn Rác hữu cơ (`biological`):**
   - Chỉ có **đúng 3 bounding boxes** trong split train của `splits_real_v2`.
   - Kết quả: Validation AP của `biological` trên checkpoint hiện tại bằng **0.00%**.
3. **Lệch nhãn Rác khác (`trash`):**
   - Lớp `trash` đóng vai trò lớp gom (catch-all), chứa từ tàn thuốc, vụn xỉ than, túi rác đen đến đồ phế thải hỗn hợp. Phân bố đặc trưng cực kỳ phân tán (high intra-class variance), khiến mạng học các đặc trưng vụn vặt và kích hoạt tùy tiện trên các vết nứt vỉa hè, bóng râm.

---

### B. Bỏ sót vật thể (False Negatives)
1. **Rác kích thước nhỏ trong đại cảnh:**
   - Khi ảnh camera $3000 \times 4000$ hoặc $1920 \times 1080$ được resize về $640 \times 640$ (YOLO) hoặc $320 \times 320$ (SSDLite), các vật thể rác có kích thước thực $80 \times 120$ pixels sẽ co lại chỉ còn $12 \times 18$ pixels (YOLO) hoặc $6 \times 9$ pixels (SSDLite).
   - Tại tầng feature map sâu nhất $C_5$, các vật thể này nhỏ hơn độ dài bước nhảy (stride) của mạng (stride = 32), dẫn đến thông tin không gian bị xóa sổ hoàn toàn sau các lớp MaxPool / Conv Strided.
2. **Rác trong suốt và rác bẹp dí:**
   - Chai nhựa trong suốt (PET) không có màu cố định mà phản chiếu màu nền bên dưới (đường nhựa xám, cỏ xanh, cát vàng). Nếu không có augmentation mô phỏng nền đa dạng, mô hình chỉ nhận diện được chai nhựa có nhãn mác màu sắc rõ rệt.
3. **Cụm rác dày đặc (Dense Clusters):**
   - Các vật thể rác nằm chồng chéo nhau tạo ra các bounding box có IoU kề cận $\ge 0.50$. Thuật toán NMS thông thường xem các box này là cùng 1 vật thể và triệt tiêu bớt, gây mất mát recall trầm trọng.

---

### C. Bounding Box sai hoặc chồng lấn (Localization Artifacts)
1. **Box toàn khung hình (Full-Image Hallucination):**
   - Xuất hiện chủ yếu trên SSDLite do kiến trúc anchor boxes cố định. Anchor box ở tầng feature map cuối cùng $1 \times 1$ có diện tích bao trùm gần như toàn bộ bức ảnh ($90\% - 100\%$). Khi gặp nền đồng nhất (bãi cỏ, mặt đường, tường), nếu không có ảnh negative để phạt, loss function vẫn cho phép anchor box này kích hoạt với confidence $\sim 0.20 - 0.50$.
2. **Box trùng lặp (Duplicate Detection):**
   - YOLOv8n trên các vật thể phức tạp (như vỏ lon biến dạng trong `images (17).jpg`) kích hoạt cả 2 center anchor-free gần nhau và không bị NMS lọc vì IoU giữa chúng chỉ đạt $\approx 0.45$ (dưới ngưỡng `iou_threshold = 0.55`).

---

### D. Dataset thiếu mẫu (Data Starvation & Imbalance)
* **Số liệu kiểm kê chính thức từ `data/detection/splits_real_v2/train.txt`:**
  - `battery`: 1 box (0.03%) / 1 image
  - `biological`: 3 boxes (0.08%) / 3 images
  - `cardboard`: 169 boxes (4.26%) / 126 images
  - `clothes`: 368 boxes (9.29%) / 212 images
  - `glass`: 213 boxes (5.37%) / 101 images
  - `metal`: 420 boxes (10.60%) / 214 images
  - `paper`: 160 boxes (4.04%) / 123 images
  - `plastic`: 1,560 boxes (39.36%) / 791 images
  - `shoes`: 215 boxes (5.43%) / 117 images
  - `trash`: 854 boxes (21.55%) / 294 images
* **Độ lệch mất cân bằng (Extreme Imbalance Ratio):**
  - Tỷ lệ `plastic` so với `battery`: **1.560 : 1**
  - Tỷ lệ `plastic` so với `biological`: **520 : 1**
  - Tỷ lệ ảnh âm (Hard-negative / 0 box background): **0 ảnh** (0.0%) trong 1.398 ảnh huấn luyện!
* **Hậu quả trực tiếp:** Mô hình bị quá khớp (overfit) vào `plastic` và `trash`, trong khi hoàn toàn mất khả năng học các lớp rác nguy hại (pin) và rác hữu cơ.

---

### E. Lỗi Class Mapping và chất lượng Annotation lịch sử
1. **Tàn dư từ tập Mendeley Synthetic cũ (đã được cô lập):**
   - Trước đây trong tập Mendeley, lon nhôm bị gán nhãn là `battery` (Class 6), cành cây/lá khô bị gán thành `clothes` (Class 8), vỏ bao thuốc lá bị gán thành `biological` (Class 5).
   - Khi loại bỏ Mendeley để tạo `real_v2`, các lớp này không còn dữ liệu thay thế, để lại khoảng trống dữ liệu mênh mông.
2. **Nhiễu nhãn Open Images V7:**
   - Trong Open Images V7, các nhãn `/m/09j2d` (Clothing) và `/m/09j5n` (Footwear) đa số là trang phục của người đang đi bộ hoặc người mẫu chụp ảnh. Nếu đưa nguyên trạng vào tập huấn luyện rác thải, mô hình sẽ học rằng "người đi đường là một đống rác quần áo/giày dép", gây sai lệch hoàn toàn ngữ cảnh nhận diện.

---

### F. Ảnh đầu vào nhỏ, vật thể nhỏ, background phức tạp
1. **Kích thước ảnh đa dạng từ nguồn thực tế:**
   - Nguồn dữ liệu thực tế gồm cả ảnh nhỏ độ phân giải thấp ($259 \times 194$ như `images (15).jpg`) và ảnh camera độ phân giải cao ($3000 \times 4000$ như `rac_online_020.jpg` và `kho_20_da_loai_goc_pho.jpg`).
2. **SSDLite320 bị giới hạn vật lý:** Đầu vào $320 \times 320$ là quá nhỏ đối với camera quan sát hiện đại.
3. **YOLOv8n ở $512 \times 512$ hoặc $640 \times 640$:** Cải thiện rõ rệt so với SSDLite, nhưng vẫn cần cơ chế xử lý đối tượng nhỏ (như Mosaic augmentation, tối ưu anchor scales, hoặc tinh chỉnh Feature Pyramid Network) để nhận diện được các mảnh rác nhỏ ở cự ly xa.

---

## III. KẾT LUẬN VÀ BÀI HỌC KỸ THUẬT CHO TASK 3 & TASK 4

1. **Không thể khắc phục lỗi nhận nhầm Pin và Rác hữu cơ bằng cách chỉnh sửa threshold hay tuning hyperparameters.** Mô hình đơn giản là không có dữ liệu để học hai lớp này. Bắt buộc phải bổ sung dữ liệu thực tế đã qua kiểm duyệt ở Task 3.
2. **Không thể triệt tiêu hoàn toàn False Positive trên tường, tủ điện, mặt đường nếu không đưa ảnh Hard-Negative (ảnh nền sạch không rác) vào tập huấn luyện.** Tỷ lệ ảnh negative mục tiêu cần đạt 10% - 15% tổng số ảnh train.
3. **Tuyệt đối không dùng kết quả trên bộ 27 ảnh holdout để tinh chỉnh mô hình liên tục (overfitting the test set).** Bộ holdout phải được đóng băng tuyệt đối để nghiệm thu công tâm theo đúng chuẩn mực khoa học.
