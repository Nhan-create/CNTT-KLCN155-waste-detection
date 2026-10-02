# SỔ TAY GIẢI THÍCH CHI TIẾT CÁC THUẬT NGỮ, KIẾN TRÚC VÀ ĐỘ ĐO
## DỰ ÁN KHÓA LUẬN TỐT NGHIỆP: CNTT-KLCN155

> **Tài liệu:** Cẩm nang thuật ngữ khoa học & kỹ thuật phục vụ báo cáo và bảo vệ khóa luận  
> **Đề tài:** *Xây dựng hệ thống phát hiện và phân loại đa đối tượng rác thải sinh hoạt trong ảnh chụp thực tế bằng mô hình học sâu nhẹ và kỹ thuật tăng cường dữ liệu*  
> **Dành cho:** Chủ dự án (PM), Nhóm sinh viên thực hiện và Báo cáo chuyên môn với GVHD  
> **Ngày lập:** 02/10/2026  

---

## MỤC LỤC TỔNG QUAN

1. [Phần 1: Các khái niệm nền tảng trong Thị giác Máy tính & Phát hiện Đối tượng](#phan-1)
   - Bounding Box (Khung bao)
   - Confidence Score (Độ tin cậy)
   - Intersection over Union (IoU)
   - Non-Maximum Suppression (NMS)
2. [Phần 2: Kỹ thuật Hợp nhất Dự đoán cốt lõi — Weighted Boxes Fusion (WBF)](#phan-2)
   - Vì sao NMS truyền thống bị giới hạn?
   - Cơ chế hoạt động của WBF
   - Đóng góp khoa học của WBF trong đề tài
3. [Phần 3: Kỹ thuật Tăng cường Dữ liệu (Data Augmentation)](#phan-3)
   - Thư viện Albumentations
   - Kỹ thuật Copy-Paste Augmentation
   - Ma trận 4 chiến lược tăng cường dữ liệu
4. [Phần 4: Hai Mô hình Học sâu Đối sánh](#phan-4)
   - SSDLite320-MobileNetV3 (Mô hình chính)
   - YOLOv8n (Mô hình đối chứng)
   - Vì sao cần phân định vai trò Chính vs Đối chứng?
5. [Phần 5: Quy trình Huấn luyện & Các Siêu tham số (Hyperparameters)](#phan-5)
   - Học chuyển giao (Transfer Learning)
   - Learning Rate, Batch Size, Epoch, Optimizer, Cosine Scheduler
   - Nguyên tắc chia tập 70/15/15 theo cảnh chụp và Seed cố định (Zero Leakage)
   - Tiêu chí lựa chọn Checkpoint trên Validation Set
6. [Phần 6: Hệ thống Độ đo Đánh giá (Metrics) Chuẩn COCO & Điều kiện Ảnh khó](#phan-6)
   - TP, FP, FN trong bài toán Object Detection
   - Precision, Recall và F1-Score
   - Average Precision (AP) và Mean Average Precision (mAP)
   - Phân biệt mAP@0.5 và mAP@0.5:0.95
   - Đánh giá trên 6 nhóm điều kiện ảnh khó thực tế
   - Đo đạc Latency batch 1, FPS, Kích thước mô hình và Tiêu thụ tài nguyên
   - Bài toán đánh đổi giữa Độ chính xác và Tốc độ (Accuracy vs. Speed Trade-off)
7. [Phần 7: Nghiên cứu Triệt tiêu (Ablation Study)](#phan-7)
   - Khái niệm Ablation Study là gì?
   - Thiết kế ma trận thực nghiệm khoa học
8. [Phần 8: Ứng dụng Web Minh họa & Sản phẩm Bàn giao](#phan-8)
   - Các chức năng bắt buộc của Web Streamlit
   - Danh mục sản phẩm bàn giao cuối cùng & Mở rộng ONNX/TensorFlow Lite

---

<a name="phan-1"></a>
## PHẦN 1: CÁC KHÁI NIỆM NỀN TẢNG TRONG PHÁT HIỆN ĐỐI TƯỢNG (OBJECT DETECTION)

### 1.1. Phân biệt Classification vs. Object Detection
- **Classification (Phân loại ảnh):** Trả lời câu hỏi *"Ảnh này chụp rác gì?"*. Toàn bộ bức ảnh chỉ nhận **một nhãn duy nhất** (ví dụ: ảnh này là "chai nhựa"). Điểm yếu: Nếu trong ảnh có 3 vỏ lon, 2 hộp sữa và 1 vỏ chuối, mô hình phân loại sẽ bối rối hoặc chỉ đoán bừa một nhãn chiếm diện tích lớn nhất.
- **Object Detection (Phát hiện đối tượng):** Trả lời đồng thời hai câu hỏi: *"Rác ở vị trí nào và đó là loại rác gì?"*. Mô hình có thể phát hiện **nhiều vật thể cùng lúc** trong cùng một bức ảnh.

### 1.2. Bounding Box (Khung bao chữ nhật)
- Là hình chữ nhật bao quanh chính xác vùng biên của vật thể rác trong ảnh.
- Có 2 định dạng biểu diễn phổ biến:
  1. **Định dạng góc (Pascal VOC / COCO pixel):** $[x_1, y_1, x_2, y_2]$ với $(x_1, y_1)$ là tọa độ góc trên bên trái, $(x_2, y_2)$ là góc dưới bên phải.
  2. **Định dạng tâm chuẩn hóa (YOLO format):** $[x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$ với các giá trị được chia cho chiều rộng/cao của ảnh để nằm trong khoảng $[0, 1]$.

### 1.3. Confidence Score (Độ tin cậy)
- Là một số thực từ $0.0$ đến $1.0$ (hoặc $0\%$ đến $100\%$) do mô hình tính toán, thể hiện: *"Mô hình tự tin bao nhiêu phần trăm rằng trong khung bao này có chứa vật thể thuộc lớp đó"*.
- **Ngưỡng suy luận (Confidence Threshold):** Bộ lọc do người dùng hoặc kỹ sư thiết lập. Ví dụ đặt ngưỡng là $0.35$ ($35\%$): Bất kỳ dự đoán nào có confidence $< 0.35$ sẽ bị loại bỏ để tránh báo động giả (False Positive).

### 1.4. Intersection over Union (IoU - Tỷ lệ phần giao trên phần hợp)
- Là độ đo toán học dùng để đánh giá mức độ trùng khớp giữa hai khung bao chữ nhật (ví dụ: khung bao do mô hình dự đoán $B_{\text{pred}}$ và khung bao nhãn chuẩn do con người vẽ $B_{\text{gt}}$).
- **Công thức tính:**
  $$\text{IoU} = \frac{\text{Diện tích vùng giao nhau (Intersection)}}{\text{Diện tích vùng hợp nhất (Union)}} = \frac{\text{Area}(B_{\text{pred}} \cap B_{\text{gt}})}{\text{Area}(B_{\text{pred}} \cup B_{\text{gt}})}$$
- **Ý nghĩa:**
  - $\text{IoU} = 0$: Hai khung bao nằm tách rời nhau hoàn toàn, không chạm nhau.
  - $\text{IoU} = 1$: Hai khung bao trùng khít nhau tuyệt đối từng điểm ảnh.
  - $\text{IoU} \ge 0.5$: Thường được coi là dự đoán "trúng đích" ở mức cơ bản.

```
       Khung bao A               Khung bao B
    ┌─────────────┐           ┌─────────────┐
    │             │           │             │
    │        ┌────┼───────────┼────┐        │
    │        │    │ Vùng Giao │    │        │
    │        │    │(Intersect)│    │        │
    └────────┼────┘           └────┼────────┘
             │                     │
             └─────────────────────┘
                 Vùng Hợp (Union)
```

### 1.5. Non-Maximum Suppression (NMS - Triệt tiêu phi cực đại)
- **Vấn đề thực tế:** Khi một mạng nơ-ron nhìn vào một cái chai nhựa, nó có thể vẽ ra 10–20 khung bao quanh cái chai đó với kích thước hơi lệch nhau một chút.
- **Cách NMS xử lý:**
  1. Sắp xếp tất cả các khung bao theo Confidence giảm dần.
  2. Chọn khung bao có Confidence cao nhất làm đại diện.
  3. So sánh IoU của khung bao này với tất cả các khung bao còn lại xung quanh nó.
  4. Nếu $\text{IoU} > \text{threshold}$ (ví dụ $> 0.5$), NMS coi đó là các khung bao vẽ trùng một vật thể và **xóa bỏ thẳng tay (loại trừ)** các khung bao phụ, chỉ giữ lại khung bao có điểm cao nhất.

---

<a name="phan-2"></a>
## PHẦN 2: WEIGHTED BOXES FUSION (WBF) — ĐÓNG GÓP CỐT LÕI CỦA ĐỀ TÀI

### 2.1. Hạn chế của NMS khi kết hợp nhiều mô hình
- Khi áp dụng mô hình kết hợp (Ensemble giữa SSDLite320 và YOLOv8n), NMS bộc lộ điểm yếu lớn:
  - Nếu SSDLite đoán cái chai ở vị trí $A$ với confidence $0.85$, và YOLOv8n đoán cái chai ở vị trí $B$ hơi lệch 2 pixel với confidence $0.84$.
  - NMS sẽ **vứt bỏ hoàn toàn** dự đoán của YOLOv8n và chỉ lấy tọa độ của SSDLite. Điều này lãng phí thông tin quý giá của mô hình thứ hai!
  - Trong tình huống rác thải thực tế nằm chồng lấn sát nhau, NMS rất dễ xóa nhầm khung bao của một vật thể thật nằm đè lên vật thể khác.

### 2.2. Weighted Boxes Fusion (WBF) là gì?
- **Weighted Boxes Fusion (WBF)** là phương pháp hợp nhất khung bao tiên tiến được công bố bởi nhóm tác giả Roman Solovyev, Weimin Wang và Tatiana Gabruseva (năm 2021).
- **Cơ chế hoạt động:** Thay vì vứt bỏ các khung bao trùng nhau như NMS, WBF **hợp nhất (trung bình cộng có trọng số)** tọa độ của các khung bao từ các mô hình khác nhau.
- **Công thức hợp nhất tọa độ:** Khung bao nào có Confidence cao hơn và đến từ mô hình có trọng số uy tín hơn sẽ có "tiếng nói" lớn hơn trong việc quyết định tọa độ cuối cùng:
  $$X_{1,\text{fused}} = \frac{\sum_{i} w_i \cdot c_i \cdot x_{1,i}}{\sum_{i} w_i \cdot c_i}, \quad X_{2,\text{fused}} = \frac{\sum_{i} w_i \cdot c_i \cdot x_{2,i}}{\sum_{i} w_i \cdot c_i}$$
  $$Y_{1,\text{fused}} = \frac{\sum_{i} w_i \cdot c_i \cdot y_{1,i}}{\sum_{i} w_i \cdot c_i}, \quad Y_{2,\text{fused}} = \frac{\sum_{i} w_i \cdot c_i \cdot y_{2,i}}{\sum_{i} w_i \cdot c_i}$$
  Trong đó: $w_i$ là trọng số mô hình (ví dụ $w_{\text{SSD}} = 1.0, w_{\text{YOLO}} = 1.5$), $c_i$ là confidence của từng khung bao.

### 2.3. Đóng góp khoa học của WBF trong đề tài CNTT-KLCN155
1. **Tận dụng tính bù trừ giữa 2 họ kiến trúc khác nhau:** SSDLite320 (Backbone MobileNetV3 với Anchor-based pyramids) có khả năng định vị tốt vật thể nhỏ; YOLOv8n (Anchor-free với C2f modules) có khả năng phân loại vật liệu chính xác hơn. WBF kết hợp điểm mạnh của cả hai.
2. **Tối ưu hóa có kiểm soát trên tập Validation:** Trọng số $w_{\text{SSD}}, w_{\text{YOLO}}$, ngưỡng IoU ghép cụm và ngưỡng confidence đầu vào được tìm kiếm bằng **Grid Search (tìm kiếm lưới)** trên tập Validation để đạt mAP@0.5:0.95 cao nhất. Sau đó cố định tham số này và đánh giá một lần duy nhất trên tập Test độc lập để đảm bảo tính khách quan khoa học.

---

<a name="phan-3"></a>
## PHẦN 3: KỸ THUẬT TĂNG CƯỜNG DỮ LIỆU (DATA AUGMENTATION)

### 3.1. Thư viện Albumentations là gì?
- Trong bài toán phân loại ảnh thông thường, bạn chỉ cần xoay ảnh, lật ảnh là xong.
- Nhưng trong **Phát hiện đối tượng (Object Detection)**, khi bạn xoay bức ảnh $90^\circ$ hoặc cắt xén (crop), tọa độ của các khung bao bounding box $[x_1, y_1, x_2, y_2]$ **bắt buộc phải được biến đổi hình học đồng bộ tương ứng**. Nếu không, cái chai nhựa quay sang phải nhưng khung bao vẫn đứng yên ở giữa ảnh $\to$ hỏng toàn bộ dữ liệu huấn luyện!
- **Albumentations** là thư viện mã nguồn mở chuẩn công nghiệp mạnh mẽ nhất hiện nay trong Python, cho phép biến đổi đồng thời cả ma trận điểm ảnh RGB lẫn danh sách tọa độ bounding box một cách chính xác tuyệt đối và tốc độ cực nhanh (nhờ tối ưu C++ qua OpenCV).

### 3.2. Kỹ thuật Copy-Paste Augmentation
- Được phát triển bởi nhóm nghiên cứu của Google Brain (Golnaz Ghiasi et al., CVPR 2021).
- **Vấn đề thực tế của dữ liệu rác:** Đa số ảnh rác tự chụp hoặc trong studio chỉ có 1–2 vật thể nằm riêng lẻ trên nền sạch. Mô hình học xong sẽ không biết phát hiện rác khi chúng nằm thành đống lộn xộn hoặc che khuất nhau.
- **Cách thức Copy-Paste hoạt động:**
  1. Trích xuất một vật thể rác (kèm mặt nạ tiền cảnh - foreground mask bóc tách sát biên) từ một ảnh này (ví dụ: một chiếc vỏ lon nhôm).
  2. Áp dụng biến đổi ngẫu nhiên (thu nhỏ, xoay nhẹ).
  3. Dán (paste) vật thể đó đè lên phông nền hoặc đè lên một vật thể rác khác trong một bức ảnh khác (ví dụ dán đè lên một đống giấy vụn).
  4. **Tính toán lại bounding box:** Tự động xác định khung bao mới ôm sát vật thể vừa dán và cập nhật lại khung bao của vật thể bị che khuất một phần bên dưới.
- **Lợi ích:** Tạo ra hàng nghìn bức ảnh mô phỏng tình huống rác thải thực tế (nhiều vật thể, rác nằm chồng đè, che khuất một phần) mà không cần tốn công sức đi chụp hàng tháng trời.

### 3.3. Ma trận 4 Chiến lược Augmentation trong đề tài
Để chứng minh đóng góp khoa học trong khóa luận (CLO 3), đề tài tổ chức thực nghiệm 4 chiến lược tăng cường có kiểm soát:
1. **Chiến lược 1 (Baseline - Không tăng cường):** Chỉ resize ảnh về kích thước chuẩn và chuẩn hóa điểm ảnh (không xoay, không lật, không đổi màu).
2. **Chiến lược 2 (Geometric - Tăng cường Hình học):** Lật ngang (Horizontal Flip), lật dọc (Vertical Flip), xoay ngẫu nhiên góc hẹp (Random Rotate), co giãn tỷ lệ (Random Scale), dịch chuyển (Translation).
3. **Chiến lược 3 (Photometric - Tăng cường Quang học/Ánh sáng):** Thay đổi độ sáng (Brightness), độ tương phản (Contrast), độ bão hòa màu (Saturation), đổi không gian màu (Hue), mô phỏng thiếu sáng (Shadow/Darkness), thêm nhiễu hạt (Gaussian Noise), làm mờ chuyển động (Motion Blur).
4. **Chiến lược 4 (Combined - Kết hợp Toàn diện):** Kết hợp cả Hình học + Ánh sáng + Copy-Paste ghép vật thể đa rác.

---

<a name="phan-4"></a>
## PHẦN 4: HAI MÔ HÌNH HỌC SÂU ĐỐI SÁNH TRONG ĐỀ TÀI

### 4.1. Mô hình Chính: SSDLite320-MobileNetV3
- **Kiến trúc:** Gồm mạng trích xuất đặc trưng (Backbone) là **MobileNetV3-Large** kết hợp với đầu dò một giai đoạn **SSDLite** (Single Shot MultiBox Detector phiên bản thu gọn).
- **Đặc điểm kỹ thuật:**
  - Sử dụng các lớp tích chập tách biệt theo chiều sâu (Depthwise Separable Convolutions) thay cho tích chập thông thường, giúp giảm $80–90\%$ số lượng phép tính nhân cộng (FLOPs).
  - Tích hợp cơ chế Squeeze-and-Excitation (SE) giúp mạng tập trung vào các kênh đặc trưng quan trọng của rác thải.
  - Kích thước ảnh đầu vào chuẩn hóa: $320 \times 320$ pixels.
- **Vì sao là mô hình chính:** Đề tài mang tên *"mô hình học sâu nhẹ"*. SSDLite320-MobileNetV3 là đại diện tiêu biểu nhất cho họ mô hình nhẹ hướng đến thiết bị nhúng (Raspberry Pi, điện thoại di động, máy quét rác mini). Việc kế thừa trọng số từ MobileNetV3 phân loại ở Giai đoạn 1 sang SSDLite320 ở Giai đoạn 2 tạo nên tính liền mạch học thuật xuyên suốt cho khóa luận.

### 4.2. Mô hình Đối chứng: YOLOv8n (YOLOv8 Nano)
- **Kiến trúc:** Phiên bản nhỏ nhất (Nano) trong dòng họ YOLOv8 của hãng Ultralytics (công bố năm 2023).
- **Đặc điểm kỹ thuật:**
  - Kiến trúc Anchor-free (không phụ thuộc vào khung neo cố định trước), giúp mô hình linh hoạt hơn khi dự đoán các hình dạng rác bất quy tắc (túi ni lông rách, vỏ chai méo mó).
  - Sử dụng module C2f (Cross-Stage Partial with two convolutions) tối ưu luồng truyền gradient.
  - Kích thước ảnh đầu vào: $640 \times 640$ (hoặc $320 \times 320$ khi so sánh công bằng với SSDLite).
- **Vai trò trong đề tài:** Đóng vai trò **Mô hình đối chứng (Benchmark/Baseline)**. Trong nghiên cứu khoa học, để khẳng định mô hình của mình (hoặc phương pháp WBF của mình) tốt hay dở, bắt buộc phải đặt cạnh một mô hình tiêu chuẩn hiện đại nhất thế giới ở cùng phân khúc nhẹ để so sánh sòng phẳng về mAP, kích thước và tốc độ.

---

<a name="phan-5"></a>
## PHẦN 5: QUY TRÌNH HUẤN LUYỆN VÀ CÁC SIÊU THAM SỐ (HYPERPARAMETERS)

### 5.1. Học chuyển giao (Transfer Learning)
- Huấn luyện một mạng nơ-ron sâu từ đầu (từ các con số ngẫu nhiên - scratch) đòi hỏi hàng triệu bức ảnh và siêu máy tính chạy cả tuần.
- **Học chuyển giao:** Sử dụng một mô hình đã được các tập đoàn lớn (Google, Microsoft) huấn luyện sẵn trên tập dữ liệu khổng lồ **ImageNet-1K** (1,2 triệu ảnh) hoặc **MS COCO** (330.000 ảnh). Mô hình này đã có sẵn khả năng nhận biết đường nét, góc cạnh, màu sắc và bề mặt vật liệu. Nhóm chỉ cần giữ lại các tầng nhận diện đặc trưng đó và huấn luyện tinh chỉnh (fine-tune) lại các tầng cuối cùng trên tập dữ liệu rác thải của mình.

### 5.2. Các siêu tham số cốt lõi
- **Learning Rate (Tốc độ học - $lr$):** Bước nhảy của thuật toán tối ưu hóa khi cập nhật trọng số mạng.
  - Nếu $lr$ quá lớn: Mô hình học không ổn định, nhảy loạn xạ và không thể hội tụ (Loss tăng vọt hoặc NaN).
  - Nếu $lr$ quá nhỏ: Mô hình học cực kỳ chậm, dễ bị mắc kẹt tại điểm tối ưu cục bộ.
  - Trong dự án: Áp dụng chiến lược **Cosine Annealing Learning Rate Scheduler** — ban đầu cho $lr$ lớn vừa phải để học nhanh, sau đó giảm dần theo đồ thị hàm cosin về rất nhỏ ở các epoch cuối để mô hình ổn định lại ở điểm tối ưu tốt nhất.
- **Batch Size (Kích thước lô):** Số lượng bức ảnh được đưa vào GPU để tính toán trong một lượt truyền thuận/nghịch. Đề tài dùng Batch size 16 hoặc 32 để tối ưu hóa bộ nhớ 4 GB VRAM của card đồ họa NVIDIA RTX 2050 mà không bị tràn bộ nhớ (Out-Of-Memory - OOM).
- **Epoch (Chu kỳ huấn luyện):** Một epoch được tính là khi toàn bộ tập dữ liệu huấn luyện đã được mạng nơ-ron nhìn qua đúng 1 lần. Đề tài cấu hình tối đa 50–120 epochs, kết hợp cơ chế dừng sớm **Early Stopping** (nếu sau 10–30 epochs mà điểm validation không tăng nữa thì tự động ngắt để tránh lãng phí điện năng và tránh Overfitting).
- **Optimizer (Thuật toán tối ưu):** Công cụ toán học cập nhật trọng số:
  - **SGD (Stochastic Gradient Descent với Momentum 0.937):** Thường dùng cho YOLO, tạo ra khả năng khái quát hóa rất tốt.
  - **AdamW (Adaptive Moment Estimation with Decoupled Weight Decay):** Thường dùng cho MobileNetV3 / SSDLite, giúp hội tụ nhanh và kiểm soát hiện tượng quá khớp (Overfitting) bằng hình phạt trọng số.

### 5.3. Nguyên tắc Chia tập dữ liệu 70/15/15 và Seed cố định
- **Tỷ lệ chia chuẩn:** 70% tập Huấn luyện (Train), 15% tập Xác thực (Validation), 15% tập Kiểm tra độc lập (Test).
- **Chia theo cảnh chụp (Scene-based Split / Group Split):**
  - *Rủi ro rò rỉ dữ liệu (Data Leakage):* Nếu bạn chụp 1 đống rác ở 5 góc máy khác nhau (burst shot), rồi để 4 ảnh ở tập Train và 1 ảnh ở tập Test. Khi đó mô hình thi đậu tập Test vì nó đã "học vẹt" đống rác đó ở tập Train, chứ không phải vì nó thông minh!
  - *Giải pháp của đề tài:* Dùng giải thuật Union-Find gom toàn bộ các ảnh chụp cùng một cảnh hoặc các ảnh có độ tương đồng băm nhận thức ($\text{pHash} \le 4$) thành các cụm nguyên tử (Atomic Clusters). Mọi ảnh trong cùng một cụm **bắt buộc phải nằm trọn vẹn trong một tập duy nhất**.
- **Cố định Seed (ví dụ `seed=42`):** Cố định bộ sinh số ngẫu nhiên của Python, NumPy và PyTorch để bất kỳ ai chạy lại mã nguồn cũng chia ra tập train/val/test giống hệt 100%, bảo đảm tính tái lập khoa học (Reproducibility).
- **Tiêu chí chọn Checkpoint:** Tuyệt đối không dùng mắt nhìn hay tập Test để chọn mô hình tốt nhất (`best.pt`). Mô hình tốt nhất phải được hệ thống tự động lưu dựa trên chỉ số cao nhất đo được trên **Tập Validation**. Tập Test được đóng băng hoàn toàn và chỉ mở ra đúng một lần duy nhất ở cuối đề tài để ghi số liệu báo cáo.

---

<a name="phan-6"></a>
## PHẦN 6: HỆ THỐNG ĐỘ ĐO ĐÁNH GIÁ (METRICS) CHUẨN COCO VÀ ĐIỀU KIỆN ẢNH KHÓ

### 6.1. Định nghĩa TP, FP, FN trong Phát hiện Đối tượng
Trong bài toán Object Detection, một dự đoán được tính là Đúng hay Sai không chỉ dựa vào tên lớp mà phụ thuộc chặt chẽ vào **Tọa độ khung bao (IoU)** và **Độ tin cậy (Confidence)**:
- **True Positive (TP - Bắt đúng):** Mô hình dự đoán khung bao có nhãn đúng và $\text{IoU} \ge \text{threshold}$ (ví dụ $\ge 0.5$) so với nhãn chuẩn thật của con người.
- **False Positive (FP - Bắt sai / Báo động giả):** Mô hình vẽ một khung bao rác vào chỗ không có rác (ví dụ nhìn cái bóng râm hoặc viên gạch lại đoán là hộp sữa), hoặc đoán sai lớp vật liệu, hoặc khung bao bị lệch quá nhiều ($\text{IoU} < 0.5$).
- **False Negative (FN - Bỏ sót rác):** Có một vật thể rác thật trên mặt đất nhưng mô hình không vẽ được khung bao nào hoặc điểm confidence quá thấp nên bị bộ lọc loại bỏ.

```
                  ┌───────────────────────────────┐
                  │       Vật thể Thực tế         │
                  ├───────────────┬───────────────┤
                  │     Có rác    │   Không rác   │
┌────────┬────────┼───────────────┼───────────────┤
│Mô hình │Có rác  │ True Positive │False Positive │
│Dự đoán │(Detect)│  (TP: Đúng)   │ (FP: Báo giả) │
│        ├────────┼───────────────┼───────────────┤
│        │Không có│False Negative │ True Negative │
│        │(Miss)  │ (FN: Bỏ sót)  │  (TN: Bỏ qua) │
└────────┴────────┴───────────────┴───────────────┘
```

### 6.2. Precision, Recall và F1-Score
- **Precision (Độ chuẩn xác):** Trong tất cả các vật thể mà mô hình hô lên là "rác", có bao nhiêu phần trăm thực sự là rác?
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *Ý nghĩa thực tế:* Precision cao nghĩa là mô hình rất chắc chắn, ít khi báo động giả làm người dọn vệ sinh đi nhặt nhầm gạch đá.
- **Recall (Độ thu hồi / Độ bao phủ):** Trong tất cả các vật thể rác đang có ngoài đời thực, mô hình đã tìm ra và bắt được bao nhiêu phần trăm?
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Ý nghĩa thực tế:* Recall cao nghĩa là mô hình mắt rất sáng, ít khi bỏ sót rác nằm trên đường phố.
- **F1-Score:** Trung bình điều hòa giữa Precision và Recall. Được dùng làm mốc so sánh khi muốn cả hai chỉ số đều cân bằng:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 6.3. Average Precision (AP) và Mean Average Precision (mAP)
- Trong thực tế, nếu bạn hạ ngưỡng confidence xuống cực thấp (ví dụ $0.05$), Recall sẽ rất cao (bắt hết rác) nhưng Precision sẽ rất tệ (báo động giả đầy đường). Ngược lại nếu đẩy confidence lên $0.95$, Precision rất cao nhưng Recall rất thấp (bỏ sót nhiều rác).
- Đồ thị biểu diễn mối quan hệ giữa Precision và Recall khi quét ngưỡng confidence từ $0$ đến $1$ gọi là **Đường cong Precision-Recall (PR Curve)**.
- **AP (Average Precision - Độ chính xác trung bình của 1 lớp):** Là diện tích nằm dưới đường cong PR của lớp đó (tính bằng tích phân). $\text{AP}$ tóm tắt toàn diện năng lực nhận diện của mô hình trên lớp rác đó ở mọi ngưỡng tự tin.
- **mAP (Mean Average Precision - Độ chính xác trung bình toàn bộ các lớp):** Là trung bình cộng chỉ số AP của tất cả các lớp rác trong hệ thống:
  $$\text{mAP} = \frac{1}{K} \sum_{i=1}^{K} \text{AP}_i \quad (K \text{ là tổng số lớp, ví dụ 6 hoặc 10 lớp})$$

### 6.4. Phân biệt mAP@0.5 và mAP@0.5:0.95 (Chuẩn COCO)
- **mAP@0.5 (hoặc mAP50):** Chỉ số tính tại ngưỡng khắt khe $\text{IoU} = 0.5$. Chỉ cần khung bao dự đoán trúng $50\%$ diện tích khung bao thật là được tính là đúng (TP). Thước đo này đánh giá khả năng **phát hiện sơ bộ vật thể**.
- **mAP@0.5:0.95 (hoặc COCO mAP):** Thước đo khắt khe và uy tín nhất thế giới trong các hội nghị thị giác máy tính đỉnh cao (CVPR, ICCV, ECCV).
  - Thuật toán tính mAP tại 10 mức ngưỡng IoU khác nhau: $0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95$.
  - Sau đó lấy trung bình cộng của 10 giá trị này lại.
  - *Ý nghĩa:* Để đạt điểm cao ở mAP@0.5:0.95, mô hình không chỉ đoán đúng loại rác mà **khung bao vẽ ra phải ôm khít từng milimet** vật thể rác ngoài đời thực.

### 6.5. Đánh giá riêng trên 6 Nhóm Điều kiện Ảnh Khó
Đề cương yêu cầu không chỉ báo cáo một con số chung chung, mà phải phân tích điểm mạnh/yếu của mô hình dưới các thử thách môi trường:
1. **Thiếu sáng (Low-light):** Ảnh chụp ban đêm, trong góc khuất, bóng cây râm.
2. **Ngược sáng (Backlight):** Ảnh chụp đối diện ánh mặt trời chói chang làm vật thể rác bị tối đen.
3. **Nền phức tạp (Complex Background):** Rác nằm trên thảm cỏ rậm, đất đá gồ ghề, sàn gạch nhiều hoa văn gây rối mắt.
4. **Vật thể nhỏ (Small Objects):** Rác có kích thước $< 32 \times 32$ pixels trong ảnh (đầu lọc thuốc lá, nắp chai, pin cúc áo).
5. **Che khuất (Occlusion):** Vật thể rác bị lá cây hoặc chướng ngại vật che mất một nửa.
6. **Chồng lấn (Overlap):** Nhiều vỏ lon, chai nhựa nằm đè lên nhau thành cụm.

### 6.6. Đo đạc Tài nguyên và Độ trễ Suy luận (Runtime Profiling)
- **Độ trễ suy luận (Inference Latency với Batch size 1):** Thời gian tính bằng mili-giây (ms) để hệ thống nhận 1 bức ảnh vào, nạp qua mô hình và trả về tọa độ khung bao. Phải đo với batch size = 1 để phản ánh trải nghiệm người dùng thực tế khi chụp từng tấm ảnh bằng điện thoại.
- **FPS (Frames Per Second - Số khung hình trên giây):**
  $$\text{FPS} = \frac{1000}{\text{Latency (ms)}}$$
  Ví dụ latency là $25\text{ ms} \implies \text{FPS} = 40$ (đủ nhanh để xử lý video mượt mà).
- **Kích thước mô hình (Model Size):** Dung lượng file trọng số tính bằng Megabytes (MB). Mô hình càng nhẹ (nhỏ hơn 20 MB) càng dễ tải và nhúng vào bộ nhớ hạn chế của vi điều khiển / điện thoại.
- **Tiêu thụ tài nguyên:** Lượng RAM (bộ nhớ hệ thống) và VRAM (bộ nhớ card đồ họa) bị chiếm dụng khi mô hình đang hoạt động.

### 6.7. Bài toán Đánh đổi giữa Độ chính xác và Tốc độ (Accuracy vs. Speed Trade-off)
- Trong kỹ thuật AI, không có mô hình nào là hoàn hảo tuyệt đối:
  - Muốn độ chính xác cực cao (mAP cao) $\implies$ Mạng phải sâu, nhiều tham số $\implies$ Chạy chậm, tốn pin, dung lượng nặng, máy nóng.
  - Muốn tốc độ cực nhanh (Latency thấp, FPS cao) $\implies$ Mạng phải nông, tỉa bớt lớp $\implies$ Dễ bỏ sót rác nhỏ hoặc đoán nhầm rác phức tạp.
- Nhiệm vụ của khóa luận là phân tích và tìm ra **điểm cân bằng tối ưu** (Pareto Frontier) giữa SSDLite320 (siêu nhẹ, tiết kiệm tài nguyên) và YOLOv8n (chính xác hơn nhưng cần cấu hình mạnh hơn), cũng như đánh giá xem việc dùng WBF có đáng để đánh đổi thêm một chút độ trễ hay không.

---

<a name="phan-7"></a>
## PHẦN 7: NGHIÊN CỨU TRIỆT TIÊU (ABLATION STUDY)

### 7.1. Ablation Study là gì?
- *"Ablation"* trong tiếng Anh chuyên ngành có gốc từ sinh học/y học (phẫu thuật cắt bỏ một bộ phận để xem cơ thể bị ảnh hưởng thế nào).
- Trong nghiên cứu Học sâu, **Ablation Study (Nghiên cứu triệt tiêu / Phân tích đóng góp từng thành phần)** là phương pháp khoa học nhằm kiểm chứng xem: **"Liệu các cải tiến mà bạn đề xuất (như Augmentation hay WBF) có thực sự mang lại hiệu quả hay chỉ là sự may rủi ngẫu nhiên?"**
- Bạn làm việc này bằng cách: Tắt bỏ từng thành phần một, giữ nguyên toàn bộ các điều kiện khác, và đo lường sự thay đổi của các con số mAP.

### 7.2. Thiết kế Bảng Ablation trong Khóa luận
Khóa luận sẽ xây dựng bảng đối sánh gồm 8 thí nghiệm huấn luyện kết hợp với 1 thí nghiệm WBF:

| Thí nghiệm | Mô hình | Chiến lược Augmentation | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | Kích thước (MB) | Đóng góp rút ra |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---|
| **EXP 1** | SSDLite320 | None (Gốc) | *Đo đạc* | *Đo đạc* | ~12 ms | ~17 MB | Mốc cơ sở của SSDLite |
| **EXP 2** | SSDLite320 | Geometric (Hình học) | *Đo đạc* | *Đo đạc* | ~12 ms | ~17 MB | Đánh giá hiệu quả của xoay/lật |
| **EXP 3** | SSDLite320 | Photometric (Quang học)| *Đo đạc* | *Đo đạc* | ~12 ms | ~17 MB | Đánh giá hiệu quả đổi sáng/nhiễu |
| **EXP 4** | SSDLite320 | Combined (+ Copy-Paste)| *Đo đạc* | *Đo đạc* | ~12 ms | ~17 MB | Đánh giá hiệu quả của Copy-Paste |
| **EXP 5** | YOLOv8n | None (Gốc) | *Đo đạc* | *Đo đạc* | ~18 ms | ~6.5 MB | Mốc cơ sở của YOLOv8n |
| **EXP 6** | YOLOv8n | Geometric (Hình học) | *Đo đạc* | *Đo đạc* | ~18 ms | ~6.5 MB | Đánh giá phản ứng của YOLO |
| **EXP 7** | YOLOv8n | Photometric (Quang học)| *Đo đạc* | *Đo đạc* | ~18 ms | ~6.5 MB | Đánh giá phản ứng của YOLO |
| **EXP 8** | YOLOv8n | Combined (+ Copy-Paste)| *Đo đạc* | *Đo đạc* | ~18 ms | ~6.5 MB | Phiên bản tốt nhất của YOLOv8n |
| **EXP 9** | **WBF Ensemble** | Best SSDLite + Best YOLO | **Cao nhất** | **Cao nhất** | ~35 ms | ~23.5 MB | **Đóng góp cốt lõi của đề tài** |

---

<a name="phan-8"></a>
## PHẦN 8: ỨNG DỤNG WEB MINH HỌA VÀ SẢN PHẨM BÀN GIAO

### 8.1. Các chức năng bắt buộc của Ứng dụng Web (Streamlit)
Theo đúng yêu cầu của Đề cương chi tiết (Đoạn 30/45) và Thang điểm Rubric CLO 3 (chiếm 4.0 điểm):
1. **Tải lên một hoặc nhiều ảnh cùng lúc:** Người dùng có thể kéo thả 5–10 tấm ảnh rác khác nhau vào giao diện một lượt.
2. **Hiển thị trực quan khung bao và nhãn:** Vẽ hộp màu phân biệt bao quanh từng vật thể rác, ghi rõ tên loại rác và điểm confidence (ví dụ: `plastic 92%`).
3. **Thống kê số lượng phát hiện theo nhóm:** Bảng tổng kết rõ ràng trong ảnh có bao nhiêu vỏ chai nhựa, bao nhiêu giấy vụn, bao nhiêu pin nguy hại.
4. **Điều chỉnh ngưỡng tin cậy (Slider Confidence):** Cho phép người dùng kéo thanh trượt từ $0.0$ đến $1.0$ để xem kết quả lọc rác thay đổi trực tiếp.
5. **Hiển thị thời gian xử lý (Processing Latency):** Hiển thị rõ ràng mô hình mất bao nhiêu mili-giây (ví dụ: `28 ms`) để xử lý tấm ảnh đó.
6. **Thông báo khi không có rác:** Nếu đưa vào ảnh sàn nhà sạch sẽ hoặc bãi biển không có rác, giao diện phải hiển thị thông báo trung thực: *"Không phát hiện vật thể rác đạt ngưỡng tin cậy trong ảnh này"* (thay vì cố tình đoán bừa một vật thể ngoài ý muốn).
7. **Tải xuống kết quả:** Nút bấm cho phép tải từng ảnh đã vẽ khung bao hoặc tải một file `.zip` chứa toàn bộ ảnh kết quả.
8. **Chỉ dẫn phân loại thùng rác:** Đưa ra khuyến nghị màu thùng rác phù hợp cho người dùng (Ví dụ: Thùng Xanh - Hữu cơ; Thùng Vàng - Tái chế nhựa/giấy/kim loại; Thùng Đỏ - Pin nguy hại).

### 8.2. Sản phẩm Bàn giao Cuối cùng & Định dạng Xuất mở rộng (ONNX / TensorFlow Lite)
- **Quyển Báo cáo khóa luận:** Định dạng chuẩn theo quy định của Khoa CNTT – HUIT, phản ánh trung thực toàn bộ cơ sở lý thuyết, quá trình xử lý dữ liệu và bảng kết quả thực nghiệm.
- **Bộ dữ liệu chuẩn hóa:** Kèm nhãn bounding box, quy tắc ánh xạ và file manifest phân chia tập không rò rỉ.
- **Mã nguồn hoàn chỉnh:** Các script huấn luyện, kiểm toán, đánh giá mAP, thuật toán WBF và ứng dụng Web Streamlit.
- **Tập trọng số tối ưu (Best Weights):** Checkpoint `.pt` hoặc `.pth` có điểm mAP cao nhất.
- **Nội dung mở rộng (ONNX / TensorFlow Lite):**
  - **ONNX (Open Neural Network Exchange):** Chuẩn mở quốc tế giúp chuyển đổi mô hình từ PyTorch sang định dạng trung gian có thể chạy siêu tốc trên CPU thông qua thư viện ONNX Runtime (đã thực nghiệm thành công với MobileNetV3 đạt 7.40 ms).
  - **TensorFlow Lite (TFLite):** Định dạng nén mô hình của Google chuyên dụng cho các ứng dụng chạy trực tiếp trên chip điện thoại Android hoặc bo mạch Raspberry Pi trong các thùng rác thông minh thực tế.

---

## LỜI KẾT DÀNH CHO PM VÀ NHÓM SINH VIÊN

Khi nắm vững 8 phần nội dung trong sổ tay này:
1. Bạn hoàn toàn tự tin khi bảo vệ trước Hội đồng chấm khóa luận tốt nghiệp HUIT: giải thích rành mạch vì sao chọn mô hình nhẹ SSDLite320, vai trò đối chứng của YOLOv8n, cách tính mAP@0.5:0.95 và cơ chế toán học của WBF.
2. Bạn có bức tranh kiến trúc tổng thể rõ ràng để cùng Tech Lead triển khai các subtasks tiếp theo mà không sợ bị lạc hướng học thuật.
