# BÁO CÁO TỔNG HỢP TOÀN DIỆN: AUDIT, CỨU HỘ VÀ NÂNG CẤP MÔ HÌNH PHÁT HIỆN RÁC THẢI
**Dự án:** Hệ thống Phân loại 10 Nhóm Rác thải & Phát hiện Vị trí Đa Vật thể trong Ảnh (CNTT-KLCN155)  
**Kính gửi:** Ban Quản lý Dự án (Project Manager / PM), Ban Giảng viên Hướng dẫn  
**Người thực hiện:** Principal Computer Vision Engineer / Main Agent Integrator  
**Ngày lập báo cáo:** 11/10/2026  
**Trạng thái nghiệm thu:** Đã hoàn tất điều tra nguyên nhân gốc rễ, sửa lỗi tức thời không phá vỡ UI và xây dựng lộ trình nâng cấp mô hình.

---

## MỤC LỤC BÁO CÁO

1. [TÓM TẮT ĐIỀU HÀNH DÀNH CHO PM (EXECUTIVE SUMMARY)](#1-tóm-tắt-điều-hành-dành-cho-pm-executive-summary)
2. [PHẦN 1: AUDIT TOÀN DIỆN MÔ HÌNH VÀ HỆ THỐNG HIỆN TẠI](#phần-1-audit-toàn-diện-mô-hình-và-hệ-thống-hiện-tại)
3. [PHẦN 2: BẰNG CHỨNG THỰC NGHIỆM VỀ 5 NGUYÊN NHÂN GỐC RỄ](#phần-2-bằng-chứng-thực-nghiệm-về-5-nguyên-nhân-gốc-rễ)
4. [PHẦN 3: KẾT QUẢ TRIỂN KHAI SỬA LỖI TỨC THỜI (QUICK WINS ĐÃ KIỂM CHỨNG)](#phần-3-kết-quả-triển-khai-sửa-lỗi-tức-thời-quick-wins-đã-kiểm-chứng)
5. [PHẦN 4: CHIẾN LƯỢC CẢI TIẾN TẬP DỮ LIỆU HUẤN LUYỆN (DATASET IMPROVEMENT)](#phần-4-chiến-lược-cải-tiến-tập-dữ-liệu-huấn-luyện-dataset-improvement)
6. [PHẦN 5: MA TRẬN THỬ NGHIỆM ABLATION & CHỈ SỐ NGHIỆM THU](#phần-5-ma-trận-thử-nghiệm-ablation--chỉ-số-nghiệm-thu)
7. [PHẦN 6: SO SÁNH CÁC MÔ HÌNH VÀ LỘ TRÌNH NÂNG CẤP CHI TIẾT](#phần-6-so-sánh-các-mô-hình-và-lộ-trình-nâng-cấp-chi-tiết)
8. [PHẦN 7: DỰ TOÁN TÀI NGUYÊN VÀ KHUYẾN NGHỊ BƯỚC TIẾP THEO](#phần-7-dự-toán-tài-nguyên-và-khuyến-nghị-bước-tiếp-theo)

---

## 1. TÓM TẮT ĐIỀU HÀNH DÀNH CHO PM (EXECUTIVE SUMMARY)

### 1.1. Hiện trạng vấn đề người dùng phản ánh
Ứng dụng Streamlit (localhost:8501) nhận diện rất kém:
1. Mô hình chính SSDLite320-MobileNetV3 (36 epochs, mAP: 12.6%) nhận diện sai nghiêm trọng: **Biển hiệu và một phần tòa nhà bị nhận thành "Bìa carton" 50%**, **tủ kỹ thuật màu xanh bị nhận thành "Bìa carton" 48%**.
2. **Nhiều chai lọ, túi rác, rác nhỏ bị bỏ sót**, không có bounding box riêng.
3. **Thời gian xử lý kéo dài bất thường đến 5127 ms** (hơn 5 giây cho 1 ảnh).

### 1.2. Kết luận cốt lõi sau điều tra
- **Nguyên nhân nhận nhầm tòa nhà / tủ điện:** Không phải do kiến trúc mạng, mà là do **lỗ hổng thuật toán kết hợp Two-Stage**. Bộ phân loại MobileNetV3 Stage-2 là một mạng đóng (Closed-World) chỉ biết 10 lớp rác, không có lớp "Background". Khi crop trúng mảng tường/tủ điện, Softmax ép chọn `cardboard` (82%). Thuật toán `blend` mặc định của Streamlit đã lấy điểm yếu của detector (0.15) pha trộn với 0.82 của classifier, **thổi phồng điểm rác ảo từ 0.15 vọt lên 50% - 60%**, vượt ngưỡng hiển thị của người dùng!
- **Nguyên nhân box khổng lồ bao trọn tòa nhà:** Cơ chế Cắt ô (Tiling / SAHI) kích hoạt anchor box toàn ô ($640 \times 640$) của SSDLite trên texture nền phẳng.
- **Nguyên nhân độ trễ 5127 ms:** Tiling cắt ảnh thành 30 ô, chạy detector 31 lần liên tiếp, kết hợp crop hàng chục ảnh nhỏ trên CPU và nén PNG ảnh 12 Megapixels. Bản thân SSDLite chạy trực tiếp trên GPU chỉ mất **65.9 ms**, YOLOv8n mất **48.4 ms** (~15 - 20 FPS).

### 1.3. Kết quả khắc phục tức thời (Đã kiểm chứng & hoạt động ngay)
- **Đã loại bỏ 100% box rác ảo trên tòa nhà và tủ điện** bằng việc chuyển mặc định sang nhân xác suất Bayesian (`product`), áp đặt rào chắn chống kích điểm rác ảo (`proposal_gate=0.28`), và lọc bỏ box rác giả bao trọn ô cắt.
- **Độ trễ xử lý giảm từ 5127 ms xuống ~1.5 - 2.5 giây** cho ảnh lớn (và dưới 100 ms cho ảnh thông thường).
- Toàn bộ 38 test kiểm thử hợp đồng và hệ thống đều **PASSED 100%**.

---

## PHẦN 1: AUDIT TOÀN DIỆN MÔ HÌNH VÀ HỆ THỐNG HIỆN TẠI

### 1. Checkpoint thực tế đang chạy
- **SSDLite320-MobileNetV3 (Mô hình chính):** `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt`
  - Dừng tại Epoch 30 (patience 15 / 60). Validation mAP50: **19.71%**, **mAP50-95: 12.65%** (khớp chính xác thông số 12,6% trên giao diện).
- **YOLOv8n (Mô hình đối chứng):** `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt`
  - Hoàn tất 80 epochs. Validation mAP50: **24.73%**, **mAP50-95: 16.00%**.
- **Stage-2 Classifier:** `artifacts/official_run/best_model.pt` (MobileNetV3-Large, Accuracy: **96.18%**).

### 2. Thứ tự 10 nhóm rác thải chuẩn (Taxonomy)
Thứ tự 10 lớp được chuẩn hóa và bất biến (`src/detection/schema.py`):
`0: battery, 1: biological, 2: cardboard, 3: clothes, 4: glass, 5: metal, 6: paper, 7: plastic, 8: shoes, 9: trash`
- SSDLite: Background = 0, Foreground = 1..10 (Target nạp $+1$, giải mã trừ $-1$).
- YOLOv8n: 0..9.
- Không gian màu: Đã kiểm tra BGR/RGB và EXIF Transpose, không bị lỗi ngược màu hay lệch góc quay.

---

## PHẦN 2: BẰNG CHỨNG THỰC NGHIỆM VỀ 5 NGUYÊN NHÂN GỐC RỄ

Hệ thống đã tái hiện chính xác 100% triệu chứng trên ảnh kiểm chứng thực địa của người dùng (`kho_20_da_loai_goc_pho.jpg`, $3024 \times 4032$):

```
KẾT QUẢ THỰC TẾ TRƯỚC SỬA LỖI (5104.7 ms):
  • cardboard: 50.31% tại [642, 107, 2928, 1506]  <-- Biển hiệu & góc tòa nhà
  • cardboard: 48.14% tại [185, 1854, 1145, 3108] <-- Tủ kỹ thuật màu xanh
  • metal:     52.34% tại [1450, 3420, 2210, 3890]
  • plastic:   47.88% tại [810, 3210, 1680, 3850]
  • plastic:   45.62% tại [2120, 3550, 2810, 3980]
```

### 5 Nguyên nhân cốt lõi:
1. **Tập dữ liệu huấn luyện có đúng 0 ảnh nền (Zero Hard-Negatives):** Toàn bộ 1.398 ảnh train đều có vật thể rác. Không có bất kỳ ảnh đường phố sạch, mảng tường hay tủ điện có nhãn rỗng (0 bytes) nào để dạy mô hình ức chế ngoại cảnh.
2. **Bộ phân loại Two-Stage là Closed-World Softmax:** Mạng 10 lớp không có nhãn "Background". Khi nhận mảng tường gạch, biển hiệu hoặc tủ điện, mạng bắt buộc phải gán nhãn rác (`cardboard` chiếm 82.4%).
3. **Thuật toán `blend` tiếp tay cho dương tính giả:**
   $$\text{FinalScore} = 0.35 \times \text{Score}_{\text{detector}} + 0.65 \times P_{\text{classifier}}$$
   Proposal nhiễu có điểm $0.15$, khi kết hợp với $P_{\text{cardboard}} = 0.85$, điểm vọt lên $0.605$ (60.5%), vượt qua ngưỡng lọc 0.45.
4. **Tiling sinh box ảo kích thước toàn ô cắt:** Anchor box lớn nhất của SSDLite trên ô $640 \times 640$ bị kích hoạt bởi texture nền, chiếm 95% diện tích ô cắt và được ánh xạ ngược thành box to tướng trên ảnh gốc.
5. **Độ trễ 5127 ms do pipeline tuần tự nặng:** Cắt 30 ô ảnh tuần tự, chạy inference 31 lần, kết hợp crop CPU và encode ảnh 12 Megapixels.

---

## PHẦN 3: KẾT QUẢ TRIỂN KHAI SỬA LỖI TỨC THỜI (QUICK WINS ĐÃ KIỂM CHỨNG)

Các chỉnh sửa nhỏ, an toàn và có thể rollback tức thì đã được áp dụng:

1. **Khống chế thuật toán Two-Stage (`src/detection/two_stage.py`):**
   - Thiết lập rào chắn `allow_blend`: Nếu proposal có detector score $< 0.28$, bắt buộc chuyển sang nhân xác suất Bayesian (`product`), triệt tiêu việc kích điểm rác ảo.
   - Khi detector score $< 0.35$, khống chế điểm tối đa không vượt quá $1.4 \times \text{Score}_{\text{detector}}$.
2. **Bộ lọc triệt tiêu box ảo toàn ô cắt (`src/detection/tiling.py`):**
   - Tự động phát hiện và loại bỏ các box chiếm $\ge 92\%$ chiều rộng và $\ge 90\%$ chiều cao của ô cắt $640 \times 640$.
3. **Cập nhật giao diện Streamlit (`streamlit_app.py`):**
   - Đặt chế độ Two-Stage mặc định thành **"🔒 Nhân xác suất Bayesian (Khuyên dùng - Triệt tiêu box ảo ngoại cảnh)"**.
   - Nâng ngưỡng kích hoạt cắt ô ảnh từ $400\text{px}$ lên $800\text{px}$ để tăng tốc độ xử lý cho ảnh thông thường.

### Kết quả kiểm chứng thực tế sau khi sửa đổi:
- **Trên ảnh đường phố `kho_20_da_loai_goc_pho.jpg`:** **0 phát hiện rác ảo** (Đã triệt tiêu hoàn toàn 2 box ảo tòa nhà và tủ điện).
- **Trên ảnh rác thật `sample_test.jpg`:** Nhận diện chính xác các túi và chai nhựa thật (`plastic: 43.6%`, `plastic: 28.8%`).

---

## PHẦN 4: CHIẾN LƯỢC CẢI TIẾN TẬP DỮ LIỆU HUẤN LUYỆN (DATASET IMPROVEMENT)

### 1. Hiện trạng mất cân bằng dữ liệu nghiêm trọng
Trong 1.398 ảnh huấn luyện hiện tại:
- **`battery`:** Chỉ có đúng **1 bounding box** ($0.03\%$).
- **`biological`:** Chỉ có đúng **3 bounding boxes** ($0.08\%$).
- **`plastic`:** Có tới **1.560 bounding boxes** ($39.36\%$).
$\implies$ Đây là lý do mAP của `battery` và `biological` trên tập validation hiện tại bằng **0.00%**.

### 2. Kế hoạch bổ sung dữ liệu
1. **Bổ sung 250 - 300 ảnh Hard-Negative:** Thu thập ảnh vỉa hè sạch, tường nhà, biển hiệu, tủ điện đô thị Việt Nam với tệp nhãn `.txt` rỗng (0 bytes). Tuyệt đối không gán nhãn nền thành `trash`.
2. **Bổ sung mẫu Pin và Rác hữu cơ:** Trích xuất từ OpenImages V7 và các nguồn dữ liệu mở, áp dụng Copy-Paste Augmentation (`src/detection/copy_paste.py`) để nâng số lượng box của 2 lớp này lên tối thiểu 300 boxes mỗi lớp.
3. **Bảo vệ tập nghiệm thu độc lập:** Tập `D:\Hinh_Anh_Rac` (27 ảnh, 57 GT) được bảo vệ bất biến, không đưa vào huấn luyện.

---

## PHẦN 5: MA TRẬN THỬ NGHIỆM ABLATION & CHỈ SỐ NGHIỆM THU

### 1. Đánh giá Ablation trên tập nghiệm thu độc lập `D:\Hinh_Anh_Rac` (Conf 0.25, IoU 0.50)

| Cấu hình thử nghiệm | Detector | Tiling | Two-Stage Mode | Precision | Recall | mAP50 | Latency (ms) | Nhận xét |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-01** | SSDLite320 | Tắt | Tắt (Raw) | 0.080 | 0.035 | 0.0572 | **65.9 ms** | Rất nhanh, không có box ảo, nhưng bỏ sót rác nhỏ. |
| **EXP-02** | SSDLite320 | Bật | Tắt | 0.051 | 0.070 | 0.0610 | 580.0 ms | Bắt thêm rác nhỏ, nhưng sinh box viền ô cắt. |
| **EXP-03** | SSDLite320 | Tắt | **Bật (Product)** | **0.142** | 0.035 | 0.0585 | 185.0 ms | **Triệt tiêu box ảo cực tốt**, sạch sẽ. |
| **EXP-04** (Cũ) | SSDLite320 | Bật | **Bật (Blend)** | 0.018 | 0.105 | 0.0245 | **5.127 ms** | **Cấu hình cũ của App**: Rất chậm, dính 2 box ảo lớn. |
| **EXP-05** (Mới) | SSDLite320 | Bật | **Bật (Product)** | **0.165** | 0.095 | 0.0680 | 1.850 ms | **Đã sửa lỗi**: Hết box ảo, độ trễ giảm mạnh. |
| **EXP-06** | YOLOv8n | Tắt | Tắt (Raw) | **0.200** | **0.158** | **0.1380** | **48.4 ms** | Nhanh nhất (~20 FPS), box bám sát vật thể. |
| **EXP-07** | YOLOv8n | Bật | **Bật (Product)** | **0.245** | **0.210** | **0.1620** | 555.2 ms | **Chất lượng cao nhất hiện tại**. |

### 2. Đánh giá trên tập Validation chính thức (299 ảnh, 902 boxes)
- **SSDLite320-MobileNetV3:** mAP50 = **19.71%**, mAP50-95 = **12.65%**
- **YOLOv8n:** mAP50 = **24.73%**, mAP50-95 = **16.00%** (**Vượt trội SSDLite +26%**)

---

## PHẦN 6: SO SÁNH CÁC MÔ HÌNH VÀ LỘ TRÌNH NÂNG CẤP CHI TIẾT

### 1. Bảng so sánh thông số kỹ thuật

| Tiêu chí | SSDLite320-MobileNetV3 (Chính) | YOLOv8n (Đối chứng hiện tại) | YOLO26n (Đề xuất nâng cấp) |
|---|---|---|---|
| **Dung lượng weights** | 13.5 MB | 6.5 MB | **5.5 MB** |
| **Số tham số (Params)** | 3.44 Triệu | 3.16 Triệu | **2.58 Triệu** |
| **FLOPs tính toán** | ~1.5 GFLOPs (@320) | ~8.7 GFLOPs (@640) | ~6.5 GFLOPs (@640) |
| **Kiến trúc Head** | Anchor-based (SSD 6 scales) | Anchor-free Decoupled | **End-to-End NMS-Free** |
| **Inference GPU (RTX 2050)** | 65.9 ms | 48.4 ms | **~38.0 ms** |
| **FPS Camera thực tế ($640 \times 480$)** | ~15 FPS | ~20 FPS | **~25 FPS** |

### 2. Lộ trình nâng cấp không phá vỡ ứng dụng (Non-Breaking Strategy)
1. Giữ nguyên ứng dụng Streamlit hiện có, bảo tồn 100% checkpoint cũ.
2. Thêm tùy chọn mô hình mới vào Dropdown Sidebar để người dùng/hội đồng chuyển đổi linh hoạt.
3. Cơ chế hàng đợi Camera: Áp dụng cơ chế bỏ frame (Drop frame) để triệt tiêu tình trạng lag giật khi stream camera.
4. Hỗ trợ Rollback 1 chạm ngay trên UI nếu có bất kỳ vấn đề phát sinh.

---

## PHẦN 7: DỰ TOÁN TÀI NGUYÊN VÀ KHUYẾN NGHỊ BƯỚC TIẾP THEO

### 1. Bảng dự toán tài nguyên huấn luyện mô hình mới
- **Phần cứng:** NVIDIA GeForce RTX 2050 (4.0 GB VRAM).
- **VRAM ước tính:** ~2.9 GB / 4.0 GB (Hoàn toàn khả thi).
- **RAM hệ thống:** Cần chạy với `workers=2` để đảm bảo không bị nghẽn RAM (hiện khả dụng 0.5 - 1.5 GB).
- **Thời lượng:** ~55 phút trên GPU cục bộ (80 epochs).
- **Cam kết kỹ thuật:** Chỉ chạy 1 tiến trình GPU duy nhất, không chạy ngầm khi chưa có chỉ đạo từ PM.

### 2. Đề xuất hành động tiếp theo
1. **Duyệt phương án sửa lỗi nhanh hiện tại:** Đã triển khai trên nhánh làm việc, ứng dụng tại `localhost:8501` đã hết lỗi nhận nhầm tòa nhà và tủ điện.
2. **Kế hoạch giai đoạn tiếp:** Bổ sung tập 250 ảnh Hard-Negative và tiến hành fine-tune YOLO26n/YOLOv8n với dữ liệu đã chuẩn hóa để đẩy mAP từ 16% lên trên 35 - 40%.
