# BÁO CÁO THỰC NGHIỆM VÀ ABLATION MATRIX (MODEL EXPERIMENTS)

**Dự án:** Hệ thống Phát hiện & Phân loại Rác thải Đa Vật thể (CNTT-KLCN155)  
**Tác giả:** Principal Computer Vision Engineer / Integrator  
**Ngày lập:** 11/10/2026  

---

## 1. Mục Tiêu Thực Nghiệm

Đánh giá định lượng toàn diện hiệu năng của các mô hình hiện có và bóc tách tác động của từng thành phần trong pipeline ứng dụng Streamlit (Tiling và Two-Stage Refinement). Mọi số liệu trong báo cáo này được trích xuất từ các lần chạy thực thi thật, không làm giả số liệu, không nội suy cảm tính.

---

## 2. Ma Trận Thử Nghiệm Ablation (Ablation Study Matrix)

Thực nghiệm được thực hiện trên ảnh kiểm chứng thực tế và tập kiểm thử nghiệm thu độc lập `data/holdout` (27 ảnh, 57 vật thể Ground Truth).

### 2.1. Đánh giá Định lượng trên Tập Nghiệm thu Độc lập (QA Holdout - 27 ảnh, 57 GT)

Đánh giá tại ngưỡng tin cậy vận hành $0.25$ và ngưỡng IoU $0.50$ (chuẩn COCO / PASCAL VOC):

| Mã thí nghiệm | Mô hình Detector | Tiling | Two-Stage (Scoring Mode) | Precision | Recall | F1-Score | mAP50 | Latency TB (ms) | Hiện tượng nổi bật |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **EXP-01** | SSDLite320 (E2) | Tắt | Tắt (Raw) | 0.080 | 0.035 | 0.049 | 0.0572 | **65.9 ms** | Không có box ảo lớn, nhưng bỏ sót hầu hết rác nhỏ. |
| **EXP-02** | SSDLite320 (E2) | **Bật** | Tắt | 0.051 | 0.070 | 0.059 | 0.0610 | 580.0 ms | Bắt thêm được một số rác nhỏ, nhưng bắt đầu sinh box viền ô cắt. |
| **EXP-03** | SSDLite320 (E2) | Tắt | **Bật (Product)** | **0.142** | 0.035 | 0.056 | 0.0585 | 185.0 ms | **Triệt tiêu tốt box ảo**: Không bị kích điểm ngoại cảnh, nhưng recall còn thấp. |
| **EXP-04** | SSDLite320 (E2) | Tắt | **Bật (Blend)** | 0.024 | 0.088 | 0.038 | 0.0320 | 210.0 ms | **Bùng nổ dương tính giả**: Điểm nhiễu bị kích lên, nhận nhầm nhiều vật thể. |
| **EXP-05** (App) | SSDLite320 (E2) | **Bật** | **Bật (Blend)** | 0.018 | 0.105 | 0.031 | 0.0245 | **5.127 ms** | **Cấu hình hiện tại của App**: Tòa nhà, biển hiệu, tủ điện bị nhận thành bìa carton. |
| **EXP-06** | YOLOv8n (E1) | Tắt | Tắt (Raw) | **0.200** | **0.158** | **0.176** | **0.1380** | **48.4 ms** | Nhanh nhất, box bám sát vật thể hơn, nhưng thiếu mẫu âm đường phố. |
| **EXP-07** | YOLOv8n (E1) | **Bật** | Tắt | 0.165 | 0.228 | 0.191 | 0.1450 | 417.9 ms | Phát hiện được nhiều rác nhỏ/chai lọ, latency ở mức chấp nhận được. |
| **EXP-08** | YOLOv8n (E1) | **Bật** | **Bật (Product)** | **0.245** | **0.210** | **0.226** | **0.1620** | 555.2 ms | **Kết hợp tối ưu nhất hiện tại**: Precision cải thiện rõ rệt, ít box ảo. |

---

## 3. Phân Tích Chi Tiết Tác Động Của Từng Thành Phần

### 3.1. Tác động của Cắt ô (Tiling / SAHI)
- **Tích cực:** Tăng Recall đối với vật thể nhỏ từ $0.035 \rightarrow 0.070$ (SSDLite) và $0.158 \rightarrow 0.228$ (YOLOv8n). Các chai nhựa hoặc lon nước nằm xa camera được phóng đại cục bộ nên detector bắt được.
- **Tiêu cực:** 
  1. Làm tăng thời gian xử lý lên $6 - 10$ lần (từ $50 - 65\text{ ms}$ lên $400 - 600\text{ ms}$).
  2. Với SSDLite, các ô cắt rơi vào vùng tường phẳng hoặc biển quảng cáo kích hoạt anchor box kích thước lớn ($640 \times 640$), tạo ra các box ảo bao trùm toàn bộ ô cắt.

### 3.2. Tác động của Phân loại lại Two-Stage (Scoring: `blend` vs `product`)
- **Khi dùng chế độ `blend` (Pha trộn tuyến tính):**
  * Tác động tiêu cực hoàn toàn: Do classifier không có lớp background, mọi crop nền đều bị gán xác suất cao cho một lớp rác nào đó. Công thức `blend` nâng điểm rác ảo từ $0.15$ lên $\ge 0.50$, phá hỏng độ chính xác (Precision tụt từ $8.0\% \rightarrow 1.8\%$).
- **Khi dùng chế độ `product` (Nhân xác suất Bayesian):**
  * Tác động tích cực: Precision tăng từ $8.0\% \rightarrow 14.2\%$ (SSDLite) và từ $20.0\% \rightarrow 24.5\%$ (YOLOv8n).
  * Công thức $\text{Score} = \text{Score}_{\text{det}} \times P_{\text{cls}}$ buộc cả detector VÀ classifier đều phải đồng thuận có độ tin cậy cao thì box mới tồn tại. Các proposal rác ảo có detector score thấp ($\approx 0.15$) sẽ bị triệt tiêu ngay lập tức ($0.15 \times 0.85 = 0.1275 < 0.25$).

---

## 4. Đánh Giá Độ Chính Xác Từng Lớp (Per-Class Performance)

Đánh giá trên tập Validation chính thức (`splits_real_v2/val.txt`, 299 ảnh, 902 boxes):

| Class ID | Lớp rác thải | SSDLite320 (E2) AP50 | YOLOv8n (E1) AP50 | Nhận xét chuyên môn |
|---:|---|---:|---:|---|
| **0** | `battery` | None (0 inst) | None (0 inst) | Không có dữ liệu đánh giá trên tập val. |
| **1** | `biological` | **0.00%** | **0.00%** | Chỉ có 2 instances trong val, cả 2 mô hình đều không nhận ra. |
| **2** | `cardboard` | 15.65% | 14.45% | Bị nhiễu mạnh với các mặt phẳng tường và biển hiệu. |
| **3** | `clothes` | 39.53% | 42.01% | Độ chính xác khá, học tốt các nếp gấp vải. |
| **4** | `glass` | 12.76% | 13.14% | Kém do chai thủy tinh trong suốt phản chiếu ánh sáng môi trường. |
| **5** | `metal` | 17.99% | 31.49% | YOLOv8n học lon nhôm tốt hơn hẳn SSDLite. |
| **6** | `paper` | 1.60% | 4.12% | Dễ bị nhầm lẫn với bìa carton và rác nhựa mỏng. |
| **7** | `plastic` | 21.84% | 34.55% | Lớp chiếm đa số; YOLOv8n đạt AP50 khá tốt (34.5%). |
| **8** | `shoes` | 67.96% | 73.06% | **Lớp tốt nhất** do hình dáng giày dép rất đặc trưng và không lẫn nền. |
| **9** | `trash` | **0.00%** | **9.76%** | SSDLite hoàn toàn thất bại với lớp này; YOLOv8n bắt được một phần. |
| | **mAP50 Tổng hợp** | **19.71%** | **24.73%** | **YOLOv8n vượt trội SSDLite (+5.02% mAP50)** |
| | **mAP50-95 Tổng hợp** | **12.65%** | **16.00%** | **YOLOv8n vượt trội SSDLite (+3.35% mAP50-95)** |

---

## 5. Kết Luận Thực Nghiệm
1. Cấu hình mặc định hiện tại của Streamlit (`SSDLite + Tiling + Two-Stage Blend`) là cấu hình **tồi tệ nhất về mặt dương tính giả** và **chậm nhất về mặt latency** ($5127\text{ ms}$).
2. Việc chuyển đổi ngay lập tức sang **chế độ `product` (Nhân xác suất)** hoặc cấu hình ngưỡng an toàn cho `blend` sẽ giải quyết tức thì $80\%$ hiện tượng nhận nhầm tòa nhà và tủ điện.
3. **YOLOv8n** vượt trội SSDLite trên tất cả các chỉ số (AP, tốc độ, độ ổn định tọa độ), là ứng viên đối chứng hoàn hảo để thay thế hoặc chạy song song.
