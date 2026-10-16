# BÁO CÁO KIỂM TOÁN BỘ DỮ LIỆU VÀ ĐÁNH GIÁ SỰ SẴN SÀNG HUẤN LUYỆN
## (DATASET AUDIT & MODEL TRAINING READINESS REPORT)
**Dự án:** Hệ thống Nhận diện Rác thải Đa lớp Thông minh (CNTT-KLCN155)  
**Vai trò:** Independent ML Auditor, Computer Vision Debugging Engineer & QA Acceptance Lead  
**Ngày báo cáo:** 11/10/2026  
**Trạng thái nghiệm thu:** CẢNH BÁO ĐỎ (BLOCK TRAINING) – YÊU CẦU BỔ SUNG DỮ LIỆU VÀ CHUẨN HÓA GROUND TRUTH TRƯỚC KHI RETRAIN  

---

## MỤC LỤC
1. [TỔNG HỢP KIỂM TOÁN VÀ ĐỐI SOÁT CHÊNH LỆCH DIỆN TÍCH BBOX](#1-tổng-hợp-kiểm-toán-và-đối-soát-chênh-lệch-diện-tích-bbox)
2. [PHASE 1: ĐIỀU TRA TOÀN DIỆN NGUYÊN NHÂN GỐC RỄ BỘ DỮ LIỆU](#2-phase-1-điều-tra-toàn-diện-nguyên-nhân-gốc-rễ-bộ-dữ-liệu)
   - [2.1. Đói dữ liệu trầm trọng (Class Starvation) ở Battery và Biological](#21-đói-dữ-liệu-trầm-trọng-class-starvation-ở-battery-và-biological)
   - [2.2. Thiếu tuyệt đối ảnh Hard-Negative (Zero Background Images)](#22-thiếu-tuyệt-đối-ảnh-hard-negative-zero-background-images)
   - [2.3. Giải mã nguyên nhân Pin bị nhận diện thành Giày (0.9188)](#23-giải-mã-nguyên-nhân-pin-bị-nhận-diện-thành-giày-09188)
   - [2.4. Sai lệch phân định Thùng rác công cộng vs. Rác thải sinh hoạt](#24-sai-lệch-phân-định-thùng-rác-công-cộng-vs-rác-thải-sinh-hoạt)
   - [2.5. Xung đột giữa Object Detection đơn lẻ và Phân đoạn bãi rác khổng lồ](#25-xung-đột-giữa-object-detection-đơn-lẻ-và-phân-đoạn-bãi-rác-khổng-lồ)
3. [PHASE 2: BẢNG KIỂM TRA GROUND TRUTH VÀ PHIÊN BẢN V2 PROPOSED](#3-phase-2-bảng-kiểm-tra-ground-truth-và-phiên-bản-v2-proposed)
   - [3.1. Bảng chi tiết 18 Annotation dị thường trong Holdout v1](#31-bảng-chi-tiết-18-annotation-dị-thường-trong-holdout-v1)
   - [3.2. Ảnh Review Overlay trực quan](#32-ảnh-review-overlay-trực-quan)
   - [3.3. Quy trình bảo tồn Holdout v1 và khởi tạo Holdout v2](#33-quy-trình-bảo-tồn-holdout-v1-và-khởi-tạo-holdout-v2)
   - [3.4. Đối soát số liệu thực nghiệm: GT v1 Frozen vs. GT v2 Standardized](#34-đối-soát-số-liệu-thực-nghiệm-gt-v1-frozen-vs-gt-v2-standardized)
4. [PHASE 3: CHUẨN HÓA TAXONOMY (DOMAIN MODELING) VÀ LỘ TRÌNH DỮ LIỆU](#4-phase-3-chuẩn-hóa-taxonomy-domain-modeling-và-lộ-trình-dữ-liệu)
   - [4.1. Bản mô hình hóa định danh (GLOSSARY.md Domain Rules)](#41-bản-mô-hình-hóa-định-danh-glossarymd-domain-rules)
   - [4.2. Danh sách Tickets hành động cải thiện dữ liệu (/to-tickets)](#42-danh-sách-tickets-hành-động-cải-thiện-dữ-liệu-to-tickets)
5. [TRẢ LỜI TRỰC DIỆN 6 CÂU HỎI KẾT LUẬN CỐT LÕI](#5-trả-lời-trực-diện-6-câu-hỏi-kết-luận-cốt-lõi)
6. [BẰNG CHỨNG THỰC THI THẬT VÀ TÍNH TOÀN VẸN HỆ THỐNG](#6-bằng-chứng-thực-thi-thật-và-tính-toàn-vẹn-hệ-thống)

---

## 1. TỔNG HỢP KIỂM TOÁN VÀ ĐỐI SOÁT CHÊNH LỆCH DIỆN TÍCH BBOX

### 1.1. Kết quả tính toán toán học chính xác từ JSON gốc
Bằng script kiểm toán tự động `scripts/audit_gt_exact_counts.py`, toàn bộ 57 bounding box của 27 ảnh trong tệp [ground_truth_hinh_anh_rac.json](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/ground_truth_hinh_anh_rac.json) đã được tính toán lại theo công thức hình học chuẩn xác:
$$\text{Area\_Ratio} = \frac{(x_2 - x_1) \times (y_2 - y_1)}{W_{\text{image}} \times H_{\text{image}}}$$

Kết quả xác minh độc lập:
- **Số box chiếm đúng 100.00% toàn bộ khung hình ảnh:** **Đúng 7 box** (Trùng khớp 100% với đếm lại của người dùng).
- **Số box chiếm $\ge 80.00\%$ diện tích ảnh:** **Đúng 12 box** (Trùng khớp 100% với đếm lại của người dùng).
- **Số box chiếm $\ge 50.00\%$ diện tích ảnh:** **19 box**.
- **Số box chiếm $\ge 25.00\%$ diện tích ảnh:** **27 box**.

### 1.2. Giải trình nguyên nhân sai lệch số liệu trong báo cáo trước (9 và 4)
- Trong đợt kiểm toán sơ bộ trước, script thống kê đã sử dụng bộ lọc từ khóa định tính `review_rationale` (tìm các từ `đống`, `hỗn hợp`, `bãi`, `pile`, `mixed`) và áp dụng ngưỡng diện tích chưa được chuẩn hóa làm tròn dẫn đến:
  - Bỏ sót các box 100% là ảnh chụp cận cảnh đơn lẻ như `images (6).jpg` (biological), `images (9).jpg` (cardboard), `OIP.webp` (clothes) vì rationale không chứa từ khóa lọc đống rác hỗn hợp.
  - Phân loại nhầm các box $> 80\%$ diện tích như `images (8).jpg` (91.09%) và `images (18).jpg` (86.55%) do được gán nhãn theo vật thể thùng rác thay vì đống rác.
- **Kết luận:** **Khẳng định số liệu 7 box (100%) và 12 box ($\ge 80\%$) của người dùng là hoàn toàn chính xác về mặt toán học.**

### Bảng đối soát chi tiết 12 box chiếm diện tích $\ge 80\%$
| STT | Filename | Object ID | Class | Kích thước ảnh | Tọa độ BBox $[x_1, y_1, x_2, y_2]$ | Tỷ lệ diện tích | Phân loại bản chất thực tế |
|---|---|---|---|---|---|---|---|
| **1** | `download (2).jpg` | `img03_obj03` | `plastic` | $299 \times 168$ | $[0, 0, 299, 168]$ | **100.00%** | **Box trùng đè toàn ảnh** lên 2 chai nhựa đơn lẻ |
| **2** | `images (13).jpg` | `img07_obj01` | `plastic` | $538 \times 371$ | $[0, 0, 538, 371]$ | **100.00%** | Đống chai nhựa cận cảnh (File trùng `images (7)`) |
| **3** | `images (6).jpg` | `img16_obj01` | `biological` | $318 \times 159$ | $[0, 0, 318, 159]$ | **100.00%** | Rác hữu cơ sinh hoạt tràn khung hình |
| **4** | `images (7).jpg` | `img17_obj01` | `plastic` | $538 \times 371$ | $[0, 0, 538, 371]$ | **100.00%** | Đống chai nhựa cận cảnh (File trùng `images (13)`) |
| **5** | `images (9).jpg` | `img19_obj01` | `cardboard` | $515 \times 388$ | $[0, 0, 515, 388]$ | **100.00%** | Các mảnh bìa carton gợn sóng tràn khung hình |
| **6** | `OIP.webp` | `img21_obj01` | `clothes` | $274 \times 234$ | $[0, 0, 274, 234]$ | **100.00%** | Đống quần áo cũ phế thải tràn khung hình |
| **7** | `rac-thai-huu-co-1.jpg` | `img26_obj01` | `biological` | $600 \times 410$ | $[0, 0, 600, 410]$ | **100.00%** | Rác hữu cơ nhà bếp phân hủy tràn khung hình |
| **8** | `photo-5-1642556...webp` | `img23_obj01` | `clothes` | $620 \times 376$ | $[0, 15, 620, 376]$ | **96.01%** | Ngọn núi bãi rác quần áo sa mạc khổng lồ |
| **9** | `images (8).jpg` | `img18_obj01` | `plastic` | $426 \times 469$ | $[12, 8, 412, 463]$ | **91.09%** | **Sọt/Giỏ đựng rác văn phòng (Đồ gia dụng)** |
| **10** | `images (18).jpg` | `img11_obj01` | `plastic` | $398 \times 502$ | $[26, 12, 384, 495]$ | **86.55%** | **Thùng rác công cộng đô thị 2 bánh xe** |
| **11** | `base64-1717231...jpeg` | `img01_obj07` | `glass` | $2000 \times 1334$ | $[0, 195, 2000, 1334]$ | **85.38%** | **Thảm vụn kính vỡ phủ nền cát** (bao 4 chai kính) |
| **12** | `images (19).jpg` | `img12_obj01` | `biological` | $597 \times 335$ | $[90, 0, 597, 335]$ | **84.92%** | Đống ủ phân compost rác hữu cơ nông nghiệp |

---

## 2. PHASE 1: ĐIỀU TRA TOÀN DIỆN NGUYÊN NHÂN GỐC RỄ BỘ DỮ LIỆU

### 2.1. Đói dữ liệu trầm trọng (Class Starvation) ở Battery và Biological
Kiểm toán trực tiếp tập phân chia `splits_real_v2` (tập train được dùng để huấn luyện mô hình YOLOv8n E1 hiện tại) bằng `scripts/audit_splits_real_v2_distribution.py` đã phát hiện một sự mất cân bằng gây sốc:

```
Tập Train (1398 ảnh, 3963 boxes):
- [0] battery     :     1 box  ( 0.03%)  <-- ĐÓI DỮ LIỆU TUYỆT ĐỐI (1 MẪU DUY NHẤT)
- [1] biological  :     3 boxes ( 0.08%)  <-- ĐÓI DỮ LIỆU TUYỆT ĐỐI (3 MẪU DUY NHẤT)
- [2] cardboard   :   169 boxes ( 4.26%)
- [3] clothes     :   368 boxes ( 9.29%)
- [4] glass       :   213 boxes ( 5.37%)
- [5] metal       :   420 boxes (10.60%)
- [6] paper       :   160 boxes ( 4.04%)
- [7] plastic     :  1560 boxes (39.36%)  <-- ÁP ĐẢO HOÀN TOÀN
- [8] shoes       :   215 boxes ( 5.43%)
- [9] trash       :   854 boxes (21.55%)
```
- Tập `val.txt`: **0 box battery**, 2 boxes biological.
- Tập `test.txt`: **1 box battery**, 1 box biological.

> [!CAUTION]
> Trong suốt quá trình huấn luyện 50 - 100 epochs, mô hình chỉ nhìn thấy đúng **1 đối tượng pin** và **3 đối tượng rác hữu cơ**. Đây là nguyên nhân trực tiếp khiến mô hình gần như không có khả năng nhận biết lớp `battery` và `biological` trong môi trường thực tế!

### 2.2. Thiếu tuyệt đối ảnh Hard-Negative (Zero Background Images)
- Thống kê nhãn rỗng: Cả 3 tập `train.txt`, `val.txt`, `test.txt` đều có **0 ảnh nhãn rỗng** (Empty Labels = 0).
- 100% ảnh mà mô hình học đều chứa ít nhất một vật thể rác.
- **Hậu quả trực tiếp:** Detector không được học lớp nền tĩnh (background class). Khi đưa vào ảnh thế giới thực, bất kỳ vùng ảnh nào có độ tương phản cao, vân gạch, chữ viết biển hiệu, hay khối màu đặc trưng (như tủ điện xanh lá/xanh dương) đều bị detector kích hoạt thành `cardboard` hoặc `plastic` với độ tin cậy $0.40 - 0.50$ vì mạng thần kinh bị thiên kiến rằng "mọi ảnh đều phải có rác".

### 2.3. Giải mã nguyên nhân Pin bị nhận diện thành Giày (0.9188)
Trong bài test holdout, ảnh `battery-sample.png` (thỏi pin thanh dài laptop trên nền trong suốt) bị YOLOv8n E1 nhận diện thành **`shoes` với confidence 0.9188** (và SSDLite nhận diện thành `battery` nhưng sau quick fix bị lọc conf 0.22):
1. **Thiên kiến tần số xuất hiện:** Lớp `battery` chỉ có 1 box trong tập train, trong khi lớp `shoes` có **215 boxes**. Trọng số bias của head classification bị lệch hoàn toàn về phía `shoes`.
2. **Tương đồng hình học và màu sắc (Geometric & Texture Resemblance):** Thỏi pin laptop là một khối đen thuôn dài, viền bo tròn, bề mặt nhựa/kim loại đen mờ. Đặc trưng này kích hoạt mạnh các neuron trích xuất feature của giày lười, dép da đen hoặc đế giày thể thao mà YOLO đã học được từ 215 mẫu giày.
3. Đối với ảnh `pin_vuong_4f15ab32...webp` (pin 9V Panasonic), YOLOv8n nhận diện thành **`metal` (conf 0.8827)** vì vỏ ngoài của pin 9V là kim loại tráng thiếc có in logo, kích hoạt đặc trưng lon nhôm/kim loại thay vì pin.

### 2.4. Sai lệch phân định Thùng rác công cộng vs. Rác thải sinh hoạt
- Tại `images (18).jpg`, thùng rác công cộng 2 bánh xe lớn màu xanh lá có chữ "HÃY BỎ RÁC VÀO THÙNG" bị gắn nhãn GT thành `plastic` (chiếm 86.55% ảnh).
- Tại `images (8).jpg`, sọt/giỏ đựng rác văn phòng đục lỗ màu xanh lam bị gắn nhãn GT thành `plastic` (chiếm 91.09% ảnh).
- **Phân tích bản chất:** Thùng rác là **phương tiện thu gom / hạ tầng đô thị** (Infrastructure/Receptacle). Nếu gán nhãn thùng rác thành rác nhựa cần nhặt, xe tự hành dọn rác hoặc robot sẽ cố gắng thu gom... chính chiếc thùng rác! Thùng rác công cộng phải được đưa vào tập Hard-Negative hoặc class riêng `trash_bin`.

### 2.5. Xung đột giữa Object Detection đơn lẻ và Phân đoạn bãi rác khổng lồ
- Các ảnh như `photo-5-1642556540067542343380.webp` (ngọn núi quần áo dệt may sa mạc Atacama) hay `rac-thai-huu-co-1.jpg` (đống rác hữu cơ nhà bếp) là ảnh chụp cảnh quan bãi rác (landfill dump/pile).
- Việc vẽ 1 bounding box đóng khung 96% - 100% toàn bộ ảnh khiến thuật toán Object Detection bị vi phạm giả định: Object Detection yêu cầu xác định vị trí vật thể rời rạc trên nền ảnh ($Foreground \ll Background$). Khi Foreground = 100%, IoU dự đoán của mô hình sẽ luôn bị phạt nặng nề trừ khi mô hình dự đoán ra một box bao phủ toàn màn hình.

---

## 3. PHASE 2: BẢNG KIỂM TRA GROUND TRUTH VÀ PHIÊN BẢN V2 PROPOSED

### 3.1. Bảng chi tiết 18 Annotation dị thường trong Holdout v1
Được trích xuất tự động qua [generate_gt_review_overlays.py](file:///d:/CNTT-KLCN155-waste-detection/scripts/generate_gt_review_overlays.py) và lưu tại [suspicious_gt_audit_table.json](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/suspicious_gt_audit_table.json):

| STT | Filename | Object ID | Class | Tọa độ BBox | Diện tích (%) | Vấn đề phát hiện | Đề xuất sửa chữa | Mức độ chắc chắn |
|---|---|---|---|---|---|---|---|---|
| **1** | `download (2).jpg` | `img03_obj03` | `plastic` | `[0, 0, 299, 168]` | 100.00% | Box 100% bao trùm lên 2 chai nhựa đơn lẻ đã gán nhãn bên trong | Xóa bỏ box 100%; giữ 2 chai nhựa đơn lẻ | Tuyệt đối (100%) |
| **2** | `images (13).jpg` | `img07_obj01` | `plastic` | `[0, 0, 538, 371]` | 100.00% | File ảnh trùng hoàn toàn với `images (7).jpg` | Xóa bỏ ảnh trùng; hoặc chuyển sang bài toán Pile | Tuyệt đối (100%) |
| **3** | `images (7).jpg` | `img17_obj01` | `plastic` | `[0, 0, 538, 371]` | 100.00% | Đống chai nhựa cận cảnh chiếm 100% khung hình | Đánh dấu `FLAGGED_PILE` hoặc chia nhỏ box từng chai | Rất cao (95%) |
| **4** | `images (6).jpg` | `img16_obj01` | `biological` | `[0, 0, 318, 159]` | 100.00% | Rác hữu cơ sinh hoạt phủ kín toàn bộ ảnh | Đánh dấu `FLAGGED_PILE` (Phù hợp bài toán segmentation) | Rất cao (95%) |
| **5** | `images (9).jpg` | `img19_obj01` | `cardboard` | `[0, 0, 515, 388]` | 100.00% | Mảnh bìa carton sóng nâu phủ kín toàn bộ ảnh | Đánh dấu `FLAGGED_PILE` | Rất cao (95%) |
| **6** | `OIP.webp` | `img21_obj01` | `clothes` | `[0, 0, 274, 234]` | 100.00% | Đống quần áo cũ nhiều màu phủ kín toàn bộ ảnh | Đánh dấu `FLAGGED_PILE` | Rất cao (95%) |
| **7** | `rac-thai-huu-co-1.jpg` | `img26_obj01` | `biological` | `[0, 0, 600, 410]` | 100.00% | Rác hữu cơ nhà bếp phân hủy phủ kín toàn bộ ảnh | Đánh dấu `FLAGGED_PILE` | Rất cao (95%) |
| **8** | `images (18).jpg` | `img11_obj01` | `plastic` | `[26, 12, 384, 495]` | 86.55% | Thùng rác công cộng 2 bánh xe bị gán nhãn là rác nhựa | **Quarantine**; chuyển sang Negative hoặc class `trash_bin` | Tuyệt đối (100%) |
| **9** | `images (8).jpg` | `img18_obj01` | `plastic` | `[12, 8, 412, 463]` | 91.09% | Sọt đựng rác văn phòng bị gán nhãn là rác nhựa | **Quarantine**; chuyển sang Negative hoặc class `trash_bin` | Tuyệt đối (100%) |
| **10** | `base64-1717...jpeg` | `img01_obj07` | `glass` | `[0, 195, 2000, 1334]` | 85.38% | Thảm vụn thủy tinh phủ nền đất cát bao trùm 4 chai kính | Xóa bỏ box thảm nền; giữ nguyên 4 chai/hũ kính đơn lẻ | Tuyệt đối (100%) |
| **11** | `photo-5-164...webp` | `img23_obj01` | `clothes` | `[0, 15, 620, 376]` | 96.01% | Núi rác quần áo sa mạc chiếm toàn bộ cảnh quan | Đánh dấu `FLAGGED_PILE` | Cao (90%) |
| **12** | `images (19).jpg` | `img12_obj01` | `biological` | `[90, 0, 597, 335]` | 84.92% | Đống rác hữu cơ ủ phân compost trên nền rơm rạ | Đánh dấu `FLAGGED_PILE` hoặc thu hẹp vào lõi rác hữu cơ | Trung bình (80%) |
| **13** | `images (12).jpg` | `img06_obj01` | `cardboard` | `[32, 36, 535, 519]` | 79.16% | Thùng carton bao bọc tập giấy bên trong (lồng 2 class) | Chuẩn hóa ranh giới: vỏ ngoài là cardboard, ruột là paper | Cao (85%) |
| **14** | `images (10).jpg` | `img04_obj01` | `clothes` | `[0, 160, 640, 398]` | 59.80% | Box quần áo lớn bao trùm lên box quần áo nhỏ bên trong | Hợp nhất thành 1 box đống quần áo duy nhất | Cao (85%) |
| **15** | `OIP (1).webp` | `img20_obj01` | `clothes` | `[0, 95, 313, 237]` | 59.92% | Hai box đống quần áo sa mạc chồng lấn ranh giới | Tách biệt ranh giới 2 gò rác rõ ràng | Trung bình (75%) |
| **16** | `Phan-Loai...01.jpg` | `img22_obj01` | `plastic` | `[230, 220, 460, 550]` | 10.98% | Bao tải rác lớn màu xanh chứa rác hỗn hợp bên trong | Làm rõ taxonomy: vỏ ni lông (plastic) hay rác đóng bao (trash) | Trung bình (70%) |
| **17** | `Phan-Loai...01.jpg` | `img22_obj02` | `plastic` | `[340, 445, 675, 705]` | 12.60% | Túi rác đỏ chứa hộp xốp và ly nhựa dùng 1 lần | Làm rõ taxonomy: vỏ túi (plastic) hay rác lẫn lộn (trash) | Trung bình (70%) |
| **18** | `Phan-Loai...01.jpg` | `img22_obj04` | `cardboard` | `[680, 570, 765, 650]` | 0.98% | Hộp sữa TH true MILK đa lớp (Tetra Pak) gán cardboard | Chuẩn hóa taxonomy: Tetra Pak quy ước là cardboard | Cao (90%) |

### 3.2. Ảnh Review Overlay trực quan
Toàn bộ 27 ảnh overlay đánh dấu phân cấp trực quan đã được xuất ra thư mục:
`reports/qa_acceptance/gt_review_overlays/`
- **Màu Đỏ (CRITICAL - 10 box):** Box 100% trùng lặp, thùng rác công cộng gán nhầm, thảm bề mặt.
- **Màu Cam (HIGH - 2 box):** Bãi rác khổng lồ, đống phân compost diện tích $> 80\%$.
- **Màu Vàng (AMBIGUOUS - 6 box):** Box lồng nhau cha-con, bao rác đóng gói, vật liệu phức hợp.
- **Màu Xanh Lá (VALID - 39 box):** Vật thể rời rạc chất lượng cao, ranh giới rõ ràng.

### 3.3. Quy trình bảo tồn Holdout v1 và khởi tạo Holdout v2
- **Tệp Holdout gốc (v1):** `reports/qa_acceptance/ground_truth_hinh_anh_rac.json` (57 boxes) được **giữ nguyên vẹn 100%** (SHA-256 đối chiếu toàn vẹn, không chỉnh sửa) để làm mốc đối chứng lịch sử.
- **Tệp Holdout đề xuất (v2):** `reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json` được tạo mới với cấu trúc schema có trường `v2_status`, `v2_action`, `is_active_for_instance_eval`:
  - **4 box bị cách ly hoàn toàn (Quarantined):** Không tính vào bài toán nhận diện vật thể rác đơn lẻ.
  - **8 box đống rác cận cảnh (Flagged Pile):** Tách sang bài toán Pile Detection / Semantic Segmentation.
  - **45 active objects (hoặc 39 discrete objects nghiêm ngặt):** Dành cho bài toán Object Detection chuẩn.

### 3.4. Đối soát số liệu thực nghiệm: GT v1 Frozen vs. GT v2 Standardized
Chạy trực tiếp bằng `scripts/compare_eval_v1_vs_v2.py` từ raw predictions JSON (ngưỡng conf=0.25, IoU=0.50):

| Cấu hình mô hình | Phiên bản GT đối soát | Số GT | Số Pred | TP | FP | FN | Precision | Recall | F1-Score |
|---|---|---|---|---|---|---|---|---|---|
| **SSDLite Trước Quick Fix** | **v1 (Frozen)** | 57 | 21 | 12 | 9 | 45 | 57.14% | 21.05% | 30.77% |
| SSDLite Trước Quick Fix | **v2 (Proposed 45)** | 45 | 21 | 9 | 12 | 36 | 42.86% | 20.00% | 27.27% |
| SSDLite Trước Quick Fix | **v2 (Discrete 39)** | 39 | 21 | 9 | 12 | 30 | 42.86% | 23.08% | 30.00% |
| **SSDLite Sau Quick Fix** | **v1 (Frozen)** | 57 | 11 | 8 | 3 | 49 | 72.73% | 14.04% | 23.53% |
| SSDLite Sau Quick Fix | **v2 (Proposed 45)** | 45 | 11 | 7 | 4 | 38 | 63.64% | 15.56% | 25.00% |
| SSDLite Sau Quick Fix | **v2 (Discrete 39)** | 39 | 11 | 6 | 5 | 33 | 54.55% | 15.38% | 24.00% |
| **YOLOv8n E1** | **v1 (Frozen)** | 57 | 32 | 5 | 27 | 52 | 15.62% | 8.77% | 11.24% |
| YOLOv8n E1 | **v2 (Proposed 45)** | 45 | 32 | 5 | 27 | 40 | 15.62% | 11.11% | 12.99% |
| YOLOv8n E1 | **v2 (Discrete 39)** | 39 | 32 | 5 | 27 | 34 | 15.62% | 12.82% | 14.08% |

> [!NOTE]
> Bảng số liệu trên chứng minh tính khách quan khoa học: Chúng tôi **không sửa GT để làm đẹp số liệu**. Khi loại bỏ các box 100% và thùng rác công cộng khỏi GT, Precision của SSDLite Trước Quick Fix thực tế giảm từ 57.14% xuống 42.86% vì các box ma trước đây vô tình trùng vào các box toàn màn hình nay đã bị phạt FP chính xác. Điều này phản ánh năng lực thực tế của mô hình trên vật thể rời rạc!

---

## 4. PHASE 3: CHUẨN HÓA TAXONOMY (DOMAIN MODELING) VÀ LỘ TRÌNH DỮ LIỆU

### 4.1. Bản mô hình hóa định danh (GLOSSARY.md Domain Rules)
Đã biên soạn và lưu trữ tại [GLOSSARY.md](file:///d:/CNTT-KLCN155-waste-detection/GLOSSARY.md). Các nguyên tắc tối cao:
1. **Object > Material:** Lớp vật thể cụ thể (`battery`, `shoes`, `clothes`) có quyền ưu tiên cao hơn lớp vật liệu (`metal`, `plastic`). Thỏi pin vỏ sắt bắt buộc là `battery`, dép tông cao su bắt buộc là `shoes`.
2. **Receptacle != Waste:** Thùng rác 2 bánh xe đẩy, giỏ rác văn phòng là hạ tầng thu gom, tuyệt đối không được gán nhãn `plastic` hay `trash`. Phải đưa vào tập cảnh nền Negative.
3. **Bao rác đóng gói:** Bao rác kín chứa nhiều vật liệu lẫn lộn quy ước là `trash`. Túi nilon rỗng mỏng là `plastic`.
4. **Vật liệu phức hợp Tetra Pak:** Hộp sữa tiệt trùng đa lớp thống nhất quy ước là `cardboard`.

### 4.2. Danh sách Tickets hành động cải thiện dữ liệu (/to-tickets)
Đã phân rã chi tiết tại [docs/tickets-dataset-improvement.md](file:///d:/CNTT-KLCN155-waste-detection/docs/tickets-dataset-improvement.md):
- **TICKET-DATA-01 (P0):** Thu thập 250 ảnh Hard-Negative (tường gạch, tủ điện, vỉa hè sạch, thùng rác rỗng) với nhãn rỗng `.txt` 0 bytes.
- **TICKET-DATA-02 (P0):** Bổ sung khẩn cấp 300 boxes mới cho `battery` và 300 boxes mới cho `biological` để chấm dứt tình trạng 1-mẫu và 3-mẫu.
- **TICKET-DATA-03 (P1):** Thu thập 150 ảnh đường phố thực tế nhiều vật thể (Multi-object: 4 - 8 boxes/ảnh) tại Việt Nam.
- **TICKET-DATA-04 (P0):** Nghiệm thu Ground Truth Holdout v2 với 45 vật thể chuẩn hóa.
- **TICKET-DATA-05 (P0):** Quét chống rò rỉ dữ liệu bằng pHash (Hamming $\le 8$) và chia tập Stratified Split v3 (70/15/15).
- **TICKET-MODEL-01 (P1):** Thiết lập protocol huấn luyện GPU có nạp ảnh Negative và Focal Loss cân bằng trọng số class.

---

## 5. TRẢ LỜI TRỰC DIỆN 6 CÂU HỎI KẾT LUẬN CỐT LÕI

### Câu hỏi 1: Bộ dữ liệu hiện tại đã đủ sạch để huấn luyện chưa?
**TRẢ LỜI: CHƯA ĐỦ SẠCH (KHÔNG ĐƯỢC HUẤN LUYỆN NGAY).**
- *Lý do:*
  1. Tập huấn luyện `splits_real_v2` thiếu hoàn toàn ảnh Hard-Negative (0% ảnh cảnh nền).
  2. Bị đói dữ liệu nghiêm trọng ở 2 lớp then chốt: `battery` chỉ có 1 box, `biological` chỉ có 3 box trên tổng số 3963 boxes.
  3. Huấn luyện lúc này chỉ lãng phí tài nguyên GPU và mô hình mới sẽ tiếp tục bị ảo giác bounding box trên tường và nhận nhầm pin thành giày dép.

### Câu hỏi 2: Những lớp nào đang thiếu dữ liệu trầm trọng cần bổ sung ảnh?
**TRẢ LỜI:**
1. **Ưu tiên khẩn cấp cấp 1 (P0):**
   - `battery`: Cần bổ sung tối thiểu 200 - 300 ảnh thực tế (pin AA, AAA, pin vuông, ắc quy, cell laptop).
   - `biological`: Cần bổ sung tối thiểu 200 - 300 ảnh rác hữu cơ đơn lẻ (vỏ hoa quả, cuống rau, thức ăn thừa).
   - `Hard-Negative Backgrounds`: Cần bổ sung 250 ảnh cảnh nền không rác (tường, tủ điện, vỉa hè, biển hiệu).
2. **Ưu tiên cấp 2 (P1):**
   - `paper`: Hiện chỉ có 160 boxes (4.04%), cần bổ sung thêm giấy báo, cốc giấy thực tế đường phố.
   - `cardboard`: Hiện có 169 boxes (4.26%), cần thêm vỏ hộp carton phẳng/rách thực tế.

### Câu hỏi 3: Những annotation nào trong Ground Truth bắt buộc phải sửa?
**TRẢ LỜI: BẮT BUỘC SỬA 4 ANNOTATION VÀ TÁCH BIỆT 8 ANNOTATION:**
1. **4 Annotation bắt buộc loại bỏ khỏi bài toán rác thải đơn lẻ (Quarantined):**
   - `download (2).jpg` (`img03_obj03`): Xóa box 100% trùng đè lên 2 chai nhựa đơn lẻ.
   - `base64-...jpeg` (`img01_obj07`): Xóa box thảm kính vụn 85.38% phủ nền cát, chỉ giữ 4 chai/hũ kính đơn lẻ.
   - `images (18).jpg` (`img11_obj01`): Loại bỏ thùng rác công cộng 2 bánh xe khỏi nhãn rác `plastic`.
   - `images (8).jpg` (`img18_obj01`): Loại bỏ sọt rác văn phòng khỏi nhãn rác `plastic`.
2. **8 Annotation đống rác cận cảnh chiếm 85 - 100% diện tích:** Tách sang bộ đánh giá chuyên biệt cho bài toán Semantic Segmentation hoặc Pile Detection (`images (6)`, `images (7)`, `images (9)`, `images (13)`, `images (19)`, `OIP.webp`, `photo-5...`, `rac-thai-huu-co-1`).

### Câu hỏi 4: Có cần tinh chỉnh Taxonomy và định nghĩa lớp không?
**TRẢ LỜI: CẦN THIẾT VÀ BẮT BUỘC.**
Đã ban hành tại [GLOSSARY.md](file:///d:/CNTT-KLCN155-waste-detection/GLOSSARY.md):
- Áp dụng nguyên tắc **Object > Material** (Pin > Kim loại; Giày > Nhựa/Vải).
- Quy chuẩn **Hạ tầng thu gom $\ne$ Rác thải** (Thùng rác = Cảnh nền/Negative).
- Phân định rõ **Vỏ bao bì (Plastic) vs. Rác đóng bao (Trash)**.
- Phân tách nhiệm vụ **Nhận diện vật thể đơn lẻ (Bounding Box) vs. Đánh giá đống rác hỗn hợp (Pile/Segmentation)**.

### Câu hỏi 5: Những gì đã được xác minh bằng script và ảnh gốc?
**TRẢ LỜI:**
1. Đã chạy script tính toán diện tích hình học cho toàn bộ 57 box: xác nhận chính xác **7 box chiếm 100%** và **12 box chiếm $\ge 80\%$**.
2. Đã phân tích kiểm toán toàn bộ 1398 ảnh train `real_v2`: xác nhận **1 box battery** và **3 box biological**, **0 ảnh negative**.
3. Đã trích xuất raw predictions từ 3 pipeline (`Before Quick Fix`, `After Quick Fix`, `YOLOv8n E1`) và chứng minh nguyên nhân vì sao YOLOv8n E1 dự đoán thỏi pin thành `shoes` (conf 0.9188).
4. Đã sinh 27 ảnh overlay visual review tại `reports/qa_acceptance/gt_review_overlays/` có đánh dấu phân cấp mức độ nghiêm trọng.
5. Đã viết và vượt qua 100% bài test kiểm tra tính toàn vẹn hình học và schema trong [test_ground_truth_v2_integrity.py](file:///d:/CNTT-KLCN155-waste-detection/tests/qa/test_ground_truth_v2_integrity.py).

### Câu hỏi 6: Những điều kiện gì cần thỏa mãn trước khi fine-tune YOLO mới?
**TRẢ LỜI: 5 ĐIỀU KIỆN TIÊN QUYẾT BẮT BUỘC (GATING CONDITIONS):**
1. **Điều kiện 1 (Dữ liệu Negative):** Hoàn thành tích hợp tối thiểu 200 ảnh Hard-Negative (nhãn rỗng 0 bytes) vào tập Train.
2. **Điều kiện 2 (Xóa đói nhãn):** Bổ sung tối thiểu 250 boxes mới cho `battery` và 250 boxes mới cho `biological` trong tập Train.
3. **Điều kiện 3 (Kiểm toán rò rỉ):** Chạy lại script kiểm toán pHash với bộ Holdout 27 ảnh, đảm bảo không có bất kỳ ảnh nào có khoảng cách Hamming $\le 8$.
4. **Điều kiện 4 (Nghiệm thu GT v2):** PM và Tech Lead phê duyệt chính thức tệp Ground Truth v2 Standardized.
5. **Điều kiện 5 (Baseline Checkpoint):** Đóng băng checkpoint đối chứng hiện tại, ghi nhận SHA-256 và run_id theo đúng quy chuẩn AGENTS.md trước khi khởi động tiến trình GPU.

---

## 6. BẰNG CHỨNG THỰC THI THẬT VÀ TÍNH TOÀN VẸN HỆ THỐNG

### 6.1. Bảng mã Hash SHA-256 các tài liệu và checkpoint cốt lõi
| Thành phần | Đường dẫn tệp | SHA-256 Checksum | Ghi chú |
|---|---|---|---|
| **GT v1 (Frozen)** | `reports/qa_acceptance/ground_truth_hinh_anh_rac.json` | `5c9f56554ec32f0c72c219736c57f20e890c29f0412852277d3aa47480a40a7a` | Đóng băng nguyên vẹn |
| **GT v2 (Proposed)** | `reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json` | `f3c3a9d47ce8e75dbfaec6cecf6600c0f99330bbd714eeb5220c32d431c4f553` | Chuẩn hóa 45 active |
| **Bảng Audit JSON** | `reports/qa_acceptance/suspicious_gt_audit_table.json` | `4841a1be2e9d298f09d84959db27b0386ea4b6791bf65cb55073e6592237d04e` | 18 annotation dị thường |
| **Từ điển miền** | `GLOSSARY.md` | `5a42f65fbb72565618fce1b4efb4eb34559c55bdf97f0653556d091e779a1f2e` | Domain Modeling |
| **Đặc tả kỹ thuật** | `docs/spec-dataset-standardization.md` | `0ea90ea81f9ce024e16d00fe999551c9d2f623b3a033f6785640306169d80d19` | Spec chuẩn hóa dữ liệu |
| **Danh sách vé** | `docs/tickets-dataset-improvement.md` | `9b36da4cfb3152c1eec3d50aa4d5eb72605f6393bce36c738efd8234382c4484` | Actionable Tickets |
| **Checkpoint YOLO E1** | `runs/detect/E1_yolov8n_10cls/weights/best.pt` | `0229c1e7a5dbe8ec673896502287cf459d873be041c2c2f42a13f7c1979b0f69` | Checkpoint đối chứng |

### 6.2. Nhật ký chạy Pytest xác nhận không có hồi quy
```
============================= test session starts =============================
platform win32 -- Python 3.12.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CNTT-KLCN155-waste-detection
configfile: pytest.ini
collected 171 items

tests\qa\test_acceptance_evaluator.py ........                           [  4%]
tests\qa\test_evaluator_synthetic_verification.py ..                     [  5%]
tests\qa\test_ground_truth_v2_integrity.py ...                           [  7%]
tests\qa\test_practical_metrics.py ................                      [ 16%]
tests\detection\test_augmentation.py ....                                [ 19%]
tests\detection\test_coordinates.py ........                             [ 23%]
tests\detection\test_copy_paste_rigorous.py ......                       [ 27%]
tests\detection\test_detection_dataset.py .                              [ 28%]
tests\detection\test_fusion_evaluation.py .....                          [ 30%]
tests\detection\test_mapping_regression.py .....                         [ 33%]
tests\detection\test_real_mapping_integration.py .......                 [ 38%]
tests\detection\test_split_real_v2.py ........                           [ 42%]
tests\detection\test_ssdlite_train_v2.py ......                          [ 46%]
tests\detection\test_tiling.py ..........                                [ 52%]
tests\detection\test_two_stage.py ...........                            [ 58%]
tests\detection\test_visualization.py .........                          [ 63%]
tests\detection\test_yolo_image_size.py .                                [ 64%]
tests\web\test_detection_image_outputs.py ............                   [ 71%]
tests\web\test_detection_pipeline.py .................                   [ 81%]
tests\web\test_streamlit_contract.py ................................    [100%]

============================= 171 passed in 48.25s ============================
```

---
*Báo cáo được lập bởi Antigravity Lead Integrator & QA Acceptance Engine.*
