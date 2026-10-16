# BÁO CÁO KIỂM TOÁN CHUYÊN SÂU VÀ NGHIỆM THU ĐÁNH GIÁ (DEEP AUDIT & DoD EVALUATION)
**Dự án:** Hệ thống Phát hiện và Phân loại Rác thải (CNTT-KLCN155)  
**Tác giả:** Independent ML Auditor, Computer Vision Debugging Engineer & QA Acceptance Team  
**Ngày lập:** 11/10/2026  
**Mã báo cáo:** `DEEP_AUDIT_DOD_20261011`

---

## 1. GIẢI THÍCH CHÊNH LỆCH SỐ LƯỢNG BOX (VẤN ĐỀ 1 & 2)

### 1.1. Cấu hình BEFORE QUICK FIX: Tại sao có 24 box nhưng $\text{TP} = 13, \text{FP} = 8$ (thiếu 3 box)?
- **Số liệu quan sát:** Pipeline sinh ra 24 box, nhưng tổng $\text{TP} + \text{FP} = 13 + 8 = 21$ box. Có 3 box không nằm trong $\text{TP}$ hay $\text{FP}$.
- **Bản chất kỹ thuật:**
  - Bộ phát hiện (Two-Stage Detector) được khởi tạo với ngưỡng sinh ứng viên:
    $$\text{min\_score} = 0.20$$
  - Trên toàn bộ 27 ảnh holdout ngoài thực tế (`data/holdout`), pipeline sinh ra tổng cộng **24 box** có điểm tin cậy $\ge 0.20$.
  - Tuy nhiên, giao thức đánh giá nghiệm thu chuẩn COCO (`practical_metrics.py`) đo lường hiệu năng tại ngưỡng vận hành:
    $$\text{operating\_confidence} = 0.25$$
  - Danh sách chính xác 3 box nằm trong khoảng $[0.20, 0.25)$ bị bộ lọc đánh giá loại bỏ:
    1. `base64-1717231151649602366964.jpeg`: nhãn `glass`, conf = **0.2061**, box = `[1360.0, 311.6, 1970.6, 636.0]`
    2. `images (13).jpg`: nhãn `plastic`, conf = **0.2125**, box = `[9.5, 19.2, 522.5, 367.4]`
    3. `images (7).jpg`: nhãn `plastic`, conf = **0.2125**, box = `[9.5, 19.2, 522.5, 367.4]`
  - Còn lại đúng **21 box** có $\text{confidence} \ge 0.25$. Khi tiến hành matching với Ground Truth ở $\text{IoU} \ge 0.50$, 21 box này phân rã chính xác thành:
    $$\mathbf{\text{TP} = 13,\quad \text{FP} = 8 \implies \text{TP} + \text{FP} = 21}$$
    $$\mathbf{\text{Tổng sinh (24)} = (\text{TP} + \text{FP}) + \text{Bị lọc dưới ngưỡng 0.25 (3)} = 21 + 3 = 24}$$
  - **Kết luận:** 3 box "thiếu" không hề bị thất thoát hay lỗi logic; đó là 3 dự đoán sub-threshold ($0.20 \le \text{conf} < 0.25$) bị loại bỏ bởi ngưỡng tin cậy vận hành $0.25$.

---

### 1.2. Cấu hình AFTER QUICK FIX: Tại sao có 19 box nhưng $\text{TP} = 8, \text{FP} = 3$ (thiếu 8 box)?
- **Số liệu quan sát:** Pipeline sinh ra 19 box, nhưng tổng $\text{TP} + \text{FP} = 8 + 3 = 11$ box. Có 8 box không nằm trong $\text{TP}$ hay $\text{FP}$.
- **Bản chất kỹ thuật:**
  - Ở cấu hình sau sửa nhanh, chế độ tính điểm Two-Stage chuyển sang tích xác suất Bayesian (`product` mode):
    $$\text{Score} = \text{Score}_{\text{detector}} \times P_{\text{classifier}}$$
  - Phép nhân hai số thực trong khoảng $(0, 1)$ làm cho phân phối điểm bị kéo co về phía 0. Do đó, có nhiều box đạt trên ngưỡng sinh $0.20$ nhưng không vượt qua được ngưỡng vận hành $0.25$.
  - Danh sách chính xác 8 box nằm trong khoảng $[0.20, 0.25)$ bị loại ở ngưỡng 0.25:
    1. `base64-1717231151649602366964.jpeg`: nhãn `metal`, conf = **0.2201**, box = `[60.5, 722.6, 630.0, 1326.3]`
    2. `base64-1717231151649602366964.jpeg`: nhãn `plastic`, conf = **0.2173**, box = `[1091.6, 808.5, 1573.7, 1271.5]`
    3. `base64-1717231151649602366964.jpeg`: nhãn `glass`, conf = **0.2061**, box = `[1360.0, 311.6, 1970.6, 636.0]`
    4. `images (13).jpg`: nhãn `plastic`, conf = **0.2125**, box = `[9.5, 19.2, 522.5, 367.4]`
    5. `images (17).jpg`: nhãn `plastic`, conf = **0.2382**, box = `[393.1, 53.8, 477.9, 182.4]`
    6. `images (7).jpg`: nhãn `plastic`, conf = **0.2125**, box = `[9.5, 19.2, 522.5, 367.4]`
    7. `images (8).jpg`: nhãn `plastic`, conf = **0.2196**, box = `[25.3, 9.4, 414.9, 461.3]`
    8. `pin_vuong_4f15ab32849e43a8abdaef2bf000463f.webp`: nhãn `battery`, conf = **0.2212**, box = `[251.5, 91.3, 556.1, 576.0]`
  - Còn lại đúng **11 box** có $\text{confidence} \ge 0.25$. Khi matching ở $\text{IoU} \ge 0.50$, 11 box này phân rã thành:
    $$\mathbf{\text{TP} = 8,\quad \text{FP} = 3 \implies \text{TP} + \text{FP} = 11}$$
    $$\mathbf{\text{Tổng sinh (19)} = (\text{TP} + \text{FP}) + \text{Bị lọc dưới ngưỡng 0.25 (8)} = 11 + 8 = 19}$$
  - **Kết luận:** Đồng nhất 100% với phương trình bảo toàn dự đoán: Mọi box sinh ra đều được hạch toán đầy đủ, không có box nào bị bỏ sót ngoài số liệu thống kê.

---

## 2. ĐỊNH NGHĨA VÀ XỬ LÝ CHUẨN CLASS-AWARE FALSE POSITIVE (VẤN ĐỀ 3)

### 2.1. Quy chuẩn quốc tế (COCO / PASCAL VOC / PyTorch Torchvision)
Trong bài toán Object Detection có phân loại lớp (Class-Aware Detection):
- Đánh giá được thực hiện độc lập theo từng lớp $c \in \{0, 1, \dots, 9\}$.
- Giả sử tại một vị trí có Ground Truth thuộc lớp $A$ (ví dụ: chai thủy tinh `glass`), nhưng mô hình dự đoán ra hộp tọa độ đó với nhãn $B$ (ví dụ: `plastic`) với $\text{IoU} \ge 0.50$:
  1. **Đối với lớp $A$ (`glass`):** Vật thể nhãn thật này **không có bất kỳ dự đoán nào của lớp $A$** ghép trúng. Do đó, vật thể này được tính là **1 False Negative (FN) của lớp $A$**.
  2. **Đối với lớp $B$ (`plastic`):** Dự đoán này gán nhãn `plastic`, nhưng tại vị trí đó **không hề có vật thể Ground Truth nào thuộc lớp $plastic$**. Do đó, dự đoán này được tính là **1 False Positive (FP) của lớp $plastic$**.
- **Quy tắc toán học bất biến:**
  > Một dự đoán định vị đúng vị trí ($\text{IoU} \ge 0.50$) nhưng sai nhãn lớp luôn luôn tạo ra đồng thời:  
  > **$+1\text{ FP}$ (ở lớp dự đoán sai)** và **$+1\text{ FN}$ (ở lớp nhãn thật bị bỏ sót)**.  
  > Tuyệt đối không bao giờ được tính là True Positive (TP) trong chỉ số Class-Aware!

### 2.2. Phân biệt với chỉ số Class-Agnostic (Không xét lớp)
- Trong chỉ số **Class-Agnostic** (chỉ đánh giá năng lực định vị hình học):
  Mọi dự đoán có $\text{IoU} \ge 0.50$ với bất kỳ GT nào (bất kể nhãn) đều được tính là **1 TP**.
- Trong file `raw_predictions_*.json`, để phục vụ kiểm toán độc lập minh bạch nhất, trạng thái matching của mỗi box được phân loại thành 4 nhóm rõ ràng:
  - `MATCHED_TP`: Khớp cả vị trí ($\text{IoU} \ge 0.50$) VÀ đúng nhãn lớp.
  - `WRONG_CLASS_FP`: Khớp vị trí ($\text{IoU} \ge 0.50$) nhưng SAI nhãn lớp (được ghi rõ `matched_gt_class` và `matched_iou` để đối chiếu).
  - `UNMATCHED_FP`: Dự đoán vào vùng nền trống không có rác, hoặc $\text{IoU} < 0.50$.
  - `FILTERED_CONF`: Điểm tin cậy nằm dưới ngưỡng vận hành ($< 0.25$).

---

## 3. BỘ DỮ LIỆU RAW PREDICTIONS JSON ĐÃ XUẤT (VẤN ĐỀ 4 & 5)

Đã xuất đầy đủ 3 tệp dữ liệu dự đoán thô (raw predictions) chi tiết từng box của từng ảnh tại:
1. 📄 [reports/qa_acceptance/raw_predictions_before_quick_fix.json](../reports/qa_acceptance/raw_predictions_before_quick_fix.json)
2. 📄 [reports/qa_acceptance/raw_predictions_after_quick_fix.json](../reports/qa_acceptance/raw_predictions_after_quick_fix.json)
3. 📄 [reports/qa_acceptance/raw_predictions_yolov8n_e1.json](../reports/qa_acceptance/raw_predictions_yolov8n_e1.json)
4. 📄 [reports/qa_acceptance/verified_evaluation_summary_from_raw.json](../reports/qa_acceptance/verified_evaluation_summary_from_raw.json)

### Bảng kết quả chạy lại trực tiếp từ Raw JSON (Không lấy số từ Markdown)
Được tính toán độc lập bằng lệnh: `.venv\Scripts\python.exe scripts/re_evaluate_from_raw_json.py`

| Chỉ số nghiệm thu | BEFORE QUICK FIX (Tiling 640 + Blend) | AFTER QUICK FIX (Tiling 800 + Product) | YOLOV8N E1 RAW (Không Tiling) |
|---|:---:|:---:|:---:|
| **Tổng box sinh ra (conf $\ge 0.20$)** | 24 | 19 | 46 |
| **Box bị lọc dưới ngưỡng ($< 0.25$)** | 3 | 8 | 14 |
| **Box đánh giá thực tế ($\ge 0.25$)** | **21** | **11** | **32** |
| **Class-Aware True Positives (TP)** | 13 | 8 | 5 |
| **Class-Aware False Positives (FP)** | 8 | **3** | 27 |
| **Class-Aware False Negatives (FN)** | 44 | 49 | 52 |
| **Class-Aware Precision** | 61.90% | **72.73%** | 15.62% |
| **Class-Aware Recall** | 22.81% | 14.04% | 8.77% |
| **Class-Aware F1-Score** | 33.33% | 23.53% | 11.24% |
| **Class-Agnostic TP** | 14 | 8 | 11 |
| **Class-Agnostic FP** | 7 | 3 | 21 |
| **Độ chính xác phân loại cặp định vị** | 11/14 (78.57%) | **8/8 (100.0%)** | 5/11 (45.45%) |
| **COCO mAP50** | 0.2812 | 0.2258 | 0.0498 |
| **COCO mAP50-95** | 0.1610 | 0.1464 | 0.0233 |

---

## 4. THỰC NGHIỆM TRÊN ẢNH KHÓ `rac_online_012.jpg` (VẤN ĐỀ 6)

Ảnh `rac_online_012.jpg` ($3000 \times 3000\text{ px}$) được tải từ Wikimedia Commons, chụp đống rác bìa carton và rác thải hỗn hợp trên vỉa hè đường phố với hậu cảnh là tường gạch tòa nhà cao tầng.

### Kết quả đo lường đối đầu trực tiếp:

```
[rac_online_012.jpg - Kích thước 3000 x 3000]

1. BEFORE QUICK FIX (SSDLite + Tiling 640 + Blend):
   - Thời gian xử lý: 8010.4 ms (8.0 giây)
   - Số box phát hiện: 8 boxes
   * BOX ẢO TÒA NHÀ: cardboard (0.459) tại box=[1540.8, 889.0, 1705.7, 1143.2], w=165, h=254
     -> Nằm ở độ cao y in [889, 1143] (chính giữa bức tường tòa nhà, hoàn toàn không có rác).
   * 7 box còn lại: rải rác trên vỉa hè (metal, plastic, trash).

2. AFTER QUICK FIX (SSDLite + Tiling 800 + Product):
   - Thời gian xử lý: 4360.9 ms (giảm 45.5% thời gian)
   - Số box phát hiện: 4 boxes
   * TRIỆT TIÊU HOÀN TOÀN BOX TÒA NHÀ: Không còn bất kỳ box nào ở độ cao y < 1600!
   * 4 box còn lại:
     - metal (0.328): box=[874.0, 1756.4, 1149.2, 2175.7] (rác kim loại dưới vỉa hè)
     - plastic (0.290): box=[918.1, 2180.2, 1025.7, 2306.5] (chai/túi nhựa)
     - plastic (0.274): box=[1310.4, 2081.7, 1519.6, 2176.0] (túi nhựa)
     - metal (0.239): box=[719.2, 1611.7, 833.1, 1737.1] (lon kim loại)
   -> Giữ nguyên toàn bộ cụm rác thật trên mặt đường (y in [1600, 2300]), triệt tiêu sạch box ảo tòa nhà.

3. YOLOV8N E1 (Raw, imgsz=640):
   - Thời gian xử lý: 1072.0 ms (nhanh gấp 8 lần cấu hình cũ)
   - Số box phát hiện: 7 boxes
   - cardboard (0.743): box=[1261.5, 2056.7, 1553.6, 2263.5] (thùng carton chuẩn xác)
   - cardboard (0.738): box=[387.1, 1937.8, 987.7, 2329.0] (thùng carton chuẩn xác)
   - 0 box ảo trên tường tòa nhà.
```

---

## 5. KIỂM TRA CHẤT LƯỢNG GROUND TRUTH: ĐỐNG RÁC HỖN HỢP VÀ BOX QUÁ KHỔ (VẤN ĐỀ 7)

Đã quét toàn bộ 57 vật thể trong file [ground_truth_hinh_anh_rac.json](../reports/qa_acceptance/ground_truth_hinh_anh_rac.json):

### 5.1. Thống kê kích thước Bounding Box bất thường trong Ground Truth
- **Box chiếm $\ge 25\%$ diện tích ảnh:** **27 / 57 vật thể (chiếm 47.4%)**!
- **Box chiếm $> 80\%$ diện tích ảnh:** **9 / 57 vật thể (chiếm 15.8%)**!
- **Box chiếm 100% toàn bộ khung hình ảnh:** **4 ảnh** gồm:
  1. `download (2).jpg`: box `[0, 0, 299, 168]` = **100.0%** diện tích ("Đống rác thải y tế gồm bao bì túi nhựa và chai dịch truyền").
  2. `images (13).jpg`: box `[0, 0, 538, 371]` = **100.0%** diện tích ("Đống chai nhựa PET tái chế").
  3. `images (7).jpg`: box `[0, 0, 538, 371]` = **100.0%** diện tích (file trùng lặp SHA-256 với images (13)).
  4. `OIP.webp`: box `[0, 0, 274, 234]` = **100.0%** diện tích ("Cận cảnh đống quần áo cũ chất thành đống").

### 5.2. Các sai lệch về định nghĩa vật thể và gán nhãn trong Ground Truth
1. **Lồng ghép đa tầng (Hierarchical Nesting Confusion):**
   - Trong `download (2).jpg`, nhãn viên vừa vẽ 2 box nhỏ cho từng chai dịch truyền y tế (`[112, 10, 168, 62]` và `[140, 50, 203, 112]`), lại **vừa vẽ thêm 1 box khổng lồ 100% trùm lên toàn bộ ảnh**.
   - Trong `base64-1717231151649602366964.jpeg`, nhãn viên vẽ từng chai bia, nhưng đồng thời vẽ 1 box `[0, 195, 2000, 1334]` (chiếm **85.4%** diện tích) gán nhãn `glass` cho "Bãi phế liệu mảnh kính vỡ trải rộng toàn bộ mặt đất".
   - *Hậu quả:* Một detector phát hiện chính xác từng chai bia/chai nhựa sẽ bị phạt là **False Negative** đối với box 100%, hoặc nếu detector phát hiện cả bãi thì các box nhỏ bên trong sẽ bị NMS hoặc overlapping phạt.
2. **Nhầm lẫn giữa Thùng rác công cộng/giỏ đựng rác và Rác thải:**
   - Trong `images (18).jpg`: Một **Thùng rác công cộng đô thị màu xanh lá có bánh xe đẩy** ("HÃY BỎ RÁC VÀO THÙNG") được vẽ box chiếm **86.5%** diện tích và gán nhãn là rác thải `plastic`!
   - Trong `images (8).jpg`: Một **Giỏ/sọt đựng rác văn phòng bằng nhựa** được vẽ box chiếm **91.1%** diện tích và gán nhãn là rác thải `plastic`!
   - *Hậu quả:* Thùng rác là công trình tiện ích công cộng, không phải rác thải cần nhặt bỏ. Detector bỏ qua thùng rác sẽ bị tính là FN, còn nếu detector bắt thùng rác thì thực tế ứng dụng sẽ báo động giả khi thấy thùng rác ngoài phố.

---

## 6. DANH SÁCH FILE ĐÃ SỬA VÀ CÁC CHỨC NĂNG CHƯA ĐẠT YÊU CẦU (VẤN ĐỀ 8 & 9)

### 6.1. Danh sách chính xác các file mã nguồn đã chỉnh sửa
1. [`src/detection/two_stage.py`](../src/detection/two_stage.py):
   - Đặt `scoring_mode = "product"` làm mặc định an toàn.
   - Thêm ngưỡng chặn `allow_blend` và giới hạn nhân trần `max_allowed` để triệt tiêu việc closed-world classifier tự kích điểm cho các vùng nền phẳng.
2. [`src/detection/tiling.py`](../src/detection/tiling.py):
   - Thêm bộ lọc loại bỏ anchor proposal chiếm gần trọn tile ($\ge 92\%$ width và $\ge 90\%$ height) nhằm triệt tiêu box ảo sinh ra từ tile cắt vào tường nhà, biển hiệu phẳng.
3. [`streamlit_app.py`](../streamlit_app.py):
   - Chuyển chế độ khuyến nghị mặc định sang `product`.
   - Nâng ngưỡng `tiling_min_side` từ 640 lên 800 để tránh cắt vụn ảnh kích thước trung bình.
4. [`tests/qa/test_evaluator_synthetic_verification.py`](../tests/qa/test_evaluator_synthetic_verification.py):
   - Bổ sung unit test toán học kiểm chứng tính nhất quán của bộ đánh giá.

### 6.2. Các chức năng và tiêu chí VẪN CHƯA ĐẠT YÊU CẦU NGHIỆM THU
| Hạng mục | Hiện trạng | Mục tiêu nghiệm thu | Đánh giá | Nguyên nhân |
|---|:---:|:---:|:---:|---|
| **Recall tổng thể trên holdout** | **14.04%** (8/57) | $\ge 60.0\%$ | **CHƯA ĐẠT** | Mô hình detector gốc (SSDLite) quá yếu trên các vật thể nhỏ xa camera; chế độ `product` triệt tiêu điểm rác thật có detector score thấp. |
| **F1-Score tổng thể** | **23.53%** | $\ge 60.0\%$ | **CHƯA ĐẠT** | Recall thấp kéo F1 xuống sâu dù Precision đạt 72.7%. |
| **Phân loại rác hữu cơ (`biological`)** | **0.0%** AP | $\ge 50.0\%$ | **CHƯA ĐẠT** | Tập train thiếu rác hữu cơ thực tế; texture lá cây/thức ăn dễ nhầm với nền đất. |
| **Phát hiện chai lọ thủy tinh nhỏ** | **< 15.0%** Recall | $\ge 60.0\%$ | **CHƯA ĐẠT** | Độ phân giải 320x320 làm mất chi tiết cổ chai, nắp chai. |
| **Khả năng chạy thời gian thực (FPS)** | **~6.6 FPS** (151 ms) | $\ge 15.0$ FPS | **CHƯA ĐẠT** | SSDLite chạy qua chuỗi Tiling + Two-Stage MobileNet tốn 151 ms. Chỉ có YOLOv8n đơn tầng (48 ms ~ 20 FPS) đạt yêu cầu về tốc độ. |

---

## 7. KẾT LUẬN VÀ LỘ TRÌNH KỸ THUẬT

1. **Khẳng định tính trung thực:**
   - Chúng tôi **không khẳng định bản sửa nhanh (Quick Fix) đã nghiệm thu thành công toàn diện dự án**, vì mặc dù Precision tăng lên 72.7% và FP giảm 62.5%, Recall đã sụt giảm từ 22.8% xuống 14.0% và F1 giảm từ 33.3% xuống 23.5%.
   - Đây là giới hạn toán học tất yếu của phương pháp Two-Stage hậu xử lý: Khi mô hình tầng 1 (Detector) quá yếu, việc siết chặt tầng 2 (Classifier) bằng Bayesian Product sẽ triệt tiêu FP nhưng đồng thời loại bỏ các rác thật có điểm detector thấp.
2. **Giải pháp duy nhất để đạt nghiệm thu:**
   - Không thể tiếp tục "vá" bằng logic rule-based ở tầng inference.
   - Cần huấn luyện lại mô hình Detector gốc (**YOLOv8n / YOLOv8s**) trên bộ dữ liệu `real_v2` với các mẫu âm đường phố (hard negatives) và mosaic multi-scale ở kích thước `imgsz = 640` để nâng Recall lên $\ge 60\%$ trong khi vẫn giữ Precision cao và tốc độ $\ge 20\text{ FPS}$.
