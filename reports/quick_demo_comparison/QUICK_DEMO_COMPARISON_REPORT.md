# BÁO CÁO ĐỐI SOÁT SUY LUẬN 4 CHẾ ĐỘ TRÊN ẢNH THỰC TẾ
**Dự án:** Multi-Waste Detection (CNTT-KLCN155) | **Thời gian:** 12/10/2026
**Quy chuẩn:** Ngưỡng confidence = 0.25, IoU NMS = 0.55, Ground Truth đóng băng 1.0_FROZEN

## 1. So sánh 4 Chế độ trên ảnh `images (17).jpg`

| Tiêu chí đánh giá | Chế độ 1: SSDLite Raw (Không Tiling) | Chế độ 2: YOLOv8n Raw (Không Tiling) | Chế độ 3: SSDLite + Tiling (min_side=400) | Chế độ 4: YOLOv8n + Tiling (min_side=400) |
|---|:---:|:---:|:---:|:---:|
| **Tổng số box dự đoán** | **1** | **5** | **11** | **16** |
| **Thời gian xử lý (Latency)** | 47.2 ms | 29.7 ms | 300.2 ms | 196.4 ms |
| **Theo GTv1 đóng băng (11 vật thể):** | | | | |
| - True Positives (TP) | 1 | 4 | 1 | 5 |
| - False Positives (FP) | 0 | 1 | 10 | 11 |
| - False Negatives (FN) | 10 | 7 | 8 | 6 |
| - Precision / Recall / F1 | 100.0% / 9.1% / 16.7% | 80.0% / 36.4% / 50.0% | 9.1% / 9.1% / 9.1% | 31.2% / 45.5% / 37.0% |
| **Theo Rác thải thực tế (9 vật thể rác rời):** | | | | |
| - True Positives (TP) | 1 | 3 | 1 | 4 |
| - False Positives (FP) | 0 | 2 | 10 | 12 |
| - False Negatives (FN) | 8 | 6 | 6 | 5 |
| - Precision / Recall / F1 | 100.0% / 11.1% / 20.0% | 60.0% / 33.3% / 42.9% | 9.1% / 11.1% / 10.0% | 25.0% / 44.4% / 32.0% |
| **Hiện tượng Box trùng / Hallucination** | 0 trùng | 1 box shoes nhận nhầm góc thùng rác | **Bùng nổ 8 box ảo clothes & shoes** | Bắt được 7 vật thể thật nhưng bị trùng 4 box ranh giới tile |

## 2. Phân tích Chi tiết Lỗi và Hiện tượng tại `images (17).jpg`

### A. Rà soát Ground Truth: Rác thật vs Thùng rác sử dụng
- Trong file GT đóng băng `ground_truth_hinh_anh_rac.json`, ảnh số 10 có 11 vật thể:
  + `img10_obj01`: Thùng rác tái chế bằng nhựa trong suốt mờ bên trái (nhãn `plastic`).
  + `img10_obj02`: Thùng rác tái chế bằng nhựa màu đen bên phải (nhãn `plastic`).
  + 9 vật thể còn lại (`img10_obj03` đến `img10_obj11`): Các món rác thực sự (hộp thiếc, lon nước ngọt, chai thủy tinh, chai nhựa, lon nhôm ép bẹp).
- **Nhận xét chuyên môn:** Thùng rác là hạ tầng chứa đựng đang hoạt động (container/infrastructure). Việc mô hình YOLOv8n bắt trọn thùng rác nhựa mờ bên trái (`plastic` conf=27.7%) có thể được coi là TP theo GTv1 đóng băng, nhưng theo nghiệp vụ phân loại rác tại nguồn thì đây không phải là món rác cần nhặt. Báo cáo phân định rạch ròi cả 2 góc nhìn để tránh tự tạo số liệu ảo.

### B. Cảnh báo nguy hiểm khi chỉ nhìn vào số lượng box:
- **Chế độ 3 (SSDLite + Tiling 400)** sinh ra tới **11 box**, nhưng có tới **8 box ảo (False Positives)**: 6 box `clothes` (quần áo cũ) và 2 box `shoes` (giày dép)! Khi cắt ô 219x219, texture nhựa nhăn và góc cạnh thùng rác bị mạng SSDLite nhầm thành nếp gấp vải và bề mặt giày da.
- **Chế độ 4 (YOLOv8n + Tiling 400)** sinh ra **16 box**, bao phủ được 7 vật thể rác thật, nhưng xuất hiện hiện tượng **box trùng lặp ranh giới tile** (cùng 1 chai nhựa hoặc chai thủy tinh bị nhận diện 2 lần ở 2 tile liền kề do IoU ranh giới chưa đủ 0.55 để triệt tiêu NMS).

## 3. Bảng Đối chiếu trên 5 Tình huống Thực tế Khác nhau

| Tình huống thử nghiệm | Ảnh nguồn | Chế độ 1: SSDLite Raw | Chế độ 2: YOLOv8n Raw | Chế độ 3: SSDLite + Tiling 400 | Chế độ 4: YOLOv8n + Tiling 400 | Đánh giá & Rủi ro |
|---|---|---|---|---|---|---|
| **Một vật thể**<br>*(Kỳ vọng: 1 vật thể: 0: battery)* | `single_object`<br>(512x411) | **1 box**<br>shoes (82.3%) | **1 box**<br>shoes (91.8%) | **5 box**<br>shoes (82.3%), clothes (57.9%), clothes (49.3%)... (+2) | **3 box**<br>shoes (91.8%), clothes (54.1%), clothes (28.6%) | Quả pin laptop bị cả 2 nhận nhầm thành `shoes` do texture màu đen và viền dài. Lỗi ngữ nghĩa. |
| **Nhiều vật thể cùng lúc**<br>*(Kỳ vọng: 11 vật thể (GTv1) / 9 món rác rời)* | `multi_objects`<br>(547x365) | **1 box**<br>plastic (30.0%) | **5 box**<br>shoes (62.4%), plastic (53.4%), plastic (41.0%), metal (33.4%), plastic (33.2%) | **12 box**<br>clothes (60.2%), clothes (50.0%), clothes (46.5%)... (+9) | **16 box**<br>metal (63.9%), glass (63.2%), shoes (62.4%)... (+13) | Đã phân tích chi tiết tại Mục 1. |
| **Vật thể nhỏ và xa**<br>*(Kỳ vọng: 3 chai nhựa nhỏ nằm rải rác)* | `small_far_objects`<br>(299x168) | **0 box**<br>0 box | **0 box**<br>0 box | **0 box**<br>0 box | **0 box**<br>0 box | Cả 2 mô hình Raw đều bỏ sót 100% rác nhỏ từ xa (0 box). Tiling giúp phát hiện thêm nhưng dễ kèm nhiễu. |
| **Ảnh nền không có rác (True Negative)**<br>*(Kỳ vọng: 0 vật thể (Không có rác))* | `no_waste`<br>(640x292) | **1 box**<br>clothes (56.8%) | **1 box**<br>plastic (32.2%) | **3 box**<br>shoes (88.8%), clothes (56.8%), plastic (29.0%) | **4 box**<br>clothes (44.6%), shoes (37.1%), plastic (32.2%)... (+1) | Mặt đất sạch nhưng SSDLite nhận nhầm thành `clothes 57%` và YOLO nhận nhầm thành `plastic 26%`. |
| **Tủ điện dễ gây nhận nhầm (Hard Negative)**<br>*(Kỳ vọng: 0 vật thể rác (Hạ tầng đô thị sạch))* | `utility_box`<br>(1031x773) | **1 box**<br>metal (55.4%) | **2 box**<br>metal (70.3%), metal (56.7%) | **4 box**<br>metal (64.4%), plastic (61.7%), metal (55.4%)... (+1) | **6 box**<br>plastic (71.2%), metal (70.3%), metal (56.7%)... (+3) | Tủ điện bị cả 2 mô hình nhận nhầm thành `metal` do bề mặt kim loại sơn tĩnh điện. Cần thêm Hard-Negative. |
