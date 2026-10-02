# TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH
## KHOA CÔNG NGHỆ THÔNG TIN
---

### TỜ TRÌNH XIN ĐIỀU CHỈNH VÀ MỞ RỘNG ĐỀ CƯƠNG KHÓA LUẬN TỐT NGHIỆP

- **Kính gửi:** ThS. Huỳnh Thị Châu Lan – Giảng viên hướng dẫn
- **Sinh viên thực hiện:** Ngô Thanh Nhân
- **Đề tài:** Nhận diện và phân loại rác thải phục vụ phân loại rác tại nguồn sử dụng học sâu
- **Mã đề tài:** CNTT-KLCN155
- **Thời gian lập tờ trình:** Ngày 02 tháng 10 năm 2026

---

### 1. NỘI DUNG ĐỀ XUẤT ĐIỀU CHỈNH

Trong đề cương khóa luận được phê duyệt ban đầu:
- Giai đoạn 1 (Phân loại ảnh đơn vật thể): Nghiên cứu và huấn luyện mô hình MobileNetV3 trên **10 nhóm rác thải**.
- Giai đoạn 2 (Phát hiện đa vật thể - Object Detection): Đề cương ban đầu định hướng **6 nhóm vật liệu chính** (Nhựa, Giấy, Kim loại, Thủy tinh, Hữu cơ, Chất thải nguy hại).

**Đề xuất điều chỉnh:**
Kính xin Giảng viên hướng dẫn cho phép **đồng bộ hóa và mở rộng bài toán Phát hiện đa vật thể (Detection) lên 10 nhóm lớp chi tiết**, thống nhất hoàn toàn với mô hình Phân loại (Classification) giai đoạn 1, bao gồm:
1. `0: battery` (Pin / Chất thải nguy hại nhỏ)
2. `1: biological` (Rác thải hữu cơ / thức ăn thừa)
3. `2: cardboard` (Bìa carton / giấy cứng)
4. `3: clothes` (Vải / quần áo thải bỏ)
5. `4: glass` (Thủy tinh / chai lọ vỡ)
6. `5: metal` (Kim loại / lon nhôm / sắt hộp)
7. `6: paper` (Giấy báo / túi giấy / khăn giấy)
8. `7: plastic` (Nhựa tái chế / chai PET / túi nilon)
9. `8: shoes` (Giày dép phế phẩm)
10. `9: trash` (Rác vô cơ khác / chất thải sinh hoạt không tái chế)

---

### 2. CƠ SỞ KHOA HỌC VÀ THỰC TIỄN CỦA ĐỀ XUẤT

1. **Tính liền mạch trong chuyển giao tri thức (Transfer Learning):**
   - Mô hình chính được xác định theo đề cương là **SSDLite320-MobileNetV3**.
   - Khi giữ nguyên 10 lớp, toàn bộ 296 trọng số đặc trưng (backbone features) được huấn luyện từ mô hình phân loại MobileNetV3 (giai đoạn 1) được nạp trực tiếp sang bộ trích xuất đặc trưng của SSDLite320 với độ sai lệch tuyệt đối bằng 0 (`diff = 0.0`).
   - Nếu ép về 6 lớp, các đặc trưng riêng biệt của các lớp như `battery` (nguy hại), `clothes` (vải sợi), `shoes` (cao su/da) sẽ bị thất thoát hoặc nhập nhằng ngữ nghĩa.

2. **Tính thực tiễn đối với phân loại rác sinh hoạt tại TP. Hồ Chí Minh:**
   - Theo quy định phân loại chất thải rắn sinh hoạt tại nguồn (Luật Bảo vệ Môi trường 2020), rác thải cần được nhận diện chi tiết hơn là các vật liệu gộp chung.
   - Việc tách riêng `battery` (pin cũ) giúp hệ thống cảnh báo nguy cơ cháy nổ và ô nhiễm kim loại nặng; tách riêng `cardboard` và `paper` giúp định giá thu mua phế liệu chính xác; tách riêng `clothes` và `shoes` hỗ trợ các chương trình tái chế sợi dệt và giày dép cũ.

3. **Cấu trúc mô hình và tính công bằng trong thực nghiệm:**
   - **Mô hình chính (Primary Detector):** SSDLite320-MobileNetV3 (10 lớp đối tượng + 1 lớp nền background = 11 logits).
   - **Mô hình đối chứng (Benchmark Detector):** YOLOv8n (10 lớp đối tượng).
   - Cả hai mô hình đều được huấn luyện trên cùng một bộ dữ liệu, cùng chia tập chống rò rỉ (Zero-Leakage Split), và cùng được đánh giá qua bộ công cụ chuẩn hóa COCO metrics (`mAP@0.5`, `mAP@0.5:0.95`).

---

### 3. HIỆN TRẠNG DỮ LIỆU VÀ KẾ HOẠCH BỔ SUNG MINH BẠCH

Quá trình kiểm toán dữ liệu độc lập (Data Audit) đợt 2 cho thấy hiện trạng phân bố 10 lớp như sau:

| Nhóm lớp | Số lượng mẫu thực tế ngoài trời | Đánh giá khả năng đánh giá trên tập kiểm thử độc lập |
| :--- | :---: | :--- |
| **7 nhóm đầy đủ:** `biological`, `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash` | $\ge 5$ bối cảnh thực tế ngoài trời độc lập | **Đầy đủ điều kiện:** Đủ chia Train / Val / Test độc lập, không rò rỉ bối cảnh. |
| **2 nhóm hạn chế:** `battery`, `shoes` | 2 bối cảnh thực tế ngoài trời | **Hạn chế:** Chỉ có 2 cụm ảnh thực tế ngoài trời (hiện chia 1 Train - 1 Test hoặc 1 Train - 1 Val). |
| **1 nhóm thiếu ảnh thực:** `clothes` | 0 bối cảnh rác thực tế ngoài trời | **Chưa đủ ảnh thực:** Dữ liệu synthetic có 520 boxes, nhưng toàn bộ ảnh candidate ngoài đời thực trước đó bị từ chối trong quá trình kiểm duyệt vì là quần áo người mặc / tủ đồ, không phải rác thải vứt bỏ. |

**Cam kết khắc phục và lộ trình thực hiện:**
1. **Tuần 6–7:** Sinh viên tổ chức đợt thu thập ảnh thực tế bổ sung tại các điểm thu gom rác, bãi phế liệu và khuôn viên trường học trên địa bàn TP.HCM bằng camera điện thoại, tập trung vào 3 nhóm: Pin cũ (`battery`), Quần áo cũ bỏ đi (`clothes`), Giày dép hỏng (`shoes`).
2. **Gán nhãn và kiểm định chất lượng:** Sử dụng nền tảng CVAT và bộ kiểm toán chất lượng nội bộ (Review Tool) để gán nhãn bounding box chuẩn xác, kiểm duyệt chéo 2 lượt trước khi đưa vào tập Test chính thức.
3. **Nguyên tắc khoa học:** Không dùng tập dữ liệu chưa hoàn thiện để công bố kết quả mAP thiếu trung thực; mọi báo cáo đều nêu rõ số lượng ảnh thật và ảnh synthetic cho từng lớp.

---

### 4. KẾT LUẬN VÀ KIẾN NGHỊ

Việc mở rộng bài toán Detection lên 10 lớp không chỉ tăng tính thực tiễn và tính hiện đại cho đề tài mà còn giữ vững tính nhất quán phương pháp luận xuyên suốt giữa Giai đoạn 1 và Giai đoạn 2 của khóa luận.

Kính mong ThS. Huỳnh Thị Châu Lan xem xét, phê duyệt định hướng điều chỉnh này để sinh viên có cơ sở vững chắc tiếp tục triển khai các nội dung thực nghiệm tiếp theo.

*Sinh viên thực hiện*  
*(Đã ký)*  
**Ngô Thanh Nhân**
