# TRƯỜNG ĐẠI HỌC CÔNG THƯƠNG TP. HỒ CHÍ MINH
## KHOA CÔNG NGHỆ THÔNG TIN
---

### TỜ TRÌNH XIN ĐIỀU CHỈNH VÀ MỞ RỘNG TAXONOMY ĐỀ TÀI KHÓA LUẬN TỐT NGHIỆP

- **Kính gửi:** ThS. Huỳnh Thị Châu Lan – Giảng viên hướng dẫn
- **Sinh viên thực hiện:** Ngô Thanh Nhân (MSSV: 200123059)
- **Tên đề tài chính thức (theo Đề cương được duyệt):** *Xây dựng hệ thống phát hiện và phân loại đa đối tượng rác thải sinh hoạt trong ảnh chụp thực tế bằng mô hình học sâu nhẹ và kỹ thuật tăng cường dữ liệu*
- **Mã đề tài:** CNTT-KLCN155
- **Thời gian lập tờ trình:** Ngày 02 tháng 10 năm 2026

---

### 1. NỘI DUNG ĐỀ XUẤT ĐIỀU CHỈNH

Trong đề cương khóa luận được phê duyệt ban đầu:
- Mục tiêu chính của đề tài là xây dựng hệ thống phát hiện đa đối tượng (Multi-Object Detection) trên ảnh chụp thực tế, định hướng tập trung vào **6 nhóm vật liệu rác thải sinh hoạt**.
- Trong quá trình triển khai thực nghiệm tiền trạm (Phần 1), nhóm nghiên cứu đã xây dựng bộ phân loại đơn vật thể MobileNetV3 trên **10 danh mục chi tiết** nhằm khai phá tối đa đặc trưng miền hình ảnh rác thải.

**Nội dung đề xuất điều chỉnh:**
Kính xin Giảng viên hướng dẫn cho phép **mở rộng bài toán Phát hiện đa vật thể (Object Detection) từ 6 nhóm vật liệu lên 10 nhóm lớp chi tiết**, bao gồm:
1. `0: battery` (Pin cũ / Rác nguy hại sinh hoạt nhỏ)
2. `1: biological` (Rác hữu cơ / thức ăn thừa / phụ phẩm nông sản)
3. `2: cardboard` (Bìa carton / thùng carton cứng)
4. `3: clothes` (Vải vụn / quần áo cũ phế phẩm)
5. `4: glass` (Thủy tinh / chai lọ vỡ)
6. `5: metal` (Kim loại / vỏ lon nhôm / đồ sắt hộp)
7. `6: paper` (Giấy báo / túi giấy / giấy văn phòng)
8. `7: plastic` (Nhựa tái chế / chai PET / hộp xốp / túi nilon)
9. `8: shoes` (Giày dép phế phẩm / cao su / da tổng hợp)
10. `9: trash` (Rác vô cơ khác / chất thải sinh hoạt không tái chế)

---

### 2. CƠ SỞ KHOA HỌC VÀ THỰC TIỄN CỦA ĐỀ XUẤT

1. **Nhu cầu phân loại rác chi tiết tại nguồn theo thực tiễn TP. Hồ Chí Minh:**
   - Theo Luật Bảo vệ Môi trường 2020 và hướng dẫn kỹ thuật của Bộ Tài nguyên và Môi trường về phân loại chất thải rắn sinh hoạt tại nguồn, rác thải cần được quản lý theo luồng xử lý và giá trị tái chế thực tế.
   - Việc phân định 10 lớp mang lại giá trị ứng dụng vượt trội:
     + Tách riêng `battery` (pin): Cảnh báo khẩn cấp rác nguy hại có độc tố kim loại nặng và nguy cơ cháy nổ cao trong thu gom rác.
     + Tách riêng `cardboard` (carton) và `paper` (giấy): Phục vụ phân luồng thu mua phế liệu công nghiệp và định giá chính xác.
     + Tách riêng `clothes` (quần áo) và `shoes` (giày dép): Phục vụ các dự án quyên góp từ thiện, tái chế sợi dệt và công nghiệp cao su/da.

2. **Kế hoạch dữ liệu chi tiết và khả thi:**
   - Nhóm nghiên cứu đã chuẩn hóa toàn bộ cấu trúc nhãn bounding box 10 lớp (1.562 file nhãn, 5.750 bounding boxes).
   - Đã khảo sát và trích xuất thành công 185 polygon segmentation masks chuẩn COCO từ tập train của TACO phục vụ kỹ thuật tăng cường dữ liệu Simple Copy-Paste (Ghiasi et al., 2021) [5] cho bài toán phát hiện đối tượng.

3. **Cấu trúc mô hình và tính công bằng trong thực nghiệm:**
   - **Mô hình chính (Primary Detector):** SSDLite320-MobileNetV3 (10 lớp đối tượng + 1 lớp nền = 11 logits).
   - **Mô hình đối chứng (Benchmark Detector):** YOLOv8n (10 lớp đối tượng).
   - Cả hai mô hình đều được huấn luyện trên cùng một bộ dữ liệu, cùng chia tập độc lập chống rò rỉ (Zero-Leakage Split), và cùng được đánh giá qua chuẩn đo lường COCO metrics (`mAP@0.5`, `mAP@0.5:0.95`).

---

### 3. HIỆN TRẠNG DỮ LIỆU VÀ KẾ HOẠCH THU THẬP THỰC ĐỊA MINH BẠCH

Quá trình kiểm toán dữ liệu độc lập (Data Audit) cho thấy:
- **Tập Train (1.317 ảnh, 4.280 boxes):** Đã có 9 lớp có số lượng boxes dồi dào ($\ge 436$ boxes/lớp), riêng lớp `shoes` có 1 box.
- **Tập Val và Test hiện tại (mỗi tập 5 ảnh, 45 boxes):** Mới chỉ mang tính chất kiểm tra luồng kỹ thuật (smoke test); hoàn toàn chưa đủ độ tin cậy thống kê để đánh giá chất lượng mô hình chính thức. Đặc biệt còn thiếu cảnh rác thực tế ngoài trời cho 3 lớp: `battery`, `clothes`, `shoes`.
- Lưu ý nguyên tắc học thuật: Số lượng "$\ge 5$ boxes" hay "$\ge 5$ cảnh" chỉ đủ điều kiện cho kiểm tra luồng (smoke validation), tuyệt đối không được coi là bằng chứng đủ để đánh giá chất lượng mô hình một cách đáng tin cậy.

**Kế hoạch thu thập thực địa và chuẩn bị Test Set độc lập:**
1. **Đối tượng thu thập cụ thể:**
   - `battery`: Chụp tại các điểm tiếp nhận pin cũ tại TP.HCM (siêu thị Co.opmart, Go!, các cơ sở của trường HUIT, sảnh chung cư).
   - `clothes`: Chụp các bọc quần áo cũ, vải vụn tại các điểm tập kết rác dân cư, thùng thu gom từ thiện hoặc ven đường.
   - `shoes`: Chụp giày dép cũ hỏng, dép xốp, dép tổ ong vứt bỏ ngoài môi trường tự nhiên, bãi rác lộ thiên.
2. **Quy mô mục tiêu:** Thu thập 200–300 ảnh thực tế tại TP.HCM phân bổ theo cảnh độc lập (scene-based), bảo đảm tối thiểu 20–30 cảnh độc lập cho mỗi lớp trên tập kiểm thử chính thức (Final Held-out Test Set).
3. **Quy trình gán nhãn:** Sử dụng nền tảng CVAT kết hợp bộ công cụ Review Tool kiểm duyệt chéo hai lượt (Double-blind Annotation Verification).

---

### 4. KẾT LUẬN VÀ KIẾN NGHỊ

Việc mở rộng bài toán Detection lên 10 lớp đáp ứng nhu cầu phân loại rác thải thực tế tại Việt Nam, đồng thời kế thừa có hệ thống công sức chuẩn bị dữ liệu và hạ tầng kỹ thuật đã xây dựng.

Kính mong ThS. Huỳnh Thị Châu Lan xem xét, phê duyệt định hướng điều chỉnh này để nhóm nghiên cứu tiếp tục triển khai các bước thực nghiệm tiếp theo theo đúng kế hoạch.

*Sinh viên thực hiện*  
*(Đã ký)*  
**Ngô Thanh Nhân**
