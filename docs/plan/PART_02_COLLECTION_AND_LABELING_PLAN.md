# KẾ HOẠCH THU THẬP VÀ GÁN NHÃN DỮ LIỆU RÁC THỰC TẾ (CNTT-KLCN155)

- **Dự án:** CNTT-KLCN155 — Phát hiện và phân loại rác thải đa đối tượng
- **Tác giả:** Tech Lead ML / Nhóm Kỹ sư ML
- **Người nhận:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_COLLECTION_AND_LABELING_PLAN.md`
- **Ngày lập:** 02/10/2026

---

## 1. MỤC TIÊU VÀ ĐỊNH HƯỚNG DỮ LIỆU ĐA RÁC

Từ kết quả kiểm toán tại `PART_02_REAL_DATA_AUDIT.md`, 100% dữ liệu detection hiện tại không phản ánh rác thải thực tế ngoài trời tại Việt Nam (91,97% là ảnh nhân tạo ghép cơ học; 8,03% là ảnh giày đang mang trên chân người từ OpenImages).

Do đó, để xây dựng mô hình phát hiện đa rác (YOLOv8n / SSDLite320) có khả năng hoạt động thực tế trong Part 3, dự án cần triển khai chiến dịch thu thập và gán nhãn dữ liệu rác thực địa tại TP.HCM.

Con số **200 – 300 ảnh** được đề xuất trước đây chỉ là **ngưỡng tối thiểu ban đầu** (Proof-of-Concept). Để đảm bảo độ hội tụ và khả năng khái quát hóa của detector, kế hoạch chuẩn hóa mục tiêu như sau:

| Chỉ tiêu | Ngưỡng tối thiểu (Minimal) | Ngưỡng khuyến nghị (Target) | Ngưỡng tối ưu (Production) |
|---|---|---|---|
| **Số lượng ảnh thực tế** | 300 ảnh | **500 ảnh** | 1.000 ảnh |
| **Tổng số Bounding Box** | $\ge 1.500$ boxes | **$\ge 2.500$ boxes** | $\ge 5.000$ boxes |
| **Mật độ rác trung bình** | 3 – 5 vật thể / ảnh | **4 – 7 vật thể / ảnh** | 5 – 10 vật thể / ảnh |
| **Số box tối thiểu mỗi lớp** | $\ge 100$ boxes / lớp | **$\ge 200$ boxes / lớp** | $\ge 400$ boxes / lớp |

---

## 2. BẢNG PHÂN BỔ MỤC TIÊU THEO 10 LỚP VẬT LIỆU

| Lớp (Class) | ID | Đối tượng thu thập thực tế trọng tâm tại TP.HCM | Số box tối thiểu | Tiêu chuẩn nhận diện |
|---|---|---|---|---|
| `battery` | 0 | Pin tiểu AA, AAA, pin cúc áo, pin sạc dự phòng cũ, pin điện thoại hỏng | 150 | Rác nguy hại, kích thước nhỏ, cần chụp cận cảnh và góc nghiêng |
| `biological` | 1 | Thức ăn thừa, vỏ chuối, vỏ cam, bã mía, cuống rau, bã cà phê | 250 | Rác hữu cơ dễ biến dạng, có dịch ẩm, lẫn lộn |
| `cardboard` | 2 | Thùng carton giao hàng (Shopee, Lazada), hộp bánh pizza, hộp sữa Tetra Pak | 300 | Bìa cứng, gợn sóng, hộp chữ nhật gấp phẳng hoặc méo mó |
| `clothes` | 3 | Áo rách vứt bỏ, giẻ lau, khăn cũ, tất rách, khẩu trang vải bỏ | 200 | Vải vụn, nếp nhăn, dính bẩn; **không chụp người đang mặc áo** |
| `glass` | 4 | Chai bia, chai nước ngọt thủy tinh, lọ tương, mảnh vỡ thủy tinh | 250 | Trong suốt hoặc có màu xanh/nâu, phản quang, có độ bóng bề mặt |
| `metal` | 5 | Lon nước ngọt (Coca, Pepsi, RedBull), lon bia, hộp sữa đặc, nắp chai sắt | 300 | Lon nhôm, hộp sắt tây dập bẹp hoặc nguyên vẹn, ánh kim |
| `paper` | 6 | Hóa đơn, giấy A4 vụn, ly giấy cà phê, tờ rơi quảng cáo, túi xi măng | 250 | Giấy mỏng phẳng hoặc nhàu nát, ly giấy uống nước |
| `plastic` | 7 | Chai nước suối Aquafina/Lavie, hộp xốp cơm, túi nilon, ống hút nhựa, nắp chai | 400 | Chai PET, hộp xốp PS, túi PE, màng bọc nilon (lớp phổ biến nhất) |
| `shoes` | 8 | Giày thể thao cũ vứt bỏ, dép lào rách, dép tổ ong hỏng, sandal đứt quai | 150 | Giày dép phế thải nằm trong thùng rác hoặc ven đường; **không chụp chân người** |
| `trash` | 9 | Đầu lọc thuốc lá, băng dính dính đất, tã bỉm phế thải, gốm sứ vỡ, rác quét nhà | 250 | Rác trơ, rác vô cơ không thể tái chế hoặc quá bẩn |

---

## 3. MA TRẬN ĐA DẠNG BỐI CẢNH (DIVERSITY MATRIX)

### 3.1. Địa điểm thu thập thực tế tại TP.HCM
1. **Khuôn viên trường ĐH Công Thương TP.HCM (HUIT):**
   - Thùng rác phân loại 2 ngăn / 3 ngăn tại các sảnh giảng đường, khu tự học.
   - Thùng rác khu vực căn tin và bãi xe sinh viên.
2. **Khu vực công cộng và đường phố:**
   - Phố đi bộ Nguyễn Huệ, Công viên Tao Đàn, Công viên Gia Định.
   - Các điểm tập kết rác dân cư tại Quận Tân Phú, Bình Thạnh, Quận 1.
3. **Quán ăn vỉa hè và chuỗi đồ uống:**
   - Bàn ăn đồ uống mang đi (takeaway) chứa ly nhựa, ống hút, hộp xốp, khăn giấy dơ.

### 3.2. Điều kiện ánh sáng (Lighting)
- **Ánh sáng tự nhiên buổi sáng (07h00 - 10h00):** 30% tổng số ảnh.
- **Ánh sáng gay gắt ban trưa (11h00 - 14h00, bóng gắt):** 25% tổng số ảnh.
- **Ánh sáng hoàng hôn / bóng râm (15h30 - 17h30):** 25% tổng số ảnh.
- **Ánh sáng nhân tạo / ban đêm (đèn đường, đèn huỳnh quang căn tin):** 20% tổng số ảnh.

### 3.3. Khoảng cách và mức độ che khuất (Scale & Occlusion)
- **Góc chụp cận cảnh (0,3m – 0,8m):** Rác nằm gọn trong khung hình, chụp chi tiết nhãn mác, bề mặt.
- **Góc chụp trung bình (1,0m – 2,0m):** Chụp thùng rác mở nắp hoặc cụm rác 3–7 món rải trên mặt đất.
- **Góc chụp rộng (2,5m – 4,0m):** Toàn cảnh điểm tập kết rác công cộng.
- **Mức độ che khuất:** 
  - 40% ảnh có vật thể đơn lẻ hoặc tiếp xúc nhẹ.
  - 40% ảnh có vật thể bị che khuất một phần ($20\% - 50\%$ diện tích bởi vật khác).
  - 20% ảnh có vật thể biến dạng nặng (lon bị cán bẹp, hộp xốp vỡ vụn, giấy bị vò tròn).

---

## 4. QUY TRÌNH GÁN NHÃN VÀ KIỂM CHUẨN (LABELING PROTOCOL)

1. **Công cụ gán nhãn duy nhất:** Sử dụng ứng dụng nội bộ **`src/ui/review_tool.py`** (Streamlit). Không phụ thuộc dịch vụ bên ngoài để bảo toàn cấu trúc dữ liệu và kiểm soát trực tiếp.
2. **Nguyên tắc bao khung (Bounding Box Rules):**
   - Khung chữ nhật phải ôm sát ranh giới biên ngoài cùng của vật thể (Tight Box).
   - Nếu vật thể bị che khuất một phần nhưng phần nhìn thấy vẫn đủ để nhận diện: Khung bao phủ toàn bộ phần nhìn thấy được.
   - Đối với vật thể tổ hợp (Composite object):
     - Ly cà phê mang đi có nắp và ống hút: Nếu các bộ phận phân tách rõ -> gán box cho ly (`plastic` hoặc `paper`), box nắp (`plastic`), box ống hút (`plastic`). Nếu chụp xa không phân biệt -> gán 1 box cho toàn bộ cụm theo vật liệu chính.
     - Hộp xốp đựng cơm thừa: Khung hộp xốp là `plastic` (PS foam); thức ăn bên trong là `biological`.
3. **Quy trình kiểm soát chất lượng (QA / QC):**
   - **Giai đoạn 1:** Gán nhãn ban đầu (nhãn nháp mang trạng thái `UNREVIEWED`).
   - **Giai đoạn 2:** Kiểm tra chéo (Peer-review). Người thứ hai dùng `review_tool.py` kiểm tra từng box. Nếu đạt -> bấm `APPROVED`; nếu sai sót -> bấm `NEEDS_RELABEL` kèm ghi chú lỗi.
   - **Giai đoạn 3:** Chạy script tự động `scripts/validate_detection_annotations.py` để đảm bảo 100% tọa độ hợp lệ trong $[0, 1]$, không có box rỗng hoặc sai class ID.
4. **Quy tắc phân chia tập Train / Val / Test cho Detection:**
   - Dùng script gom nhóm `scripts/cluster_detection_groups.py` để gán `group_id` cho các ảnh cùng góc chụp hoặc chụp liên tiếp trong 1 phiên.
   - **Toàn bộ ảnh trong 1 `group_id` bắt buộc phải nằm trọn trong cùng 1 split** (chống rò rỉ bối cảnh).
   - Tỷ lệ phân chia: **70% Train - 15% Val - 15% Test**.
   - **Tập Val và Test bắt buộc phải chứa 100% ảnh thật đã được `APPROVED`**, không đưa ảnh Synthetic vào Val/Test.
