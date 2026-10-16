# TỪ ĐIỂN MIỀN VÀ MÔ HÌNH HÓA ĐỊNH DANH RÁC THẢI (CNTT-KLCN155)
> Hệ thống phân loại 10 lớp, quy tắc phân định ranh giới (Domain Modeling) và tiêu chuẩn nhãn thực tế.

---

## 1. HỆ THỐNG 10 LỚP RÁC CHUẨN HOÁ (TAXONOMY)

Thứ tự 10 lớp cố định trong toàn bộ pipeline (Index 0..9):

| Class ID | Tên tiếng Anh | Tên tiếng Việt | Nhóm bản chất | Định nghĩa kỹ thuật chuẩn |
|---|---|---|---|---|
| **0** | `battery` | Pin / Ắc quy | **Vật thể độc hại** | Pin tiểu (AA, AAA), pin vuông 9V, pin cúc áo, pin lithium, ắc quy xe máy, cell pin laptop. Vỏ kim loại/nhựa nhưng chứa hóa chất độc hại, ưu tiên định danh vật thể. |
| **1** | `biological` | Rác hữu cơ | **Hữu cơ phân hủy** | Thức ăn thừa, vỏ rau củ quả, cơm nguội, lá cây, hoa héo, xác sinh vật có khả năng ủ compost. Không bao gồm bao bì bọc ngoài. |
| **2** | `cardboard` | Bìa carton | **Vật liệu xenlulo** | Thùng giấy carton sóng (corrugated), hộp carton đóng gói hàng, bìa cứng kraft nhiều lớp. |
| **3** | `clothes` | Quần áo / Vải dệt | **Vật thể dệt may** | Áo sơ mi, áo thun, quần jean, áo khoác, khăn, vải vụn vứt bỏ. Không bao gồm giày dép. |
| **4** | `glass` | Thủy tinh | **Vật liệu khoáng** | Chai bia thủy tinh, chai rượu, hũ lọ thủy tinh, ly cốc thủy tinh, mảnh vỡ thủy tinh. |
| **5** | `metal` | Kim loại | **Vật liệu vô cơ** | Lon bia/nước ngọt nhôm, nắp chai kim loại, hộp sữa đặc, đồ dùng sắt thép phế liệu, dây kẽm. |
| **6** | `paper` | Giấy | **Vật liệu xenlulo** | Giấy viết, sách báo cũ, tài liệu, tờ rơi, cốc giấy dùng 1 lần, bao bì giấy mỏng 1 lớp. |
| **7** | `plastic` | Nhựa | **Vật liệu polymer** | Chai nhựa PET, nắp chai nhựa, ly nhựa trà sữa, ống hút, hộp xốp EPS, màng bọc nilon, túi nilon rỗng. |
| **8** | `shoes` | Giày dép | **Vật thể mang mặc** | Giày thể thao, dép quai hậu, dép lê, xăng-đan, ủng cao su. Bất kể vật liệu là cao su, da hay vải. |
| **9** | `trash` | Rác thải hỗn hợp | **Nhóm vét / Tạp chất** | Rác vô cơ không tái chế được, đầu lọc thuốc lá, khẩu trang y tế, tã bỉm, túi rác đen buộc kín chứa nhiều loại rác lẫn lộn. |

---

## 2. QUY TẮC PHÂN ĐỊNH RANH GIỚI (DISAMBIGUATION RULES)

### Quy tắc 1: Ưu tiên Vật thể cụ thể hơn Vật liệu nền (Object > Material)
- **Pin (`battery`)**: Dù pin có vỏ bằng kim loại (`metal`) hoặc bọc nhựa (`plastic`), nhãn BẮT BUỘC là `battery` (Class 0). Tuyệt đối không gán `metal` cho pin 9V hay cell pin laptop.
- **Giày dép (`shoes`)**: Dù dép tông làm bằng nhựa dẻo/cao su (`plastic`) hay giày vải (`clothes`), nhãn BẮT BUỘC là `shoes` (Class 8).
- **Quần áo (`clothes`)**: Quần áo làm từ sợi polyester (bản chất polymer) vẫn BẮT BUỘC là `clothes` (Class 3), không được gán `plastic`.

### Quy tắc 2: Thùng chứa rác / Thiết bị đô thị KHÔNG PHẢI LÀ RÁC (Receptacle != Waste)
- **Thùng rác công cộng (`utility_receptacle`)**: Thùng rác 2 bánh xe, thùng rác inox công viên, thùng composite đô thị là **Hạ tầng thu gom** (Municipal Infrastructure), KHÔNG PHẢI rác cần nhặt.
  - Phân loại: Xếp vào tập **Hard-Negative (Cảnh nền)** hoặc nhãn riêng `trash_bin` nếu mở rộng hệ thống.
  - Tuyệt đối KHÔNG gán nhãn `plastic` hay `trash` cho thùng rác công cộng.
- **Sọt rác / Giỏ rác văn phòng**: Là đồ nội thất văn phòng, thuộc cảnh nền Negative.

### Quy tắc 3: Bao rác / Túi nilon chứa rác (Bagged Waste)
- **Túi rác buộc kín / Bao tải chứa rác hỗn hợp**: Gán nhãn `trash` (Class 9) vì bên trong là hỗn hợp không thể phân tách thành từng vật liệu riêng.
- **Túi nilon rỗng / Bao bì nhựa phế thải**: Gán nhãn `plastic` (Class 7).

### Quy tắc 4: Bao bì phức hợp đa lớp (Composite Packaging)
- **Hộp sữa tươi (Tetra Pak / Combibloc)**: Cấu tạo 75% giấy bìa cứng, 20% màng PE, 5% nhôm.
  - Quy ước chuẩn hóa: Gán nhãn `cardboard` (Class 2) do kết cấu dạng hộp bìa cứng chịu lực, hoặc `paper` nếu quy chuẩn địa phương yêu cầu giấy tái chế. Trong dự án CNTT-KLCN155, thống nhất là `cardboard`.

### Quy tắc 5: Vật thể đơn lẻ vs. Đống rác hỗn hợp (Instance vs. Waste Pile)
- **Nhiệm vụ Object Detection (Bounding Box)**: Chỉ áp dụng cho các vật thể riêng biệt có ranh giới rõ ràng (chai, lon, hộp, giày, pin...).
- **Đống rác bãi rác khổng lồ / Thảm kính vỡ / Đống compost**:
  - Không thể vẽ bounding box chính xác bao toàn bộ khung hình (100% canvas).
  - Quy chuẩn: Nếu đống rác chiếm > 50% ảnh và không thể phân rã vật thể, chuyển sang bài toán **Semantic Segmentation** hoặc nhãn vùng `waste_pile`.
  - Không đưa box 100% toàn ảnh vào đánh giá mAP của mô hình Object Detection vật thể đơn lẻ.

---

## 3. ĐỊNH NGHĨA TẬP DỮ LIỆU HARD-NEGATIVE

Tập Negative là những ảnh **hoàn toàn không chứa rác**, có nhãn rỗng (file `.txt` rỗng 0 bytes):
1. **Kiến trúc công trình**: Tường gạch, đá ốp, mặt tiền tòa nhà, biển hiệu quảng cáo.
2. **Hạ tầng đô thị**: Tủ kỹ thuật điện màu xanh, cột đèn, nắp cống sạch, vỉa hè lát gạch sạch, bậc tam cấp bê tông.
3. **Thùng chứa rác**: Thùng rác công cộng sạch, giỏ rác để bàn văn phòng rỗng.
4. **Nội thất / Vật dụng sinh hoạt**: Bàn gỗ, thảm trải sàn, sofa, kệ sách.

Mục đích: Huấn luyện detector học cách kích hoạt background (Class 0 trong SSDLite) và không xuất bounding box ma (False Positives) trên cảnh nền.
