# QUY CHUẨN GÁN NHÃN VÀ PHÂN ĐỊNH TAXONOMY 10 LỚP (ANNOTATION POLICY)
**Tiêu chuẩn Kỹ thuật Dán nhãn Dữ liệu Nhận diện Rác Thực tế**  
**Dự án:** `CNTT-KLCN155-waste-detection`  
**Giai đoạn:** Sprint 0 — Chuẩn hóa Domain Modeling & Annotation Policy  
**Tác giả:** Antigravity Integrator (Multi-Agent Team — Chuyên trách: Agent A Data & Agent D QA)  
**Ngày lập:** 2026-10-12  

---

## 1. NGUYÊN TẮC BẤT BIẾN CỦA HỆ THỐNG PHÂN LỚP (TAXONOMY)

Hệ thống tuân thủ nghiêm ngặt **10 lớp rác thải cố định** đã đóng băng trong `src/detection/schema.py`:

```python
DETECTION_CLASS_NAMES = (
    "battery",     # ID 0: Pin / Ắc quy (Rác nguy hại)
    "biological",  # ID 1: Rác sinh học / Hữu cơ
    "cardboard",   # ID 2: Bìa carton
    "clothes",     # ID 3: Quần áo cũ / Vải vụn
    "glass",       # ID 4: Thủy tinh
    "metal",       # ID 5: Kim loại
    "paper",       # ID 6: Giấy
    "plastic",     # ID 7: Nhựa
    "shoes",       # ID 8: Giày dép
    "trash",       # ID 9: Rác hỗn hợp khác
)
```

### 1.1 Ánh xạ nhãn giữa các mô hình
- **YOLOv8n / YOLO26n:** Sử dụng `class_id` từ `0` đến `9`.
- **SSDLite320-MobileNetV3:** Sử dụng `0: background`, các lớp rác Foreground có ID từ `1` đến `10`. Quá trình nạp dữ liệu tự động `+1`, giải mã inference tự động `-1`.
- **Định dạng file nhãn YOLO:** `<class_id> <x_center> <y_center> <width> <height>` (tọa độ chuẩn hóa $[0.0, 1.0]$, làm tròn 6 chữ số thập phân).

---

## 2. QUY TẮC ĐỊNH NGHĨA VÀ PHÂN LOẠI CHI TIẾT TỪNG LỚP (DOMAIN MODELING)

| Lớp (ID & Tên) | Đối tượng CHẤP NHẬN (Included Positive) | Đối tượng LOẠI TRỪ (Excluded Negative) | Ghi chú tránh nhầm lẫn |
|---|---|---|---|
| **0: battery** *(Pin / Ắc quy)* | Pin trụ (AA, AAA, 18650, C, D), pin vuông 9V, pin cúc áo (coin cell), pin lithium phồng, pin sạc dự phòng hỏng, bình ắc quy xe máy/xe đạp điện cũ bị vứt bỏ. | Đèn pin còn nguyên vẹn, đồ chơi có pin bên trong chưa tháo rời, ngăn chứa pin trên thiết bị điện tử. | Rác nguy hại cao. Nếu pin nằm rời ngoài thiết bị $\to$ gán `0: battery`. Nếu cả đồ chơi bị vứt $\to$ gán `9: trash` hoặc `7: plastic`. |
| **1: biological** *(Rác sinh học)* | Vỏ trái cây (chuối, cam, dưa hấu), cùi bắp, rau củ quả dập thối, xương động vật, thức ăn thừa vứt bỏ trên mặt đường/thùng rác, bã chè, bã cà phê, lá cây khô gom đống. | Cây cỏ tự nhiên đang sống, thức ăn tươi nguyên đặt trên bàn ăn nhà hàng, hộp xốp/túi nilon chứa thức ăn. | Phải có ngữ cảnh "bị vứt bỏ/rác". Hộp xốp chứa cơm thừa: hộp là `7: plastic`, cơm bên trong nếu nhìn rõ là `1: biological`. |
| **2: cardboard** *(Bìa carton)* | Thùng carton sóng, vỏ hộp chuyển phát nhanh, vỉ trứng bằng bìa ép, hộp bánh pizza, ống lõi giấy carton dày. | Tờ rơi mỏng, giấy in A4, ly giấy mỏng, bao thư giấy. | Bìa cứng có độ dày và kết cấu sóng/ép nhiều lớp. Nếu mỏng và dẻo $\to$ xếp vào `6: paper`. |
| **3: clothes** *(Quần áo cũ)* | Áo thun, quần jean, khăn lau, vải vụn, ga trải giường cũ bị vứt trên mặt đất hoặc trong bãi rác. | **Quần áo đang mặc trên người người đi bộ**, rèm cửa đang treo trong nhà, cờ treo trên cột. | **Tuyệt đối cấm gán nhãn quần áo đang mặc của con người** để tránh sinh lỗi dương tính giả lên người thật. |
| **4: glass** *(Thủy tinh)* | Chai bia/rượu thủy tinh, lọ nước hoa, mảnh kính vỡ, hũ mứt thủy tinh, bóng đèn sợi đốt/huỳnh quang vỡ. | Chai nhựa trong suốt (PET), cốc nhựa dẻo trong suốt, kính chắn gió xe hơi đang sử dụng. | Phân biệt kỹ với chai nhựa trong: chai thủy tinh có độ phản xạ ánh sáng mạnh, viền đáy dày và thành cứng. |
| **5: metal** *(Kim loại)* | Lon nước ngọt/bia nhôm, đồ hộp cá/thịt kim loại, nắp chai sắt tây, dây thép gai vụn, xoong chảo rỉ sét vứt bỏ, đinh ốc cũ. | Nắp cống cố định trên đường, cột đèn kim loại, tủ biến áp điện, hàng rào sắt của nhà dân. | Chỉ gán nhãn đồ kim loại đã vứt bỏ thành rác. Không gán nhãn cơ sở hạ tầng đô thị. |
| **6: paper** *(Giấy)* | Báo in cũ, hóa đơn tính tiền, giấy vở học sinh, tờ rơi quảng cáo, túi giấy mua sắm mỏng, ly giấy uống nước. | Thùng bìa carton dầy, bao bì nhựa tráng bạc bên trong, giấy vệ sinh đã qua sử dụng bẩn nặng. | Bao bì thực phẩm composite (giấy tráng màng nhựa/nhôm) nếu màng nhựa chiếm chủ đạo $\to$ xếp `7: plastic`. |
| **7: plastic** *(Nhựa)* | Chai nhựa PET, túi nilon (bịch xốp), ly nhựa dùng 1 lần, hộp cơm xốp (EPS), ống hút, muỗng nĩa nhựa, màng bọc thực phẩm, chai dầu gội/nước giặt. | Chai thủy tinh, túi vải canvas, bình kim loại. | Lớp có tần suất xuất hiện cao nhất. Toàn bộ hộp xốp đựng thức ăn (styrofoam) xếp vào `7: plastic`. |
| **8: shoes** *(Giày dép)* | Giày thể thao cũ, dép tổ ong, sandal đứt quai, ủng cao su hỏng bị vứt bỏ. | **Giày dép đang mang trên chân người đang đi lại trong khung hình.** | Tương tự quần áo: chỉ gán nhãn giày dép đã bị vứt bỏ độc lập; không dán nhãn lên chân người. |
| **9: trash** *(Rác khác)* | Tàn thuốc lá (đầu lọc), rác vô cơ tổng hợp không thể phân tách, tã lót dùng 1 lần, gốm sứ vỡ, rác quét đường hỗn hợp nhỏ vụn. | Các vật thể xác định rõ ràng thuộc 9 lớp trên. | Lớp gom rác hỗn hợp và các chất thải khó phân loại vào 9 nhóm chính. |

---

## 3. TIÊU CHUẨN KỸ THUẬT VẼ BOUNDING BOX (GEOMETRIC ACCURACY)

### 3.1 Độ khít của khung bao (Snug Bounding Box)
- Bounding box phải bao trọn các pixel ngoài cùng của vật thể ($x_{\min}, y_{\min}, x_{\max}, y_{\max}$).
- Không vẽ khung bao quá rộng chứa nhiều nền trống ($\text{IoU} \ge 0.85$ so với diện tích thực của vật thể).
- Không vẽ khung bao cắt cụt các phần nhô ra quan trọng của rác (ví dụ: cuống chuối, quai quai dép, cổ chai).

### 3.2 Quy tắc chạm viền ảnh (Image Edge Clipping)
- Khi rác nằm ở rìa ảnh và một phần vật thể bị cắt ra ngoài khung hình:
  - Tọa độ khung bao phải chạm đúng vào giới hạn ảnh ($x=0$, $y=0$, $x=W$, hoặc $y=H$).
  - **Bài học từ Phase 4E:** Ảnh `wiki_bio_banana_peel_ground.jpg` ($5184\times 3456$), dải vỏ chuối trải dài đến tận mép phải. Bounding box chuẩn xác bắt buộc phải đặt $x_2 = 5184$. Nếu để $x_2 = 5010$ sẽ vi phạm tiêu chuẩn nghiệm thu hình học.

### 3.3 Quy tắc xử lý rác bị che khuất (Occlusion & Overlap)
- **Che khuất nhẹ ($< 30\%$):** Vẽ một bounding box bao phủ trọn vẹn toàn bộ vật thể như thể nó không bị che khuất, miễn là hình dạng ước lượng có độ tin cậy cao.
- **Che khuất nặng ($\ge 30\%$):** Nếu vật thể bị cắt rời thành hai mảnh nhìn thấy riêng biệt (ví dụ cây gậy đè ngang chai nhựa):
  - Nếu hai phần kết nối rõ ràng: Vẽ một box bao trọn cả hai phần.
  - Nếu hai phần trông như hai mảnh rác vụn tách rời: Vẽ từng box riêng lẻ.

### 3.4 Quy tắc Gán nhãn Đa rác Toàn diện (Exhaustive Multi-Waste Labeling)
- **Quy tắc tuyệt đối:** Trong một ảnh có nhiều rác, **MỌI VẬT THỂ RÁC THUỘC TAXONOMY 10 LỚP ĐỀU BẮT BUỘC PHẢI ĐƯỢC GẮN NHÃN**.
- Không bao giờ chỉ gán nhãn 1 vật thể to ở giữa mà bỏ qua 3 chai nhựa nhỏ ở góc. Việc bỏ sót rác sẽ khiến thuật toán học hiểu nhầm những chai nhựa đó là "background" (False Negative Penalty), làm tê liệt khả năng học của mô hình.

---

## 4. QUY TRÌNH GÁN NHÃN ẢNH CẢNH ÂM TÍNH KHÓ (HARD-NEGATIVE POLICY)

Mục đích của Hard-Negative là triệt tiêu lỗi nhận diện nhầm ngoại cảnh công cộng thành rác.

### 4.1 Danh mục bối cảnh Hard-Negative cho phép
1. Vỉa hè gạch lát, mặt đường nhựa, lối đi công viên sạch sẽ không có rác.
2. Tủ cáp viễn thông, trạm biến áp, tủ điện kim loại gắn tường.
3. Nắp cống thoát nước kim loại hoặc bê tông.
4. Thùng rác công cộng sạch (rỗng ruột, chưa có rác).
5. Sàn nhà gạch men, thảm văn phòng sạch rác.

### 4.2 Tiêu chuẩn kiểm duyệt 100% Full-Frame
- File nhãn của ảnh Hard-Negative là **file `.txt` rỗng 0-byte**.
- Trước khi xác nhận một ảnh là Hard-Negative, kỹ sư QA phải phóng to $100\%$ độ phân giải gốc để quét toàn bộ 4 góc ảnh.
- **Quy tắc chặn lỗi:** Nếu phát hiện dù chỉ một mẩu đầu lọc thuốc lá, nắp chai nhựa nhỏ hay mẩu giấy vụn rơi trên cỏ $\to$ **DỪNG LẠI NGAY LẬP TỨC**. Có 2 lựa chọn:
  1. Gán nhãn bổ sung cho mẩu rác đó (chuyển thành ảnh thường).
  2. Loại bỏ ảnh khỏi danh sách Hard-Negative. Tuyệt đối không để một ảnh có rác mang nhãn 0-byte.

---

## 5. ĐỊNH DẠNG ĐẦU RA VÀ KIỂM THỬ TỰ ĐỘNG

Mọi file nhãn sau khi gán đều phải vượt qua bộ kiểm thử hình học tự động trong `tests/qa/test_phase4e_intake_safety.py`:
1. Mỗi dòng nhãn có đúng 5 phần tử: `<class_id> <x_c> <y_c> <w> <h>`.
2. `class_id` phải là số nguyên $0 \le \text{class\_id} \le 9$.
3. Các tọa độ chuẩn hóa phải thỏa mãn: $0.0 < x_c, y_c, w, h \le 1.0$.
4. Tọa độ góc hợp lệ: $0.0 \le x_c - w/2$ và $x_c + w/2 \le 1.0$; $0.0 \le y_c - h/2$ và $y_c + h/2 \le 1.0$.
5. Diện tích bounding box chuẩn hóa tối thiểu $\ge 0.0001$ và tối đa $\le 0.98$.
