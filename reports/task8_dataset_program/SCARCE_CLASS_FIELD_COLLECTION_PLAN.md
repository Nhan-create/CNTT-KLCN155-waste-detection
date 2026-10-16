# PHƯƠNG ÁN THU THẬP DỮ LIỆU TẠI HIỆN TRƯỜNG CHO 5 LỚP KHAN HIẾM (TASK 8H — SECTION 4)

> **Cam kết trung thực dữ liệu:** Tài liệu này là **kế hoạch triển khai thực địa khả thi** cho phần thiếu hụt còn lại sau khi các nguồn công khai đã cạn kiệt (`EXHAUSTED` / `STOPPED_AFTER_PILOT`). **Tuyệt đối không giả lập ảnh đã thu thập (`simulated_or_fabricated_images = 0`).**

## 1. Những Lớp Bắt Buộc Phải Tự Thu Thập Tại Hiện Trường & Khối Lượng Cần Bù Đắp

| Class ID | Tên Lớp | Trước 8H (`Train`) | Tăng Thêm Ở 8H (`+`) | Sau 8H (`Train`) | Sau 8H (`Val`) | Còn Thiếu Đến `M1=500` | Còn Thiếu Đến `M4=5.000` | Số Mẫu Vật Độc Lập Cần Thu (`M1`) | Số Buổi Chụp Ước Tính (`M1`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | **`battery`** | `52` | **`+34`** | **`86`** | `31` | **`414`** | `4914` | `414` mẫu | `~11` buổi |
| `3` | **`clothes`** | `34` | **`+21`** | **`55`** | `4` | **`445`** | `4945` | `445` mẫu | `~12` buổi |
| `8` | **`shoes`** | `55` | **`+23`** | **`78`** | `15` | **`422`** | `4922` | `422` mẫu | `~11` buổi |
| `1` | **`biological`** | `90` | **`+76`** | **`166`** | `40` | **`334`** | `4834` | `334` mẫu | `~9` buổi |
| `4` | **`glass`** | `90` | **`+23`** | **`113`** | `32` | **`387`** | `4887` | `387` mẫu | `~10` buổi |

## 2. Vị Trí Và Loại Bối Cảnh Chụp Thực Địa Theo Từng Lớp
- **`0: battery`:** Điểm thu hồi pin cũ tại siêu thị/chung cư/trường đại học, tiệm sửa chữa điện tử - đồng hồ - xe máy, gara ô tô (ắc quy chì 12V, pin tiểu AA/AAA/9V/cúc áo rỉ sét, pin Li-ion tháo rời), khay rác nguy hại hộ gia đình, vỉa hè/bàn kỹ thuật.
- **`3: clothes`:** Điểm tập kết rác sinh hoạt/rác cồng kềnh, thùng rác ký túc xá/khu trọ, chợ dân sinh cuối ngày (găng tay vải/cao su rách, tất/vớ cũ rách, giẻ lau dầu mỡ, áo quần rách bỏ đi trên nền bê tông/nhựa đường/đất/cỏ).
- **`8: shoes`:** Điểm tập kết rác đầu ngõ, bãi đất trống/vỉa hè, khu thu gom phế liệu ký túc xá, bờ kênh/bãi biển (giày thể thao rách đế, dép lê/tông đứt quai, ủng cao su thủng, giày da bong keo nằm lẻ chiếc hoặc theo đôi hỏng).
- **`1: biological`:** Khu vực thùng rác hữu cơ nhà ăn sinh viên / căng-tin / quán cơm / chợ truyền thống (vỏ chuối, lõi táo, vỏ cam/bưởi, bánh mì mốc, rau củ héo thối, vỏ trứng, thức ăn thừa đổ tại miệng thùng/khay thu gom/nền đất).
- **`4: glass`:** Điểm thu gom ve chai/tái chế, thùng rác cạnh quán nước/nhà hàng, lề đường/bãi cỏ (chai bia/rượu/nước ngọt thủy tinh rỗng nằm nghiêng/vỡ, lọ thủy tinh đựng gia vị/mứt bỏ đi, mảnh thủy tinh vỡ lớn).

## 3. Quy Tắc Mẫu Vật Độc Lập, Góc Chụp Và Điều Kiện Ánh Sáng
- **Tính độc lập mẫu vật (`specimen_group` & `session_group`):** Mỗi mẫu vật vật lý được gán mã `specimen_group` duy nhất tại hiện trường. Chỉ đưa **01 ảnh đại diện chính/mẫu vật** vào tập `Train` (hoặc nếu chụp 2 góc khác bối cảnh thì bắt buộc khóa cùng `specimen_group` và `session_group`, tuyệt đối không chia cắt giữa `Train` và `Val`).
- **Phân bổ góc chụp:**
  - `50%` góc nhìn đi bộ xiên (`30°–55°`, khoảng cách `0.8m–2.0m`).
  - `35%` góc nhìn từ trên xuống miệng thùng/nền đất (`70°–90°`, khoảng cách `0.5m–1.2m`).
  - `15%` góc thấp ngang tầm mắt vật thể (`10°–25°`, khoảng cách `0.5m–1.5m`).
- **Phân bổ ánh sáng:** `35%` nắng trực tiếp có bóng đổ, `35%` trời râm/bóng râm tán cây, `20%` đèn huỳnh quang/LED khu nhà rác trong nhà, `10%` chiều muộn/đèn đường.

## 4. Thiết Kế Ảnh Đa Vật Thể / Đa Lớp Cùng Lúc (Multi-Class Scenes)
- Tối thiểu **`40%` số khung hình** phải chứa từ **`2 – 5` vật thể thuộc `2 – 4` lớp khác nhau** (không vượt quá 6 vật thể rời rạc/ảnh để tránh biến thành đống rác dày không gán nhãn instance chuẩn được).
- Tổ hợp ưu tiên: `[0:battery + 5:metal + 7:plastic]`, `[3:clothes + 8:shoes + 2:cardboard]`, `[1:biological + 4:glass + 6:paper]` với mức che khuất tự nhiên `10%–35%`.

## 5. Năng Suất Ước Tính Mỗi Buổi Thu Thập
- **Quy mô 1 buổi (Session):** `2.0 – 2.5 giờ`, nhóm 2 người.
- **Đầu ra mỗi buổi:** `35 – 50` mẫu vật độc lập (`90 – 120` ảnh thô) -> thu được **`35 – 50` ảnh `REAL_WASTE_PASS` độc lập** sau lọc trùng và QA.

## 6. Hướng Dẫn Gán Nhãn Instance & Quyền Sử Dụng / An Toàn / Riêng Tư
- **Gán nhãn Instance:** Chuẩn YOLO 10 lớp (`0..9`), bao sát phần nhìn thấy của từng vật thể (`margin <= 2%`), gán nhãn **100% vật thể rác đồng xuất hiện trong ảnh** (kể cả `plastic`/`paper`/`metal` đi kèm), giữ nguyên chồng lấp thật (`VALID_SAME_CLASS_OVERLAP`, `VALID_CROSS_CLASS_OVERLAP`), và giữ `APPROVED_FOR_TRAIN = False`.
- **Quyền sử dụng ảnh:** Ảnh tự chụp phát hành dưới giấy phép `CC0-1.0` / `CC-BY-4.0` (`FIRST_PARTY_FIELD_CAPTURE`).
- **An toàn hiện trường:** Đeo găng tay chống cắt/chống hóa chất (Nitrile + Kevlar cấp 3+) và dùng kẹp gắp dài khi xử lý pin rò rỉ/ắc quy (`0: battery`), mảnh chai vỡ (`4: glass`), và rác hữu cơ mốc/phân hủy (`1: biological`). Không chọc thủng pin Li-ion phồng.
- **Bảo vệ riêng tư (PII):** Không chụp mặt người, biển số xe, hoặc nhãn địa chỉ/số điện thoại trên vỏ hộp chuyển phát; xóa tọa độ GPS chi tiết trong EXIF trước khi phát hành.
