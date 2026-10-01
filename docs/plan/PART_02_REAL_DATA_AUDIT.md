# BÁO CÁO KIỂM TOÁN TẬP DỮ LIỆU DETECTION VÀ THẨM ĐỊNH TRỰC QUAN 114 ẢNH THỰC TẾ (OPENIMAGES)

- **Dự án:** CNTT-KLCN155 — Phát hiện và phân loại rác thải đa đối tượng
- **Tác giả:** Tech Lead ML
- **Người nhận:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_REAL_DATA_AUDIT.md` (Cập nhật R3)
- **Ngày lập:** 02/10/2026

---

## 1. TỔNG QUAN HIỆN TRẠNG VÀ TRUY NGUYÊN NGUỒN GỐC (PROVENANCE)

Tập dữ liệu bounding box ban đầu được kế thừa từ thư mục `D:\waste-training\dataset-v1`, gồm **1.419 bức ảnh** với **4.602 bounding box**. 

Kết quả truy nguyên nguồn gốc dựa trên tệp cấu hình tải dữ liệu gốc tại `D:\waste-training` như sau:

| Nguồn dữ liệu | Số lượng ảnh | Tỷ lệ ảnh | Số lượng BBox | Tỷ lệ BBox | Các lớp hiện diện | Truy nguyên nguồn gốc chính xác |
|---|---|---|---|---|---|---|
| **Synthetic** | 1.305 | **91,97%** | **4.095** | **88,98%** | 9 lớp (0, 1, 2, 3, 4, 5, 6, 7, 9). **0 box lớp 8 (`shoes`)** | Tải từ Mendeley Data (Dataset ID: `2x69gjbcz6`), ghi nhận tại `D:\waste-training\synthetic-image-jobs.json`. Ảnh ghép nhân tạo từ vật thể rác dán trên các nền khác nhau. |
| **Real** | 114 | **8,03%** | **507** | **11,02%** | **Duy nhất lớp 8 (`shoes`)**. **0 box của 9 lớp còn lại** | Tải từ Google OpenImages V7 (tập validation qua Amazon S3), ghi nhận tại `D:\waste-training\shoe-image-jobs.json`. |
| **Ảnh thực địa TP.HCM** | **0** | **0,00%** | **0** | **0,00%** | **Chưa có** | Kế hoạch thu thập 500+ ảnh đang trong giai đoạn chuẩn bị, chưa có trong dữ liệu hiện tại. |
| **TỔNG CỘNG** | **1.419** | **100%** | **4.602** | **100%** | - | - |

> [!NOTE]
> **Giải trình về số lượng box:** Số lượng bounding box chính xác trong tập synthetic là **4.095 box** (không phải 4.417 như một số tính toán nhầm lẫn trước đây). Tổng số box của toàn bộ 1.419 ảnh là $4.095 + 507 = 4.602$ box, đã được kiểm toán 100% không có lỗi hình học hay cú pháp.

---

## 2. BẰNG CHỨNG MAPPING VÀ CHUYỂN ĐỔI TỌA ĐỘ TỪ OPENIMAGES

### 2.1. Đối chiếu Ontology và Class ID nguồn
Dựa trên tệp ontology gốc `D:\waste-training\openimages-boxable-classes.csv`:
- `/m/09j5n` $\rightarrow$ **Footwear** (Được chuyển thành lớp 8: `shoes`)
- `/m/01b638` $\rightarrow$ **Boot** (Được chuyển thành lớp 8: `shoes`)
- `/m/03nfch` $\rightarrow$ **Sandal** (Được chuyển thành lớp 8: `shoes`)

### 2.2. Công thức chuyển đổi hình học (OpenImages $\rightarrow$ YOLO)
OpenImages lưu tọa độ góc chuẩn hóa: $[X_{min}, X_{max}, Y_{min}, Y_{max}] \in [0, 1]$.  
Định dạng YOLO chuẩn yêu cầu: $[class\_id, x_c, y_c, w, h]$, trong đó:
$$x_c = \frac{X_{min} + X_{max}}{2}, \quad y_c = \frac{Y_{min} + Y_{max}}{2}, \quad w = X_{max} - X_{min}, \quad h = Y_{max} - Y_{min}$$

**Bằng chứng đối soát mẫu trên ảnh `oi_034f71ee4e111261.jpg` (OpenImages ID: `034f71ee4e111261`):**
- Trong `openimages-validation-boxes.csv`:
  - Box 1: $X_{min} = 0.5203125, X_{max} = 0.6265625, Y_{min} = 0.80625, Y_{max} = 0.9604167$
    $\rightarrow x_c = 0.57343750, y_c = 0.88333335, w = 0.10625000, h = 0.15416670$
  - Box 2: $X_{min} = 0.7953125, X_{max} = 0.878125, Y_{min} = 0.95, Y_{max} = 0.99791664$
    $\rightarrow x_c = 0.83671875, y_c = 0.97395832, w = 0.08281250, h = 0.04791664$
- Khớp chính xác 100% từng chữ số với file nhãn baseline: `D:\waste-training\dataset-v1\labels\train\oi_034f71ee4e111261.txt`.

---

## 3. KẾT QUẢ THẨM ĐỊNH TRỰC QUAN TOÀN BỘ 114 ẢNH THỰC TẾ

Tech Lead ML đã trực tiếp kiểm tra bằng mắt toàn bộ **114/114 bức ảnh** kèm bounding box thông qua hệ thống **10 contact sheets** lưu tại `artifacts/part02/real_data_audit/contact_sheets/` (`contact_sheet_01.jpg` đến `contact_sheet_10.jpg`).

Bảng thẩm định chi tiết 114 dòng được lưu tại [`real_images_audit_table.csv`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/real_data_audit/real_images_audit_table.csv).

### 3.1. Phân loại thực tế sau thẩm định trực quan

| Nhóm thẩm định trực quan | Số lượng ảnh | Tỷ lệ | Mô tả ngữ cảnh thực tế | Phù hợp bài toán rác thải? |
|---|:---:|:---:|---|:---:|
| **`UNSUITABLE_WORN_BY_PERSON`** | **103** | **90,35%** | Giày đang được con người hoặc mannequin/tượng mang trên chân trong sinh hoạt: chạy bộ, thi đấu thể thao (bóng đá, điền kinh, đấu vật, trượt băng), trình diễn thời trang, đi lại ngoài đường, học sinh trong lớp. | **KHÔNG PHÙ HỢP** (Vật dụng đang sử dụng, không phải rác) |
| **`UNSUITABLE_COMMERCIAL_PRODUCT`** | **10** | **8,77%** | Ảnh chụp sản phẩm studio/catalog quảng cáo thương mại: giày lười da sang trọng, bốt da cao bồi, giày Converse, giày cao gót, giày thể thao trên phông nền trắng/studio sạch sẽ. | **KHÔNG PHÙ HỢP** (Sản phẩm thương mại nguyên vẹn, không phải rác) |
| **`SUITABLE_DISCARDED_OUTDOORS`** | **1** | **0,88%** | Ảnh `oi_6e9fabfb47047286.jpg`: Duy nhất một chiếc giày cũ màu đen bị vứt bỏ nằm đơn độc ngoài trời trên thảm lá cây mùa thu rụng. | **CÓ THỂ PHÙ HỢP** (Rác thải ngoài môi trường tự nhiên) |
| **TỔNG CỘNG** | **114** | **100%** | 100% ảnh đã được thẩm định trực quan độc lập. | **Chỉ 1 ảnh rác thực tế** |

> [!CAUTION]
> **Phát hiện lỗi bỏ sót nhãn nghiêm trọng (Critical Annotation Leakage):**
> Trong 103 bức ảnh có người mang giày, OpenImages vốn ghi nhận **239 bounding box quần áo (`/m/09j2d: Clothing`)** và **187 bounding box con người (`/m/01g317: Person`)**. Quá trình trích xuất dữ liệu cũ chỉ lấy nhãn giày và **bỏ toàn bộ nhãn quần áo**.  
> **Hậu quả:** Khi đưa vào huấn luyện mô hình phát hiện đa rác, các vùng quần áo trên người sẽ bị gán nhãn là vùng nền (background/negative), triệt tiêu hoàn toàn khả năng nhận dạng lớp 3 (`clothes`) của mô hình.

---

## 4. CÂU HỎI VỀ PHẠM VI ỨNG DỤNG VÀ VÙNG NHÌN CAMERA

Để xác lập ranh giới dữ liệu chuẩn xác cho Phần 3, Tech Lead kính đề nghị PM Ngô Thanh Nhân phê duyệt một trong hai định hướng:

### Phương án 1 (Khuyến nghị của Tech Lead): Hệ thống phân loại tại nguồn / Thùng rác thông minh / Băng chuyền thu gom
- **Vùng nhìn camera:** Khoang kín bên trong thùng rác hoặc băng chuyền trạm trung chuyển.
- **Quy tắc phân loại:** Mọi vật thể xuất hiện trong khoang camera đều mặc định là đồ vật đã bị thải bỏ. Không có người hoặc chân người lọt vào khoang này.
- **Xử lý 114 ảnh OpenImages:** **Cách ly toàn bộ 103 ảnh người mang giày và 10 ảnh studio** khỏi tập dữ liệu huấn luyện và kiểm thử (coi là out-of-domain) để tránh mô hình học sai ngữ cảnh.

### Phương án 2: Hệ thống camera giám sát nơi công cộng / Đường phố
- **Vùng nhìn camera:** Góc rộng quan sát vỉa hè, công viên, lòng đường.
- **Quy tắc phân loại:** Mô hình phải phân biệt được "giày đang mang trên chân người" (không báo động) với "giày bị vứt bỏ dưới đất" (báo động rác).
- **Yêu cầu kỹ thuật:** Bắt buộc phải bổ sung nhãn `Person`, `Clothing` và nhãn thuộc tính ngữ cảnh `is_discarded`. Quy mô và độ phức tạp của bài toán sẽ tăng gấp nhiều lần.

---

## 5. KẾT LUẬN VÀ KIẾN NGHỊ

1. **Bảo toàn dữ liệu nguồn:** Giữ nguyên 114 ảnh OpenImages tại thư mục nguồn nhưng gắn cờ trạng thái cách ly, không đưa vào tập Train/Val/Test chính thức của Phần 3.
2. **Không huấn luyện detector trên dữ liệu hiện tại:** Việc chỉ có 1 lớp `shoes` trong ảnh thật và 9 lớp còn lại 100% synthetic sẽ tạo ra mô hình lệch lạc nghiêm trọng.
3. **Thực thi đợt thu thập rác thực địa tại TP.HCM:** Cần tối thiểu 500+ ảnh rác thực tế đa lớp theo kế hoạch tại `PART_02_COLLECTION_AND_LABELING_PLAN.md` trước khi bước vào Phần 3.
