# BÁO CÁO KIỂM TOÁN TẬP DỮ LIỆU DETECTION VÀ PHÂN TÍCH 114 ẢNH THỰC TẾ (OPENIMAGES)

- **Dự án:** CNTT-KLCN155 — Phát hiện và phân loại rác thải đa đối tượng
- **Tác giả:** Tech Lead ML / Nhóm Kỹ sư ML
- **Người nhận:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_REAL_DATA_AUDIT.md`
- **Ngày lập:** 02/10/2026

---

## 1. TỔNG QUAN HIỆN TRẠNG DỮ LIỆU PHÁT HIỆN ĐA ĐỐI TƯỢNG (DETECTION)

Toàn bộ dữ liệu bounding box ban đầu được kế thừa từ thư mục `D:\waste-training\dataset-v1`, gồm **1.419 bức ảnh** với **4.602 bounding box**. 

Sau khi rà soát và bóc tách nguồn gốc chi tiết, cơ cấu dữ liệu thực tế như sau:

| Nguồn dữ liệu | Số lượng ảnh | Tỷ lệ ảnh | Số lượng BBox | Tỷ lệ BBox | Các lớp hiện diện | Ghi chú nguồn gốc |
|---|---|---|---|---|---|---|
| **Synthetic (Mendeley)** | 1.305 | **91,97%** | 4.095 | **88,98%** | 9 lớp (0, 1, 2, 3, 4, 5, 6, 7, 9). **HOÀN TOÀN KHÔNG CÓ LỚP 8 (`shoes`)** | Ảnh ghép nhân tạo từ Mendeley Waste, các vật thể rác được dán ghép cơ học lên nền trắng hoặc bối cảnh nhân tạo. |
| **Real (OpenImages V7)** | 114 | **8,03%** | 507 | **11,02%** | **DUY NHẤT LỚP 8 (`shoes`)**. **HOÀN TOÀN KHÔNG CÓ 9 LỚP CÒN LẠI** | Ảnh tải về từ tập validation của Google OpenImages V7 dựa trên job tải `shoe-image-jobs.json`. |
| **Ảnh thực địa TP.HCM** | **0** | **0,00%** | **0** | **0,00%** | **Không có** | Chưa thu thập hoặc chưa được đưa vào pipeline. |
| **TỔNG CỘNG** | **1.419** | **100%** | **4.602** | **100%** | - | - |

---

## 2. KẾT QUẢ KIỂM TOÁN TẬN GỐC 114 ẢNH THỰC TẾ (OPENIMAGES)

Đội ngũ kỹ thuật đã đối chiếu 114 mã định danh ảnh trong `data/detection/images/real/` với toàn bộ cơ sở dữ liệu nhãn gốc của OpenImages V7 (`D:\waste-training\openimages-validation-boxes.csv` và `openimages-boxable-classes.csv`).

### 2.1. Thống kê tất cả các đối tượng thực sự có mặt trong 114 bức ảnh
Bảng thống kê các nhãn gốc của OpenImages trên 114 bức ảnh này cho thấy:

| Nhãn thực tế trong OpenImages V7 | Mã Ontology | Số lượng phát hiện | Ý nghĩa ngữ cảnh |
|---|---|---|---|
| **Footwear (Giày dép)** | `/m/09j5n` | 496 | Được chuyển đổi thành lớp 8 (`shoes`) trong dự án |
| **Mammal (Động vật có vú)** | `/m/04rky` | 266 | Người và động vật trong ảnh |
| **Clothing (Quần áo, trang phục)** | `/m/01g317` | **239** | **BỊ BỎ SÓT HOÀN TOÀN (Lớp 3 của dự án)** |
| **Person (Con người)** | `/m/01g317` | **187** | Người xuất hiện trực tiếp trong khung hình |
| **Human body / Human leg** | `/m/02p0tk3` | 334 | Chân người, thân người đang vận động |
| **Man / Woman / Girl** | - | 261 | Nam giới, phụ nữ, trẻ em |
| **Sports equipment / Roller skates** | - | 198 | Dụng cụ thể thao, giày trượt patin |

### 2.2. Kết luận thẩm định tính phù hợp (Suitability Assessment)
Phân tích chi tiết từng ảnh (lưu tại `artifacts/part02/real_data_audit/real_images_audit_table.csv`) cho kết quả:
- **101 / 114 ảnh (88,6%): `UNSUITABLE_SHOES_BEING_WORN`**
  - **Hiện trạng:** Là ảnh chụp người đi đường, người mẫu thời trang, vận động viên, người đang khiêu vũ hoặc đi lại ngoài phố. Đôi giày đang được con người **MANG TRÊN CHÂN VÀ ĐANG SỬ DỤNG**, hoàn toàn **KHÔNG PHẢI RÁC THẢI BỊ VỨT BỎ**.
  - **Lỗi bỏ sót nhãn nghiêm trọng:** Trong 101 bức ảnh này có ít nhất **239 vật thể quần áo (`clothing`)** đang được mặc trên người. Tuy nhiên, người tạo `dataset-v1` trước đây đã cắt bỏ toàn bộ nhãn quần áo, chỉ giữ lại nhãn giày.
  - **Hệ quả mô hình:** Khi huấn luyện detector, mô hình bị ép học rằng: Quần áo trên người là *Background (Vùng nền âm tính)*, trong khi giày là *Object (Vật thể cần phát hiện)*. Điều này sẽ phá hủy hoàn toàn khả năng học lớp 3 (`clothes`) của detector trong Part 3.
- **13 / 114 ảnh (11,4%): `REQUIRES_MANUAL_INSPECTION`**
  - **Hiện trạng:** Là ảnh chụp sản phẩm giày thời trang trong cửa hàng, ảnh phụ kiện, ảnh chụp thú cưng (chó đi giày), hoặc ảnh giày cạnh xe cộ.
  - **Kết luận:** Hoàn toàn không có bức ảnh nào phản ánh bối cảnh rác thải đường phố, bãi rác sinh hoạt hay thùng rác công cộng.
- **0 / 114 ảnh (0,0%): Đạt tiêu chuẩn rác thải sinh hoạt thực tế.**

---

## 3. CÂU HỎI THIẾT KẾ CỐT LÕI CẦN PM XÁC NHẬN

Từ kết quả kiểm toán trên, Tech Lead gửi câu hỏi làm rõ quan trọng nhất tới PM Ngô Thanh Nhân:

> **CÂU HỎI CHO PM:**  
> Hệ thống phát hiện rác đa đối tượng CNTT-KLCN155 có phạm vi nhận diện rác thải được định nghĩa là:  
> - **Phương án A (Nhận diện theo chủng loại vật thể/vật liệu):** Nhận diện 10 loại vật liệu bất kể vật thể đó đang được con người sử dụng hay đã bị thải bỏ (ví dụ: phát hiện đôi giày trên chân người là `shoes`, chiếc áo đang mặc là `clothes`).  
> - **Phương án B (Nhận diện rác thải đã bị vứt bỏ - Discarded Waste Context):** Chỉ phát hiện các vật thể rác thải nằm trong bối cảnh vứt bỏ (trong thùng rác, trên mặt đường, vỉa hè, bãi tập kết, đã qua sử dụng/hư hỏng).  

### Phân tích tác động kỹ thuật:
- Nếu chọn **Phương án B (Khuyến nghị của Tech Lead)**: Toàn bộ 114 ảnh OpenImages hiện tại là **NGOÀI PHẠM VI (Out of Scope)** vì chúng là người đang mang giày sinh hoạt thường ngày. Bắt buộc phải loại bỏ khỏi tập dữ liệu huấn luyện rác thải và thay thế bằng ảnh giày cũ vứt bỏ trong thùng rác.
- Nếu chọn **Phương án A**: Bắt buộc phải **GÁN NHÃN LẠI (Relabel)** toàn bộ 239 vị trí quần áo, áo khoác, quần dài trong 114 ảnh OpenImages để tránh detector coi quần áo là background.

---

## 4. KẾ HOẠCH HÀNH ĐỘNG KHẮC PHỤC DỮ LIỆU

1. **Tuyệt đối không dùng 114 ảnh OpenImages hiện tại cho tập Test / Validation của detector:** Dữ liệu này không đại diện cho bài toán rác thải thực tế tại Việt Nam.
2. **Cô lập 1.305 ảnh Synthetic Mendeley:** Chỉ sử dụng cho mục đích **Pretraining / Data Augmentation khởi tạo**, không đưa vào tập đánh giá Benchmark cuối cùng.
3. **Triển khai đợt thu thập dữ liệu thực tế tại TP.HCM:** Cần kế hoạch thu thập có cấu trúc (trình bày chi tiết trong `PART_02_COLLECTION_AND_LABELING_PLAN.md`).
