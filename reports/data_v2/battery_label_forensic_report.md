# BÁO CÁO PHÁP Y NHÃN PIN (BATTERY LABEL FORENSIC REPORT)
**Thời điểm thực hiện:** 2026-10-12T02:00:00Z  
**Mục tiêu:** Làm rõ nguồn gốc nhãn Battery giữa `taco_0082`, `taco_0456` và `taco_0499`.  

---

## 1. KẾT QUẢ ĐỐI CHIẾU GỐC TỪ TACO COCO RAW (`taco_annotations_raw.json`)

Trong toàn bộ dataset TACO (1.500 ảnh, 4.784 annotations), **Category ID 1 (`Battery`) chỉ xuất hiện đúng 2 lần**:

### A. Trường hợp 1: `taco_0082.jpg` (TACO Image ID 82)
- **File ảnh trong COCO:** `batch_1/000104.JPG` (kích thước $3264 \times 2448$).
- **Annotation ID 207:** `category_id: 1` (`Battery`), bbox = `[1322.0, 425.0, 311.0, 215.0]`.
- **Tọa độ YOLO chuẩn hóa:**
  - $cx = (1322 + 311/2) / 3264 = 0.452665$
  - $cy = (425 + 215/2) / 2448 = 0.217525$
  - $bw = 311 / 3264 = 0.095282$
  - $bh = 215 / 2448 = 0.087827$
- **Nhãn trên đĩa (`data/detection/labels/real/taco_0082.txt`):**
  ```text
  5 0.686275 0.192198 0.172794 0.279820
  0 0.452665 0.217525 0.095282 0.087827   <-- PIN (BATTERY) THỰC SỰ
  5 0.407475 0.740400 0.365196 0.470997
  7 0.731311 0.822304 0.087010 0.110294
  7 0.803615 0.464461 0.079657 0.075163
  7 0.869332 0.677492 0.211091 0.502042
  7 0.530790 0.505515 0.114890 0.185866
  ```
- **Thuộc split:** `TEST` trong cả Dataset v2 và v3 candidate.
- **Hình ảnh thực tế & Vật thể:** Viên pin hình trụ Panasonic cỡ AA/AAA (vỏ màu vàng kim - đen, in chữ thương hiệu 'Panasonic' màu trắng nằm ngang) đặt trên mặt bàn trắng cùng vỉ thuốc nhôm, tuýp keo/kem đánh răng, chai xịt và nắp chai. **Không phải pin cúc áo (button cell).**
- **Kết luận:** `taco_0082.jpg` là **ẢNH CHỨA PIN 100% XÁC THỰC** từ TACO, nhưng thuộc tập TEST nên tuyệt đối không được gộp vào tập TRAIN.

### B. Trường hợp 2: `taco_0456.jpg` (TACO Image ID 456)
- **File ảnh trong COCO:** `batch_13/000055.jpg` (kích thước $3120 \times 4160$).
- **Annotation ID 1616:** `category_id: 1` (`Battery`), bbox = `[1402.0, 991.0, 470.0, 1636.0]`.
- **Tọa độ YOLO chuẩn hóa:**
  - $cx = (1402 + 470/2) / 3120 = 0.524679$
  - $cy = (991 + 1636/2) / 4160 = 0.434856$
  - $bw = 470 / 3120 = 0.150641$
  - $bh = 1636 / 4160 = 0.393269$
- **Nhãn trên đĩa (`data/detection/labels/real/taco_0456.txt`):**
  ```text
  0 0.524679 0.434856 0.150641 0.393269   <-- PIN (BATTERY) THỰC SỰ
  ```
- **Thuộc split:** `TRAIN` trong cả Dataset v2 và v3 candidate.
- **Kết luận:** `taco_0456.jpg` là **ẢNH CHỨA PIN 100% XÁC THỰC** từ TACO.

### C. Trường hợp ảnh `taco_0499.jpg` (TACO Image ID 499) — Thực chất là gì?
- **File ảnh trong COCO:** `batch_13/000098.jpg` (kích thước $4160 \times 3120$).
- **Annotation ID 1698:** `category_id: 45` (`Disposable food container`, `supercategory: Plastic container`).
- **Nhãn trên đĩa:** `7 0.495793 0.555449 0.538702 0.707051` (lớp `plastic`).
- **Kết luận:** `taco_0499.jpg` **KHÔNG PHẢI LÀ PIN**, mà là hộp đựng thức ăn dùng một lần bằng nhựa/xốp. Việc một bản thảo trước đây ghi 'taco_0499' là pin chỉ là **lỗi đánh máy nhầm lẫn con số (typo)** thay cho `taco_0082`.

---

## 2. TỔNG KẾT VÀ BẢO TOÀN PHÂN BỐ PIN
- **Tổng số box Pin trong toàn bộ dataset:** Đúng **2 boxes** (`taco_0456` và `taco_0082`).
- **Phân bố split:**
  - 1 box nằm ở `TRAIN` (`taco_0456.jpg`).
  - 1 box nằm ở `TEST` (`taco_0082.jpg`).
- **Ràng buộc bất biến:** Tuyệt đối không chuyển `taco_0082.jpg` từ Test sang Train để tránh làm ô nhiễm tập kiểm thử.
