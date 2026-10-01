# 02 — DATA INVENTORY AND TAXONOMY: KIỂM KÊ DỮ LIỆU VÀ QUY CHUẨN DANH MỤC LỚP (R2)

**Dự án:** Phân loại và phát hiện rác thải sinh hoạt (`CNTT-KLCN155`)  
**Tác giả:** Tech Lead & Machine Learning Engineer  
**Phiên:** Task R2 — Reconcile, Audit & Rectify  
**Trạng thái:** `VERIFIED_AND_LOCKED` (Đã kiểm toán thực tế và đối soát từng dòng)

---

## 1. Mục tiêu và vấn đề cần giải quyết

### 1.1. Mục tiêu
Thiết lập hồ sơ kiểm kê dữ liệu định lượng chính xác 100% dựa trên các tệp thực tế có trên máy tính; phân định rõ ranh giới giữa bản gốc và các bản sao resize; kiểm toán tính trùng lặp (exact & near duplicate); xây dựng ma trận ánh xạ danh mục lớp (Taxonomy Mapping) khoa học, thống nhất **10 lớp là danh mục chuẩn xuyên suốt cả Giai đoạn A và Giai đoạn B**, đồng thời làm rõ bảng ánh xạ sang **6 nhóm vật liệu của đề cương cũ** như một phương án đối chiếu phụ.

### 1.2. Vấn đề cần giải quyết
1. **Ảo tưởng về số lượng ảnh (Dataset Inflation):** Bộ Garbage V2 thường bị tính nhầm lên 48.868 tệp do chứa các thư mục con resize (`standardized_256`, `standardized_384`). Đã loại trừ dứt điểm 24.518 bản sao resize.
2. **Loại bỏ sai sót tính toán trong R1:** Sửa toàn diện các số liệu:
   - Tổng sạch: **14.831 ảnh** (Train: **10.381**, Val: **2.225**, Test: **2.225**, tổng bằng 14.831, không bị lệch thành 14.832).
   - Tổng thô chính xác: **15.754 ảnh** (không phải 15.448).
   - Khi trừ 3 lớp `clothes` (1.892) + `shoes` (1.449) + `trash` (503) = 3.844 ảnh, số lượng còn lại chính xác là $14.831 - 3.844 = \mathbf{10.987\text{ ảnh}}$ (sửa lỗi 13.760 ở R1).
3. **Thống nhất taxonomy 10 lớp:** Không tự ý hạ Giai đoạn B xuống 6 lớp. Giữ vững 10 lớp rác sinh hoạt thực tế làm mục tiêu chính.

---

## 2. Hiện trạng đã kiểm kê trên ổ đĩa (`VERIFIED`)

### 2.1. Nguồn dữ liệu đã tải về máy
Toàn bộ dữ liệu thô được lưu trữ tại thư mục [D:\HK7\Đồ án khóa luận\Data\raw](file:///D:/HK7/Đồ%20án%20khóa%20luận/Data/raw):
1. **Garbage Classification V2:**
   - Nguồn Kaggle: `sumn2u/garbage-classification-v2` (Version 12, MIT License).
   - Thư mục trên máy: `D:\HK7\Đồ án khóa luận\Data\raw\garbage_v2`.
   - Tổng số ảnh gốc độc lập: **12.259 ảnh**.
   - Các thư mục biến thể resize tồn tại song song: `standardized_256` (12.259 ảnh), `standardized_384` (12.259 ảnh) $\rightarrow$ **ĐÃ LOẠI TRỪ 24.518 ẢNH BẢN SAO**.
2. **VN Trash Classification:**
   - Nguồn Kaggle: `mrgetshjtdone/vn-trash-classification` (Version 1, MIT License).
   - Thư mục trên máy: `D:\HK7\Đồ án khóa luận\Data\raw\vn_trash`.
   - Tổng số ảnh: **3.495 ảnh**.
3. **Tổng cộng thô:** $12.259 + 3.495 = \mathbf{15.754\text{ ảnh thô}}$.

### 2.2. Kiểm kê phân bố lớp thô, trùng lặp và ảnh sạch từng lớp (`VERIFIED`)

Bảng dưới đây trích xuất từ [data/audit/class_counts.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/class_counts.csv) được quét trực tiếp trên 15.754 file thô:

| STT | Tên lớp chuẩn (10 lớp) | Thô Garbage V2 | Thô VN Trash | Tổng ảnh thô | Trùng lặp loại bỏ | Tổng sạch (Clean) | Tập Train (70%) | Tập Val (15%) | Tập Test (15%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `battery` | 756 | 0 | 756 | 0 | **756** | 529 | 113 | 114 |
| 2 | `biological` | 699 | 0 | 699 | 0 | **699** | 489 | 105 | 105 |
| 3 | `cardboard` | 1.411 | 1.495 | 2.906 | 838 | **2.068** | 1.448 | 311 | 309 |
| 4 | `clothes` | 1.892 | 0 | 1.892 | 0 | **1.892** | 1.324 | 284 | 284 |
| 5 | `glass` | 1.736 | 0 | 1.736 | 0 | **1.736** | 1.215 | 260 | 261 |
| 6 | `metal` | 930 | 1.345 | 2.275 | 17 | **2.258** | 1.581 | 339 | 338 |
| 7 | `paper` | 1.336 | 211 | 1.547 | 63 | **1.484** | 1.039 | 222 | 223 |
| 8 | `plastic` | 1.597 | 394 | 1.991 | 5 | **1.986** | 1.390 | 298 | 298 |
| 9 | `shoes` | 1.449 | 0 | 1.449 | 0 | **1.449** | 1.014 | 217 | 218 |
| 10 | `trash` | 453 | 50 | 503 | 0 | **503** | 352 | 76 | 75 |
| | **TỔNG CỘNG** | **12.259** | **3.495** | **15.754** | **923** | **14.831** | **10.381** | **2.225** | **2.225** |

- **Phép cộng kiểm toán:**  
  $$10.381 + 2.225 + 2.225 = \mathbf{14.831} \text{ (Khớp 100\%)}$$
- **Kiểm toán loại 3 lớp `clothes`, `shoes`, `trash`:**  
  Số ảnh 3 lớp: $1.892 + 1.449 + 503 = 3.844\text{ ảnh}$.  
  Số ảnh còn lại: $14.831 - 3.844 = \mathbf{10.987\text{ ảnh}}$ (sửa dứt điểm con số 13.760 ở R1).

### 2.3. Chi tiết 923 tệp trùng lặp (Deduplication Audit)
- **897 tệp trùng tuyệt đối (MD5 exact match)**: Cùng nội dung nhị phân.
- **26 tệp gần trùng (pHash Hamming distance $\le 4$)**: Cùng một góc chụp hoặc thay đổi độ sáng nhẹ.
- **Vị trí trùng lặp:** Toàn bộ 923 tệp trùng lặp nằm bên trong bộ dữ liệu `vn_trash` (838 Carton, 63 Paper, 17 Alu, 4 Foam_box, 1 PET).
- **Trùng lặp giữa hai nguồn:** **0 ảnh** (Garbage V2 và VN Trash hoàn toàn độc lập về nguồn ảnh).
- Chi tiết từng nhóm trùng lặp được lưu tại [data/audit/duplicate_groups.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/duplicate_groups.csv) và danh sách tệp bị loại tại [data/audit/excluded_samples.csv](file:///D:/CNTT-KLCN155-waste-detection/data/audit/excluded_samples.csv).

---

## 3. Bảng quy chuẩn Taxonomy và Ánh xạ nhãn

### 3.1. Bảng ánh xạ nguồn sang 10 lớp chuẩn (Mặc định cho cả Giai đoạn A và B)

| Nhãn nguồn | Dataset nguồn | Lớp dự án (10 lớp) | Tiêu chí nhận diện | Xử lý trường hợp ranh giới |
|:---|:---|:---|:---|:---|
| `battery` | Garbage V2 | `battery` | Pin tiểu, ắc quy, pin cúc áo | Giữ nguyên, không gộp |
| `biological` | Garbage V2 | `biological` | Rác thực phẩm, vỏ trái cây, lá cây | Nếu rác đựng trong bọc nilon, gắn nhãn chính là biological nếu thức ăn chiếm $>70\%$ |
| `cardboard` | Garbage V2 | `cardboard` | Thùng bìa carton gợn sóng, hộp bánh carton cứng | Phân biệt với paper bằng độ dày và cấu trúc sóng |
| `clothes` | Garbage V2 | `clothes` | Vải vụn, quần áo cũ, khăn lau | Không gộp với giày vải |
| `glass` | Garbage V2 | `glass` | Chai, lọ, ly thủy tinh | Chai nhựa trong suốt đặt cạnh không bị gán nhầm sang glass |
| `metal` | Garbage V2 | `metal` | Lon thiếc, nắp chai kim loại, chìa khóa, ốc vít | Đồ kim loại sơn màu vẫn tính là metal |
| `paper` | Garbage V2 | `paper` | Sách báo, tài liệu A4 mỏng, hóa đơn | Giấy vụn mỏng không có lớp phủ nilon dày |
| `plastic` | Garbage V2 | `plastic` | Chai nhựa, can nhựa, chai mỹ phẩm | Các loại nhựa PET, HDPE, PP |
| `shoes` | Garbage V2 | `shoes` | Giày thể thao, dép quai hậu, dép xốp | Giữ riêng vì hình thái đặc thù |
| `trash` | Garbage V2 | `trash` | Rác vô cơ hỗn hợp không tái chế được (tàn thuốc, băng gạc...) | Lớp rác tạp, không ép nhãn sang lớp khác |
| `Alu` | VN Trash | `metal` | Lon nhôm nước ngọt bia | Thuộc tập con kim loại |
| `Carton` | VN Trash | `cardboard` | Hộp bìa carton chuyển phát nhanh | Thuộc tập con cardboard |
| `Foam_box` | VN Trash | `plastic` | Hộp xốp đựng cơm (nhựa Polystyrene - PS) | Bản chất hóa học là nhựa, ánh xạ vào plastic |
| `Milk_box` | VN Trash | `cardboard` | Hộp sữa giấy (Tetra Pak / Aseptic carton) | Cấu trúc đa lớp (75% bìa giấy), ánh xạ vào cardboard |
| `Other` | VN Trash | `trash` | Rác thải sinh hoạt vụn vặt | Ánh xạ vào trash |
| `PET` | VN Trash | `plastic` | Chai nước suối, nước ngọt trong suốt | Thuộc tập con plastic |
| `Paper` | VN Trash | `paper` | Giấy viết, tập vở học sinh | Thuộc tập con paper |
| `Paper_cup` | VN Trash | `paper` | Ly giấy dùng một lần | Ánh xạ vào paper |
| `Plastic_cup` | VN Trash | `plastic` | Ly nhựa trà sữa, ly nước mía | Ánh xạ vào plastic |

### 3.2. Bảng ánh xạ đối chiếu sang 6 nhóm vật liệu đề cương cũ (Secondary / Fallback View)

Nếu Hội đồng bảo vệ khóa luận yêu cầu đối chiếu với đề cương Word gốc (6 nhóm vật liệu: `plastic`, `paper`, `metal`, `glass`, `organic`, `hazardous`), hệ thống có sẵn bảng ánh xạ quy đổi tự động:

| Nhãn 10 lớp chuẩn | Nhãn 6 lớp đề cương cũ | Ghi chú quy đổi |
|:---|:---|:---|
| `plastic` | `plastic` | Giữ nguyên |
| `cardboard` | `paper` | Gộp chung bìa carton vào nhóm Giấy / Bìa |
| `paper` | `paper` | Giữ nguyên |
| `metal` | `metal` | Giữ nguyên |
| `glass` | `glass` | Giữ nguyên |
| `biological` | `organic` | Ánh xạ sang nhóm Hữu cơ |
| `battery` | `hazardous` | Ánh xạ sang nhóm Rác nguy hại |
| `clothes` | *Loại bỏ (Exclude)* | Đề cương cũ loại bỏ quần áo cũ |
| `shoes` | *Loại bỏ (Exclude)* | Đề cương cũ loại bỏ giày dép |
| `trash` | *Loại bỏ (Exclude)* | Đề cương cũ loại bỏ rác hỗn hợp không rõ nguồn |

---

## 4. Phân tích hiện trạng tập Detection v1 (1.419 ảnh) (`VERIFIED`)

Kiểm toán dữ liệu bounding box tại [data/audit/audit_summary.json](file:///D:/CNTT-KLCN155-waste-detection/data/audit/audit_summary.json):
- **Tổng số ảnh:** 1.419 ảnh.
- **Tổng số bounding box:** 4.602 hộp.
- **Nguồn gốc ảnh:**
  - **1.305 ảnh (91,97%)** là ảnh tổng hợp từ **Mendeley Synthetic** (ảnh đồ họa ghép rác nhân tạo trên mặt sàn phẳng).
  - **114 ảnh (8,03%)** là ảnh chụp thực tế từ **OpenImages**.
- **Đánh giá rủi ro khoa học:**  
  Nếu dùng tập v1 này làm thước đo đánh giá độ chính xác (mAP) của hệ thống thì kết quả sẽ bị sai lệch nghiêm trọng vì mô hình chỉ học cách phát hiện rác nhân tạo trên nền studio ghép.  
- **Biện pháp xử lý:**  
  1. Tập v1 này chỉ được dùng làm dữ liệu Pre-training bước đệm cho Giai đoạn B.
  2. Bắt buộc thu thập và gán nhãn tập **200–300 ảnh thực tế TP.HCM** (tại trường HUIT, căn tin, thùng rác công cộng) để làm tập Test Set thực tế độc lập (Blind Test).

---

## 5. Kết luận nghiệm thu kiểm kê

- Mọi số liệu trong R2 đã được đối soát khớp từng dòng trên ổ đĩa và lưu trữ trong [data/audit/](file:///D:/CNTT-KLCN155-waste-detection/data/audit).
- Không còn mâu thuẫn số học hay nhầm lẫn phân loại nào tồn tại trong tài liệu kế hoạch.
