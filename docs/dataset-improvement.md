# CHIẾN LƯỢC VÀ KẾ HOẠCH NÂNG CẤP TẬP DỮ LIỆU HUẤN LUYỆN (DATASET IMPROVEMENT)

**Dự án:** Hệ thống Phát hiện & Phân loại Rác thải Đa Vật thể (CNTT-KLCN155)  
**Tác giả:** Principal Computer Vision Engineer / Agent A (Data & Mapping Lead)  
**Ngày lập:** 11/10/2026  

---

## 1. Kiểm Kê Hiện Trạng Tập Dữ Liệu Hiện Tại (`v2.0-real-only`)

Tập dữ liệu đang được sử dụng để huấn luyện `E1_yolov8n_10cls` và `E2_ssdlite320_10cls` được quy định tại `configs/detection_dataset_real_v2.yaml`:

### 1.1. Thống kê theo phân đoạn (Splits)
- **Tổng số ảnh:** 1.996 ảnh (Train: 1.398, Val: 299, Test: 299).
- **Tổng số bounding boxes:** 5.585 boxes.
- **Nguồn dữ liệu:** TACO (official annotations) + OpenImages V7 (bổ sung có chọn lọc).
- **Trạng thái Mendeley:** Cách ly hoàn toàn (`QUARANTINED`, 0 ảnh) do nhiễu nội lớp cực kỳ nghiêm trọng (Class 6 lon nhôm bị gán thành Pin; Class 8 cành cây/vỏ hoa quả bị gán thành Quần áo; Class 4 pizza box/túi giấy bị gán thành Thủy tinh).

### 1.2. Phân bố số lượng Bounding Box theo từng lớp trong tập Train (1.398 ảnh)

| ID | Tên lớp (Taxonomy) | Tên tiếng Việt | Số lượng Box | Tỉ lệ % | Đánh giá tình trạng |
|---:|---|---|---:|---:|---|
| **0** | `battery` | Pin / Ắc quy | **1** | 0.03% | **Cực kỳ thiếu hụt (Không thể học được)** |
| **1** | `biological` | Rác hữu cơ / Thức ăn | **3** | 0.08% | **Cực kỳ thiếu hụt (Không thể học được)** |
| **2** | `cardboard` | Bìa carton | 169 | 4.26% | Thiếu hụt |
| **3** | `clothes` | Quần áo / Vải | 368 | 9.29% | Tạm ổn |
| **4** | `glass` | Thủy tinh | 213 | 5.37% | Thiếu hụt |
| **5** | `metal` | Kim loại / Lon nhôm | 420 | 10.60% | Trung bình |
| **6** | `paper` | Giấy / Báo | 160 | 4.04% | Thiếu hụt |
| **7** | `plastic` | Nhựa / Chai túi | **1.560** | **39.36%** | **Áp đảo (Mất cân bằng dữ liệu)** |
| **8** | `shoes` | Giày dép | 215 | 5.43% | Thiếu hụt |
| **9** | `trash` | Rác vô cơ khác | **854** | **21.55%** | Hỗn tạp cao |
| | **TỔNG CỘNG** | | **3.963** | **100%** | |

### 1.3. Khuyết tật chí mạng: 0 Ảnh Nền (Zero Negative Background Images)
- Toàn bộ 1.398 ảnh train đều có vật thể rác. **Số lượng ảnh rỗng (empty annotations) = 0**.
- Đây là nguyên nhân trực tiếp khiến mô hình không học được vector đặc trưng của nền, dẫn đến hiện tượng hallucination (dương tính giả) trên tường gạch, biển hiệu và tủ điện đô thị.

---

## 2. Kế Hoạch Bổ Sung Tập Hard-Negative (Mẫu Âm Ngoại Cảnh)

Để triệt tiêu hoàn toàn lỗi nhận nhầm công trình đô thị thành rác, cần xây dựng tập **Hard-Negative** với các quy chuẩn kỹ thuật nghiêm ngặt:

### 2.1. Danh mục các chủ đề ảnh Hard-Negative bắt buộc
1. **Kiến trúc & Tường:** Mặt tường gạch thô, tường sơn vàng/trắng, mảng bê tông, cửa sổ kính, ban công.
2. **Biển hiệu đô thị:** Biển quảng cáo kim loại/hộp đèn, biển báo giao thông, bảng hiệu cửa hàng (loại trừ nguyên nhân nhầm thành `cardboard`).
3. **Thiết bị hạ tầng kỹ thuật:** Tủ điện xanh/xám trên vỉa hè, cột đèn, nắp cống kim loại, đồng hồ nước (loại trừ nguyên nhân nhầm thành `cardboard`/`metal`).
4. **Nền đô thị sạch:** Vỉa hè lát gạch sạch, lòng đường nhựa sạch, lối đi công viên không rác.
5. **Đồ đạc thông thường:** Xe máy, xe đạp, ô tô đỗ trên đường, ghế đá, chậu cây cảnh.

### 2.2. Quy chuẩn kỹ thuật của tệp annotation
- Định dạng chuẩn YOLO: Tệp `.txt` tương ứng với ảnh phải tồn tại nhưng **hoàn toàn rỗng (0 bytes)**.
- Đối với PyTorch/Torchvision: Trả về tensor `boxes = torch.zeros((0, 4))`, `labels = torch.zeros((0,), dtype=torch.int64)`.
- **Ràng buộc cấm:** Tuyệt đối không gán nhãn nền ngoại cảnh thành class 9 (`trash`). Nếu gán thành `trash`, mô hình sẽ học rằng "tòa nhà là rác thải" và gây hậu quả phá hủy toàn bộ hệ thống.
- **Tỉ lệ đề xuất:** Bổ sung khoảng $250 - 300$ ảnh Hard-Negative (chiếm khoảng $15\% - 20\%$ tổng số ảnh tập train).

---

## 3. Chiến Lược Cân Bằng Dữ Liệu Các Lớp Thiếu Hụt

### 3.1. Giải pháp cho lớp `battery` (Pin) và `biological` (Rác hữu cơ)
- Hiện tại: `battery` (1 box) và `biological` (3 boxes) là nguyên nhân khiến mAP của 2 lớp này bằng $0.0\%$.
- Phương án bổ sung:
  1. Trích xuất bổ sung từ tập OpenImages V7 (chỉ lấy các nhãn đơn lẻ, rõ nét, không dính người).
  2. Tận dụng tập ảnh rác thực tế nội địa từ các nguồn công khai có bản quyền mở, tập trung vào:
     * Pin tiểu (AA/AAA), pin vuông 9V, pin cúc áo, ắc quy xe máy.
     * Vỏ chuối, cùi bắp, thức ăn thừa, rau củ quả dập, lá cây rụng.
  3. Áp dụng kỹ thuật Copy-Paste Augmentation (đã có khung code sẵn tại `src/detection/copy_paste.py` với `DonorBank`): Cắt dán các vật thể Pin và Rác hữu cơ đã được kiểm định chất lượng cao lên các nền đường phố đa dạng.

### 3.2. Chuẩn hóa lại ranh giới Bounding Box
- Loại bỏ các box bao trùm toàn bộ ảnh (toàn cảnh bãi rác) gây nhiễu receptive field.
- Bounding box phải bám khít (tight, snug) quanh từng vật thể riêng biệt: từng chai nhựa, từng lon nước, từng túi nilon thay vì một box khổng lồ ôm trọn cả đống rác.

---

## 4. Bảo Vệ Tuyệt Đối Tập Kiểm Chứng Độc Lập `data/holdout`

- Tập ảnh `data/holdout` (27 ảnh, 57 vật thể Ground Truth) do Sub-Agent D quản lý là **bộ chuẩn nghiệm thu độc lập (Unseen Test Holdout)**.
- **Quy tắc bất biến:**
  1. Tuyệt đối không đưa bất kỳ ảnh nào từ `data/holdout` vào tập huấn luyện hoặc tập validation.
  2. Mã băm SHA-256 của từng file trong tập nghiệm thu được kiểm tra tự động trước mỗi lần huấn luyện để ngăn chặn hoàn toàn rò rỉ dữ liệu (Data Leakage).
  3. Không tối ưu hóa siêu tham số (Hyperparameter Tuning) trực tiếp trên tập nghiệm thu này.

---

## 5. Dự Toán Tài Nguyên Phần Cứng Cho Quá Trình Huấn Luyện Mới

Trước khi tiến hành huấn luyện, bảng dự toán tài nguyên được xác định rõ ràng nhằm đảm bảo an toàn hệ thống:

| Hạng mục | Tài nguyên khả dụng | Yêu cầu dự kiến | Khả năng đáp ứng |
|---|---|---|---|
| **GPU / VRAM** | NVIDIA RTX 2050 (4.0 GB VRAM) | Batch size 8 (SSDLite) hoặc Batch size 16 (YOLOv8n) $\rightarrow$ Cần 2.8 - 3.2 GB VRAM | **ĐỦ (Có thể chạy cục bộ)** |
| **System RAM** | 15.31 GB (Khả dụng hiện tại: 0.43 - 1.5 GB) | Cần tối thiểu 4.0 GB RAM trống cho DataLoader workers | **Cần dọn dẹp RAM trước khi train (workers=0 hoặc 2)** |
| **Thời lượng Train** | GPU cục bộ | SSDLite: ~45 phút (60 epochs); YOLOv8n: ~60 phút (80 epochs) | **Khả thi, chạy trong chế độ nền hoặc ngoài giờ** |
| **Quy định ràng buộc** | Main Agent Integrator | **Chỉ chạy 1 tác vụ huấn luyện GPU duy nhất tại một thời điểm** | **Tuân thủ tuyệt đối** |
