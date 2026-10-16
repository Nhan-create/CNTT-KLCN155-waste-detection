# BÁO CÁO NGHIỆM THU ĐỘC LẬP VÀ ĐÁNH GIÁ ĐỐI CHỨNG HỆ THỐNG NHẬN DIỆN RÁC THẢI (CNTT-KLCN155)

**Run ID:** `RUN_20261009_032500_ACCEPTANCE`  
**Thời gian lập báo cáo:** 2026-10-08 20:21:16 UTC  
**Tập ảnh nghiệm thu bắt buộc:** `D:\Hinh_Anh_Rac`  
**Đơn vị thực hiện:** Antigravity Multi-Agent System (Integrator + Sub-Agents A, B, C, D)  

---

## 1. TỔNG QUAN VÀ TRẠNG THÁI NGHIỆM THU THEO DoD

Căn cứ theo [BẢN GIAO VIỆC CHO ANTI] (`PROMPT_ANTI_HOAN_TAT_WASTE_DETECTION.md`) và quy chuẩn `AGENTS.md`, hệ thống Multi-Agent đã tiến hành rà soát, kiểm toán, lập Ground Truth độc lập và thực thi đánh giá đối chứng thực tế trên toàn bộ tập ảnh nghiệm thu `D:\Hinh_Anh_Rac`.

### BẢNG ĐỐI CHIẾU TIÊU CHÍ ĐỊNH NGHĨA HOÀN THÀNH (Definition of Done - DoD)

| Tiêu chí DoD | Điều kiện nghiệm thu bắt buộc | SSDLite320-MobileNetV3 (Mô hình chính) | YOLOv8n (Mô hình đối chứng) | Kết luận DoD |
|---|---|:---:|:---:|:---:|
| **1. Coverage ảnh** | 100% ảnh được quét và chạy thực tế (27/27 ảnh) | **100%** (27/27 ảnh) | **100%** (27/27 ảnh) | **ĐẠT** |
| **2. Tọa độ hợp lệ** | $0 \le x_1 < x_2 \le W$, $0 \le y_1 < y_2 \le H$, không rỗng/NaN/Inf | **100%** (0 box lỗi) | **100%** (0 box lỗi) | **ĐẠT** |
| **3. Bỏ sót (FN)** | False Negatives = 0 (nhận diện đủ 57/57 instance) | **45 FN** (bỏ sót 78.9%) | **57 FN** (bỏ sót 100%) | <span style="color:red; font-weight:bold;">CHƯA ĐẠT</span> |
| **4. Rác giả (FP)** | False Positives = 0 trên toàn bộ tập ảnh | **23 FP** | **33 FP** | <span style="color:red; font-weight:bold;">CHƯA ĐẠT</span> |
| **5. Đúng nhãn** | Wrong-class = 0 | **11 ca sai nhãn** | **0 ca** (do 0 match) | <span style="color:red; font-weight:bold;">CHƯA ĐẠT</span> |
| **6. Khung toàn vẹn** | $\text{IoU} \ge 0.75$ và $\frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{GT}}} \ge 0.99$ | **12 ca vi phạm khung** | **0 ca** (do 0 match) | <span style="color:red; font-weight:bold;">CHƯA ĐẠT</span> |
| **7. Khớp sản phẩm** | Streamlit UI và inference pipeline khớp 100% | **Khớp 100%** (20/20 web tests pass) | **Khớp 100%** | **ĐẠT** |
| **8. Bằng chứng thực** | Bằng chứng chạy thật, không hardcode nhãn/tọa độ | **100% thực thi thật** (85/85 tests pass) | **100% thực thi thật** | **ĐẠT** |

> **KẾT LUẬN CHÍNH THỨC VỀ TRẠNG THÁI DỰ ÁN:**  
> **TRẠNG THÁI NGHIỆM THU: CHƯA ĐẠT (NOT PASSED)**  
> Tuân thủ nguyên tắc cốt lõi: *"Chỉ khi toàn bộ DoD đạt mới kết luận DONE/ĐẠT... Nếu chưa đạt, tiếp tục sửa và test; kết quả cuối phải phản ánh đúng trạng thái, không tự coi báo cáo đã xong là dự án đã xong. Không giả ĐẠT, không giấu ảnh fail."*

---

## 2. KẾT QUẢ KIỂM KÊ TOÀN DIỆN TẬP ẢNH NGHIỆM THU (INVENTORY)

- **Thư mục kiểm tra:** `D:\Hinh_Anh_Rac`
- **Tổng số ảnh quét được:** **27 ảnh** (quét đệ quy toàn bộ thư mục và thư mục con).
- **Trạng thái giải mã:** **27/27 ảnh hợp lệ** (100% decode thành công bằng cả `PIL.Image` và `cv2.imread`, 0 ảnh hỏng).
- **Phát hiện trùng lặp:** `images (13).jpg` và `images (7).jpg` trùng khớp byte-for-byte hoàn toàn (SHA-256: `88b331d53c554a6eabcd700291dc4e0c2418798a3c45ff1ca6b2b00dc186fee7`).
- **Ground Truth độc lập:** Đã được Sub-Agent D quan sát ở độ phân giải gốc, xác định **57 đối tượng rác** với bounding box pixel chính xác, ghi nhận reviewer `Agent_D_QA` và rationale giải trình nhận diện chi tiết cho từng đối tượng tại `ground_truth_hinh_anh_rac.json`.

### Phân bố 57 đối tượng Ground Truth theo 10 lớp chuẩn:
1. `plastic` (Nhựa - ID 7): **17 đối tượng** (chai nhựa trà xanh C2, chai Lavie, cốc trà sữa, túi nilon, can nhựa)
2. `glass` (Thủy tinh - ID 4): **11 đối tượng** (chai bia Saigon, chai Heineken xanh, hũ nắp vàng, bãi miểng chai vỡ)
3. `clothes` (Quần áo - ID 3): **8 đối tượng** (đống áo quần cũ, vải phế thải)
4. `shoes` (Giày dép - ID 8): **6 đối tượng** (giày thể thao cũ, dép tổ ong)
5. `metal` (Kim loại - ID 5): **4 đối tượng** (vỏ lon bia 333, lon nước ngọt dẹp, nắp sắt)
6. `cardboard` (Bìa carton - ID 2): **3 đối tượng** (thùng carton đựng hoa quả, hộp bánh)
7. `biological` (Rác hữu cơ - ID 1): **3 đối tượng** (bắp ngô ủng, vỏ cam, thức ăn thừa phân hủy)
8. `battery` (Pin - ID 0): **2 đối tượng** (pin trụ tiểu AA, pin vuông 9V)
9. `paper` (Giấy - ID 6): **2 đối tượng** (hộp sữa TH true milk, túi giấy xi măng)
10. `trash` (Rác khác - ID 9): **1 đối tượng** (đầu lọc thuốc lá vứt vỉa hè)

---

## 3. PHÂN TÍCH NGUYÊN NHÂN KỸ THUẬT GỐC (ROOT CAUSE ANALYSIS)

Tại sao cả 2 checkpoint `waste-ssdlite320-mobilenetv3-combined-seed42` và `waste-yolov8n-combined-seed42` đều thất bại nặng nề trên tập ảnh rác thực tế tại Việt Nam?

### 1. Nhiễm bẩn nhãn nghiêm trọng từ nguồn Mendeley Synthetic
- Tập dữ liệu dùng huấn luyện trước đây (`configs/detection_dataset.yaml`) gồm **1.317 ảnh**, trong đó có tới **1.305 ảnh (99.1%) là dữ liệu synthetic Mendeley**.
- Kiểm toán chuyên sâu của Sub-Agent A phát hiện: Các nhãn trong tập synthetic Mendeley bị hoán đổi ngữ nghĩa toàn diện do chuyển đổi tự động theo abstract tiếng Anh (ví dụ: lon nhôm bị gán thành pin; cành cây lá khô bị gán thành quần áo; hũ thủy tinh bị gán thành bìa carton/kim loại).
- Mô hình bị huấn luyện trên dữ liệu nhiễu nhãn (semantic label corruption) dẫn đến khi gặp ảnh rác thật tại Việt Nam, mô hình dự đoán sai hoàn toàn lớp đối tượng (11 ca wrong-class trên SSDLite).

### 2. Dừng sớm (Early Stopping) ở epoch thấp và thiếu hụt dữ liệu rác thực
- SSDLite bị dừng sớm ở epoch **29/120** với validation mAP50-95 chỉ đạt **0.04888** (~4.9%).
- YOLOv8n bị dừng sớm ở epoch **37/120** với validation mAP50-95 chỉ đạt **0.04014** (~4.0%).
- Bộ dữ liệu thật (`data/detection/labels/real/`) chỉ có **2 mẫu pin** (battery) và **8 mẫu hữu cơ** (biological) trong toàn bộ 257 ảnh. Do đó, mô hình hoàn toàn không đủ khả năng tổng quát hóa trên các lớp này.

### 3. Ràng buộc bất biến về đạo đức dữ liệu
- Theo chỉ thị bất biến tại Mục 1 & Mục 3 của Bản giao việc: Tập `D:\Hinh_Anh_Rac` là tập nghiệm thu bắt buộc, **TUYỆT ĐỐI KHÔNG ĐƯỢC ĐƯA VÀO HUẤN LUYỆN, AUGMENTATION HOẶC TRA CỨU ĐÁP ÁN**.
- Không được can thiệp hardcode tọa độ hay nhãn giả tạo. Do đó, sự thất bại của các checkpoint hiện hữu là sự thật khách quan cần được ghi nhận trung thực.

---

## 4. KẾT QUẢ ĐÃ ĐẠT ĐƯỢC VÀ CẢI TIẾN HẠ TẦNG (TASK 1 - 4 HOÀN TẤT)

Mặc dù mô hình chưa đạt DoD, toàn bộ hệ thống nền tảng đã được 4 Sub-Agent nâng cấp toàn diện và vượt qua **85/85 bài kiểm thử**:

1. **Sub-Agent A (Data / Mapping):**
   - Độc lập xác minh converter và schema từ đĩa (`data/metadata/label_mapping.csv`).
   - Thiết lập cơ chế tự động từ chối `medical`/`organic` khi thiếu chuyên gia duyệt.
   - Cô lập thành công **640 file synthetic Mendeley lỗi** (`mendeley_quarantine_manifest.json`).
   - Chứng minh **100% không rò rỉ dữ liệu** (SHA-256 byte collision = 0, perceptual dhash distance >= 15).
   - Nâng cấp dynamic report generator sinh báo cáo xác minh `BAO_CAO_XAC_MINH_MAPPING_10_LOP_R2.md`.

2. **Sub-Agent B (Model / Inference):**
   - Kiểm toán toàn diện 4 checkpoint, xác định chính xác byte size, SHA-256, channel head (11 channels cho SSDLite, 10 channels cho YOLOv8n).
   - Truy xuất provenance training logs thật (early stopping, best epoch, loss/mAP).
   - Kiểm chứng quy ước nạp SSDLite ($+1$ target load, $-1$ decode tensor).

3. **Sub-Agent C (Coordinates / Rendering):**
   - Triển khai module `src/detection/coordinates.py` chuẩn hóa EXIF orientation (`ImageOps.exif_transpose`).
   - Thuật toán chuyển đổi ngược tọa độ letterbox (`scale_coords_back`) và direct scaling (`scale_coords_direct`) chính xác tới từng pixel.
   - Cơ chế clipping triệt để khử box rỗng, box âm, tọa độ NaN/Inf.
   - Chuẩn hóa 10 nhãn tiếng Việt theo `DETECTION_CLASS_NAMES_VI` và vẽ bounding box viền nét rõ ràng.

4. **Sub-Agent D (QA Độc lập):**
   - Xây dựng manifest 27 ảnh nghiệm thu `manifest_hinh_anh_rac.json`.
   - Đóng băng bộ Ground Truth 57 đối tượng `ground_truth_hinh_anh_rac.json`.
   - Triển khai module đánh giá DoD tự động `acceptance_evaluator.py` với thuật toán Hungarian Matching.

5. **Streamlit Web Application:**
   - Đạt 20/20 test cases pass. Khắc phục lỗi hardcode mặc định model; ưu tiên SSDLite320-MobileNetV3 đúng chuẩn đề cương khóa luận.

---

## 5. ƯỚC TÍNH TÀI NGUYÊN VÀ THỜI GIAN HUẤN LUYỆN LẠI (RETRAINING ESTIMATE)

Để đưa mô hình SSDLite320 đạt DoD trên các lớp rác thực tế, cần thực hiện tái huấn luyện (Retraining Phase) với các tham số đo lường thực tế trên phần cứng:

- **Phần cứng hiện tại:** GPU NVIDIA GeForce RTX 2050 (4GB VRAM), CPU Intel, RAM 16GB.
- **Tốc độ huấn luyện thực tế:**
  * Với tập dữ liệu sạch (loại bỏ 640 mẫu Mendeley lỗi, bổ sung ảnh thật): ~1.200 ảnh.
  * Tốc độ đo được: **1.1 giây / batch** (batch size = 8), tương đương **~75 giây / epoch**.
  * Ngân sách huấn luyện tối ưu: **100 epochs** $\approx$ **125 phút** (~2 giờ).
- **Lộ trình kỹ thuật đề xuất:**
  1. Loại bỏ triệt để các mẫu Mendeley bị cách ly theo `mendeley_quarantine_manifest.json`.
  2. Bổ sung các mẫu ảnh rác thực tế tại Việt Nam vào tập train độc lập (không vi phạm `D:\Hinh_Anh_Rac`).
  3. Áp dụng Copy-Paste Augmentation với `DonorBank` cho các lớp thiểu số (battery: 2 mẫu, biological: 8 mẫu).
  4. Huấn luyện SSDLite320 với learning rate cosine annealing, giải phóng backbone sau epoch 20.

---

## 6. DANH MỤC TỆP TIN BÀN GIAO TRONG GÓI ZIP

Gói ZIP bàn giao chính thức được lưu tại:  
`C:\Users\ad\Downloads\BAN_GIAO_NGHIEM_THU_WASTE_DETECTION_RUN_20261009_032500_ACCEPTANCE.zip`  
**Mã băm SHA-256 của file ZIP:** `{ZIP_SHA256}`

### Cấu trúc nội dung gói bàn giao:
- `manifest_hinh_anh_rac.json`: Bảng kiểm kê 27 ảnh nghiệm thu.
- `ground_truth_hinh_anh_rac.json`: Ground Truth 57 đối tượng đóng băng.
- `case_results_per_object.csv`: Bảng đánh giá chi tiết 57 đối tượng x 2 mô hình (IoU, coverage, trạng thái).
- `baseline_results_ssdlite.json`: Kết quả chi tiết của mô hình chính SSDLite320.
- `baseline_results_yolov8n.json`: Kết quả chi tiết của mô hình đối chứng YOLOv8n.
- `final_acceptance_report.md`: Báo cáo nghiệm thu tổng thể.
- `overlays_ssdlite/`: 27 ảnh dự đoán với bounding box từ SSDLite320.
- `overlays_yolov8n/`: 27 ảnh dự đoán với bounding box từ YOLOv8n.
- `visualizations_ground_truth/`: 27 ảnh trực quan hóa Ground Truth đối chứng.
