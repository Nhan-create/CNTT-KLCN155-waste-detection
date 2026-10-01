# BÁO CÁO THỰC NGHIỆM TỐI ƯU HÓA RUNTIME CPU CHO MOBILENETV3-LARGE (NHÁNH A)

- **Dự án:** CNTT-KLCN155 — Phát hiện và phân loại rác thải đa đối tượng
- **Tác giả:** Tech Lead ML / Nhóm Kỹ sư ML
- **Người nhận:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_CPU_OPTIMIZATION.md`
- **Ngày lập:** 02/10/2026
- **Trạng thái thực nghiệm:** `OPTIMIZATION_SUCCESSFUL` (Khẳng định giả thuyết Sub-10ms trên CPU vật lý)
- **Trạng thái Gate A gốc:** `GATE_A_FAILED_PRESERVED` (Bảo toàn lịch sử đánh giá gốc 23,86 ms trên final test; báo cáo độc lập thí nghiệm tối ưu trên tập validation)

---

## 1. MÔI TRƯỜNG THỰC NGHIỆM VÀ ĐIỀU KIỆN PHẦN CỨNG

Các phép đo được thực hiện trực tiếp trên phần cứng máy trạm phát triển của dự án:
- **Hệ điều hành:** Microsoft Windows 11 Home Single Language (64-bit), Version 10.0.26200.
- **Vi xử lý (CPU):** AMD Ryzen 5 6600H with Radeon Graphics (6 nhân vật lý / 12 luồng logic, xung nhịp cơ bản 3.3 GHz, boost tối đa 4.5 GHz).
- **Bộ nhớ RAM:** 16.0 GB (khả dụng ~15.3 GB, bus 4800 MHz DDR5).
- **Trạng thái nguồn điện:** Cắm nguồn AC trực tiếp (`PowerOnline = True`), chế độ Balanced/High Performance.
- **Môi trường phần mềm:**
  - Python: 3.12.10 (64-bit)
  - PyTorch: 2.5.1+cu121 (chạy chế độ CPU)
  - Torchvision: 0.20.1+cu121
  - ONNX: 1.23.1
  - ONNX Runtime: 1.30.0 (CPUExecutionProvider)

---

## 2. QUY CHUẨN ĐÓNG BĂNG GIAO THỨC (PROTOCOL FROZEN TRƯỚC KHI CHẠY)

Tuân thủ nghiêm ngặt chỉ đạo của PM Ngô Thanh Nhân:
1. **Lưu và băm giao thức trước khi thực thi:** File cấu hình `artifacts/part02/runtime_optimization/optimization_protocol_frozen.json` (SHA-256: `11abbd851f392beace4e58561f99f0ad3b3a3b2b0c4a97391879bf8a66dfa9a3`) được tạo và băm vào lúc bắt đầu chạy script, trước khi bất kỳ phép đo nào diễn ra.
2. **Dữ liệu đo đạc:** Sử dụng **100% ảnh thật từ tập validation** (`data/processed_v2/val/`, 2.223 ảnh), không dùng tensor ngẫu nhiên (`torch.randn`).
3. **Thiết lập chuẩn:**
   - Kích thước ảnh: $224 \times 224$ pixels.
   - Chuẩn hóa: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
   - Batch size: 1 (mô phỏng đúng trễ thời gian thực cho từng ảnh tại biên).
   - Warmup: 50 lượt trước khi ghi nhận dữ liệu.
   - Số lượt đo chính thức: 200 lượt với các mẫu ảnh validation được cố định ngẫu nhiên (seed = 42).
   - Khảo sát các cấu hình luồng: $1, 2, 4, 6, 8, 12$ luồng.

---

## 3. KẾT QUẢ THỰC NGHIỆM SO SÁNH RUNTIME VÀ LUỒNG XỬ LÝ

### 3.1. Bảng tổng hợp hiệu năng Forward Pass (Batch size 1, FP32)

| Framework | Số luồng (Threads) | Mean Latency (ms) | Median Latency (ms) | P95 Latency (ms) | Min (ms) | Max (ms) | Tốc độ (FPS) | Đạt ngưỡng $<20\text{ ms}$ | Đạt giả thuyết $<10\text{ ms}$ |
|---|---|---|---|---|---|---|---|---|---|
| **PyTorch FP32** (`eval` + `inference_mode`) | 1 | 40,36 | 38,88 | 52,09 | 28,88 | 94,79 | 24,8 | Không | Không |
| PyTorch FP32 | 2 | 36,49 | 33,65 | 53,77 | 25,57 | 100,21 | 27,4 | Không | Không |
| PyTorch FP32 | 4 | 40,12 | 36,03 | 65,50 | 26,63 | 148,62 | 24,9 | Không | Không |
| PyTorch FP32 | 6 | 46,10 | 41,72 | 71,17 | 31,69 | 263,60 | 21,7 | Không | Không |
| PyTorch FP32 | 8 | 73,81 | 65,74 | 130,02 | 38,52 | 375,05 | 13,5 | Không | Không |
| PyTorch FP32 | 12 | 85,58 | 71,87 | 163,31 | 46,67 | 241,54 | 11,7 | Không | Không |
| **ONNX Runtime FP32** | 1 | 10,80 | 10,64 | 12,92 | 8,67 | 14,86 | 92,6 | **ĐẠT** | Gần đạt |
| ONNX Runtime FP32 | 2 | 7,93 | 7,81 | 9,95 | 6,04 | 15,18 | 126,0 | **ĐẠT** | **ĐẠT** |
| ONNX Runtime FP32 | 4 | 7,64 | 7,20 | 11,01 | 5,08 | 17,73 | 130,9 | **ĐẠT** | **ĐẠT** |
| **ONNX Runtime FP32** | **6 (Tối ưu)** | **7,40** | **6,67** | **11,64** | **5,24** | **18,93** | **135,2** | **ĐẠT** | **ĐẠT XUẤT SẮC** |
| ONNX Runtime FP32 | 8 | 11,48 | 10,52 | 17,08 | 6,22 | 83,06 | 87,1 | **ĐẠT** | Không |
| ONNX Runtime FP32 | 12 | 21,26 | 16,30 | 49,45 | 4,83 | 86,55 | 47,0 | Không | Không |

### 3.2. Phân tích nguyên nhân kỹ thuật chuyên sâu:
1. **Tại sao PyTorch gốc không vượt qua 20 ms trên CPU này:**
   - Bộ lập lịch luồng (Thread Scheduler / OpenMP) của PyTorch có chi phí đồng bộ hóa (synchronization overhead) lớn khi kích thước tensor nhỏ ($1 \times 3 \times 224 \times 224$). 
   - Với kiến trúc MobileNetV3 gồm nhiều khối Inverted Residual và Depthwise Separable Convolutions liên tiếp, lượng tính toán trên mỗi layer rất nhỏ (FLOPs thấp) nhưng số lượng kernel launch rất cao. PyTorch CPU backend mất phần lớn thời gian cho việc điều phối luồng và quản lý bộ nhớ đệm thay vì tính toán thuần túy.
2. **Tại sao ONNX Runtime đạt tốc độ vượt trội (7,40 ms, 135 FPS):**
   - ONNX Runtime áp dụng kỹ thuật **Graph Fusion / Constant Folding** gộp các chuỗi layer liên tiếp (Conv + BatchNorm + Hardswish) thành các kernel C++ tối ưu hóa AVX-512 / AVX2 trực tiếp trên thanh ghi CPU.
   - **Số luồng tối ưu là 6:** Trùng khớp chính xác với **6 nhân vật lý** của chip AMD Ryzen 5 6600H.
   - Khi tăng lên 8 hoặc 12 luồng (vượt qua số nhân vật lý sang nhân ảo SMT), độ trễ bị suy thoái từ $7,40\text{ ms} \to 21,26\text{ ms}$ do tranh chấp bộ nhớ đệm L3 Cache (L3 Cache Thrashing) và context switching giữa các luồng.

---

## 4. KIỂM ĐỊNH TÍNH TƯƠNG ĐƯƠNG TOÀN DIỆN (EQUIVALENCE AUDIT TRÊN 2.223 ẢNH)

Để đảm bảo việc chuyển đổi sang ONNX Runtime không làm suy thoái chất lượng phân loại hoặc sai lệch nhãn, đội ngũ kỹ thuật đã chạy suy luận đối đầu song song giữa PyTorch FP32 và ONNX Runtime FP32 trên **toàn bộ 2.223 bức ảnh tập validation**:

| Chỉ tiêu kiểm định tương đương | Kết quả đo đạc thực tế | Đánh giá |
|---|---|---|
| **Tổng số mẫu validation đối soát** | **2.223 / 2.223 ảnh** | Toàn bộ tập validation |
| **Số mẫu dự đoán khớp nhau 100% (Top-1 Match)** | **2.223 / 2.223 ảnh** | **100,0000% khớp tuyệt đối** |
| **Số mẫu bị sai lệch nhãn dự đoán** | **0 ảnh** | Không có bất kỳ sai lệch nào |
| **Validation Accuracy (PyTorch gốc)** | **96,22%** (2.139 / 2.223) | Chuẩn đối chiếu |
| **Validation Accuracy (ONNX Runtime)** | **96,22%** (2.139 / 2.223) | **Giữ nguyên 100%** |
| **Sai lệch Logits tuyệt đối lớn nhất ($\max \|z_{\text{py}} - z_{\text{onnx}}\|$)** | **$6,592 \times 10^{-5}$** | Sai số dấu phẩy động cực nhỏ ($< 0,0001$) |
| **Sai lệch Logits trung bình ($\text{mean} \|z_{\text{py}} - z_{\text{onnx}}\|$)** | **$1,860 \times 10^{-6}$** | Thực chất tương đương hoàn toàn |
| **Sai lệch Xác suất Softmax lớn nhất ($\max \|p_{\text{py}} - p_{\text{onnx}}\|$)** | **$1,597 \times 10^{-5}$** | Không ảnh hưởng đến ngưỡng quyết định |
| **Sai lệch Xác suất Softmax trung bình ($\text{mean} \|p_{\text{py}} - p_{\text{onnx}}\|$)** | **$7,527 \times 10^{-8}$** | Gần như bằng 0 |

---

## 5. KẾT LUẬN VÀ KHUYẾN NGHỊ TRIỂN KHAI

1. **Khẳng định giả thuyết:** Giả thuyết “Mô hình có thể đạt dưới 10 ms trên chính máy này” đã được **CHỨNG MINH THỰC NGHIỆM VÀ XÁC NHẬN THÀNH CÔNG (CONFIRMED)** bằng ONNX Runtime FP32 với cấu hình 6 luồng (Mean: **7,40 ms**, Median: **6,67 ms**, P95: **11,64 ms**).
2. **Độc lập và bảo toàn lịch sử:** 
   - Lần đánh giá Gate A ban đầu trên final test vẫn giữ nguyên trạng thái **`GATE_A_FAILED`** (do PyTorch gốc trên 4 luồng đo được 23,86 ms).
   - Thí nghiệm tối ưu hóa này là một nghiên cứu độc lập được thực hiện trên tập validation, chứng minh tính khả thi của việc đóng gói triển khai (deployment ready).
3. **Khuyến nghị cho ứng dụng thực tế:** Khi triển khai hệ thống phân loại tại biên hoặc nhúng vào ứng dụng camera Web/Desktop, dự án nên dùng model file `artifacts/part02/runtime_optimization/mobilenetv3_large_waste.onnx` (SHA-256: `5b506a6e627a98742e5b3d4d616d095bcad46d6703604cb3b6db33e3a68663b5`) với runtime ONNX 6 luồng để đạt thông lượng **135 FPS**, hoàn toàn vượt yêu cầu thời gian thực.
