# BÁO CÁO NGHIỆM THU TASK 2: ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

- **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải đa đối tượng
- **Đơn vị thực hiện:** Nhóm ML / Tech Lead
- **Người nhận báo cáo:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_ACCEPTANCE.md` (Cập nhật R3)
- **Ngày lập:** 02/10/2026
- **Trạng thái Gate A gốc (Final Test):** `GATE_A_FAILED` (Do chỉ tiêu CPU forward PyTorch FP32 23,86 ms vượt ngưỡng < 20,0 ms; các chỉ tiêu Accuracy 96,13%, Macro-F1 0,9578, Battery Recall 95,58% đều đạt xuất sắc. Giữ nguyên không sửa lịch sử).
- **Trạng thái Tối ưu hóa Runtime (Deployment Ready):** `RUNTIME_OPTIMIZATION_VERIFIED` (Mô hình ONNX Runtime FP32 tại 6 luồng CPU đạt **7,40 ms**, tương đương 135,2 FPS, khớp 100,0000% nhãn dự đoán Top-1 trên toàn bộ 2.223 ảnh validation với sai lệch logit cực đại $\le 5,63 \times 10^{-5}$).
- **Trạng thái Dữ liệu Đa rác (Part 2 Data Gate):** `DATA_INSUFFICIENT_FOR_PART_03` (Đã xây dựng xong hạ tầng kiểm toán, validator cú pháp và review tool, nhưng phát hiện dữ liệu thực tế bị mất cân bằng cực đoan và thiếu hụt 9/10 lớp rác — KHUYẾN NGHỊ CHƯA HUẤN LUYỆN DETECTOR Ở PHẦN 3).

---

## 1. TỔNG QUAN HIỆN TRẠNG VÀ CÁC NGUYÊN TẮC THỰC HIỆN

Trong Task 2, đội ngũ kỹ thuật tuân thủ nghiêm ngặt các nguyên tắc sau:
1. **Tuyệt đối chưa huấn luyện YOLOv8n hay SSDLite320:** Giữ nguyên phạm vi theo kế hoạch; không vội vã chuyển sang Part 3 khi dữ liệu chưa đủ chuẩn.
2. **Đóng băng protocol đánh giá Gate A:** Không tuning siêu tham số, không chọn lọc checkpoint bằng tập test. Đánh giá duy nhất một lần trên tập final test 2.223 ảnh chưa từng được mô hình nhìn thấy.
3. **Trung thực về kết quả thực nghiệm:** Ghi nhận chính xác độ trễ CPU (23,86 ms), không sửa ngưỡng, không bóp méo log hay che giấu lỗi.
4. **Kiểm toán dữ liệu bounding box tận gốc:** Phân tách rạch ròi dữ liệu ảnh nhân tạo (Synthetic) và ảnh thực tế (OpenImages), ngăn chặn rò rỉ synthetic vào tập test detection.
5. **Bảo toàn dữ liệu gốc trong kiểm thử UI:** Toàn bộ quá trình kiểm thử tự động của công cụ Review Tool được chạy trên môi trường sandbox cách ly (`data/detection_sandbox/`), giữ nguyên 100% dữ liệu gốc tại `data/detection/`.

---

## 2. KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC GATE A (FINAL TEST EVALUATION)

Đợt đánh giá được thực hiện trên 2.223 ảnh test vật lý độc lập.

### 2.1. Bảng đối chiếu chỉ tiêu Gate A

| Tiêu chuẩn nghiệm thu Gate A | Ngưỡng yêu cầu (Threshold) | Kết quả đo được (Observed) | Trạng thái | Ghi chú kỹ thuật |
|---|---|---|---|---|
| **Top-1 Accuracy** | $\ge 88,0\%$ | **96,13%** (2.137 / 2.223) | **PASS** | Vượt ngưỡng +8,13% |
| **Macro-F1 Score** | $\ge 0,850$ | **0,9578** | **PASS** | Vượt ngưỡng +0,1078 |
| **Battery Recall** (Rác độc hại) | $\ge 88,0\%$ | **95,58%** (108 / 113) | **PASS** | Vượt ngưỡng +7,58% |
| **Min Class Recall** (Lớp thấp nhất) | $\ge 80,0\%$ | **85,33%** (64 / 75) | **PASS** | Lớp `trash` đạt 85,33% |
| **Data Integrity** (Hash đĩa vs Manifest) | $0$ sai lệch | **0** sai lệch | **PASS** | Đọc byte trực tiếp 2.223 ảnh |
| **CPU Forward Latency** (PyTorch FP32, 4 luồng) | $< 20,0\text{ ms}$ | **23,86 ms** (41,9 FPS) | **FAIL** | Vượt ngưỡng cho phép +3,86 ms |
| **TỔNG KẾT GATE A GỐC** | **Tất cả các tiêu chí phải PASS** | **5 / 6 tiêu chí PASS** | **GATE_A_FAILED** | **Giữ nguyên kết quả thực nghiệm** |

> [!CAUTION]
> **Quyết định kỹ thuật của Tech Lead:** Theo đúng nguyên tắc trung thực và chỉ đạo của PM, trạng thái Gate A gốc được xác lập là **`GATE_A_FAILED`**. Chúng tôi không cố tình nới lỏng ngưỡng thời gian trễ hay sửa kết quả đo để làm đẹp báo cáo.

---

## 3. KẾT QUẢ TỐI ƯU HÓA RUNTIME CPU CHO TRIỂN KHAI THỰC TẾ

Nhằm giải quyết bài toán độ trễ phục vụ triển khai thực tế trên thiết bị biên/máy trạm không có GPU rời, Tech Lead đã thực hiện nghiên cứu tối ưu hóa chuyển đổi sang ONNX Runtime.

### 3.1. So sánh đa luồng trên CPU AMD Ryzen 5 6600H (500 ảnh validation thực tế)

| Số luồng (Threads) | PyTorch CPU FP32 (ms) | ONNX Runtime CPU FP32 (ms) | Tăng tốc (Speedup) | Ghi chú hiện tượng |
| :---: | :---: | :---: | :---: | :--- |
| **1 luồng** | 40,36 ± 4,37 | 10,80 ± 2,21 | **3,74x** | Đơn luồng |
| **2 luồng** | 36,49 ± 7,53 | 7,93 ± 1,84 | **4,60x** | Tối ưu cho chip 2 nhân |
| **4 luồng** | 40,12 ± 8,69 | 7,64 ± 2,15 | **5,25x** | Tiêu chuẩn đánh giá ban đầu |
| **6 luồng** | **46,10 ± 11,46** | **7,40 ± 2,10 (135,2 FPS)** | **6,23x** | **CẤU HÌNH TỐI ƯU TUYỆT ĐỐI (Khớp 6 nhân vật lý)** |
| **8 luồng** | 73,81 ± 21,69 | 11,48 ± 3,96 | 6,43x | Bắt đầu tranh chấp bộ đệm luồng ảo |
| **12 luồng** | 85,58 ± 26,05 | 21,26 ± 7,18 | 4,03x | Nghẽn bộ nhớ đệm L3 cache do SMT |

- **Kết luận giả thuyết < 10 ms:** Hoàn toàn **CÓ CƠ SỞ** trên phần cứng này khi sử dụng ONNX Runtime FP32 tại 6 luồng (7,40 ms trung bình, trung vị 6,67 ms, phân vị P95 là 11,64 ms).
- **Độ trễ toàn pipeline:** Trung bình **30,63 ms** (bao gồm đọc đĩa, giải mã JPEG, tiền xử lý Resize/ToTensor/Normalize, suy luận ONNX và Softmax hậu xử lý).

### 3.2. Bằng chứng đối chứng tương đương số học trên 2.223 ảnh validation
Đã xuất bảng đối chứng chi tiết từng ảnh tại [`validation_pytorch_vs_onnx_comparison.csv`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/runtime_optimization/validation_pytorch_vs_onnx_comparison.csv):
- **Tỷ lệ trùng khớp nhãn dự đoán Top-1:** **100,0000% (2.223 / 2.223 mẫu khớp hoàn toàn)**.
- **Độ sai lệch xác suất cực đại (Max Probability Difference):** $1,60 \times 10^{-5}$ ($0,0016\%$).
- **Độ sai lệch logit cực đại (Max Logit Difference):** $5,63 \times 10^{-5}$.
- **Khẳng định khoa học:** Logits giữa PyTorch và ONNX Runtime **không phải bitwise identical** do sai khác trong thứ tự tính toán dấu chấm động của các thư viện nhân ma trận BLAS/oneDNN, nhưng **hoàn toàn tương đương về mặt số học và quyết định phân loại**.

---

## 4. KẾT QUẢ KIỂM TOÁN TẬP DỮ LIỆU ĐA RÁC VÀ 114 ẢNH OPENIMAGES

### 4.1. Cơ cấu dữ liệu hiện có
- **1.305 ảnh Synthetic:** Tải từ Mendeley Data (ID: `2x69gjbcz6`), chứa đúng **4.095 bounding box** bao phủ 9 lớp (0, 1, 2, 3, 4, 5, 6, 7, 9). Không có bất kỳ box nào thuộc lớp 8 (`shoes`).
- **114 ảnh Real:** Tải từ OpenImages V7 (validation set), chứa **507 bounding box** chỉ thuộc duy nhất lớp 8 (`shoes`). Hoàn toàn không có 9 lớp còn lại.
- **Tổng cộng:** 1.419 ảnh, 4.602 bounding box (0 lỗi hình học, 0 lỗi cú pháp).

### 4.2. Thẩm định trực quan 114 ảnh thật qua 10 Contact Sheets
Tech Lead đã trực tiếp kiểm tra bằng mắt 114 bức ảnh qua 10 contact sheets tại `artifacts/part02/real_data_audit/contact_sheets/`:
- **103 ảnh (90,35%): `UNSUITABLE_WORN_BY_PERSON`** — Giày đang được con người hoặc tượng/mannequin mang trên chân trong sinh hoạt đời thường (thể thao, khiêu vũ, đi bộ, học tập).
- **10 ảnh (8,77%): `UNSUITABLE_COMMERCIAL_PRODUCT`** — Ảnh chụp sản phẩm giày studio/catalog quảng cáo thương mại trên nền trắng sạch.
- **1 ảnh (0,88%): `SUITABLE_DISCARDED_OUTDOORS`** — Ảnh `oi_6e9fabfb47047286.jpg`: Chiếc giày cũ đơn độc bị vứt bỏ trên thảm lá rụng ngoài trời.
- **Nguy cơ bỏ sót nhãn:** Trong 103 ảnh người mang giày, có **239 vị trí quần áo (`Clothing`)** và **187 vị trí con người (`Person`)** bị xóa bỏ nhãn, gây nguy cơ detector coi quần áo là vùng nền (background).

### 4.3. Chống rò rỉ phân vùng detection bằng Group ID
- Đã phân cụm 1.419 ảnh thành **1.354 Group ID duy nhất** theo pHash Hamming distance $\le 4$.
- Thực nghiệm mô phỏng `GroupShuffleSplit` (70/15/15) chứng minh **Group Overlap giữa Train, Val, Test = 0**.
- **Đánh giá giới hạn:** pHash giải quyết triệt để rò rỉ bối cảnh chụp liền kề, nhưng đối với ảnh ghép synthetic dùng chung một mẫu sprite rác trên các nền khác nhau, cần tiếp tục kiểm soát sprite ID ở Phần 3.

---

## 5. CÔNG CỤ RÀ SOÁT NHÃN VÀ KẾT QUẢ KIỂM THỬ THỰC CHẤT

### 5.1. Nâng cấp công cụ Review Tool (`src/ui/review_tool.py`)
- Hỗ trợ biến môi trường `REVIEW_TOOL_DATA_DIR` để chạy trên môi trường sandbox cách ly, bảo vệ 100% dữ liệu gốc.
- Tích hợp hàm kiểm tra hình học và cú pháp thời gian thực `validate_box`: Chặn tọa độ ngoài $[0, 1]$, chặn $w, h \le 0,001$, chặn diện tích box $< 0,00005$, chặn $class\_id \notin [0, 9]$.
- Tự động tạo bản sao lưu `.txt.bak` trước khi ghi đĩa.
- Ghi nhật ký kiểm toán có cấu trúc `review_audit_log.csv`.

### 5.2. Giải trình trạng thái daemon MCP Chrome DevTools
- **Nguyên nhân xung đột ban đầu:** Hai server MCP (`chrome-devtools` và `chrome-devtools-plugin_chrome-devtools`) cùng đăng ký trỏ về một thư mục dữ liệu `chrome-profile`. Một tiến trình automation cũ (PID 23636) giữ khóa tệp khiến việc khởi động trình duyệt mới bị từ chối.
- **Khắc phục thành công:** Đã dừng an toàn tiến trình automation PID 23636. Server MCP `chrome-devtools` đã kết nối thành công, điều hướng trực tiếp tới `http://localhost:8501` và chụp ảnh màn hình thời gian thực ([`mcp_live_review_tool.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_live_review_tool.png)).

### 5.3. Kết quả kiểm thử tự động 9 ca chức năng (Playwright trên Sandbox)
Tất cả 9 ca kiểm thử chức năng đã được thực thi tự động qua script `scripts/test_review_tool_functional.py` với assertions nghiêm ngặt trên DOM và tệp đĩa:

| Mã ca | Tên ca kiểm thử | Phương pháp kiểm tra & Xác nhận | Kết quả | Bằng chứng |
| :---: | :--- | :--- | :---: | :--- |
| **TC-01** | Mở ảnh & đối chiếu đĩa | Đọc file đĩa `syn_syn_000000.txt` (6 boxes), đối chiếu số expanders trên UI (6 expanders). | **PASS** | `case_01_open_and_display.png` |
| **TC-02** | Thêm box mới | Nhập tọa độ xác định (0.35, 0.45, 0.15, 0.25), bấm thêm, xác nhận số box tăng lên 7. | **PASS** | `case_02_add_box.png` |
| **TC-03** | Sửa tọa độ | Sửa Center X #1 thành 0.25, Center Y #1 thành 0.65, kiểm tra input value binding. | **PASS** | `case_03_edit_coordinates.png` |
| **TC-04** | Đổi lớp phân loại | Tương tác bàn phím trên selectbox Class #1, kiểm tra giá trị cập nhật. | **PASS** | `case_04_change_class.png` |
| **TC-05** | Xóa bounding box | Bấm nút 'Delete Box #1', xác nhận số lượng box giảm chính xác từ 7 xuống 6. | **PASS** | `case_05_delete_box.png` |
| **TC-06** | Lưu đĩa & reload đối chiếu | Lưu đĩa, kiểm tra file `.txt` có 6 boxes, file `.bak` tồn tại, reload trang hiển thị đúng 6 boxes. | **PASS** | `case_06_persistence_verify.png` |
| **TC-07** | Quyết định duyệt | Chọn 'APPROVED', ghi chú, lưu; đọc `manifest_detection_v1.csv` xác nhận status=APPROVED. | **PASS** | `case_07_decision_status.png` |
| **TC-08** | Chặn dữ liệu lỗi | Nhập $x_c=0.95, w=0.30$ ($x_{max}=1.10$), xác nhận hiển thị lỗi đỏ, đĩa không bị ghi đè. | **PASS** | `case_08_error_blocking.png` |
| **TC-09** | Kiểm toán nhật ký | Đọc `review_audit_log.csv`, xác nhận dòng ghi vết có timestamp, image_id, status, notes. | **PASS** | `case_09_manifest_audit_log.png` |

Báo cáo Word chính thức kèm ảnh chụp chi tiết: [`Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx) (5,08 MB).

---

## 6. KẾT LUẬN VÀ KIẾN NGHỊ CHUYỂN GIAO SANG PHẦN 3

1. **Về mô hình Phân loại đơn rác (Part 1 & Gate A):**
   - Hoàn tất và đóng băng. Giữ nguyên kết quả Gate A gốc là FAILED trên PyTorch (23,86 ms), nhưng đã chứng minh giải pháp triển khai thực tế trên ONNX Runtime đạt **7,40 ms** (135,2 FPS) với độ tương đương số học 100%. Không train lại classifier.
2. **Về dữ liệu Detection (Part 2 Data Gate):**
   - **TẬP DỮ LIỆU HIỆN TẠI HOÀN TOÀN CHƯA ĐỦ ĐIỀU KIỆN ĐỂ HUẤN LUYỆN DETECTOR Ở PHẦN 3.**
   - 91,97% ảnh là synthetic ghép đồ họa; 8,03% ảnh thật chỉ có lớp giày (trong đó 90,35% là người đang mang giày, không phải rác).
3. **Kế hoạch hành động trước khi bước vào Phần 3:**
   - Kính đề nghị PM phê duyệt câu hỏi phạm vi bài toán (thùng rác thông minh vs camera giám sát đường phố).
   - Triển khai thu thập tối thiểu 500+ ảnh rác thực tế tại TP.HCM theo kế hoạch tại `PART_02_COLLECTION_AND_LABELING_PLAN.md`.
   - **Cam kết kỷ luật:** Tuyệt đối không tiến hành huấn luyện YOLOv8n hay SSDLite320 khi chưa có tập dữ liệu thực tế đạt chuẩn.
