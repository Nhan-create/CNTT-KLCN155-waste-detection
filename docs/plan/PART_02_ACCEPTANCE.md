# BÁO CÁO NGHIỆM THU TASK 2: ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

- **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải đa đối tượng
- **Đơn vị thực hiện:** Nhóm ML / Tech Lead
- **Người nhận báo cáo:** PM Ngô Thanh Nhân (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã tài liệu:** `PART_02_ACCEPTANCE.md` (Cập nhật R3)
- **Ngày lập:** 02/10/2026
- **Trạng thái Gate A gốc (Final Test):** `GATE_A_FAILED` (Do chỉ tiêu CPU forward PyTorch FP32 23,86 ms vượt ngưỡng < 20,0 ms; các chỉ tiêu Accuracy 96,13%, Macro-F1 0,9578, Battery Recall 95,58% đều đạt xuất sắc. Giữ nguyên không sửa lịch sử).
- **Trạng thái Tối ưu hóa Runtime (Deployment Ready):** `RUNTIME_OPTIMIZATION_VERIFIED` (Mô hình ONNX Runtime FP32 tại 6 luồng CPU đạt **7,40 ms**, tương đương 135,2 FPS, khớp 100,0000% nhãn dự đoán Top-1 trên toàn bộ 2.223 ảnh validation với sai lệch logit cực đại $\le 5,63 \times 10^{-5}$).
- **Trạng thái Công cụ Review (Review Tool Verification):** `REVIEW_TOOL_VERIFIED` (Đạt 9/9 ca kiểm thử chức năng tự động Playwright trên sandbox cô lập; kiểm chứng thao tác live bằng MCP Chrome DevTools; bảo toàn 100% hash SHA-256 của 2.841 tệp dữ liệu gốc).
- **Trạng thái Dữ liệu Đa rác (Part 2 Data Gate):** `DATA_PENDING_MULTICLASS_SUPPLEMENT` (Phát hiện ảnh thật hiện tại chỉ có duy nhất lớp giày và 88,6% là giày đang mang; thiếu hụt 9/10 lớp rác thực tế. Đang chờ bổ sung nguồn dữ liệu thật và quyết định phạm vi từ PM).
- **KẾT LUẬN TRẠNG THÁI TASK 2:** **`TOOL_VERIFIED_DATA_PENDING`** (Công cụ đạt chuẩn, dữ liệu detection cần bổ sung trước khi train detector Phần 3).

---

## 1. TỔNG QUAN HIỆN TRẠNG VÀ CÁC NGUYÊN TẮC THỰC HIỆN

Trong Task 2, đội ngũ kỹ thuật tuân thủ nghiêm ngặt các nguyên tắc sau:
1. **Tuyệt đối chưa huấn luyện YOLOv8n hay SSDLite320:** Giữ nguyên phạm vi theo kế hoạch; không vội vã chuyển sang Part 3 khi dữ liệu chưa đủ chuẩn.
2. **Đóng băng protocol đánh giá Gate A:** Không tuning siêu tham số, không chọn lọc checkpoint bằng tập test. Đánh giá duy nhất một lần trên tập final test 2.223 ảnh chưa từng được mô hình nhìn thấy.
3. **Trung thực về kết quả thực nghiệm:** Ghi nhận chính xác độ trễ CPU (23,86 ms), không sửa ngưỡng, không bóp méo log hay che giấu lỗi.
4. **Kiểm toán dữ liệu bounding box tận gốc:** Phân tách rạch ròi dữ liệu ảnh nhân tạo (Synthetic) và ảnh thực tế (OpenImages), ngăn chặn rò rỉ synthetic vào tập test detection.
5. **Bảo toàn dữ liệu gốc trong kiểm thử UI:** Toàn bộ quá trình kiểm thử tự động của công cụ Review Tool được chạy trên môi trường sandbox cách ly (`data/detection_sandbox/`), đối soát mã băm trước và sau chứng minh 100% tệp dữ liệu gốc tại `data/detection/` không bị thay đổi bit nào.

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

## 4. KẾT QUẢ THẨM ĐỊNH TRỰC QUAN 114 ẢNH OPENIMAGES

Tech Lead ML đã thẩm định trực tiếp bằng mắt 114/114 ảnh qua 10 contact sheets tại `artifacts/part02/real_data_audit/contact_sheets/` và đối soát metadata gốc:

| Phân nhóm bối cảnh trực quan | Số lượng ảnh | Tỷ lệ | Đặc điểm bối cảnh thực tế | Đánh giá kỹ thuật & Hệ quả |
|---|:---:|:---:|---|---|
| **`WORN_BY_PERSON`** | **101** | **88,60%** | Giày đang mang trên chân người sống/mannequin (thể thao, khiêu vũ, học đường). | Đã bị tước bỏ 239 nhãn `Clothing` và 187 nhãn `Person`. Nguy cơ cao đầu độc lớp `clothes` (class 3). |
| **`COMMERCIAL_PRODUCT_STUDIO`** | **10** | **8,77%** | Ảnh sản phẩm thương mại studio/catalog trên phông nền trắng/xám sạch. | Vật phẩm thương mại đơn lập, không phản ánh rác thải bỏ. |
| **`UNRESOLVED`** | **2** | **1,75%** | `oi_3e6ea8c52a3e9792`, `oi_e15b3f94b4d3e3eb`: Cảnh ngoài trời phức tạp có xe cộ, đàn chó dạo phố; box nhỏ ở xa. | Ngữ cảnh thải bỏ chưa thể xác định chắc chắn từ ảnh chụp. |
| **`APPEARS_DISCARDED_OUTDOORS`** | **1** | **0,88%** | `oi_6e9fabfb47047286.jpg`: Chiếc giày cũ đơn độc nằm ngoài trời trên thảm lá rụng mùa thu. | Có vẻ bị bỏ ngoài trời; lưu ý metadata OpenImages không có nhãn xác nhận đây là rác thải bỏ. |
| **TỔNG CỘNG** | **114** | **100%** | 100% ảnh đã được đối soát qua 10 contact sheets. | **Trạng thái thẩm định: PENDING_PM_SCOPE_DECISION** |

---

## 5. BẢNG KIỂM KÊ 10 LỚP VÀ PHƯƠNG ÁN BỔ SUNG DỮ LIỆU ĐA RÁC

### 5.1. Bảng kiểm kê hiện trạng theo 10 lớp vật liệu

| ID | Tên lớp (Class) | Số ảnh Real đã duyệt | Số BBox Real | Số ảnh Synthetic | Số BBox Synthetic | Đánh giá hiện trạng | Nguồn & Phương án bổ sung khả thi (Không yêu cầu PM tự chụp toàn bộ) |
|:---:|---|:---:|:---:|:---:|:---:|---|---|
| 0 | `battery` | 0 | 0 | 397 | 462 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/01c648` (Battery) + tập TACO (Batteries). |
| 1 | `biological` | 0 | 0 | 379 | 446 | **Thiếu hoàn toàn ảnh thật** | Trích xuất TACO (Food waste) + gán nhãn ảnh phân loại VN-trash sẵn có trong repo. |
| 2 | `cardboard` | 0 | 0 | 379 | 452 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/025dyy` (Box/Cardboard) + TACO (Corrugated cardboard). |
| 3 | `clothes` | 0 | 0 | 389 | 462 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/01g317` (Clothing) đơn lập; loại bỏ ảnh người mặc. |
| 4 | `glass` | 0 | 0 | 383 | 476 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/09tvcd` (Wine glass) + TACO (Glass bottle, Broken glass). |
| 5 | `metal` | 0 | 0 | 372 | 448 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/02jnhm` (Tin can) + TACO (Drink can, Food can). |
| 6 | `paper` | 0 | 0 | 353 | 422 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/02w3_ws` (Paper towel) + TACO (Newspaper, Paper cup). |
| 7 | `plastic` | 0 | 0 | 400 | 472 | **Thiếu hoàn toàn ảnh thật** | Trích xuất OpenImages `/m/04dr76x` (Bottle), `/m/054_l` (Bag) + TACO (Plastic). |
| 8 | `shoes` | 114 | 507 | 0 | 0 | **Đã có ảnh thật (cần lọc)** | Đã có 114 ảnh OpenImages; giữ 10 ảnh studio + 1 ảnh lá; cách ly 101 ảnh mang trên chân. |
| 9 | `trash` | 0 | 0 | 385 | 455 | **Thiếu hoàn toàn ảnh thật** | Trích xuất TACO (Unlabeled litter, Cigarette) + gán nhãn tập TrashNet sẵn có. |
| **Σ** | **TỔNG HỢP** | **114** | **507** | **1.305** | **4.095** | **Lệch cực đoan: 9/10 lớp là 0 box thật** | **Sử dụng 4 nguồn dữ liệu mở + 50-100 ảnh tự chụp kiểm thử tại HUIT** |

> [!IMPORTANT]
> **Vai trò của Synthetic:** 1.305 ảnh synthetic (4.095 box) chỉ được sử dụng làm tập tăng cường huấn luyện (Training augmentation). **Tuyệt đối không đưa synthetic vào tập Validation hoặc Test thực tế** để đánh giá detector ở Phần 3.

---

## 6. KẾT QUẢ KIỂM THỬ THỰC CHẤT CÔNG CỤ REVIEW TOOL

Công cụ `src/ui/review_tool.py` đã vượt qua toàn bộ 9 ca kiểm thử chức năng tự động với Playwright trên sandbox cô lập và được kiểm chứng thao tác live trực tiếp bằng MCP Chrome DevTools:

| Mã ca | Tên ca kiểm thử | Hành động thực hiện | Phương pháp đối chiếu & Xác minh | Kết quả | Bằng chứng |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **TC-01** | Mở ảnh & đối chiếu đĩa | Tải trang Streamlit, đọc ảnh mặc định `syn_syn_000000.jpg`. | Đọc file đĩa `syn_syn_000000.txt` (6 boxes), đối chiếu số expanders trên UI (6 expanders). Khớp 100%. | **PASS** | `case_01_open_and_display.png` |
| **TC-02** | Thêm box mới có tọa độ cụ thể | Nhập form thêm box: xc=0.35, yc=0.45, w=0.15, h=0.25, class=cardboard. Bấm Add Box. | Số lượng box expander tăng từ 6 lên 7; Box #7 hiển thị nhãn cardboard. | **PASS** | `case_02_add_box.png` |
| **TC-03** | Sửa tọa độ Box #1 (Bảo tồn) | Sửa Center X #1=0.25, Center Y #1=0.65, Width #1=0.18, Height #1=0.22. Không xóa box. | Form input phản hồi và lưu giữ giá trị mới (0.25, 0.65, 0.18, 0.22). Box #1 được giữ nguyên để đối soát lưu đĩa. | **PASS** | `case_03_edit_coordinates.png` |
| **TC-04** | Đổi lớp Box #1 (Assert khác biệt) | Đổi lớp của Box #1 từ `4: glass` sang `6: paper` bằng phím điều hướng. | Khẳng định `class_before ('4: glass') != class_after ('6: paper')`. Giữ nguyên Box #1 để kiểm tra việc ghi đĩa. | **PASS** | `case_04_change_class.png` |
| **TC-06** | Lưu đĩa & reload đối chiếu | Bấm 'Save Changes & Update Manifest', đọc từng dòng file `.txt` trên đĩa, reload trang web. | File đĩa lưu chính xác Box #1 (class=6, xc=0.25, yc=0.65, w=0.18, h=0.22) và Box #7 (class=2, xc=0.35, yc=0.45). File `.bak` được tạo. Reload trang hiển thị đủ 7 boxes với đúng tọa độ đã lưu. | **PASS** | `case_06_persistence_verify.png` |
| **TC-05** | Xóa box và xác nhận lưu đĩa | Bấm '🗑️ Delete Box #7', bấm Lưu để cập nhật ổ cứng. | Số lượng box trên UI giảm từ 7 xuống 6. Đọc file đĩa xác nhận còn đúng 6 boxes, Box #1 vẫn được bảo toàn nguyên vẹn với class=6 và tọa độ đã sửa. | **PASS** | `case_05_delete_box.png` |
| **TC-07** | Chuyển trạng thái thẩm định thật | Thực hiện chuỗi chuyển trạng thái thật: APPROVED -> REJECTED -> APPROVED, kèm ghi chú cho từng bước. | Đọc trực tiếp `manifest_detection_v1.csv` trên đĩa sau mỗi lần lưu: xác nhận trạng thái chuyển dịch chuẩn xác (APPROVED -> REJECTED -> APPROVED), num_boxes=6. | **PASS** | `case_07_decision_status.png` |
| **TC-08** | Chặn dữ liệu lỗi (Add & Edit) | Kiểm tra 2 trường hợp lỗi: (1) Nhập tọa độ vượt biên khi thêm box mới (xc=0.95, w=0.30 -> xmax=1.10); (2) Sửa Box #1 thành tọa độ lỗi rồi bấm Lưu thay đổi. | Cả hai trường hợp đều kích hoạt cảnh báo lỗi màu đỏ 'Validation Failed / exceed image boundaries [0, 1]'. Đối soát mã băm SHA-256 tệp nhãn và manifest: hoàn toàn không đổi. | **PASS** | `case_08_error_blocking.png` |
| **TC-09** | Nhật ký kiểm toán có cấu trúc | Đọc và thẩm định toàn bộ các dòng ghi trong file `review_audit_log.csv` trên ổ cứng. | Ghi nhận đủ 5 bản ghi có cấu trúc chuẩn; ghi nhận đầy đủ chu kỳ chuyển trạng thái (APPROVED, REJECTED), số lượng boxes, timestamp ISO và ghi chú của kiểm định viên. | **PASS** | `case_09_manifest_audit_log.png` |

- **Bằng chứng thao tác trực tiếp qua Chrome DevTools MCP:**
  - Ảnh thao tác MCP sửa tọa độ, đổi lớp và lưu thành công: [`mcp_live_interaction_saved.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_live_interaction_saved.png).
  - Ảnh thao tác MCP nhập tọa độ vượt biên và hiển thị cảnh báo đỏ chặn lưu: [`mcp_live_error_blocking.png`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/mcp_test_evidence/mcp_live_error_blocking.png).
- **Kiểm tra tính toàn vẹn dữ liệu gốc:** Đối soát 2.841 file trong `data/detection/` trước và sau toàn bộ quá trình kiểm thử: **100% bitwise identical**.
- **Báo cáo Word chính thức:** Đã tạo và cập nhật đầy đủ tại [`Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx`](file:///D:/CNTT-KLCN155-waste-detection/artifacts/part02/Bao_Cao_Kiem_Thu_Cong_Cu_Review_Tool.docx) (4,58 MB).

---

## 7. KẾ HOẠCH BÀN GIAO VÀ PHÂN ĐỊNH TRÁCH NHIỆM

### 7.1. Phân định công việc

| Hạng mục công việc | Người thực hiện | Thời gian dự kiến | Điều kiện phụ thuộc / Ghi chú |
|---|:---:|:---:|---|
| **1. Chốt phạm vi ứng dụng & vùng nhìn camera** | **PM Ngô Thanh Nhân** | 0,5 giờ | Quyết định bài toán: Khoang thu gom/Thùng thông minh vs Camera đường phố mở |
| **2. Viết script trích xuất 9 lớp rác từ OpenImages V7 & TACO** | Tech Lead (Tự chủ) | 4 giờ | Phụ thuộc metadata OpenImages có sẵn trên máy và API/bộ lọc TACO |
| **3. Gán nhãn bounding box tập rác nội bộ (TrashNet / VN-trash)** | Tech Lead / Nhóm ML | 6 – 8 giờ | Sử dụng chính công cụ Review Tool đã được kiểm chứng |
| **4. Thu thập tập ảnh thực nghiệm thực địa HUIT (50 – 100 ảnh)** | Nhóm sinh viên thực hiện | 1 – 2 buổi | Chụp tại sảnh giảng đường, căn tin, phòng lab trường ĐH Công Thương TP.HCM |
| **5. Rà soát chất lượng (QA/QC 2 vòng) và đóng gói tập dữ liệu Part 2** | Tech Lead (Tự chủ) | 4 giờ | Thẩm định 100% box bằng Review Tool, xuất manifest V2 và phân cụm Group ID |
| **6. Khởi động huấn luyện YOLOv8n & SSDLite320 (Part 3)** | Tech Lead / Nhóm ML | 12 – 16 giờ | **Bắt đầu sau khi hoàn thành mục 1 đến 5** |

### 7.2. Dự kiến mốc thời gian hoàn tất
- **Mốc hoàn tất kiểm chứng công cụ Review Tool:** **ĐÃ HOÀN TẤT 100%** (02/10/2026).
- **Mốc hoàn tất bổ sung và chuẩn bị dữ liệu đa rác đạt chuẩn:** **Dự kiến 3 ngày làm việc** sau khi PM chốt định hướng phạm vi ứng dụng.
