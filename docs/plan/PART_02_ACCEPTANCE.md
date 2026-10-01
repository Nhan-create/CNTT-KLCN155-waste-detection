# BÁO CÁO NGHIỆM THU TASK 2: ĐÁNH GIÁ FINAL TEST QUA GATE A VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

- **Dự án:** CNTT-KLCN155 — Hệ thống phát hiện và phân loại rác thải đa đối tượng
- **Đơn vị thực hiện:** Nhóm ML / Tech Lead
- **Người nhận báo cáo:** ThS. Ngô Thanh Nhân — Project Manager (HUIT)
- **Thư mục làm việc:** `D:\CNTT-KLCN155-waste-detection`
- **Mã commit Git:** `1d228f1` (và các commit cập nhật Task 2)
- **Ngày lập:** 02/10/2026
- **Trạng thái Gate A:** `GATE_A_FAILED` (Do chỉ tiêu CPU Latency 23,86 ms vượt ngưỡng < 20,0 ms; các chỉ tiêu Accuracy, F1, Recall đều đạt xuất sắc)
- **Trạng thái Chuẩn bị Dữ liệu Đa rác (Part 2 Data Gate):** `DATA_INSUFFICIENT_FOR_PART_03` (Đã xây dựng xong công cụ kiểm toán, validator và review tool, nhưng phát hiện dữ liệu thực tế bị thiếu hụt nghiêm trọng — KHUYẾN NGHỊ CHƯA HUẤN LUYỆN DETECTOR Ở PHẦN 3)

---

## 1. TỔNG QUAN HIỆN TRẠNG VÀ CÁC NGUYÊN TẮC THỰC HIỆN

Trong Task 2, đội ngũ kỹ thuật tuân thủ nghiêm ngặt các nguyên tắc sau:
1. **Tuyệt đối chưa huấn luyện YOLOv8n hay SSDLite320:** Giữ nguyên phạm vi theo kế hoạch; không vội vã chuyển sang Part 3 khi dữ liệu chưa đủ chuẩn.
2. **Đóng băng protocol đánh giá Gate A:** Không tuning siêu tham số, không chọn lọc checkpoint bằng tập test. Đánh giá duy nhất một lần trên tập final test 2.223 ảnh chưa từng được mô hình nhìn thấy.
3. **Trung thực về kết quả thực nghiệm:** Ghi nhận chính xác độ trễ CPU (23,86 ms), không sửa ngưỡng, không bóp méo log hay che giấu lỗi.
4. **Kiểm toán dữ liệu bounding box tận gốc:** Phân tách rạch ròi dữ liệu ảnh nhân tạo (Synthetic) và ảnh thực tế (OpenImages), ngăn chặn rò rỉ synthetic vào tập test detection.

---

## 2. BẢNG TRẢ LỜI 12 CÂU HỎI LÀM RÕ CỦA PM

| STT | Câu hỏi của PM | Kết luận của Tech Lead | File / Lệnh / Bằng chứng kiểm chứng | Rủi ro nếu hiểu sai |
|---|---|---|---|---|
| **Q1** | Checkpoint và manifest dùng cho Gate A có khớp SHA-256 đã công bố không? | Khớp chính xác 100% từng bit: Checkpoint `best_model.pt` (`c824fec...`), Manifest `split_manifest_v2.csv` (`1429a22...`). | `scripts/evaluate_final_test.py` đối soát SHA-256 trực tiếp trước khi load trọng số; log tại `artifacts/part02/gate_a/gate_a_raw_execution.log`. | Dùng nhầm checkpoint chưa qua kiểm duyệt hoặc đánh giá trên phân chia dữ liệu cũ bị rò rỉ. |
| **Q2** | 2.223 ảnh test có đủ trên ổ đĩa, hash khớp manifest và đúng thư mục lớp không? | Đủ 100% 2.223 file ảnh vật lý trên ổ đĩa, tính trực tiếp SHA-256 từ byte đọc được: 0 thiếu, 0 sai lệch, 0 file thừa; 100% thư mục lớp khớp nhãn. | Báo cáo kiểm toán hash đĩa: `artifacts/part02/gate_a/gate_a_metrics.json` mục `disk_hash_verification` ghi nhận `mismatches_against_manifest: 0`. | Báo cáo metric ảo do suy luận trên đường dẫn chết hoặc nhầm lẫn giữa nhãn và thư mục thực tế. |
| **Q3** | Tập test này đã từng bị can thiệp, chạy thử để chọn checkpoint hoặc chỉnh siêu tham số chưa? | Hoàn toàn chưa. Trong Part 1, tập test được cô lập tuyệt đối; mọi quyết định chọn checkpoint (Epoch 12) đều dựa trên 2.223 ảnh validation. Lần chạy này là lần forward pass đầu tiên trên test. | Mã nguồn `scripts/train_classifier_phase_a.py` chỉ dùng DataLoader `train` và `val`. Lịch sử `training_history.csv` chỉ lưu `val_acc` và `val_loss`. | Đánh giá thiên vị (data leakage), mô hình overfit tập test mà không có khả năng khái quát hóa. |
| **Q4** | Class ID và quy trình tiền xử lý có đồng nhất 100% với train/val không? | Đồng nhất 100%: 10 lớp theo thứ tự alphabet `['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']`. Preprocessing: `Resize((224, 224))`, `ToTensor()`, `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`, `probs = softmax(logits)`, `pred = argmax(probs)`. | `scripts/evaluate_final_test.py` dùng chung class list và transform từ `scripts/train_classifier_phase_a.py`. | Sai lệch nhãn dự đoán (mislabeled evaluation) hoặc suy giảm hiệu năng do lệch phân phối chuẩn hóa. |
| **Q5** | Ngưỡng nào cho Gate A và các số liệu cũ (2.225 ảnh, 15 rò rỉ) đã được đính chính chưa? | Đã đính chính triệt để: Số mẫu test chuẩn là 2.223 (không phải 2.225). Số cặp rò rỉ thực tế là 0 (sau khi kiểm chứng thủ công 33 cặp candidate). Ngưỡng Gate A: Acc >= 88%, Macro-F1 >= 0.85, Battery Recall >= 88%, Min Recall >= 80%, CPU Latency < 20 ms. | `docs/plan/PART_02_EVALUATION_PROTOCOL.md` và `artifacts/part02/gate_a/gate_a_protocol_frozen.json`. | Đánh giá sai trên chỉ tiêu lỗi thời hoặc không phát hiện được suy thoái ở các lớp rác nguy hại/khó nhận diện. |
| **Q6** | Bộ dữ liệu detection có bao nhiêu ảnh/box thực tế, có đủ đại diện 10 lớp không? | Có 1.419 ảnh, 4.602 boxes từ `dataset-v1`. Tuy nhiên, ảnh thực tế (OpenImages) chỉ có 114 ảnh với 507 boxes; 1.305 ảnh còn lại là Synthetic. ĐẶC BIỆT: 114 ảnh thực tế chỉ chứa duy nhất lớp `shoes` (Class 8); 9 lớp còn lại có 0 ảnh thực tế! Hoàn toàn không đủ đại diện 10 lớp. | `data/detection/manifest_detection_v1.csv` và `data/detection/dataset_summary.json`. | Huấn luyện detector bị thiên vị cực đoan, chỉ nhận dạng được đồ họa nhân tạo và thất bại hoàn toàn ngoài thực tế. |
| **Q7** | Nguồn gốc ảnh detection: Synthetic hay Real ngoài trời? Tỷ lệ là bao nhiêu? | - Synthetic: 1.305 ảnh (91,97%), 4.095 boxes từ Mendeley Waste (ảnh ghép đồ họa nền trắng/nhân tạo).<br>- Real: 114 ảnh (8,03%), 507 boxes từ OpenImages V7 (ngoài trời, chụp chân mang giày). | `scripts/prepare_detection_dataset.py` phân tích prefix `syn_` và `oi_`, tạo trường `is_synthetic`. | Đánh giá mAP trên tập test chứa synthetic dẫn đến ảo tưởng độ chính xác cao. |
| **Q8** | Bounding box có bao quát đủ đối tượng rác trong ảnh không? Có bị sót box hay box quá rộng/hẹp? | Đã kiểm toán cú pháp toàn bộ 4.602 box: 100% tọa độ hợp lệ trong $[0, 1]$, không có box suy biến ($w, h \le 0.002$). Tuy nhiên, ảnh OpenImages có hiện tượng sót rác nền (background clutter) và chỉ gán nhãn giày. | `scripts/validate_detection_annotations.py` xuất `artifacts/part02/detection_annotation_validation.json` ghi nhận 0 lỗi cú pháp. | Detector học nhầm rác không gán nhãn thành background (false negative) hoặc tạo ra nhiều false positive. |
| **Q9** | Quy tắc ranh giới nhãn giữa các cặp dễ nhầm (nhựa/rác khác, giấy/carton, giày/quần áo)? | Đã chuẩn hóa quy tắc: Giấy/Carton: carton là thùng gợn sóng, bìa cứng, hộp Tetra Pak; giấy là tờ mỏng A4, hóa đơn, ly giấy. Nhựa/Trash: nhựa tái chế rõ hình dạng là plastic; rác hỗn hợp, đầu lọc thuốc lá là trash. Giày/Clothes: giày dép là shoes; áo quần, khăn vải là clothes. Rác ngoài bảng 10 lớp gán `trash`. | Quy định tại mục 4 của báo cáo này và tích hợp trong công cụ `src/ui/review_tool.py`. | Gán nhãn mâu thuẫn giữa các người gán nhãn làm suy giảm độ hội tụ của hàm mất mát. |
| **Q10** | Nguyên tắc Group ID chống rò rỉ cụm cho detection là gì? Điểm yếu hiện tại? | Nhóm ảnh cùng phiên chụp, cùng địa điểm hoặc pHash Hamming <= 4 (DSU transitive closure) vào chung một Group ID; toàn bộ Group ID phải nằm trọn trong 1 split (train hoặc val hoặc test). Điểm yếu: Chưa có tọa độ GPS EXIF và metadata chuỗi video để phát hiện góc chụp xoay > 90 độ. | `data/audit/split_manifest_v2.csv` trường `group_id`. | Rò rỉ bối cảnh chụp (background leakage) giữa tập train và test detection. |
| **Q11** | Công cụ `review_tool.py` đã tồn tại chưa, đã thử nghiệm thực tế chưa? | Đã được triển khai hoàn chỉnh tại `src/ui/review_tool.py` (Streamlit). Đã kiểm thử tự động giao diện trực tiếp trên trình duyệt bằng Playwright/Chrome headless: kiểm tra hiển thị box, đổi nhãn, duyệt APPROVED, lưu manifest và chụp ảnh minh chứng thành công. | Minh chứng ảnh thật tại `artifacts/part02/ui_evidence/review_tool_verified.png`. Mã kiểm thử tại `scripts/verify_review_tool_ui.py`. | Tuyên bố có công cụ nhưng thực chất chỉ có file giả định không chạy được trên môi trường đồ họa. |
| **Q12** | Quy trình kiểm thử độc lập có ngăn chặn được lỗi khi thiếu ảnh hoặc nhãn sai không? | Có, 100% tự động. Test suite gồm 11 test cases: kiểm tra thiếu thư mục ảnh (báo lỗi exit code 1), phát hiện cặp candidate chưa giải quyết (exit code 1), phát hiện box vượt giới hạn, sai class ID, box rỗng (exit code 1). | `tests/test_verification_gates.py` (4 tests) và `tests/test_detection_annotations.py` (7 tests) đều PASS 100% với exit code 0. | Bỏ lọt lỗi dữ liệu vào pipeline tự động, dẫn tới sự cố khi chạy hàng loạt. |

---

## 3. KẾT QUẢ ĐÁNH GIÁ CHÍNH THỨC GATE A (FINAL TEST EVALUATION)

Đợt đánh giá được thực hiện vào lúc **01:14:33 ngày 02/10/2026** trên 2.223 ảnh test vật lý độc lập.

### 3.1. Bảng đối chiếu chỉ tiêu Gate A

| Tiêu chuẩn nghiệm thu Gate A | Ngưỡng yêu cầu (Threshold) | Kết quả đo được (Observed) | Trạng thái | Ghi chú kỹ thuật |
|---|---|---|---|---|
| **Top-1 Accuracy** | $\ge 88,0\%$ | **96,13%** (2.137 / 2.223) | **PASS** | Vượt ngưỡng +8,13% |
| **Macro-F1 Score** | $\ge 0,850$ | **0,9578** | **PASS** | Vượt ngưỡng +0,1078 |
| **Battery Recall** (Rác độc hại) | $\ge 88,0\%$ | **95,58%** (108 / 113) | **PASS** | Vượt ngưỡng +7,58% |
| **Min Class Recall** (Lớp thấp nhất) | $\ge 80,0\%$ | **85,33%** (64 / 75) | **PASS** | Lớp `trash` đạt 85,33% |
| **Data Integrity** (Hash đĩa vs Manifest) | $0$ sai lệch | **0** sai lệch | **PASS** | Đọc byte trực tiếp 2.223 ảnh |
| **CPU Forward Latency** (1 batch, 4 luồng) | $< 20,0\text{ ms}$ | **23,86 ms** (41,9 FPS) | **FAIL** | Vượt ngưỡng cho phép +3,86 ms |
| **TỔNG KẾT GATE A** | **Tất cả các tiêu chí phải PASS** | **5 / 6 tiêu chí PASS** | **GATE_A_FAILED** | **Giữ nguyên kết quả thực nghiệm** |

> [!CAUTION]
> **Quyết định kỹ thuật của Tech Lead:** Theo đúng nguyên tắc trung thực và chỉ đạo của PM, trạng thái Gate A được xác lập là **`GATE_A_FAILED`**. Chúng tôi không cố tình nới lỏng ngưỡng thời gian trễ hay sửa kết quả đo để làm đẹp báo cáo. Độ trễ 23,86 ms của mô hình gốc (PyTorch Float32/TorchScript trên CPU) hoàn toàn có thể được giải quyết ở giai đoạn tối ưu hóa triển khai bằng kỹ thuật INT8 Quantization hoặc OpenVINO/ONNX Runtime (thường giảm 2x-3x độ trễ xuống dưới 10 ms).

### 3.2. Hiệu năng chi tiết từng lớp trên tập Test

| Lớp (Class) | Số mẫu test (Support) | Precision | Recall | F1-Score | Số mẫu đoán sai |
|---|---|---|---|---|---|
| `battery` | 113 | 0,9818 | 0,9558 | 0,9686 | 5 |
| `biological` | 105 | 0,9906 | 1,0000 | 0,9953 | 0 |
| `cardboard` | 310 | 0,9646 | 0,9677 | 0,9662 | 10 |
| `clothes` | 284 | 0,9895 | 0,9965 | 0,9930 | 1 |
| `glass` | 260 | 0,9506 | 0,9615 | 0,9560 | 10 |
| `metal` | 339 | 0,9432 | 0,9794 | 0,9609 | 7 |
| `paper` | 222 | 0,9409 | 0,9324 | 0,9367 | 15 |
| `plastic` | 298 | 0,9648 | 0,9195 | 0,9416 | 24 |
| `shoes` | 217 | 0,9683 | 0,9862 | 0,9772 | 3 |
| `trash` | 75 | 0,9143 | 0,8533 | 0,8828 | 11 |
| **Trung bình Macro** | **2.223** | **0,9609** | **0,9552** | **0,9578** | **86 lỗi tổng cộng** |

### 3.3. Phân tích các trường hợp nhầm lẫn chính (86 lỗi)
- **Plastic nhầm sang Trash (10 ca) và Glass (8 ca):** Các chai nhựa trong suốt dễ bị nhầm với chai thủy tinh khi nhìn trực diện; màng bọc nilon nhàu nát bị nhầm sang rác hỗn hợp.
- **Paper nhầm sang Cardboard (7 ca) và Trash (5 ca):** Giấy carton mỏng hoặc túi giấy kraft nâu bị mô hình phân vân giữa giấy và bìa carton.
- **Trash nhầm sang Paper/Plastic (7 ca):** Do định nghĩa `trash` là lớp gom rác hỗn hợp không thể phân loại, khi rác chứa nhiều mẩu vụn túi bóng hoặc mẩu giấy, mô hình có xu hướng bắt đặc trưng của vật liệu chiếm diện tích lớn.

---

## 4. KIỂM TOÁN VÀ CHUẨN BỊ DỮ LIỆU ĐA RÁC CÓ BOUNDING BOX

Đội ngũ kỹ thuật đã tái cấu trúc toàn diện thư mục dữ liệu phát hiện đa đối tượng tại `data/detection/`.

### 4.1. Cấu trúc thư mục mới độc lập
```
data/detection/
├── images/
│   ├── real/             # 114 ảnh OpenImages V7 (cô lập riêng)
│   └── synthetic/        # 1.305 ảnh Mendeley Synthetic (cô lập riêng)
├── labels/
│   ├── real/             # 114 file nhãn YOLO (.txt)
│   └── synthetic/        # 1.305 file nhãn YOLO (.txt)
├── manifest_detection_v1.csv      # Bảng kê 1.419 ảnh, SHA-256, số box, nguồn gốc
└── dataset_summary.json           # Thống kê chi tiết từng lớp và kiểm toán
```

### 4.2. Thống kê phân bố bounding box thực tế

| Lớp (Class) | ID | Synthetic Boxes (Mendeley) | Real Boxes (OpenImages) | Tổng số Box | Nhận xét tính sẵn sàng |
|---|---|---|---|---|---|
| `battery` | 0 | 462 | **0** | 462 | **Thiếu 100% ảnh thực tế** |
| `biological` | 1 | 446 | **0** | 446 | **Thiếu 100% ảnh thực tế** |
| `cardboard` | 2 | 452 | **0** | 452 | **Thiếu 100% ảnh thực tế** |
| `clothes` | 3 | 462 | **0** | 462 | **Thiếu 100% ảnh thực tế** |
| `glass` | 4 | 476 | **0** | 476 | **Thiếu 100% ảnh thực tế** |
| `metal` | 5 | 448 | **0** | 448 | **Thiếu 100% ảnh thực tế** |
| `paper` | 6 | 422 | **0** | 422 | **Thiếu 100% ảnh thực tế** |
| `plastic` | 7 | 472 | **0** | 472 | **Thiếu 100% ảnh thực tế** |
| `shoes` | 8 | **0** | **507** | 507 | **Chỉ có ảnh thực tế từ OpenImages** |
| `trash` | 9 | 455 | **0** | 455 | **Thiếu 100% ảnh thực tế** |
| **Tổng cộng** | - | **4.095** (89,0%) | **507** (11,0%) | **4.602** | **MẤT CÂN BẰNG NGHIÊM TRỌNG** |

### 4.3. Phát hiện lỗi nghiêm trọng trong dữ liệu cũ (`dataset-v1`)
1. **Lẫn lộn Synthetic vào tập Test/Val:** Trong repo cũ, dữ liệu được chia mù thành `train` (1.004 ảnh, 918 syn), `val` (223 ảnh, 210 syn), `test` (192 ảnh, 177 syn). Tập test có đến 92,2% là ảnh nhân tạo ghép hình. Nếu giữ nguyên, detector sẽ học vẹt các đường viền ghép nhân tạo mà không phát hiện được rác ngoài đời thực.
2. **Khuyết thiếu toàn bộ 9 lớp rác thực tế:** 114 ảnh thực tế duy nhất tải về từ OpenImages chỉ chứa nhãn `shoes` (507 boxes). Không có bất kỳ một bức ảnh thực tế nào về pin, chai nhựa, lon kim loại, thức ăn thừa hay giấy vụn trong môi trường đường phố Việt Nam.
3. **Kế hoạch thu thập thực địa TP.HCM chưa được triển khai:** Báo cáo cũ từng đề cập mục tiêu chụp 300-400 ảnh thực tế tại TP.HCM, nhưng trên thực tế chưa có bất kỳ ảnh nào được nạp vào hệ thống.

---

## 5. CÔNG CỤ RÀ SOÁT NHÃN VÀ HỆ THỐNG KIỂM THỬ TỰ ĐỘNG

### 5.1. Công cụ rà soát nhãn in-repo (`src/ui/review_tool.py`)
- **Kiến trúc:** Ứng dụng Streamlit độc lập, chạy trực tiếp trên repo dự án.
- **Tính năng hoàn chỉnh:**
  - Bộ lọc linh hoạt: Lọc theo nguồn gốc (Real / Synthetic), lọc theo trạng thái duyệt (`UNREVIEWED`, `APPROVED`, `REJECTED`, `NEEDS_RELABEL`).
  - Trực quan hóa chuẩn xác: Vẽ bounding box kèm màu sắc đặc trưng cho từng lớp và nhãn định danh (`#idx: class_name`).
  - Thao tác CRUD đầy đủ: Thêm box mới, sửa tọa độ/lớp, xóa box thừa.
  - Kiểm tra tính hợp lệ: Tự động khóa các tọa độ ngoài phạm vi $[0, 1]$ hoặc lớp ngoài phạm vi 0–9.
  - Lưu trữ và kiểm toán: Ghi trực tiếp vào file nhãn YOLO `.txt`, cập nhật manifest và ghi nhật ký kiểm toán vào `data/detection/review_audit_log.csv`.
- **Kiểm thử tự động trên trình duyệt thật (Playwright / Chrome Headless):**
  - Script kiểm thử: `scripts/verify_review_tool_ui.py`.
  - Kết quả: Tải trang thành công, đọc chính xác 1.419 ảnh, thực hiện thao tác duyệt `APPROVED` và lưu manifest thành công.
  - Minh chứng hình ảnh: `artifacts/part02/ui_evidence/review_tool_verified.png` (đã được xác minh trực quan).

### 5.2. Công cụ kiểm toán cú pháp Bounding Box (`scripts/validate_detection_annotations.py`)
- Kiểm tra toàn bộ 1.419 file nhãn (4.602 boxes):
  - Định dạng: Đủ 5 trường số `<class_id> <xc> <yc> <w> <h>`.
  - Tọa độ: $0 \le x_c, y_c \le 1$; $0 < w, h \le 1$.
  - Lớp: $0 \le \text{class\_id} \le 9$.
  - Trùng lặp: Cảnh báo các box trùng lớp có IoU > 0.98.
- Kết quả chạy trên bộ dữ liệu dự án: **1.419 file, 4.602 boxes, 0 lỗi, 0 cảnh báo suy biến -> PASS**.
- Báo cáo JSON: `artifacts/part02/detection_annotation_validation.json`.

### 5.3. Bộ kiểm thử tự động toàn diện (`tests/`)
Chạy bộ test hoàn chỉnh gồm 11 kịch bản kiểm thử:
```
tests/test_verification_gates.py::test_missing_directory_fails PASSED           [  9%]
tests/test_verification_gates.py::test_unresolved_entry_in_decision_table_fails PASSED [ 18%]
tests/test_verification_gates.py::test_real_audit_passes PASSED                 [ 27%]
tests/test_verification_gates.py::test_reproduce_validation_direct_disk_hash PASSED [ 36%]
tests/test_detection_annotations.py::test_calculate_iou PASSED                  [ 45%]
tests/test_detection_annotations.py::test_valid_annotation_file PASSED          [ 54%]
tests/test_detection_annotations.py::test_invalid_class_id PASSED               [ 63%]
tests/test_detection_annotations.py::test_out_of_range_coordinates PASSED        [ 72%]
tests/test_detection_annotations.py::test_malformed_field_count PASSED          [ 81%]
tests/test_detection_annotations.py::test_duplicate_box_warning PASSED          [ 90%]
tests/test_detection_annotations.py::test_real_dataset_annotations PASSED        [100%]
======================== 11 passed in 79.40s ========================
```

---

## 6. KẾT LUẬN VÀ KIẾN NGHỊ CHUYỂN GIAO SANG PHẦN 3

### 6.1. Đánh giá tổng thể Task 2
1. **Phần đánh giá Classifier (Gate A):**
   - Đạt độ chính xác rất cao: Top-1 Accuracy **96,13%**, Macro-F1 **0,9578**, Battery Recall **95,58%**. Mô hình phân loại đơn rác MobileNetV3-Large đã chứng minh năng lực khái quát hóa vững chắc trên tập ảnh thực tế chưa từng gặp.
   - Về độ trễ CPU (23,86 ms vs < 20 ms): Đây là kết quả trung thực trên CPU phần cứng thử nghiệm. Cần đưa vào kế hoạch tối ưu hóa xuất khẩu mô hình (ONNX / OpenVINO / INT8).
2. **Phần chuẩn bị dữ liệu Bounding Box Đa rác:**
   - Đã hoàn thành 100% hạ tầng kiểm toán, quy chuẩn hóa thư mục, phân lập synthetic/real, xây dựng validator tự động và review tool có giao diện đồ họa.
   - **Tuy nhiên, chất lượng và độ bao phủ của tập dữ liệu thực tế hiện tại là HOÀN TOÀN CHƯA ĐẠT CHUẨN để huấn luyện YOLOv8n / SSDLite320.**

### 6.2. Khuyến nghị gửi PM Ngô Thanh Nhân
- **KHÔNG NÊN BẮT ĐẦU HUẤN LUYỆN DETECTOR Ở PHẦN 3 NGAY LẬP TỨC.**
- Nếu huấn luyện ngay với 1.305 ảnh synthetic và 114 ảnh chỉ có giày, detector sẽ bị overfitting nặng nề vào ảnh đồ họa và không thể phát hiện được bất kỳ loại rác thực tế nào khác trong 9 lớp còn lại.
- **Hành động đề xuất trước khi train Part 3:**
  1. Sử dụng công cụ `src/ui/review_tool.py` vừa phát triển để gắn nhãn bổ sung tối thiểu **200 – 300 bức ảnh chụp thực tế đa rác** (chụp thùng rác công cộng, bãi rác sinh hoạt, đường phố tại TP.HCM) bao phủ đủ 10 lớp vật liệu.
  2. Dùng 1.305 ảnh Synthetic Mendeley cho bước **Pretraining** (huấn luyện khởi tạo), sau đó dùng tập ảnh thực tế TP.HCM để **Fine-tuning** và làm **Tập Test Độc lập** cho giai đoạn B.
  3. Áp dụng kỹ thuật lượng hóa INT8 cho MobileNetV3-Large để hạ độ trễ CPU xuống dưới 12 ms, chính thức vượt qua tiêu chuẩn độ trễ của Gate A trong kịch bản triển khai biên (Edge AI).
