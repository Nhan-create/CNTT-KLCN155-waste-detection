# BÁO CÁO TOÀN DIỆN TIẾP QUẢN PHASE 3: KIỂM TOÁN TẬP DỮ LIỆU, KHẮC PHỤC MÔI TRƯỜNG & ĐÁNH GIÁ TÍNH SẴN SÀNG HUẤN LUYỆN
## (ANTIGRAVITY CONTINUATION REPORT: PHASE 3 DATASET INTEGRITY, ENVIRONMENT REPAIR & TRAINING READINESS)

**Dự án:** Nhận diện Rác thải Đa lớp Thời gian thực (CNTT-KLCN155)  
**Tác giả:** Antigravity Principal Computer Vision & Independent QA Auditor  
**Thời gian lập:** 11/10/2026  
**Trạng thái bàn giao:** Tiếp nhận từ Codex, hoàn thành khắc phục môi trường, truy xuất nguồn gốc dữ liệu, kiểm toán toàn vẹn và lập hồ sơ thẩm định.

---

## 1. TÌNH TRẠNG TIẾP NHẬN TỪ CODEX (STATUS OF HANDOVER FROM CODEX)

Codex đã thực hiện kiểm toán ban đầu và phát hiện các điểm sai lệch giữa các bản báo cáo Markdown trước đó:
- **Tập dữ liệu Holdout độc lập:** Thư mục `data/holdout` gồm **27 ảnh** (trong đó có 2 ảnh trùng lặp byte hoàn toàn: `images (7).jpg` và `images (13).jpg`, SHA-256: `9ee5ca3468bead7c48529f79024f21db5976b97e937d37ef67223e75e110b9db`).
- **Ground Truth v1 (Frozen Baseline):** Tệp `reports/qa_acceptance/ground_truth_hinh_anh_rac.json` gồm **57 bounding boxes**, mã băm SHA-256 bất biến:
  `f390f0f9440a977d0a0afe9e55b862562f1aafd3011ab18313e1772b18d470c0`.
- **Gói bàn giao từ Codex:** Tệp nén `~/Downloads\CNTT-KLCN155_CODEX_AUDIT_20261011_225845.zip` đã được đối soát thực tế tại thư mục Downloads của hệ thống.
- **Vấn đề Codex để lại:**
  1. Chưa giải thích triệt để nguyên nhân cảnh báo import module ngoại lai `waste-classifier-mobilenetv3`.
  2. Bị vướng 10 lỗi pytest collection errors khi chạy qua công cụ gọi test tự động.
  3. Chưa chứng minh toán học nguồn gốc tọa độ âm TACO ($y = -1.00$) trên dữ liệu gốc COCO.
  4. Chưa giải quyết dứt điểm 4 ảnh Open Images thiếu nguồn và sự chênh lệch số lượng box.
  5. GT v2 mới chỉ ở dạng sơ thảo, còn 6 đối tượng tranh chấp và 10 ảnh không có active object chưa được phân loại rõ ràng.

---

## 2. CÁC BẢN SỬA CỦA CODEX ĐÃ ĐỐI SOÁT VÀ XÁC MINH TRONG REPOSITORY

Antigravity đã kiểm tra trực tiếp mã nguồn và chạy kiểm thử để xác minh các bản sửa của Codex:
1. **Bộ đánh giá độc lập (Evaluator):**
   - Tệp `scripts/compare_eval_v1_vs_v2.py`: Đã chuẩn hóa quy tắc tính toán IoU, xử lý class-aware matching chặt chẽ (dự đoán đúng vị trí nhưng sai nhãn bị tính 1 FP cho nhãn dự đoán và 1 FN cho nhãn ground truth).
   - Kiểm thử tự động `tests/qa/test_compare_eval_v1_vs_v2.py`: Chạy thực tế bằng `.venv\Scripts\python.exe -m pytest` đạt **2/2 PASSED (100%)**.
2. **Khớp đúng số liệu Baseline:**
   - Số liệu đánh giá trước và sau Quick Fix trên tập Holdout đã được tái lập nhất quán:
     - **Before Quick Fix:** 24 predictions $\rightarrow$ TP = 13, FP = 8, Unmatched GT = 44, Recall = 22.81%, Precision = 61.90%, F1 = 33.33%.
     - **After Quick Fix:** 19 predictions $\rightarrow$ TP = 8, FP = 3, Unmatched GT = 49, Recall = 14.04%, Precision = 72.73%, F1 = 23.53%.
   - Hiện tượng sụt giảm Recall và F1 được xác nhận là do việc tăng ngưỡng lọc (filtering) giúp giảm FP ở cảnh nền nhưng làm triệt tiêu các phát hiện rác thật yếu.

---

## 3. NGUYÊN NHÂN GỐC RỄ VẤN ĐỀ IMPORT MÔI TRƯỜNG PYTHON (ROOT CAUSE OF IMPORT ALARMS)

Trong các log lỗi trước đây xuất hiện đường dẫn lạ:
`~/Downloads\Do-an-deeplearning\waste-classifier-mobilenetv3\...`
gây lo ngại rằng môi trường `.venv` đang import nhầm mã nguồn từ một dự án khác.

### Kết quả điều tra thực nghiệm:
1. **Kiểm tra `sys.path` và module `src`:**
   Chạy lệnh kiểm toán trực tiếp:
   ```python
   import src
   print(src.__file__)  # src\__init__.py
   ```
   Hoàn toàn không có bất kỳ thư mục ngoại lai nào nằm trong `sys.path`.
2. **Nguyên nhân cốt lõi – 149 tệp bytecode `.pyc` lỗi thời (Stale Bytecode):**
   - Khi dự án được copy hoặc clone từ thư mục gốc ban đầu, toàn bộ các tệp `.pyc` trong thư mục `__pycache__` đã bị sao chép theo.
   - Trong kiến trúc Python, mỗi tệp bytecode `.pyc` chứa cấu trúc `code object` lưu cứng chuỗi `co_filename` tại thời điểm biên dịch (`compile-time absolute path`).
   - Khi xảy ra ngoại lệ, Python traceback in chuỗi `co_filename` được nhúng sẵn trong `.pyc`, dẫn đến việc hiển thị đường dẫn thư mục Downloads cũ mặc dù mã nguồn Python `.py` đang chạy thực tế 100% nằm tại `.`.
3. **Biện pháp xử lý triệt để:**
   - Đã phát triển và thực thi script `scripts/recompile_clean_bytecode.py`.
   - Quét và xóa sạch toàn bộ 149 tệp `.pyc` cũ trong tất cả các thư mục `__pycache__`.
   - Biên dịch lại toàn bộ mã nguồn tại chỗ bằng `compileall.compile_dir(force=True)`.
   - Kết quả: 100% tệp `.pyc` mới đều trỏ chính xác về `.`.
   - Xây dựng bộ test hồi quy `tests/test_environment_integrity.py` gồm 4 bài kiểm tra tự động xác nhận không còn bất kỳ tham chiếu bytecode ngoại lai nào.

---

## 4. CHẨN ĐOÁN VÀ GIẢI QUYẾT LỖI PYTEST COLLECTION (PYTEST COLLECTION ERRORS)

### Hiện tượng:
Báo cáo trước đó ghi nhận có 10 lỗi Collection Errors khi chạy pytest.

### Chẩn đoán kỹ thuật:
1. Môi trường ảo `.venv` (Python 3.12.10) đã có đầy đủ 100% thư viện phụ thuộc:
   - `imagehash 4.3.2`
   - `seaborn 0.13.2`
   - `PyQt5 5.15.11`
   - `nbformat 5.10.4`
   - `kagglehub 1.0.2`
2. Tuy nhiên, khi gọi trực tiếp tệp nhị phân script `.venv\Scripts\pytest.exe` từ một số môi trường shell Windows, dòng shebang đầu tệp (`#!`) có thể điều hướng sang trình thông dịch Python mặc định của hệ thống (`Python 3.12.9`), vốn không được cài đặt các thư viện trên.
3. Khi thực thi qua lệnh chuẩn của môi trường ảo:
   ```powershell
   .venv\Scripts\python.exe -m pytest --collect-only
   ```
   Kết quả: **Thu thập thành công 297/297 test items, 0 lỗi collection error**.

---

## 5. CÁC LỖI DATASET PHÁT HIỆN ĐƯỢC (DATASET ERRORS & PROVENANCE)

### A. Chứng minh toán học lỗi tọa độ âm TACO ($y = -1.00$)
- **Nghi vấn ban đầu:** Tệp `taco_0112.txt` và `taco_0457.txt` bị nghi ngờ lỗi parser khi chuyển đổi từ COCO sang YOLO.
- **Thực chứng:** Truy xuất trực tiếp tệp COCO nguyên bản của nhóm tác giả TACO `data/audit/taco_annotations_raw.json`:
  1. `taco_0112` (Image ID 112, Kích thước $1824 \times 1824$, Annotation ID 387):
     Tọa độ COCO gốc: `bbox: [1225.0, -1.0, 107.0, 92.0]` $\rightarrow y_1 = -1.0$.
     Tọa độ chuẩn hóa: $y_{norm} = -1.0 / 1824 = -0.000548 < 0.0$.
  2. `taco_0457` (Image ID 457, Kích thước $4160 \times 4160$, Annotation ID 1621):
     Tọa độ COCO gốc: `bbox: [1167.0, -1.0, 246.0, 269.0]` $\rightarrow y_1 = -1.0$.
     Tọa độ chuẩn hóa: $y_{norm} = -1.0 / 4160 = -0.000240 < 0.0$.
- **Kết luận:** Đây là lỗi làm tròn 1 pixel chạm mép trên của chính tập dữ liệu TACO công bố quốc tế. Bộ nhãn thực tế của dự án tại `data/detection/labels/real/` đã có hàm kẹp biên `max(0.0, y)` đưa về $y=0.0$, hoàn toàn hợp lệ và không làm mất dữ liệu.

### B. Kiểm toán nguồn gốc Open Images (Provenance & Box Counts)
- Kiểm tra tệp manifest `data/detection/real_v2_source_manifest.csv` (2,027 dòng) đối chiếu với danh sách 1,996 ảnh trong `splits_real_v2`:
  - Phát hiện **4 ảnh** Open Images có trong split nhưng chưa được ghi trong `real_v2_source_manifest.csv`:
    `oi_19c5c4dae0da0f77`, `oi_2e62e7ce1ce182b5`, `oi_48fe726552b07707`, `oi_57f0572aa1a3cba5` (đều thuộc lớp `shoes`).
  - Giải thích **30 dòng Open Images có `num_boxes = 0`**: Khi thu thập từ Open Images V7, bộ lọc chỉ trích xuất các box thuộc danh mục `clothing` và `footwear` của người mẫu thời trang, bỏ qua các box gán nhãn toàn thân `Person`/`Human body`. Do đó các ảnh này không sinh box rác trong bộ nhãn.

### C. Phát hiện Box trùng lặp tuyệt đối (Duplicate Box Anomaly)
- Qua việc chạy script quét toàn diện `scripts/audit_dataset_integrity_full.py` trên toàn bộ **1,996 ảnh** và **5,585 bounding boxes**:
  - Không có ảnh hỏng.
  - Không có box âm hoặc tràn biên.
  - Không có nhãn sai dải $[0, 9]$.
  - **Phát hiện duy nhất 1 lỗi box trùng lặp:** Tệp `oi_2734e597675d8a97.txt` chứa 2 dòng giống hệt nhau:
    `3 0.500000 0.500000 1.000000 1.000000`
    `3 0.500000 0.500000 1.000000 1.000000`

---

## 6. BẢN VÁ ĐÃ ÁP DỤNG KÈM REGRESSION TESTS

1. **Khắc phục Duplicate Box `oi_2734e597675d8a97`:**
   - Đã loại bỏ dòng trùng lặp thứ 2 trong:
     - `data/detection/labels/real/oi_2734e597675d8a97.txt`
     - `data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt`
     - `data/detection_agnostic/labels/real/oi_2734e597675d8a97.txt`
   - Cập nhật số lượng box (`num_boxes`: $2 \rightarrow 1$) và mã băm SHA-256 mới (`ce34d49fd17aa36d23fa185ffc7c5b7db806f9f91b79d59bc6c580499c0d5964`) trong cả `real_v2_source_manifest.csv` và `real_v2_split_manifest.csv`.
2. **Kiểm tra hồi quy toàn vẹn dữ liệu:**
   - Chạy lại `scripts/audit_dataset_integrity_full.py`: Tổng số box còn lại là **5,584 boxes**, số box trùng lặp giảm về **0**.
   - `tests/test_detection_annotations.py`: **7/7 PASSED**.
   - `tests/detection/test_split_real_v2.py`: **14/14 PASSED**.

---

## 7. CHÚ THÍCH ĐANG CHỜ PHÊ DUYỆT THỦ CÔNG TỪ STAKEHOLDER (GT V2 REVIEW)

Báo cáo chuyên sâu độc lập đã được xuất tại:
[`reports/GT_V2_REVIEW_FOR_APPROVAL.md`](../reports/GT_V2_REVIEW_FOR_APPROVAL.md)

### Tóm tắt cơ cấu 57 đối tượng Holdout trong đề xuất v2:
- **39 đối tượng ACCEPTED (68.42%):** Các vật thể rác đơn lẻ, rõ ràng ranh giới (chai, lon, hộp, giấy báo).
- **4 đối tượng QUARANTINED (7.02%):**
  - Thùng rác công cộng 2 bánh (`images (18).jpg`) và sọt rác văn phòng (`images (8).jpg`): Không phải rác cần dọn dẹp, thuộc hạ tầng tiếp nhận.
  - Box nền chiếm 100% canvas trong `download (2).jpg` và `base64...`: Box dư thừa bọc ngoài các box con.
- **8 đối tượng FLAGGED_UNSEGREGATED_PILE (14.04%):** Đống rác/bãi rác cận cảnh chiếm từ 85% đến 100% khung hình. **Đề xuất tách riêng sang bài toán Pile Detection/Segmentation**, không coi là ảnh nền sạch (Negative) để tránh phạt oan mô hình.
- **6 đối tượng Active đang chờ Chủ dự án quyết định:**
  1. `images (12).jpg` (`img06_obj01`): Thùng carton chứa hồ sơ giấy $\rightarrow$ Đề xuất giữ quan hệ Cha-Con (`cardboard` chứa `paper`).
  2. `images (10).jpg` (`img04_obj01`): Hai box quần áo lồng nhau $\rightarrow$ Đề xuất gộp thành 1 box `clothes` duy nhất.
  3. `OIP (1).webp` (`img20_obj01`): Gò quần áo sa mạc chồng lấn $\rightarrow$ Đề xuất cắt tỉa ranh giới rõ ràng.
  4. `Phan-Loai-Rac-Tai-Ng-01.jpg` (`img22_obj04`): Hộp sữa Tetra Pak (vừa giấy vừa nhôm vừa nhựa) $\rightarrow$ Đề xuất thống nhất nhãn `cardboard`.
  5. & 6. `Phan-Loai-Rac-Tai-Ng-01.jpg` (`img22_obj01`, `img22_obj02`): Bao tải xanh và túi rác đỏ chứa rác sinh hoạt hỗn hợp $\rightarrow$ Đề xuất gán `trash` hay giữ `plastic`.
- **10 ảnh có 0 active object đơn lẻ:** Đã phân định rõ giữa ảnh hạ tầng (2 ảnh) và ảnh đống rác cận cảnh (8 ảnh).

---

## 8. CÁC THAY ĐỔI MÃ NGUỒN ĐÃ THỰC HIỆN (SOURCE CODE & ARTIFACTS)

### Tệp tạo mới:
1. `scripts/recompile_clean_bytecode.py`: Công cụ dọn dẹp bytecode `.pyc` lỗi thời.
2. `tests/test_environment_integrity.py`: 4 bài test bảo vệ môi trường Python và tính cục bộ của workspace.
3. `scripts/audit_taco_coco_exact.py`: Script chứng minh nguồn gốc tọa độ âm TACO.
4. `scripts/audit_exact_openimages_provenance.py`: Script đối soát nguồn gốc 2,027 mẫu Open Images.
5. `scripts/audit_dataset_integrity_full.py`: Bộ kiểm toán tự động toàn diện 1,996 ảnh dataset.
6. `reports/DATASET_INTEGRITY_AUDIT_REPORT.json`: Báo cáo JSON chi tiết kết quả quét 5,584 boxes.
7. `reports/GT_V2_REVIEW_FOR_APPROVAL.md`: Bản trình duyệt nghiệm thu Ground Truth v2.
8. `reports/ANTI_PHASE3_CONTINUATION.md`: Báo cáo tổng kết toàn diện Phase 3.

### Tệp đã hiệu chỉnh (Surgical Patches):
1. `data/detection/labels/real/oi_2734e597675d8a97.txt`: Loại bỏ box trùng lặp.
2. `data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt`: Loại bỏ box trùng lặp.
3. `data/detection_agnostic/labels/real/oi_2734e597675d8a97.txt`: Loại bỏ box trùng lặp.
4. `data/detection/real_v2_source_manifest.csv`: Cập nhật `num_boxes` $2 \rightarrow 1$.
5. `data/detection/real_v2_split_manifest.csv`: Cập nhật `num_boxes` và mã SHA-256 mới.

*Bảo toàn 100% các thay đổi uncommitted trước đó của người dùng; không có thao tác xóa dữ liệu thô hay git reset.*

---

## 9. TÓM TẮT KẾT QUẢ KIỂM THỬ PYTEST (TEST RESULTS SUMMARY)

Tổng số bài test được thu thập trong toàn dự án: **297 tests**.
Trong phiên làm việc của Antigravity, **120 bài test trọng yếu** đã được thực thi thực tế bằng `.venv\Scripts\python.exe -m pytest`:

| Phân hệ kiểm thử | Tệp test | Kết quả | Thời gian |
|---|---|---|---|
| **Môi trường & Bytecode** | `tests/test_environment_integrity.py` | **4/4 PASSED** | 0.82s |
| **Tính toàn vẹn Bounding Box** | `tests/test_detection_annotations.py` | **7/7 PASSED** | 0.97s |
| **Đánh giá Baseline Evaluator** | `tests/qa/test_compare_eval_v1_vs_v2.py` | **2/2 PASSED** | 1.15s |
| **Toán học tọa độ & Geometry** | `tests/detection/test_coordinates.py` | **8/8 PASSED** | 0.85s |
| **Cắt lát Tiling & Tái hợp Bbox** | `tests/detection/test_tiling.py` | **19/19 PASSED** | 1.84s |
| **Suy luận Two-Stage Refinement** | `tests/detection/test_two_stage.py` | **17/17 PASSED** | 11.2s |
| **Hồi quy Mapping 10 Lớp** | `tests/detection/test_mapping_regression.py` | **5/5 PASSED** | 0.76s |
| **Tích hợp Nguồn dữ liệu thật** | `tests/detection/test_real_mapping_integration.py` | **14/14 PASSED** | 2.10s |
| **Toàn vẹn Split Dataset v2** | `tests/detection/test_split_real_v2.py` | **14/14 PASSED** | 8.74s |
| **Pipeline Streamlit Web Backend** | `tests/web/test_detection_pipeline.py` | **15/15 PASSED** | 18.4s |
| **Đầu ra ảnh & Render màu** | `tests/web/test_detection_image_outputs.py` | **4/4 PASSED** | 4.2s |
| **Streamlit UI Contract** | `tests/web/test_streamlit_contract.py` | **2/2 PASSED** | 3.1s |
| **Hiển thị Bounding Box** | `tests/detection/test_visualization.py` | **5/5 PASSED** | 1.9s |
| **Cổng kiểm soát chất lượng** | `tests/test_verification_gates.py` | **3 PASSED, 1 SKIPPED** | 29.3s |
| **TỔNG CỘNG** | **14 Test Suites** | **119 PASSED, 1 SKIPPED, 0 FAILED** | **~84s** |

---

## 10. TÌNH TRẠNG DATASET VÀ PHÂN BỐ CLASS TRONG REAL_V2

Tập dữ liệu hiện hữu `splits_real_v2` gồm **1,996 ảnh** (Train: 1,398, Val: 299, Test: 299).  
Thống kê phân bố 10 lớp trong tập Train (**1,398 ảnh, 3,963 boxes**):

| Class ID | Tên lớp | Số lượng Box | Tỷ lệ (%) | Đánh giá mức độ |
|---|---|---|---|---|
| **0** | `battery` | **1** | **0.03%** | **Cực kỳ thiếu thốn (Starvation)** |
| **1** | `biological` | **3** | **0.08%** | **Cực kỳ thiếu thốn (Starvation)** |
| **2** | `cardboard` | 189 | 4.77% | Tương đối thấp |
| **3** | `clothes` | 645 | 16.28% | Đủ mẫu (nhưng 101 ảnh là ảnh người mẫu) |
| **4** | `glass` | 272 | 6.86% | Trung bình |
| **5** | `metal` | 355 | 8.96% | Trung bình |
| **6** | `paper` | 326 | 8.23% | Trung bình |
| **7** | `plastic` | **1,972** | **49.77%** | **Áp đảo 50% tập dữ liệu** |
| **8** | `shoes` | 144 | 3.63% | Tương đối thấp |
| **9** | `trash` | 56 | 1.41% | Thấp |
| **Tổng** | **10 lớp** | **3,963** | **100.0%** | Mất cân bằng nghiêm trọng |

### Điểm nghẽn đặc biệt nguy hiểm:
- **Số lượng ảnh cảnh nền sạch (Background/Negative Images) = 0 ảnh (0.0%).**  
  Tất cả 1,398 ảnh trong tập Train đều có ít nhất 1 nhãn rác. Không có bất kỳ ảnh nào chỉ chứa nền gạch, vỉa hè, mặt đường, tường nhà hay tủ điện mà không có rác.
- Đây chính là **nguyên nhân cốt lõi khiến mô hình SSDLite và YOLO liên tục bị False Positive (nhận diện nhầm biển hiệu, tường nhà, tủ điện thành bìa carton hoặc rác)** trên ứng dụng thực tế.

---

## 11. PHÁN QUYẾT TÍNH SẴN SÀNG HUẤN LUYỆN (TRAINING READINESS VERDICT)

### **KẾT LUẬN: BỊ CHẶN (BLOCKED)**

Tuyệt đối **CHƯA ĐƯỢC TIẾN HÀNH HUẤN LUYỆN** mô hình mới tại thời điểm này vì 4 lý do kỹ thuật sau:
1. **Thiếu hoàn toàn Hard Negatives (0% ảnh nền):** Nếu huấn luyện ngay, mô hình mới sẽ tiếp tục bị ảo giác (hallucination), bắt nhầm các chi tiết đô thị thành rác. Cần tối thiểu 150 - 200 ảnh nền sạch không có rác đưa vào tập Train.
2. **Mất cân bằng dữ liệu cực đoan:** Lớp `battery` chỉ có 1 box, `biological` chỉ có 3 box trong tập Train. Một mạng nơ-ron học sâu không thể hội tụ đặc trưng cho một lớp chỉ với 1 đến 3 mẫu.
3. **Nhiễu chú thích từ Open Images:** 101/114 ảnh Open Images là chân dung thời trang có người mặc quần áo nhưng người mẫu không được gán nhãn, khiến mô hình học tín hiệu mâu thuẫn (vừa coi quần áo là rác, vừa coi người mặc quần áo là nền).
4. **Ground Truth Benchmark chưa được đóng băng chính thức:** Tập GT v2 trên Holdout vẫn đang ở trạng thái Proposed chờ Chủ dự án phê duyệt 6 đối tượng tranh chấp. Chưa thể huấn luyện mô hình khi thước đo nghiệm thu chưa được chuẩn hóa.

---

## 12. CÁC BƯỚC TIẾP THEO VÀ QUYẾT ĐỊNH CẦN NGƯỜI DÙNG PHÊ DUYỆT

Để mở khóa giai đoạn huấn luyện (Phase 4), kính đề nghị Chủ dự án xem xét và phê duyệt:

### A. Quyết định nghiệm thu Ground Truth v2:
1. **Phê duyệt phân loại:** Chấp thuận đưa 39 vật thể ACCEPTED vào GT v2 chính thức; chuyển 4 vật thể hạ tầng/box nền vào QUARANTINED; tách 8 ảnh đống rác cận cảnh thành tập đánh giá riêng (Pile/Segmentation).
2. **Biểu quyết 6 đối tượng tranh chấp:**
   - `images (12).jpg`: Chấp thuận quy ước lồng nhau `cardboard` chứa `paper` (Có/Không).
   - `images (10).jpg`: Chấp thuận gộp 2 box quần áo lồng nhau thành 1 box duy nhất (Có/Không).
   - `OIP (1).webp`: Chấp thuận chia đôi ranh giới 2 gò quần áo sa mạc (Có/Không).
   - `Phan-Loai-Rac-Tai-Ng-01.jpg` obj 04: Thống nhất gán vỏ hộp sữa Tetra Pak là `cardboard` (Có/Không).
   - `Phan-Loai-Rac-Tai-Ng-01.jpg` obj 01 & 02: Thống nhất gán bao tải/túi rác chứa rác hỗn hợp là `trash` hay `plastic`.

### B. Kế hoạch chuẩn bị tập dữ liệu `splits_real_v3`:
1. **Bổ sung 150 - 200 ảnh Negative (Hard Negatives):** Thu thập ảnh vỉa hè, lòng đường, tủ kỹ thuật, góc phòng sạch không có rác (tạo tệp nhãn rỗng kích thước 0 byte).
2. **Bổ sung mẫu thực tế cho `battery` và `biological`:** Tích hợp các mẫu pin và rác thực phẩm từ tập Mendeley đã được làm sạch và verify thủ công để nâng số lượng box lên tối thiểu 80 - 100 boxes mỗi lớp.
3. **Loại bỏ/cách ly các ảnh Open Images người mẫu thời trang:** Thay thế bằng ảnh quần áo, giày dép bị vứt bỏ thực tế ngoài môi trường.

Sau khi nhận được phê duyệt các mục trên, Antigravity sẽ tiến hành đóng băng GT v2, sinh tập `splits_real_v3` và chuyển sang **Phase 4: Huấn luyện chính thức SSDLite320-MobileNetV3 và YOLOv8n đối chứng**.
