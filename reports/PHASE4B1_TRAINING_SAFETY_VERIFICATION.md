# BÁO CÁO NGHIỆM THU AN TOÀN DỮ LIỆU & HUẤN LUYỆN
## PHASE 4B.1: DATASET SAFETY & TRAINING VALIDATION
**Dự án:** CNTT-KLCN155 — Hệ thống Nhận diện Rác Đa Lớp (Waste Detection)  
**Thời điểm thực thi:** 2026-10-12  
**Môi trường phần cứng:** NVIDIA GeForce RTX 2050 (4.0 GB VRAM), 15.31 GB System RAM  
**Môi trường phần mềm:** Python 3.12.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.171  

---

## 1. KẾT LUẬN TỔNG THỂ & PHÁN QUYẾT NGHIỆM THU (VERDICT)

| Hạng mục kiểm tra | Phán quyết | Ràng buộc & Hành động kế tiếp |
|---|:---:|---|
| **Chất lượng & Cách ly Dataset v3 Candidate** | **VERIFIED** | 1.820 ảnh sạch (1.277 train, 269 val, 274 test). Độc lập Inode/MFT 100%, 0 hardlink trên label, 0 rò rỉ SHA-256/pHash qua splits và holdout. |
| **Kiểm tra Huấn luyện Thật (Dry-Run SSDLite & YOLO)** | **VERIFIED** | 1-step forward/loss/backward SSDLite trên GPU RTX 2050 với batch hỗn hợp (ảnh có rác + ảnh rỗng 0-object) hoàn thành trong 2.52s. YOLO DataLoader nạp 10 batches train/val sạch 100% không lỗi. |
| **An toàn Phần cứng Huấn luyện Cục bộ** | **BLOCKED** | **Hệ thống RAM đang chiếm dụng 97.1% (chỉ còn ~440 MB khả dụng)**. Khóa tuyệt đối lệnh train nhiều epoch (`multi-epoch`) trên máy local để bảo vệ hệ điều hành khỏi thrashing/crash. Huấn luyện chính thức bắt buộc thực hiện trên Google Colab GPU hoặc giải phóng RAM trước. |
| **Kế hoạch Mở rộng Dữ liệu Ngoài (Battery, Biological, Hard-Negatives)** | **NEEDS APPROVAL** | Đã lập danh mục chi tiết 9 nguồn ứng viên chuẩn license (CC BY 4.0, CC0). **Tuân thủ nghiêm ngặt chính sách 0 download trái phép khi chưa có phê duyệt.** |

---

## 2. P0: BẢO VỆ DATASET V2 & SỬA LỖI HARDLINK TRÊN FILE NHÃN

### 2.1. Phân tích nguyên nhân gốc (Root Cause)
Trên hệ điều hành Windows NTFS, hàm `os.link(src, dst)` tạo một hardlink trỏ trực tiếp vào cùng một entry MFT (Master File Table) và Inode với file gốc.
Khi áp dụng hardlink lên file nhãn text (`.txt`):
- Bất kỳ thao tác chỉnh sửa, chuẩn hóa tọa độ hoặc lọc box nào trên dataset v3 sẽ **trực tiếp làm thay đổi nhãn gốc của dataset v2**.
- Điều này vi phạm nghiêm trọng nguyên tắc **Bất biến của dữ liệu gốc (Immutability)**.

### 2.2. Giải pháp khắc phục & Bằng chứng Inode độc lập
File xử lý: [scripts/build_dataset_v3_candidate.py](../scripts/build_dataset_v3_candidate.py)
1. **File nhãn (`.txt`):** Thay thế toàn bộ bằng `shutil.copy2(src, dst)`. Mỗi file nhãn v3 được cấp phát MFT/Inode độc lập hoàn toàn.
2. **File ảnh (`.jpg`, `.png`):** Áp dụng chính sách **Read-Only / Immutable**. Ảnh dùng hardlink để tiết kiệm dung lượng đĩa nhưng bị khóa thuộc tính chỉ đọc (`stat.S_IREAD`), ngăn chặn mọi hành vi ghi đè.
3. **Làm sạch cache tái tạo:** Khi build lại, script tự động quét và xóa toàn bộ thư mục đích cũ cùng các file cache `.cache` của YOLO để ngăn chặn rác dữ liệu tồn đọng.

**Kiểm tra Inode thực tế trên NTFS:**
- `v2 label inode`: `562949954313679`
- `v3 label inode`: `281474977729218` (Khác nhau hoàn toàn, độc lập 100%).
- Thử nghiệm sửa đổi nội dung file nhãn v3: File nhãn gốc v2 không bị biến đổi bất kỳ byte nào.

**Kiểm thử tự động:** Bộ test [tests/qa/test_dataset_v3_isolation.py](../tests/qa/test_dataset_v3_isolation.py) đã chạy và đạt `3/3 PASSED`.

---

## 3. P0: THỰC THI TRAINING DRY-RUN THẬT & ĐO LƯỜNG TÀI NGUYÊN PHẦN CỨNG

### 3.1. Thiết kế bài test Training Dry-Run cách ly
File thực thi: [scripts/verify_training_dry_run_environment.py](../scripts/verify_training_dry_run_environment.py)  
Báo cáo chi tiết: [reports/model_v2/training_dry_run_report.json](../reports/model_v2/training_dry_run_report.json)

Khác với các lần chạy `model.eval()` giả lập trước đây, bài test này thực hiện quy trình huấn luyện thật 100%:
- Khởi tạo kiến trúc mô hình chuẩn: `ssdlite320_mobilenet_v3_large(num_classes=11)` (Background = 0, Foreground = 1..10).
- Chuyển mô hình sang chế độ huấn luyện: `model.train()`.
- Cấu hình Optimizer: `torch.optim.AdamW(lr=1e-4)`.
- Thiết bị thực thi: `cuda:0` (NVIDIA GeForce RTX 2050 Laptop GPU).

### 3.2. Kiểm thử Batch hỗn hợp (Positive + Hard-Negative)
Để kiểm tra khả năng xử lý của mô hình khi gặp ảnh không có rác (0-object / Hard-Negative), batch huấn luyện gồm:
1. **Mẫu Positive thật:** `oi_0010c714a5da358a.jpg` chứa 3 bounding box rác thực tế.
2. **Mẫu Hard-Negative (0-object):** Khung hình ảnh sạch với target `boxes: zeros((0, 4))` và `labels: zeros((0,), dtype=torch.int64)`.

**Kết quả thực thi 1 bước tối ưu (Single-Step Forward/Loss/Backward):**
- **Thời gian xử lý Forward:** `1.6580 giây`
- **Thời gian xử lý Backward:** `0.8318 giây`
- **Tổng thời gian 1 step:** `2.5253 giây`
- **Loss chi tiết:**
  - `bbox_regression`: `3.0311`
  - `classification`: `14.7162`
  - `total_loss`: `17.7473`
- **Trạng thái toán học:** Gradient tính toán đầy đủ trên toàn bộ tham số, **0 lỗi NaN, 0 lỗi Inf, 0 lỗi chia cho 0 khi batch chứa mẫu 0-object**.

### 3.3. Đo lường CUDA VRAM & Cảnh báo System RAM
- **CUDA VRAM (RTX 2050 4GB):**
  - VRAM Allocated ban đầu: `13.56 MB`
  - VRAM Allocated sau forward/backward: `46.85 MB`
  - **Peak VRAM sử dụng:** `217.62 MB` (Chiếm ~5.3% dung lượng GPU 4GB, cực kỳ an toàn về mặt bộ nhớ VRAM).
- **System Host RAM (CẢNH BÁO QUAN TRỌNG):**
  - Tổng RAM máy: `15.31 GB`
  - Đang sử dụng: `14.88 GB` (**97.1% utilization**)
  - Bộ nhớ RAM khả dụng: **chỉ còn 444.6 MB**!
- **Hành động bảo vệ:** Hệ thống tự động kích hoạt cờ `SYSTEM_RAM_CRITICAL` và **KHÓA TOÀN BỘ TIẾN TRÌNH HUẤN LUYỆN DÀI HẠN** trên máy cục bộ để ngăn chặn tràn RAM và crash hệ điều hành.

---

## 4. P1: ĐỐI SOÁT & GIẢI TRÌNH CÁC TẬP SPLITS V2 VS V3 CANDIDATE

Báo cáo phân tích: [reports/data_v2/v3_splits_reconciliation_audit.json](../reports/data_v2/v3_splits_reconciliation_audit.json)

### 4.1. Bảng đối soát số lượng ảnh theo Split

| Tập dữ liệu | Dataset v2 (Gốc) | Dataset v3 Candidate (Đã lọc) | Chênh lệch (Excluded) | Tỷ lệ loại bỏ |
|---|:---:|:---:|:---:|:---:|
| **Train** | 1.398 ảnh | 1.277 ảnh | -121 ảnh | -8.66% |
| **Validation** | 299 ảnh | 269 ảnh | -30 ảnh | -10.03% |
| **Test** | 299 ảnh | 274 ảnh | -25 ảnh | -8.36% |
| **TỔNG CỘNG** | **1.996 ảnh** | **1.820 ảnh** | **-176 ảnh** | **-8.82%** |

### 4.2. Giải trình chi tiết 176 ảnh bị loại khỏi v3 Candidate
Toàn bộ 176 ảnh này đều bị Quality Gate gắn cờ `FLAG_FOR_REVIEW` do vi phạm các tiêu chuẩn hình học và chất lượng chú thích:
1. **Box quá lớn (Unusually Large Box - diện tích > 80% canvas):** `59 ảnh`. Các box này phủ gần như toàn bộ khung hình, thường là các đống rác cận cảnh hoặc chú thích bao phủ cả bối cảnh xung quanh gây nhiễu detector.
2. **Box siêu nhỏ dưới ngưỡng phân giải (Sub-resolution Box - diện tích < 0.01% canvas):** `105 ảnh`. Các box này chỉ có kích thước vài pixel, là nhiễu do gán nhãn tự động từ Open Images.
3. **Box trùng lặp hình học (Duplicate Box - IoU >= 0.95 cùng class):** `12 ảnh`. Các box bị duplicate nhiều lần đè lên nhau.

### 4.3. Tác động của việc loại bỏ 176 ảnh lên phân bố 10 lớp

| Lớp (Class) | Số box trong v2 (1.996 ảnh) | Số box trong 176 ảnh loại bỏ | Số box còn lại trong v3 (1.820 ảnh) | Đánh giá an toàn |
|---|:---:|:---:|:---:|---|
| `0: battery` | 2 | **0 (0.0%)** | **2 (100% giữ nguyên)** | **Bảo toàn tuyệt đối 100%** |
| `1: biological` | 5 | **0 (0.0%)** | **5 (100% giữ nguyên)** | **Bảo toàn tuyệt đối 100%** |
| `2: cardboard` | 258 | 36 (14.0%) | 222 | Loại bỏ box nhiễu kích thước siêu nhỏ |
| `3: clothes` | 506 | 90 (17.8%) | 416 | Loại bỏ các box người mặc quần áo thời trang |
| `4: glass` | 313 | 122 (39.0%) | 191 | Làm sạch box trùng lặp trên kính vỡ |
| `5: metal` | 599 | 97 (16.2%) | 502 | Loại bỏ box sub-resolution |
| `6: paper` | 252 | 42 (16.7%) | 210 | Loại bỏ box nhiễu |
| `7: plastic` | 2.228 | 326 (14.6%) | 1.902 | Giảm tải thiên vị (bias) mà không mất đa dạng |
| `8: shoes` | 268 | 8 (3.0%) | 260 | Bảo toàn 97.0% |
| `9: trash` | 1.159 | 532 (45.9%) | 627 | **Làm sạch mạnh mẽ lớp rác chung (catch-all) gây nhiễu** |
| **Tổng số box** | **5.590** | **1.253** | **4.337** | **Dữ liệu sạch, triệt tiêu box lỗi** |

> [!IMPORTANT]
> **Kết luận quan trọng:** Việc loại bỏ 176 ảnh có nhãn bất thường hoàn toàn **không làm mất bất kỳ một bounding box nào của hai lớp đói dữ liệu nghiêm trọng là `battery` (giữ nguyên 100%) và `biological` (giữ nguyên 100%)**. Đồng thời, việc cắt giảm 532 box `trash` và 326 box `plastic` giúp giảm bớt sự mất cân bằng cực đoan trong dataset.

### 4.4. Giải trình 147 ảnh trên đĩa không thuộc splits v2 & Đối soát con số "129 ảnh"
Tổng số ảnh thực tế trong thư mục `data/detection/images/real`: **2.143 ảnh**.  
Số ảnh trong `splits_real_v2`: **1.996 ảnh**.  
Chênh lệch trên đĩa: **147 ảnh** (118 ảnh tiền tố `oi_`, 29 ảnh tiền tố `taco_`).

Toàn bộ 147 ảnh này đã được ghi nhận trong file kiểm soát cách ly [data/detection/quarantine_real_v2.csv](../data/detection/quarantine_real_v2.csv) với các lý do cụ thể:
- `oi_human_present_in_use_object`: **101 ảnh** (Ảnh Open Images người đang đi giày hoặc mặc đồ, không phải rác thải vứt bừa bãi).
- `review_status_REJECTED`: **14 ảnh** (Bị đánh giá REJECTED trong Phase 1 do nhãn sai lệch).
- `empty_label`: **13 ảnh** (File nhãn rỗng 0 box).
- `taco_orientation_mismatch`: **10 ảnh** (Lệch khung tọa độ do cờ xoay EXIF trong TACO).
- `review_status_NEEDS_RELABEL`: **4 ảnh** (Cần chỉnh sửa lại viền nhãn).
- `oi_groupof_or_depiction_box`: **4 ảnh** (Nhãn đám đông/tranh vẽ).
- `exif_orientation_3`: **1 ảnh** (Lỗi xoay ảnh 180 độ).

**Công thức đối soát con số 129 ảnh của QA Reviewer:**
$$101\ (\text{oi\_human}) + 14\ (\text{REJECTED}) + 10\ (\text{orientation}) + 4\ (\text{NEEDS\_RELABEL}) = \mathbf{129\ \text{ảnh có nhãn bị giữ lại trong hàng đợi QA}}$$
(147 ảnh tổng cộng = 129 ảnh có nhãn cần xem xét + 13 ảnh nhãn rỗng + 5 ảnh depiction/exif).

**Vấn đề 2 ảnh chứa nhãn `biological` trong hàng đợi cách ly:**
- `taco_0072.jpg`: Lý do cách ly `review_status_REJECTED` (chứa 1 box biological).
- `taco_0092.jpg`: Lý do cách ly `review_status_NEEDS_RELABEL` (chứa 1 box biological).  
**Quyết định:** Cả 2 ảnh này đều có vấn đề về chất lượng nhãn đã bị từ chối từ Phase 1. Tuyệt đối không được tự ý đưa vào tập train khi chưa qua bước re-annotation độc lập.

---

## 5. P1: KIỂM TOÁN CÁC CỔNG CHỐNG RÒ RỈ DỮ LIỆU (LEAKAGE GATES)

Thực hiện kiểm tra toán học và tri giác toàn diện trên 1.820 ảnh của Dataset v3 Candidate:
1. **Kiểm tra va chạm SHA-256 xuyên Split:**
   - Train vs Val: `0 va chạm`
   - Train vs Test: `0 va chạm`
   - Val vs Test: `0 va chạm`
2. **Kiểm tra va chạm SHA-256 với Tập kiểm thử độc lập (Holdout data/holdout):**
   - So sánh với 27 ảnh trong `reports/qa_acceptance/manifest_hinh_anh_rac.json`: `0 va chạm`.
3. **Kiểm tra rò rỉ nhóm gần trùng lặp pHash (Perceptual Hash):**
   - Khoảng cách Hamming $\le 8$ giữa các ảnh thuộc các nhóm (groups) khác split: `0 nhóm bị rò rỉ`.

**Kết luận Leakage Gate:** **VERIFIED_ZERO_LEAKAGE (Đạt chuẩn cô lập 100%)**.

---

## 6. P1: KIỂM ĐỊNH CHUYÊN SÂU ULTRALYTICS YOLO DATALOADER TRÊN DATASET V3

Báo cáo chi tiết: [reports/model_v2/yolo_dataloader_verification_report.json](../reports/model_v2/yolo_dataloader_verification_report.json)  
Script thực thi: [scripts/verify_yolo_dataloader_batches.py](../scripts/verify_yolo_dataloader_batches.py)

Đã khởi tạo đối tượng `YOLODataset` và `build_dataloader` chính thức từ thư viện Ultralytics trên file cấu hình [data/detection_v3_candidate/yolo/data.yaml](../data/detection_v3_candidate/yolo/data.yaml):
- **Tập Train (kèm Augmentations Albumentations, batch_size=4, imgsz=640):**
  - Quét 1.277 file nhãn: `100% valid`, `0 background`, `0 corrupt`. Đã tạo cache `.cache` chuẩn.
  - Chạy thực nghiệm 5 batch (103 bounding boxes):
    - Kích thước tensor ảnh: `torch.Size([4, 3, 640, 640])`
    - Kích thước tensor nhãn: `torch.Size([N, 1])`, class IDs $\in [0..9]$
    - Kích thước tensor box: `torch.Size([N, 4])`, chuẩn hóa $[0.0..1.0]$
    - Vector ánh xạ `batch_idx`: `torch.Size([N])`, index $\in [0..3]$
    - Lỗi phát hiện: **0 lỗi, 0 cảnh báo**.
- **Tập Validation (chế độ Rectangular, batch_size=4, imgsz=640):**
  - Quét 269 file nhãn: `100% valid`, `0 corrupt`.
  - Chạy thực nghiệm 5 batch (41 bounding boxes): **0 lỗi, 0 cảnh báo**.

**Kết luận:** Dataset v3 Candidate hoàn toàn tương thích và sẵn sàng 100% cho việc huấn luyện Ultralytics YOLOv8 khi có điều kiện tài nguyên phần cứng phù hợp.

---

## 7. P1: DANH MỤC NGUỒN DỮ LIỆU MỞ RỘNG (EXPANSION PROPOSAL)

Báo cáo chi tiết: [reports/data_v2/candidate_expansion_sources.json](../reports/data_v2/candidate_expansion_sources.json)

> [!CAUTION]
> **CHÍNH SÁCH TUÂN THỦ:** Nhóm kỹ sư cam kết **KHÔNG TỰ Ý DOWNLOAD DỮ LIỆU NGOÀI LỚN** khi chưa có phê duyệt từ PM / Product Owner. Toàn bộ các nguồn dưới đây được thiết kế sẵn theo dạng Catalog để sẵn sàng kích hoạt ngay khi được phê duyệt.

| Nhóm dữ liệu | Lớp mục tiêu | Tên nguồn / Dataset | Giấy phép (License) | Số lượng đề xuất | Định dạng nhãn & Phương pháp thu nạp | Trạng thái |
|---|:---:|---|:---:|:---:|---|:---:|
| **Pin / Ắc quy** | `battery` | Open Images V7 (`/m/01cydv`) | CC BY 4.0 | 100 - 150 ảnh | Bbox chuẩn hóa. Lọc bỏ người cầm pin, lấy pin phế thải ngoài trời. | PENDING APPROVAL |
| **Pin / Ắc quy** | `battery` | TACO Dataset (Category 60) | CC BY 4.0 | 30 - 40 ảnh | COCO annotation. Pin lẫn trong rác thực tế trên đường phố. | PENDING APPROVAL |
| **Hữu cơ / Thực phẩm** | `biological` | Open Images V7 (`/m/0f4s2w`, `/m/09qck`) | CC BY 4.0 | 80 - 120 ảnh | Vỏ chuối, cuống táo, rau củ phế thải bị vứt bỏ trên mặt đất. | PENDING APPROVAL |
| **Hữu cơ / Thực phẩm** | `biological` | TACO Dataset (Cat 63 & 64) | CC BY 4.0 | 40 - 50 ảnh | Rác thực phẩm, bã thức ăn phân hủy ngoài trời. | PENDING APPROVAL |
| **Hữu cơ / Thực phẩm** | `biological` | Local Quarantined (`taco_0072`, `taco_0092`) | CC BY 4.0 | 2 ảnh | Re-annotation lại 2 ảnh có sẵn trong hàng đợi cách ly (0 cần download). | PENDING APPROVAL |
| **Hard-Negative** | `0-object` | Vỉa hè, lòng đường sạch | CC0 / Tự chụp | 30 ảnh | File nhãn rỗng 0-byte. Triệt tiêu FP nhận vết nứt, bóng râm thành rác. | PENDING APPROVAL |
| **Hard-Negative** | `0-object` | Tường gạch, mặt tiền tòa nhà | CC0 / Tự chụp | 25 ảnh | File nhãn rỗng 0-byte. Triệt tiêu FP nhận biển hiệu, tường thành bìa carton. | PENDING APPROVAL |
| **Hard-Negative** | `0-object` | Tủ điện kỹ thuật màu xanh | CC0 / Tự chụp | 20 ảnh | File nhãn rỗng 0-byte. Khắc phục triệt để lỗi ảnh `rac_online_012.jpg`. | PENDING APPROVAL |
| **Hard-Negative** | `0-object` | Thùng rác công cộng sạch | CC0 / Tự chụp | 20 ảnh | File nhãn rỗng 0-byte. Dạy mô hình không nhận diện chiếc thùng rác thành rác. | PENDING APPROVAL |

---

## 8. TỔNG HỢP KIỂM THỬ HỆ THỐNG (QUALITY GATE VERIFICATION SUITE)

### 8.1. Thống kê kiểm thử pytest
- **Tổng số tests thu thập trong toàn bộ repository:** **343 tests collected** (tăng từ 332 tests sau khi bổ sung 11 tests kiểm toán chuyên sâu).
- **Bộ kiểm thử trọng tâm (Targeted Suite):** **198 PASSED, 1 SKIPPED, 0 FAILED** (trong 71.83 giây).
- **Danh sách các test case mới được thiết lập trong Phase 4B.1:**
  1. `tests/qa/test_dataset_v3_isolation.py`: 3 tests (Bảo vệ dataset v2, kiểm tra Inode độc lập trên NTFS, kiểm tra tính toàn vẹn khi rebuild).
  2. `tests/qa/test_training_dry_run_api.py`: 1 test (Kiểm tra API forward/loss/backward SSDLite với batch rỗng và hỗn hợp).
  3. `tests/qa/test_v3_splits_audit.py`: 5 tests (Đối soát 1.820 ảnh v3 vs 1.996 ảnh v2, bảo toàn 100% pin và hữu cơ, đối soát 147 ảnh cách ly, kiểm tra 0 rò rỉ SHA/pHash).
  4. `tests/qa/test_yolo_dataloader_batches.py`: 2 tests (Kiểm tra Ultralytics DataLoader batch generation, tensor shapes, và valid coordinates).

### 8.2. Danh mục Bằng chứng đính kèm (Evidence Artifacts)
Tất cả các tài liệu và bằng chứng đã được đóng gói hoàn chỉnh tại:  
`~/Downloads\PHASE4B1_SAFETY_VERIFICATION_EVIDENCE.zip`
- `reports/PHASE4B1_TRAINING_SAFETY_VERIFICATION.md` (Báo cáo này)
- `reports/model_v2/training_dry_run_report.json` (Kết quả đo VRAM & 1-step dry run)
- `reports/model_v2/yolo_dataloader_verification_report.json` (Kết quả nạp batch YOLO)
- `reports/data_v2/v3_splits_reconciliation_audit.json` (Báo cáo đối soát splits & exclusions)
- `reports/data_v2/candidate_expansion_sources.json` (Danh mục nguồn mở rộng)
- `data/detection_v3_candidate/manifests/candidate_image_manifest.csv`
- `data/detection_v3_candidate/manifests/candidate_annotation_manifest.csv`
- `data/detection_v3_candidate/yolo/data.yaml`
- Toàn bộ source code scripts và QA regression tests.
