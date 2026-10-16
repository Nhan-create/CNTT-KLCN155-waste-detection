# BÁO CÁO NGHIỆM THU PHASE 4C CLOSEOUT & KẾ HOẠCH MỞ RỘNG DỮ LIỆU THẬT
**Mã dự án:** CNTT-KLCN155 — Waste Detection System  
**Repository:** `.`  
**Tài liệu:** `reports/PHASE4C_CLOSEOUT_AND_DATA_ACQUISITION.md`  
**Thời điểm hoàn thành:** 2026-10-12T06:20:00Z  
**Vai trò:** Principal Computer Vision Engineer, Senior ML Data Engineer & Independent QA Auditor  

---

## 1. TỔNG QUAN KẾT QUẢ VÀ TUÂN THỦ NGUYÊN TẮC BẤT BIẾN

Giai đoạn **Phase 4C Closeout** đã khắc phục triệt để các rủi ro kỹ thuật còn tồn tại sau đợt review độc lập đối với gói chứng cứ Phase 4C:
1. **Triệt tiêu lỗ hổng Quality Gate:** Tách biệt hoàn toàn kiểm tra hình thức (syntactic) và kiểm tra ngữ nghĩa (semantic). Không còn tình trạng file nhãn 0-byte tự động được cấp chứng chỉ "Hard Negative verified".
2. **Ngăn chặn rò rỉ Train/Test (Contamination Prevention):** Xây dựng module kiểm soát xuất dữ liệu `src/detection/export_train_manifest.py` với cơ chế chặn cứng ngoại lệ `SplitContaminationError` và `UnmappedWasteLeakageError`.
3. **Hiệu chỉnh mô tả hình ảnh chính xác theo thực tế (Ground Truth Reality):** Sửa toàn diện các mô tả chưa chuẩn xác về `taco_0072` (thực phẩm nướng trong túi nilon, không phải lõi táo), `taco_0092` (lõi táo cắn dở đang cầm trên tay, không phải vỏ quả trên đường), và `taco_0082` (pin Panasonic hình trụ AA/AAA, không phải pin cúc áo).
4. **Xác lập ranh giới dữ liệu trung thực:** Dừng hoàn toàn việc tái chế nguồn TACO đã cạn kiệt; lập kế hoạch thu thập 100 ảnh mới từ các nguồn xác thực ngoài TACO cho 4 nhóm ưu tiên.
5. **Không có bất kỳ phát ngôn khẳng định độ chính xác tăng:** Khi chưa tiến hành huấn luyện thực tế, mọi nhận định chỉ dừng ở mức độ an toàn dữ liệu và tính sẵn sàng của kiến trúc.

---

## 2. KẾT QUẢ SỬA QUALITY GATE & PHÂN LOẠI NGỮ NGHĨA (P0)

### 2.1. Lỗ hổng cũ được loại bỏ
Trong `scripts/validate_pilot_quality_gates.py` phiên bản trước, đoạn mã:
```python
if len(lbl_text) == 0:
    # Hard negative verified
    continue
```
đã bị gỡ bỏ hoàn toàn. File nhãn 0-byte chỉ được coi là hợp lệ về mặt cú pháp định dạng YOLO, nhưng **KHÔNG THỂ tự động chứng minh ảnh là negative sample**.

### 2.2. Cơ chế phân loại 2 tầng (Two-Tier Architecture)
Hệ thống Quality Gate mới trong `scripts/validate_pilot_quality_gates.py` phân tách thành:
- **Tầng 1 (Syntactic Gates):** `format_and_decodability_gate`, `geometry_gate`, `taxonomy_gate`, `intra_pilot_duplicate_gate`, `holdout_leakage_gate`.
- **Tầng 2 (Semantic & Split Safety Gates):**
  - `semantic_negative_gate`: Tra cứu manifest. Chỉ ảnh có trạng thái `VERIFIED_TRUE_NEGATIVE` mới được chấp nhận làm negative. Tám (8) ảnh TACO chứa Category 51 (`Rope & strings`) bị gắn cờ vi phạm ngữ nghĩa vì chứa rác chưa gán nhãn.
  - `split_safety_gate`: Phát hiện 2 ảnh thuộc TEST split (`taco_0068.jpg`, `taco_0082.jpg`) và cấm tuyệt đối nạp vào train.
  - `approval_status_gate`: Phát hiện 2 ảnh đề xuất sửa nhãn biological (`taco_0072.jpg`, `taco_0092.jpg`) đang ở diện chờ duyệt (`PROPOSED_PENDING_USER_APPROVAL`).

### 2.3. Bảng phân loại chính thức 15 ảnh Pilot
- **`EXISTING_TEST` (2 ảnh):** `taco_0068.jpg`, `taco_0082.jpg` $\to$ Cấm nạp vào Train.
- **`EXISTING_TRAIN` (3 ảnh):** `taco_0841.jpg`, `taco_0924.jpg`, `taco_0456.jpg` $\to$ Đã có sẵn trong Train, không tính là mới.
- **`QUARANTINED` / `PENDING_REVIEW` (2 ảnh):** `taco_0072.jpg`, `taco_0092.jpg` $\to$ Chờ người dùng phê duyệt.
- **`EMPTY_LABEL_UNMAPPED_WASTE` / `REJECTED` (8 ảnh):** `taco_0170`, `taco_0431`, `taco_0556`, `taco_0909`, `taco_0912`, `taco_0916`, `taco_0917`, `taco_0925` $\to$ Từ chối làm negative vì chứa dây thừng phế thải.
- **`VERIFIED_TRUE_NEGATIVE` (0 ảnh):** Hiện tại chưa có ảnh negative nào được xác thực trong pilot.

**Kết luận Quality Gate:** Trạng thái tổng thể của Pilot Dataset v3 được ghi nhận chính thức là **`TRAINING_BLOCKED`** (Khóa huấn luyện, ngăn chặn việc mô hình tự ý nạp cả thư mục pilot).

---

## 3. KIẾN TRÚC NGĂN CHẶN NHIỄM CHÉO TRAIN/TEST (P0)

Nhóm kỹ thuật đã triển khai module kiểm soát xuất dữ liệu:
[`src/detection/export_train_manifest.py`](../src/detection/export_train_manifest.py)

### 3.1. Các lớp ngoại lệ an toàn (Exception Guards)
- `SplitContaminationError`: Kích hoạt ngay lập tức nếu phát hiện ảnh thuộc split `TEST` hoặc `VAL` trong danh sách xuất Train.
- `UnmappedWasteLeakageError`: Kích hoạt nếu phát hiện ảnh có nhãn 0-byte nhưng chứa rác chưa gán nhãn (Category 51 rope).
- `PendingApprovalError`: Kích hoạt nếu cố tình xuất các ảnh đang chờ duyệt (`PROPOSED_PENDING_USER_APPROVAL`) mà không có cờ `--allow-pending`.

### 3.2. Bằng chứng kiểm thử hồi quy (Regression Test Proof)
Bộ test suite chuyên trách [`tests/qa/test_split_contamination_safety.py`](../tests/qa/test_split_contamination_safety.py) gồm 6 bài kiểm thử nghiêm ngặt:
1. `test_test_images_strictly_blocked_from_train_export` $\to$ **PASSED** (chứng minh `taco_0068` và `taco_0082` kích hoạt `SplitContaminationError`).
2. `test_unmapped_waste_images_strictly_blocked_from_train_export` $\to$ **PASSED** (chứng minh 8 ảnh dây thừng kích hoạt `UnmappedWasteLeakageError`).
3. `test_proposed_biological_reannotations_blocked_until_user_approval` $\to$ **PASSED** (chứng minh `taco_0072` và `taco_0092` bị chặn mặc định).
4. `test_naive_pilot_directory_loading_blocked` $\to$ **PASSED** (xác nhận verdict của pilot là `TRAINING_BLOCKED`).
5. `test_pilot_train_eligible_manifest_contains_zero_test_or_unmapped_images` $\to$ **PASSED** (xác nhận file xuất sạch chỉ có 3 ảnh train cũ, 0 ảnh test, 0 ảnh unmapped).
6. `test_official_splits_contain_zero_test_contamination` $\to$ **PASSED** (tập Train và Test chính thức có độ giao nhau bằng 0).

Kết quả chạy thực tế: **6/6 passed** trong `0.21s`.

---

## 4. HIỆU CHỈNH MÔ TẢ ẢNH THEO THỰC TẾ GROUND TRUTH (P1)

Dựa trên việc kiểm tra trực tiếp từng pixel của các file ảnh:

### 4.1. Ảnh `taco_0072.jpg`
- **Mô tả cũ:** "Lõi táo + Túi nilon trên mặt đường nhựa".
- **Thực tế pixel:** Thực phẩm nướng màu nâu sẫm (bánh mì nguyên cám / bánh bun) nằm bên trong túi nilon bao bì trong suốt, được cầm bằng tay người trong nhà, phía trên chậu cây húng quế và tường trắng. Hoàn toàn không phải lõi táo và không nằm trên mặt đường.
- **Phân loại nhãn:** Giữ nguyên Lớp 1 (`biological`) cho phần bánh bên trong và Lớp 7 (`plastic`) cho bao bì nilon bên ngoài.
- **Trạng thái:** Giữ nguyên `PROPOSED_PENDING_USER_APPROVAL`, không tự ý ghi đè vào `data/detection/labels/real/taco_0072.txt`.

### 4.2. Ảnh `taco_0092.jpg`
- **Mô tả cũ:** "Vỏ trái cây trên mặt đường nhựa".
- **Thực tế pixel:** Lõi táo đã bị cắn dở (apple core) đang được các ngón tay người cầm thẳng đứng trên không trung. Nền phía dưới là mặt đường nhựa có lá rụng và mũi giày.
- **Vấn đề quy chuẩn rác thải:** Việc vật thể đang được cầm trên tay người (in-hand object) đặt ra câu hỏi về tiêu chuẩn nhãn rác môi trường (rác vứt tự do vs vật dụng đang cầm). Nhóm kỹ thuật giữ ảnh ở trạng thái `PROPOSED_PENDING_USER_APPROVAL` chờ người dùng quyết định tiêu chuẩn.

### 4.3. Ảnh `taco_0082.jpg`
- **Mô tả cũ:** "Chứa pin cúc áo (button cell)".
- **Thực tế pixel:** Viên pin hình trụ Panasonic cỡ AA/AAA (vỏ màu vàng kim - đen, in chữ thương hiệu 'Panasonic' màu trắng nằm ngang) đặt trên mặt bàn trắng cùng vỉ thuốc nhôm, tuýp keo/kem đánh răng, chai xịt và nắp chai. Hoàn toàn không phải pin cúc áo.
- **Thuộc split:** `TEST`. Giữ nguyên trong Test để đánh giá khách quan.

---

## 5. KẾ HOẠCH MỞ RỘNG DỮ LIỆU THỰC TẾ (P1)

Tài liệu chi tiết: [`reports/data_v2/real_data_acquisition_plan.md`](../reports/data_v2/real_data_acquisition_plan.md)

Nhóm kỹ thuật đề xuất kế hoạch thu thập có kiểm soát gồm 4 nhóm mục tiêu với quy mô **25 ảnh/nhóm** (tổng cộng 100 ảnh thí điểm), đáp ứng đầy đủ tiêu chí pháp lý, kỹ thuật và an toàn chống rò rỉ:

| Nhóm | Mục tiêu lớp | Số ảnh | Số box dự kiến | Nguồn khả thi | License | Biện pháp chống rò rỉ |
|---|---|:---:|:---:|---|:---:|---|
| **Nhóm A** | `battery` (Lớp 0) | 25 ảnh | 35–50 boxes | Roboflow E-Waste + Chụp thực địa TP.HCM | CC BY 4.0 / CC0 | Đối chiếu SHA-256 + pHash với Holdout |
| **Nhóm B** | `biological` (Lớp 1) | 25 ảnh | 30–45 boxes | Open Images V7 Validation Pool | CC BY 4.0 | Lọc nghiêm ngặt rác rơi vãi tự do |
| **Nhóm C** | Hard-Negatives | 25 ảnh | 0 boxes (0-byte) | Chụp thực địa đô thị sạch + CC0 | CC0 / Public Domain | Thẩm định 2-stage QA negative |
| **Nhóm D** | Multi-waste Co-occurrence | 25 ảnh | 75–100 boxes | TrashNet Multi-Object Open Subset | Academic / MIT | Lọc pHash chống trùng với Test |
| **TỔNG** | **4 nhóm ưu tiên** | **100 ảnh** | **140–195 boxes** | **Dung lượng dự kiến: ~100 MB** | Đã thẩm định rõ ràng | Zero unapproved downloads |

---

## 6. THÔNG SỐ VÀ CẤU HÌNH THỬ NGHIỆM MODEL (P2 - PREPARATION ONLY)

> [!NOTE]
> **Không có bất kỳ tác vụ huấn luyện nào được kích hoạt trong đợt này.** Toàn bộ cấu hình và lệnh dưới đây được chuẩn bị sẵn sàng cho giai đoạn sau khi dữ liệu mới được duyệt.

### 6.1. Dataset Version & Split Target
- Dataset: `configs/detection_dataset_real_v2.yaml` (hoặc candidate v3 sau khi tích hợp 100 ảnh mới qua `src.detection.export_train_manifest`).
- Split tỷ lệ: 80% Train, 20% Val, 0% Test (tập Test và 27 ảnh Holdout tại `data/holdout` giữ nguyên 100% không đổi).

### 6.2. Kiểm soát phần cứng (Preflight Guard)
- RAM hệ thống hiện tại: ~16 GB, khả dụng: ~800–1200 MB (nguy cơ OOM cao nếu chạy tác vụ nền nặng).
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM).
- Trước khi chạy training, bắt buộc kích hoạt guard:
  ```powershell
  .venv\Scripts\python.exe -m src.detection.preflight_guard
  ```

### 6.3. Lệnh huấn luyện chuẩn bị sẵn
- **Mô hình chính (SSDLite320-MobileNetV3):**
  ```powershell
  .venv\Scripts\python.exe scripts/train_detection_v2.py ssdlite --run E2_ssdlite320_10cls --data configs/detection_dataset_real_v2.yaml
  ```
- **Mô hình đối chứng (YOLOv8n):**
  ```powershell
  .venv\Scripts\python.exe scripts/train_detection_v2.py yolo --run E1_yolov8n_10cls --data configs/detection_dataset_real_v2.yaml
  ```
  *(Cả hai lệnh đều tự động kích hoạt `enforce_preflight_guard`, ngăn chặn hoàn toàn nguy cơ tràn RAM/VRAM. Tuyệt đối không gọi lệnh raw `YOLO.train()` trực tiếp).*

### 6.4. Giao thức đánh giá trước/sau (Before/After Evaluation Protocol)
- **Baseline hiện tại:** Đánh giá trên checkpoint gốc với Dataset v2 (Battery $N=1$, Bio $N=3$, Negatives $=0$).
- **Experiment 1 (Post-Acquisition):** Đánh giá lại trên cùng tập Test và Holdout sau khi nạp 100 ảnh thật đã duyệt.
- **Tiêu chí so sánh:** Delta Recall trên lớp Battery, Delta Recall trên lớp Biological, và Tỷ lệ giảm False Positive trên các mẫu tủ điện và nền gạch sạch.

---

## 7. ĐỐI CHIẾU 6 TIÊU CHÍ KẾT THÚC (DOD CONCLUDING CRITERIA)

| Tiêu chí | Trạng thái | Bằng chứng kỹ thuật |
|---|:---:|---|
| **1. Không còn chuyện file 0-byte tự động công nhận là true negative** | **ĐẠT (PASSED)** | Đã xóa logic bypass trong `scripts/validate_pilot_quality_gates.py`; 8 ảnh dây thừng bị đánh trượt ngữ nghĩa (`semantic_negative_gate: FAILED`). |
| **2. Ảnh thuộc test không thể lọt vào train export** | **ĐẠT (PASSED)** | Module `src/detection/export_train_manifest.py` chặn cứng `taco_0068` và `taco_0082` bằng `SplitContaminationError`; 6 regression tests passed. |
| **3. Pilot được ghi đúng trạng thái BLOCKED nếu còn ảnh rejected** | **ĐẠT (PASSED)** | Verdict của Pilot Dataset v3 ghi nhận chính thức: `TRAINING_BLOCKED`. Báo cáo JSON ghi nhận rõ 3 lý do chặn. |
| **4. Không có dữ liệu gốc bị ghi đè** | **ĐẠT (PASSED)** | File nhãn gốc `data/detection/labels/real/taco_0072.txt` và `taco_0092.txt` nguyên vẹn 100%. Các đề xuất chỉ nằm trong pilot và báo cáo. |
| **5. Có kế hoạch thu thập ảnh mới khả thi, không dựa vào TACO cạn** | **ĐẠT (PASSED)** | Đã hoàn thành `reports/data_v2/real_data_acquisition_plan.md` cho 4 nhóm ưu tiên (100 ảnh) từ Roboflow, OpenImages V7 và chụp thực tế đô thị. |
| **6. Không có lời khẳng định độ chính xác tăng khi chưa training** | **ĐẠT (PASSED)** | Toàn bộ báo cáo giữ tính trung thực tuyệt đối: 0 khẳng định tăng mAP; chỉ nêu kế hoạch so sánh trước/sau khi huấn luyện thật. |

---

## 8. CÁC QUYẾT ĐỊNH CẦN NGƯỜI DÙNG PHÊ DUYỆT (PENDING USER DECISIONS)

Để tiến hành bước tiếp theo và dừng các vòng lặp kiểm toán, nhóm kỹ thuật kính trình 3 quyết định cụ thể:

1. **Quyết định 1 (Phê duyệt re-annotation cho `taco_0072`):**
   - Nội dung: Chấp thuận cặp nhãn siết viền cho bánh nướng (Lớp 1: `biological`) và bao bì nilon (Lớp 7: `plastic`) trong `taco_0072.jpg` để chuyển từ Quarantine sang Train.
   - Lựa chọn: `[APPROVE_TACO_0072]` hoặc `[REJECT_TACO_0072]`.
2. **Quyết định 2 (Quy chuẩn rác cầm trên tay cho `taco_0092`):**
   - Nội dung: Có chấp nhận ảnh rác cầm trên tay người (`taco_0092.jpg` lõi táo) vào tập Train không, hay chỉ chấp nhận rác rơi vãi tự do ngoài môi trường?
   - Lựa chọn: `[APPROVE_HANDHELD_WASTE]` hoặc `[EXCLUDE_HANDHELD_WASTE]`.
3. **Quyết định 3 (Phê duyệt thu thập 100 ảnh thật theo kế hoạch):**
   - Nội dung: Cho phép nhóm kỹ thuật triển khai thu thập thí điểm 100 ảnh (25 Pin thải, 25 Rác hữu cơ ngoài trời, 25 Hard-negatives đường phố sạch, 25 Đống rác hỗn hợp) theo đúng catalog tại `reports/data_v2/real_data_acquisition_plan.md`.
   - Lựa chọn: `[APPROVE_ACQUISITION_100_IMAGES]`.
