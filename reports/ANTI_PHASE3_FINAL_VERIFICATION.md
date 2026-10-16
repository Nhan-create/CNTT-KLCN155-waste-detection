# BÁO CÁO XÁC MINH DỨT ĐIỂM PHASE 3
## (PHASE 3 FINAL VERIFICATION & EVIDENCE REPORT)

**Dự án:** Nhận diện Rác thải Đa lớp Thời gian thực (CNTT-KLCN155)  
**Tác giả:** Antigravity Principal Computer Vision & Independent QA Auditor  
**Thời gian lập:** 12/10/2026  
**Trạng thái tổng thể:** HOÀN TẤT XÁC MINH TOÀN DIỆN, BẢO TOÀN DỮ LIỆU GỐC, KHÔNG TỰ DUYỆT GT V2

---

## 1. MỤC TIÊU VÀ PHẠM VI XÁC MINH DỨT ĐIỂM

Báo cáo này hoàn thiện và giải quyết dứt điểm các yêu cầu kỹ thuật phát sinh sau `ANTI_PHASE3_CONTINUATION.md`, bao gồm:
1. **P0 — Sửa đổi triệt để quy tắc Evaluator GT v2:** Xây dựng 3 chính sách rõ ràng (**NEGATIVE, EXCLUDED, IGNORE REGION**), chấm dứt việc tự động coi mọi ảnh zero-active là negative.
2. **P1 — Bằng chứng phẫu thuật bản vá Dataset:** Cung cấp unified diff và mã băm SHA-256 trước/sau cho việc xóa box trùng trong `oi_2734e597675d8a97.txt`.
3. **P1 — Nhật ký kiểm thử nguyên bản (Test Evidence):** Xuất toàn bộ `pytest_collection.log`, `pytest_targeted.log`, `python_environment.json`, `tests_summary.json` với lệnh chạy và interpreter path thực tế.
4. **P1 — Kiểm toán chất lượng dữ liệu & rò rỉ:** Chứng minh 0 trùng lặp SHA-256, 0 rò rỉ chéo split, 0 rò rỉ holdout; làm rõ nguồn gốc Open Images và hiện trạng 0% ảnh negative.
5. **P1 — Chứng minh toán học hình học TACO:** Chứng minh việc clip $y = -1.0$ bảo toàn đáy box và giảm chiều cao đúng 1 pixel, không dịch chuyển tọa độ.
6. **P2 — Thẩm định Ground Truth 6 đối tượng tranh chấp:** Không suy đoán vật liệu bên trong túi kín, xuất phương án preview rõ ràng cho chủ dự án.
7. **P2 — Bảng tiêu chí sẵn sàng huấn luyện:** Đánh giá cụ thể theo 3 trạng thái: `VERIFIED`, `NEEDS HUMAN APPROVAL`, `BLOCKED`.

---

## 2. P0 — SỬA ĐỔI QUY TẮC EVALUATION GT V2 VÀ ĐỐI SOÁT METRICS

### A. Vấn đề của Evaluator trước đây
Trong phiên bản evaluator trước, các đối tượng không active đã được lọc bỏ, nhưng khi một ảnh trong GT v2 không còn active object nào (zero-active), evaluator mặc định coi ảnh đó là **Negative** (ảnh nền sạch). Do đó, bất kỳ dự đoán nào của mô hình trên ảnh này đều bị tính là **False Positive (FP)**.

Thực tế trong 10 ảnh zero-active của GT v2:
- **2 ảnh thùng/giỏ rác:** `images (8).jpg` (sọt rác đục lỗ) và `images (18).jpg` (thùng rác công cộng 2 bánh). Đây là hạ tầng thu gom, không chứa rác đơn lẻ trên thân vỏ $\rightarrow$ **Có thể coi là NEGATIVE benchmark** để kiểm tra việc mô hình không bắt nhầm thiết bị thu gom thành rác.
- **8 ảnh đống rác cận cảnh:** `images (6)`, `images (7)`, `images (9)`, `images (13)`, `images (19)`, `OIP.webp`, `photo-5...webp`, `rac-thai-huu-co-1.jpg`. Các ảnh này chứa toàn bộ là rác thải (bìa carton, chai nhựa, quần áo dệt, rác hữu cơ) chiếm từ 85% đến 100% canvas. Việc coi các ảnh này là nền sạch khiến cho dự đoán đúng loại rác (ví dụ: SSDLite đoán `cardboard` trên đống bìa `images (9).jpg`) bị phạt oan thành False Positive!

### B. Ba chính sách đánh giá rõ ràng trong `scripts/compare_eval_v1_vs_v2.py`
Mã nguồn tại [`scripts/compare_eval_v1_vs_v2.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/compare_eval_v1_vs_v2.py) đã được nâng cấp hỗ trợ đầy đủ 3 chính sách:
1. **NEGATIVE BENCHMARK:** Áp dụng cho ảnh không chứa rác mục tiêu (ví dụ 2 ảnh thùng rác). Active GT = 0. Mọi dự đoán trên ảnh đều được đưa vào đánh giá và tính là **False Positive (FP)**.
2. **EXCLUDED BENCHMARK:** Áp dụng cho ảnh nằm ngoài phạm vi bài toán instance detection (8 ảnh đống rác cận cảnh). Ảnh này được loại hoàn toàn khỏi mẫu đánh giá instance: không tính GT, không tính predictions vào `total_preds`, không phạt FP hay TP.
3. **IGNORE REGION:** Áp dụng cho vùng không gian đống rác hoặc vùng thảm vụn. Các dự đoán có tâm hoặc phần lớn diện tích giao cắt (IoU $\ge 0.1$ hoặc containment $\ge 0.3$) nằm trong hộp ignore-region sẽ được **bỏ qua** (`ignored_preds += 1`), không tính là TP và không phạt FP.

> **Bảo toàn GT v1:** Evaluator giữ nguyên 100% logic đánh giá cho GT v1 frozen (không áp dụng lọc hay loại trừ). Kết quả GT v1 tái lập chính xác từng số liệu gốc: Before (13 TP, 8 FP, 44 FN), After (8 TP, 3 FP, 49 FN).

### C. Bảng đối so sánh toàn diện các chính sách đánh giá (Confidence $\ge 0.25$, IoU $\ge 0.50$)

| Pipeline | Phiên bản GT & Chính sách | GT | Preds | Ign | TP | FP | FN | Precision % | Recall % | F1 % |
|---|---|---|---|---|---|---|---|---|---|---|
| **SSDLite Before** | `v1_frozen` (Baseline gốc) | 57 | 21 | 0 | 13 | 8 | 44 | **61.90** | **22.81** | **33.33** |
| | `v2_preliminary_all_negative` (Cũ: đống rác bị tính FP) | 45 | 21 | 0 | 10 | 11 | 35 | 47.62 | 22.22 | 30.30 |
| | `v2_approved_active` (Đề xuất: Thùng rác=Neg, Đống rác=Excl) | 45 | 20 | 0 | 10 | 10 | 35 | **50.00** | **22.22** | **30.77** |
| | `v2_approved_discrete_only` (Chỉ tính 39 vật thể Accepted) | 39 | 20 | 0 | 9 | 11 | 30 | 45.00 | 23.08 | 30.51 |
| | `v2_ignore_region_active` (Đống rác là Ignore Region) | 45 | 20 | 1 | 10 | 10 | 35 | **50.00** | **22.22** | **30.77** |
| **SSDLite After** | `v1_frozen` (Baseline gốc) | 57 | 11 | 0 | 8 | 3 | 49 | **72.73** | **14.04** | **23.53** |
| | `v2_preliminary_all_negative` (Cũ: đống rác bị tính FP) | 45 | 11 | 0 | 7 | 4 | 38 | 63.64 | 15.56 | 25.00 |
| | `v2_approved_active` (Đề xuất: Thùng rác=Neg, Đống rác=Excl) | 45 | 10 | 0 | 7 | 3 | 38 | **70.00** | **15.56** | **25.45** |
| | `v2_approved_discrete_only` (Chỉ tính 39 vật thể Accepted) | 39 | 10 | 0 | 6 | 4 | 33 | 60.00 | 15.38 | 24.49 |
| | `v2_ignore_region_active` (Đống rác là Ignore Region) | 45 | 10 | 1 | 7 | 3 | 38 | **70.00** | **15.56** | **25.45** |
| **YOLOv8n E1** | `v1_frozen` (Baseline gốc) | 57 | 32 | 0 | 5 | 27 | 52 | **15.62** | **8.77** | **11.24** |
| | `v2_preliminary_all_negative` (Cũ: đống rác bị tính FP) | 45 | 32 | 0 | 5 | 27 | 40 | 15.62 | 11.11 | 12.99 |
| | `v2_approved_active` (Đề xuất: Thùng rác=Neg, Đống rác=Excl) | 45 | 26 | 0 | 5 | 21 | 40 | **19.23** | **11.11** | **14.08** |
| | `v2_approved_discrete_only` (Chỉ tính 39 vật thể Accepted) | 39 | 26 | 0 | 5 | 21 | 34 | 19.23 | 12.82 | 15.38 |
| | `v2_ignore_region_active` (Đống rác là Ignore Region) | 45 | 26 | 6 | 5 | 21 | 40 | **19.23** | **11.11** | **14.08** |

### D. Minh chứng Regression Tests (7/7 PASSED)
Đã triển khai bộ test tự động tại [`tests/qa/test_compare_eval_v1_vs_v2.py`](file:///D:/CNTT-KLCN155-waste-detection/tests/qa/test_compare_eval_v1_vs_v2.py):
1. `test_best_match_must_have_the_prediction_class`: Khớp nhãn class-aware một-đối-một.
2. `test_real_before_quick_fix_v1_matching`: Bảo toàn kết quả GT v1 frozen (13 TP, 8 FP, 44 FN).
3. `test_negative_benchmark_policy_counts_fp`: Kiểm chứng ảnh negative benchmark bị phạt FP khi có dự đoán sai.
4. `test_excluded_image_policy_discards_predictions`: Kiểm chứng ảnh đống rác bị excluded không phạt FP.
5. `test_ignore_region_policy_discards_contained_prediction`: Kiểm chứng dự đoán rơi vào hộp ignore region bị bỏ qua, không phạt FP.
6. `test_ignore_region_policy_penalizes_outside_prediction`: Kiểm chứng dự đoán rơi ra ngoài ignore region vẫn bị phạt FP.
7. `test_v2_approved_policy_on_real_predictions`: Kiểm chứng số liệu thực tế trên bản vá SSDLite After Quick Fix (7 TP, 3 FP, Precision 70.0%).

---

## 3. P1 — BẰNG CHỨNG PHẪU THUẬT BẢN VÁ DATASET (`oi_2734e597675d8a97.txt`)

Do thư mục `data/detection/` nằm trong `.gitignore` (dòng 49), lệnh `git diff` thông thường không in ra sự thay đổi của file nhãn. Antigravity đã tạo script [`scripts/generate_patch_evidence_oi.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/generate_patch_evidence_oi.py) xuất bản vá độc lập tại [`reports/patches/patch_oi_2734e597675d8a97.diff`](file:///D:/CNTT-KLCN155-waste-detection/reports/patches/patch_oi_2734e597675d8a97.diff) và hồ sơ JSON tại [`reports/patches/patch_oi_2734e597675d8a97_evidence.json`](file:///D:/CNTT-KLCN155-waste-detection/reports/patches/patch_oi_2734e597675d8a97_evidence.json).

### Bảng đối chiếu mã băm SHA-256 và số lượng box:

| Đường dẫn tệp | Trạng thái | Số Bbox | Mã băm SHA-256 |
|---|---|---|---|
| `data/detection/labels/real/oi_2734e597675d8a97.txt` | **TRƯỚC** | 2 | `608e6f3df80f2d84711fe9f3151ec85c88b64a51e60472e50c4aa301b44ec4df` |
| | **SAU** | 1 | `ce34d49fd17aa36d23fa185ffc7c5b7db806f9f91b79d59bc6c580499c0d5964` |
| `data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt` | **TRƯỚC** | 2 | `a90623a637a89e900994f31cfa25c1dafd2f99583b27b9c97a4a2bb34ff441f9` |
| | **SAU** | 1 | `b21be8ce423fc1fc15b7993ab98b991dae745d431ead63139c812e3c43095011` |
| `data/detection_agnostic/labels/real/oi_2734e597675d8a97.txt` | **TRƯỚC** | 2 | `a90623a637a89e900994f31cfa25c1dafd2f99583b27b9c97a4a2bb34ff441f9` |
| | **SAU** | 1 | `b21be8ce423fc1fc15b7993ab98b991dae745d431ead63139c812e3c43095011` |
| `data/detection/real_v2_source_manifest.csv` (dòng 108) | **TRƯỚC** | `num_boxes: 2` | `oi_2734e597675d8a97.jpg,...,new,2,0` |
| | **SAU** | `num_boxes: 1` | `oi_2734e597675d8a97.jpg,...,new,1,0` |
| `data/detection/real_v2_split_manifest.csv` (dòng 104) | **TRƯỚC** | `num_boxes: 2` | `...be1d1ed75cb7ae89994956272a680e1c60ce918d47e1d5c4b2539fa73b92c9c0,2,clothes` |
| | **SAU** | `num_boxes: 1` | `...ce34d49fd17aa36d23fa185ffc7c5b7db806f9f91b79d59bc6c580499c0d5964,1,clothes` |

### Unified Diff nguyên bản:
```diff
--- a/data/detection/labels/real/oi_2734e597675d8a97.txt
+++ b/data/detection/labels/real/oi_2734e597675d8a97.txt
@@ -1,2 +1 @@
 3 0.500000 0.500000 1.000000 1.000000
-3 0.500000 0.500000 1.000000 1.000000
--- a/data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt
+++ b/data/detection/labels_agnostic/real/oi_2734e597675d8a97.txt
@@ -1,2 +1 @@
 0 0.500000 0.500000 1.000000 1.000000
-0 0.500000 0.500000 1.000000 1.000000
--- a/data/detection/real_v2_source_manifest.csv (line 108)
+++ b/data/detection/real_v2_source_manifest.csv (line 108)
@@ -1 +1 @@
-oi_2734e597675d8a97.jpg,openimages_v7,2734e597675d8a97,validation,https://open-images-dataset.s3.amazonaws.com/validation/2734e597675d8a97.jpg,new,2,0
+oi_2734e597675d8a97.jpg,openimages_v7,2734e597675d8a97,validation,https://open-images-dataset.s3.amazonaws.com/validation/2734e597675d8a97.jpg,new,1,0
--- a/data/detection/real_v2_split_manifest.csv (line 104)
+++ b/data/detection/real_v2_split_manifest.csv (line 104)
@@ -1 +1 @@
-oi_2734e597675d8a97.jpg,val,openimages_v7,g00102,openimages_v7|clothes,858c1d6d695a3e65,be1d1ed75cb7ae89994956272a680e1c60ce918d47e1d5c4b2539fa73b92c9c0,2,clothes
+oi_2734e597675d8a97.jpg,val,openimages_v7,g00102,openimages_v7|clothes,858c1d6d695a3e65,ce34d49fd17aa36d23fa185ffc7c5b7db806f9f91b79d59bc6c580499c0d5964,1,clothes
```

---

## 4. P1 — NHẬT KÝ KIỂM THỬ NGUYÊN BẢN (TEST EVIDENCE)

Đã xuất đầy đủ 4 tệp bằng chứng tại thư mục [`reports/test_evidence/`](file:///D:/CNTT-KLCN155-waste-detection/reports/test_evidence/):
1. **Thông tin môi trường (`python_environment.json`):**
   - Trình thông dịch: `D:\CNTT-KLCN155-waste-detection\.venv\Scripts\python.exe`
   - Phiên bản: `Python 3.12.10 (tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
   - Phần cứng: `NVIDIA GeForce RTX 2050 (CUDA 12.1, PyTorch 2.5.1+cu121)`
   - Thư viện: `Torchvision 0.20.1+cu121`, `Ultralytics 8.4.171`, `imagehash 4.3.2`, `PyQt5 5.15.11`
2. **Pytest Collection Log (`pytest_collection.log`):**
   - Lệnh: `.venv\Scripts\python.exe -m pytest --collect-only`
   - Kết quả: **311 tests collected, 0 errors, Exit code: 0** (Thời gian: 20.95s).
3. **Pytest Targeted Run Log (`pytest_targeted.log`):**
   - Lệnh: `.venv\Scripts\python.exe -m pytest -v <16 targeted test files>`
   - Kết quả: **132 PASSED, 1 SKIPPED, 0 FAILED, Exit code: 0** (Thời gian: 80.36s).
   - *(Lưu ý: Test bị skip là `test_hardware_training_budget` do gate kiểm soát phần cứng chủ động bỏ qua để tránh chiếm dụng GPU trong phiên kiểm toán).*
4. **Bản tổng kết máy đọc (`tests_summary.json`):**
   - Khẳng định phạm vi: Chạy trọn vẹn 133 bài test mục tiêu trọng yếu của toàn bộ hệ thống phát hiện, mapping, tọa độ, tiling, evaluator, web pipeline và data integrity.

---

## 5. P1 — CHẤT LƯỢNG TẬP DỮ LIỆU VÀ RÀ SOÁT RÒ RỈ (QUALITY & LEAKAGE)

Báo cáo JSON độc lập: [`reports/DATASET_QUALITY_AND_LEAKAGE_REPORT.json`](file:///D:/CNTT-KLCN155-waste-detection/reports/DATASET_QUALITY_AND_LEAKAGE_REPORT.json).

### Kết quả kiểm toán 1,996 ảnh và 5,584 bounding boxes trong `splits_real_v2`:
1. **Trùng lặp SHA-256 nội bộ:** **0 ảnh** trùng mã băm SHA-256 trong cùng split hoặc giữa các split.
2. **Rò rỉ giữa các tập Train/Val/Test (Cross-split Leakage):** **0 trường hợp** (cả theo filename, theo SHA-256 và theo nhóm pHash near-duplicate).
3. **Rò rỉ với tập Holdout độc lập (`D:\Hinh_Anh_Rac`):** **0 trường hợp** (không có bất kỳ ảnh hoặc mã băm nào của 27 ảnh holdout xuất hiện trong train, val hay test).
4. **Lỗi cú pháp nhãn:** 0 ảnh hỏng, 0 nhãn sai class ID ngoài $[0, 9]$, 0 box âm, 0 box tràn biên, 0 box trùng lặp sau bản vá.
5. **Hiện trạng ảnh cảnh nền (Hard-negatives):**
   - Số lượng ảnh nhãn rỗng (negative images): **0 ảnh (0.0%)**.
   - **Lưu ý khoa học bắt buộc:** Thiếu ảnh negative là **giả thuyết ứng viên hàng đầu** giải thích hiện tượng false positives trên biển hiệu và tủ điện đô thị. Tuy nhiên, theo đúng phương pháp luận khoa học, đây là một giả thuyết cần được kiểm chứng thực nghiệm bằng việc so sánh mô hình train có và không có 150 ảnh negative trong Phase 4, không được khẳng định chủ quan là nguyên nhân duy nhất.
6. **Mất cân bằng lớp (Class Imbalance) ở tập Train (1,398 ảnh, 3,963 boxes):**
   - `battery`: **1 box (0.03%)** $\rightarrow$ Đói dữ liệu trầm trọng.
   - `biological`: **3 boxes (0.08%)** $\rightarrow$ Đói dữ liệu trầm trọng.
   - `cardboard`: 189 boxes (4.77%).
   - `clothes`: 645 boxes (16.28%).
   - `glass`: 272 boxes (6.86%).
   - `metal`: 355 boxes (8.96%).
   - `paper`: 326 boxes (8.23%).
   - `plastic`: **1,972 boxes (49.77%)** $\rightarrow$ Chiếm gần 50% toàn bộ dataset.
   - `shoes`: 144 boxes (3.63%).
   - `trash`: 56 boxes (1.41%).
7. **Nguồn gốc và bản quyền Open Images V7:**
   - Dữ liệu trích xuất từ Open Images V7 thuộc giấy phép CC-BY 2.0 / CC-BY 4.0.
   - 101/114 ảnh Open Images là ảnh người mẫu thời trang đang mặc trang phục nhưng cơ thể người không được gán nhãn.
   - **Cam kết tuân thủ:** Không xóa hàng loạt hay thay đổi nhãn của 101 ảnh này khi chưa có phương án bù đắp mẫu thay thế và sự phê duyệt chính thức từ Chủ dự án.

---

## 6. P1 — HÌNH HỌC CẮT TỌA ĐỘ TACO (CLIPPING GEOMETRY PROOF)

### A. Bản chất toán học của tọa độ âm $y = -1.00$
Trong tệp nhãn COCO gốc của TACO ([`data/audit/taco_annotations_raw.json`](file:///D:/CNTT-KLCN155-waste-detection/data/audit/taco_annotations_raw.json)):
- `taco_0112` (Image 112, Annotation 387, Kích thước $1824 \times 1824$): COCO bbox = `[1225.0, -1.0, 107.0, 92.0]`.
- `taco_0457` (Image 457, Annotation 1621, Kích thước $4160 \times 4160$): COCO bbox = `[1167.0, -1.0, 246.0, 269.0]`.

Tọa độ COCO định nghĩa `[x, y, w, h]` với đỉnh trên-trái là $(x, y)$, đỉnh dưới-phải là $(x + w, y + h)$.  
Khi $y = -1.0$, đỉnh trên-trái nằm ngoài khung ảnh 1 pixel ở phía trên. Đỉnh dưới-phải nằm tại:
$$y_2 = y + h = -1.0 + 92.0 = 91.0 \text{ pixels}$$

### B. Chứng minh hình học của hàm `coco_to_yolo`
Hàm chuyển đổi trong [`scripts/collect_real_detection_v2.py`](file:///D:/CNTT-KLCN155-waste-detection/scripts/collect_real_detection_v2.py):
```python
def coco_to_yolo(bbox, W, H):
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)
```

1. **Đỉnh trên được cắt vào mép ảnh:** $y_1' = \max(0.0, -1.0) = 0.0$.
2. **Đỉnh dưới được giữ nguyên vẹn:** $y_2' = \min(H, -1.0 + 92.0) = 91.0$.
3. **Chiều cao sau khi cắt giảm đúng 1 pixel:**
   $$h' = y_2' - y_1' = 91.0 - 0.0 = 91.0 \text{ pixels (giảm đúng 1 pixel không tồn tại trên canvas)}$$
4. **Tâm đối tượng được tính lại chính xác:**
   $$y_c' = \frac{y_1' + y_2'}{2} = \frac{0.0 + 91.0}{2} = 45.5 \text{ pixels}$$
5. **So sánh với phương pháp kẹp ngây thơ (Naive Clamping):**
   - Nếu ai đó chỉ gán $y_{new} = \max(0.0, y) = 0.0$ và giữ nguyên chiều cao $h = 92.0$:
     Đáy của hộp sẽ trở thành $y_{bottom} = 0.0 + 92.0 = 92.0 \text{ pixels}$.
     Điều này khiến **toàn bộ bounding box bị trôi dịch chuyển xuống dưới 1 pixel**, lấn quá ranh giới thực của vật thể rác!
   - Hàm `coco_to_yolo` của dự án đã tính lại $(y_2 - y_1)$, đảm bảo đáy hộp tái lập từ YOLO $(y_c + h/2) \times H$ đúng bằng $91.0$ pixels, bảo toàn 100% hình học vật thể.

Bộ test hồi quy toán học tại [`tests/detection/test_taco_clipping_geometry.py`](file:///D:/CNTT-KLCN155-waste-detection/tests/detection/test_taco_clipping_geometry.py) đã chạy và đạt **5/5 PASSED**.

---

## 7. P2 — THẨM ĐỊNH GROUND TRUTH VÀ 6 ĐỐI TƯỢNG TRANH CHẤP

> **Nguyên tắc cốt lõi:** Tuyệt đối không tự tiện đóng băng GT v2 khi chưa có sự phê duyệt chính thức từ Chủ dự án. Tuyệt đối không suy đoán thành phần rác bên trong các túi buộc kín chỉ bằng hình ảnh bên ngoài.

### Chi tiết 6 đối tượng tranh chấp và phương án trình duyệt:

1. **`images (12).jpg` (`img06_obj01` vs `img06_obj02`):**
   - Bản chất: Thùng bìa carton mở nắp đặt trên sàn, bên trong chứa các tập hồ sơ giấy.
   - Box `img06_obj01`: `cardboard` (79.16% diện tích, bọc ngoài). Box `img06_obj02`: `paper` (42.22% diện tích, bên trong).
   - Đề xuất: Giữ quan hệ cha-con (Container `cardboard` chứa `paper`). Nếu detector phát hiện cả 2 box thì cả 2 đều hợp lệ.
2. **`images (10).jpg` (`img04_obj01` vs `img04_obj02`):**
   - Bản chất: Đống quần áo vải dệt bị bỏ lại. Box lớn (`img04_obj01`, 59.8% diện tích) và box nhỏ (`img04_obj02`, 22.3% diện tích) đều cùng gán nhãn `clothes`.
   - Đề xuất: Hợp nhất thành 1 box `clothes` duy nhất bao quát toàn bộ đống rác vải, tránh việc phạt oan 1 FN khi detector chỉ dự đoán 1 box hợp lý.
3. **`OIP (1).webp` (`img20_obj01` vs `img20_obj02`):**
   - Bản chất: Hai gò quần áo trên sườn sa mạc Atacama có phần giao nhau ở sườn trái.
   - Đề xuất: Cắt tỉa ranh giới rõ ràng: Box 1 cho gò tiền cảnh bên phải, Box 2 cho gò hậu cảnh bên trái.
4. **`Phan-Loai-Rac-Tai-Ng-01.jpg` (`img22_obj04`):**
   - Bản chất: Vỏ hộp sữa TH true MILK tiệt trùng (Tetra Pak). Cấu tạo phức hợp gồm 75% giấy bìa cứng, 20% màng PE và 5% lá nhôm.
   - Đề xuất: Thống nhất chuẩn hóa nhãn là `cardboard` theo quy ước GLOSSARY.md (vật liệu định hình cứng chính).
5. & 6. **`Phan-Loai-Rac-Tai-Ng-01.jpg` (`img22_obj01` và `img22_obj02`):**
   - Bản chất: Bao tải dứa xanh buộc túm (`img22_obj01`, 10.98%) và túi nilon đỏ buộc kín miệng (`img22_obj02`, 12.60%).
   - **Xác minh không suy đoán:** Hai túi này được buộc kín hoàn toàn, camera chỉ ghi nhận được màng nhựa/bao dệt bên ngoài. Việc khẳng định bên trong chứa thức ăn thừa hay giấy vụn là suy đoán vô căn cứ.
   - **Hai phương án lựa chọn cho Chủ dự án:**
     - **Phương án A (Theo hình thức thị giác thuần túy - Visual Material):** Gán nhãn `plastic` (vỏ bao bì nhựa nhìn thấy được).
     - **Phương án B (Theo nghiệp vụ thu gom đô thị - Waste Management Handling):** Gán nhãn `trash` (bao rác sinh hoạt hỗn hợp, không phân loại tái chế màng film).

Tất cả 27 ảnh overlay kiểm duyệt đã được kết xuất sẵn sàng tại [`reports/qa_acceptance/gt_review_overlays/`](file:///D:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/gt_review_overlays/).

---

## 8. P2 — BẢNG ĐIỀU KIỆN SẴN SÀNG HUẤN LUYỆN (TRAINING READINESS CHECKLIST)

| Tiêu chuẩn điều kiện | Mục tiêu nghiệm thu | Bằng chứng kiểm toán | Trạng thái kỹ thuật |
|---|---|---|:---:|
| **1. Cấu trúc 10 Lớp** | Taxonomy cố định 10 lớp `[0..9]` khớp giữa dataset, SSDLite, YOLO và Streamlit | `tests/detection/test_mapping_regression.py` | **VERIFIED** |
| **2. Tọa độ & Geometry** | Chuẩn hóa xyxy/xywh, clip biên bảo toàn hình học, chuyển đổi xuôi ngược nhất quán | `tests/detection/test_taco_clipping_geometry.py`, `test_coordinates.py` | **VERIFIED** |
| **3. Cô lập Rò rỉ (Leakage)** | 0% trùng lặp SHA-256, 0% rò rỉ Train-Val-Test, 0% rò rỉ Holdout | `reports/DATASET_QUALITY_AND_LEAKAGE_REPORT.json` | **VERIFIED** |
| **4. Checkpoint Baseline** | Bảo toàn nguyên vẹn checkpoint SSDLite 35 epochs và YOLOv8n E1, không ghi đè | `tests/qa/test_compare_eval_v1_vs_v2.py` | **VERIFIED** |
| **5. Tài nguyên Phần cứng** | NVIDIA RTX 2050 4GB VRAM, CUDA 12.1, PyTorch 2.5.1, RAM 16GB | `reports/test_evidence/python_environment.json` | **VERIFIED** |
| **6. Giao thức Đánh giá GT v2** | Thống nhất chính sách: Thùng rác=Negative, Đống rác=Excluded/Ignore-Region | `scripts/compare_eval_v1_vs_v2.py`, `test_compare_eval_v1_vs_v2.py` | **NEEDS HUMAN APPROVAL** |
| **7. Phê duyệt 6 Vật thể GT** | Quyết định gán nhãn cho thùng carton lồng giấy, đống quần áo và túi rác buộc kín | `reports/GT_V2_REVIEW_FOR_APPROVAL.md` | **NEEDS HUMAN APPROVAL** |
| **8. Bổ sung Hard-Negatives** | Bổ sung 150-200 ảnh nền sạch (vỉa hè, mặt đường, tủ điện) để kiểm chứng giả thuyết FP | Hiện tại 0 ảnh nền sạch trong tập Train | **BLOCKED** |
| **9. Cân bằng Dữ liệu Train** | Bổ sung mẫu thực tế cho `battery` (1 box) và `biological` (3 boxes) đạt tối thiểu 80+ boxes | Tập train hiện hữu quá đói dữ liệu | **BLOCKED** |

---

## 9. ĐÓNG GÓI HỒ SƠ BẰNG CHỨNG (DELIVERABLE ARCHIVE)

Đã đóng gói tệp nén tinh gọn, độc lập tại:  
[`C:\Users\ad\Downloads\ANTI_PHASE3_FINAL_VERIFICATION_EVIDENCE.zip`](file:///C:/Users/ad/Downloads/ANTI_PHASE3_FINAL_VERIFICATION_EVIDENCE.zip)

### Danh mục tệp trong gói ZIP:
1. `00_ANTI_PHASE3_FINAL_VERIFICATION.md`: Bản báo cáo này.
2. `01_GT_V2_REVIEW_FOR_APPROVAL.md`: Tài liệu danh mục 57 đối tượng trình duyệt.
3. `diffs/patch_oi_2734e597675d8a97.diff`: Unified diff bản vá dataset.
4. `diffs/patch_oi_2734e597675d8a97_evidence.json`: Bằng chứng mã băm SHA-256 trước/sau.
5. `test_evidence/pytest_collection.log`: Nhật ký 311 tests collected nguyên bản.
6. `test_evidence/pytest_targeted.log`: Nhật ký 132 passed / 1 skipped nguyên bản.
7. `test_evidence/python_environment.json`: Cấu hình môi trường Python, CUDA, GPU.
8. `test_evidence/tests_summary.json`: Bản tổng hợp máy đọc.
9. `data_audit/DATASET_QUALITY_AND_LEAKAGE_REPORT.json`: Báo cáo rò rỉ và phân bố 10 lớp.
10. `data_audit/DATASET_INTEGRITY_AUDIT_REPORT.json`: Nhật ký quét 5,584 boxes.
11. `evaluation/verified_evaluation_summary_from_raw.json`: Kết quả đánh giá đa chính sách.
12. `tests/`: 3 tệp test hồi quy cốt lõi (`test_compare_eval_v1_vs_v2.py`, `test_taco_clipping_geometry.py`, `test_ground_truth_v2_integrity.py`).
13. `gt_review_overlays/`: 4 ảnh overlay tiêu biểu cho 6 đối tượng tranh chấp (`images (10)`, `images (12)`, `OIP (1)`, `Phan-Loai-Rac-Tai-Ng-01`).
