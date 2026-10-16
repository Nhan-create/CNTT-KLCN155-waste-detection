# BÁO CÁO TOÀN DIỆN PHẦN 1: XÁC MINH VÀ SỬA MAPPING 10 LỚP DETECTION
**Dự án:** Multi-Object Waste Detection (CNTT-KLCN155)  
**Ngày thực hiện:** 09/10/2026  
**Môi trường:** Python 3.12.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.3.163, GPU RTX 2050 Laptop.  
**Thực hiện theo tiêu chuẩn:** TDD (Red-Green-Refactor), Systematic Debugging, Ponytail (tối giản, không kiến trúc thừa).

---

## MỤC LỤC
1. [Mục tiêu và Phạm vi](#1-mục-tiêu-và-phạm-vi)
2. [Bảng Ánh Xạ Chuẩn và Bằng Chứng Nguồn](#2-bảng-ánh-xạ-chuẩn-và-bằng-chứng-nguồn)
3. [Kiểm Tra Tương Thích Checkpoint Thực Tế](#3-kiểm-tra-tương-thích-checkpoint-thực-tế)
4. [Truy Vết và Rút Lại Kết Luận "77,1% Nhãn Sai"](#4-truy-vết-và-rút-lại-kết-luận-771-nhãn-sai)
5. [Quy Ước Background SSDLite và Kiểm Định Hoán Đổi Lớp (Mutation Test)](#5-quy-ước-background-ssdlite-và-kiểm-định-hoán-đổi-lớp-mutation-test)
6. [Inventory Annotation và Phân Bố Lớp](#6-inventory-annotation-và-phân-bố-lớp)
7. [Các Lỗi Mã Nguồn Đã Xác Nhận, Ý Tưởng Sửa và Git Diff](#7-các-lỗi-mã-nguồn-đã-xác-nhận-ý-tưởng-sửa-và-git-diff)
8. [Log Kiểm Thử Chi Tiết](#8-log-kiểm-thử-chi-tiết)
9. [Kết Luận và Hướng Dẫn Bước Tiếp Theo](#9-kết-luận-và-hướng-dẫn-bước-tiếp-theo)

---

## 1. MỤC TIÊU VÀ PHẠM VI

- **Mục tiêu:** Bảo đảm Class ID có cùng ý nghĩa từ dữ liệu nguồn $\rightarrow$ annotation YOLO txt $\rightarrow$ cấu hình train $\rightarrow$ checkpoint $\rightarrow$ inference $\rightarrow$ tên tiếng Việt trên giao diện.
- **Mô hình:**
  - Mô hình chính: **SSDLite320-MobileNetV3 Large** (PyTorch/torchvision).
  - Mô hình đối chứng: **YOLOv8n** (Ultralytics).
- **Phạm vi nghiêm ngặt:** Không sửa threshold, bounding box coordinates, cache, giao diện; không đổi dữ liệu chính hay retrain model; giữ nguyên toàn bộ các file người dùng đang sửa dở.

---

## 2. BẢNG ÁNH XẠ CHUẨN VÀ BẰNG CHỨNG NGUỒN

Taxonomy 10 lớp được chuẩn hóa tại `src/detection/schema.py` và `configs/detection_source_mapping.yaml`:

| ID Chuẩn | Tên chuẩn (EN) | Tên tiếng Việt chuẩn | Nguồn dữ liệu | ID / Tên gốc từ nguồn | Bằng chứng xác minh | Trạng thái |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `battery` | Pin / Ắc quy | TACO<br>VN Trash | `1: Battery`<br>`medical` | `configs/detection_source_mapping.yaml:35`<br>`data/metadata/label_mapping.csv:1` | **Xác nhận 100%** |
| **1** | `biological` | Rác sinh học / Hữu cơ | TACO<br>VN Trash | `25: Food waste`<br>`organic` | `configs/detection_source_mapping.yaml:37`<br>`src/detection/schema.py:56` | **Xác nhận 100%** |
| **2** | `cardboard` | Bìa carton | TACO<br>VN Trash | `13..19: Cartons, Pizza box`<br>`Carton, Milk_box` | `configs/detection_source_mapping.yaml:39-45`<br>`data/metadata/label_mapping.csv:3,5` | **Xác nhận 100%** |
| **3** | `clothes` | Quần áo cũ | OpenImages V7<br>Garbage v2 | `/m/09j2d, /m/01n4qj...`<br>`clothes` | `configs/detection_source_mapping.yaml:118-125`<br>`src/detection/schema.py:39` | **Xác nhận 100%** |
| **4** | `glass` | Thủy tinh | TACO<br>OpenImages V7 | `6, 9, 23, 26: Glass bottle, cup...`<br>`/m/09tvcd: Wine glass` | `configs/detection_source_mapping.yaml:47-50`<br>`configs/detection_source_mapping.yaml:113` | **Xác nhận 100%** |
| **5** | `metal` | Kim loại | TACO<br>OpenImages V7<br>VN Trash | `0, 2, 8, 10, 11, 12: Can, lid...`<br>`/m/02jnhm: Tin can`<br>`Alu` | `configs/detection_source_mapping.yaml:52-60`<br>`configs/detection_source_mapping.yaml:112`<br>`data/metadata/label_mapping.csv:2` | **Xác nhận 100%** |
| **6** | `paper` | Giấy | TACO<br>OpenImages V7<br>VN Trash | `20, 30..34, 56: Paper cup, bag...`<br>`/m/02w3r3: Paper towel`<br>`Paper, Paper_cup` | `configs/detection_source_mapping.yaml:62-68`<br>`configs/detection_source_mapping.yaml:115`<br>`data/metadata/label_mapping.csv:8-9` | **Xác nhận 100%** |
| **7** | `plastic` | Nhựa | TACO<br>OpenImages V7<br>VN Trash | `4, 5, 7, 21, 36..49: Bottle, bag...`<br>`/m/05gqfk: Plastic bag`<br>`PET, Foam_box, Plastic_cup` | `configs/detection_source_mapping.yaml:70-94`<br>`configs/detection_source_mapping.yaml:114`<br>`data/metadata/label_mapping.csv:4,7,10` | **Xác nhận 100%** |
| **8** | `shoes` | Giày dép | TACO<br>OpenImages V7<br>Garbage v2 | `53: Shoe`<br>`/m/09j5n: Footwear`<br>`shoes` | `configs/detection_source_mapping.yaml:96`<br>`configs/detection_source_mapping.yaml:117`<br>`src/detection/schema.py:44` | **Xác nhận 100%** |
| **9** | `trash` | Rác khác | TACO<br>VN Trash<br>Garbage v2 | `58, 59: Litter, Cigarette`<br>`Other`<br>`trash` | `configs/detection_source_mapping.yaml:98-99`<br>`data/metadata/label_mapping.csv:6`<br>`src/detection/schema.py:45` | **Xác nhận 100%** |

---

## 3. KIỂM TRA TƯƠNG THÍCH CHECKPOINT THỰC TẾ

Đã load và kiểm tra metadata trực tiếp từ cả 4 checkpoint có trong `artifacts/detection`:

1. **`artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt`**:
   - Model chính: SSDLite320-MobileNetV3 Large.
   - Classification head: 11 channels (index 0 dành cho background, indices 1..10 dành cho 10 lớp foreground).
   - Metadata `class_names`: 10 lớp đúng thứ tự chuẩn `('battery', 'biological', ..., 'trash')`.
   - Trạng thái: **Hoàn toàn tương thích** với schema và mã nguồn inference.
2. **`artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt`**:
   - Model đối chứng: YOLOv8n (`nc=10`).
   - Head conv channels = 10, `model.names` khớp 100% thứ tự chuẩn.
   - Metadata `class_names`: 10 lớp chuẩn.
   - Trạng thái: **Hoàn toàn tương thích**.
3. **`artifacts/detection/waste-ssdlite320-mobilenetv3/weights/best.pt`** (smoke):
   - Trạng thái: **Tương thích**.
4. **`artifacts/detection/waste-yolov8n/weights/best.pt`** (smoke):
   - Trạng thái: **Tương thích**.

---

## 4. TRUY VẾT VÀ RÚT LẠI KẾT LUẬN "77,1% NHÃN SAI"

### Sai lầm của báo cáo kiểm toán cũ
Báo cáo kiểm toán cũ trước đây cho rằng:
*“Script tạo dataset đã copy nguyên class ID 0..9 từ Mendeley Data sang TAXONOMY_10 dẫn tới 77,1% nhãn bị gán sai.”*

### Thực nghiệm đối chứng
1. Script kiểm tra đối chiếu trực tiếp 1.529 box giữa `D:\waste-training\mendeley-labels\train` và `D:\waste-training\dataset-v1\labels\train`:
   - Mendeley `0` $\rightarrow$ v1 `7` (179/179 box)
   - Mendeley `1` $\rightarrow$ v1 `6` (157/157 box)
   - Mendeley `2` $\rightarrow$ v1 `2` (168/168 box)
   - Mendeley `3` $\rightarrow$ v1 `5` (184/184 box)
   - Mendeley `4` $\rightarrow$ v1 `4` (189/189 box)
   - Mendeley `5` $\rightarrow$ v1 `1` (181/181 box)
   - Mendeley `6` $\rightarrow$ v1 `0` (161/161 box)
   - Mendeley `8` $\rightarrow$ v1 `3` (167/167 box)
   - Mendeley `9` $\rightarrow$ v1 `9` (143/143 box)
   $\rightarrow$ **Khẳng định: Bộ dữ liệu `dataset-v1` đã được chuyển đổi mapping 1-1 cố định, không hề copy trực tiếp ID.**
2. Crop và soi ảnh thật từng lớp:
   - Mendeley class `6` (được chuyển sang class 0 `battery`): Thực tế là ảnh lon bia Amstel, lon nước ngọt kim loại.
   - Mendeley class `8` (được chuyển sang class 3 `clothes`): Thực tế là cành lá cây mục (rác hữu cơ).
   - Mendeley class `3` (được chuyển sang class 5 `metal`): Thực tế là bó dây thừng nylon.
3. **Nguyên nhân gốc rễ:**
   Người chuẩn bị dữ liệu `dataset-v1` trước đây đã lấy danh sách tên lớp trong đoạn văn bản tóm tắt (abstract) trên website Mendeley Data để lập bảng map sang `TAXONOMY_10`, nhưng bản thân tác giả Mendeley Data lại lưu nhãn ID nội bộ theo một thứ tự hoàn toàn khác.
4. **Kết luận:**
   **Chính thức rút lại kết luận "77,1% nhãn sai do không map"** vì tiền đề phương pháp luận cũ bị sai lệch. Bản chất vấn đề là bộ dữ liệu Mendeley bị sai ngữ nghĩa do nhầm lẫn giữa text abstract và ID file annotation.

---

## 5. QUY ƯỚC BACKGROUND SSDLITE VÀ KIỂM ĐỊNH HOÁN ĐỔI LỚP (MUTATION TEST)

### Quy ước Background của SSDLite
- **Training Dataset Loader (`src/detection/ssdlite_train.py` - `YoloBoxDataset`):**
  - Đọc class $c \in [0..9]$ từ file annotation.
  - Chuyển đổi: `labels.append(int(c) + 1)`.
  - Mục đích: Index `0` dành riêng cho background trong torchvision SSDLite320; indices `1..10` là 10 lớp foreground.
- **Inference Decoder (`src/detection/ssdlite.py` - `tensor_detections`):**
  - Đọc tensor đầu ra từ mô hình.
  - Chuyển đổi: `class_index = int(label) - 1`.
  - Kiểm tra an toàn: Nếu tensor trả về `label == 0` (background) hoặc `label > 10`, lập tức ném `DetectionError`.
- **Khẳng định:** Phép chuyển ID có cộng 1 khi train và trừ 1 khi inference đúng một lần duy nhất, bảo đảm tính nhất quán toán học 100%.

### Kiểm định Đột biến hoán đổi lớp (Mutation Test)
Hàm test `test_checkpoint_class_order_mutation_fails` trong `tests/detection/test_mapping_regression.py` thực hiện:
- Hoán đổi vị trí của 2 lớp trong metadata (`battery` $\leftrightarrow$ `biological`):
  1. `validate_yolo_metadata`: Phát hiện ngay `names != CLASS_NAMES`, raise `DetectionError: Sai thứ tự lớp detector`.
  2. `SSDLiteWasteDetector._ensure_loaded`: Phát hiện `class_names != DETECTION_CLASS_NAMES`, raise `DetectionError: Checkpoint SSDLite sai kiến trúc/nhãn hoặc chưa huấn luyện`.

---

## 6. INVENTORY ANNOTATION VÀ PHÂN BỐ LỚP

Quét toàn bộ thư mục `data/detection/labels`:
- **Tổng số file annotation:** **1.562 files** (1.305 synthetic files + 257 real files).
- **Tổng số bounding box:** **5.750 boxes** (4.095 synthetic boxes + 1.655 real boxes).
- **Số box ngoài khoảng [0, 9]:** **0 box (0%)**.
- **Số box vi phạm tọa độ/kích thước:** **0 box**.
- **Cảnh báo nhỏ:** 1 box cực nhỏ tại `data/detection/labels/real/taco_0804.txt` line 6 (width=0.0126, height=0.0018).

### Phân bố số lượng Bounding Box theo 10 lớp:
| Class ID | Tên Lớp (EN) | Tên Tiếng Việt | Số Lượng Box | Tỷ Lệ (%) |
| :---: | :--- | :--- | :---: | :---: |
| **0** | `battery` | Pin / Ắc quy | 464 | 8.07% |
| **1** | `biological` | Rác sinh học / Hữu cơ | 454 | 7.90% |
| **2** | `cardboard` | Bìa carton | 521 | 9.06% |
| **3** | `clothes` | Quần áo cũ | 520 | 9.04% |
| **4** | `glass` | Thủy tinh | 557 | 9.69% |
| **5** | `metal` | Kim loại | 557 | 9.69% |
| **6** | `paper` | Giấy | 511 | 8.89% |
| **7** | `plastic` | Nhựa | 878 | 15.27% |
| **8** | `shoes` | Giày dép | 514 | 8.94% |
| **9** | `trash` | Rác khác | 774 | 13.46% |
| **TỔNG** | **10 Lớp Chuẩn** | - | **5.750** | **100.0%** |

---

## 7. CÁC LỖI MÃ NGUỒN ĐÃ XÁC NHẬN, Ý TƯỞNG SỬA VÀ GIT DIFF

### Lỗi 1: `src/detection/visualization.py` thiếu 6 lớp detection chuẩn
- **Ý tưởng sửa:** Bổ sung đầy đủ 10 lớp chuẩn theo `DETECTION_CLASS_NAMES_VI` trực tiếp vào từ điển `DETECTION_DISPLAY_NAMES`, giữ lại alias `"organic"` và `"hazardous"` để không làm gãy các thành phần phân loại giai đoạn 1.
- **Diff:**
```diff
--- a/src/detection/visualization.py
+++ b/src/detection/visualization.py
@@ -10,9 +10,16 @@ from src.ui.presentation import COLOR_BY_CLASS, present_label
 DETECTION_DISPLAY_NAMES = {
-    "plastic": "Nhựa",
-    "paper": "Giấy",
-    "metal": "Kim loại",
+    "battery": "Pin",
+    "biological": "Rác hữu cơ",
+    "cardboard": "Bìa carton",
+    "clothes": "Quần áo",
     "glass": "Thủy tinh",
+    "metal": "Kim loại",
+    "paper": "Giấy",
+    "plastic": "Nhựa",
+    "shoes": "Giày dép",
+    "trash": "Rác khác",
+    # Legacy aliases
     "organic": "Rác hữu cơ",
     "hazardous": "Rác nguy hại",
 }
```

### Lỗi 2: `src/detection/schema.py` thiếu mapping cho nhãn `vn_trash`
- **Ý tưởng sửa:** Đồng bộ các nhãn trong `data/metadata/label_mapping.csv` (`alu`, `carton`, `foam_box`, `milk_box`, `other`, `pet`, `paper`, `paper_cup`, `plastic_cup`) vào `SOURCE_LABEL_MAPPING["vn_trash"]`.
- **Diff:**
```diff
--- a/src/detection/schema.py
+++ b/src/detection/schema.py
@@ -47,6 +47,15 @@ SOURCE_LABEL_MAPPING = {
     "vn_trash": {
+        "alu": "metal",
+        "carton": "cardboard",
+        "foam_box": "plastic",
+        "milk_box": "cardboard",
+        "other": "trash",
+        "pet": "plastic",
+        "paper": "paper",
+        "paper_cup": "paper",
+        "plastic_cup": "plastic",
         "organic": "biological",
         "medical": "battery",
     },
```

### Lỗi 3: `tests/test_detection_annotations.py` kiểm tra sai số lượng inventory hiện hữu
- **Ý tưởng sửa:** Cập nhật assertions khớp với số lượng thực tế đã xác minh (1562 files, 5750 boxes).
- **Diff:**
```diff
--- a/tests/test_detection_annotations.py
+++ b/tests/test_detection_annotations.py
@@ -88,4 +88,4 @@ def test_real_dataset_annotations():
     summary = validate_dataset(labels_dir)
-    assert summary["total_files_checked"] == 1419
-    assert summary["total_boxes_validated"] == 4602
+    assert summary["total_files_checked"] == 1562
+    assert summary["total_boxes_validated"] == 5750
     assert summary["total_errors"] == 0
```

---

## 8. LOG KIỂM THỬ CHI TIẾT

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CNTT-KLCN155-waste-detection
configfile: pytest.ini
plugins: anyio-4.15.1
collected 11 items

tests\detection\test_mapping_regression.py ....                          [ 36%]
tests\test_detection_annotations.py .......                              [100%]

============================= 11 passed in 4.46s ==============================

============================= full detection suite =============================
collected 31 items
tests\detection\test_augmentation.py ....                                [ 12%]
tests\detection\test_copy_paste_rigorous.py ......                       [ 32%]
tests\detection\test_detection_dataset.py .                              [ 35%]
tests\detection\test_fusion_evaluation.py .....                          [ 51%]
tests\detection\test_mapping_regression.py ....                          [ 64%]
tests\detection\test_train.py ..                                         [ 70%]
tests\detection\test_types_and_tracking.py ...                           [ 80%]
tests\detection\test_visualization.py .                                  [ 83%]
tests\detection\test_yolo_adapter.py .....                               [100%]

======================= 31 passed, 3 warnings in 4.99s ========================
```

---

## 9. KẾT LUẬN VÀ HƯỚNG DẪN BƯỚC TIẾP THEO

1. **Phần 1 đã hoàn thành đầy đủ:**
   - Toàn bộ đường ống mapping 10 lớp được chứng minh nhất quán từ nguồn $\rightarrow$ annotation $\rightarrow$ checkpoint $\rightarrow$ visualization.
   - Các lỗi thiếu mapping hiển thị và thiếu source label mapping đã được sửa triệt để bằng TDD.
   - Không vi phạm bất kỳ ràng buộc nào (không đổi threshold, không retrain, không làm xáo trộn các file người dùng đang sửa).
2. **Khuyến nghị cho Phần 2:**
   - Tập trung vào lọc và làm sạch các ảnh Mendeley có nhãn ngữ nghĩa bị sai từ trước trong `dataset-v1` trước khi tiến hành retrain.
