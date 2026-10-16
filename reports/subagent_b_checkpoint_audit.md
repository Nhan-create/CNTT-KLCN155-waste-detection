# BÁO CÁO KIỂM ĐỊNH CHECKPOINT VÀ SUY LUẬN (SUB-AGENT B)
**Dự án:** CNTT-KLCN155 - Hệ thống phát hiện rác thải đa nguồn  
**Chuyên trách:** Sub-Agent B (Model / Inference Specialist)  
**Ngày thực hiện:** 09/10/2026  
**Môi trường thực thi:** Python 3.12.10, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, Ultralytics 8.4.171, CUDA: NVIDIA GeForce RTX 2050 (4GB)

---

## 1. Kết quả kiểm tra 4 Checkpoint trong `artifacts/detection/`

| STT | Checkpoint Path | Kích thước (bytes) | SHA-256 Hash | Phân loại & Số channels Head | Trạng thái |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | `artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt` | **28,456,982** | `584c5980eb4ebe3718da7b3e87a3c95edfee3d78f186def3e5feba32cf03dff1` | **SSDLite320 (Chính)**: 11 channels (Background=0, 10 Foreground classes 1..10). Head có 6 levels x 11 = 66 outputs/anchor. | **HỢP LỆ (Mô hình chính thức)** |
| 2 | `artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt` | **6,217,379** | `97f46ccaaa0a63c5934225d51ccd399c1c18eb3e1af8a63b9f14bec1aaa085c9` | **YOLOv8n (Đối chứng)**: 10 channels (10 classes 0..9). cv3[0..2] out_conv weight shape `[10, 64, 1, 1]`. | **HỢP LỆ (Mô hình chính thức)** |
| 3 | `artifacts/detection/waste-ssdlite320-mobilenetv3/weights/best.pt` | **28,456,918** | `c9a079d630bdff7919d476c960784af885662fd12fa28fee7c8fef03bd551c9b` | **SSDLite320 (Smoke Run)**: 11 channels (Background=0, 10 Foreground classes 1..10). | **SMOKE RUN (2 Epochs)** |
| 4 | `artifacts/detection/waste-yolov8n/weights/best.pt` | **6,208,931** | `84b677fcb4de6cb0725973aa7de21f0a594dfea57c20c875256f393fff2a06dd` | **YOLOv8n (Smoke Run)**: 10 channels (10 classes 0..9). | **SMOKE RUN (2 Epochs)** |

---

## 2. Phân tích Provenance & Training Logs Thực tế

### 2.1. Mô hình SSDLite320-MobileNetV3 Combined Seed42 (`artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42`)
- **Tập dữ liệu & Fingerprint:** `configs/detection_dataset.yaml`, SHA-256 fingerprint: `7d452da6a90d9ce248bafc5bf0a9638fd09334bb247b8ec909fdaabf9010ca16`.
- **Cấu hình huấn luyện (Budget):** 120 epochs, batch size 8, imgsz 320, AdamW (lr0=0.001, lrf=0.01, weight_decay=0.0005), patience=30.
- **Số epoch thực tế hoàn thành:** **59 epochs** (Dừng sớm do early stopping sau 30 epochs không cải thiện).
- **Best Epoch:** **Epoch 29**.
- **Chỉ số Validation tốt nhất:**
  - `mAP50-95`: **0.04888** (4.89%)
  - `mAP50`: **0.08650** (8.65%)
  - AP từng lớp tại best epoch:
    * `cardboard`: AP50-95 = 0.3713, AP50 = 0.6425
    * `plastic`: AP50-95 = 0.0198, AP50 = 0.0495
    * Các lớp khác: 0.0 hoặc không có instance trong tập validation 5 ảnh.
  - Training loss tại best epoch: **2.8797** (Loss epoch 1: 8.7297, loss epoch 59: 2.6485).
- **Thời gian thực thi:** 4,149.67 giây (~69.2 phút) trên GPU NVIDIA RTX 2050 (Peak VRAM: 481 MB).
- **File nhật ký gốc kèm theo:** `training_summary.json`, `run_metadata.json`, `history.json`.

### 2.2. Mô hình YOLOv8n Combined Seed42 (`artifacts/detection/waste-yolov8n-combined-seed42`)
- **Tập dữ liệu & Fingerprint:** `configs/detection_dataset.yaml`, SHA-256 fingerprint: `7d452da6a90d9ce248bafc5bf0a9638fd09334bb247b8ec909fdaabf9010ca16`.
- **Cấu hình huấn luyện (Budget):** 120 epochs, batch size 16, imgsz 320, AdamW (lr0=0.001, lrf=0.01, weight_decay=0.0005), patience=30.
- **Số epoch thực tế hoàn thành:** **67 epochs** (Dừng sớm do early stopping sau 30 epochs không cải thiện).
- **Best Epoch:** **Epoch 37** (theo `results.csv`: mAP50-95 = 0.04456, mAP50 = 0.05776, val/box_loss = 1.9356, val/cls_loss = 4.4725).
- **Chỉ số Validation sau huấn luyện (`model.val()` trên `best.pt`):**
  - `mAP50-95(B)`: **0.04014** (4.01%)
  - `mAP50(B)`: **0.05251** (5.25%)
  - `Precision(B)`: **0.20122** (20.12%)
  - `Recall(B)`: **0.05398** (5.40%)
  - Khuyến nghị ngưỡng tối ưu F1 từ validation: `confidence = 0.0470`.
- **Thời gian thực thi:** 846.0 giây (~14.1 phút) trên GPU NVIDIA RTX 2050.
- **File nhật ký gốc kèm theo:** `training_summary.json`, `run_metadata.json`, `results.csv`, `effective_train_args.json`, `args.yaml`.

### 2.3. Hai Checkpoint Smoke Run (`waste-ssdlite320-mobilenetv3` & `waste-yolov8n`)
- **Nguồn gốc (Provenance):** Hai checkpoint này nằm trong thư mục thiếu file logs rời (`MISSING training_summary.json, history.json, results.csv`). Tuy nhiên, kiểm tra trường `metadata` (SSDLite) và `phase2_metadata` (YOLO) được đóng gói bên trong file checkpoint `.pt` cho thấy:
  + Cả hai được khởi tạo từ dự án smoke test `artifacts/detection_smoke`, run name `smoke-ssdlite320` và `smoke-yolov8n`.
  + Cấu hình `epochs: 2` (Smoke test).
  + Đã hoàn thành đúng 2 epochs vào ngày 02/10/2026.
  + SSDLite smoke: validation `mAP50-95 = 0.00728`, `mAP50 = 0.01354`.
  + YOLOv8n smoke: validation mAP sơ bộ chưa hội tụ, `best_fitness = None`.

---

## 3. Rà soát Quy ước Mã nguồn Suy luận và Tiền xử lý

### 3.1. Quy ước chỉ số nhãn (Class Mapping)
1. **SSDLite (`src/detection/ssdlite.py` & `src/detection/ssdlite_train.py`):**
   - *Tập dữ liệu nạp vào:* File YOLO nhãn `0..9` được nạp trong `YoloBoxDataset.__getitem__` qua lệnh:
     `labels.append(int(c) + 1)` -> Index `0` được bảo lưu làm `background`, nhãn Foreground trở thành `1..10`.
   - *Khởi tạo mô hình:* `ssdlite320_mobilenet_v3_large(num_classes=11)`.
   - *Bộ giải mã (Decoder `tensor_detections`):*
     Lấy nhãn dự đoán `label` (thuộc `1..10`), kiểm tra tính hợp lệ `1 <= int(label) <= 10`, sau đó giải mã về nhãn chuẩn:
     `class_index = int(label) - 1` (khoảng `0..9`) -> Ánh xạ chính xác vào `DETECTION_CLASS_NAMES[class_index]`.
   - *Kết luận:* Hoàn toàn khớp với Ràng buộc Bất biến số 2 & 3 trong `AGENTS.md`.

2. **YOLOv8n (`src/detection/yolo.py`):**
   - Head phân loại gồm 10 channels.
   - Nhãn dự đoán `0..9` được map trực tiếp `CLASS_NAMES[class_index]`.

### 3.2. Tiền xử lý & Chuẩn hóa (Preprocessing)
1. **SSDLite:**
   - Đầu vào: Ảnh PIL được xử lý EXIF qua `ImageOps.exif_transpose(image).convert("RGB")`.
   - Tensor chuẩn hóa: `pil_to_tensor(prepared).to(device) / 255.0` (dải `[0.0, 1.0]`).
   - Lớp nội tại của Torchvision `GeneralizedRCNNTransform`:
     * `Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])` đưa dữ liệu về `[-1.0, 1.0]`.
     * `Resize(min_size=(320,), max_size=320, mode='bilinear')` scale ảnh về đúng 320x320.
   - Hậu xử lý: Torchvision tự động unscale bounding boxes về hệ tọa độ pixel gốc của ảnh ban đầu; hàm `box.clamp(width, height)` bảo đảm tọa độ không bị tràn biên.

2. **YOLOv8n:**
   - Đầu vào: `ImageOps.exif_transpose(image).convert("RGB")`, chuyển mảng BGR `np.asarray(prepared)[:, :, ::-1].copy()`.
   - Kích thước suy luận: Hỗ trợ linh hoạt (mặc định 640 qua factory, hoặc 320 khớp với cấu hình training).
   - Ultralytics tự động letterbox padding, chuẩn hóa `1/255.0`, và unscale boxes về tọa độ ảnh gốc.

---

## 4. Công cụ Kiểm tra Suy luận (`scripts/benchmark_inference.py`)

Đã phát triển script [benchmark_inference.py](../scripts/benchmark_inference.py) hỗ trợ đo kiểm suy luận đa nền tảng:
- Chế độ chạy: Hỗ trợ `--device cpu`, `--device cuda` hoặc `--device both`.
- Mô hình: Hỗ trợ `--backend ssdlite`, `--backend yolov8n` hoặc `--backend both`.
- Tùy chỉnh: Ngưỡng `--conf`, `--iou`, kích thước `--limit` và thư mục ảnh `--images-dir` (mặc định `data/holdout`).
- Xuất dữ liệu: Lưu toàn bộ metadata và bounding boxes dạng JSON cấu trúc.

### Kết quả đo kiểm thực tế trên tập ảnh `data/holdout`:
| Mô hình | Phần cứng | Độ trễ trung bình / ảnh (ms) | Tốc độ (FPS) |
|---|:---:|:---:|:---:|
| **SSDLite320-MobileNetV3** | CPU (Intel Core i5 6 threads) | 381.19 ms | **2.62 FPS** |
| **SSDLite320-MobileNetV3** | GPU (NVIDIA RTX 2050 4GB) | 296.94 ms | **3.37 FPS** |
| **YOLOv8n** | CPU (Intel Core i5 6 threads) | 3350.08 ms | **0.30 FPS** |
| **YOLOv8n** | GPU (NVIDIA RTX 2050 4GB) | 508.59 ms | **1.97 FPS** |

*Ghi chú:* SSDLite320 đạt tốc độ vượt trội trên cả CPU và GPU do kiến trúc MobileNetV3 rút gọn Tail và độ phân giải 320x320. Cả hai mô hình sẵn sàng để Agent D sử dụng chạy đánh giá Baseline và nghiệm thu trên 27 ảnh của `data/holdout`.
