# YOLOv8n box anomaly & training readiness (Agent B — Model/Inference)

Date: 2026-10-11 · Device: CPU only (no training run) · Python: `.venv\Scripts\python.exe`

## 1. Kết luận nguyên nhân gốc

**Không phải lỗi tọa độ/pipeline.** Box nhỏ là **hành vi thật của mô hình**, do dữ liệu huấn luyện; cộng thêm một **lệch cấu hình imgsz (640 khi suy luận vs 320 khi huấn luyện)** làm tình hình tệ hơn.

| Giả thuyết | Kết quả | Bằng chứng |
|---|---|---|
| Box ở hệ imgsz không được unscale | ❌ Loại | Wrapper `WasteDetector` vs `ultralytics.predict` thô: chênh lệch **0.0 px** trên tất cả 13 ảnh × 2 imgsz (`yolo_box_anomaly_probe.json`, trường `raw_vs_wrapper_max_abs_px`). Ultralytics trả `boxes.xyxy` đã ở hệ ảnh gốc. |
| Sai xywh/xyxy | ❌ Loại | Trên ảnh train có nhãn, box dự đoán trùng GT: `taco_0082` IoU 0.87–0.99 (6/6 box, đúng lớp); `syn_syn_000003` IoU 0.93–0.98. |
| BGR/RGB | ❌ Loại | Wrapper đảo `[:, :, ::-1]` đúng quy ước Ultralytics (numpy = BGR); tái lập chính xác 33 box của QA (SHA `97f46cca…`) trên CPU. |
| imgsz khác lúc train | ⚠️ **Có, góp phần** | `args.yaml`: `imgsz: 320`; baseline QA & factory dùng 640. Trên 27 ảnh QA: 640 → 0 loc-hit, median diện tích box 0.18% ảnh; 320 → 5 loc-hit (IoU≥0.2), 2 TP, median 0.81% (box lớn ~4.5×). Trên ảnh synthetic train, 640 làm mất/đổi lớp box lớn (`syn_syn_000000`). |
| Mô hình kém thật | ✅ **Nguyên nhân chính** | Xem mục 2. |

Ghi chú nhỏ: mô tả "toàn bộ 33 box ~10–50 px" hơi quá — có 4 box lớn (626×384, 282×178, 268×136, 195×127); 29 box còn lại nhỏ.

## 2. Vì sao mô hình kém (bằng chứng dữ liệu)

1. **Train set = 1305 synthetic + 12 real** (`train-val-train-abs.txt`; `data/detection/splits/train.txt` cũng vậy). Val = 5 ảnh TACO, test = 5 ảnh.
2. **Ảnh synthetic là patch chữ nhật dán lên nền TACO, vật thể thật trên nền không được gán nhãn** (xem `artifacts/detection/waste-yolov8n-combined-seed42/train_batch0.jpg`: `syn_syn_001743` có chai + 2 lon rõ ràng nhưng chỉ một patch nhỏ được gán nhãn). Mô hình học “patch dán = vật, vật thật = nền” → trên ảnh thật chỉ bắn box nhỏ vào vùng có kết cấu lạ.
3. **Val thật cũng thất bại**: `val_batch0_pred.jpg` cho thấy cùng kiểu box nhỏ trên TACO; mAP50 val = 0.053, mAP50-95 = 0.040; dừng ở epoch 67 (patience 30). Chọn checkpoint trên chỉ 5 ảnh val → rất nhiễu.
4. **Không có augmentation trong YOLO**: mọi aug Ultralytics bị ép = 0 (`mosaic/scale/fliplr/hsv…=0`, `train.py` L251-257); chỉ có `AblationAugmentationTransform` (Albumentations flip/affine ±15°/scale 0.9–1.1/ColorJitter/blur + copy-paste). Không có multi-scale → không học được vật lớn chiếm nửa ảnh như QA.
5. Có sẵn **2001 ảnh real có nhãn** (`data/detection/labels/real`: 1358 TACO + 643 OpenImages, 5776 box) nhưng split chỉ dùng 12.

### SSDLite (kiểm tra nhanh)
- Ánh xạ lớp đúng: train `+1` (`ssdlite_train.py` L82), decode `-1` (`ssdlite.py` L116). Trên 40 ảnh synthetic train: 59 cặp khớp IoU≥0.5, hiệu lớp (pred−gt) = 0 chiếm 31/59, **không có đỉnh ở ±1** → không lệch nhãn.
- Tọa độ đúng: torchvision `GeneralizedRCNNTransform` resize 320 và `postprocess` scale ngược về kích thước gốc; box khớp GT.
- Mô hình **underfit**: best val mAP50-95 = 0.049 tại epoch 29/59; QA: 1 TP, 11 wrong-class, 12 loc-hit (IoU≥0.2).

## 3. Thay đổi đã làm (phạm vi B)

- [`src/detection/yolo.py`](../../src/detection/yolo.py): `image_size: int | None = None`; khi load, đọc `ckpt["train_args"]["imgsz"]` (`_checkpoint_train_imgsz`) → dùng làm imgsz suy luận; fallback 640 nếu không có. Thêm `train_image_size`. Giá trị truyền tường minh vẫn được tôn trọng.
- [`src/detection/factory.py`](../../src/detection/factory.py): `image_size` mặc định `None` (không ép 640).
- Test mới [`tests/detection/test_yolo_image_size.py`](../../tests/detection/test_yolo_image_size.py): 3 test (red trước sửa: 3 failed → green).
- `pytest tests/detection`: **98 passed, 1 failed** — lỗi `test_split_real_v2.py::test_no_holdout_images` **có sẵn, không liên quan**: manifest QA có 2 file trùng SHA-256 (`images (7).jpg` và `images (13).jpg`, cùng 48757 byte) → 26 SHA duy nhất ≠ 27. Thuộc Agent D/A.

### Việc cần bên khác (không sửa)
- **Agent C**: `src/web/detection_settings.py` L33/L211 mặc định `image_size=640` và truyền tường minh → web vẫn chạy YOLO ở 640. Đề nghị mặc định `None`/320 (hoặc lấy theo checkpoint).
- **Agent D**: `run_baseline_evaluation.py` L161 hard-code `image_size=640` cho YOLO → nên dùng 320 (= train imgsz) hoặc bỏ tham số. Manifest có ảnh trùng (xem trên).

## 4. Sẵn sàng huấn luyện lại

(a) **SSDLite init**: `build_ssdlite(pretrained=True)` nạp **toàn bộ detector COCO** (backbone + extra layers + regression head), chỉ bỏ `head.classification_head`. **Nhưng** config hiện tại còn đặt `backbone_weights: artifacts/official_run/best_model.pt` → sau đó **ghi đè backbone** bằng classifier stage-1 (khối 0–12 và một phần 13–15 khớp shape) → backbone không còn đồng bộ với extra layers/regression head COCO.
(b) **Augmentation**: `combined` = HorizontalFlip p0.5, Affine(scale 0.9–1.1, translate ±6.25%, rotate ±15°) p0.5, ColorJitter p0.7, Gaussian/Motion blur p0.3, copy-paste p0.5 (DonorBank TACO). Không có random crop/zoom-out/multi-scale. YOLO: toàn bộ aug native = 0, chỉ dùng transform trên (áp lên ảnh BGR của Ultralytics → hue lệch chiều, nhỏ).
(c) **Đổi input size SSDLite**: **Không hỗ trợ** — `SSDLiteWasteDetector` và `train_ssdlite` raise nếu ≠ 320; `build_ssdlite` dùng kích thước mặc định torchvision. Muốn 512 cần: torchvision **hard-code** `size = (320, 320)` trong `ssdlite320_mobilenet_v3_large` (không nhận qua kwargs), nên sau khi build phải đặt `model.transform.fixed_size = (512, 512)`, `model.transform.min_size = (512,)`, `model.transform.max_size = 512` (DefaultBoxGenerator sinh anchor theo kích thước feature map/ảnh nên tự thích ứng); lưu `image_size` vào metadata checkpoint, áp lại khi load; bỏ ràng buộc ==320 ở wrapper/factory/trainer. Lưu ý tên kiến trúc vẫn là “SSDLite320” theo AGENTS.md → nên giữ 320 làm cấu hình chính, 512 chỉ là thử nghiệm.
(d) **YOLO**: `model: yolov8n.pt` (COCO pretrained, `pretrained: true`), **imgsz 320**, epochs 120, batch 16, AdamW lr0 1e-3, patience 30.

### Đề xuất cụ thể (chưa thực hiện)
1. **Dữ liệu (quan trọng nhất, phối hợp Agent A)**: dựng split mới dùng ~2000 ảnh real đã review (TACO + OI), stratified theo lớp; val ≥ 150–200 ảnh real; synthetic chỉ ≤ 30–50% train hoặc loại bỏ các ảnh synthetic có vật thể nền không nhãn. Ghi fingerprint split mới.
2. **YOLOv8n**: `imgsz: 640` (QA có vật lớn & nhỏ; 4 GB RTX 2050 đủ với batch 8–16, amp true); bật aug native: `mosaic 1.0, close_mosaic 10, scale 0.5, translate 0.1, fliplr 0.5, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, degrees 10`; tắt AblationAugmentationTransform hoặc chỉ giữ copy-paste; `epochs 150, patience 40, cos_lr true`. Inference phải dùng cùng imgsz (đã tự động nhờ sửa trên).
3. **SSDLite320**: `pretrained: true`, **bỏ `backbone_weights`** (giữ detector COCO đồng bộ); thêm aug zoom-out/RandomIoUCrop (chuẩn SSD, torchvision `transforms.v2.RandomZoomOut` + `RandomIoUCrop`) để học vật lớn/nhỏ; lr0 1e-3 cosine + warmup, epochs 200–300 (SSD hội tụ chậm), batch 16–32, `freeze_bn: false` khi batch ≥ 16. Tùy chọn thí nghiệm 512 như (c).
4. **Chọn checkpoint** trên val real đủ lớn; báo cáo mAP50/mAP50-95 + đánh giá QA 27 ảnh sau train, cùng conf/imgsz.

## 5. Tái lập

```
.venv\Scripts\python.exe reports/model_v2/scripts/yolo_box_anomaly_probe.py        # -> yolo_box_anomaly_probe.json
.venv\Scripts\python.exe reports/model_v2/scripts/qa_imgsz_and_ssdlite_sanity.py   # -> qa_imgsz_and_ssdlite_sanity.json
.venv\Scripts\python.exe -m pytest tests/detection -q
```
Checkpoint YOLO SHA-256 `97f46ccaaa0a63c5934225d51ccd399c1c18eb3e1af8a63b9f14bec1aaa085c9`; SSDLite `584c5980eb4e…` (theo baseline QA).
