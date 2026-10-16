# PRACTICAL METRICS REPORT — Holdout `D:\Hinh_Anh_Rac` (27 ảnh, 57 vật thể)

> Agent D (QA Độc lập). Bổ sung bộ chỉ số nghiệm thu thực tế. **DoD nghiêm ngặt** (FN=0, FP=0, sai lớp=0, IoU≥0.75, Intersection/GT≥0.99, coverage 100%) trong `acceptance_evaluator.py` **giữ nguyên** và được coi là *tiêu chí lý tưởng*; nó vẫn được tính lại trong mục `strict_dod_reference` của mỗi kết quả.

## 1. Công cụ

- Module + CLI: [practical_metrics.py](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/practical_metrics.py)
- Unit test: [test_practical_metrics.py](file:///d:/CNTT-KLCN155-waste-detection/tests/qa/test_practical_metrics.py) (16 case tính tay; `pytest tests/qa` → **24 passed**)
- Kết quả baseline: [practical_metrics_baseline.json](file:///d:/CNTT-KLCN155-waste-detection/reports/qa_acceptance/practical_metrics_baseline.json) (gộp), cùng `practical_metrics_ssdlite.json`, `practical_metrics_yolov8n.json`

```powershell
.venv\Scripts\python.exe reports/qa_acceptance/practical_metrics.py --predictions <file> --out <file> `
    [--conf 0.25] [--min-recall 0.60] [--min-precision 0.60] [--min-class-acc 0.75] [--gt ...] [--manifest ...]
```

Định dạng predictions chấp nhận (tự nhận diện), class_id 0..9 theo thứ tự cố định (SSDLite phải decode label−1 trước):

1. `{"<filename>": [{"class_id": int, "confidence": float, "bbox_xyxy": [x1,y1,x2,y2]}, ...]}` — tọa độ pixel ảnh gốc sau EXIF transpose. Có thể bọc `{"predictions": {...}, "confidence_threshold": <min lưu>, "backend": ..., "checkpoint_sha256": ...}`.
2. `{"images": [{"filename": ..., "objects": [...]}]}` (định dạng của `QAAcceptanceEvaluator`).
3. `baseline_results_*.json` (dựng lại toàn bộ prediction từ `matched_pairs` + `false_positives`, kiểm tra khớp `pred_count`).

**Khuyến nghị cho mô hình mới:** xuất predictions với confidence tối thiểu thấp (≈0.001–0.05) để AP không bị cắt cụt.

## 2. Định nghĩa chỉ số

| Chỉ số | Định nghĩa |
|---|---|
| AP50 / mAP50-95 | COCO: greedy theo confidence giảm dần, mỗi prediction khớp GT cùng lớp chưa dùng có IoU lớn nhất ≥ t; nội suy 101 điểm recall; IoU t ∈ {0.50:0.05:0.95}. mAP = trung bình trên các lớp có GT (lớp không có GT bị loại). Không giới hạn maxDets. |
| Class-agnostic AP50 | Như trên nhưng bỏ qua lớp (chất lượng định vị thuần). |
| P / R / F1 @IoU0.5 | Lọc confidence ≥ ngưỡng, khớp có lớp như COCO. FN = GT không khớp; ảnh thiếu prediction ⇒ toàn bộ GT là FN. |
| Class accuracy trên cặp khớp | Khớp 1-1 không xét lớp, IoU ≥ 0.5, greedy theo IoU giảm dần; accuracy = cặp đúng lớp / số cặp. |
| Ma trận nhầm lẫn 11×11 | Hàng = GT, cột = prediction; index 10 = background (hàng background = FP, cột background = FN). |
| View `dedup_sha256` | Loại ảnh trùng SHA-256 (giữ ảnh đầu theo thứ tự GT; loại `images (7).jpg`, trùng với `images (13).jpg`) → 26 ảnh, 56 vật thể. |

**Tiêu chí nghiệm thu thực tế (cấu hình được, mặc định):** Recall@0.5 ≥ 0.60, Precision@0.5 ≥ 0.60, class accuracy trên cặp khớp ≥ 0.75, và đủ prediction cho mọi ảnh — tại confidence vận hành (mặc định 0.25).

## 3. Kết quả baseline (conf vận hành 0.25, view 27 ảnh)

| Chỉ số | SSDLite320 (chính) | YOLOv8n (đối chứng) |
|---|---:|---:|
| mAP50 (10 lớp có GT) | 0.0572 | 0.0000 |
| mAP50-95 | 0.0057 | 0.0000 |
| Class-agnostic AP50 / AP50-95 | 0.0561 / 0.0084 | 0.0000 / 0.0000 |
| TP / FP / FN @0.5 (có lớp) | 2 / 23 / 55 | 0 / 28 / 57 |
| Precision / Recall / F1 @0.5 | 0.080 / 0.035 / 0.049 | 0.000 / 0.000 / 0.000 |
| P / R @0.5 không xét lớp | 0.280 / 0.123 | 0.000 / 0.000 |
| Class accuracy trên cặp khớp | 0.143 (1/7) | n/a (0 cặp) |
| **Nghiệm thu thực tế** | **FAIL** | **FAIL** |
| DoD nghiêm ngặt (tham chiếu) | FAIL (FN=44, FP=12 theo matcher evaluator, IoU≥0.10) | FAIL (FN=56, FP=27) |

View `dedup_sha256` (26 ảnh/56 GT): SSDLite R=0.036, P=0.080, mAP50=0.0572 — kết luận không đổi.

SSDLite AP50 theo lớp: battery 0.505, cardboard 0.067, các lớp còn lại 0.000. Nhầm lẫn trên 7 cặp định vị đúng: battery→battery 1; cardboard→battery, metal→battery, glass→metal, metal→cardboard, paper→cardboard, plastic→cardboard (mỗi loại 1).

Quét confidence (SSDLite, có lớp): 0.20: P 0.057 / R 0.035 · 0.30: 0.091 / 0.018 · 0.45 (mặc định app): 0.500 / 0.018 (1 TP, 1 FP).

## 4. Nhận xét QA

- Cả hai baseline đều rất xa tiêu chí thực tế; khoảng cách lớn nhất là **recall** (SSDLite chỉ định vị được 7/57 vật thể ở IoU≥0.5 dù bỏ qua lớp).
- **YOLOv8n**: toàn bộ 33 box (conf≥0.2) đều rất nhỏ (thường 10–50 px) và IoU tối đa với GT ≤ 0.17 → 0 cặp khớp. Đây là quan sát từ dữ liệu baseline; QA không kết luận nguyên nhân (mô hình hay hậu xử lý tọa độ) — đề nghị Agent B/C kiểm tra riêng.
- Lớp `paper`, `trash`, `battery` có ≤2 GT ⇒ AP theo lớp rất nhiễu; ưu tiên đọc mAP cùng P/R tổng và class-agnostic AP.

> [!WARNING]
> Hạn chế: `baseline_results_*.json` chỉ lưu detection có confidence ≥ 0.20. AP tính từ các file này bị cắt cụt (là cận dưới), và P/R/F1 chỉ hợp lệ với ngưỡng ≥ 0.20. Mỗi kết quả JSON có mục `warnings` ghi rõ điều này.
