# BÁO CÁO HIỆU CHỈNH VÀ THỐNG NHẤT SỐ LIỆU BASELINE (BASELINE_CORRECTION.md)
**Sprint:** SPRINT AI-UPGRADE-01 — Task 4: Real Image Acquisition & QA  
**Repository:** `D:\CNTT-KLCN155-waste-detection`  
**Ngày lập:** 12/10/2026  
**Người thực hiện:** Antigravity AI Engineer  
**Đối tượng thẩm định:** ChatGPT PM / Reviewer  

---

## I. MỤC ĐÍCH VÀ PHẠM VI

Báo cáo này giải trình nguồn gốc thấu đáo, có bằng chứng raw log / file hash đối với các số liệu baseline mâu thuẫn giữa `AI_UPGRADE_BASELINE.md` và các báo cáo demo trước đây:
1. Số liệu `mAP50 = 0.1380` (13.80%).
2. Số liệu `mAP50 ~ 0.057` (5.70% - 5.72%).
3. Số liệu `Recall ~ 8.8%` (8.77%).
4. Số liệu `Precision = 0.200`, `Recall = 0.158`.

Tuyệt đối không sửa Ground Truth (`data/real_test/ground_truth_v1.json` và `v2.json`), không tinh chỉnh ngưỡng tin cậy (threshold tuning) trên tập holdout, và dựa 100% trên raw artifacts hiện hữu.

---

## II. ĐỐI CHIẾU CHECKPOINT VÀ NGUỒN GỐC TỪNG CHỈ SỐ

### 1. Bảng xác thực Checkpoint gốc (Immutable Hashes)

| Mô hình | Đường dẫn trọng số | Dung lượng | SHA-256 Checksum |
|---|---|---|---|
| **YOLOv8n E1** | `artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt` | 6.228.451 bytes | `7e986deed90ba4208f4662039c1e78b2d03ffb41a6e57e8235184db78aec32d8` |
| **SSDLite320 E2** | `artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt` | 28.459.990 bytes | `c9eb52efe6c5592340843d7678ae7d742eb2250d5ee9aaebbae620b8ebd465e7` |
| **YOLOv8n v1** *(Legacy synthetic)* | `artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt` | 6.230.000 bytes | `97f46ccaaa0a63c5934225d51ccd399c1c18eb3e1af8a63b9f14bec1aaa085c9` |

---

### 2. Phân rã nguồn gốc và bóc tách các sai lệch số liệu

#### A. Chỉ số `0.1380` (13.80%)
* **Bản chất thực tế:** Đây là chỉ số **$\text{mAP50-95}$** (chính xác là `0.1380492811576202`), đo trên **tập Internal Test split (299 ảnh, 774 boxes)** của dataset Mendeley v2, được sinh ra từ bộ đánh giá COCO chuẩn (`pycocotools.COCOeval`).
* **Bằng chứng gốc:** File `artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json`, dòng `test.map50_95`.
* **Sai lệch xảy ra trong tài liệu cũ:**
  1. Đã nhầm lẫn giữa chỉ số khắt khe $\text{mAP50-95}$ (0.1380) và $\text{mAP50}$ (trên cùng tập test 299 ảnh này, $\text{mAP50} = 0.222756 \approx 22.28\%$).
  2. Đã đánh tráo tập dữ liệu: Lấy kết quả đánh giá trên tập **Internal Test split 299 ảnh** gán nhãn thành hiệu năng trên tập **External Holdout 27 ảnh**.
* **Kết luận:** `0.1380` **KHÔNG PHẢI** là holdout mAP50 của YOLOv8n.

#### B. Chỉ số `0.0572` (~5.7%)
* **Bản chất thực tế:** Đây là chỉ số **$\text{mAP50}$** của mô hình **SSDLite320-MobileNetV3 E2** trên tập **External Holdout 27 ảnh / 57 GT**, được đánh giá tại ngưỡng vận hành `conf=0.25`, `iou=0.50`.
* **Bằng chứng gốc:** File `reports/qa_acceptance/PRACTICAL_METRICS_REPORT.md` và `reports/qa_acceptance/practical_metrics_baseline.json`.
* **Sai lệch xảy ra trong tài liệu cũ:** Báo cáo demo trước đây nhắc đến con số "mAP50 khoảng 5–6%" cho cả hệ thống hoặc bị gán nhầm lẫn giữa SSDLite và YOLOv8n.

#### C. Chỉ số `Recall ~ 8.8%` và `Precision = 0.1562`
* **Bản chất thực tế:** Đây là chỉ số vận hành thực tế của **YOLOv8n E1** trên tập **External Holdout 27 ảnh / 57 GT** ở ngưỡng chuẩn `conf=0.25`, `iou=0.50`.
* **Bằng chứng gốc:** File `reports/qa_acceptance/raw_predictions_yolov8n_e1.json`, khi chạy qua bộ đo chuẩn `reports/qa_acceptance/practical_metrics.py`:
  - Số lượng True Positives: $\text{TP} = 5$
  - Số lượng False Positives: $\text{FP} = 27$
  - Số lượng False Negatives: $\text{FN} = 52$
  - $\text{Recall} = \frac{5}{57} = 0.087719... \approx \mathbf{8.77\%}$ (~8.8%)
  - $\text{Precision} = \frac{5}{5 + 27} = \frac{5}{32} = \mathbf{0.1562}$ (15.62%)
  - $\text{F1-score} = \mathbf{0.1124}$ (11.24%)
  - $\text{mAP50 (all 27)} = \mathbf{0.0498}$ (~5.0%)
  - $\text{mAP50 (dedup 26)} = \mathbf{0.0514}$ (~5.1%)

#### D. Chỉ số `Precision = 0.200`, `Recall = 0.158`
* **Bản chất thực tế:** Đây là kết quả đo thực nghiệm trên 27 ảnh holdout nhưng tại **ngưỡng tin cậy phi chuẩn `conf = 0.15` và `iou = 0.20`** ($\text{TP}=9, \text{FP}=36, \text{FN}=36, \text{Wrong Class}=12$).
* **Kết luận:** Không sử dụng cặp số này vì vi phạm ngưỡng vận hành chuẩn $\text{conf}=0.25$ và tiêu chuẩn COCO $\text{IoU}=0.50$.

---

## III. BẢNG BASELINE CHÍNH THỨC ĐƯỢC ĐỐI CHIẾU ĐỒNG NHẤT

Dưới đây là bảng số liệu duy nhất được công nhận cho Baseline Sprint AI-UPGRADE-01:

| Mô hình / Checkpoint | Tập dữ liệu đánh giá | Evaluator & Cấu hình | mAP50 | mAP50-95 | Precision | Recall | F1-Score | Ghi chú & Nguồn gốc |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8n E1**<br>`7e986deed90b...` | **Internal Val**<br>(299 ảnh, 901 GT) | `pycocotools`<br>`iou=0.50:0.95` | **0.2473**<br>(24.73%) | **0.1600**<br>(16.00%) | N/A | N/A | N/A | `eval_val_test.json`<br>Val split Mendeley v2 |
| **YOLOv8n E1**<br>`7e986deed90b...` | **Internal Test**<br>(299 ảnh, 774 GT) | `pycocotools`<br>`iou=0.50:0.95` | **0.2228**<br>(22.28%) | **0.1380**<br>(13.80%) | N/A | N/A | N/A | `eval_val_test.json`<br>Nguồn gốc số `0.1380` |
| **YOLOv8n E1**<br>`7e986deed90b...` | **External Holdout**<br>(27 ảnh, 57 GT) | `practical_metrics.py`<br>`conf=0.25, iou=0.50` | **0.0498**<br>(~5.0%) | **0.0233**<br>(2.33%) | **0.1562**<br>(15.62%) | **0.0877**<br>(**8.77%**) | **0.1124**<br>(11.24%) | `raw_predictions_yolov8n_e1.json`<br>TP=5, FP=27, FN=52 |
| **SSDLite320 E2**<br>`c9eb52efe6c5...` | **Internal Val**<br>(299 ảnh, 901 GT) | `pycocotools`<br>`iou=0.50:0.95` | **0.1971**<br>(19.71%) | **0.1265**<br>(12.65%) | N/A | N/A | N/A | Validation log v2<br>(Khớp UI Streamlit 12.6%) |
| **SSDLite320 E2**<br>`c9eb52efe6c5...` | **External Holdout**<br>(27 ảnh, 57 GT) | `practical_metrics.py`<br>`conf=0.25, iou=0.50` | **0.0572**<br>(**5.72%**) | **0.0278**<br>(2.78%) | **0.0800**<br>(8.00%) | **0.0351**<br>(3.51%) | **0.0488**<br>(4.88%) | `practical_metrics_baseline.json`<br>TP=2, FP=23, FN=55 |

---

## IV. KẾT LUẬN VÀ CAM KẾT

1. Số liệu mAP50 trên external holdout của **YOLOv8n E1** chính xác là **0.0498 (5.0%)**, Recall **8.77%** tại ngưỡng vận hành chuẩn `conf=0.25`.
2. Con số `0.1380` là $\text{mAP50-95}$ của tập test 299 ảnh, chính thức được đính chính và loại bỏ khỏi bảng chỉ số holdout.
3. Con số `0.0572` (5.72%) là mAP50 của **SSDLite320 E2** trên holdout, không phải của YOLOv8n.
4. Mọi chỉ số nâng cấp sau này của Task 5 / Sprint sẽ được so sánh trực tiếp với bộ số liệu đã thống nhất tại Mục III.
