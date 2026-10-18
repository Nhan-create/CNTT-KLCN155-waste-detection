# Bộ 6 Biểu Đồ Thực Nghiệm Bảo Vệ Đồ Án (CNTT-KLCN155)

Toàn bộ 6 biểu đồ trong thư mục này được trích xuất trực tiếp từ nhật ký huấn luyện và kết quả đánh giá thực nghiệm thật của dự án bằng script [`scripts/generate_thesis_charts.py`](../../scripts/generate_thesis_charts.py).

## Danh sách 6 biểu đồ và nguồn dữ liệu gốc

| STT | Tên file biểu đồ | Nội dung biểu đồ | Nguồn dữ liệu thực tế trong Repo |
|:---:|---|---|---|
| 1 | [`yolov8n_loss_map_curve.png`](./yolov8n_loss_map_curve.png) | Đường cong Training/Validation Loss & Validation mAP của **YOLOv8n (`E1_yolov8n_10cls`)** qua **80 epochs** thực tế (Best Checkpoint tại **Epoch 70**: `mAP@0.5 = 27.41%` Ultralytics / `24.73%` COCOeval, `mAP@0.5:0.95 = 19.38%` Ultralytics / `16.00%` COCOeval). | [`artifacts/detection_v2/E1_yolov8n_10cls/results.csv`](../../artifacts/detection_v2/E1_yolov8n_10cls/results.csv), [`eval_val_test.json`](../../artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json) |
| 2 | [`ssdlite320_loss_map_curve.png`](./ssdlite320_loss_map_curve.png) | Đường cong Training Total Loss & Validation mAP của **SSDLite320-MobileNetV3 (`E2_ssdlite320_10cls`)** qua **36 epochs** thực tế (Best Checkpoint tại **Epoch 30**: `mAP@0.5 = 19.71%`, `mAP@0.5:0.95 = 12.65%`, Early Stopping tại Epoch 36). | [`artifacts/detection_v2/E2_ssdlite320_10cls/history.json`](../../artifacts/detection_v2/E2_ssdlite320_10cls/history.json), [`training_summary.json`](../../artifacts/detection_v2/E2_ssdlite320_10cls/training_summary.json) |
| 3 | [`normalized_confusion_matrix.png`](./normalized_confusion_matrix.png) | Ma trận nhầm lẫn chuẩn hóa thực nghiệm:<br>• **(A) Stage-2 Classifier (MobileNetV3-Large):** Accuracy **96.18%** trên 2.223 mẫu Validation.<br>• **(B) Object Detector (YOLOv8n 10-Class + Background):** Ma trận 11×11 chuẩn hóa theo cột Ground Truth (`E3_yolov8n_pilot`). | [`artifacts/official_run/val_confusion_matrix.csv`](../../artifacts/official_run/val_confusion_matrix.csv), [`artifacts/detection_v3_pilot/E3_yolov8n_pilot/confusion_matrix_normalized.png`](../../artifacts/detection_v3_pilot/E3_yolov8n_pilot/confusion_matrix_normalized.png) |
| 4 | [`per_class_ap50_comparison.png`](./per_class_ap50_comparison.png) | So sánh `AP@0.5` trên từng lớp giữa **SSDLite320-MobileNetV3 (`19.71%`)** và **YOLOv8n (`24.73%`)** trên tập Validation chính thức (299 ảnh, 902 vật thể, chuẩn `pycocotools.COCOeval`). | [`artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json`](../../artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json), [`artifacts/detection_v2/E2_ssdlite320_10cls/history.json`](../../artifacts/detection_v2/E2_ssdlite320_10cls/history.json) |
| 5 | [`ablation_study_chart.png`](./ablation_study_chart.png) | Nghiên cứu Ablation so sánh `Precision`, `Recall`, `mAP@0.5` và `Latency (ms) / FPS` qua 6 cấu hình Pipeline trên tập nghiệm thu độc lập `data/holdout` (27 ảnh, 57 vật thể). | [`reports/qa_acceptance/verified_evaluation_summary_from_raw.json`](../qa_acceptance/verified_evaluation_summary_from_raw.json), [`docs/BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md`](../../docs/BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md) |
| 6 | [`pr_and_f1_curves.png`](./pr_and_f1_curves.png) | Đường cong `Precision-Recall` bậc thang thực nghiệm và đường cong `F1-Confidence` trích xuất trực tiếp từ các tệp dự đoán thật trên tập `data/holdout`. | [`reports/qa_acceptance/raw_predictions_before_quick_fix.json`](../qa_acceptance/raw_predictions_before_quick_fix.json), [`raw_predictions_after_quick_fix.json`](../qa_acceptance/raw_predictions_after_quick_fix.json), [`raw_predictions_yolov8n_e1.json`](../qa_acceptance/raw_predictions_yolov8n_e1.json) |

## Lệnh tái tạo bộ biểu đồ

```powershell
.\.venv\Scripts\python.exe scripts/generate_thesis_charts.py --out-dir reports/figures
```
