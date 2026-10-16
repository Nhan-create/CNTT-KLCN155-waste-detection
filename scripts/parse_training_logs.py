import json
import csv
from pathlib import Path

# SSDLite combined seed42
ssdlite_sum_path = Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/training_summary.json")
if ssdlite_sum_path.is_file():
    with open(ssdlite_sum_path, encoding="utf-8") as f:
        sum_data = json.load(f)
    print("=== SSDLITE COMBINED SEED42 TRAINING SUMMARY ===")
    for k, v in sum_data.items():
        if k not in ["class_names", "dataset_report", "environment", "config"]:
            print(f"  {k}: {v}")
    print("  config.train:", sum_data.get("config", {}).get("train"))

# History length
hist_path = Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/history.json")
if hist_path.is_file():
    with open(hist_path, encoding="utf-8") as f:
        hist = json.load(f)
    print(f"  Total history epochs logged: {len(hist)}")
    if hist:
        print(f"  Epoch 1: {hist[0]}")
        print(f"  Last logged epoch ({len(hist)}): {hist[-1]}")
        best_in_hist = max(hist, key=lambda x: x["validation"]["map50_95"])
        print(f"  Best in history: epoch={best_in_hist['epoch']}, map50_95={best_in_hist['validation']['map50_95']}, map50={best_in_hist['validation']['map50']}")

# YOLO combined seed42
yolo_sum_path = Path("artifacts/detection/waste-yolov8n-combined-seed42/training_summary.json")
if yolo_sum_path.is_file():
    with open(yolo_sum_path, encoding="utf-8") as f:
        yolo_sum = json.load(f)
    print("\n=== YOLOv8N COMBINED SEED42 TRAINING SUMMARY ===")
    for k, v in yolo_sum.items():
        if k not in ["class_names", "dataset_report", "environment", "config"]:
            print(f"  {k}: {v}")
    print("  config.train:", yolo_sum.get("config", {}).get("train"))

# YOLO results.csv
csv_path = Path("artifacts/detection/waste-yolov8n-combined-seed42/results.csv")
if csv_path.is_file():
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    print(f"  Total epochs in results.csv: {len(reader)}")
    if reader:
        clean_reader = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]
        print(f"  CSV columns: {list(clean_reader[0].keys())}")
        print(f"  Epoch 1: {clean_reader[0]}")
        print(f"  Last Epoch: {clean_reader[-1]}")
        best_row = max(clean_reader, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))
        print(f"  Best in CSV by mAP50-95(B): epoch={best_row.get('epoch')}, mAP50={best_row.get('metrics/mAP50(B)')}, mAP50-95={best_row.get('metrics/mAP50-95(B)')}, box_loss={best_row.get('val/box_loss')}, cls_loss={best_row.get('val/cls_loss')}, dfl_loss={best_row.get('val/dfl_loss')}")
