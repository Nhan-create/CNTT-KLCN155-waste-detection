# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `print_baseline_summary`.
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

with open('reports/baseline_demo_results/baseline_report.json', encoding='utf-8') as f:
    rep = json.load(f)

for m_key in ['ssdlite', 'yolov8n']:
    d = rep[m_key]
    disp = d['display_name']
    print('================================================================================')
    print('MODEL:', disp)
    print('================================================================================')
    lat = d['latency']
    print(f"Latency GPU (mean): {lat['gpu_mean_ms']:.2f} ms | FPS GPU: {lat['gpu_fps']}")
    print(f"Latency CPU (mean): {lat['cpu_sample_mean_ms']:.2f} ms | FPS CPU: {lat['cpu_fps']}")
    cmap = d['coco_map']
    print(f"COCO mAP50: {cmap['mAP50']}% | mAP50-95: {cmap['mAP50_95']}% | Class-Agnostic AP50: {cmap['class_agnostic_AP50']}%")
    ov = d['overall_metrics_conf025_iou050']
    print(f"Operating Point (conf>=0.25, IoU>=0.50):")
    print(f"  Total GT: {ov['total_gt_objects']} | TP: {ov['true_positives']} | FP: {ov['false_positives']} | FN: {ov['false_negatives']}")
    print(f"  Precision: {ov['precision']}% | Recall: {ov['recall']}% | F1: {ov['f1']}%")
    loc = d['localization_accuracy']
    print(f"Localization Accuracy: {loc['class_accuracy']}% ({loc['correct_class_pairs']}/{loc['matched_pairs']} pairs)")
    print(f"Strict DoD Status: {d['strict_dod']['status']}")
    print("\nPer-Class Breakdown:")
    print(f"  {'Class':<12} {'Vietnamese':<24} {'GT':<4} {'TP':<4} {'FP':<4} {'FN':<4} {'P (%)':<8} {'R (%)':<8} {'F1 (%)':<8} {'AP50 (%)':<8}")
    print("  " + "-" * 88)
    for c_name, c_data in d['per_class_metrics'].items():
        vi = c_data['vietnamese']
        gt = c_data['n_gt']
        tp = c_data['tp']
        fp = c_data['fp']
        fn = c_data['fn']
        p = c_data['precision'] * 100
        r = c_data['recall'] * 100
        f1 = c_data['f1'] * 100
        ap = c_data['ap50'] * 100
        print(f"  {c_name:<12} {vi:<24} {gt:<4} {tp:<4} {fp:<4} {fn:<4} {p:<8.1f} {r:<8.1f} {f1:<8.1f} {ap:<8.1f}")
    print()

print("================================================================================")
print("TASK 2 FUNCTIONAL TEST CASES")
print("================================================================================")
with open('reports/baseline_demo_results/task2_results.json', encoding='utf-8') as f:
    t2 = json.load(f)

for case in t2:
    print(f"\nCase: {case['case_id']} ({case['case_name']}) - File: {case['filename']}")
    print(f"  SSDLite: {case['ssdlite']['count']} detections in {case['ssdlite']['latency_ms']} ms ({case['ssdlite']['fps']} FPS)")
    for det in case['ssdlite']['detections']:
        print(f"    - {det['vietnamese']} ({det['class']}): conf={det['conf']:.3f}, box={det['box']}")
    print(f"  YOLOv8n: {case['yolov8n']['count']} detections in {case['yolov8n']['latency_ms']} ms ({case['yolov8n']['fps']} FPS)")
    for det in case['yolov8n']['detections']:
        print(f"    - {det['vietnamese']} ({det['class']}): conf={det['conf']:.3f}, box={det['box']}")
