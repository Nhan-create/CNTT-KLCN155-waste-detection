# Mô tả: [Lưu trữ — Kịch bản kiểm toán lịch sử] Thực thi tác vụ hỗ trợ kiểm toán và xử lý dữ liệu lịch sử trong `build_visual_dashboard`.
"""
Build an interactive, modern, standalone HTML dashboard for baseline evaluation results
(CNTT-KLCN155 Waste Detection).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from reports.qa_acceptance.practical_metrics import CLASS_NAMES
RESULTS_DIR = ROOT / "reports/baseline_demo_results"
REPORT_JSON = RESULTS_DIR / "baseline_report.json"
TASK2_JSON = RESULTS_DIR / "task2_results.json"
MANIFEST_JSON = ROOT / "reports/qa_acceptance/manifest_hinh_anh_rac.json"
GT_JSON = ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json"

with open(REPORT_JSON, "r", encoding="utf-8") as f:
    report = json.load(f)

with open(TASK2_JSON, "r", encoding="utf-8") as f:
    task2 = json.load(f)

with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(GT_JSON, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

gt_by_fn = {img["filename"]: img for img in gt_data["images"]}

ssd = report["ssdlite"]
yolo = report["yolov8n"]

html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CNTT-KLCN155 — Báo cáo Nghiệm thu Baseline & Demo Nhận diện Rác</title>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #3b82f6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5; }}
  .container {{ max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 2rem; margin-bottom: 8px; color: #fff; display: flex; align-items: center; gap: 12px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 1rem; }}
  .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
  .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid var(--danger); }}
  .badge-success {{ background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid var(--success); }}
  .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fde68a; border: 1px solid var(--warning); }}
  
  .grid-2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 24px; margin-bottom: 32px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
  .card-title {{ font-size: 1.25rem; font-weight: 700; color: #fff; }}
  
  .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
  .metric-box {{ background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }}
  .metric-label {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }}
  .metric-val {{ font-size: 1.35rem; font-weight: 700; color: #fff; }}
  .metric-sub {{ font-size: 0.75rem; color: var(--muted); margin-top: 2px; }}
  
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.875rem; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: rgba(15, 23, 42, 0.5); color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; }}
  tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
  
  .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 20px; margin-top: 20px; }}
  .gallery-item {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }}
  .gallery-img {{ width: 100%; height: 260px; object-fit: contain; background: #000; }}
  .gallery-body {{ padding: 14px; flex: 1; display: flex; flex-direction: column; justify-content: space-between; }}
  .gallery-title {{ font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; word-break: break-all; }}
  .gallery-meta {{ font-size: 0.8rem; color: var(--muted); margin-bottom: 8px; }}
  
  .tabs {{ display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
  .tab-btn {{ background: transparent; border: 1px solid var(--border); color: var(--muted); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 0.875rem; transition: all 0.2s; }}
  .tab-btn.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}
  
  .alert-box {{ background: rgba(239, 68, 68, 0.1); border-left: 4px solid var(--danger); padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 0.9rem; color: #fca5a5; }}
  .legend {{ display: flex; gap: 16px; align-items: center; margin-top: 10px; font-size: 0.8rem; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .color-dot {{ width: 12px; height: 12px; border-radius: 2px; }}
</style>
</head>
<body>
<div class="container">
  <h1>
    <span>Báo cáo Baseline Hiện tại & Đánh giá Demo Nhận diện Rác</span>
    <span class="badge badge-danger">DoD: FAILED (Độ chính xác thấp)</span>
  </h1>
  <div class="subtitle">
    Dự án: CNTT-KLCN155 — Tập dữ liệu nghiệm thu đóng băng: <strong>data/holdout</strong> (27 ảnh, 57 Ground Truth objects) — Ngưỡng vận hành: conf ≥ 0.25, IoU ≥ 0.50
  </div>

  <div class="alert-box">
    <strong>KẾT LUẬN QUAN TRỌNG TỪ ĐÁNH GIÁ THỰC TẾ:</strong><br>
    Cả hai mô hình hiện tại (SSDLite320 và YOLOv8n) đều <strong>chưa đạt độ chính xác thực tế để triển khai thương mại/sản xuất</strong>.
    • SSDLite mAP50 đạt <strong>{ssd['coco_map']['mAP50']}%</strong>, F1 đạt <strong>{ssd['overall_metrics_conf025_iou050']['f1']}%</strong> (Bỏ sót 52/57 vật thể rác).<br>
    • YOLOv8n mAP50 đạt <strong>{yolo['coco_map']['mAP50']}%</strong>, F1 đạt <strong>{yolo['overall_metrics_conf025_iou050']['f1']}%</strong> (Bỏ sót 52/57 vật thể rác, sinh 27 false alarms).<br>
    • Cả hai mô hình đều nhận nhầm pin thành "Giày dép", nhận nhầm tủ điện thành "Kim loại", và bỏ sót hoàn toàn rác nhỏ / xa.
  </div>

  <!-- SO SÁNH HAI MÔ HÌNH -->
  <div class="grid-2">
    <!-- SSDLITE -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">1. {ssd['display_name']}</div>
          <div style="font-size:0.75rem; color:var(--muted);">Mô hình chính (MobileNetV3 Backbone, 320x320) — Dung lượng: {ssd['checkpoint_size_mb']} MB</div>
        </div>
        <span class="badge badge-danger">mAP50: {ssd['coco_map']['mAP50']}%</span>
      </div>
      
      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">Precision</div>
          <div class="metric-val">{ssd['overall_metrics_conf025_iou050']['precision']}%</div>
          <div class="metric-sub">TP: {ssd['overall_metrics_conf025_iou050']['true_positives']} | FP: {ssd['overall_metrics_conf025_iou050']['false_positives']}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Recall</div>
          <div class="metric-val">{ssd['overall_metrics_conf025_iou050']['recall']}%</div>
          <div class="metric-sub">FN: {ssd['overall_metrics_conf025_iou050']['false_negatives']} / 57</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">F1-Score</div>
          <div class="metric-val">{ssd['overall_metrics_conf025_iou050']['f1']}%</div>
          <div class="metric-sub">Harmonic mean</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">COCO mAP50-95</div>
          <div class="metric-val">{ssd['coco_map']['mAP50_95']}%</div>
          <div class="metric-sub">Agnostic: {ssd['coco_map']['class_agnostic_AP50']}%</div>
        </div>
      </div>

      <div class="metrics-grid" style="grid-template-columns: repeat(2, 1fr);">
        <div class="metric-box">
          <div class="metric-label">GPU Latency / FPS (RTX 3060)</div>
          <div class="metric-val" style="color:var(--success);">{ssd['latency']['gpu_mean_ms']:.1f} ms <span style="font-size:0.9rem; font-weight:normal;">({ssd['latency']['gpu_fps']} FPS)</span></div>
        </div>
        <div class="metric-box">
          <div class="metric-label">CPU Latency / FPS</div>
          <div class="metric-val" style="color:var(--warning);">{ssd['latency']['cpu_sample_mean_ms']:.1f} ms <span style="font-size:0.9rem; font-weight:normal;">({ssd['latency']['cpu_fps']} FPS)</span></div>
        </div>
      </div>

      <div style="font-size:0.85rem; font-weight:600; margin-top:12px;">Bảng chỉ số 10 lớp (SSDLite):</div>
      <table>
        <thead>
          <tr>
            <th>Lớp</th>
            <th>Tên Việt</th>
            <th>GT</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>P (%)</th>
            <th>R (%)</th>
            <th>F1 (%)</th>
            <th>AP50 (%)</th>
          </tr>
        </thead>
        <tbody>
"""

for c_name, c_data in ssd["per_class_metrics"].items():
    html += f"""
          <tr>
            <td><code>{c_name}</code></td>
            <td>{c_data['vietnamese']}</td>
            <td>{c_data['n_gt']}</td>
            <td>{c_data['tp']}</td>
            <td>{c_data['fp']}</td>
            <td>{c_data['fn']}</td>
            <td>{c_data['precision']*100:.1f}</td>
            <td>{c_data['recall']*100:.1f}</td>
            <td>{c_data['f1']*100:.1f}</td>
            <td>{c_data['ap50']*100:.1f}</td>
          </tr>
    """

html += f"""
        </tbody>
      </table>
    </div>

    <!-- YOLOV8N -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">2. {yolo['display_name']}</div>
          <div style="font-size:0.75rem; color:var(--muted);">Mô hình đối chứng (640x640) — Dung lượng: {yolo['checkpoint_size_mb']} MB</div>
        </div>
        <span class="badge badge-danger">mAP50: {yolo['coco_map']['mAP50']}%</span>
      </div>

      <div class="metrics-grid">
        <div class="metric-box">
          <div class="metric-label">Precision</div>
          <div class="metric-val">{yolo['overall_metrics_conf025_iou050']['precision']}%</div>
          <div class="metric-sub">TP: {yolo['overall_metrics_conf025_iou050']['true_positives']} | FP: {yolo['overall_metrics_conf025_iou050']['false_positives']}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Recall</div>
          <div class="metric-val">{yolo['overall_metrics_conf025_iou050']['recall']}%</div>
          <div class="metric-sub">FN: {yolo['overall_metrics_conf025_iou050']['false_negatives']} / 57</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">F1-Score</div>
          <div class="metric-val">{yolo['overall_metrics_conf025_iou050']['f1']}%</div>
          <div class="metric-sub">Harmonic mean</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">COCO mAP50-95</div>
          <div class="metric-val">{yolo['coco_map']['mAP50_95']}%</div>
          <div class="metric-sub">Agnostic: {yolo['coco_map']['class_agnostic_AP50']}%</div>
        </div>
      </div>

      <div class="metrics-grid" style="grid-template-columns: repeat(2, 1fr);">
        <div class="metric-box">
          <div class="metric-label">GPU Latency / FPS (RTX 3060)</div>
          <div class="metric-val" style="color:var(--success);">{yolo['latency']['gpu_mean_ms']:.1f} ms <span style="font-size:0.9rem; font-weight:normal;">({yolo['latency']['gpu_fps']} FPS)</span></div>
        </div>
        <div class="metric-box">
          <div class="metric-label">CPU Latency / FPS</div>
          <div class="metric-val" style="color:var(--warning);">{yolo['latency']['cpu_sample_mean_ms']:.1f} ms <span style="font-size:0.9rem; font-weight:normal;">({yolo['latency']['cpu_fps']} FPS)</span></div>
        </div>
      </div>

      <div style="font-size:0.85rem; font-weight:600; margin-top:12px;">Bảng chỉ số 10 lớp (YOLOv8n):</div>
      <table>
        <thead>
          <tr>
            <th>Lớp</th>
            <th>Tên Việt</th>
            <th>GT</th>
            <th>TP</th>
            <th>FP</th>
            <th>FN</th>
            <th>P (%)</th>
            <th>R (%)</th>
            <th>F1 (%)</th>
            <th>AP50 (%)</th>
          </tr>
        </thead>
        <tbody>
"""

for c_name, c_data in yolo["per_class_metrics"].items():
    html += f"""
          <tr>
            <td><code>{c_name}</code></td>
            <td>{c_data['vietnamese']}</td>
            <td>{c_data['n_gt']}</td>
            <td>{c_data['tp']}</td>
            <td>{c_data['fp']}</td>
            <td>{c_data['fn']}</td>
            <td>{c_data['precision']*100:.1f}</td>
            <td>{c_data['recall']*100:.1f}</td>
            <td>{c_data['f1']*100:.1f}</td>
            <td>{c_data['ap50']*100:.1f}</td>
          </tr>
    """

html += f"""
        </tbody>
      </table>
    </div>
  </div>

  <!-- TASK 2 FUNCTIONAL TEST CASES -->
  <div class="card" style="margin-bottom: 32px;">
    <div class="card-header">
      <div>
        <div class="card-title">Kết quả 5 Tình huống Thực tế Đặc thù (TASK 2)</div>
        <div style="font-size:0.75rem; color:var(--muted);">Kiểm tra nhận diện trên các điều kiện biên: 1 vật thể, nhiều vật thể, vật thể xa, không có rác, tủ điện ngoại cảnh</div>
      </div>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px;">
"""

for case in task2:
    ssd_txt = ", ".join(f"{d['vietnamese']} ({d['conf']:.2f})" for d in case['ssdlite']['detections']) if case['ssdlite']['detections'] else "Không phát hiện (0)"
    yolo_txt = ", ".join(f"{d['vietnamese']} ({d['conf']:.2f})" for d in case['yolov8n']['detections']) if case['yolov8n']['detections'] else "Không phát hiện (0)"
    html += f"""
      <div style="background: rgba(15,23,42,0.4); border: 1px solid var(--border); border-radius: 8px; padding: 14px;">
        <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 4px; color: #fff;">{case['case_name']}</div>
        <div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 10px;">File: <code>{case['filename']}</code></div>
        <div style="margin-bottom: 8px;">
          <a href="{case['ssdlite']['image_rel']}" target="_blank">
            <img src="{case['ssdlite']['image_rel']}" style="width: 100%; height: 160px; object-fit: contain; background: #000; border-radius: 4px; margin-bottom: 6px;">
          </a>
        </div>
        <div style="font-size: 0.8rem; margin-bottom: 4px;">
          <strong style="color: #60a5fa;">SSDLite ({case['ssdlite']['fps']} FPS):</strong> {ssd_txt}
        </div>
        <div style="font-size: 0.8rem;">
          <strong style="color: #f87171;">YOLOv8n ({case['yolov8n']['fps']} FPS):</strong> {yolo_txt}
        </div>
      </div>
    """

html += f"""
    </div>
  </div>

  <!-- THƯ VIỆN ẢNH OVERLAY TOÀN BỘ 27 ẢNH -->
  <div class="card">
    <div class="card-header">
      <div>
        <div class="card-title">Thư viện Ảnh Overlay 27 Ảnh Holdout (Ground Truth vs Prediction)</div>
        <div style="font-size:0.75rem; color:var(--muted);">Khung Xanh lá: Ground Truth thật — Khung Màu: Model dự đoán (conf ≥ 0.25)</div>
      </div>
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-ssdlite')">SSDLite320 (27 ảnh)</button>
        <button class="tab-btn" onclick="switchTab('tab-yolov8n')">YOLOv8n (27 ảnh)</button>
      </div>
    </div>

    <!-- TAB SSDLITE -->
    <div id="tab-ssdlite" class="tab-content active">
      <div class="gallery-grid">
"""

for f_info in manifest["files"]:
    fn = f_info["filename"]
    stem = Path(fn).stem
    overlay_path = f"overlays_ssdlite/{stem}_overlay_ssdlite.jpg"
    gt_objs = gt_by_fn.get(fn, {}).get("objects", [])
    gt_classes = ", ".join(CLASS_NAMES[o["class_id"]] for o in gt_objs) if gt_objs else "Không có rác"
    html += f"""
        <div class="gallery-item">
          <a href="{overlay_path}" target="_blank">
            <img class="gallery-img" src="{overlay_path}" alt="{fn}" loading="lazy">
          </a>
          <div class="gallery-body">
            <div class="gallery-title">{fn}</div>
            <div class="gallery-meta">
              <strong>Ground Truth ({len(gt_objs)}):</strong> {gt_classes}
            </div>
            <div style="font-size:0.75rem; color:var(--muted);">
              <a href="{overlay_path}" target="_blank" style="color:var(--accent); text-decoration:none;">🔍 Xem ảnh kích thước đầy đủ &rarr;</a>
            </div>
          </div>
        </div>
    """

html += f"""
      </div>
    </div>

    <!-- TAB YOLOV8N -->
    <div id="tab-yolov8n" class="tab-content">
      <div class="gallery-grid">
"""

for f_info in manifest["files"]:
    fn = f_info["filename"]
    stem = Path(fn).stem
    overlay_path = f"overlays_yolov8n/{stem}_overlay_yolov8n.jpg"
    gt_objs = gt_by_fn.get(fn, {}).get("objects", [])
    gt_classes = ", ".join(CLASS_NAMES[o["class_id"]] for o in gt_objs) if gt_objs else "Không có rác"
    html += f"""
        <div class="gallery-item">
          <a href="{overlay_path}" target="_blank">
            <img class="gallery-img" src="{overlay_path}" alt="{fn}" loading="lazy">
          </a>
          <div class="gallery-body">
            <div class="gallery-title">{fn}</div>
            <div class="gallery-meta">
              <strong>Ground Truth ({len(gt_objs)}):</strong> {gt_classes}
            </div>
            <div style="font-size:0.75rem; color:var(--muted);">
              <a href="{overlay_path}" target="_blank" style="color:var(--accent); text-decoration:none;">🔍 Xem ảnh kích thước đầy đủ &rarr;</a>
            </div>
          </div>
        </div>
    """

html += f"""
      </div>
    </div>
  </div>
</div>

<script>
function switchTab(tabId) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>
"""

dashboard_path = RESULTS_DIR / "index.html"
with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Visual dashboard built successfully at: {dashboard_path}")
