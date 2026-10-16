"""
Tạo cases.csv và ảnh overlay cho các ca kiểm thử đa dạng từ data/holdout.
"""
import csv
import hashlib
import sys
from pathlib import Path
from PIL import Image, ImageOps

WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE_ROOT))

from src.detection.factory import create_detector
from src.detection.visualization import draw_detections, detection_label


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def main():
    out_dir = Path(__file__).resolve().parents[1]
    images_dir = out_dir / "images"
    csv_file = out_dir / "cases.csv"
    
    test_cases = [
        {
            "id": "CASE_001",
            "path": (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[5] / "data" / "holdout"))) / "rac-1562828175-1562828212-3852-1562828313.webp"),
            "scenario": "Ảnh người dùng báo lỗi (Bãi chai nhựa/thủy tinh, nền nhà)",
            "ground_truth_note": "Nhiều chai nhựa/thủy tinh nằm la liệt; không có quần áo"
        },
        {
            "id": "CASE_002",
            "path": (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[5] / "data" / "holdout"))) / "rac-thai-huu-co-1.jpg"),
            "scenario": "Rác hữu cơ thực tế (Vỏ hoa quả, rau củ)",
            "ground_truth_note": "Rác hữu cơ / sinh học"
        },
        {
            "id": "CASE_003",
            "path": (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[5] / "data" / "holdout"))) / "rac-thai-quan-ao-4.jpg"),
            "scenario": "Rác quần áo thực tế (Đống quần áo cũ ngoài trời)",
            "ground_truth_note": "Quần áo cũ phế thải"
        },
        {
            "id": "CASE_004",
            "path": (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[5] / "data" / "holdout"))) / "Phan-Loai-Rac-Tai-Ng-01.jpg"),
            "scenario": "Ảnh tổng hợp nhiều loại rác (Thùng rác phân loại)",
            "ground_truth_note": "Nhiều loại rác khác nhau đặt cạnh nhau"
        }
    ]
    
    yolo_model_path = WORKSPACE_ROOT / "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"
    ssd_model_path = WORKSPACE_ROOT / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"
    
    det_yolo_020 = create_detector("yolov8n", yolo_model_path, confidence_threshold=0.20, device="cpu")
    det_yolo_045 = create_detector("yolov8n", yolo_model_path, confidence_threshold=0.45, device="cpu")
    det_ssd_020 = create_detector("ssdlite", ssd_model_path, confidence_threshold=0.20, device="cpu")
    det_ssd_045 = create_detector("ssdlite", ssd_model_path, confidence_threshold=0.45, device="cpu")
    det_wbf_045 = create_detector("wbf", ssd_model_path, secondary_model_path=yolo_model_path, confidence_threshold=0.45, device="cpu")
    
    csv_rows = []
    
    for tc in test_cases:
        p = tc["path"]
        if not p.exists():
            continue
        sha = get_sha256(p)
        with Image.open(p) as opened:
            img = ImageOps.exif_transpose(opened).convert("RGB")
            
        res_y_020 = det_yolo_020.detect_pil(img)
        res_y_045 = det_yolo_045.detect_pil(img)
        res_s_020 = det_ssd_020.detect_pil(img)
        res_s_045 = det_ssd_045.detect_pil(img)
        res_w_045 = det_wbf_045.detect_pil(img)
        
        # Save overlay for YOLO @ 0.20 and 0.45
        img_y020 = draw_detections(img, res_y_020, style="box")
        img_y020.save(images_dir / f"{tc['id']}_yolo_conf020.jpg", quality=90)
        
        def summarize_dets(dets):
            if not dets:
                return "0 box"
            items = [f"{detection_label(d.class_id)}:{d.confidence:.0%}" for d in dets]
            return f"{len(dets)} box ({', '.join(items[:4])}{'...' if len(items)>4 else ''})"
            
        csv_rows.append({
            "case_id": tc["id"],
            "file_name": p.name,
            "sha256": sha,
            "scenario": tc["scenario"],
            "ground_truth_note": tc["ground_truth_note"],
            "yolo_conf020": summarize_dets(res_y_020.detections),
            "yolo_conf045_ui": summarize_dets(res_y_045.detections),
            "ssdlite_conf020": summarize_dets(res_s_020.detections),
            "ssdlite_conf045_ui": summarize_dets(res_s_045.detections),
            "wbf_conf045_ui": summarize_dets(res_w_045.detections),
            "overlay_path": f"images/{tc['id']}_yolo_conf020.jpg"
        })
        
    with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "file_name", "sha256", "scenario", "ground_truth_note",
            "yolo_conf020", "yolo_conf045_ui", "ssdlite_conf020", "ssdlite_conf045_ui",
            "wbf_conf045_ui", "overlay_path"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print("cases.csv created successfully with", len(csv_rows), "cases.")


if __name__ == "__main__":
    main()
