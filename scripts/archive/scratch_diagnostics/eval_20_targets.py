import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')

from PIL import Image
from src.detection.factory import create_detector
from src.detection.visualization import DETECTION_DISPLAY_NAMES

detector = create_detector('yolov8n', 'artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt', device='cpu', confidence_threshold=0.20)

manifest = json.loads(Path('reports/qa_acceptance/manifest_20_anh_kho.json').read_text(encoding='utf-8'))

print('=== KẾT QUẢ SUY LUẬN SƠ BỘ TRÊN 20 ẢNH MỚI TẢI VỀ (YOLOv8n v2 - Ngưỡng 0.20) ===\n')
for item in manifest:
    p = Path(item['saved_path'])
    with Image.open(p) as img:
        res = detector.detect_pil(img)
    dets = res.detections
    fname = item['filename']
    ctype = item['category_type'].upper()
    clabel = item['class_label']
    print(f"{fname} [{ctype}] - Mục tiêu: {clabel}")
    if dets:
        for d in dets:
            vi = DETECTION_DISPLAY_NAMES.get(d.class_id, d.class_id)
            print(f"   ✓ Phát hiện: {vi} ({d.class_id}) - Độ tin cậy: {d.confidence:.1%}, bbox=[{d.box.x1:.0f}, {d.box.y1:.0f}, {d.box.x2:.0f}, {d.box.y2:.0f}]")
    else:
        print("   ✗ Chưa bắt được ở ngưỡng >= 20% (Ca khó / vật thể ngụy trang)")
