import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

files = [
    ('Before Quick Fix', 'reports/qa_acceptance/raw_predictions_before_quick_fix.json'),
    ('After Quick Fix', 'reports/qa_acceptance/raw_predictions_after_quick_fix.json'),
    ('YOLOv8n E1', 'reports/qa_acceptance/raw_predictions_yolov8n_e1.json')
]

for name, p in files:
    data = json.load(open(p, encoding='utf-8'))
    images = data['images'] if isinstance(data, dict) and 'images' in data else data
    print(f"\n==================== {name} ====================")
    for img in images:
        fn = img.get('filename', '')
        if 'battery' in fn or 'pin' in fn:
            print(f"File: {fn}")
            preds = img.get('predictions', [])
            if not preds:
                print("  No predictions.")
            for pr in preds:
                cname = pr.get('class_name')
                conf = pr.get('confidence')
                box = pr.get('bbox_xyxy')
                match = pr.get('match_status')
                print(f"  -> {cname:10s} (conf={conf:.4f}) | box={box} | status={match}")
