import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

# Load raw predictions
raw_before = json.load(open('reports/qa_acceptance/raw_predictions_before_quick_fix.json', encoding='utf-8'))
raw_after = json.load(open('reports/qa_acceptance/raw_predictions_after_quick_fix.json', encoding='utf-8'))
raw_yolo = json.load(open('reports/qa_acceptance/raw_predictions_yolov8n_e1.json', encoding='utf-8'))

gt_v1 = json.load(open('reports/qa_acceptance/ground_truth_hinh_anh_rac.json', encoding='utf-8'))
gt_v2 = json.load(open('reports/qa_acceptance/ground_truth_hinh_anh_rac_v2_proposed.json', encoding='utf-8'))

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def evaluate_pipeline(raw_data, gt_data, conf_thresh=0.25, iou_thresh=0.5, class_aware=True, v2_filter='all_active'):
    # Prepare GT map
    gt_map = {}
    total_gt = 0
    for img in gt_data['images']:
        fn = img['filename']
        objs = []
        for o in img['objects']:
            if 'is_active_for_instance_eval' in o:
                if v2_filter == 'discrete_only' and o.get('v2_status') != 'ACCEPTED':
                    continue
                if v2_filter == 'all_active' and not o.get('is_active_for_instance_eval', True):
                    continue
            objs.append(o)
            total_gt += 1
        gt_map[fn] = objs

    images_preds = raw_data['images'] if isinstance(raw_data, dict) and 'images' in raw_data else raw_data

    tp = 0
    fp = 0
    total_preds_evaluated = 0

    for img in images_preds:
        fn = img['filename']
        gt_objs = gt_map.get(fn, [])
        preds = [p for p in img.get('predictions', []) if p['confidence'] >= conf_thresh]
        preds.sort(key=lambda x: -x['confidence'])
        total_preds_evaluated += len(preds)

        matched_gt = set()
        for p in preds:
            p_box = p['bbox_xyxy']
            p_cls = p['class_id']
            best_iou = 0.0
            best_gt_idx = -1

            for idx, g in enumerate(gt_objs):
                if idx in matched_gt:
                    continue
                iou = compute_iou(p_box, g['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                g_cls = gt_objs[best_gt_idx]['class_id']
                if not class_aware or p_cls == g_cls:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            else:
                fp += 1

    fn_count = total_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'total_gt': total_gt,
        'total_preds': total_preds_evaluated,
        'tp': tp,
        'fp': fp,
        'fn': fn_count,
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2)
    }

pipelines = [
    ('SSDLite Before Quick Fix', raw_before, 0.25),
    ('SSDLite After Quick Fix', raw_after, 0.25),
    ('YOLOv8n E1', raw_yolo, 0.25)
]

print("=== COMPARATIVE EVALUATION: GT V1 (FROZEN 57 OBJS) vs GT V2 (PROPOSED 45 ACTIVE OBJS) ===")
print(f"{'Pipeline':26s} | {'GT Version':15s} | {'GT':4s} | {'Pred':4s} | {'TP':3s} | {'FP':3s} | {'FN':3s} | {'Prec %':7s} | {'Rec %':7s} | {'F1 %':7s}")
print("-" * 105)

for name, raw, conf in pipelines:
    res_v1 = evaluate_pipeline(raw, gt_v1, conf_thresh=conf, iou_thresh=0.5, class_aware=True)
    print(f"{name:26s} | {'v1 (Frozen)':15s} | {res_v1['total_gt']:4d} | {res_v1['total_preds']:4d} | {res_v1['tp']:3d} | {res_v1['fp']:3d} | {res_v1['fn']:3d} | {res_v1['precision']:6.2f}% | {res_v1['recall']:6.2f}% | {res_v1['f1']:6.2f}%")
    
    res_v2 = evaluate_pipeline(raw, gt_v2, conf_thresh=conf, iou_thresh=0.5, class_aware=True, v2_filter='all_active')
    print(f"{name:26s} | {'v2 (Proposed 45)':15s} | {res_v2['total_gt']:4d} | {res_v2['total_preds']:4d} | {res_v2['tp']:3d} | {res_v2['fp']:3d} | {res_v2['fn']:3d} | {res_v2['precision']:6.2f}% | {res_v2['recall']:6.2f}% | {res_v2['f1']:6.2f}%")
    
    res_v2_disc = evaluate_pipeline(raw, gt_v2, conf_thresh=conf, iou_thresh=0.5, class_aware=True, v2_filter='discrete_only')
    print(f"{name:26s} | {'v2 (Discrete 39)':15s} | {res_v2_disc['total_gt']:4d} | {res_v2_disc['total_preds']:4d} | {res_v2_disc['tp']:3d} | {res_v2_disc['fp']:3d} | {res_v2_disc['fn']:3d} | {res_v2_disc['precision']:6.2f}% | {res_v2_disc['recall']:6.2f}% | {res_v2_disc['f1']:6.2f}%")
    print("-" * 105)
