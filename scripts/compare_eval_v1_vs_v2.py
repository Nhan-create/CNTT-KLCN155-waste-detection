# Mô tả: So sánh định lượng kết quả đánh giá giữa các phiên bản pipeline phát hiện.
"""
scripts/compare_eval_v1_vs_v2.py
---------------------------------
Evaluates detection pipeline predictions against Ground Truth versions
with explicit policies for negative benchmark images, excluded images, and ignore regions.

Evaluation Policies:
1. NEGATIVE: Image contains no target waste objects. Tested as hard negative benchmark.
   Predictions are evaluated and counted as False Positives.
2. EXCLUDED: Image is outside instance benchmark scope (e.g. unsegregated bulk dumps).
   Neither ground truth nor predictions are counted towards instance metrics.
3. IGNORE REGION: Designated spatial regions (e.g. unsegregated pile bounding box).
   Predictions falling within these regions are discarded (neither TP nor FP).
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / 'reports' / 'qa_acceptance'


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


def evaluate_pipeline(
    raw_data,
    gt_data,
    conf_thresh=0.25,
    iou_thresh=0.5,
    class_aware=True,
    v2_filter='all_active',
    pile_policy='excluded',        # 'excluded', 'ignore_region', 'negative'
    receptacle_policy='negative',  # 'negative', 'excluded'
    ignore_iou_thresh=0.1,
):
    """
    Evaluates detection predictions against ground truth with explicit image/region policies.
    """
    is_v2 = 'metadata' in gt_data and 'v2' in gt_data.get('metadata', {}).get('version', '')

    gt_map = {}
    ignore_map = {}
    image_effective_policy = {}
    total_gt = 0

    for img in gt_data['images']:
        fn = img['filename']
        eval_policy = img.get('evaluation_policy', 'INSTANCE_BENCHMARK')

        if not is_v2:
            # GT v1 frozen: all images evaluated normally
            effective_policy = 'instance'
        else:
            if eval_policy == 'NEGATIVE_BENCHMARK' or (
                any(o.get('v2_status') == 'QUARANTINED_UTILITY_RECEPTACLE' for o in img['objects'])
                and not any(o.get('is_active_for_instance_eval') for o in img['objects'])
            ):
                effective_policy = receptacle_policy
            elif eval_policy == 'UNSEGREGATED_PILE_EXCLUDED' or (
                any(o.get('v2_status') == 'FLAGGED_UNSEGREGATED_PILE' for o in img['objects'])
                and not any(o.get('is_active_for_instance_eval') for o in img['objects'])
            ):
                effective_policy = pile_policy
            else:
                effective_policy = 'instance'

        image_effective_policy[fn] = effective_policy

        if effective_policy == 'excluded':
            continue

        objs = []
        ignore_boxes = []
        for o in img['objects']:
            if is_v2 and 'is_active_for_instance_eval' in o:
                if v2_filter == 'discrete_only' and o.get('v2_status') != 'ACCEPTED':
                    continue
                if v2_filter == 'all_active' and not o.get('is_active_for_instance_eval', True):
                    if o.get('is_ignore_region') or o.get('v2_status') == 'FLAGGED_UNSEGREGATED_PILE':
                        ignore_boxes.append(o['bbox_xyxy'])
                    continue
            objs.append(o)
            total_gt += 1

        gt_map[fn] = objs
        ignore_map[fn] = ignore_boxes

    images_preds = raw_data['images'] if isinstance(raw_data, dict) and 'images' in raw_data else raw_data

    tp = 0
    fp = 0
    ignored_count = 0
    total_preds_evaluated = 0

    for img in images_preds:
        fn = img['filename']
        eff_policy = image_effective_policy.get(fn, 'instance')
        if eff_policy == 'excluded':
            continue

        gt_objs = gt_map.get(fn, [])
        ignore_boxes = ignore_map.get(fn, [])
        preds = [p for p in img.get('predictions', []) if p['confidence'] >= conf_thresh]
        preds.sort(key=lambda x: -x['confidence'])

        matched_gt = set()
        for p in preds:
            p_box = p['bbox_xyxy']
            p_cls = p['class_id']
            best_iou = 0.0
            best_gt_idx = -1

            for idx, g in enumerate(gt_objs):
                if idx in matched_gt:
                    continue
                if class_aware and p_cls != g['class_id']:
                    continue
                iou = compute_iou(p_box, g['bbox_xyxy'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp += 1
                total_preds_evaluated += 1
                matched_gt.add(best_gt_idx)
            else:
                # Check ignore region policy
                is_ignored = False
                if eff_policy == 'ignore_region' or pile_policy == 'ignore_region':
                    for ibox in ignore_boxes:
                        iou_ign = compute_iou(p_box, ibox)
                        inter_area = max(0.0, min(p_box[2], ibox[2]) - max(p_box[0], ibox[0])) * max(0.0, min(p_box[3], ibox[3]) - max(p_box[1], ibox[1]))
                        pred_area = max(1.0, (p_box[2] - p_box[0]) * (p_box[3] - p_box[1]))
                        containment = inter_area / pred_area
                        if iou_ign >= ignore_iou_thresh or containment >= 0.3:
                            is_ignored = True
                            break

                if is_ignored:
                    ignored_count += 1
                else:
                    fp += 1
                    total_preds_evaluated += 1

    fn_count = total_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'total_gt': total_gt,
        'total_preds': total_preds_evaluated,
        'ignored_preds': ignored_count,
        'tp': tp,
        'fp': fp,
        'fn': fn_count,
        'precision': round(prec * 100, 2),
        'recall': round(rec * 100, 2),
        'f1': round(f1 * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare thresholded detection counts on GT versions')
    parser.add_argument('--output', type=Path, help='Write machine-readable results to this JSON file')
    parser.add_argument('--pile-policy', choices=['excluded', 'ignore_region', 'negative'], default='excluded',
                        help='Policy for unsegregated bulk waste pile images in v2 (default: excluded)')
    parser.add_argument('--receptacle-policy', choices=['negative', 'excluded'], default='negative',
                        help='Policy for trash/recycling receptacle images in v2 (default: negative)')
    args = parser.parse_args()

    inputs = {
        'SSDLite Before Quick Fix': QA_DIR / 'raw_predictions_before_quick_fix.json',
        'SSDLite After Quick Fix': QA_DIR / 'raw_predictions_after_quick_fix.json',
        'YOLOv8n E1': QA_DIR / 'raw_predictions_yolov8n_e1.json',
    }
    gt_paths = {
        'v1_frozen': QA_DIR / 'ground_truth_hinh_anh_rac.json',
        'v2_preliminary_all_negative': QA_DIR / 'ground_truth_hinh_anh_rac_v2_proposed.json',
        'v2_proposed_active': QA_DIR / 'ground_truth_hinh_anh_rac_v2_proposed.json',
        'v2_proposed_discrete_only': QA_DIR / 'ground_truth_hinh_anh_rac_v2_proposed.json',
        'v2_ignore_region_active': QA_DIR / 'ground_truth_hinh_anh_rac_v2_proposed.json',
    }
    loaded = {path: json.loads(path.read_text(encoding='utf-8')) for path in set(inputs.values()) | set(gt_paths.values())}
    rows = []

    for pipeline, raw_path in inputs.items():
        for gt_version, gt_path in gt_paths.items():
            if gt_version == 'v1_frozen':
                p_policy = 'negative'
                r_policy = 'negative'
                v_filter = 'all_active'
            elif gt_version == 'v2_preliminary_all_negative':
                p_policy = 'negative'
                r_policy = 'negative'
                v_filter = 'all_active'
            elif gt_version == 'v2_proposed_active':
                p_policy = args.pile_policy
                r_policy = args.receptacle_policy
                v_filter = 'all_active'
            elif gt_version == 'v2_proposed_discrete_only':
                p_policy = args.pile_policy
                r_policy = args.receptacle_policy
                v_filter = 'discrete_only'
            elif gt_version == 'v2_ignore_region_active':
                p_policy = 'ignore_region'
                r_policy = args.receptacle_policy
                v_filter = 'all_active'

            result = evaluate_pipeline(
                loaded[raw_path], loaded[gt_path],
                conf_thresh=0.25, iou_thresh=0.5,
                class_aware=True,
                v2_filter=v_filter,
                pile_policy=p_policy,
                receptacle_policy=r_policy,
            )
            rows.append({
                'pipeline': pipeline,
                'gt_version': gt_version,
                'pile_policy': p_policy,
                'receptacle_policy': r_policy,
                **result,
            })

    print('Pipeline | GT version | GT | Pred | Ign | TP | FP | FN | Precision % | Recall % | F1 %')
    for row in rows:
        print(f"{row['pipeline']} | {row['gt_version']} | {row['total_gt']} | {row['total_preds']} | {row.get('ignored_preds', 0)} | "
              f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.2f} | "
              f"{row['recall']:.2f} | {row['f1']:.2f}")

    if args.output:
        paths = sorted(set(inputs.values()) | set(gt_paths.values()))
        artifact = {
            'protocol': 'one-to-one greedy by descending confidence, same class, max IoU; confidence >= 0.25, IoU >= 0.50',
            'metric_type': 'operating-threshold precision/recall/F1; not AP or mAP',
            'policies': {
                'v1_frozen': 'All 27 images evaluated, 57 GT objects (frozen baseline)',
                'v2_preliminary_all_negative': 'Legacy v2 preliminary: zero-active pile images penalized as negative FP',
                'v2_proposed_active': f'Proposed v2 (unapproved): receptacles={args.receptacle_policy}, piles={args.pile_policy}, active=45',
                'v2_proposed_discrete_only': f'Proposed v2 (unapproved): receptacles={args.receptacle_policy}, piles={args.pile_policy}, discrete=39',
                'v2_ignore_region_active': 'Proposed v2 (unapproved): pile boxes treated as ignore regions instead of image-level exclusion',
            },
            'inputs': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'results': rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
