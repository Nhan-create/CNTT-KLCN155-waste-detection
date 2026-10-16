import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from PIL import Image

from src.detection.ssdlite import SSDLiteWasteDetector
from src.web.classifier_service import MobileNetClassifier
from src.detection.two_stage import TwoStageDetector, MobileNetBatchAdapter
from src.detection.tiling import TiledDetector
from reports.qa_acceptance.practical_metrics import (
    load_ground_truth,
    precision_recall_f1,
    box_iou,
    CLASS_NAMES,
)

def inspect_boxes():
    gt_dict, _ = load_ground_truth(PROJECT_ROOT / "reports/qa_acceptance/ground_truth_hinh_anh_rac.json")
    ckpt_det = PROJECT_ROOT / "artifacts/detection_v2/E2_ssdlite320_10cls/weights/best.pt"
    ckpt_cls = PROJECT_ROOT / "artifacts/official_run/best_model.pt"

    base_det = SSDLiteWasteDetector(ckpt_det, confidence_threshold=0.20, image_size=320, device="cpu")
    classifier = MobileNetClassifier(ckpt_cls, device="cpu")
    adapter = MobileNetBatchAdapter(classifier)

    # Config Before
    ts_before = TwoStageDetector(
        base_det, adapter, scoring_mode="blend", blend_weight=0.65, min_proposal_gate=0.0, min_score=0.20
    )
    tiled_before = TiledDetector(ts_before, tile_size=640, overlap=0.20, min_side_to_tile=640, adaptive=True)

    # Config After
    ts_after = TwoStageDetector(
        base_det, adapter, scoring_mode="product", min_score=0.20
    )
    tiled_after = TiledDetector(ts_after, tile_size=640, overlap=0.20, min_side_to_tile=800, adaptive=True)

    for cfg_name, pipeline in [("BEFORE", tiled_before), ("AFTER", tiled_after)]:
        print(f"\n=================================================================")
        print(f"PIPELINE: {cfg_name}")
        all_preds = {}
        for fn in gt_dict:
            img_p = Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[1] / "data" / "holdout"))) / fn
            with Image.open(img_p) as im:
                res = pipeline.detect_pil(im)
            all_preds[fn] = [
                {
                    "class_id": d.class_index,
                    "class_name": d.class_id,
                    "confidence": d.confidence,
                    "bbox_xyxy": [d.box.x1, d.box.y1, d.box.x2, d.box.y2]
                }
                for d in res.detections
            ]

        total_boxes = sum(len(v) for v in all_preds.values())
        print(f"Total boxes generated (at min_score=0.20): {total_boxes}")

        boxes_below_025 = []
        boxes_ge_025 = []
        for fn, preds in all_preds.items():
            for p in preds:
                entry = {**p, "filename": fn}
                if p["confidence"] < 0.25:
                    boxes_below_025.append(entry)
                else:
                    boxes_ge_025.append(entry)

        print(f"Boxes with confidence in [0.20, 0.25): {len(boxes_below_025)}")
        for b in boxes_below_025:
            print(f"  [<0.25] {b['filename']}: {b['class_name']} ({b['confidence']:.4f}) box={b['bbox_xyxy']}")

        print(f"Boxes with confidence >= 0.25: {len(boxes_ge_025)}")
        for b in boxes_ge_025:
            print(f"  [>=0.25] {b['filename']}: {b['class_name']} ({b['confidence']:.4f}) box={b['bbox_xyxy']}")

        # Class-aware matching at conf 0.25
        prf = precision_recall_f1(gt_dict, all_preds, conf=0.25, iou_thr=0.5, agnostic=False)
        print(f"At conf=0.25, iou=0.5: TP={prf['tp']}, FP={prf['fp']}, TP+FP={prf['tp']+prf['fp']}")
        print(f"Difference: Total ({total_boxes}) - (TP+FP={prf['tp']+prf['fp']}) = {total_boxes - (prf['tp']+prf['fp'])}")

if __name__ == "__main__":
    inspect_boxes()
