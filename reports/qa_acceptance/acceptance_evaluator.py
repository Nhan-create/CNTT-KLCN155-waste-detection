"""DoD Acceptance Evaluation Module for Waste Detection (CNTT-KLCN155).

Evaluates predictions against Frozen Ground Truth on the mandatory acceptance dataset:
data/holdout

DoD Criteria:
1. Coverage: 100% of manifest images evaluated.
2. Valid Bounding Boxes: Coordinates within image dimensions, x1 < x2 and y1 < y2.
3. 1-to-1 Matching (Hungarian algorithm / Greedy IoU matching).
4. Strict DoD Thresholds:
   - False Negatives (FN) = 0
   - False Positives (FP) = 0
   - Wrong-Class Detections = 0
   - IoU >= 0.75 for all matched pairs
   - Intersection / Area(GT) >= 0.99 for all matched pairs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def bbox_area(box: List[float]) -> float:
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    return w * h


def compute_iou(box1: List[float], box2: List[float]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    area1 = bbox_area(box1)
    area2 = bbox_area(box2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_intersection_over_gt(pred_box: List[float], gt_box: List[float]) -> float:
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    inter_w = max(0.0, float(xB - xA))
    inter_h = max(0.0, float(yB - yA))
    inter_area = inter_w * inter_h

    gt_area = bbox_area(gt_box)
    if gt_area <= 0.0:
        return 0.0
    return inter_area / gt_area


def validate_bbox(box: List[float], width: int, height: int) -> Tuple[bool, str]:
    if len(box) != 4:
        return False, f"Expected 4 coordinates, got {len(box)}"
    x1, y1, x2, y2 = box
    if x1 >= x2:
        return False, f"x1 ({x1}) >= x2 ({x2})"
    if y1 >= y2:
        return False, f"y1 ({y1}) >= y2 ({y2})"
    if x1 < 0 or x2 > width:
        return False, f"x bounds [{x1}, {x2}] exceed width {width}"
    if y1 < 0 or y2 > height:
        return False, f"y bounds [{y1}, {y2}] exceed height {height}"
    return True, "OK"


@dataclass
class MatchResult:
    gt_object_id: str
    gt_class_id: int
    gt_class_name: str
    gt_bbox: List[float]
    pred_object_id: Optional[str] = None
    pred_class_id: Optional[int] = None
    pred_class_name: Optional[str] = None
    pred_bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    iou: float = 0.0
    intersection_over_gt: float = 0.0
    is_matched: bool = False
    is_wrong_class: bool = False
    passed_dod: bool = False
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class ImageEvaluationReport:
    image_index: int
    filename: str
    sha256: str
    width: int
    height: int
    total_gt: int
    total_pred: int
    matched_count: int
    fn_count: int
    fp_count: int
    wrong_class_count: int
    passed_all_dod: bool
    matches: List[MatchResult] = field(default_factory=list)
    unmatched_preds: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class QAAcceptanceEvaluator:
    def __init__(
        self,
        ground_truth_path: str | Path,
        manifest_path: str | Path,
        min_iou_dod: float = 0.75,
        min_inter_gt_dod: float = 0.99,
        matching_iou_threshold: float = 0.10,
    ):
        self.gt_path = Path(ground_truth_path)
        self.manifest_path = Path(manifest_path)
        self.min_iou_dod = min_iou_dod
        self.min_inter_gt_dod = min_inter_gt_dod
        self.matching_iou_threshold = matching_iou_threshold

        with open(self.gt_path, "r", encoding="utf-8") as f:
            self.gt_data = json.load(f)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        self.gt_by_filename = {img["filename"]: img for img in self.gt_data["images"]}
        self.manifest_by_filename = {f["filename"]: f for f in self.manifest_data["files"]}
        self.classes = self.gt_data["metadata"]["classes"]

    def match_predictions(
        self,
        gt_objects: List[Dict[str, Any]],
        pred_objects: List[Dict[str, Any]],
        img_w: int,
        img_h: int
    ) -> Tuple[List[MatchResult], List[Dict[str, Any]], List[str]]:
        errors: List[str] = []
        matches: List[MatchResult] = []
        unmatched_preds: List[Dict[str, Any]] = []

        # Validate predicted boxes
        valid_preds: List[Dict[str, Any]] = []
        for i, p in enumerate(pred_objects):
            bbox = p.get("bbox_xyxy", [])
            valid, msg = validate_bbox(bbox, img_w, img_h)
            if not valid:
                errors.append(f"Invalid pred box {bbox} at index {i}: {msg}")
            else:
                valid_preds.append(p)

        n_gt = len(gt_objects)
        n_pred = len(valid_preds)

        if n_gt == 0:
            for p in valid_preds:
                unmatched_preds.append(p)
            return matches, unmatched_preds, errors

        if n_pred == 0:
            for gt in gt_objects:
                mr = MatchResult(
                    gt_object_id=gt["object_id"],
                    gt_class_id=gt["class_id"],
                    gt_class_name=gt["class_name"],
                    gt_bbox=gt["bbox_xyxy"],
                    is_matched=False,
                    passed_dod=False,
                    failure_reasons=["False Negative (Unmatched GT object)"]
                )
                matches.append(mr)
            return matches, unmatched_preds, errors

        # Compute IoU matrix
        iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
        inter_gt_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

        for i, gt in enumerate(gt_objects):
            for j, pred in enumerate(valid_preds):
                iou_matrix[i, j] = compute_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
                inter_gt_matrix[i, j] = compute_intersection_over_gt(pred["bbox_xyxy"], gt["bbox_xyxy"])

        # Perform 1-to-1 matching
        # Cost is -IoU. Preference to same-class matches.
        cost_matrix = -iou_matrix.copy()
        for i, gt in enumerate(gt_objects):
            for j, pred in enumerate(valid_preds):
                if gt["class_id"] != pred["class_id"]:
                    cost_matrix[i, j] += 2.0  # penalty for class mismatch

        if HAS_SCIPY:
            gt_indices, pred_indices = linear_sum_assignment(cost_matrix)
        else:
            # Greedy matching fallback
            matched_gt = set()
            matched_pred = set()
            gt_indices = []
            pred_indices = []
            flat_indices = np.argsort(cost_matrix, axis=None)
            for flat_idx in flat_indices:
                gi, pj = np.unravel_index(flat_idx, cost_matrix.shape)
                if gi not in matched_gt and pj not in matched_pred:
                    matched_gt.add(gi)
                    matched_pred.add(pj)
                    gt_indices.append(gi)
                    pred_indices.append(pj)

        matched_gt_set = set()
        matched_pred_set = set()

        for gi, pj in zip(gt_indices, pred_indices):
            iou = float(iou_matrix[gi, pj])
            inter_gt = float(inter_gt_matrix[gi, pj])
            gt = gt_objects[gi]
            pred = valid_preds[pj]

            # Matching threshold
            if iou < self.matching_iou_threshold:
                continue

            matched_gt_set.add(gi)
            matched_pred_set.add(pj)

            is_wrong_class = (gt["class_id"] != pred["class_id"])
            fail_reasons = []
            if is_wrong_class:
                fail_reasons.append(f"Wrong class: predicted {pred.get('class_name')} ({pred.get('class_id')}) != GT {gt['class_name']} ({gt['class_id']})")
            if iou < self.min_iou_dod:
                fail_reasons.append(f"IoU violation: {iou:.4f} < {self.min_iou_dod}")
            if inter_gt < self.min_inter_gt_dod:
                fail_reasons.append(f"Intersection/GT violation: {inter_gt:.4f} < {self.min_inter_gt_dod}")

            passed = (len(fail_reasons) == 0)

            mr = MatchResult(
                gt_object_id=gt["object_id"],
                gt_class_id=gt["class_id"],
                gt_class_name=gt["class_name"],
                gt_bbox=gt["bbox_xyxy"],
                pred_object_id=pred.get("object_id", f"pred_{pj}"),
                pred_class_id=pred.get("class_id"),
                pred_class_name=pred.get("class_name"),
                pred_bbox=pred.get("bbox_xyxy"),
                confidence=pred.get("confidence", 1.0),
                iou=iou,
                intersection_over_gt=inter_gt,
                is_matched=True,
                is_wrong_class=is_wrong_class,
                passed_dod=passed,
                failure_reasons=fail_reasons
            )
            matches.append(mr)

        # Unmatched GT -> FN
        for gi, gt in enumerate(gt_objects):
            if gi not in matched_gt_set:
                mr = MatchResult(
                    gt_object_id=gt["object_id"],
                    gt_class_id=gt["class_id"],
                    gt_class_name=gt["class_name"],
                    gt_bbox=gt["bbox_xyxy"],
                    is_matched=False,
                    passed_dod=False,
                    failure_reasons=["False Negative (Unmatched GT object)"]
                )
                matches.append(mr)

        # Unmatched Pred -> FP
        for pj, pred in enumerate(valid_preds):
            if pj not in matched_pred_set:
                unmatched_preds.append(pred)

        return matches, unmatched_preds, errors

    def evaluate(self, predictions_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a full set of predictions against the frozen Ground Truth."""
        preds_by_filename = {item["filename"]: item for item in predictions_data.get("images", [])}

        total_manifest_images = len(self.manifest_data["files"])
        evaluated_images = 0
        missing_images = []

        image_reports: List[ImageEvaluationReport] = []
        total_gt = 0
        total_pred = 0
        total_matched = 0
        total_fn = 0
        total_fp = 0
        total_wrong_class = 0
        total_iou_violations = 0
        total_inter_gt_violations = 0
        total_invalid_boxes = 0

        all_ious = []
        all_inter_gts = []

        for mfile in self.manifest_data["files"]:
            fn = mfile["filename"]
            sha256 = mfile["sha256"]
            gt_item = self.gt_by_filename.get(fn)

            if gt_item is None:
                continue

            w = gt_item["width"]
            h = gt_item["height"]
            gt_objs = gt_item["objects"]
            total_gt += len(gt_objs)

            pred_item = preds_by_filename.get(fn)
            if pred_item is None:
                missing_images.append(fn)
                # All GTs are FN
                rep = ImageEvaluationReport(
                    image_index=gt_item["image_index"],
                    filename=fn,
                    sha256=sha256,
                    width=w,
                    height=h,
                    total_gt=len(gt_objs),
                    total_pred=0,
                    matched_count=0,
                    fn_count=len(gt_objs),
                    fp_count=0,
                    wrong_class_count=0,
                    passed_all_dod=False,
                    matches=[
                        MatchResult(
                            gt_object_id=gt["object_id"],
                            gt_class_id=gt["class_id"],
                            gt_class_name=gt["class_name"],
                            gt_bbox=gt["bbox_xyxy"],
                            is_matched=False,
                            passed_dod=False,
                            failure_reasons=["Image missing from predictions"]
                        )
                        for gt in gt_objs
                    ],
                    errors=["Image not present in predictions"]
                )
                image_reports.append(rep)
                total_fn += len(gt_objs)
                continue

            evaluated_images += 1
            pred_objs = pred_item.get("objects", [])
            total_pred += len(pred_objs)

            matches, unmatched_preds, errors = self.match_predictions(gt_objs, pred_objs, w, h)
            total_invalid_boxes += len(errors)

            fn_cnt = sum(1 for m in matches if not m.is_matched)
            fp_cnt = len(unmatched_preds)
            wc_cnt = sum(1 for m in matches if m.is_wrong_class)

            for m in matches:
                if m.is_matched:
                    all_ious.append(m.iou)
                    all_inter_gts.append(m.intersection_over_gt)
                    if m.iou < self.min_iou_dod:
                        total_iou_violations += 1
                    if m.intersection_over_gt < self.min_inter_gt_dod:
                        total_inter_gt_violations += 1

            total_matched += sum(1 for m in matches if m.is_matched)
            total_fn += fn_cnt
            total_fp += fp_cnt
            total_wrong_class += wc_cnt

            passed_dod = (fn_cnt == 0 and fp_cnt == 0 and wc_cnt == 0 and
                          all(m.passed_dod for m in matches) and len(errors) == 0)

            rep = ImageEvaluationReport(
                image_index=gt_item["image_index"],
                filename=fn,
                sha256=sha256,
                width=w,
                height=h,
                total_gt=len(gt_objs),
                total_pred=len(pred_objs),
                matched_count=sum(1 for m in matches if m.is_matched),
                fn_count=fn_cnt,
                fp_count=fp_cnt,
                wrong_class_count=wc_cnt,
                passed_all_dod=passed_dod,
                matches=matches,
                unmatched_preds=unmatched_preds,
                errors=errors
            )
            image_reports.append(rep)

        coverage_ratio = (evaluated_images / total_manifest_images) if total_manifest_images > 0 else 0.0
        dod_passed = (
            coverage_ratio >= 1.0 and
            total_fn == 0 and
            total_fp == 0 and
            total_wrong_class == 0 and
            total_iou_violations == 0 and
            total_inter_gt_violations == 0 and
            total_invalid_boxes == 0
        )

        summary = {
            "dod_status": "PASS" if dod_passed else "FAIL",
            "acceptance_criteria": {
                "coverage_target": "100%",
                "coverage_actual": f"{coverage_ratio * 100:.2f}% ({evaluated_images}/{total_manifest_images})",
                "coverage_passed": (coverage_ratio >= 1.0),
                "fn_target": 0,
                "fn_actual": total_fn,
                "fn_passed": (total_fn == 0),
                "fp_target": 0,
                "fp_actual": total_fp,
                "fp_passed": (total_fp == 0),
                "wrong_class_target": 0,
                "wrong_class_actual": total_wrong_class,
                "wrong_class_passed": (total_wrong_class == 0),
                "min_iou_target": f">= {self.min_iou_dod}",
                "iou_violations_count": total_iou_violations,
                "iou_passed": (total_iou_violations == 0),
                "min_inter_gt_target": f">= {self.min_inter_gt_dod}",
                "inter_gt_violations_count": total_inter_gt_violations,
                "inter_gt_passed": (total_inter_gt_violations == 0),
                "invalid_boxes_count": total_invalid_boxes,
                "invalid_boxes_passed": (total_invalid_boxes == 0)
            },
            "overall_statistics": {
                "total_manifest_images": total_manifest_images,
                "evaluated_images": evaluated_images,
                "missing_images_count": len(missing_images),
                "total_gt_objects": total_gt,
                "total_pred_objects": total_pred,
                "total_matched_pairs": total_matched,
                "mean_iou_matched": float(np.mean(all_ious)) if all_ious else 0.0,
                "mean_inter_gt_matched": float(np.mean(all_inter_gts)) if all_inter_gts else 0.0,
            },
            "missing_images": missing_images,
            "image_reports": [
                {
                    "image_index": r.image_index,
                    "filename": r.filename,
                    "sha256": r.sha256,
                    "width": r.width,
                    "height": r.height,
                    "total_gt": r.total_gt,
                    "total_pred": r.total_pred,
                    "matched_count": r.matched_count,
                    "fn_count": r.fn_count,
                    "fp_count": r.fp_count,
                    "wrong_class_count": r.wrong_class_count,
                    "passed_all_dod": r.passed_all_dod,
                    "errors": r.errors,
                    "matches": [
                        {
                            "gt_object_id": m.gt_object_id,
                            "gt_class": f"{m.gt_class_name}:{m.gt_class_id}",
                            "gt_bbox": m.gt_bbox,
                            "pred_object_id": m.pred_object_id,
                            "pred_class": f"{m.pred_class_name}:{m.pred_class_id}" if m.pred_class_id is not None else None,
                            "pred_bbox": m.pred_bbox,
                            "confidence": m.confidence,
                            "iou": round(m.iou, 4),
                            "intersection_over_gt": round(m.intersection_over_gt, 4),
                            "is_matched": m.is_matched,
                            "is_wrong_class": m.is_wrong_class,
                            "passed_dod": m.passed_dod,
                            "failure_reasons": m.failure_reasons
                        }
                        for m in r.matches
                    ],
                    "unmatched_preds": r.unmatched_preds
                }
                for r in image_reports
            ]
        }
        return summary
