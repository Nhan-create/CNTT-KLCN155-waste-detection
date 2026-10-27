# Mô tả: Tính toán bộ chỉ số nghiệm thu thực tế (AP, Precision, Recall theo ngưỡng IoU) trên tập đối chứng data/holdout.
"""Practical acceptance metrics for the data/holdout holdout (CNTT-KLCN155).

The strict DoD in ``acceptance_evaluator.py`` (FN=0, FP=0, wrong-class=0,
IoU>=0.75, Intersection/GT>=0.99) is kept unchanged as the *ideal* criterion.
This module adds standard, graded detection metrics that are more informative
for comparing models:

* COCO-style AP50 and mAP50-95 (101-point interpolated, per class + mean).
* Class-agnostic AP50 (localisation quality independent of class).
* Precision / Recall / F1 @ IoU 0.5 at a given confidence threshold
  (class-aware, greedy one-to-one matching by descending confidence, COCO rule).
* Class accuracy on localised pairs (class-agnostic one-to-one matching with
  IoU >= 0.5, greedy by descending IoU) and an 11x11 confusion matrix
  (10 classes + background; rows = GT, columns = prediction).
* Configurable practical acceptance criteria
  (default Recall@0.5 >= 0.60, Precision@0.5 >= 0.60, class accuracy >= 0.75).

Accepted prediction inputs (auto-detected by :func:`load_predictions`):

1. Simple mapping ``{filename: [{"class_id", "confidence", "bbox_xyxy"}, ...]}``
   (optionally wrapped as ``{"predictions": {...}}``).
2. Evaluator format ``{"images": [{"filename", "objects": [...]}, ...]}``.
3. ``baseline_results_*.json`` produced by ``run_baseline_evaluation.py``
   (predictions are reconstructed from ``matched_pairs`` + ``false_positives``).

Class ids are 0..9 in the fixed order of AGENTS.md (YOLO convention; SSDLite
outputs must already be decoded, i.e. foreground label - 1).

Limitation: baseline files only store detections with confidence >= the
inference threshold used at generation time (0.20 for current baselines).
AP values computed from such files are therefore truncated (low-confidence tail
missing) and are a lower bound of the true AP; P/R/F1 are only valid for
thresholds >= that stored minimum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

CLASS_NAMES: List[str] = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
NUM_CLASSES = len(CLASS_NAMES)
BACKGROUND_INDEX = NUM_CLASSES  # row/col index for background in confusion matrix
COCO_IOU_THRESHOLDS: np.ndarray = np.linspace(0.5, 0.95, 10)
RECALL_POINTS: np.ndarray = np.linspace(0.0, 1.0, 101)

DEFAULT_GT_PATH = Path(__file__).resolve().parent / "ground_truth_hinh_anh_rac.json"
DEFAULT_CRITERIA: Dict[str, float] = {
    "min_recall_at_iou50": 0.60,
    "min_precision_at_iou50": 0.60,
    "min_class_accuracy_on_matched": 0.75,
}

Prediction = Dict[str, Any]
GroundTruth = Dict[str, Dict[str, Any]]  # filename -> {"objects": [...], "sha256": ...}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _norm_pred(p: Dict[str, Any]) -> Prediction:
    return {
        "class_id": int(p["class_id"]),
        "confidence": float(p.get("confidence", p.get("pred_confidence", 1.0))),
        "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
    }


def predictions_from_baseline(data: Dict[str, Any]) -> Dict[str, List[Prediction]]:
    """Reconstruct the full prediction list per image from a baseline_results_*.json."""
    out: Dict[str, List[Prediction]] = {}
    for case in data["cases"]:
        matching = case.get("matching", {})
        preds: List[Prediction] = []
        for pair in matching.get("matched_pairs", []):
            preds.append({
                "class_id": int(pair["pred_class_id"]),
                "confidence": float(pair["pred_confidence"]),
                "bbox_xyxy": [float(v) for v in pair["pred_bbox"]],
            })
        for fp in matching.get("false_positives", []):
            preds.append(_norm_pred(fp))
        expected = case.get("pred_count")
        if expected is not None and expected != len(preds):
            raise ValueError(
                f"Baseline reconstruction mismatch for {case['filename']}: "
                f"pred_count={expected} but reconstructed {len(preds)}"
            )
        out[case["filename"]] = preds
    return out


def load_predictions(path: str | Path) -> Tuple[Dict[str, List[Prediction]], Dict[str, Any]]:
    """Load predictions from any supported format. Returns (preds, source_info)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    info: Dict[str, Any] = {"path": str(path)}
    if isinstance(data, dict) and "cases" in data:
        info.update(format="baseline_results",
                    backend=data.get("backend"),
                    checkpoint_sha256=data.get("checkpoint_sha256"),
                    stored_min_confidence=data.get("confidence_threshold"))
        return predictions_from_baseline(data), info
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        info.update(format="evaluator_images",
                    stored_min_confidence=data.get("confidence_threshold"))
        return {img["filename"]: [_norm_pred(o) for o in img.get("objects", [])]
                for img in data["images"]}, info
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        info.update(format="simple_wrapped",
                    stored_min_confidence=data.get("confidence_threshold"),
                    backend=data.get("backend"),
                    checkpoint_sha256=data.get("checkpoint_sha256"))
        data = data["predictions"]
    else:
        info.update(format="simple")
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError(f"Unrecognised predictions format in {path}")
    return {fn: [_norm_pred(p) for p in plist] for fn, plist in data.items()}, info


def load_ground_truth(path: str | Path = DEFAULT_GT_PATH, dedupe_sha256: bool = False
                      ) -> Tuple[GroundTruth, List[str]]:
    """Load frozen GT. If dedupe_sha256, keep only the first image per SHA-256.

    Returns (gt_by_filename, excluded_duplicate_filenames).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt: GroundTruth = {}
    seen: Dict[str, str] = {}
    excluded: List[str] = []
    for img in data["images"]:
        sha = img.get("sha256")
        if dedupe_sha256 and sha and sha in seen:
            excluded.append(img["filename"])
            continue
        if sha:
            seen[sha] = img["filename"]
        gt[img["filename"]] = {
            "sha256": sha,
            "width": img.get("width"),
            "height": img.get("height"),
            "objects": [{"class_id": int(o["class_id"]),
                         "bbox_xyxy": [float(v) for v in o["bbox_xyxy"]]}
                        for o in img["objects"]],
        }
    return gt, excluded


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def _match_by_confidence(gts: List[Dict[str, Any]], preds: List[Prediction],
                         iou_thr: float) -> Tuple[List[bool], int]:
    """COCO greedy matching inside one image & one class group.

    Predictions are processed by descending confidence; each takes the unmatched GT
    with the highest IoU (>= iou_thr). Returns (is_tp per pred in sorted order, n_gt).
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i]["confidence"])
    used = [False] * len(gts)
    flags: List[bool] = []
    for pi in order:
        best_iou, best_g = iou_thr, -1
        for gi, g in enumerate(gts):
            if used[gi]:
                continue
            iou = box_iou(preds[pi]["bbox_xyxy"], g["bbox_xyxy"])
            if iou >= best_iou:
                best_iou, best_g = iou, gi
        if best_g >= 0:
            used[best_g] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags, len(gts)


def average_precision_101(scores: Sequence[float], is_tp: Sequence[bool], n_pos: int) -> float:
    """COCO 101-point interpolated AP from scored detections."""
    if n_pos == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")
    tp = np.asarray(is_tp, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_pos
    precision = tp_c / np.maximum(tp_c + fp_c, np.finfo(np.float64).eps)
    # precision envelope (monotonically non-increasing)
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    q = np.array([precision[i] if i < len(precision) else 0.0 for i in idx])
    return float(q.mean())


def _collect(gt: GroundTruth, preds: Dict[str, List[Prediction]], iou_thr: float,
             class_id: Optional[int], agnostic: bool) -> Tuple[List[float], List[bool], int]:
    scores: List[float] = []
    flags: List[bool] = []
    n_pos = 0
    for fn, g in gt.items():
        gobjs = g["objects"] if (agnostic or class_id is None) else \
            [o for o in g["objects"] if o["class_id"] == class_id]
        pobjs = preds.get(fn, [])
        if not agnostic and class_id is not None:
            pobjs = [p for p in pobjs if p["class_id"] == class_id]
        f, n = _match_by_confidence(gobjs, pobjs, iou_thr)
        pobjs_sorted = sorted(pobjs, key=lambda p: -p["confidence"])
        scores.extend(p["confidence"] for p in pobjs_sorted)
        flags.extend(f)
        n_pos += n
    return scores, flags, n_pos


def compute_ap_table(gt: GroundTruth, preds: Dict[str, List[Prediction]]) -> Dict[str, Any]:
    """Per-class AP50 / AP75 / AP50-95, mean over classes present in GT, plus agnostic AP50."""
    per_class: Dict[str, Any] = {}
    ap50s, ap5095s = [], []
    for c, name in enumerate(CLASS_NAMES):
        aps = []
        n_pos = 0
        for t in COCO_IOU_THRESHOLDS:
            s, f, n_pos = _collect(gt, preds, float(t), c, agnostic=False)
            aps.append(average_precision_101(s, f, n_pos))
        n_pred = sum(1 for fn in gt for p in preds.get(fn, []) if p["class_id"] == c)
        entry = {"n_gt": n_pos, "n_pred": n_pred}
        if n_pos == 0:
            entry.update(ap50=None, ap75=None, ap50_95=None, note="no GT instances; excluded from mean")
        else:
            entry.update(ap50=aps[0], ap75=aps[5], ap50_95=float(np.mean(aps)))
            ap50s.append(aps[0])
            ap5095s.append(float(np.mean(aps)))
        per_class[name] = entry
    s, f, n = _collect(gt, preds, 0.5, None, agnostic=True)
    agn50 = average_precision_101(s, f, n)
    agn_all = [average_precision_101(*_collect(gt, preds, float(t), None, agnostic=True))
               for t in COCO_IOU_THRESHOLDS]
    return {
        "per_class": per_class,
        "mAP50": float(np.mean(ap50s)) if ap50s else 0.0,
        "mAP50_95": float(np.mean(ap5095s)) if ap5095s else 0.0,
        "classes_in_mean": len(ap50s),
        "class_agnostic_AP50": agn50,
        "class_agnostic_AP50_95": float(np.mean(agn_all)),
    }


def filter_by_confidence(preds: Dict[str, List[Prediction]], conf: float
                         ) -> Dict[str, List[Prediction]]:
    return {fn: [p for p in plist if p["confidence"] >= conf] for fn, plist in preds.items()}


def precision_recall_f1(gt: GroundTruth, preds: Dict[str, List[Prediction]],
                        conf: float, iou_thr: float = 0.5, agnostic: bool = False
                        ) -> Dict[str, Any]:
    """Class-aware (or agnostic) P/R/F1 at a confidence threshold."""
    fp_ = filter_by_confidence(preds, conf)
    tp = fp = n_gt = 0
    for fn, g in gt.items():
        pl = fp_.get(fn, [])
        if agnostic:
            f, n = _match_by_confidence(g["objects"], pl, iou_thr)
            tp += sum(f); fp += len(f) - sum(f); n_gt += n
            continue
        for c in range(NUM_CLASSES):
            go = [o for o in g["objects"] if o["class_id"] == c]
            po = [p for p in pl if p["class_id"] == c]
            f, n = _match_by_confidence(go, po, iou_thr)
            tp += sum(f); fp += len(f) - sum(f); n_gt += n
    fn_ = n_gt - tp
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"confidence_threshold": conf, "iou_threshold": iou_thr, "class_agnostic": agnostic,
            "tp": tp, "fp": fp, "fn": fn_, "n_gt": n_gt,
            "precision": p, "recall": r, "f1": f1}


def localization_class_analysis(gt: GroundTruth, preds: Dict[str, List[Prediction]],
                                conf: float, iou_thr: float = 0.5) -> Dict[str, Any]:
    """Class-agnostic 1-1 matching (greedy by IoU desc) -> class accuracy + confusion matrix."""
    fp_ = filter_by_confidence(preds, conf)
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)
    matched = correct = 0
    for fn, g in gt.items():
        gobjs, pobjs = g["objects"], fp_.get(fn, [])
        cand = []
        for gi, go in enumerate(gobjs):
            for pi, po in enumerate(pobjs):
                iou = box_iou(go["bbox_xyxy"], po["bbox_xyxy"])
                if iou >= iou_thr:
                    cand.append((iou, po["confidence"], gi, pi))
        cand.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
        ug, up = set(), set()
        for _, _, gi, pi in cand:
            if gi in ug or pi in up:
                continue
            ug.add(gi); up.add(pi)
            gc, pc = gobjs[gi]["class_id"], pobjs[pi]["class_id"]
            cm[gc, pc] += 1
            matched += 1
            correct += int(gc == pc)
        for gi, go in enumerate(gobjs):
            if gi not in ug:
                cm[go["class_id"], BACKGROUND_INDEX] += 1
        for pi, po in enumerate(pobjs):
            if pi not in up:
                cm[BACKGROUND_INDEX, po["class_id"]] += 1
    labels = CLASS_NAMES + ["background"]
    return {
        "confidence_threshold": conf,
        "iou_threshold": iou_thr,
        "matched_pairs": matched,
        "correct_class_pairs": correct,
        "class_accuracy_on_matched": (correct / matched) if matched else 0.0,
        "confusion_matrix": {"labels": labels, "rows": "ground_truth", "cols": "prediction",
                             "matrix": cm.tolist()},
    }


def evaluate_practical(gt: GroundTruth, preds: Dict[str, List[Prediction]],
                       conf: float = 0.25,
                       criteria: Optional[Dict[str, float]] = None,
                       sweep: Iterable[float] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
                       stored_min_confidence: Optional[float] = None) -> Dict[str, Any]:
    crit = dict(DEFAULT_CRITERIA)
    crit.update(criteria or {})
    missing = sorted(fn for fn in gt if fn not in preds)
    extra = sorted(fn for fn in preds if fn not in gt)
    prf = precision_recall_f1(gt, preds, conf)
    prf_agn = precision_recall_f1(gt, preds, conf, agnostic=True)
    loc = localization_class_analysis(gt, preds, conf)
    checks = {
        "recall_at_iou50": {"actual": prf["recall"], "target": f">= {crit['min_recall_at_iou50']}",
                            "passed": prf["recall"] >= crit["min_recall_at_iou50"]},
        "precision_at_iou50": {"actual": prf["precision"],
                               "target": f">= {crit['min_precision_at_iou50']}",
                               "passed": prf["precision"] >= crit["min_precision_at_iou50"]},
        "class_accuracy_on_matched": {"actual": loc["class_accuracy_on_matched"],
                                      "target": f">= {crit['min_class_accuracy_on_matched']}",
                                      "passed": loc["class_accuracy_on_matched"]
                                      >= crit["min_class_accuracy_on_matched"]},
        "coverage_all_images_predicted": {"actual": len(gt) - len(missing), "target": len(gt),
                                          "passed": not missing},
    }
    warnings = []
    if stored_min_confidence is not None:
        warnings.append(f"Predictions only contain detections with confidence >= "
                        f"{stored_min_confidence}; AP is truncated (lower bound) and P/R/F1 are "
                        f"only valid for thresholds >= {stored_min_confidence}.")
        if conf < stored_min_confidence:
            warnings.append(f"Requested confidence {conf} < stored minimum {stored_min_confidence}.")
    return {
        "n_images": len(gt),
        "n_gt_objects": sum(len(g["objects"]) for g in gt.values()),
        "n_predictions_total": sum(len(preds.get(fn, [])) for fn in gt),
        "missing_prediction_images": missing,
        "ignored_prediction_images_not_in_gt": extra,
        "ap": compute_ap_table(gt, preds),
        "operating_point": {"class_aware": prf, "class_agnostic": prf_agn},
        "localization_class_analysis": loc,
        "confidence_sweep": [
            {**{k: v for k, v in precision_recall_f1(gt, preds, c).items()
                if k in ("confidence_threshold", "tp", "fp", "fn", "precision", "recall", "f1")},
             "class_accuracy_on_matched":
                 localization_class_analysis(gt, preds, c)["class_accuracy_on_matched"]}
            for c in sweep
        ],
        "practical_acceptance": {
            "criteria": crit,
            "checks": checks,
            "status": "PASS" if all(v["passed"] for v in checks.values()) else "FAIL",
        },
        "warnings": warnings,
    }


def strict_dod_summary(preds: Dict[str, List[Prediction]], conf: float,
                       gt_path: Path, manifest_path: Path) -> Optional[Dict[str, Any]]:
    """Strict (ideal) DoD via the unchanged QAAcceptanceEvaluator, for reference."""
    try:
        from reports.qa_acceptance.acceptance_evaluator import QAAcceptanceEvaluator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from acceptance_evaluator import QAAcceptanceEvaluator  # type: ignore
    if not manifest_path.exists():
        return None
    ev = QAAcceptanceEvaluator(gt_path, manifest_path)
    images = [{"filename": fn,
               "objects": [{**p, "class_name": CLASS_NAMES[p["class_id"]]}
                           for p in plist if p["confidence"] >= conf]}
              for fn, plist in preds.items()]
    res = ev.evaluate({"images": images})
    return {"dod_status": res["dod_status"], "acceptance_criteria": res["acceptance_criteria"],
            "note": "Ideal criterion (unchanged). Computed on all manifest images."}


def run(predictions_path: Path, out_path: Optional[Path], gt_path: Path = DEFAULT_GT_PATH,
        conf: float = 0.25, criteria: Optional[Dict[str, float]] = None,
        manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    preds, info = load_predictions(predictions_path)
    stored = info.get("stored_min_confidence")
    result: Dict[str, Any] = {"source": info, "ground_truth": str(gt_path),
                              "operating_confidence": conf, "views": {}}
    for key, dedupe in (("all_27", False), ("dedup_sha256", True)):
        gt, excluded = load_ground_truth(gt_path, dedupe_sha256=dedupe)
        view = evaluate_practical(gt, preds, conf=conf, criteria=criteria,
                                  stored_min_confidence=stored)
        view["excluded_duplicates"] = excluded
        result["views"][key] = view
    mp = manifest_path or (gt_path.parent / "manifest_hinh_anh_rac.json")
    result["strict_dod_reference"] = strict_dod_summary(preds, conf, gt_path, mp)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def _print_summary(res: Dict[str, Any]) -> None:
    for key, v in res["views"].items():
        ap, op = v["ap"], v["operating_point"]["class_aware"]
        loc = v["localization_class_analysis"]
        print(f"[{key}] images={v['n_images']} gt={v['n_gt_objects']} "
              f"mAP50={ap['mAP50']:.4f} mAP50-95={ap['mAP50_95']:.4f} "
              f"agnAP50={ap['class_agnostic_AP50']:.4f} | conf>={op['confidence_threshold']}: "
              f"P={op['precision']:.4f} R={op['recall']:.4f} F1={op['f1']:.4f} "
              f"clsAcc={loc['class_accuracy_on_matched']:.4f} ({loc['matched_pairs']} pairs) "
              f"-> practical {v['practical_acceptance']['status']}")
    if res.get("strict_dod_reference"):
        print(f"[strict DoD] {res['strict_dod_reference']['dod_status']}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Practical acceptance metrics on data/holdout")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT_PATH)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25, help="operating confidence threshold")
    ap.add_argument("--min-recall", type=float, default=DEFAULT_CRITERIA["min_recall_at_iou50"])
    ap.add_argument("--min-precision", type=float,
                    default=DEFAULT_CRITERIA["min_precision_at_iou50"])
    ap.add_argument("--min-class-acc", type=float,
                    default=DEFAULT_CRITERIA["min_class_accuracy_on_matched"])
    a = ap.parse_args(argv)
    res = run(a.predictions, a.out, gt_path=a.gt, conf=a.conf, manifest_path=a.manifest,
              criteria={"min_recall_at_iou50": a.min_recall,
                        "min_precision_at_iou50": a.min_precision,
                        "min_class_accuracy_on_matched": a.min_class_acc})
    _print_summary(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
