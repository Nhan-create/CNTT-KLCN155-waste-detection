"""
scripts/validate_detection_annotations.py
-----------------------------------------
Validates YOLO bounding box annotations for the multi-object waste detection dataset.

Validation Rules:
1. File format: Each line must contain exactly 5 space-separated values:
   <class_id> <x_center> <y_center> <width> <height>
2. Class ID constraint: Integer in [0, 9] (10-class taxonomy).
3. Coordinate normalization:
   0.0 <= x_center <= 1.0
   0.0 <= y_center <= 1.0
   0.0 < width <= 1.0
   0.0 < height <= 1.0
4. Boundary constraints:
   x_center - width/2 >= -0.05 (allowing minor edge bounding)
   x_center + width/2 <= 1.05
   y_center - height/2 >= -0.05
   y_center + height/2 <= 1.05
5. Degeneracy check:
   width > 0.002 and height > 0.002
6. Duplicate box check:
   Detect exact duplicates or IoU > 0.98 for the same class in an image.

Exit Code:
- 0: All files passed or only harmless warnings.
- 1: Critical errors detected (out-of-bounds, invalid classes, unparseable lines).
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

TAXONOMY_10 = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    # box format: [xc, yc, w, h]
    x1_min, x1_max = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    y1_min, y1_max = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    x2_min, x2_max = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    y2_min, y2_max = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def validate_annotation_file(file_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    errors = []
    warnings = []
    box_count = 0

    if not file_path.exists():
        errors.append({
            "file": str(file_path),
            "line": 0,
            "type": "FILE_NOT_FOUND",
            "message": f"Label file does not exist: {file_path}"
        })
        return errors, warnings, 0

    boxes = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append({
                "file": str(file_path),
                "line": idx,
                "type": "INVALID_FIELD_COUNT",
                "message": f"Expected 5 fields, got {len(parts)}: '{raw_line}'"
            })
            continue

        try:
            cls_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError as e:
            errors.append({
                "file": str(file_path),
                "line": idx,
                "type": "PARSE_FLOAT_ERROR",
                "message": f"Could not parse numerical fields: {e}"
            })
            continue

        box_count += 1

        # Class ID
        if cls_id < 0 or cls_id >= len(TAXONOMY_10):
            errors.append({
                "file": str(file_path),
                "line": idx,
                "type": "OUT_OF_BOUNDS_CLASS",
                "message": f"Class ID {cls_id} is outside taxonomy [0, {len(TAXONOMY_10)-1}]"
            })

        # Coordinates
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            errors.append({
                "file": str(file_path),
                "line": idx,
                "type": "CENTER_OUT_OF_RANGE",
                "message": f"Center coordinates ({xc}, {yc}) must be in [0.0, 1.0]"
            })

        if w <= 0.0 or w > 1.0 or h <= 0.0 or h > 1.0:
            errors.append({
                "file": str(file_path),
                "line": idx,
                "type": "DIMENSIONS_OUT_OF_RANGE",
                "message": f"Dimensions w={w}, h={h} must be in (0.0, 1.0]"
            })

        # Degenerate boxes
        if w <= 0.002 or h <= 0.002:
            warnings.append({
                "file": str(file_path),
                "line": idx,
                "type": "DEGENERATE_BOX",
                "message": f"Extremely tiny box: w={w}, h={h}"
            })

        # Boundary overflow
        xmin, xmax = xc - w / 2, xc + w / 2
        ymin, ymax = yc - h / 2, yc + h / 2
        if xmin < -0.05 or xmax > 1.05 or ymin < -0.05 or ymax > 1.05:
            warnings.append({
                "file": str(file_path),
                "line": idx,
                "type": "BBOX_EXTENDS_OFFSCREEN",
                "message": f"Box boundaries [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}] exceed frame"
            })

        boxes.append((cls_id, [xc, yc, w, h], idx))

    # Duplicate box check
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            c1, b1, l1 = boxes[i]
            c2, b2, l2 = boxes[j]
            if c1 == c2:
                iou = calculate_iou(b1, b2)
                if iou > 0.98:
                    warnings.append({
                        "file": str(file_path),
                        "line": f"{l1},{l2}",
                        "type": "NEAR_DUPLICATE_BOX",
                        "message": f"Box on line {l1} and {l2} have IoU {iou:.4f} for class {c1}"
                    })

    return errors, warnings, box_count

def validate_dataset(labels_dir: Path) -> Dict[str, Any]:
    all_label_files = sorted(list(labels_dir.glob("**/*.txt")))
    total_files = len(all_label_files)
    total_boxes = 0
    all_errors = []
    all_warnings = []

    for f in all_label_files:
        errs, warns, bcnt = validate_annotation_file(f)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        total_boxes += bcnt

    summary = {
        "labels_directory": str(labels_dir),
        "total_files_checked": total_files,
        "total_boxes_validated": total_boxes,
        "total_errors": len(all_errors),
        "total_warnings": len(all_warnings),
        "passed": len(all_errors) == 0,
        "errors": all_errors[:100],  # cap to first 100 for readability
        "warnings": all_warnings[:100]
    }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Validate YOLO bounding box annotations.")
    parser.add_argument("--labels-dir", type=str, default="data/detection/labels",
                        help="Directory containing YOLO label files")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save JSON summary report")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Exit code 1 on any error")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    labels_dir = Path(args.labels_dir)
    if not labels_dir.is_absolute():
        labels_dir = project_root / labels_dir

    print(f"[INFO] Validating annotations in: {labels_dir}")
    if not labels_dir.exists():
        print(f"[ERROR] Directory does not exist: {labels_dir}")
        sys.exit(1)

    summary = validate_dataset(labels_dir)

    print(f"\n--- Validation Summary ---")
    print(f"Files checked: {summary['total_files_checked']}")
    print(f"Boxes checked: {summary['total_boxes_validated']}")
    print(f"Errors:        {summary['total_errors']}")
    print(f"Warnings:      {summary['total_warnings']}")
    print(f"Status:        {'PASS' if summary['passed'] else 'FAIL'}")

    if summary["errors"]:
        print("\nErrors encountered (first 10):")
        for err in summary["errors"][:10]:
            print(f"  [{err['type']}] {err['file']}:{err['line']} - {err['message']}")

    if summary["warnings"]:
        print("\nWarnings encountered (first 5):")
        for warn in summary["warnings"][:5]:
            print(f"  [{warn['type']}] {warn['file']}:{warn['line']} - {warn['message']}")

    if args.output_json:
        out_p = Path(args.output_json)
        if not out_p.is_absolute():
            out_p = project_root / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport written to: {out_p}")

    if args.strict and not summary["passed"]:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
