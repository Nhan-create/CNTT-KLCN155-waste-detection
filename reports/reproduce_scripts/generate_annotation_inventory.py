"""
Generates an authoritative, itemized inventory of all 1,562 YOLO annotation files in data/detection/labels.
Includes:
- File relative path
- SHA-256 hash of the label file
- Source (synthetic_mendeley, real_taco, real_openimages)
- Number of boxes
- Box counts per class [0..9]
- Errors and warnings per file
Outputs:
- reports/annotation_inventory.csv
- reports/annotation_inventory_summary.json
"""

import hashlib
import json
import sys
import csv
from collections import Counter
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.validate_detection_annotations import validate_annotation_file, TAXONOMY_10
labels_dir = repo_root / "data" / "detection" / "labels"

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

all_txt_files = sorted(list(labels_dir.glob("**/*.txt")))
print(f"Total annotation files found: {len(all_txt_files)}")

rows = []
total_boxes = 0
global_class_counts = Counter()
files_with_warnings = 0
files_with_errors = 0

for idx, f in enumerate(all_txt_files):
    rel_path = f.relative_to(repo_root).as_posix()
    f_sha = sha256_file(f)

    # Determine source
    name = f.name
    if "synthetic" in rel_path or name.startswith("syn_"):
        source = "synthetic_mendeley"
    elif name.startswith("taco_"):
        source = "real_taco"
    elif name.startswith("oi_"):
        source = "real_openimages"
    else:
        source = "unknown"

    errors, warnings, box_count = validate_annotation_file(f)
    total_boxes += box_count

    # Count classes in file
    file_class_counts = Counter()
    with open(f, "r", encoding="utf-8") as lf:
        for line in lf:
            parts = line.strip().split()
            if parts:
                try:
                    cid = int(parts[0])
                    file_class_counts[cid] += 1
                    global_class_counts[cid] += 1
                except ValueError:
                    pass

    if errors:
        files_with_errors += 1
    if warnings:
        files_with_warnings += 1

    error_str = "; ".join(f"[{e['type']}] L{e['line']}: {e['message']}" for e in errors) if errors else "NONE"
    warning_str = "; ".join(f"[{w['type']}] L{w['line']}: {w['message']}" for w in warnings) if warnings else "NONE"

    rows.append({
        "file_path": rel_path,
        "filename": name,
        "sha256": f_sha,
        "source": source,
        "num_boxes": box_count,
        "class_counts_json": json.dumps(dict(sorted(file_class_counts.items()))),
        "errors": error_str,
        "warnings": warning_str
    })

# Write CSV
csv_out = repo_root / "reports" / "annotation_inventory.csv"
csv_out.parent.mkdir(parents=True, exist_ok=True)
with open(csv_out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file_path", "filename", "sha256", "source", "num_boxes",
        "class_counts_json", "errors", "warnings"
    ])
    writer.writeheader()
    writer.writerows(rows)

# Summary JSON
summary = {
    "total_files": len(rows),
    "total_boxes": total_boxes,
    "files_with_errors": files_with_errors,
    "files_with_warnings": files_with_warnings,
    "boxes_by_class": {
        str(cid): {
            "name": TAXONOMY_10[cid],
            "count": global_class_counts[cid]
        }
        for cid in range(10)
    },
    "by_source": {
        "synthetic_mendeley": {
            "files": sum(1 for r in rows if r["source"] == "synthetic_mendeley"),
            "boxes": sum(r["num_boxes"] for r in rows if r["source"] == "synthetic_mendeley")
        },
        "real_taco": {
            "files": sum(1 for r in rows if r["source"] == "real_taco"),
            "boxes": sum(r["num_boxes"] for r in rows if r["source"] == "real_taco")
        },
        "real_openimages": {
            "files": sum(1 for r in rows if r["source"] == "real_openimages"),
            "boxes": sum(r["num_boxes"] for r in rows if r["source"] == "real_openimages")
        }
    }
}

json_out = repo_root / "reports" / "annotation_inventory_summary.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\n[DONE] Wrote CSV inventory to: {csv_out}")
print(f"[DONE] Wrote JSON summary to: {json_out}")
print(f"Summary:")
print(f"  Total files: {len(rows)}")
print(f"  Total boxes: {total_boxes}")
print(f"  Files with errors: {files_with_errors}")
print(f"  Files with warnings: {files_with_warnings}")
for cid in range(10):
    print(f"  Class {cid} ({TAXONOMY_10[cid]}): {global_class_counts[cid]} boxes")
