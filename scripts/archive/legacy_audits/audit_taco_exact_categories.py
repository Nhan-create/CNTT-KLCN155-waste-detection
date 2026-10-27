"""
scripts/audit_taco_exact_categories.py
--------------------------------------
Inspects data/audit/taco_annotations_raw.json directly to audit:
- Exact categories: id, name, supercategory
- Total annotations and unique images per category
- Exact counts for Battery and Food/Biological categories
- Verification against public TACO dataset definitions

Outputs: reports/data_v2/taco_exact_categories_audit.json
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TACO_JSON_PATH = ROOT / "data" / "audit" / "taco_annotations_raw.json"
REPORT_DIR = ROOT / "reports" / "data_v2"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Reading TACO annotations from: {TACO_JSON_PATH}")
    if not TACO_JSON_PATH.exists():
        raise FileNotFoundError(f"Missing {TACO_JSON_PATH}")

    with open(TACO_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    print(f"Total Images in TACO: {len(images)}")
    print(f"Total Annotations in TACO: {len(annotations)}")
    print(f"Total Categories in TACO: {len(categories)}")

    cat_lookup = {c["id"]: c for c in categories}
    ann_counts = Counter()
    img_counts = defaultdict(set)

    for a in annotations:
        cid = a["category_id"]
        ann_counts[cid] += 1
        img_counts[cid].add(a["image_id"])

    # Detailed table of all categories
    cat_summary = []
    for c in sorted(categories, key=lambda x: x["id"]):
        cid = c["id"]
        cname = c["name"]
        supercat = c.get("supercategory", "None")
        n_ann = ann_counts[cid]
        n_img = len(img_counts[cid])
        cat_summary.append({
            "category_id": cid,
            "name": cname,
            "supercategory": supercat,
            "annotations_count": n_ann,
            "unique_images_count": n_img
        })

    # Search for battery and biological categories
    battery_cats = []
    biological_cats = []

    for c in cat_summary:
        name_l = c["name"].lower()
        super_l = c["supercategory"].lower()
        if "batter" in name_l or "batter" in super_l:
            battery_cats.append(c)
        if any(w in name_l or w in super_l for w in ["food", "organic", "fruit", "vegetable", "bio"]):
            biological_cats.append(c)

    print("\n--- BATTERY CATEGORIES FOUND ---")
    for bc in battery_cats:
        print(f"  ID {bc['category_id']:2d}: name='{bc['name']}', super='{bc['supercategory']}' -> {bc['annotations_count']} annotations across {bc['unique_images_count']} images")

    print("\n--- BIOLOGICAL / FOOD WASTE CATEGORIES FOUND ---")
    for bg in biological_cats:
        print(f"  ID {bg['category_id']:2d}: name='{bg['name']}', super='{bg['supercategory']}' -> {bg['annotations_count']} annotations across {bg['unique_images_count']} images")

    # Mapping to 10-class project taxonomy
    mapping_yaml = ROOT / "configs" / "detection_source_mapping.yaml"
    mapping_info = {}
    if mapping_yaml.exists():
        import yaml
        with open(mapping_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            mapping_info = cfg.get("sources", {}).get("taco", {}).get("class_mapping", {})

    report = {
        "dataset_name": "TACO (Trash Annotations in Context)",
        "source_file": str(TACO_JSON_PATH.name),
        "total_images": len(images),
        "total_annotations": len(annotations),
        "total_categories": len(categories),
        "battery_categories": battery_cats,
        "biological_categories": biological_cats,
        "all_categories": cat_summary,
        "mapping_to_10_classes": {
            k: v for k, v in mapping_info.items() if v.get("class_id") in [0, 1]
        }
    }

    out_file = REPORT_DIR / "taco_exact_categories_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved exact TACO audit to: {out_file}")

if __name__ == "__main__":
    main()
