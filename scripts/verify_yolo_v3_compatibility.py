"""
scripts/verify_yolo_v3_compatibility.py
---------------------------------------
Verifies Ultralytics YOLO dataset compatibility for Dataset v3 Candidate:
1. Validates data.yaml existence and schema (nc: 10, names: 0..9).
2. Verifies 1-to-1 correspondence between images and labels across train/val/test.
3. Tests Ultralytics parser compatibility directly.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = ROOT / "data" / "detection_v3_candidate" / "yolo"
DATA_YAML = YOLO_DIR / "data.yaml"

EXPECTED_CLASSES = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash"
]

def verify_yolo_compatibility() -> dict:
    results = {
        "data_yaml_exists": False,
        "schema_valid": False,
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": 0,
        "orphaned_labels": 0,
        "ultralytics_parsable": False,
        "errors": []
    }
    
    # 1. Check data.yaml
    if not DATA_YAML.exists():
        results["errors"].append(f"Missing data.yaml at {DATA_YAML}")
        return results
    results["data_yaml_exists"] = True
    
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    nc = cfg.get("nc")
    names = cfg.get("names")
    
    if nc != 10:
        results["errors"].append(f"nc is {nc}, expected 10")
    if isinstance(names, dict):
        class_list = [names[i] for i in range(10) if i in names]
    elif isinstance(names, list):
        class_list = names
    else:
        class_list = []
        
    if class_list != EXPECTED_CLASSES:
        results["errors"].append(f"Class names mismatch: {class_list} vs {EXPECTED_CLASSES}")
        
    if not results["errors"]:
        results["schema_valid"] = True

    # 2. Check 1-to-1 images and labels
    for sp in ["train", "val", "test"]:
        img_dir = YOLO_DIR / "images" / sp
        lbl_dir = YOLO_DIR / "labels" / sp
        
        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))
        
        img_stems = {p.stem: p for p in imgs}
        lbl_stems = {p.stem: p for p in lbls}
        
        missing = [stem for stem in img_stems if stem not in lbl_stems]
        orphaned = [stem for stem in lbl_stems if stem not in img_stems]
        
        results["splits"][sp] = {
            "images_count": len(imgs),
            "labels_count": len(lbls),
            "missing_labels_count": len(missing),
            "orphaned_labels_count": len(orphaned)
        }
        results["total_images"] += len(imgs)
        results["total_labels"] += len(lbls)
        results["missing_labels"] += len(missing)
        results["orphaned_labels"] += len(orphaned)
        
        if missing:
            results["errors"].append(f"Split {sp} has {len(missing)} images without label files")
        if orphaned:
            results["errors"].append(f"Split {sp} has {len(orphaned)} labels without image files")

    # 3. Test Ultralytics dataset resolution
    try:
        from ultralytics.data.utils import check_det_dataset
        dataset_info = check_det_dataset(str(DATA_YAML))
        results["ultralytics_parsable"] = True
        results["ultralytics_info"] = {
            "train": str(dataset_info.get("train", "")),
            "val": str(dataset_info.get("val", "")),
            "test": str(dataset_info.get("test", "")),
            "nc": dataset_info.get("nc")
        }
    except Exception as e:
        results["errors"].append(f"Ultralytics check_det_dataset failed: {e}")
        
    return results

if __name__ == "__main__":
    print("Verifying YOLO dataset compatibility...")
    res = verify_yolo_compatibility()
    print("Results:", yaml.dump(res, sort_keys=False))
    if res["schema_valid"] and res["missing_labels"] == 0 and res["ultralytics_parsable"]:
        print(">>> YOLO V3 COMPATIBILITY: 100% GREEN <<<")
    else:
        print(">>> YOLO V3 COMPATIBILITY: FAILED <<<")
