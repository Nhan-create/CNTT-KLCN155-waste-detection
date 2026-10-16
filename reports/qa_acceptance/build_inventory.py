import os
import sys
import json
import hashlib
from pathlib import Path
from PIL import Image
import cv2

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def scan_acceptance_images(root_dir: str, output_path: str):
    root = Path(root_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    all_files = sorted(list(root.rglob("*")))
    file_records = []
    
    total_files = 0
    valid_images = 0
    corrupt_images = 0

    for p in all_files:
        if not p.is_file():
            continue
        total_files += 1
        rel_path = str(p.relative_to(root))
        size_bytes = p.stat().st_size
        sha256 = sha256_file(p)
        
        # PIL decode check
        pil_ok = False
        pil_info = {}
        pil_err = None
        try:
            with Image.open(p) as img:
                pil_ok = True
                pil_info = {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                }
        except Exception as e:
            pil_err = str(e)
            
        # OpenCV decode check
        cv_ok = False
        cv_info = {}
        cv_err = None
        try:
            # handle non-ascii or windows path safely using imdecode
            with open(p, "rb") as f:
                bytes_data = bytearray(f.read())
            import numpy as np
            arr = np.asarray(bytes_data, dtype=np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                cv_ok = True
                cv_info = {
                    "height": cv_img.shape[0],
                    "width": cv_img.shape[1],
                    "channels": cv_img.shape[2] if len(cv_img.shape) > 2 else 1
                }
            else:
                cv_err = "cv2.imdecode returned None"
        except Exception as e:
            cv_err = str(e)

        if pil_ok and cv_ok:
            status = "VALID"
            valid_images += 1
        elif pil_ok or cv_ok:
            status = "PARTIAL_DECODE"
            corrupt_images += 1
        else:
            status = "CORRUPT"
            corrupt_images += 1

        rec = {
            "index": total_files,
            "filename": p.name,
            "relative_path": rel_path.replace("\\", "/"),
            "absolute_path": str(p),
            "file_extension": p.suffix.lower(),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "status": status,
            "pil_decode": {
                "success": pil_ok,
                "info": pil_info,
                "error": pil_err
            },
            "opencv_decode": {
                "success": cv_ok,
                "info": cv_info,
                "error": cv_err
            }
        }
        file_records.append(rec)

    manifest = {
        "metadata": {
            "root_directory": str(root),
            "total_files_scanned": total_files,
            "valid_images_count": valid_images,
            "corrupt_images_count": corrupt_images,
            "generator": "Agent_D_QA",
            "timestamp": "2026-10-09T03:00:00Z"
        },
        "files": file_records
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"Scanned {total_files} files: {valid_images} valid, {corrupt_images} corrupt.")
    print(f"Manifest saved to: {out_p}")

if __name__ == "__main__":
    scan_acceptance_images("data/holdout", "reports/qa_acceptance/manifest_hinh_anh_rac.json")
