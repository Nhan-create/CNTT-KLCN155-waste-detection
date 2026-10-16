"""
Package Task 4 deliverables into a single zip archive in ~/Downloads.
"""

import sys
import zipfile
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    dest_zip = (Path.home() / "Downloads" / "CNTT_KLCN155_Task4_Acquisition_Va_QA.zip")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    
    files_to_pack = [
        (root / "BASELINE_CORRECTION.md", "BASELINE_CORRECTION.md"),
        (root / "TASK4_ACQUISITION_QA_REPORT.md", "TASK4_ACQUISITION_QA_REPORT.md"),
        (root / "data/acquisition_staging/batch_a_mini01/manifest.json", "manifest.json"),
        (root / "reports/source_verification/minibatch01_download_log.json", "minibatch01_download_log.json"),
        (root / "reports/source_verification/contact_sheets/minibatch01_contact_sheet.jpg", "contact_sheets/minibatch01_contact_sheet.jpg"),
        (root / "reports/source_verification/contact_sheets/minibatch01_annotated_contact_sheet.jpg", "contact_sheets/minibatch01_annotated_contact_sheet.jpg"),
    ]
    
    # Add 10 raw images
    images_dir = root / "data/acquisition_staging/batch_a_mini01/images"
    for img_p in images_dir.glob("*.jpg"):
        files_to_pack.append((img_p, f"images/{img_p.name}"))
        
    # Add 10 labels
    labels_dir = root / "data/acquisition_staging/batch_a_mini01/labels"
    for lbl_p in labels_dir.glob("*.txt"):
        files_to_pack.append((lbl_p, f"labels/{lbl_p.name}"))
        
    # Add overlays
    overlays_dir = root / "reports/source_verification/overlays"
    for ov_p in overlays_dir.glob("*.jpg"):
        files_to_pack.append((ov_p, f"overlays/{ov_p.name}"))
        
    print(f"Packaging {len(files_to_pack)} items into {dest_zip}...")
    with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path, arc_name in files_to_pack:
            if src_path.exists():
                zf.write(src_path, arcname=arc_name)
                print(f"  + {arc_name} ({src_path.stat().st_size:,} bytes)")
            else:
                print(f"  ! MISSING: {src_path}")
                
    print(f"\nSuccessfully created: {dest_zip} ({dest_zip.stat().st_size:,} bytes)")

if __name__ == "__main__":
    main()
