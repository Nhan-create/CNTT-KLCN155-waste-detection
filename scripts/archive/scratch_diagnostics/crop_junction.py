"""
Inspect base of junction boxes in wiki_neg_junction_boxes.jpg
"""
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_neg_junction_boxes.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]

# Crop base of green and black boxes
crop = bgr[int(0.65*h):h, 0:w]
cv2.imwrite(str(ROOT / "scratch_junction_base.jpg"), crop)
print("Saved scratch_junction_base.jpg")
