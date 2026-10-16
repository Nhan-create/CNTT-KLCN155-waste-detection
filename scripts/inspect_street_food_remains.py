"""
scripts/inspect_street_food_remains.py
Examine the exact objects in wiki_bio_street_food_remains.jpg
"""
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_street_food_remains.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]
print(f"Street food remains image size: {w}x{h}")

# The foreground has several styrofoam bowls with food residue/sauce and wooden skewers
# Let's save a clear crop of the cups/bowls
crop = bgr[int(0.1*h):int(0.85*h), int(0.0*w):int(0.85*w)]
cv2.imwrite(str(ROOT / "scratch_street_food_crop.jpg"), crop)
print("Saved scratch_street_food_crop.jpg")
