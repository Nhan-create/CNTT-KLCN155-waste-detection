"""
Inspect individual contours and bounding boxes
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([12, 40, 50])
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask[:int(0.50 * h), :] = 0
mask[int(0.85 * h):, :] = 0

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in contours]
areas.sort(key=lambda x: x[0], reverse=True)

for i, (area, bbox) in enumerate(areas[:10]):
    x, y, bw, bh = bbox
    print(f"Contour {i}: area={area:.0f}, bbox=({x}, {y}, {bw}, {bh}), x2={x+bw}, y2={y+bh}")
