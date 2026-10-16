"""
Measure exact bounding box of the banana peel on scratch_banana_crop2.jpg
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CROP_P = ROOT / "scratch_banana_crop2.jpg"

bgr = cv2.imread(str(CROP_P))
h_crop, w_crop = bgr.shape[:2]

# Crop2 global offset: x0 = 800, y0 = 600
x0, y0 = 800, 600

# Color thresholding in HSV inside crop2
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# Yellow hue range: 15 to 40, saturation >= 60, value >= 70
lower = np.array([12, 60, 70])
upper = np.array([45, 255, 255])
mask = cv2.inRange(hsv, lower, upper)

# Only keep the lower half of crop2 (below y = 500) where the banana peel is
mask[:400, :] = 0

# Clean up mask
kernel = np.ones((9, 9), np.uint8)
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 2000]

all_pts = np.vstack(contours)
cx, cy, cw, ch = cv2.boundingRect(all_pts)

# Translate to full image coordinates
gx1 = x0 + cx
gy1 = y0 + cy
gx2 = gx1 + cw
gy2 = gy1 + ch

print(f"Crop-relative bounding box: x={cx}, y={cy}, w={cw}, h={ch}")
print(f"Full-image bounding box (5184x3456):")
print(f"  x1={gx1}, y1={gy1}, x2={gx2}, y2={gy2}")
print(f"  w={cw}, h={ch}")

# Normalized YOLO coordinates
img_w, img_h = 5184, 3456
xc = (gx1 + gx2) / (2.0 * img_w)
yc = (gy1 + gy2) / (2.0 * img_h)
wn = cw / img_w
hn = ch / img_h
print(f"  YOLO format: 1 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

# Draw and verify
vis = bgr.copy()
cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 4)
cv2.putText(vis, f"1: biological ({cw}x{ch})", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.imwrite(str(ROOT / "scratch_banana_crop_verified.jpg"), vis)
print("Saved scratch_banana_crop_verified.jpg")
