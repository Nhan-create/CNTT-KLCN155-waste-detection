# Mô tả: [Lưu trữ — Tiện ích chẩn đoán cục bộ] Đo đạc tọa độ điểm ảnh và cắt vùng mẫu kiểm tra cho kịch bản `locate_banana_peel_exact`.
"""
scripts/locate_banana_peel_exact.py
Find exact pixel boundaries of the banana peel in wiki_bio_banana_peel_ground.jpg
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
IMG_P = ROOT / "data" / "acquisition_staging" / "phase4e_staging" / "images" / "wiki_bio_banana_peel_ground.jpg"

bgr = cv2.imread(str(IMG_P))
h, w = bgr.shape[:2]

# Convert to HSV to detect the yellow banana peel
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Yellow/banana color in HSV: Hue ~15-35, Saturation > 50, Value > 50
lower_yellow = np.array([12, 40, 50])
upper_yellow = np.array([45, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Focus only on the foreground / middle where the peel is located (y from 0.5*h to 0.85*h)
# Zero out background (people/houses at the top)
mask[:int(0.50 * h), :] = 0
mask[int(0.85 * h):, :] = 0

# Morphological clean up
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

# Find contours
contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Filter contours by size, combine nearby peel parts
    peel_contours = [c for c in contours if cv2.contourArea(c) > 5000]
    all_points = np.vstack([c for c in peel_contours])
    x, y, bw, bh = cv2.boundingRect(all_points)

    print(f"Detected Banana Peel Bounding Box (pixels):")
    print(f"  x1={x}, y1={y}, x2={x+bw}, y2={y+bh}")
    print(f"  width={bw}, height={bh} (Image: {w}x{h})")
    
    # Normalized YOLO format: xc, yc, w_norm, h_norm
    xc = (x + bw / 2.0) / w
    yc = (y + bh / 2.0) / h
    w_norm = bw / w
    h_norm = bh / h
    print(f"  Normalized YOLO: {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}")

    # Draw on copy and save
    vis = bgr.copy()
    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 6)
    cv2.putText(vis, "1: biological", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # Save visual verification crop
    pad = 100
    y1_p = max(0, y - pad)
    y2_p = min(h, y + bh + pad)
    x1_p = max(0, x - pad)
    x2_p = min(w, x + bw + pad)
    crop_vis = vis[y1_p:y2_p, x1_p:x2_p]
    cv2.imwrite(str(ROOT / "scratch_banana_detected.jpg"), crop_vis)
    print("Saved scratch_banana_detected.jpg")
