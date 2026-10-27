# Mô tả: [Lưu trữ — Kịch bản đóng gói & bàn giao Phase 4] Sinh báo cáo, bảng tổng hợp hoặc ảnh đối chiếu trực quan (contact sheets/overlays) cho `mini pilot qa overlays`.
"""
scripts/generate_mini_pilot_qa_overlays.py
Generate visual overlay evidence for all 4 images:
1. wiki_bio_banana_peel_ground.jpg: exact biological bounding box fully enclosing peel to right image edge
2. wiki_bio_street_food_remains.jpg: styrofoam container boxes + conflict annotation
3. wiki_neg_utility_box_leeds.jpg: highlight unconfirmed metallic cylindrical object at cabinet base (PENDING_OBJECT_IDENTITY_REVIEW)
4. wiki_neg_junction_boxes.jpg: full-frame clean confirmation
"""
import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGING_DIR = ROOT / "data" / "acquisition_staging" / "phase4e_staging"
IMAGES_DIR = STAGING_DIR / "images"
OVERLAYS_DIR = STAGING_DIR / "overlays"
OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

# 1. BANANA PEEL (Task 1: corrected bbox fully enclosing peel to right edge)
p1 = IMAGES_DIR / "wiki_bio_banana_peel_ground.jpg"
im1 = cv2.imread(str(p1))
h1, w1 = im1.shape[:2]
# Exact measured box: [2100, 1280, 5184, 2300]
bx1, by1, bx2, by2 = 2100, 1280, 5184, 2300
cv2.rectangle(im1, (bx1, by1), (bx2, by2), (0, 255, 0), 8)
label_text = f"1: biological (banana_peel) [{bx1},{by1},{bx2},{by2}] TOUCHES_RIGHT_EDGE"
cv2.putText(im1, label_text, (bx1, by1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 6)
cv2.putText(im1, "STATUS: PROPOSED_PENDING_USER_APPROVAL", (bx1, by1 - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
out1 = OVERLAYS_DIR / "overlay_wiki_bio_banana_peel_ground.jpg"
cv2.imwrite(str(out1), im1)
print(f"Saved: {out1.name}")

# 2. STREET FOOD REMAINS
p2 = IMAGES_DIR / "wiki_bio_street_food_remains.jpg"
im2 = cv2.imread(str(p2))
h2, w2 = im2.shape[:2]
cv2.rectangle(im2, (0, 420), (450, 1260), (0, 0, 255), 4)
cv2.putText(im2, "7: plastic (styrofoam bowl #1)", (15, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.rectangle(im2, (60, 440), (860, 930), (0, 0, 255), 4)
cv2.putText(im2, "7: plastic (styrofoam bowl #2)", (70, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.putText(im2, "QA VERDICT: REJECT FOR 1:biological (TAXONOMY CONFLICT: EPS PLASTIC CONTAINER)", 
            (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
out2 = OVERLAYS_DIR / "overlay_wiki_bio_street_food_remains.jpg"
cv2.imwrite(str(out2), im2)
print(f"Saved: {out2.name}")

# 3. LEEDS UTILITY BOX (Highlight unconfirmed metallic object, PENDING_OBJECT_IDENTITY_REVIEW)
p3 = IMAGES_DIR / "wiki_neg_utility_box_leeds.jpg"
im3 = cv2.imread(str(p3))
h3, w3 = im3.shape[:2]
can_x1, can_y1, can_x2, can_y2 = 3580, 2800, 4100, 3100
cv2.rectangle(im3, (can_x1, can_y1), (can_x2, can_y2), (0, 165, 255), 8) # Orange box for unconfirmed
cv2.putText(im3, "UNIDENTIFIED METALLIC CYLINDRICAL OBJECT AT CABINET BASE", (can_x1 - 350, can_y1 - 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 165, 255), 5)
cv2.putText(im3, "QA VERDICT: PENDING_OBJECT_IDENTITY_REVIEW (NOT CLEAN NEGATIVE / NOT CONFIRMED METAL)", 
            (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 165, 255), 6)
out3 = OVERLAYS_DIR / "overlay_wiki_neg_utility_box_leeds.jpg"
cv2.imwrite(str(out3), im3)
print(f"Saved: {out3.name}")

# 4. JUNCTION BOXES (Clean hard negative)
p4 = IMAGES_DIR / "wiki_neg_junction_boxes.jpg"
im4 = cv2.imread(str(p4))
cv2.putText(im4, "QA VERDICT: CLEAN HARD-NEGATIVE CANDIDATE (0 BOXES)", 
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(im4, "STATUS: PENDING_USER_APPROVAL (NO PREMATURE LABEL ON DISK)", 
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
out4 = OVERLAYS_DIR / "overlay_wiki_neg_junction_boxes.jpg"
cv2.imwrite(str(out4), im4)
print(f"Saved: {out4.name}")

print("\nAll 4 QA overlays generated successfully.")
