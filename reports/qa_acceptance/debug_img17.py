import cv2

im = cv2.imread("data/holdout/images (17).jpg")
h, w = im.shape[:2]

# Objects in images (17):
# 1. Left bin: x: ~0 to 258, y: ~170 to 365
# 2. Right bin: x: ~248 to 535, y: ~180 to 365
# 3. Snowflake tin can (metal): x: ~0 to 115, y: ~170 to 250
# 4. Red soda can (metal): x: ~195 to 258, y: ~220 to 250
# 5. Green bottle (glass): x: ~68 to 150, y: ~105 to 240
# 6. Clear glass bottle behind green: x: ~41 to 86, y: ~110 to 190
# 7. Clear bottle with green cap: x: ~128 to 170, y: ~145 to 220
# 8. Large clear plastic bottle in left bin: x: ~135 to 255, y: ~115 to 250
# 9. Clear water bottle in right bin (top left): x: ~250 to 330, y: ~18 to 205
# 10. White plastic bottle (dispenser/bleach): x: ~328 to 405, y: ~52 to 200
# 11. Blue water bottle (top right): x: ~400 to 470, y: ~45 to 200
# 12. Plastic pump bottle in front: x: ~240 to 365, y: ~110 to 205
# 13. Crushed can (metal): x: ~360 to 415, y: ~175 to 210
# 14. Clear plastic bottle (far right): x: ~425 to 510, y: ~110 to 210

# Let's write crops
cv2.imwrite("reports/qa_acceptance/debug_17_cans.jpg", im[170:255, 0:260])
cv2.imwrite("reports/qa_acceptance/debug_17_right_top.jpg", im[15:215, 245:515])
print("Saved debug 17 crops")
