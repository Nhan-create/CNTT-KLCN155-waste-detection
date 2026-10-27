# Mô tả: Tiện ích kiểm tra hiển thị nhãn tiếng Việt có dấu trên ảnh kết quả kiểm định QA.
import cv2

im = cv2.imread("data/holdout/Phan-Loai-Rac-Tai-Ng-01.jpg")
h, w = im.shape[:2]

# In 960x720:
# 1. blue_trash_bag:
# approx x: 235 to 480, y: 220 to 640
# 2. red_trash_bag:
# approx x: 350 to 675, y: 445 to 880 (wait, y max is 720) -> y: 445 to 705
# 3. clear_plastic_cup:
# approx x: 590 to 750, y: 710? Let's check coords
# 4. milk box:
# approx x: 685 to 765, y: 710? Let's check
# 5. yellow cap on curb:
# approx x: 680 to 710, y: 380 to 400

# Let's save crops to find exact pixel boundaries
cv2.imwrite("reports/qa_acceptance/debug_vn_bag_blue.jpg", im[210:650, 220:480])
cv2.imwrite("reports/qa_acceptance/debug_vn_bag_red.jpg", im[430:715, 340:680])
cv2.imwrite("reports/qa_acceptance/debug_vn_cup.jpg", im[690:720, 580:760])
cv2.imwrite("reports/qa_acceptance/debug_vn_cap.jpg", im[375:405, 675:715])
print("Saved Phan-Loai crops")
