"""Inspect visual content of controversial pairs:
Pair 10, Pair 12, Pair 18, Pair 11, Pair 14.
"""

from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(r"D:\HK7\Đồ án khóa luận")

pairs_to_check = [
    (1, "data/raw/garbage_v2/metal/metal_155.jpg", "data/raw/vn_trash/Alu/train_AluCan94.jpg"),
    (2, "data/raw/garbage_v2/metal/metal_186.jpg", "data/raw/vn_trash/Alu/train_AluCan105.jpg"),
    (3, "data/raw/vn_trash/Alu/train_AluCan773.jpg", "data/raw/vn_trash/Alu/train_AluCan770.jpg"),
    (4, "data/raw/vn_trash/Alu/train_beverage_cans373.jpg", "data/raw/vn_trash/Alu/train_beverage_cans357.jpg"),
    (5, "data/raw/vn_trash/Alu/train_beverage_cans701.jpg", "data/raw/garbage_v2/metal/metal_183.jpg"),
    (6, "data/raw/garbage_v2/paper/paper_1118.jpg", "data/raw/vn_trash/Paper/test_paper 879.jpg"),
    (7, "data/raw/vn_trash/Paper/test_paper 891.jpg", "data/raw/garbage_v2/paper/paper_720.jpg"),
    (8, "data/raw/garbage_v2/plastic/plastic_601.jpg", "data/raw/vn_trash/PET/test_plastic_bottle 544.jpg"),
    (9, "data/raw/vn_trash/Carton/train_cardboard245.jpg", "data/raw/garbage_v2/cardboard/cardboard_1036.jpg"),
    (10, "data/raw/vn_trash/Carton/train_cardboard438.jpg", "data/raw/garbage_v2/paper/paper_582.jpg"),
    (11, "data/raw/garbage_v2/glass/glass_499.jpg", "data/raw/garbage_v2/cardboard/cardboard_1333.jpg"),
    (12, "data/raw/garbage_v2/metal/metal_435.jpg", "data/raw/garbage_v2/metal/metal_158.jpg"),
    (13, "data/raw/vn_trash/Alu/train_AluCan755.jpg", "data/raw/vn_trash/Alu/train_AluCan756.jpg"),
    (14, "data/raw/vn_trash/Alu/train_beverage_cans174.jpg", "data/raw/vn_trash/Paper_cup/test_paper_cups 599.jpg"),
    (15, "data/raw/vn_trash/Alu/train_beverage_cans89.jpg", "data/raw/vn_trash/Alu/train_beverage_cans904.jpg"),
    (16, "data/raw/vn_trash/Paper_cup/test_00000010.jpg", "data/raw/garbage_v2/paper/paper_185.jpg"),
    (17, "data/raw/vn_trash/Paper/test_paper 978.jpg", "data/raw/garbage_v2/paper/paper_1326.jpg"),
    (18, "data/raw/vn_trash/Foam_box/train_00000024.jpg", "data/raw/vn_trash/Foam_box/train_00000012.jpg"),
]

def describe_image(img: Image.Image) -> str:
    # Downsample to 16x16 to get coarse visual pattern
    thumb = img.convert("L").resize((8, 8))
    arr = np.array(thumb)
    bright_ratio = float((arr > 200).mean())
    dark_ratio = float((arr < 50).mean())
    std_val = float(arr.std())
    return f"size={img.size}, bright%={bright_ratio*100:.0f}%, dark%={dark_ratio*100:.0f}%, contrast={std_val:.1f}"

for idx, p1_rel, p2_rel in pairs_to_check:
    im1 = Image.open(BASE_DIR / p1_rel)
    im2 = Image.open(BASE_DIR / p2_rel)
    print(f"\n[Pair {idx:02d}]")
    print(f"  Img1 ({p1_rel}): {describe_image(im1)}")
    print(f"  Img2 ({p2_rel}): {describe_image(im2)}")
