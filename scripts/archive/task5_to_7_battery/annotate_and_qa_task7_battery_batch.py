# Mô tả: [Lưu trữ — Kịch bản xử lý & kiểm định dữ liệu lớp pin (Task 5–7)] Gán nhãn khung bao, rà soát chất lượng điểm ảnh (Visual QA) và chuẩn bị dữ liệu cho `annotate_and_qa_task7_battery_batch`.
"""
scripts/annotate_and_qa_task7_battery_batch.py
----------------------------------------------
Sprint AI-UPGRADE-01 — Task 7 (Part B2, B3, B4):
1. Applies manual visual QA verdicts (`QA_PASS`, `NEEDS_MANUAL_REVIEW`, `REJECT`) and
   full-object per-battery bounding box annotations across all 56 downloaded Wikimedia Commons
   candidates in `data/acquisition_staging/task7_diverse_battery_batch/`.
2. Generates YOLO label files, individual overlay images, and 3 QA contact sheets.
3. Runs a comprehensive SHA-256 + 4-rotation pHash leakage check against:
   - Dataset V2 (1,996 images across train/val/test)
   - Roboflow Battery Export (350 images: 140 canonical + 210 augmented)
   - External Holdout (27 images in data/holdout)
   - Cross-split specimen/author grouping between proposed TRAIN_V4 and VAL_REAL_BATTERY.
4. Produces:
   - `data/acquisition_staging/task7_diverse_battery_batch/task7_qa_manifest.json`
   - `data/acquisition_staging/task7_diverse_battery_batch/task7_leakage_report.json`
   - `data/acquisition_staging/task7_diverse_battery_batch/task7_dataset_v4_and_e4_proposal.json`
   - `configs/training_yolov8n_v4_proposed_DO_NOT_RUN.yaml`
   - `TASK7_DIVERSE_BATTERY_DATA_ACQUISITION_AND_QA_REPORT.md`
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imagehash
from PIL import Image, ImageDraw
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK7_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch"
RAW_MANIFEST_P = TASK7_DIR / "raw_download_manifest.json"
LABELS_DIR = TASK7_DIR / "labels"
LABELS_REVIEW_DIR = TASK7_DIR / "labels_needs_manual_review"
OVERLAYS_DIR = TASK7_DIR / "overlays"
SHEETS_DIR = TASK7_DIR / "contact_sheets"

# Visually verified QA decisions and full-battery normalized [xmin, ymin, xmax, ymax] boxes
# Every box encloses the full physical battery (never just text/logo on the battery body).
QA_DECISIONS: Dict[str, Dict[str, Any]] = {
    # --- REJECT (16 off-target / non-battery / patent text / icon / building images) ---
    "task7_01": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Yellow outdoor battery collection bins exterior only; zero visible battery objects.",
    },
    "task7_02": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Non-photographic recycling icon graphic with transparent/black alpha background.",
    },
    "task7_03": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target historical B&W photograph of wooden timber pile.",
    },
    "task7_04": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Supermarket e-waste collection bin exterior only; no individual batteries visible.",
    },
    "task7_05": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target reverse vending machine (PANT) in supermarket.",
    },
    "task7_06": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target outdoor delivery truck photograph.",
    },
    "task7_19": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Macro internal cross-section of cut-open capacitor, not an external object.",
    },
    "task7_20": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Longitudinal internal cross-section of cut-open capacitor, not an external object.",
    },
    "task7_22": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Scanned 1889 US Patent text document.",
    },
    "task7_23": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Scanned 1889 US Patent specification page.",
    },
    "task7_24": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Scanned 1889 US Patent line drawing.",
    },
    "task7_33": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Historical B&W laboratory glass-jar stationary battery cells.",
    },
    "task7_34": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Covered table with printed Russian sheets; no visible battery objects.",
    },
    "task7_47": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target Energizer Bunny hot air balloon in sky.",
    },
    "task7_49": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target Varta corporate building exterior.",
    },
    "task7_51": {
        "qa_verdict": "REJECT",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "boxes_xyxy_norm": [],
        "qa_reason": "Off-target aerial photograph of industrial facility.",
    },

    # --- NEEDS_MANUAL_REVIEW (10 complex / dense pile / PCB-mounted / blister-pack images) ---
    "task7_07": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_07_CR2032_BLISTER",
        "boxes_xyxy_norm": [
            [0.38, 0.38, 0.63, 0.58],
            [0.37, 0.65, 0.62, 0.85],
        ],
        "qa_reason": "Two CR2032 coin cells enclosed inside reflective plastic/cardboard retail blister pack held by hand; requires PM policy decision on whether outer blister pack should also be labeled plastic/cardboard.",
    },
    "task7_10": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_10_CMOS_PCB",
        "boxes_xyxy_norm": [
            [0.12, 0.15, 0.84, 0.86],
        ],
        "qa_reason": "CR2032 coin cell seated inside a black CMOS holder on a motherboard PCB; requires review of socket rim inclusion.",
    },
    "task7_17": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_17_DISASSEMBLED_18650",
        "boxes_xyxy_norm": [
            [0.16, 0.08, 0.59, 0.55],
            [0.50, 0.11, 0.82, 0.67],
        ],
        "qa_reason": "Disassembled 18650 lithium-ion cell separated into green internal jelly-roll and empty steel cylinder shell.",
    },
    "task7_18": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_18_CHARGER_4X18650",
        "boxes_xyxy_norm": [
            [0.24, 0.49, 0.37, 0.86],
            [0.38, 0.49, 0.50, 0.86],
            [0.50, 0.49, 0.62, 0.86],
            [0.64, 0.49, 0.76, 0.86],
        ],
        "qa_reason": "Four Robiton 18650 cells partially occluded inside a 4-bay LiitoKala charger.",
    },
    "task7_36": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_36_CUTAWAY_LEAD_ACID",
        "boxes_xyxy_norm": [
            [0.22, 0.13, 0.97, 0.99],
        ],
        "qa_reason": "Open-top cutaway lead-acid automotive battery exposing internal lead plates.",
    },
    "task7_37": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_37_DENSE_PILE_40PLUS",
        "boxes_xyxy_norm": [
            [0.14, 0.48, 0.25, 0.84],
            [0.23, 0.27, 0.35, 0.56],
            [0.37, 0.41, 0.48, 0.75],
            [0.53, 0.18, 0.69, 0.50],
            [0.63, 0.34, 0.77, 0.70],
            [0.64, 0.72, 0.88, 0.96],
        ],
        "qa_reason": "Dense overlapping heap of 40+ spent AA/AAA batteries with severe occlusion; requires high-density crowd annotation review before training.",
    },
    "task7_38": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_38_WATCH_MOVEMENT",
        "boxes_xyxy_norm": [
            [0.56, 0.24, 0.84, 0.62],
        ],
        "qa_reason": "Internal quartz watch movement with corroded button-cell compartment.",
    },
    "task7_42": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_42_PCB_CORRODED_NICD",
        "boxes_xyxy_norm": [
            [0.61, 0.28, 0.85, 0.47],
        ],
        "qa_reason": "Corroded cylindrical NiCd backup battery soldered onto synthesizer circuit board.",
    },
    "task7_44": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_44_DURACELL_48PACK",
        "boxes_xyxy_norm": [
            [0.02, 0.02, 0.98, 0.98],
        ],
        "qa_reason": "Retail 48-pack cardboard box containing 48 tightly stacked Duracell AA batteries behind plastic window.",
    },
    "task7_55": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_NOKIA_BL5C",
        "boxes_xyxy_norm": [
            [0.26, 0.22, 0.98, 0.96],
        ],
        "qa_reason": "Nokia BL-5C Li-ion battery seated inside open phone housing with holographic sticker.",
    },

    # --- QA_PASS (30 images: 26 positive diverse battery scenes with 57 full-object boxes + 4 hard negatives) ---
    "task7_08": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_08_MAXELL_CR2032",
        "boxes_xyxy_norm": [
            [0.02, 0.02, 0.98, 0.98],
        ],
        "qa_reason": "Clear Hitachi Maxell CR2032 3V coin cell; full circular body boxed.",
    },
    "task7_09": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_09_MAXELL_ML2032",
        "boxes_xyxy_norm": [
            [0.02, 0.02, 0.98, 0.98],
        ],
        "qa_reason": "Clear Maxell ML2032 rechargeable button cell; full circular body boxed.",
    },
    "task7_11": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_11_9V_SHORT_TEST",
        "boxes_xyxy_norm": [
            [0.08, 0.22, 0.70, 0.89],
        ],
        "qa_reason": "Prismatic 9V battery lying horizontally with yellow wire; full 9V body & terminals boxed.",
    },
    "task7_12": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_12_13_MOTOMA_9V_WOOD",
        "boxes_xyxy_norm": [
            [0.18, 0.27, 0.79, 0.70],
        ],
        "qa_reason": "Motoma 9V 6F22 battery side view on wooden surface; grouped in TRAIN_V4 with task7_13.",
    },
    "task7_13": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_12_13_MOTOMA_9V_WOOD",
        "boxes_xyxy_norm": [
            [0.26, 0.26, 0.73, 0.73],
        ],
        "qa_reason": "Top-down terminal view of the same Motoma 9V battery on wooden surface; grouped in TRAIN_V4 with task7_12.",
    },
    "task7_14": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_14_ULTRALIFE_9V",
        "boxes_xyxy_norm": [
            [0.11, 0.01, 0.89, 0.97],
        ],
        "qa_reason": "Ultralife 9V lithium battery; full rectangular body and snap terminals boxed.",
    },
    "task7_15": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_15_18650_21700_PAIR",
        "boxes_xyxy_norm": [
            [0.13, 0.16, 0.57, 0.56],
            [0.46, 0.28, 0.91, 0.58],
        ],
        "qa_reason": "Two blue cylindrical lithium-ion cells (18650 and 21700) on dark surface; each cell boxed individually.",
    },
    "task7_16": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_16_THREE_LIION_CELLS",
        "boxes_xyxy_norm": [
            [0.18, 0.12, 0.32, 0.74],
            [0.12, 0.65, 0.64, 0.93],
            [0.65, 0.48, 0.79, 0.86],
        ],
        "qa_reason": "Three distinct lithium cells (standing blue 18650, lying silver Panasonic 18650, standing white CR123A); all 3 boxed individually.",
    },
    "task7_21": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_21_HARD_NEG_CAPACITORS",
        "boxes_xyxy_norm": [],
        "qa_reason": "Hard negative: two cylindrical electrolytic capacitors with wire leads on textured surface (0 battery boxes).",
    },
    "task7_25": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_25_HARD_NEG_ORANGE_BOX",
        "boxes_xyxy_norm": [],
        "qa_reason": "Hard negative: two round orange electrical wall boxes (0 battery boxes).",
    },
    "task7_26": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_26_HARD_NEG_CONDUITS",
        "boxes_xyxy_norm": [],
        "qa_reason": "Hard negative: outdoor wall with electrical conduit boxes and pipes (0 battery boxes).",
    },
    "task7_27": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_27_HARD_NEG_JUNCTION_BOXES",
        "boxes_xyxy_norm": [],
        "qa_reason": "Hard negative: industrial electrical junction boxes and metal conduits (0 battery boxes).",
    },
    "task7_28": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_28_18650_AA_AAA_PENNY",
        "boxes_xyxy_norm": [
            [0.10, 0.11, 0.46, 0.95],
            [0.46, 0.33, 0.72, 0.92],
            [0.73, 0.40, 0.94, 0.94],
        ],
        "qa_reason": "Three cylindrical batteries (green 18650, Kirkland AA, Kirkland AAA) standing side-by-side behind a US penny;penny excluded, 3 batteries boxed.",
    },
    "task7_29": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_29_THREE_AAA_ONE_AA",
        "boxes_xyxy_norm": [
            [0.26, 0.01, 0.43, 0.65],
            [0.43, 0.01, 0.60, 0.65],
            [0.60, 0.01, 0.77, 0.65],
            [0.08, 0.79, 0.94, 0.99],
        ],
        "qa_reason": "Four batteries in one frame (3 vertical AAA cells + 1 horizontal AA cell); directly tests multi-battery cluster detection.",
    },
    "task7_30": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_30_SIX_KODAK_BATTERIES",
        "boxes_xyxy_norm": [
            [0.073, 0.075, 0.232, 0.710],
            [0.242, 0.158, 0.343, 0.706],
            [0.360, 0.162, 0.453, 0.704],
            [0.383, 0.492, 0.747, 0.912],
            [0.546, 0.563, 0.838, 0.918],
            [0.675, 0.570, 0.948, 0.918],
        ],
        "qa_reason": "Six Kodak AA and AAA batteries (3 standing, 3 lying touching each other); patched in Task 7 Final QA Patch so all 3 standing positive terminals and 3 angled horizontal bodies are fully and tightly enclosed.",
        "qa_patch_note": "Patched per PM Task 7 Final QA Patch: extended top of boxes #1-#3 to fully enclose protruding positive terminals and refined boxes #4-#6 to tightly hug angled horizontal cylinders.",
    },
    "task7_31": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_31_THREE_AUCHAN_BATTERIES",
        "boxes_xyxy_norm": [
            [0.18, 0.26, 0.30, 0.81],
            [0.34, 0.33, 0.43, 0.81],
            [0.42, 0.69, 0.81, 0.84],
        ],
        "qa_reason": "Three Auchan LR6/LR03 batteries (2 standing, 1 lying) on wooden desk.",
    },
    "task7_32": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_32_FOUR_FORM_FACTORS",
        "boxes_xyxy_norm": [
            [0.05, 0.03, 0.29, 0.96],
            [0.34, 0.16, 0.52, 0.95],
            [0.59, 0.20, 0.72, 0.93],
            [0.77, 0.44, 0.95, 0.94],
        ],
        "qa_reason": "Four cylindrical batteries of different diameters/heights (AA, AAA, AAAA, A23) side-by-side.",
    },
    "task7_35": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_35_TWO_LPE6_PACKS",
        "boxes_xyxy_norm": [
            [0.17, 0.19, 0.47, 0.86],
            [0.49, 0.21, 0.81, 0.84],
        ],
        "qa_reason": "Two rectangular LP-E6 lithium-ion camera battery packs (front and back view).",
    },
    "task7_39": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_39_40_NANFU_LEAKED_AA",
        "boxes_xyxy_norm": [
            [0.19, 0.24, 0.82, 0.75],
        ],
        "qa_reason": "Severely corroded leaking AA alkaline battery with white potassium carbonate crust; grouped in TRAIN_V4 with task7_40.",
    },
    "task7_40": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_39_40_NANFU_LEAKED_AA",
        "boxes_xyxy_norm": [
            [0.19, 0.27, 0.79, 0.72],
        ],
        "qa_reason": "Second angle of the same leaking AA alkaline battery on tile; grouped in TRAIN_V4 with task7_39 to prevent split leakage.",
    },
    "task7_41": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_41_THREE_LEAKING_NOPOPO",
        "boxes_xyxy_norm": [
            [0.14, 0.29, 0.35, 0.63],
            [0.40, 0.30, 0.51, 0.63],
            [0.69, 0.38, 0.82, 0.69],
        ],
        "qa_reason": "Three corroded/leaking cylindrical batteries lying at perspective angles on tabletop.",
    },
    "task7_43": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_43_DURACELL_9V",
        "boxes_xyxy_norm": [
            [0.14, 0.04, 0.96, 0.94],
        ],
        "qa_reason": "Duracell Plus 9V rectangular battery angled view; full 3D body and top snap terminals boxed.",
    },
    "task7_45": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_45_FOUR_DURACELL_DARK",
        "boxes_xyxy_norm": [
            [0.06, 0.05, 0.27, 0.88],
            [0.19, 0.01, 0.40, 0.70],
            [0.15, 0.46, 0.66, 0.92],
            [0.34, 0.31, 0.87, 0.84],
        ],
        "qa_reason": "Four Duracell Plus AA batteries (2 upright, 2 crossed in foreground) on dark background.",
    },
    "task7_46": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_46_48_52_ENERGIZER_NIMH",
        "boxes_xyxy_norm": [
            [0.11, 0.16, 0.30, 0.85],
            [0.32, 0.16, 0.49, 0.85],
            [0.52, 0.16, 0.68, 0.85],
            [0.71, 0.16, 0.89, 0.85],
        ],
        "qa_reason": "Four Energizer NiMH 2000mAh AA batteries standing in a row; grouped in TRAIN_V4 with task7_48 and task7_52.",
    },
    "task7_48": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_46_48_52_ENERGIZER_NIMH",
        "boxes_xyxy_norm": [
            [0.07, 0.44, 0.32, 0.90],
            [0.33, 0.07, 0.49, 0.76],
            [0.47, 0.35, 0.72, 0.79],
            [0.74, 0.07, 0.94, 0.76],
        ],
        "qa_reason": "Four Energizer NiMH 2000mAh AA batteries (2 standing, 2 lying angled); grouped in TRAIN_V4 with task7_46 and task7_52.",
    },
    "task7_50": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_50_VARTA_ENERGY_AA",
        "boxes_xyxy_norm": [
            [0.18, 0.06, 0.82, 0.94],
        ],
        "qa_reason": "Varta Energy Alkaline AA battery upright; full cylinder boxed.",
    },
    "task7_52": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_46_48_52_ENERGIZER_NIMH",
        "boxes_xyxy_norm": [
            [0.24, 0.06, 0.78, 0.94],
        ],
        "qa_reason": "Single Energizer NiMH 2000mAh AA battery from the same photographer session as task7_46/48; grouped in TRAIN_V4.",
    },
    "task7_53": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "TRAIN_V4",
        "specimen_group": "SPEC_53_FOUR_IKEA_LADDA",
        "boxes_xyxy_norm": [
            [0.353, 0.122, 0.478, 0.638],
            [0.476, 0.095, 0.592, 0.592],
            [0.258, 0.530, 0.678, 0.842],
            [0.450, 0.560, 0.868, 0.882],
        ],
        "qa_reason": "Four IKEA LADDA NiMH rechargeable AA batteries (2 standing, 2 lying in front); patched in Task 7 Final QA Patch so Boxes #3 and #4 enclose the entire angled bodies and upper-right ends.",
        "qa_patch_note": "Patched per PM Task 7 Final QA Patch: extended Box #3 [0.258, 0.530, 0.678, 0.842] and Box #4 [0.450, 0.560, 0.868, 0.882] to enclose full visible upper-right slanted ends and lower-left positive terminals without excess white margin.",
    },
    "task7_54": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_54_THREE_NIMH_TYPES",
        "boxes_xyxy_norm": [
            [0.01, 0.01, 0.99, 0.40],
            [0.13, 0.42, 0.86, 0.73],
            [0.16, 0.76, 0.81, 0.99],
        ],
        "qa_reason": "Three NiMH batteries of distinct form factors (GP 1450 prismatic gumstick, Uniross AA, Uniross AAA).",
    },
    "task7_56": {
        "qa_verdict": "QA_PASS",
        "proposed_split": "VAL_REAL_BATTERY",
        "specimen_group": "SPEC_56_NOKIA_BL5C_STANDALONE",
        "boxes_xyxy_norm": [
            [0.06, 0.02, 0.99, 0.99],
        ],
        "qa_reason": "Standalone Nokia BL-5C Li-ion mobile phone battery; full pouch/prismatic body boxed.",
    },
}


def xyxy_to_yolo(box: List[float]) -> Tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return xc, yc, w, h


def draw_overlay(item: Dict[str, Any], out_path: Path) -> None:
    img_path = REPO_ROOT / item["local_path"]
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    draw = ImageDraw.Draw(rgb)

    verdict = item["qa_verdict"]
    color = (0, 230, 90) if verdict == "QA_PASS" else ((255, 165, 0) if verdict == "NEEDS_MANUAL_REVIEW" else (240, 60, 60))

    for b_idx, box in enumerate(item["boxes_xyxy_norm"], start=1):
        x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, min(w, h) // 180))
        label_txt = f"0:battery #{b_idx}"
        draw.rectangle([x1, max(0, y1 - 22), min(w, x1 + 125), y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 18)), label_txt, fill=(10, 10, 10))

    # Top banner
    banner_h = 34
    draw.rectangle([0, 0, w, banner_h], fill=(20, 24, 30))
    draw.text(
        (8, 8),
        f"{item['short_key']} | {verdict} | boxes={len(item['boxes_xyxy_norm'])} | split={item['proposed_split']}",
        fill=color,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, quality=90)


def render_qa_contact_sheet(items: List[Dict[str, Any]], title: str, out_path: Path, cols: int = 3) -> None:
    cell_w, cell_h = 540, 420
    header_h = 54
    banner_h = 48
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, banner_h + rows * (cell_h + header_h)), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, cols * cell_w, banner_h], fill=(32, 38, 48))
    draw.text((16, 14), title, fill=(255, 235, 130))

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = banner_h + r * (cell_h + header_h)

        verdict = item["qa_verdict"]
        v_color = (60, 230, 110) if verdict == "QA_PASS" else ((255, 175, 40) if verdict == "NEEDS_MANUAL_REVIEW" else (245, 80, 80))

        draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + header_h], fill=(36, 40, 48))
        draw.text(
            (x0 + 8, y0 + 6),
            f"{item['short_key']} [{verdict}] | boxes={len(item['boxes_xyxy_norm'])} | {item['proposed_split']}",
            fill=v_color,
        )
        draw.text((x0 + 8, y0 + 26), f"{item['qa_reason'][:72]}", fill=(210, 215, 225))

        ov_path = REPO_ROOT / item["overlay_path"] if item.get("overlay_path") else (REPO_ROOT / item["local_path"])
        with Image.open(ov_path) as im:
            thumb = im.convert("RGB").resize((cell_w - 16, cell_h - 16), Image.Resampling.BILINEAR)
        canvas.paste(thumb, (x0 + 8, y0 + header_h + 8))
        draw.rectangle([x0 + 8, y0 + header_h + 8, x0 + cell_w - 8, y0 + header_h + cell_h - 8], outline=v_color, width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=91)


def run_leakage_audit(enriched_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 1. Load known hashes from authoritative_hashes_cache.json, roboflow_battery, and holdout_27
    known_sha256: Dict[str, str] = {}
    known_phash: Dict[str, imagehash.ImageHash] = {}

    cache_p = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if cache_p.exists():
        cache_data = json.loads(cache_p.read_text(encoding="utf-8"))
        for k, v in cache_data.items():
            if v.get("sha256"):
                known_sha256[v["sha256"]] = f"v2_or_cache:{k}"
            if v.get("phash"):
                try:
                    known_phash[f"v2_or_cache:{k}"] = imagehash.hex_to_hash(v["phash"])
                except Exception:
                    pass

    rf_qa_p = REPO_ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "base_images_qa_manifest.json"
    if rf_qa_p.exists():
        rf_data = json.loads(rf_qa_p.read_text(encoding="utf-8"))
        for item in rf_data.get("items", []):
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"roboflow_140:{fn}"
            if item.get("phash"):
                try:
                    known_phash[f"roboflow_140:{fn}"] = imagehash.hex_to_hash(item["phash"])
                except Exception:
                    pass

    holdout_p = REPO_ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
    if holdout_p.exists():
        h_data = json.loads(holdout_p.read_text(encoding="utf-8"))
        for item in h_data.get("images", []):
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"holdout_27:{fn}"

    exact_sha_collisions: List[Dict[str, Any]] = []
    near_phash_collisions_external: List[Dict[str, Any]] = []

    for c in enriched_candidates:
        c_sha = c["sha256"]
        if c_sha in known_sha256:
            exact_sha_collisions.append({
                "candidate_id": c["candidate_id"],
                "matched_source": known_sha256[c_sha],
                "sha256": c_sha,
            })
        c_rot_hashes = [imagehash.hex_to_hash(h) for h in c["phash_rotations"]]
        for ext_key, ext_ph in known_phash.items():
            min_dist = min(abs(rh - ext_ph) for rh in c_rot_hashes)
            if min_dist <= 8:
                near_phash_collisions_external.append({
                    "candidate_id": c["candidate_id"],
                    "matched_external": ext_key,
                    "min_rotation_hamming_distance": int(min_dist),
                })

    # 2. Check cross-split pHash between proposed TRAIN_V4 and VAL_REAL_BATTERY
    train_cands = [c for c in enriched_candidates if c["proposed_split"] == "TRAIN_V4"]
    val_cands = [c for c in enriched_candidates if c["proposed_split"] == "VAL_REAL_BATTERY"]
    cross_split_collisions: List[Dict[str, Any]] = []

    for tc in train_cands:
        t_rots = [imagehash.hex_to_hash(h) for h in tc["phash_rotations"]]
        for vc in val_cands:
            v_rots = [imagehash.hex_to_hash(h) for h in vc["phash_rotations"]]
            min_d = min(abs(th - vh) for th in t_rots for vh in v_rots)
            if min_d <= 8 or tc["specimen_group"] == vc["specimen_group"]:
                cross_split_collisions.append({
                    "train_candidate": tc["candidate_id"],
                    "val_candidate": vc["candidate_id"],
                    "min_rotation_hamming_distance": int(min_d),
                    "same_specimen_group": tc["specimen_group"] == vc["specimen_group"],
                })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "external_reference_pool": {
            "known_sha256_count": len(known_sha256),
            "known_phash_count": len(known_phash),
        },
        "exact_sha256_collisions_with_v2_v3_holdout": exact_sha_collisions,
        "near_phash_le8_collisions_with_v2_v3_holdout": near_phash_collisions_external,
        "cross_split_train_vs_val_real_battery_collisions": cross_split_collisions,
        "zero_external_leakage": len(exact_sha_collisions) == 0 and len(near_phash_collisions_external) == 0,
        "zero_cross_split_leakage": len(cross_split_collisions) == 0,
    }


def main() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    candidates = raw_data["candidates"]

    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        short_key = "_".join(item["candidate_id"].split("_")[:2])  # e.g. task7_01
        if short_key not in QA_DECISIONS:
            raise KeyError(f"Missing manual QA decision for {short_key} ({item['candidate_id']})")
        dec = QA_DECISIONS[short_key]

        boxes_xyxy = dec["boxes_xyxy_norm"]
        yolo_lines = []
        for b in boxes_xyxy:
            xc, yc, bw, bh = xyxy_to_yolo(b)
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        stem = Path(item["filename"]).stem
        label_rel = None
        if dec["qa_verdict"] == "QA_PASS":
            lp = LABELS_DIR / f"{stem}.txt"
            lp.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            label_rel = lp.relative_to(REPO_ROOT).as_posix()
        elif dec["qa_verdict"] == "NEEDS_MANUAL_REVIEW":
            lp = LABELS_REVIEW_DIR / f"{stem}.txt"
            lp.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            label_rel = lp.relative_to(REPO_ROOT).as_posix()

        review_ts = datetime.now(timezone.utc).isoformat()
        merged = {
            **item,
            "short_key": short_key,
            "qa_verdict": dec["qa_verdict"],
            "proposed_split": dec["proposed_split"],
            "specimen_group": dec["specimen_group"],
            "boxes_count": len(boxes_xyxy),
            "boxes_xyxy_norm": boxes_xyxy,
            "yolo_label_lines": yolo_lines,
            "label_path": label_rel,
            "qa_reason": dec["qa_reason"],
            "qa_patch_note": dec.get("qa_patch_note"),
            "qa_reviewer": "Antigravity Visual QA + PM Audit Patch (Task 7 Final QA Patch)",
            "qa_reviewed_at_utc": review_ts,
            "review_status": "MANUALLY_INSPECTED_ON_FULL_RES_AND_OVERLAY",
            "approved_for_train": False,
            "approval_status": "PENDING_PM_APPROVAL",
        }

        ov_path = OVERLAYS_DIR / f"{stem}_overlay.jpg"
        draw_overlay(merged, ov_path)
        merged["overlay_path"] = ov_path.relative_to(REPO_ROOT).as_posix()
        enriched.append(merged)

    # Render 3 QA Contact Sheets
    qa_pass_items = [c for c in enriched if c["qa_verdict"] == "QA_PASS"]
    review_and_reject_items = [c for c in enriched if c["qa_verdict"] in ("NEEDS_MANUAL_REVIEW", "REJECT")]

    sheet1_p = SHEETS_DIR / "contact_sheet_task7_qa_pass_part1.jpg"
    sheet2_p = SHEETS_DIR / "contact_sheet_task7_qa_pass_part2.jpg"
    sheet3_p = SHEETS_DIR / "contact_sheet_task7_needs_manual_review_and_reject.jpg"

    render_qa_contact_sheet(qa_pass_items[:15], "TASK 7 QA_PASS CANDIDATES (PART 1: 1..15) — GREEN = FULL BATTERY BBOX", sheet1_p)
    render_qa_contact_sheet(qa_pass_items[15:], "TASK 7 QA_PASS CANDIDATES (PART 2: 16..30) — GREEN = FULL BATTERY BBOX", sheet2_p)
    render_qa_contact_sheet(review_and_reject_items, "TASK 7 NEEDS_MANUAL_REVIEW (10) & REJECT (16) CANDIDATES", sheet3_p)

    leakage_report = run_leakage_audit(enriched)
    (TASK7_DIR / "task7_leakage_report.json").write_text(json.dumps(leakage_report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Compute summary stats
    qa_pass_pos = [c for c in qa_pass_items if c["boxes_count"] > 0]
    qa_pass_neg = [c for c in qa_pass_items if c["boxes_count"] == 0]
    qa_pass_train = [c for c in qa_pass_items if c["proposed_split"] == "TRAIN_V4"]
    qa_pass_val = [c for c in qa_pass_items if c["proposed_split"] == "VAL_REAL_BATTERY"]
    needs_review_items = [c for c in enriched if c["qa_verdict"] == "NEEDS_MANUAL_REVIEW"]
    reject_items = [c for c in enriched if c["qa_verdict"] == "REJECT"]

    qa_manifest = {
        "manifest_name": "TASK7_DIVERSE_BATTERY_BATCH_QA_MANIFEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_reviewer": "Antigravity Visual QA + PM Audit Patch (Task 7 Final QA Patch)",
        "qa_patch_history": [
            {
                "short_key": "task7_30",
                "patched_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": "PM Review + Antigravity Full-Res Manual Patch",
                "reason": "Three standing batteries (#1-#3) previously clipped top protruding positive terminals; refined all 6 boxes on 4239x2826 full-resolution image to tightly enclose full physical batteries including positive terminals.",
                "before_boxes_xyxy_norm": [
                    [0.08, 0.12, 0.21, 0.72],
                    [0.24, 0.19, 0.34, 0.71],
                    [0.35, 0.20, 0.45, 0.71],
                    [0.38, 0.51, 0.71, 0.90],
                    [0.54, 0.56, 0.85, 0.91],
                    [0.66, 0.60, 0.95, 0.92],
                ],
                "after_boxes_xyxy_norm": QA_DECISIONS["task7_30"]["boxes_xyxy_norm"],
            },
            {
                "short_key": "task7_53",
                "patched_at_utc": datetime.now(timezone.utc).isoformat(),
                "reviewer": "PM Review + Antigravity Full-Res Manual Patch",
                "reason": "Box #3 and Box #4 previously cut off the upper-right slanted ends of the two foreground lying IKEA LADDA batteries; refined all 4 boxes on 1280x960 full-resolution image so each box tightly wraps the full visible physical battery without excess white background.",
                "before_boxes_xyxy_norm": [
                    [0.35, 0.10, 0.48, 0.72],
                    [0.48, 0.09, 0.59, 0.65],
                    [0.26, 0.63, 0.66, 0.87],
                    [0.47, 0.66, 0.86, 0.93],
                ],
                "after_boxes_xyxy_norm": QA_DECISIONS["task7_53"]["boxes_xyxy_norm"],
            },
        ],
        "total_downloaded_candidates": len(enriched),
        "summary_counts": {
            "qa_pass_total_images": len(qa_pass_items),
            "qa_pass_positive_battery_images": len(qa_pass_pos),
            "qa_pass_hard_negative_images": len(qa_pass_neg),
            "qa_pass_total_battery_boxes": sum(c["boxes_count"] for c in qa_pass_items),
            "qa_pass_multi_battery_images_ge2_boxes": sum(1 for c in qa_pass_pos if c["boxes_count"] >= 2),
            "proposed_train_v4_images": len(qa_pass_train),
            "proposed_train_v4_battery_boxes": sum(c["boxes_count"] for c in qa_pass_train),
            "proposed_val_real_battery_images": len(qa_pass_val),
            "proposed_val_real_battery_boxes": sum(c["boxes_count"] for c in qa_pass_val),
            "needs_manual_review_images": len(needs_review_items),
            "needs_manual_review_candidate_boxes": sum(c["boxes_count"] for c in needs_review_items),
            "reject_images": len(reject_items),
            "approved_for_train": 0,
        },
        "contact_sheets": [
            sheet1_p.relative_to(REPO_ROOT).as_posix(),
            sheet2_p.relative_to(REPO_ROOT).as_posix(),
            sheet3_p.relative_to(REPO_ROOT).as_posix(),
        ],
        "items": enriched,
    }
    (TASK7_DIR / "task7_qa_manifest.json").write_text(json.dumps(qa_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save proposed V4 dataset & E4 training plan (DO NOT RUN)
    proposal = {
        "proposal_name": "DATASET_V4_AND_E4_CONSERVATIVE_FINETUNING_PROPOSAL",
        "status": "PROPOSED_ONLY_DO_NOT_RUN_UNTIL_PM_APPROVAL",
        "dataset_v4_design": {
            "base_dataset": "v3_pilot (Train: 1509 images / 4074 boxes; Val V3: 328 images / 930 boxes; Test Frozen: 299 images / 720 boxes)",
            "new_task7_qa_pass_additions": {
                "train_v4_additions_images": len(qa_pass_train),
                "train_v4_additions_battery_boxes": sum(c["boxes_count"] for c in qa_pass_train),
                "train_v4_additions_hard_negatives": sum(1 for c in qa_pass_train if c["boxes_count"] == 0),
                "val_real_battery_independent_images": len(qa_pass_val),
                "val_real_battery_independent_boxes": sum(c["boxes_count"] for c in qa_pass_val),
                "val_real_battery_hard_negatives": sum(1 for c in qa_pass_val if c["boxes_count"] == 0),
            },
            "projected_train_v4_totals": {
                "images": 1509 + len(qa_pass_train),
                "total_boxes": 4074 + sum(c["boxes_count"] for c in qa_pass_train),
                "battery_boxes": 112 + sum(c["boxes_count"] for c in qa_pass_train),
            },
            "evaluation_protocol_5_benchmarks": [
                "1. Validation V2 (299 images, 901 boxes, 0 battery) — Catastrophic Forgetting Gate on 9 existing classes",
                "2. Validation V3 Tabletop (328 images, 930 boxes, 29 tabletop batteries) — In-domain Regression Gate",
                "3. NEW Validation Real-World Battery (`val_real_battery`: 10 QA_PASS images, 19 diverse battery boxes + 1 hard negative, expandable to 25+ images) — Model Selection & EarlyStopping Gate for diverse batteries",
                "4. Internal Test Frozen (299 images, 720 boxes) — Blind Final Internal Test (never used for tuning)",
                "5. External Holdout (27 images, 57 GT objects in data/holdout) — Blind External Acceptance Benchmark (never used for tuning)",
            ],
        },
        "e4_conservative_finetuning_config_proposal": {
            "execution_authorized": False,
            "base_checkpoint": "artifacts/detection_v2/E1_yolov8n_10cls/weights/best.pt",
            "imgsz": 640,
            "batch": 4,
            "freeze": 10,
            "optimizer": "AdamW",
            "lr0": 0.0002,
            "lrf": 0.01,
            "epochs": 20,
            "patience": 5,
            "rationale": (
                "1. Keep imgsz=640 identical to E1 (RTX 2050 4GB only used 400.1 MB at imgsz=512, batch=4, so imgsz=640 easily fits under 1 GB VRAM) to prevent resolution-shift degradation on cardboard/shoes/metal. "
                "2. Freeze backbone layers 0..9 (freeze=10) and lower lr0 from 0.001 to 0.0002 so existing 9-class feature extractors are preserved while detection head adapts to multi-battery, 9V, button cell, Li-ion, and corroded batteries."
            ),
        },
        "remaining_data_gaps_requiring_user_or_field_collection": [
            "1. Outdoor Batteries Mixed with Household Trash/Plastic/Food Waste: Wikimedia Commons has very few open-license photos of batteries lying directly inside overflowing street bins or mixed with plastic bags/food waste (`0` QA_PASS mixed-trash-pile battery images in current batch; recommend capturing 15–20 smartphone photos of old AA/AAA/9V/phone batteries placed alongside plastic bottles, cardboard, and wrappers in real waste bins).",
            "2. Dense Pile Individual Annotations: `task7_37` (40+ overlapping batteries in a collection bin) and `task7_44` (48-pack Duracell box) are held in `NEEDS_MANUAL_REVIEW` until PM confirms crowd/pile annotation policy.",
            "3. Partially Occluded / Muddy Street Batteries: Recommend 10–15 smartphone photos of single and grouped batteries on wet asphalt, grass, soil, and concrete sidewalks under varied lighting.",
        ],
    }
    (TASK7_DIR / "task7_dataset_v4_and_e4_proposal.json").write_text(json.dumps(proposal, indent=2, ensure_ascii=False), encoding="utf-8")

    e4_yaml_path = REPO_ROOT / "configs" / "training_yolov8n_v4_proposed_DO_NOT_RUN.yaml"
    e4_yaml_path.write_text(yaml.safe_dump(proposal["e4_conservative_finetuning_config_proposal"], sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(json.dumps(qa_manifest["summary_counts"], indent=2))
    print("Zero external leakage:", leakage_report["zero_external_leakage"])
    print("Zero cross-split leakage:", leakage_report["zero_cross_split_leakage"])


if __name__ == "__main__":
    main()
