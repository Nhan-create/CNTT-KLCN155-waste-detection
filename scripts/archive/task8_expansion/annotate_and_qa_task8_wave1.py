"""
scripts/annotate_and_qa_task8_wave1.py
--------------------------------------
Executes Task 3 (Dataset Standardization) and Task 4 (Quality & Balance Control)
on the Wave 1 multi-source acquisition batch (`data/acquisition_staging/task8a_multiclass_batch`).

Enforces:
- Strict 10-class taxonomy (`0..9`).
- Multi-class exhaustive labeling where multiple taxonomy classes appear in an image.
- Full physical object bounding boxes (never collapsing distinguishable items into a single `9: trash` box).
- Uncertain/incomplete/dense-pile annotations separated into `NEEDS_REVIEW` (`labels_needs_manual_review/`), never mixed into `QA_PASS` (`labels/`).
- Quarantined/rejected images excluded (`REJECT`), never used as negatives.
- Explicit hard negatives (`0` boxes) verified to contain zero waste objects.
- SHA-256 + 4-rotation pHash (hamming <= 8) + specimen/session grouping leakage enforcement across all existing splits (`real_v2`, `v3_candidate`, `roboflow_140`, `task7`, `batch_a_mini01`, `phase4e`, `holdout_27`).
- `approved_for_train = False` on all records pending PM/User approval gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imagehash
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
BATCH_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch"
IMAGES_DIR = BATCH_DIR / "images"
LABELS_DIR = BATCH_DIR / "labels"
LABELS_REVIEW_DIR = BATCH_DIR / "labels_needs_manual_review"
OVERLAYS_DIR = BATCH_DIR / "overlays"
SHEETS_DIR = BATCH_DIR / "contact_sheets"
RAW_MANIFEST_P = BATCH_DIR / "raw_download_manifest.json"
QA_MANIFEST_P = BATCH_DIR / "task8_wave1_qa_manifest.json"
QA_CSV_P = BATCH_DIR / "task8_wave1_qa_manifest.csv"

CLASS_NAMES = {
    0: "battery",
    1: "biological",
    2: "cardboard",
    3: "clothes",
    4: "glass",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "shoes",
    9: "trash",
}

CLASS_COLORS = {
    0: (255, 60, 60),     # battery - red
    1: (60, 220, 90),     # biological - green
    2: (230, 160, 40),    # cardboard - amber
    3: (180, 90, 240),    # clothes - purple
    4: (60, 210, 230),    # glass - cyan
    5: (190, 200, 210),   # metal - silver
    6: (255, 235, 70),    # paper - yellow
    7: (70, 130, 255),    # plastic - blue
    8: (255, 120, 180),   # shoes - pink
    9: (255, 140, 90),    # trash - orange
}

# Tuple schema for QA_DECISIONS:
# short_key -> (
#     verdict,              # "QA_PASS" | "NEEDS_REVIEW" | "REJECT"
#     proposed_split,       # "TRAIN_WAVE1" | "VAL_REAL_WAVE1" | "MANUAL_REVIEW_HOLD" | "EXCLUDED"
#     specimen_group,       # Specimen/session ID to prevent split leakage
#     is_outdoor,           # bool
#     is_complex_bg,        # bool
#     is_dirty_or_damaged,  # bool
#     is_hard_negative,     # bool
#     boxes_xyxy,           # List[Tuple[int, float, float, float, float]] -> (cls_id, xmin, ymin, xmax, ymax)
#     qa_note               # Detailed audit note
# )
QA_DECISIONS: Dict[str, Tuple[str, str, str, bool, bool, bool, bool, List[Tuple[int, float, float, float, float]], str]] = {
    # --- Sheet 01 (001..018) ---
    "task8a_001": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_001", True, True, True, False,
                   [(1, 0.08, 0.14, 0.90, 0.87)], "Rotten brown apple half on rocky ground"),
    "task8a_002": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_002", True, True, True, False,
                   [(1, 0.31, 0.44, 0.61, 0.69)], "Black mummified rotten apple on branch"),
    "task8a_003": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_003", True, True, True, False,
                   [(1, 0.25, 0.12, 0.78, 0.88)], "Decaying apple with white mold spores on soil"),
    "task8a_004": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_004", True, True, True, False,
                   [(1, 0.12, 0.10, 0.30, 0.25), (1, 0.56, 0.23, 0.75, 0.46), (1, 0.30, 0.62, 0.65, 0.92)],
                   "Three rotting cashew fruits on outdoor stone"),
    "task8a_005": ("REJECT", "EXCLUDED", "SPEC_W1_005", False, False, True, False, [], "exclude_microscope_petri_dish_view"),
    "task8a_006": ("REJECT", "EXCLUDED", "SPEC_W1_006", False, False, True, False, [], "exclude_microscope_petri_dish_view"),
    "task8a_007": ("REJECT", "EXCLUDED", "SPEC_W1_007", False, False, True, False, [], "exclude_microscope_slide_view"),
    "task8a_008": ("REJECT", "EXCLUDED", "SPEC_W1_008", False, False, True, False, [], "exclude_microscope_petri_dish_view"),
    "task8a_009": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_009", False, False, True, False,
                   [(1, 0.01, 0.39, 0.98, 0.99)], "Green Penicillium mold covering decaying lemon"),
    "task8a_010": ("REJECT", "EXCLUDED", "SPEC_W1_010", False, False, False, False, [], "exclude_document_text_poster"),
    "task8a_011": ("REJECT", "EXCLUDED", "SPEC_W1_011", False, False, False, False, [], "exclude_insect_museum_specimen"),
    "task8a_012": ("REJECT", "EXCLUDED", "SPEC_W1_012", False, False, False, False, [], "exclude_insect_museum_specimen"),
    "task8a_013": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_013", False, True, True, False,
                   [(1, 0.28, 0.21, 0.62, 0.72)], "Eaten apple core indoor"),
    "task8a_014": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_014", True, True, True, False,
                   [(1, 0.28, 0.54, 0.38, 0.70)], "Discarded eaten apple core on concrete curb"),
    "task8a_015": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_015", False, True, True, False,
                   [(1, 0.08, 0.20, 0.90, 0.95)], "Bruised cut apples in bowl for disposal"),
    "task8a_016": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_016", False, False, True, False,
                   [(1, 0.06, 0.12, 0.96, 0.87)], "Moldy slice of bread"),
    "task8a_017": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_017", False, False, True, False,
                   [(1, 0.09, 0.32, 0.50, 0.91), (1, 0.49, 0.17, 0.93, 0.79)], "Two moldy rotting tomatoes"),
    "task8a_018": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_018", False, True, True, False,
                   [(1, 0.18, 0.12, 0.84, 0.97)], "Moldy rotting strawberry"),

    # --- Sheet 02 (019..036) ---
    "task8a_019": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_019", False, True, True, False,
                   [(1, 0.01, 0.13, 0.28, 0.88), (1, 0.12, 0.02, 0.83, 0.96), (1, 0.80, 0.12, 0.99, 0.76)],
                   "Three moldy strawberries"),
    "task8a_020": ("REJECT", "EXCLUDED", "SPEC_W1_020", True, True, False, False, [], "exclude_street_vehicles_scene"),
    "task8a_021": ("REJECT", "EXCLUDED", "SPEC_W1_021", True, True, False, False, [], "exclude_distant_street_scene"),
    "task8a_022": ("REJECT", "EXCLUDED", "SPEC_W1_022", False, True, False, False, [], "exclude_industrial_crane_machinery"),
    "task8a_023": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_023", False, False, True, False,
                   [(6, 0.17, 0.15, 0.87, 0.89)], "Crumpled white paper ball"),
    "task8a_024": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_024", False, False, True, False,
                   [(6, 0.02, 0.08, 0.98, 0.92)], "Crumpled lined notebook paper ball"),
    "task8a_025": ("REJECT", "EXCLUDED", "SPEC_W1_025", False, False, False, False, [], "exclude_vintage_book_scan"),
    "task8a_026": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_026", True, True, True, False,
                   [(6, 0.07, 0.01, 0.67, 0.85)], "Discarded dirty paper coffee cup on soil"),
    "task8a_027": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_YELLOW_CUP_27_28", True, True, False, False,
                   [(6, 0.31, 0.06, 0.78, 0.93), (7, 0.31, 0.05, 0.78, 0.25)],
                   "Multi-class: paper coffee cup + white plastic lid outdoors"),
    "task8a_028": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_YELLOW_CUP_27_28", True, True, False, False,
                   [(6, 0.68, 0.34, 0.88, 0.73), (7, 0.68, 0.34, 0.88, 0.43)],
                   "Multi-class: paper coffee cup + plastic lid on bench (grouped with 027)"),
    "task8a_029": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_029", False, True, True, False,
                   [(6, 0.01, 0.11, 0.99, 0.96)], "Dense shredded paper strips inside wicker basket"),
    "task8a_030": ("REJECT", "EXCLUDED", "SPEC_W1_030", False, False, False, False, [], "exclude_non_photographic_cartoon"),
    "task8a_031": ("REJECT", "EXCLUDED", "SPEC_W1_031", False, False, False, False, [], "exclude_non_photographic_cartoon"),
    "task8a_032": ("REJECT", "EXCLUDED", "SPEC_W1_032", False, False, False, False, [], "exclude_poster_graphic"),
    "task8a_033": ("REJECT", "EXCLUDED", "SPEC_W1_033", True, True, False, False, [], "exclude_building_architecture"),
    "task8a_034": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_034", False, True, True, False,
                   [(2, 0.01, 0.35, 0.99, 0.99)], "Dense room of 30+ dumped cardboard boxes"),
    "task8a_035": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_BALE_35_36", False, True, True, False,
                   [(2, 0.04, 0.24, 0.99, 0.96)], "Compressed industrial cardboard bale"),
    "task8a_036": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_BALE_35_36", False, True, True, False,
                   [(2, 0.02, 0.10, 0.99, 0.99)], "Compressed industrial cardboard bale stack"),

    # --- Sheet 03 (037..054) ---
    "task8a_037": ("REJECT", "EXCLUDED", "SPEC_W1_037", False, True, False, False, [], "exclude_in_use_person_operating_baler"),
    "task8a_038": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_038", True, True, True, False,
                   [(2, 0.30, 0.22, 0.66, 0.68)], "Damaged crushed Fragile cardboard box on step"),
    "task8a_039": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_039", True, True, True, False,
                   [(2, 0.01, 0.18, 0.99, 0.95)], "Huge outdoor heap of cardboard boxes with people"),
    "task8a_040": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_040", True, True, True, False,
                   [(2, 0.01, 0.01, 0.82, 0.99)], "Wet damaged cardboard box outdoors"),
    "task8a_041": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_041", False, True, False, False,
                   [(2, 0.31, 0.22, 0.80, 0.49), (2, 0.33, 0.45, 0.81, 0.65), (2, 0.25, 0.59, 0.79, 0.82), (2, 0.39, 0.72, 0.83, 0.94)],
                   "Four stacked molded cardboard egg cartons"),
    "task8a_042": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_042", False, True, False, False,
                   [(2, 0.01, 0.12, 0.86, 0.90)], "Open cardboard egg carton filled with 12 intact fresh eggs"),
    "task8a_043": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_043", False, True, True, False,
                   [(2, 0.06, 0.21, 0.57, 0.64), (2, 0.10, 0.49, 0.89, 0.91)], "Two cardboard boxes stacked on scale"),
    "task8a_044": ("REJECT", "EXCLUDED", "SPEC_W1_044", False, True, False, False, [], "exclude_in_use_person"),
    "task8a_045": ("REJECT", "EXCLUDED", "SPEC_W1_045", True, True, False, False, [], "exclude_in_use_person"),
    "task8a_046": ("REJECT", "EXCLUDED", "SPEC_W1_046", True, True, False, False, [], "exclude_historical_bw_photo"),
    "task8a_047": ("REJECT", "EXCLUDED", "SPEC_W1_047", False, False, False, False, [], "exclude_historical_engraving"),
    "task8a_048": ("REJECT", "EXCLUDED", "SPEC_W1_048", False, False, False, False, [], "exclude_historical_drawing"),
    "task8a_049": ("REJECT", "EXCLUDED", "SPEC_W1_049", False, False, False, False, [], "exclude_historical_engraving"),
    "task8a_050": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_050", False, True, True, False,
                   [(4, 0.22, 0.53, 0.49, 0.88), (4, 0.43, 0.13, 0.77, 0.60)], "Broken glass bottle in two pieces on carpet"),
    "task8a_051": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_051", True, True, True, False,
                   [(4, 0.01, 0.01, 0.99, 0.99)], "Dense crushed glass cullet pile"),
    "task8a_052": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_052", True, True, True, False,
                   [(4, 0.02, 0.08, 0.98, 0.98)], "Broken glass shards cemented onto wall top"),
    "task8a_053": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_053", False, True, False, False,
                   [(4, 0.28, 0.64, 0.67, 0.99)], "Clear glass preserving jar"),
    "task8a_054": ("REJECT", "EXCLUDED", "SPEC_W1_054", True, True, False, False, [], "exclude_vintage_postcard"),

    # --- Sheet 04 (055..067) ---
    "task8a_055": ("REJECT", "EXCLUDED", "SPEC_W1_055", False, False, False, False, [], "exclude_medical_xray"),
    "task8a_056": ("REJECT", "EXCLUDED", "SPEC_W1_056", False, True, False, False, [], "exclude_in_use_hand_holding"),
    "task8a_057": ("REJECT", "EXCLUDED", "SPEC_W1_057", True, True, False, False, [], "exclude_in_use_person_cart"),
    "task8a_058": ("REJECT", "EXCLUDED", "SPEC_W1_058", False, False, False, False, [], "exclude_non_photographic_cartoon"),
    "task8a_059": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_059", True, True, True, False,
                   [(8, 0.44, 0.86, 0.52, 0.93), (8, 0.52, 0.86, 0.59, 0.94), (8, 0.28, 0.72, 0.39, 0.80)],
                   "15+ discarded shoes piled on and around street litter bin with pedestrians"),
    "task8a_060": ("REJECT", "EXCLUDED", "SPEC_W1_060", True, True, False, False, [], "exclude_historical_bw_crowd"),
    "task8a_061": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_061", True, True, True, False,
                   [(8, 0.16, 0.53, 0.36, 0.70), (8, 0.28, 0.52, 0.46, 0.71), (8, 0.41, 0.52, 0.63, 0.73), (8, 0.57, 0.51, 0.81, 0.75)],
                   "Row of worn black shoes on grass curb"),
    "task8a_062": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_062", False, True, True, False,
                   [(8, 0.28, 0.04, 0.66, 0.53), (8, 0.30, 0.44, 0.76, 0.98)], "Pair of muddy winter boots on tile floor"),
    "task8a_063": ("REJECT", "EXCLUDED", "SPEC_W1_063", False, False, False, False, [], "exclude_painting_artwork"),
    "task8a_064": ("REJECT", "EXCLUDED", "SPEC_W1_064", True, True, True, False, [], "exclude_in_use_worn_on_feet"),
    "task8a_065": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_065", True, True, True, False,
                   [(8, 0.15, 0.27, 0.95, 0.78)], "Abandoned muddy brown shoe on dirt ground"),
    "task8a_066": ("REJECT", "EXCLUDED", "SPEC_W1_066", True, True, False, False, [], "exclude_bronze_monument_sculpture"),
    "task8a_067": ("REJECT", "EXCLUDED", "SPEC_W1_067", False, True, False, False, [], "exclude_historical_archive_photo"),

    # --- Sheet 05 (068..083) ---
    "task8a_068": ("REJECT", "EXCLUDED", "SPEC_W1_068", True, True, False, False, [], "exclude_wildlife_butterfly_subject"),
    "task8a_069": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_069", False, True, True, False,
                   [(1, 0.21, 0.10, 0.97, 0.93)], "Hand holding banana peel"),
    "task8a_070": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_070", True, True, True, False,
                   [(1, 0.32, 0.31, 0.38, 0.40)], "Small discarded banana peel on sidewalk pavement"),
    "task8a_071": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_071", True, True, True, False,
                   [(1, 0.41, 0.22, 0.97, 0.67)], "Crushed banana peel on wet street asphalt"),
    "task8a_072": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_072", True, True, True, False,
                   [(1, 0.09, 0.23, 0.99, 0.83)], "Discarded banana peel on dark stone"),
    "task8a_073": ("REJECT", "EXCLUDED", "SPEC_W1_073", True, True, False, False, [], "exclude_art_sculpture"),
    "task8a_074": ("REJECT", "EXCLUDED", "SPEC_W1_074", False, False, False, False, [], "exclude_diagram_graphic"),
    "task8a_075": ("REJECT", "EXCLUDED", "SPEC_W1_075", True, True, False, False, [], "exclude_wildlife_bird_subject"),
    "task8a_076": ("REJECT", "EXCLUDED", "SPEC_W1_076", False, False, False, False, [], "exclude_abstract_pattern"),
    "task8a_077": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_077", True, True, True, False,
                   [(1, 0.26, 0.32, 0.69, 0.81)], "Rotting apple among dead leaves outdoors"),
    "task8a_078": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_078", False, True, True, False,
                   [(7, 0.08, 0.12, 0.90, 0.99), (1, 0.20, 0.24, 0.81, 0.94)],
                   "Multi-class: moldy peaches inside clear plastic punnet tray"),
    "task8a_079": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_079", True, True, True, False,
                   [(1, 0.01, 0.01, 0.99, 0.99)], "Dense pile of 40+ rotting oranges"),
    "task8a_080": ("REJECT", "EXCLUDED", "SPEC_W1_080", False, True, False, False, [], "exclude_wildlife_butterfly_subject"),
    "task8a_081": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_081", True, True, True, False,
                   [(1, 0.11, 0.11, 0.97, 0.82)], "Rotting pear half on grass"),
    "task8a_082": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_082", False, True, True, False,
                   [(1, 0.01, 0.04, 0.99, 0.99), (2, 0.01, 0.01, 0.99, 0.35)],
                   "Multi-class: moldy tomatoes decomposing inside corrugated cardboard crate"),
    "task8a_083": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_083", False, True, False, False,
                   [(6, 0.09, 0.40, 0.42, 0.92)], "Disposable paper coffee cup next to ceramic mug"),

    # --- Sheet 06 (084..101) ---
    "task8a_084": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_084", False, False, True, False,
                   [(1, 0.08, 0.04, 0.94, 0.98)], "Overripe brown banana peel"),
    "task8a_085": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_085", True, True, True, False,
                   [(6, 0.02, 0.48, 0.98, 0.98)], "Dense marathon water station cup litter with runners"),
    "task8a_086": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_BANANA_86_87_88", False, False, True, False,
                   [(1, 0.07, 0.04, 0.95, 0.89)], "Freshly discarded banana peel"),
    "task8a_087": ("REJECT", "EXCLUDED", "SPEC_W1_BANANA_86_87_88", False, False, True, False, [], "duplicate_session_crop_of_086"),
    "task8a_088": ("REJECT", "EXCLUDED", "SPEC_W1_BANANA_86_87_88", False, False, True, False, [], "duplicate_session_crop_of_086"),
    "task8a_089": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_089", False, False, True, False,
                   [(1, 0.32, 0.05, 0.81, 0.94)], "Eaten apple core"),
    "task8a_090": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_090", False, False, True, False,
                   [(1, 0.24, 0.22, 0.56, 0.87)], "Oxidized eaten apple core"),
    "task8a_091": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_091", True, True, True, False,
                   [(1, 0.01, 0.22, 0.98, 0.98), (6, 0.72, 0.01, 0.99, 0.65), (7, 0.20, 0.01, 0.54, 0.34)],
                   "Multi-class: pile of eaten apple cores + paper cup + plastic wrapper"),
    "task8a_092": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_092", False, False, True, False,
                   [(1, 0.11, 0.08, 0.77, 0.89)], "Eaten red apple core"),
    "task8a_093": ("REJECT", "EXCLUDED", "SPEC_W1_093", False, False, False, False, [], "exclude_fresh_intact_orange_not_waste"),
    "task8a_094": ("REJECT", "EXCLUDED", "SPEC_W1_094", False, False, False, False, [], "exclude_extreme_macro_zest"),
    "task8a_095": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_ORANGE_PEEL_95_96", False, True, True, False,
                   [(1, 0.17, 0.38, 0.86, 0.85)], "Torn orange peels on wooden table (grouped with 096)"),
    "task8a_096": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_ORANGE_PEEL_95_96", False, True, True, False,
                   [(1, 0.13, 0.09, 0.91, 0.93)], "Torn orange peels top-down view (grouped with 095)"),
    "task8a_097": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_097", True, True, True, False,
                   [(1, 0.31, 0.39, 0.62, 0.72)], "Broken bird eggshell on outdoor moss"),
    "task8a_098": ("REJECT", "EXCLUDED", "SPEC_W1_098", False, False, False, False, [], "exclude_easter_cake_not_eggshell"),
    "task8a_099": ("REJECT", "EXCLUDED", "SPEC_W1_099", False, False, False, False, [], "exclude_vintage_illustration_postcard"),
    "task8a_100": ("REJECT", "EXCLUDED", "SPEC_W1_100", False, False, False, False, [], "exclude_vintage_illustration_postcard"),
    "task8a_101": ("REJECT", "EXCLUDED", "SPEC_W1_101", False, True, False, False, [], "exclude_in_use_people_event"),

    # --- Sheet 07 (102..119) ---
    "task8a_102": ("REJECT", "EXCLUDED", "SPEC_W1_102", False, True, False, False, [], "exclude_in_use_people_event"),
    "task8a_103": ("REJECT", "EXCLUDED", "SPEC_W1_103", False, True, False, False, [], "exclude_in_use_people_event"),
    "task8a_104": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_104", True, True, True, False,
                   [(1, 0.01, 0.01, 0.99, 0.99)], "Rich organic kitchen compost waste with peels and eggshells"),
    "task8a_105": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_105", False, True, True, False,
                   [(6, 0.01, 0.01, 0.99, 0.99)], "Dense heap of 30+ crumpled paper sheets"),
    "task8a_106": ("REJECT", "EXCLUDED", "SPEC_W1_106", False, False, False, False, [], "exclude_flat_wallpaper_texture"),
    "task8a_107": ("REJECT", "EXCLUDED", "SPEC_W1_107", False, True, False, False, [], "exclude_in_use_person_reading"),
    "task8a_108": ("REJECT", "EXCLUDED", "SPEC_W1_108", False, True, False, False, [], "exclude_in_use_person_reading"),
    "task8a_109": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_109", False, True, False, False,
                   [(6, 0.26, 0.02, 0.77, 0.97), (7, 0.25, 0.02, 0.77, 0.15)],
                   "Multi-class: paper coffee cup + white plastic lid"),
    "task8a_110": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_CUP_SLEEVE_110_111", False, True, False, False,
                   [(6, 0.23, 0.03, 0.61, 0.79), (7, 0.22, 0.03, 0.61, 0.13), (2, 0.59, 0.72, 0.99, 0.90)],
                   "Multi-class: paper cup + plastic lid + corrugated cardboard sleeve"),
    "task8a_111": ("REJECT", "EXCLUDED", "SPEC_W1_CUP_SLEEVE_110_111", False, True, False, False, [], "duplicate_session_photo_of_110"),
    "task8a_112": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_112", False, True, False, False,
                   [(6, 0.34, 0.07, 0.69, 0.94), (6, 0.13, 0.11, 0.34, 0.62), (7, 0.13, 0.11, 0.34, 0.18), (7, 0.34, 0.07, 0.70, 0.21), (2, 0.14, 0.25, 0.33, 0.48), (2, 0.36, 0.42, 0.67, 0.76)],
                   "Multi-class: two paper coffee cups + two white plastic lids + two cardboard sleeves"),
    "task8a_113": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_113", False, True, True, False,
                   [(6, 0.40, 0.14, 0.63, 0.58), (6, 0.65, 0.13, 0.89, 0.58), (6, 0.10, 0.41, 0.69, 0.87), (7, 0.01, 0.06, 0.99, 0.98)],
                   "Multi-class: used paper cups and paper napkins on plastic tray"),
    "task8a_114": ("REJECT", "EXCLUDED", "SPEC_W1_114", False, True, False, False, [], "exclude_in_use_person_holding_cup"),
    "task8a_115": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_115", False, True, True, False,
                   [(2, 0.02, 0.20, 0.85, 0.94), (7, 0.02, 0.05, 0.43, 0.65), (7, 0.44, 0.02, 0.79, 0.53)],
                   "Multi-class: cardboard pizza box + two clear plastic clamshell containers"),
    "task8a_116": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_116", False, True, True, False,
                   [(2, 0.10, 0.36, 0.36, 0.87), (2, 0.34, 0.18, 0.66, 0.89), (2, 0.64, 0.12, 0.99, 0.91), (1, 0.11, 0.65, 0.28, 0.81), (1, 0.37, 0.56, 0.64, 0.75), (1, 0.67, 0.57, 0.95, 0.83)],
                   "Multi-class: three open greasy cardboard pizza boxes with leftover food"),
    "task8a_117": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_117", False, True, True, False,
                   [(2, 0.01, 0.45, 0.99, 0.98)], "Stack of used cardboard pizza boxes"),
    "task8a_118": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_118", False, True, False, False,
                   [(2, 0.17, 0.20, 0.88, 0.91)], "Stack of four cardboard pizza boxes"),
    "task8a_119": ("REJECT", "EXCLUDED", "SPEC_W1_119", True, True, False, False, [], "exclude_distant_bin_signage"),

    # --- Sheet 08 (120..137) ---
    "task8a_120": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_120", False, True, True, False,
                   [(2, 0.14, 0.10, 0.89, 0.92)], "Stack of used cardboard banana boxes"),
    "task8a_121": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_121", True, True, True, False,
                   [(2, 0.17, 0.03, 0.59, 0.40)], "B&W photo of cardboard banana box on plastic crates"),
    "task8a_122": ("REJECT", "EXCLUDED", "SPEC_W1_122", False, False, False, False, [], "exclude_19th_century_museum_artifact"),
    "task8a_123": ("REJECT", "EXCLUDED", "SPEC_W1_123", False, False, False, False, [], "exclude_19th_century_museum_artifact"),
    "task8a_124": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_124", True, True, True, False,
                   [(4, 0.07, 0.07, 0.96, 0.86)], "Broken glass bottle shard on ground"),
    "task8a_125": ("REJECT", "EXCLUDED", "SPEC_W1_125", True, True, False, False, [], "exclude_distant_trucks_plaza"),
    "task8a_126": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_126", True, True, True, False,
                   [(4, 0.45, 0.27, 0.52, 0.61), (4, 0.26, 0.50, 0.43, 0.58), (4, 0.53, 0.53, 0.66, 0.62), (4, 0.66, 0.53, 0.83, 0.69), (6, 0.56, 0.59, 0.69, 0.67)],
                   "Multi-class: four empty green glass bottles + discarded paper cigarette pack"),
    "task8a_127": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_127", True, True, True, False,
                   [(4, 0.10, 0.04, 0.90, 0.98)], "Shattered brown glass beer bottles on grass"),
    "task8a_128": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_GLASS_BEACH_128_129", True, True, True, False,
                   [(4, 0.01, 0.01, 0.99, 0.99)], "Dense dump of ratusan glass fragments on beach"),
    "task8a_129": ("REJECT", "EXCLUDED", "SPEC_W1_GLASS_BEACH_128_129", True, True, True, False, [], "duplicate_of_128"),
    "task8a_130": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_130", True, True, True, False,
                   [(4, 0.36, 0.60, 0.57, 0.72)], "Broken green glass bottle neck on sand"),
    "task8a_131": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_131", True, True, True, False,
                   [(4, 0.35, 0.33, 0.80, 0.78)], "Broken clear glass flip-top bottle on dirt ground"),
    "task8a_132": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_132", False, True, True, False,
                   [(4, 0.24, 0.73, 0.39, 0.87), (4, 0.39, 0.33, 0.57, 0.46)], "Shattered glass bottle pieces in spilled liquid"),
    "task8a_133": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_133", False, True, True, False,
                   [(4, 0.38, 0.52, 0.52, 0.68)], "Broken brown glass bottle on floor"),
    "task8a_134": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_134", True, True, True, False,
                   [(4, 0.12, 0.38, 0.95, 0.98)], "Dense cluster of 50+ bottles and bags next to street bin"),
    "task8a_135": ("REJECT", "EXCLUDED", "SPEC_W1_135", True, True, False, False, [], "exclude_tree_signage"),
    "task8a_136": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_136", False, True, True, False,
                   [(7, 0.33, 0.48, 0.66, 0.69)], "Remapped to 7:plastic — discarded clear PET bottle on floor"),
    "task8a_137": ("REJECT", "EXCLUDED", "SPEC_W1_137", False, True, False, False, [], "exclude_in_use_person_drinking"),

    # --- Sheet 09 (138..150) & Sheet 10 (151..154) ---
    "task8a_138": ("REJECT", "EXCLUDED", "SPEC_W1_138", True, True, False, False, [], "exclude_night_crowd_scene"),
    "task8a_139": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_PINK_SHOES_139_140", True, True, True, False,
                   [(8, 0.17, 0.33, 0.40, 0.69), (8, 0.41, 0.45, 0.83, 0.82)], "Pair of abandoned pink shoes on stone ledge"),
    "task8a_140": ("REJECT", "EXCLUDED", "SPEC_W1_PINK_SHOES_139_140", True, True, True, False, [], "duplicate_file_of_139"),
    "task8a_141": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_141", True, True, True, False,
                   [(8, 0.28, 0.19, 0.70, 0.72)], "Weathered abandoned shoe washed up on rocky shore"),
    "task8a_142": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_142", True, True, True, False,
                   [(8, 0.33, 0.33, 0.45, 0.56)], "Multiple pairs of shoes hanging from overhead wire"),
    "task8a_143": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_143", True, True, True, False,
                   [(8, 0.11, 0.85, 0.17, 0.99)], "Slipper in foreground with dense trash heap in background"),
    "task8a_144": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_144", True, True, True, False,
                   [(8, 0.28, 0.23, 0.68, 0.78)], "Abandoned black flat shoe on pavement among dead leaves"),
    "task8a_145": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_145", False, True, True, False,
                   [(8, 0.04, 0.10, 0.58, 0.56), (8, 0.22, 0.18, 0.78, 0.88)], "Pair of worn dirty brown leather shoes"),
    "task8a_146": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_146", True, True, True, False,
                   [(8, 0.05, 0.55, 0.98, 0.95)], "Dense row of 25+ old shoes against brick wall"),
    "task8a_147": ("REJECT", "EXCLUDED", "SPEC_W1_147", False, False, False, False, [], "exclude_extreme_macro_shoelace"),
    "task8a_148": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_148", True, True, True, False,
                   [(8, 0.01, 0.01, 0.99, 0.99)], "Dense shoe rack at market stall"),
    "task8a_149": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_149", True, True, True, False,
                   [(8, 0.06, 0.31, 0.84, 0.71)], "Muddy worn black leather boot outdoors"),
    "task8a_150": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_150", False, True, True, False,
                   [(8, 0.01, 0.18, 0.80, 0.74), (8, 0.01, 0.55, 0.70, 0.99), (8, 0.25, 0.03, 0.89, 0.44), (8, 0.37, 0.01, 0.95, 0.26)],
                   "Four worn dirty canvas sneakers"),
    "task8a_151": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_151", True, True, False, False,
                   [(4, 0.35, 0.10, 0.52, 0.90)], "Empty clear glass bottle outdoors"),
    "task8a_152": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_BROWN_BOTTLES_152_153", False, False, False, False,
                   [(4, 0.06, 0.07, 0.22, 0.70), (4, 0.23, 0.07, 0.38, 0.72), (4, 0.37, 0.08, 0.52, 0.79), (4, 0.52, 0.09, 0.70, 0.88), (4, 0.72, 0.09, 0.93, 0.95)],
                   "Five empty brown glass bottles"),
    "task8a_153": ("REJECT", "EXCLUDED", "SPEC_W1_BROWN_BOTTLES_152_153", False, False, False, False, [], "duplicate_session_angle_of_152"),
    "task8a_154": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_154", True, True, True, False,
                   [(4, 0.12, 0.35, 0.93, 0.69)], "Empty brown glass bottle lying on concrete ground"),

    # --- Sheets 11..13 (155..204, Open Images V7 + Wikimedia Supplement) ---
    "task8a_155": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_155", False, True, False, False,
                   [(2, 0.0325, 0.0642, 0.9688, 0.9758)], "Open Images V7 verified: cardboard laptop box"),
    "task8a_156": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_156", False, True, False, False,
                   [(2, 0.1602, 0.1120, 0.7910, 0.8980)], "Open Images V7 verified: open cardboard box"),
    "task8a_157": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_157", False, True, False, False,
                   [(2, 0.0906, 0.0208, 0.3906, 0.2500), (2, 0.3406, 0.0000, 0.8844, 0.5000), (2, 0.4406, 0.1708, 1.0000, 0.9792), (2, 0.0100, 0.2000, 0.5600, 0.9800)],
                   "Open Images V7 verified: four molded cardboard egg cartons"),
    "task8a_158": ("REJECT", "EXCLUDED", "SPEC_W1_OI_158", False, True, False, False, [], "exclude_wooden_metal_chest_not_cardboard"),
    "task8a_159": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_159", False, True, False, False,
                   [(2, 0.0900, 0.0500, 0.8900, 0.8900)], "Open Images V7 verified: patterned cardboard box"),
    "task8a_160": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_160", False, True, False, False,
                   [(2, 0.0000, 0.3900, 0.7200, 0.9700), (4, 0.6600, 0.4400, 0.8600, 0.7900)],
                   "Open Images V7 multi-class: cardboard packaging box + glass perfume bottle"),
    "task8a_161": ("REJECT", "EXCLUDED", "SPEC_W1_OI_161", False, True, False, False, [], "exclude_ceramic_mug_with_drink_not_paper"),
    "task8a_162": ("REJECT", "EXCLUDED", "SPEC_W1_OI_162", False, True, False, False, [], "exclude_ceramic_cup_with_drink_not_paper"),
    "task8a_163": ("REJECT", "EXCLUDED", "SPEC_W1_OI_163", False, True, False, False, [], "exclude_ceramic_mug_not_paper"),
    "task8a_164": ("REJECT", "EXCLUDED", "SPEC_W1_OI_164", False, True, False, False, [], "exclude_ceramic_cup_and_food_not_paper"),
    "task8a_165": ("REJECT", "EXCLUDED", "SPEC_W1_OI_165", False, True, False, False, [], "exclude_porcelain_gravy_boat"),
    "task8a_166": ("REJECT", "EXCLUDED", "SPEC_W1_OI_166", False, True, False, False, [], "exclude_ceramic_teacup_not_paper"),
    "task8a_167": ("REJECT", "EXCLUDED", "SPEC_W1_OI_167", False, True, False, False, [], "exclude_library_bookshelves_in_use"),
    "task8a_168": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_168", False, True, False, False,
                   [(6, 0.0000, 0.2500, 0.7300, 1.0000)], "Open Images V7 verified: open paper book/notebook"),
    "task8a_169": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_169", False, True, False, False,
                   [(4, 0.0000, 0.0100, 0.9400, 0.4500), (4, 0.0000, 0.4500, 0.9900, 0.9900)],
                   "Open Images V7 verified: two glass bottles"),
    "task8a_170": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_170", False, True, True, False,
                   [(7, 0.0300, 0.0400, 0.4600, 0.9600), (7, 0.4600, 0.0600, 0.9000, 0.9600)],
                   "Open Images V7 verified (remapped to 7:plastic): two large clear plastic jugs/bottles"),
    "task8a_171": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_171", False, True, False, False,
                   [(4, 0.0000, 0.0100, 0.1400, 0.4900), (4, 0.8900, 0.0800, 0.9900, 0.4900), (4, 0.2400, 0.3600, 0.4900, 0.9200), (4, 0.5300, 0.0200, 0.8300, 0.9100)],
                   "Open Images V7 verified: glass bottles and glass mugs"),
    "task8a_172": ("REJECT", "EXCLUDED", "SPEC_W1_OI_172", False, False, False, False, [], "exclude_commercial_product_collage"),
    "task8a_173": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_173", False, True, False, False,
                   [(4, 0.3100, 0.1200, 0.6400, 0.9700)], "Open Images V7 verified: glass bottle"),
    "task8a_174": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_174", True, True, False, False,
                   [(4, 0.3500, 0.1700, 0.5900, 0.6700)], "Open Images V7 verified: dark glass bottle in snow outdoors"),
    "task8a_175": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_OI_175", False, True, False, False,
                   [(4, 0.2200, 0.0300, 0.3900, 0.7200)], "Multiple bottles on refrigerator shelf with partial OI boxes"),
    "task8a_176": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_176", False, True, False, False,
                   [(4, 0.3600, 0.0100, 0.5800, 0.8200), (4, 0.5000, 0.1700, 0.6200, 0.8600), (4, 0.6800, 0.0300, 0.8800, 0.9000)],
                   "Open Images V7 verified: dark glass wine bottle and two glass wine flutes"),
    "task8a_177": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_177", False, True, False, False,
                   [(4, 0.3300, 0.1200, 0.6100, 0.8200), (4, 0.5800, 0.1200, 0.8700, 0.7800), (4, 0.0800, 0.1300, 0.3700, 0.7800)],
                   "Open Images V7 verified: glass wine bottle and two wine glasses"),
    "task8a_178": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_178", False, True, True, False,
                   [(8, 0.2300, 0.1700, 0.4900, 0.8200), (8, 0.4700, 0.1500, 0.8400, 0.7700)],
                   "Open Images V7 verified: pair of worn brown leather boots"),
    "task8a_179": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_179", False, True, True, False,
                   [(8, 0.0600, 0.1500, 0.6400, 0.9200), (8, 0.3900, 0.1400, 0.7600, 0.7300)],
                   "Open Images V7 verified: pair of grey leather boots"),
    "task8a_180": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_180", False, True, False, False,
                   [(8, 0.0600, 0.1300, 0.8900, 0.9300)], "Open Images V7 verified: tan suede ankle boot"),
    "task8a_181": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_181", False, False, False, False,
                   [(8, 0.0500, 0.0900, 0.5200, 0.9600), (8, 0.5700, 0.0700, 0.9700, 0.8700)],
                   "Open Images V7 verified: pair of black boots"),
    "task8a_182": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_182", False, True, False, False,
                   [(8, 0.0300, 0.1700, 0.8600, 0.8200)], "Open Images V7 verified: black sandal"),
    "task8a_183": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_183", False, True, False, False,
                   [(8, 0.0800, 0.1200, 0.9000, 0.8900)], "Open Images V7 verified: black/red high-heel sandal"),
    "task8a_184": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_184", False, True, True, False,
                   [(8, 0.2900, 0.4000, 0.6700, 0.7700), (8, 0.4300, 0.4300, 0.8400, 0.8000)],
                   "Open Images V7 verified: pair of white slippers/sandals on dark tile"),
    "task8a_185": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_185", False, True, False, False,
                   [(8, 0.2100, 0.1000, 0.5400, 0.9900), (8, 0.4500, 0.0100, 0.8600, 0.9900)],
                   "Open Images V7 verified: pair of flip-flop sandals"),
    "task8a_186": ("REJECT", "EXCLUDED", "SPEC_W1_OI_186", False, True, False, False, [], "exclude_plush_toy_not_clothes"),
    "task8a_187": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_187", False, True, True, False,
                   [(3, 0.0600, 0.1100, 0.8500, 0.8700)], "Open Images V7 verified: worn black glove"),
    "task8a_188": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_188", False, True, False, False,
                   [(3, 0.0500, 0.1400, 0.5800, 0.9700), (3, 0.4600, 0.0800, 0.9500, 0.9100)],
                   "Open Images V7 verified: pair of blue gloves"),
    "task8a_189": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_189", False, True, False, False,
                   [(3, 0.0000, 0.1100, 0.6200, 0.8900), (3, 0.3400, 0.1300, 0.9900, 0.9100)],
                   "Open Images V7 verified: pair of knitted mittens"),
    "task8a_190": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_190", False, True, False, False,
                   [(3, 0.0900, 0.0500, 0.9900, 0.9900)], "Open Images V7 verified: stack of cloth towels"),
    "task8a_191": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_191", False, False, False, False,
                   [(3, 0.0300, 0.0600, 0.4300, 0.8400), (3, 0.4800, 0.0600, 0.8700, 0.8600)],
                   "Open Images V7 verified: two beige cloth hand towels"),
    "task8a_192": ("REJECT", "EXCLUDED", "SPEC_W1_OI_192", False, True, False, False, [], "exclude_cocktail_drink_in_use"),
    "task8a_193": ("REJECT", "EXCLUDED", "SPEC_W1_OI_193", False, True, False, False, [], "exclude_cocktail_drink_in_use"),
    "task8a_194": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_194", True, True, False, True,
                   [], "Hard negative (0 boxes): clean outdoor waste container on grass with zero exposed waste"),
    "task8a_195": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_OI_195", True, True, False, True,
                   [], "Hard negative (0 boxes): clean blue recycling bin outdoors with zero exposed waste"),
    "task8a_196": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_OI_196", True, True, False, True,
                   [], "Hard negative (0 boxes): closed green wheelie bins against brick wall with zero waste"),
    "task8a_197": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_197", True, True, True, False,
                   [(3, 0.02, 0.30, 0.98, 0.99)], "Large mixed heap of dumped clothing and bags on street"),
    "task8a_198": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_198", False, True, True, False,
                   [(3, 0.25, 0.29, 0.80, 0.96)], "Discarded red sports jersey and dark cloth on floor"),
    "task8a_199": ("REJECT", "EXCLUDED", "SPEC_W1_199", False, True, False, False, [], "exclude_dark_indoor_room"),
    "task8a_200": ("REJECT", "EXCLUDED", "SPEC_W1_200", True, True, False, False, [], "exclude_ambiguous_distant_tree_branch"),
    "task8a_201": ("REJECT", "EXCLUDED", "SPEC_W1_201", False, False, False, False, [], "exclude_vintage_poster_illustration"),
    "task8a_202": ("REJECT", "EXCLUDED", "SPEC_W1_202", False, False, False, False, [], "exclude_vintage_poster_illustration"),
    "task8a_203": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_203", True, True, True, False,
                   [(5, 0.88, 0.93, 0.96, 0.99)], "Many tiny cigarette butts in tram groove + crushed can at corner"),
    "task8a_204": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_204", True, True, True, False,
                   [(9, 0.51, 0.47, 0.63, 0.59)], "Discarded cigarette butt among dry leaves outdoors"),

    # --- Sheet 14 (205..211) ---
    "task8a_205": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_205", False, True, True, False,
                   [(7, 0.02, 0.02, 0.98, 0.98)], "Collage of 35 river plastic fragments"),
    "task8a_206": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_206", True, True, True, False,
                   [(7, 0.18, 0.29, 0.83, 0.82)], "Discarded white plastic bag on gravel ground"),
    "task8a_207": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_207", True, True, True, False,
                   [(7, 0.16, 0.19, 0.80, 0.85)], "Discarded plastic bag in outdoor leaf litter"),
    "task8a_208": ("REJECT", "EXCLUDED", "SPEC_W1_208", True, True, False, False, [], "exclude_split_composite_photo"),
    "task8a_209": ("REJECT", "EXCLUDED", "SPEC_W1_209", True, True, True, False, [], "duplicate_of_task7_corroded_batteries_imgp6146"),
    "task8a_210": ("REJECT", "EXCLUDED", "SPEC_W1_210", True, True, True, False, [], "duplicate_of_task7_corroded_batteries_imgp6157"),
    "task8a_211": ("REJECT", "EXCLUDED", "SPEC_W1_211", True, True, True, False, [], "duplicate_of_task7_corroded_batteries_dscn6962"),

    # --- Sheet 15 (212..221) ---
    "task8a_212": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_VARTA_CR2032_212_213", False, False, True, False,
                   [(0, 0.09, 0.29, 0.51, 0.94), (0, 0.48, 0.14, 0.95, 0.79)],
                   "Two used VARTA CR2032 lithium button cell batteries"),
    "task8a_213": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_VARTA_CR2032_212_213", False, False, True, False,
                   [(0, 0.06, 0.17, 0.55, 0.93), (0, 0.50, 0.14, 0.99, 0.87)],
                   "Two used VARTA CR2032 button cell batteries in blister wrap (grouped with 212)"),
    "task8a_214": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_214", False, False, True, False,
                   [(0, 0.07, 0.08, 0.94, 0.93)],
                   "Used SR626W silver-oxide button cell battery from medical sensor"),
    "task8a_215": ("NEEDS_REVIEW", "MANUAL_REVIEW_HOLD", "SPEC_W1_215", False, True, False, False,
                   [(5, 0.01, 0.01, 0.99, 0.98)], "Dense table of 100+ beverage cans"),
    "task8a_216": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_216", False, False, False, False,
                   [(5, 0.01, 0.01, 0.67, 0.83)], "Aluminum beverage can top with pull tab"),
    "task8a_217": ("REJECT", "EXCLUDED", "SPEC_W1_217", True, True, False, False, [], "exclude_art_sculpture_trash_man"),
    "task8a_218": ("REJECT", "EXCLUDED", "SPEC_W1_218", True, True, False, False, [], "exclude_in_use_person_costume"),
    "task8a_219": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_219", True, True, True, False,
                   [(7, 0.03, 0.23, 0.90, 0.91)], "Remapped to 7:plastic — torn white plastic bag littered in grass"),
    "task8a_220": ("QA_PASS", "VAL_REAL_WAVE1", "SPEC_W1_220", False, True, True, False,
                   [
                       (5, 0.05, 0.35, 0.28, 0.50),
                       (5, 0.26, 0.39, 0.57, 0.52),
                       (5, 0.55, 0.38, 0.95, 0.65),
                       (5, 0.26, 0.52, 0.60, 0.66),
                       (5, 0.09, 0.65, 0.66, 0.80),
                       (7, 0.13, 0.17, 0.54, 0.41),
                       (7, 0.63, 0.62, 0.96, 0.79),
                       (7, 0.58, 0.11, 0.94, 0.38),
                       (4, 0.52, 0.20, 0.71, 0.42),
                   ],
                   "Multi-class litter collected from walk: 5 crushed aluminum cans + plastic bottle + plastic cup + plastic bag + glass bottle"),
    "task8a_221": ("QA_PASS", "TRAIN_WAVE1", "SPEC_W1_221", True, True, True, False,
                   [(9, 0.44, 0.31, 0.70, 0.58)], "Crushed deteriorated litter packet flattened on asphalt"),
}


def xyxy_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float) -> Tuple[float, float, float, float]:
    xmin = max(0.0, min(1.0, xmin))
    ymin = max(0.0, min(1.0, ymin))
    xmax = max(0.0, min(1.0, xmax))
    ymax = max(0.0, min(1.0, ymax))
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    bw = max(0.002, xmax - xmin)
    bh = max(0.002, ymax - ymin)
    return round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)


def render_overlay_image(img_path: Path, boxes_xyxy: List[Tuple[int, float, float, float, float]], verdict: str, split: str, out_path: Path) -> None:
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    draw = ImageDraw.Draw(rgb)

    for cls_id, xmin, ymin, xmax, ymax in boxes_xyxy:
        x0, y0 = int(xmin * w), int(ymin * h)
        x1, y1 = int(xmax * w), int(ymax * h)
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))
        for offset in range(3):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
        label_str = f"{cls_id}:{CLASS_NAMES[cls_id]}"
        tw = len(label_str) * 7 + 8
        ty = max(0, y0 - 18)
        draw.rectangle([x0, ty, x0 + tw, ty + 18], fill=color)
        draw.text((x0 + 4, ty + 2), label_str, fill=(10, 10, 14))

    banner_color = (24, 120, 55) if verdict == "QA_PASS" else ((180, 120, 20) if verdict == "NEEDS_REVIEW" else (160, 35, 35))
    draw.rectangle([0, 0, w, 26], fill=banner_color)
    draw.text((8, 5), f"{img_path.stem[:42]} | {verdict} | {split} | boxes={len(boxes_xyxy)}", fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, quality=90)


def render_contact_sheet(records: List[Dict[str, Any]], title: str, out_path: Path, cols: int = 4) -> None:
    if not records:
        return
    cell_w, cell_h = 420, 340
    header_h = 46
    top_banner = 44
    rows = (len(records) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, top_banner + rows * (cell_h + header_h)), (20, 22, 26))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, cols * cell_w, top_banner], fill=(34, 40, 52))
    draw.text((14, 13), f"{title} ({len(records)} images)", fill=(255, 235, 130))

    for idx, rec in enumerate(records):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = top_banner + r * (cell_h + header_h)

        verdict = rec["qa_verdict"]
        hdr_bg = (28, 92, 48) if verdict == "QA_PASS" else ((130, 90, 24) if verdict == "NEEDS_REVIEW" else (115, 32, 32))
        draw.rectangle([x0 + 2, y0 + 2, x0 + cell_w - 3, y0 + header_h], fill=hdr_bg)
        src_tag = "OI_V7" if rec["source_family"] == "Open Images V7" else "WIKI"
        draw.text((x0 + 8, y0 + 6), f"{rec['short_key']} [{src_tag}] {verdict} -> {rec['proposed_split']}", fill=(255, 255, 255))
        cls_summary = ", ".join(rec["present_class_names"]) if rec["present_class_names"] else ("HARD_NEG (0 box)" if rec["is_hard_negative"] else rec["qa_note"][:32])
        draw.text((x0 + 8, y0 + 24), f"Cls: {cls_summary[:44]} | boxes={rec['box_count']}", fill=(220, 230, 240))

        ov_p = REPO_ROOT / rec["overlay_path"] if rec.get("overlay_path") else (REPO_ROOT / rec["local_path"])
        with Image.open(ov_p) as im:
            thumb = im.convert("RGB").resize((cell_w - 12, cell_h - 12), Image.Resampling.BILINEAR)
        canvas.paste(thumb, (x0 + 6, y0 + header_h + 4))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def load_external_hashes() -> Tuple[Dict[str, str], List[Tuple[str, imagehash.ImageHash]]]:
    sha_map: Dict[str, str] = {}
    phash_list: List[Tuple[str, imagehash.ImageHash]] = []

    # 1. Existing staging manifests
    manifest_paths = [
        ("batch_a_mini01", REPO_ROOT / "data/acquisition_staging/batch_a_mini01/batch_a_mini01_manifest.json"),
        ("phase4e_staging", REPO_ROOT / "data/acquisition_staging/phase4e_staging/phase4e_Manifest.json"),
        ("task7_battery", REPO_ROOT / "data/acquisition_staging/task7_diverse_battery_batch/task7_battery_qa_manifest.json"),
    ]
    for tag, mp in manifest_paths:
        if mp.exists():
            d = json.loads(mp.read_text(encoding="utf-8"))
            recs = d.get("records") or d.get("candidates") or []
            for r in recs:
                if r.get("qa_verdict") == "REJECT":
                    continue
                s = r.get("sha256")
                if s:
                    sha_map[s] = f"{tag}:{r.get('candidate_id', r.get('filename', ''))}"
                ph = r.get("phash_0") or r.get("phash")
                if ph:
                    try:
                        phash_list.append((f"{tag}:{r.get('candidate_id', '')}", imagehash.hex_to_hash(ph)))
                    except Exception:
                        pass

    # 2. Holdout 27 manifest
    holdout_p = REPO_ROOT / "reports/qa_acceptance/manifest_27_images.json"
    if holdout_p.exists():
        hd = json.loads(holdout_p.read_text(encoding="utf-8"))
        for r in hd if isinstance(hd, list) else hd.get("images", []):
            s = r.get("sha256")
            if s:
                sha_map[s] = f"holdout27:{r.get('filename', '')}"

    return sha_map, phash_list


def main() -> None:
    for d in [LABELS_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for old_f in list(LABELS_DIR.glob("*.txt")) + list(LABELS_REVIEW_DIR.glob("*.txt")):
        old_f.unlink(missing_ok=True)

    raw_data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    candidates = raw_data["candidates"]

    ext_sha_map, ext_phash_list = load_external_hashes()
    records: List[Dict[str, Any]] = []

    for cand in candidates:
        sk = cand["short_key"]
        if sk not in QA_DECISIONS:
            raise KeyError(f"Missing QA decision for {sk} ({cand['filename']})")

        verdict, split, spec_group, is_outdoor, is_complex_bg, is_dirty, is_hard_neg, boxes_xyxy, note = QA_DECISIONS[sk]
        img_path = REPO_ROOT / cand["local_path"]
        stem = img_path.stem

        yolo_lines: List[str] = []
        norm_boxes: List[Dict[str, Any]] = []
        present_classes_set = set()

        for cls_id, xmin, ymin, xmax, ymax in boxes_xyxy:
            cx, cy, bw, bh = xyxy_to_yolo(xmin, ymin, xmax, ymax)
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            present_classes_set.add(cls_id)
            norm_boxes.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "xyxy_norm": [round(xmin, 4), round(ymin, 4), round(xmax, 4), round(ymax, 4)],
                "yolo_xywh_norm": [cx, cy, bw, bh],
            })

        label_rel_path = None
        if verdict == "QA_PASS":
            lp = LABELS_DIR / f"{stem}.txt"
            lp.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            label_rel_path = f"data/acquisition_staging/task8a_multiclass_batch/labels/{stem}.txt"
        elif verdict == "NEEDS_REVIEW":
            lp = LABELS_REVIEW_DIR / f"{stem}.txt"
            lp.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            label_rel_path = f"data/acquisition_staging/task8a_multiclass_batch/labels_needs_manual_review/{stem}.txt"

        ov_path = OVERLAYS_DIR / f"{stem}_overlay.jpg"
        render_overlay_image(img_path, boxes_xyxy, verdict, split, ov_path)
        ov_rel_path = f"data/acquisition_staging/task8a_multiclass_batch/overlays/{stem}_overlay.jpg"

        present_cls_ids = sorted(present_classes_set)
        present_cls_names = [f"{cid}: {CLASS_NAMES[cid]}" for cid in present_cls_ids]
        src_family = "Open Images V7" if "Open Images" in cand.get("source_name", "") else "Wikimedia Commons"

        rec = {
            **cand,
            "source_family": src_family,
            "qa_verdict": verdict,
            "proposed_split": split,
            "specimen_group": spec_group,
            "is_outdoor": is_outdoor,
            "is_complex_background": is_complex_bg,
            "is_dirty_or_damaged": is_dirty,
            "is_hard_negative": is_hard_neg,
            "is_multi_class": len(present_cls_ids) >= 2,
            "present_class_ids": present_cls_ids,
            "present_class_names": present_cls_names,
            "box_count": len(norm_boxes),
            "boxes": norm_boxes,
            "label_path": label_rel_path,
            "overlay_path": ov_rel_path,
            "qa_note": note,
            "approved_for_train": False,
        }
        records.append(rec)

    # Leakage verification on QA_PASS records
    qa_pass_recs = [r for r in records if r["qa_verdict"] == "QA_PASS"]
    external_sha_hits = []
    external_phash_hits = []
    for r in qa_pass_recs:
        if r["sha256"] in ext_sha_map:
            external_sha_hits.append((r["short_key"], ext_sha_map[r["sha256"]]))
        for rot_hex in r["phash_rotations"]:
            rh = imagehash.hex_to_hash(rot_hex)
            for ext_id, ext_h in ext_phash_list:
                if rh - ext_h <= 8:
                    external_phash_hits.append((r["short_key"], ext_id, int(rh - ext_h)))
                    break

    # Cross-split leakage check between TRAIN_WAVE1 and VAL_REAL_WAVE1
    train_recs = [r for r in qa_pass_recs if r["proposed_split"] == "TRAIN_WAVE1"]
    val_recs = [r for r in qa_pass_recs if r["proposed_split"] == "VAL_REAL_WAVE1"]
    train_specimens = {r["specimen_group"] for r in train_recs}
    val_specimens = {r["specimen_group"] for r in val_recs}
    shared_specimens = sorted(train_specimens & val_specimens)

    cross_split_phash_hits = []
    for tr in train_recs:
        for vr in val_recs:
            for rot_hex in tr["phash_rotations"]:
                dist = imagehash.hex_to_hash(rot_hex) - imagehash.hex_to_hash(vr["phash_0"])
                if dist <= 8:
                    cross_split_phash_hits.append((tr["short_key"], vr["short_key"], int(dist)))

    # Render contact sheets
    pass_chunks = [qa_pass_recs[i : i + 16] for i in range(0, len(qa_pass_recs), 16)]
    sheet_files = []
    for idx, ch in enumerate(pass_chunks, 1):
        sp = SHEETS_DIR / f"task8_wave1_qa_pass_sheet_{idx:02d}.jpg"
        render_contact_sheet(ch, f"Task 8 Wave 1 — QA_PASS Sheet {idx:02d}/{len(pass_chunks)}", sp)
        sheet_files.append(f"data/acquisition_staging/task8a_multiclass_batch/contact_sheets/{sp.name}")

    rev_recs = [r for r in records if r["qa_verdict"] == "NEEDS_REVIEW"]
    if rev_recs:
        sp = SHEETS_DIR / "task8_wave1_needs_review_sheet.jpg"
        render_contact_sheet(rev_recs, "Task 8 Wave 1 — NEEDS_REVIEW Hold Sheet", sp)
        sheet_files.append(f"data/acquisition_staging/task8a_multiclass_batch/contact_sheets/{sp.name}")

    rej_recs = [r for r in records if r["qa_verdict"] == "REJECT"]
    rej_chunks = [rej_recs[i : i + 20] for i in range(0, len(rej_recs), 20)]
    for idx, ch in enumerate(rej_chunks, 1):
        sp = SHEETS_DIR / f"task8_wave1_reject_sheet_{idx:02d}.jpg"
        render_contact_sheet(ch, f"Task 8 Wave 1 — REJECT Audit Sheet {idx:02d}/{len(rej_chunks)}", sp)
        sheet_files.append(f"data/acquisition_staging/task8a_multiclass_batch/contact_sheets/{sp.name}")

    # Per-class summary for Wave 1 QA_PASS
    wave1_per_class: Dict[str, Any] = {}
    for cid, cname in CLASS_NAMES.items():
        key = f"{cid}: {cname}"
        c_pass = [r for r in qa_pass_recs if cid in r["present_class_ids"]]
        c_train = [r for r in c_pass if r["proposed_split"] == "TRAIN_WAVE1"]
        c_val = [r for r in c_pass if r["proposed_split"] == "VAL_REAL_WAVE1"]
        c_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid) for r in c_pass)
        wave1_per_class[key] = {
            "qa_pass_independent_images": len(c_pass),
            "train_wave1_images": len(c_train),
            "val_wave1_images": len(c_val),
            "total_boxes": c_boxes,
            "sources": sorted({r["source_family"] for r in c_pass}),
            "specimen_groups": len({r["specimen_group"] for r in c_pass}),
            "multi_class_images": sum(1 for r in c_pass if r["is_multi_class"]),
            "complex_bg_images": sum(1 for r in c_pass if r["is_complex_background"]),
            "outdoor_images": sum(1 for r in c_pass if r["is_outdoor"]),
            "dirty_or_damaged_images": sum(1 for r in c_pass if r["is_dirty_or_damaged"]),
        }

    summary = {
        "task": "TASK 8 — LARGE-SCALE BALANCED 10-CLASS DATASET (Wave 1 Multi-Source Acquisition & QA)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "approved_for_train": False,
        "total_candidates_evaluated": len(records),
        "verdict_counts": {
            "QA_PASS": len(qa_pass_recs),
            "NEEDS_REVIEW": len(rev_recs),
            "REJECT": len(rej_recs),
        },
        "split_counts_qa_pass": {
            "TRAIN_WAVE1": len(train_recs),
            "VAL_REAL_WAVE1": len(val_recs),
            "hard_negatives_qa_pass": sum(1 for r in qa_pass_recs if r["is_hard_negative"]),
            "multi_class_qa_pass": sum(1 for r in qa_pass_recs if r["is_multi_class"]),
        },
        "source_family_breakdown_qa_pass": {
            "Open Images V7": sum(1 for r in qa_pass_recs if r["source_family"] == "Open Images V7"),
            "Wikimedia Commons": sum(1 for r in qa_pass_recs if r["source_family"] == "Wikimedia Commons"),
        },
        "leakage_audit": {
            "external_sha256_collisions": external_sha_hits,
            "external_phash_le8_collisions": external_phash_hits,
            "cross_split_shared_specimens": shared_specimens,
            "cross_split_phash_le8_pairs": cross_split_phash_hits,
            "zero_external_leakage": len(external_sha_hits) == 0 and len(external_phash_hits) == 0,
            "zero_cross_split_leakage": len(shared_specimens) == 0 and len(cross_split_phash_hits) == 0,
        },
        "wave1_per_class_qa_pass": wave1_per_class,
        "contact_sheets": sheet_files,
        "records": records,
    }

    QA_MANIFEST_P.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also write flat CSV manifest for easy inspection
    csv_cols = [
        "short_key", "filename", "source_family", "source_name", "license",
        "qa_verdict", "proposed_split", "specimen_group", "present_class_names",
        "box_count", "is_multi_class", "is_outdoor", "is_complex_background",
        "is_dirty_or_damaged", "is_hard_negative", "sha256", "phash_0",
        "approved_for_train", "qa_note",
    ]
    with open(QA_CSV_P, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k) for k in csv_cols}
            row["present_class_names"] = ";".join(r["present_class_names"])
            writer.writerow(row)

    print("=== TASK 8 WAVE 1 QA & STANDARDIZATION COMPLETE ===")
    print(f"Total evaluated: {len(records)} | QA_PASS: {len(qa_pass_recs)} (TRAIN_WAVE1={len(train_recs)}, VAL_REAL_WAVE1={len(val_recs)}) | NEEDS_REVIEW: {len(rev_recs)} | REJECT: {len(rej_recs)}")
    print("Source family QA_PASS:", summary["source_family_breakdown_qa_pass"])
    print("Leakage audit:", json.dumps(summary["leakage_audit"], indent=2))
    print("Per-class Wave 1 QA_PASS:", json.dumps(wave1_per_class, indent=2))


if __name__ == "__main__":
    main()

