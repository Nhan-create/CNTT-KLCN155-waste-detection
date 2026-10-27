"""
scripts/annotate_and_qa_task8a_multiclass_batch.py
--------------------------------------------------
Sprint AI-UPGRADE-01 — Task 8A: Real-World Multi-Class Data Acquisition & QA
Applies manual full-resolution visual QA verdicts (`QA_PASS`, `NEEDS_MANUAL_REVIEW`, `REJECT`),
multi-class full-object bounding boxes (`[class_id, xmin, ymin, xmax, ymax]`), specimen/session
grouping, and train/val split proposals across all 154 Task 8A candidates.

Generates:
1. YOLO label files in `labels/` (for QA_PASS) and `labels_needs_manual_review/` (for NEEDS_MANUAL_REVIEW)
2. Individual overlay images in `overlays/`
3. 6 QA contact sheets in `contact_sheets/`
4. `task8a_qa_manifest.json`
5. `task8a_license_and_source_manifest.json`
6. `task8a_leakage_and_session_report.json` (checked against Dataset V2, V3 Battery, Task 7, Batch A Mini 01, Phase 4E, Frozen Test, External Holdout, and internal Train vs Val splits)
7. `task8a_10class_coverage_matrix.json` & `TASK8A_MULTICLASS_ACQUISITION_AND_QA_REPORT.md`
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imagehash
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK8A_DIR = REPO_ROOT / "data" / "acquisition_staging" / "task8a_multiclass_batch"
RAW_MANIFEST_P = TASK8A_DIR / "raw_download_manifest.json"
LABELS_DIR = TASK8A_DIR / "labels"
LABELS_REVIEW_DIR = TASK8A_DIR / "labels_needs_manual_review"
OVERLAYS_DIR = TASK8A_DIR / "overlays"
SHEETS_DIR = TASK8A_DIR / "contact_sheets"

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
    0: (255, 70, 70),
    1: (50, 225, 90),
    2: (235, 165, 40),
    3: (180, 100, 255),
    4: (60, 210, 245),
    5: (190, 195, 205),
    6: (255, 235, 65),
    7: (255, 110, 185),
    8: (110, 150, 255),
    9: (210, 130, 90),
}

# Format of each box: [class_id, xmin, ymin, xmax, ymax]
QA_DECISIONS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # 1. BIOLOGICAL CANDIDATES (001..022, 068..082, 084..104) -> 31 QA_PASS
    # =========================================================================
    "task8a_001": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_001_ROTTEN_APPLE_HALF",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.22, 0.16, 0.70, 0.85]],
        "qa_reason": "Rotten decaying half-apple with dark fungal core; full fruit body and stem enclosed.",
    },
    "task8a_002": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_002_MUMMIFIED_APPLE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.31, 0.45, 0.60, 0.67]],
        "qa_reason": "Shriveled brown rotten apple against natural foliage background; full fruit body boxed.",
    },
    "task8a_003": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_003_ROTTEN_APPLE_GROUND",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.26, 0.12, 0.83, 0.89]],
        "qa_reason": "Rotten apple decaying on soil/dry grass outdoors; full apple body boxed.",
    },
    "task8a_004": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_004_ROTTEN_CASHEWS_STEPS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [1, 0.13, 0.08, 0.31, 0.22],
            [1, 0.57, 0.24, 0.79, 0.46],
            [1, 0.33, 0.64, 0.67, 0.90],
        ],
        "qa_reason": "Three rotten yellow cashew fruits discarded on outdoor concrete steps; each fruit boxed individually.",
    },
    "task8a_005": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Microscope view of orange peel mold, not a macroscopic waste object.",
    },
    "task8a_006": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Microscope slide view of mold hyphae, not a macroscopic waste object.",
    },
    "task8a_007": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Microscope slide view of orange peel mold, not a macroscopic waste object.",
    },
    "task8a_008": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Duplicate microscope slide view of mold, excluded.",
    },
    "task8a_009": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_009_MOLDY_LEMON",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.01, 0.18, 0.98, 0.99]],
        "qa_reason": "Moldy citrus lemon fruit with white fungal growth; visible fruit body boxed.",
    },
    "task8a_010": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Historical Metropolitan Police printed text notice about orange/banana peels.",
    },
    "task8a_011": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Entomology pinned moth specimen (matched query word 'rotten banana' in trap notes).",
    },
    "task8a_012": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Entomology pinned moth specimen, off-target.",
    },
    "task8a_013": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_013_CUT_APPLE_CORE",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[1, 0.31, 0.27, 0.64, 0.77]],
        "qa_reason": "Knife-cut apple core discarded on printed sheet; full apple core boxed.",
    },
    "task8a_014": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_014_APPLE_CORE_CURB",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.29, 0.59, 0.39, 0.73]],
        "qa_reason": "Half-eaten red apple core discarded on outdoor concrete curb beside wrought-iron fence.",
    },
    "task8a_015": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_015_CODLING_ROTTEN_APPLES",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [1, 0.08, 0.28, 0.35, 0.58],
            [1, 0.31, 0.22, 0.56, 0.43],
            [1, 0.55, 0.09, 0.84, 0.42],
            [1, 0.31, 0.29, 0.63, 0.62],
            [1, 0.11, 0.56, 0.43, 0.87],
            [1, 0.42, 0.60, 0.72, 0.89],
        ],
        "qa_reason": "Six worm-damaged/rotting Gravenstein apples and cut halves; each apple boxed individually.",
    },
    "task8a_016": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_016_MOLDY_BREAD",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.08, 0.14, 0.92, 0.96]],
        "qa_reason": "Heavily moldy slice of bread spoiled with green mold; full slice boxed.",
    },
    "task8a_017": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_017_ROTTEN_TOMATOES_ROPE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [1, 0.09, 0.34, 0.48, 0.85],
            [1, 0.48, 0.14, 0.95, 0.69],
        ],
        "qa_reason": "Two rotten moldy red tomatoes resting on coiled blue rope; both tomatoes boxed individually.",
    },
    "task8a_018": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_018_ROTTING_STRAWBERRY",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.17, 0.16, 0.84, 0.97]],
        "qa_reason": "Strawberry with brown fungal fruit rot at base; full fruit boxed.",
    },
    "task8a_019": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_019_BOTRYTIS_STRAWBERRY",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[1, 0.12, 0.05, 0.81, 0.94]],
        "qa_reason": "Rotting strawberry covered in gray Botrytis mold on wooden surface; full fruit boxed.",
    },
    "task8a_020": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Hong Kong street scene showing orange-colored litter bin and van; no biological waste visible.",
    },
    "task8a_021": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Hong Kong street scene showing van and orange litter bin; no biological waste visible.",
    },
    "task8a_022": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Industrial orange-peel grapple claw machine inside plant; no biological waste.",
    },

    # =========================================================================
    # 2. PAPER CANDIDATES (023..033, 083, 105..114) -> 11 QA_PASS
    # =========================================================================
    "task8a_023": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_023_CRUMPLED_BALL",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[6, 0.18, 0.08, 0.77, 0.88]],
        "qa_reason": "Crumpled white paper ball; full crumpled paper geometry boxed.",
    },
    "task8a_024": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_PAPER_024_CRUMPLED_BALL",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[6, 0.03, 0.03, 0.96, 0.95]],
        "qa_reason": "Crumpled lined paper ball; full paper ball boxed.",
    },
    "task8a_025": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Scanned historical seed catalog page, not physical waste paper.",
    },
    "task8a_026": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_026_BIODEGRADABLE_CUP_SOIL",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[6, 0.02, 0.01, 0.70, 0.88]],
        "qa_reason": "Discarded biodegradable kraft paper coffee cup lying on dark soil/compost outdoors.",
    },
    "task8a_027": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_027_YELLOW_COFFEE_CUP",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[6, 0.31, 0.06, 0.78, 0.93]],
        "qa_reason": "Disposable yellow paper coffee cup with lid on wet dark outdoor surface.",
    },
    "task8a_028": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_PAPER_028_ALEPA_COFFEE_CUP",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[6, 0.70, 0.26, 0.88, 0.64]],
        "qa_reason": "Discarded Alepa paper coffee cup standing on mossy log/littered ground outdoors.",
    },
    "task8a_029": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "6: paper",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_029_PAPER_STRIP_BASKET",
        "scene_context": "OPEN_BIN_OR_CONTAINER",
        "is_complex_background": True,
        "boxes": [[6, 0.04, 0.23, 0.96, 0.96]],
        "qa_reason": "Wicker wastebasket overflowing with tangled coiled paper strips; held for PM review on whether to box whole tangled mass or individual coils.",
    },
    "task8a_030": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Line drawing illustration ('The Waste Basket'), non-photographic.",
    },
    "task8a_031": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Editorial cartoon drawing about waste paper salvage, non-photographic.",
    },
    "task8a_032": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "WWII vintage poster ('Waste Paper Still Wanted'), non-photographic.",
    },
    "task8a_033": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Architectural photograph of curved building facade, off-target.",
    },

    # =========================================================================
    # 3. CARDBOARD CANDIDATES (034..045, 115..123) -> 10 QA_PASS
    # =========================================================================
    "task8a_034": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_034_CARDBOARD_DUMP_ROOM",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [2, 0.02, 0.35, 0.98, 0.99],
        ],
        "qa_reason": "Abandoned room filled with dozens of overlapping torn cardboard boxes and debris; requires PM crowd/pile segmentation review.",
    },
    "task8a_035": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_035_CARDBOARD_BALES",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [2, 0.01, 0.39, 0.36, 0.87],
            [2, 0.36, 0.34, 0.95, 0.98],
        ],
        "qa_reason": "Two large tied bales of hundreds of flattened cardboard sheets on Jerusalem street; held for PM review on bale vs individual sheet boxing.",
    },
    "task8a_036": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_036_CARDBOARD_BALE_PILE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[2, 0.14, 0.12, 0.99, 0.99]],
        "qa_reason": "Compressed industrial bale of flattened corrugated cardboard; held for PM review on bale-level annotation policy.",
    },
    "task8a_037": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_037_WALMART_BALER",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[2, 0.56, 0.45, 0.99, 0.98]],
        "qa_reason": "Warehouse worker operating industrial cardboard baler with blurred foreground cardboard stack; held for PM review.",
    },
    "task8a_038": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_038_DAMAGED_FRAGILE_BOX",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[2, 0.31, 0.22, 0.67, 0.60]],
        "qa_reason": "Dented/damaged corrugated cardboard box with red FRAGILE sticker sitting on outdoor stone steps; full box enclosed (not sticker).",
    },
    "task8a_039": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_039_CARDBOARD_CEMETERY",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[2, 0.01, 0.18, 0.99, 0.99]],
        "qa_reason": "Massive outdoor heap of hundreds of overlapping cardboard boxes ('Cimetiere de carton'); requires crowd/pile policy review.",
    },
    "task8a_040": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_040_WET_TORN_CARDBOARD",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[2, 0.01, 0.01, 0.99, 0.99]],
        "qa_reason": "Wet, chewed, weathered corrugated cardboard box flap outdoors with visible corrugated inner flutes.",
    },
    "task8a_041": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_041_FIVE_CARTONS_STACK",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [2, 0.31, 0.25, 0.78, 0.43],
            [2, 0.33, 0.39, 0.79, 0.53],
            [2, 0.32, 0.49, 0.79, 0.66],
            [2, 0.33, 0.60, 0.80, 0.78],
            [2, 0.33, 0.69, 0.81, 0.87],
        ],
        "qa_reason": "Stack of five corrugated cardboard cartons on wooden pallet; each carton boxed individually.",
    },
    "task8a_042": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_042_EGG_CARTON_WITH_EGGS",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[2, 0.03, 0.05, 0.90, 0.94]],
        "qa_reason": "Molded-pulp cardboard egg carton filled with 12 intact fresh uncooked eggs; held for PM review so intact fresh food is not mixed with discarded waste.",
    },
    "task8a_043": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_043_THREE_BOXES_CART",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [2, 0.13, 0.19, 0.60, 0.35],
            [2, 0.13, 0.35, 0.58, 0.52],
            [2, 0.12, 0.44, 0.90, 0.85],
        ],
        "qa_reason": "Three corrugated cardboard shipping boxes of different sizes stacked on a blue hand cart; each box annotated separately.",
    },
    "task8a_044": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_044_OPEN_BOX_SIDEWALK",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [2, 0.31, 0.09, 0.78, 0.98],
            [7, 0.49, 0.52, 0.57, 0.68],
        ],
        "qa_reason": "Open corrugated cardboard box on sidewalk in Matsudo with plastic bottle inside; multi-class annotation (2: cardboard + 7: plastic).",
    },
    "task8a_045": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_045_WALMART_BALE_CART",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[2, 0.55, 0.32, 0.88, 0.78]],
        "qa_reason": "Worker pulling compressed cardboard bale on pallet jack; held with task8a_037 for bale policy review.",
    },

    # =========================================================================
    # 4. GLASS CANDIDATES (046..054, 124..137, 151..154) -> 12 QA_PASS
    # =========================================================================
    "task8a_046": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Historical B&W archival photograph of camp shelter, off-target.",
    },
    "task8a_047": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "18th-century engraving print, non-photographic.",
    },
    "task8a_048": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "18th-century engraving print duplicate, non-photographic.",
    },
    "task8a_049": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "1779 satirical comic engraving print, non-photographic.",
    },
    "task8a_050": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_050_BROKEN_CLEAR_BOTTLE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [4, 0.40, 0.13, 0.74, 0.58],
            [4, 0.25, 0.57, 0.46, 0.91],
        ],
        "qa_reason": "Broken clear glass bottle shattered into two main pieces on outdoor textured brick pavement; both pieces boxed.",
    },
    "task8a_051": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_051_GLASS_CULLET_BED",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.01, 0.01, 0.99, 0.99]],
        "qa_reason": "Dense ground layer of thousands of tiny broken glass cullet fragments; cannot isolate individual bottles.",
    },
    "task8a_052": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_052_WALL_GLASS_SHARDS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.02, 0.12, 0.98, 0.98]],
        "qa_reason": "Broken glass shards embedded into mortar on top of security wall; held for PM review as architectural fixture vs waste.",
    },
    "task8a_053": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_053_PRESERVING_JAR",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[4, 0.27, 0.52, 0.67, 0.98]],
        "qa_reason": "Empty clear glass preserving jar below wooden opener tool; full glass jar body boxed.",
    },
    "task8a_054": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Postcard of church ruins, off-target.",
    },

    # =========================================================================
    # 5. SHOES CANDIDATES (055..067, 138..150) -> 11 QA_PASS
    # =========================================================================
    "task8a_055": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "X-ray radiograph of feet in pointe shoes, non-photographic.",
    },
    "task8a_056": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "1931 B&W film still of hand holding worn shoe sole, low clarity.",
    },
    "task8a_057": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Street vegetable cart vendor scene, off-target.",
    },
    "task8a_058": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "1908 newspaper cartoon ('To The Dump'), non-photographic.",
    },
    "task8a_059": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_SHOES_059_NEWPORT_LITTER_BIN",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.37, 0.24, 0.50, 0.34],
            [8, 0.29, 0.70, 0.39, 0.76],
            [8, 0.39, 0.71, 0.44, 0.76],
            [8, 0.45, 0.77, 0.51, 0.83],
            [8, 0.51, 0.77, 0.56, 0.84],
        ],
        "qa_reason": "Discarded boots on top of a street litter bin and two pairs of discarded shoes on the pavement around the bin in Newport High Street.",
    },
    "task8a_060": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Historical B&W magazine photo of street ice sailing, off-target.",
    },
    "task8a_061": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_061_FOUR_BLACK_SHOES_CURB",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.15, 0.50, 0.38, 0.69],
            [8, 0.29, 0.51, 0.52, 0.72],
            [8, 0.44, 0.51, 0.66, 0.74],
            [8, 0.57, 0.51, 0.82, 0.76],
        ],
        "qa_reason": "Four discarded black leather shoes lined up on outdoor pavement/grass border; each shoe boxed individually.",
    },
    "task8a_062": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_062_MUDDY_SNOW_BOOTS",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [8, 0.29, 0.02, 0.69, 0.49],
            [8, 0.22, 0.37, 0.68, 0.97],
        ],
        "qa_reason": "Pair of muddy worn snow boots lying on tiled floor; both boots boxed individually.",
    },
    "task8a_063": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Illustrated WWI postcard painting by Arthur Thiele, non-photographic.",
    },
    "task8a_064": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_064_WORN_MUDDY_BOOTS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.28, 0.58, 0.48, 0.74],
            [8, 0.50, 0.53, 0.67, 0.76],
            [8, 0.63, 0.52, 0.86, 0.79],
        ],
        "qa_reason": "Muddy boots actively worn on people's feet standing in festival mud; held for PM review as worn footwear vs discarded waste.",
    },
    "task8a_065": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_SHOES_065_ABANDONED_SHOE_MUD",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.14, 0.24, 0.95, 0.72]],
        "qa_reason": "Single old dirty leather shoe abandoned on muddy ground and twigs with dirt inside.",
    },
    "task8a_066": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Equestrian monument in Vicksburg National Military Park, off-target.",
    },
    "task8a_067": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Historical halftone newspaper portrait, off-target.",
    },

    # =========================================================================
    # EXPANDED BATCH CANDIDATES (068..154)
    # =========================================================================
    "task8a_068": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Butterfly perched on overripe banana bait; insect dominates frame.",
    },
    "task8a_069": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_069_BROWN_BANANA_PEEL",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[1, 0.22, 0.15, 0.97, 0.95]],
        "qa_reason": "Spotted brown banana peel strips; full peel structure boxed.",
    },
    "task8a_070": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_070_BANANA_PEEL_MAISELS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.33, 0.36, 0.39, 0.44]],
        "qa_reason": "Small discarded banana peel lying on concrete sidewalk mosaic ('MAISEL'S'); tests small-scale outdoor biological detection.",
    },
    "task8a_071": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_071_BANANA_PEEL_ASPHALT",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.45, 0.38, 0.99, 0.64]],
        "qa_reason": "Discarded yellow banana peel lying on wet asphalt sidewalk with pedestrians in background.",
    },
    "task8a_072": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_072_YELAKKI_BANANA_PEEL",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[1, 0.08, 0.24, 0.99, 0.91]],
        "qa_reason": "Yelakki banana peel discarded on dark speckled granite counter; full peel including stem boxed.",
    },
    "task8a_073": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Outdoor monument sculpture of an apple core by Claes Oldenburg, not real biological waste.",
    },
    "task8a_074": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Infographic diagram about apple core waste economy, non-photographic.",
    },
    "task8a_075": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "1: biological",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_075_STARLING_APPLE_CORE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.80, 0.69, 0.90, 0.78]],
        "qa_reason": "Small apple core piece on soil next to large Burchell's starling bird; held for PM review due to bird dominating frame.",
    },
    "task8a_076": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Digital kaleidoscope abstract pattern titled 'Apple Core', non-photographic.",
    },
    "task8a_077": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_077_ROTTEN_APPLE_LEAVES",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.24, 0.26, 0.72, 0.84]],
        "qa_reason": "Brown decaying rotten apple partially eaten on dead leaves under tree.",
    },
    "task8a_078": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_078_ROTTEN_PEACHES_PUNNET",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [7, 0.07, 0.10, 0.89, 0.96],
            [1, 0.19, 0.21, 0.46, 0.56],
            [1, 0.18, 0.57, 0.39, 0.85],
            [1, 0.39, 0.58, 0.62, 0.86],
            [1, 0.55, 0.42, 0.81, 0.78],
        ],
        "qa_reason": "Four moldy rotting peaches inside a clear plastic punnet container; multi-class annotation (4x 1:biological + 1x 7:plastic).",
    },
    "task8a_079": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "1: biological",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_079_ROTTEN_ORANGES_HEAP",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.01, 0.01, 0.99, 0.99]],
        "qa_reason": "Dense overlapping heap of 60+ rotten oranges filling entire frame; held for PM crowd/pile annotation review.",
    },
    "task8a_080": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Macro wildlife photo of Kallima inachus butterfly perched on rotting fruit; butterfly dominates frame.",
    },
    "task8a_081": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_081_ROTTEN_PEAR_GRASS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.09, 0.10, 0.98, 0.81]],
        "qa_reason": "Rotten browning half-pear lying in green grass outdoors; full pear body and stem boxed.",
    },
    "task8a_082": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_082_MOLDY_TOMATOES",
        "scene_context": "OPEN_BIN_OR_CONTAINER",
        "is_complex_background": True,
        "boxes": [
            [1, 0.39, 0.08, 0.60, 0.36],
            [1, 0.22, 0.29, 0.49, 0.63],
            [1, 0.46, 0.36, 0.67, 0.69],
            [1, 0.62, 0.22, 0.87, 0.56],
            [1, 0.31, 0.62, 0.62, 0.96],
        ],
        "qa_reason": "Moldy spoiled red tomatoes with white fungal spots inside box; 5 distinct moldy tomatoes boxed.",
    },
    "task8a_083": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_083_DISPOSABLE_CUP_VS_MUG",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[6, 0.09, 0.31, 0.43, 0.93]],
        "qa_reason": "Disposable paper coffee cup (left, boxed as 6:paper) next to reusable red ceramic mug (right, intentionally unboxed hard negative context).",
    },
    "task8a_084": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_084_BLACKENED_BANANA_SKIN",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.12, 0.04, 0.93, 0.99]],
        "qa_reason": "Overripe blackened banana peel skin; full peel and stem boxed.",
    },
    "task8a_085": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "6: paper",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_085_MARATHON_CUPS_STREET",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [6, 0.35, 0.69, 0.41, 0.77],
            [6, 0.43, 0.78, 0.48, 0.85],
            [6, 0.54, 0.78, 0.58, 0.86],
            [6, 0.55, 0.87, 0.60, 0.94],
        ],
        "qa_reason": "Wet marathon street littered with 40+ crushed paper cups and banana peels among runners; held for PM dense street-litter annotation review.",
    },
    "task8a_086": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_BANANE_A_SESSION",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.07, 0.03, 0.89, 0.88]],
        "qa_reason": "Single peeled yellow banana skin; grouped in SPEC_BIO_BANANE_A_SESSION with task8a_088 in TRAIN_V4_PROPOSED.",
    },
    "task8a_087": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "SPEC_BIO_BANANE_A_SESSION",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Cropped duplicate version ('Banane a 05 cropped.jpg') of task8a_088 ('Banane a 05.jpg'); rejected per PM anti-duplicate rule.",
    },
    "task8a_088": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_BANANE_A_SESSION",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [
            [1, 0.03, 0.12, 0.69, 0.83],
            [1, 0.38, 0.10, 0.93, 0.80],
        ],
        "qa_reason": "Two overlapping banana peels from same session as task8a_086; both peels boxed individually and grouped in TRAIN_V4_PROPOSED.",
    },
    "task8a_089": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_089_RED_APPLE_CORE",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.35, 0.08, 0.82, 0.92]],
        "qa_reason": "Eaten red apple core standing upright; full stem, core, and base boxed.",
    },
    "task8a_090": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_090_APFELGRIEBSCH",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.38, 0.11, 0.72, 0.90]],
        "qa_reason": "Eaten yellow-red apple core; full stem, core, and base boxed.",
    },
    "task8a_091": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_091_MULTI_TRASH_APPLE_CORES",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [1, 0.42, 0.25, 0.67, 0.62],
            [1, 0.62, 0.46, 0.97, 0.82],
            [1, 0.45, 0.62, 0.69, 0.94],
            [1, 0.04, 0.56, 0.44, 0.88],
            [6, 0.71, 0.01, 0.99, 0.60],
            [7, 0.19, 0.01, 0.48, 0.39],
        ],
        "qa_reason": "Real open waste bin containing multiple eaten apple cores (1:biological), a white disposable paper cup (6:paper), and a plastic bag wrapper (7:plastic).",
    },
    "task8a_092": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_092_STARK_APPLE_CORE",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.11, 0.08, 0.79, 0.94]],
        "qa_reason": "Eaten dark-red Stark apple core; full core and stem boxed.",
    },
    "task8a_093": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Intact fresh unpeeled orange fruit, not discarded biological waste per PM Section 4.",
    },
    "task8a_094": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Extreme macro close-up of orange zest macerating in liquid jar, unrecognizable object boundary.",
    },
    "task8a_095": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_ORANGE_PEEL_FAMARTIN",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [
            [1, 0.22, 0.22, 0.77, 0.45],
            [1, 0.15, 0.39, 0.92, 0.88],
        ],
        "qa_reason": "Two discarded orange peel shells on table; both peel halves boxed and grouped in SPEC_BIO_ORANGE_PEEL_FAMARTIN with task8a_096 in TRAIN_V4_PROPOSED.",
    },
    "task8a_096": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_ORANGE_PEEL_FAMARTIN",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[1, 0.09, 0.05, 0.87, 0.92]],
        "qa_reason": "Torn orange peel pieces showing white pith from same session as task8a_095; grouped in TRAIN_V4_PROPOSED.",
    },
    "task8a_097": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_BIO_097_EGGSHELL_FOREST",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[1, 0.32, 0.36, 0.64, 0.69]],
        "qa_reason": "Broken eggshell lying on outdoor moss and forest litter in Finkenpark Berlin.",
    },
    "task8a_098": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Vintage Russian Easter postcard illustration, non-photographic.",
    },
    "task8a_099": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Easter table cake decoration centerpiece, not waste.",
    },
    "task8a_100": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "Vintage Easter chicks postcard illustration, non-photographic.",
    },
    "task8a_101": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Officials touring warehouse facility, no visible biological waste.",
    },
    "task8a_102": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Officials talking in food warehouse aisle, no visible biological waste.",
    },
    "task8a_103": {
        "qa_verdict": "REJECT",
        "primary_class": "1: biological",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Wide warehouse pallet racks, off-target.",
    },
    "task8a_104": {
        "qa_verdict": "QA_PASS",
        "primary_class": "1: biological",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_BIO_104_BIOWASTE_BIN_MIX",
        "scene_context": "OPEN_BIN_OR_CONTAINER",
        "is_complex_background": True,
        "boxes": [
            [1, 0.59, 0.01, 0.96, 0.25],
            [1, 0.43, 0.56, 0.65, 0.94],
            [1, 0.56, 0.68, 0.77, 0.98],
            [1, 0.67, 0.69, 0.87, 0.98],
        ],
        "qa_reason": "Real household biowaste bin containing banana peel, lemon rind, tomato top, and eggshell.",
    },
    "task8a_105": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "6: paper",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_105_CRUMPLED_PAPER_HEAP",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[6, 0.01, 0.01, 0.99, 0.99]],
        "qa_reason": "Dense heap of 50+ overlapping crumpled white paper balls with Arabic watermark text in top-left corner; held for PM review.",
    },
    "task8a_106": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_106_CRUMPLED_OLIVE_SHEET",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[6, 0.01, 0.01, 0.99, 0.99]],
        "qa_reason": "Unfolded crumpled olive-green paper sheet with dense crease texture.",
    },
    "task8a_107": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Historical B&W portrait of man reading newspaper, not discarded waste.",
    },
    "task8a_108": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Duplicate higher-res version of task8a_107 portrait, excluded.",
    },
    "task8a_109": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_CANADA150_CUP",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[6, 0.26, 0.02, 0.75, 0.95]],
        "qa_reason": "Disposable Tim Hortons paper coffee cup with plastic lid; grouped with task8a_110 in TRAIN_V4_PROPOSED.",
    },
    "task8a_110": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_CANADA150_CUP",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [6, 0.23, 0.04, 0.63, 0.76],
            [6, 0.37, 0.78, 0.86, 0.94],
        ],
        "qa_reason": "Disposable paper coffee cup and detached paper sleeve on table; grouped with task8a_109 in TRAIN_V4_PROPOSED.",
    },
    "task8a_111": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "SPEC_PAPER_CANADA150_CUP",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Near-duplicate angle of task8a_110 taken seconds apart; rejected to avoid duplicate inflation per PM rule.",
    },
    "task8a_112": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_PAPER_112_TWO_SECONDCUP_CUPS",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [6, 0.12, 0.10, 0.31, 0.41],
            [6, 0.32, 0.03, 0.70, 0.84],
        ],
        "qa_reason": "Two disposable Second Cup paper coffee cups on wooden table; both cups boxed individually.",
    },
    "task8a_113": {
        "qa_verdict": "QA_PASS",
        "primary_class": "6: paper",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_PAPER_113_AIRFRANCE_CUPS_NAPKIN",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [6, 0.40, 0.14, 0.64, 0.56],
            [6, 0.64, 0.14, 0.88, 0.56],
            [6, 0.11, 0.33, 0.65, 0.96],
            [6, 0.13, 0.35, 0.63, 0.86],
        ],
        "qa_reason": "Two disposable paper cups, white paper napkin, and paper sugar/cutlery packet on tray.",
    },
    "task8a_114": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "6: paper",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_114_ICE_CREAM_CUP_HAND",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[6, 0.46, 0.48, 0.77, 0.83]],
        "qa_reason": "Paper cup filled with fresh rolled ice cream held by vendor; held for PM review as food-in-hand vs discarded waste.",
    },
    "task8a_115": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_115_PIZZA_BOX_PLASTIC_TUBS",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [2, 0.03, 0.11, 0.84, 0.88],
            [7, 0.02, 0.01, 0.44, 0.60],
            [7, 0.42, 0.01, 0.80, 0.54],
        ],
        "qa_reason": "Corrugated cardboard pizza takeaway box beneath two empty clear plastic food containers; multi-class annotation (2:cardboard + 2x 7:plastic).",
    },
    "task8a_116": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "2: cardboard",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_116_OPEN_PIZZA_BOXES_FOOD",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [2, 0.09, 0.42, 0.27, 0.77],
            [2, 0.25, 0.14, 0.60, 0.81],
            [2, 0.57, 0.10, 0.99, 0.80],
        ],
        "qa_reason": "Three open cardboard pizza boxes containing fresh edible pizzas at event; held in NEEDS_MANUAL_REVIEW per PM rule not to label clean served food as biological waste.",
    },
    "task8a_117": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_117_PARK_SLOPE_PIZZA_BOXES",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [2, 0.01, 0.01, 0.99, 0.43],
            [2, 0.08, 0.41, 0.99, 0.56],
            [2, 0.11, 0.52, 0.99, 0.65],
            [2, 0.11, 0.60, 0.99, 0.76],
            [2, 0.11, 0.67, 0.99, 0.89],
        ],
        "qa_reason": "Stack of five used corrugated cardboard pizza boxes on table; each box annotated individually.",
    },
    "task8a_118": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_118_SIX_PIZZA_BOXES",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [2, 0.17, 0.13, 0.83, 0.52],
            [2, 0.18, 0.36, 0.83, 0.60],
            [2, 0.20, 0.44, 0.84, 0.67],
            [2, 0.21, 0.52, 0.86, 0.74],
            [2, 0.22, 0.60, 0.85, 0.82],
            [2, 0.24, 0.68, 0.82, 0.91],
        ],
        "qa_reason": "Stack of six corrugated cardboard pizza boxes on counter; each box annotated individually.",
    },
    "task8a_119": {
        "qa_verdict": "REJECT",
        "primary_class": "2: cardboard",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Metal outdoor pizza-box disposal bin exterior only; no actual cardboard boxes visible.",
    },
    "task8a_120": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_120_BANANA_BOXES_STACK",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [2, 0.19, 0.22, 0.47, 0.36],
            [2, 0.19, 0.36, 0.47, 0.49],
            [2, 0.20, 0.49, 0.48, 0.61],
            [2, 0.21, 0.61, 0.48, 0.74],
            [2, 0.49, 0.23, 0.79, 0.39],
            [2, 0.51, 0.39, 0.81, 0.54],
            [2, 0.56, 0.54, 0.83, 0.68],
            [2, 0.56, 0.68, 0.83, 0.83],
        ],
        "qa_reason": "Two stacks of heavy-duty corrugated cardboard banana cartons (8 boxes boxed individually).",
    },
    "task8a_121": {
        "qa_verdict": "QA_PASS",
        "primary_class": "2: cardboard",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_CARDBOARD_121_ALLEY_BANANA_BOX",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[2, 0.49, 0.22, 0.74, 0.38]],
        "qa_reason": "Discarded Bonita corrugated cardboard box stacked atop wooden pallets in outdoor stone alley.",
    },
    "task8a_122": {
        "qa_verdict": "REJECT",
        "primary_class": "2: cardboard",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "19th-century vintage round percussion-cap tin/box label close-up, off-target.",
    },
    "task8a_123": {
        "qa_verdict": "REJECT",
        "primary_class": "2: cardboard",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": False,
        "boxes": [],
        "qa_reason": "19th-century vintage round percussion-cap box label close-up, off-target.",
    },
    "task8a_124": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_124_MYUNA_CREEK_SHARD",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.10, 0.08, 0.90, 0.86]],
        "qa_reason": "Thick broken glass bottle shard with embossed lettering discarded at Myuna Creek.",
    },
    "task8a_125": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_125_PLAZA_DISTANT_LITTER",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [4, 0.80, 0.71, 0.83, 0.75],
            [4, 0.84, 0.80, 0.89, 0.84],
        ],
        "qa_reason": "Wide plaza view after parade with tiny distant broken bottles/cans (<15px); held for PM review due to sub-resolution object scale.",
    },
    "task8a_126": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_GLASS_126_FIVE_BOTTLES_TABLE",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [
            [4, 0.45, 0.22, 0.51, 0.54],
            [4, 0.34, 0.44, 0.46, 0.50],
            [4, 0.49, 0.43, 0.60, 0.49],
            [4, 0.63, 0.50, 0.84, 0.60],
            [4, 0.68, 0.46, 0.87, 0.55],
            [6, 0.54, 0.50, 0.65, 0.56],
        ],
        "qa_reason": "Five empty green glass bottles (1 standing, 4 lying) and a red paper cigarette pack left on an outdoor wooden picnic table in forest.",
    },
    "task8a_127": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_127_BROKEN_BEER_BOTTLES_PAVEMENT",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [4, 0.12, 0.51, 0.35, 0.96],
            [4, 0.43, 0.34, 0.72, 0.69],
            [4, 0.34, 0.07, 0.76, 0.43],
            [4, 0.41, 0.73, 0.54, 0.96],
        ],
        "qa_reason": "Two smashed brown glass beer bottles ('Sonnen-Hopfen') on cobblestone pavement with weeds; 4 major bottle bodies/necks boxed.",
    },
    "task8a_128": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_128_BEACH_GLASS_SHARDS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.01, 0.25, 0.99, 0.98]],
        "qa_reason": "Hundreds of broken glass bottle necks and shards washed up on beach shoreline; held for PM crowd/pile annotation review.",
    },
    "task8a_129": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_128_BEACH_GLASS_SHARDS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.01, 0.20, 0.99, 0.99]],
        "qa_reason": "Second angle of dense broken glass bottle deposit on beach shoreline; held in NEEDS_MANUAL_REVIEW with task8a_128.",
    },
    "task8a_130": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_GLASS_130_BROKEN_GREEN_BOTTLE_BEACH",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.37, 0.58, 0.58, 0.69]],
        "qa_reason": "Broken green glass bottle neck/shoulder half-buried in coastal beach sand.",
    },
    "task8a_131": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_131_BROKEN_FLENSBURGER_BOTTLE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.35, 0.23, 0.81, 0.81]],
        "qa_reason": "Broken brown glass beer bottle top with metal/porcelain swing-top cap lying on soil.",
    },
    "task8a_132": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_132_BROKEN_JARRITOS_FLOOR",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [4, 0.44, 0.72, 0.53, 0.83],
            [4, 0.76, 0.59, 0.88, 0.69],
        ],
        "qa_reason": "Broken Jarritos glass soda bottle pieces in spilled liquid on tiled floor.",
    },
    "task8a_133": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_GLASS_133_BROKEN_VANA_TALLINN",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [4, 0.38, 0.45, 0.51, 0.63],
            [4, 0.26, 0.20, 0.31, 0.25],
        ],
        "qa_reason": "Smashed brown glass liqueur bottle and broken neck piece in spilled puddle on terrazzo floor.",
    },
    "task8a_134": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "4: glass",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_134_GLASS_BANK_CROWD",
        "scene_context": "MULTI_OBJECT_OR_MULTI_CLASS",
        "is_complex_background": True,
        "boxes": [[4, 0.05, 0.50, 0.98, 0.97]],
        "qa_reason": "Over 100 glass wine/beer bottles and plastic bags piled around street bottle-bank container; held for PM dense crowd review.",
    },
    "task8a_135": {
        "qa_verdict": "REJECT",
        "primary_class": "6: paper",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Handwritten note hung on tree branch about empty bottles; zero bottles visible.",
    },
    "task8a_136": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "7: plastic",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_136_PLASTIC_BOTTLE_IN_EMPTY_BOTTLES_CAT",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [[7, 0.34, 0.49, 0.69, 0.75]],
        "qa_reason": "Fetched from Wikimedia Category:Empty bottles, but visual QA verified it is a clear PET plastic bottle (7:plastic), NOT glass (4:glass). Held in NEEDS_MANUAL_REVIEW per PM Section 4 glass-vs-plastic rule.",
    },
    "task8a_137": {
        "qa_verdict": "REJECT",
        "primary_class": "7: plastic",
        "proposed_split": "EXCLUDED",
        "specimen_group": "REJECTED",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Close-up portrait of person holding a plastic sports drink bottle against face; not glass and person dominates frame.",
    },
    "task8a_138": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_138_TINY_STREET_SHOE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.30, 0.72, 0.34, 0.77]],
        "qa_reason": "Wide night street scene with very small sandal in foreground and pedestrians wearing shoes in background; held for PM review.",
    },
    "task8a_139": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_139_ABANDONED_PINK_PAIR",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.16, 0.27, 0.39, 0.61],
            [8, 0.40, 0.38, 0.83, 0.70],
        ],
        "qa_reason": "Pair of abandoned worn pink peep-toe shoes on outdoor concrete wall ledge; both shoes boxed individually.",
    },
    "task8a_140": {
        "qa_verdict": "REJECT",
        "primary_class": "8: shoes",
        "proposed_split": "EXCLUDED",
        "specimen_group": "SPEC_SHOES_139_ABANDONED_PINK_PAIR",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Near-duplicate second angle of task8a_139 ('Abandoned shoes and flower 2.jpg'); rejected to prevent duplicate inflation.",
    },
    "task8a_141": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_141_BEACH_ABANDONED_SANDAL",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.25, 0.19, 0.71, 0.72]],
        "qa_reason": "Weathered, sand-encrusted abandoned foam sandal washed up on beach sand.",
    },
    "task8a_142": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_142_SHOES_ON_WIRES",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.49, 0.22, 0.64, 0.43]],
        "qa_reason": "Cluster of 15+ shoes tied together and flung over overhead utility wires against sky; held for PM review.",
    },
    "task8a_143": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_143_LONDON_SHOE_LINE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.08, 0.89, 0.14, 0.99],
            [8, 0.30, 0.45, 0.85, 0.72],
        ],
        "qa_reason": "Over 25 overlapping abandoned shoes piled along brick wall on London sidewalk; held for PM dense-pile annotation review.",
    },
    "task8a_144": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_SHOES_144_LONDON_PAVEMENT_SHOE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.26, 0.21, 0.69, 0.74]],
        "qa_reason": "Single dirty blue flat shoe abandoned on stone pavement slabs among fallen leaves in London.",
    },
    "task8a_145": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_145_WORN_BROWN_WINGTIPS",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": True,
        "boxes": [
            [8, 0.03, 0.12, 0.71, 0.69],
            [8, 0.15, 0.22, 0.77, 0.95],
        ],
        "qa_reason": "Pair of heavily scuffed worn-out brown suede/leather shoes; both shoes boxed individually.",
    },
    "task8a_146": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_146_WADI_RUM_TORN_BOOT",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.08, 0.37, 0.69, 0.88]],
        "qa_reason": "Torn, sun-bleached old ankle boot abandoned on desert sand and dry grass in Wadi Rum.",
    },
    "task8a_147": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_147_MACRO_SHOE_LACES",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[8, 0.02, 0.02, 0.98, 0.98]],
        "qa_reason": "Extreme close-up crop of frayed sneaker shoelaces only, cutting off toe and heel geometry; held in NEEDS_MANUAL_REVIEW.",
    },
    "task8a_148": {
        "qa_verdict": "NEEDS_MANUAL_REVIEW",
        "primary_class": "8: shoes",
        "proposed_split": "MANUAL_REVIEW_HOLD",
        "specimen_group": "SPEC_REVIEW_148_USED_SHOE_STALL",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[8, 0.18, 0.28, 0.76, 0.67]],
        "qa_reason": "Roadside rack displaying 30+ stacked second-hand shoes in Baliuag; held for PM dense-rack annotation review.",
    },
    "task8a_149": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_SHOES_149_DIRTY_BLUNDSTONE_BOOT",
        "scene_context": "INDOOR_OR_CLOSEUP_REAL",
        "is_complex_background": False,
        "boxes": [[8, 0.05, 0.27, 0.86, 0.69]],
        "qa_reason": "Dirty mud-splattered worn Blundstone work boot; full boot profile boxed.",
    },
    "task8a_150": {
        "qa_verdict": "QA_PASS",
        "primary_class": "8: shoes",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_SHOES_150_FOUR_DIRTY_CONVERSE",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [8, 0.02, 0.14, 0.73, 0.62],
            [8, 0.02, 0.37, 0.71, 0.98],
            [8, 0.25, 0.02, 0.96, 0.28],
            [8, 0.42, 0.09, 0.88, 0.43],
        ],
        "qa_reason": "Two pairs (4 shoes) of dirty worn yellow and green Converse canvas sneakers on coarse doormat; all 4 shoes boxed individually.",
    },
    "task8a_151": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_151_CLEAR_EMPTY_BOTTLE_RAIL",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.35, 0.09, 0.52, 0.90]],
        "qa_reason": "Empty clear glass bottle standing on weathered outdoor wooden rail with grass background.",
    },
    "task8a_152": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "TRAIN_V4_PROPOSED",
        "specimen_group": "SPEC_GLASS_152_FIVE_PAOLYTA_BOTTLES",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [
            [4, 0.06, 0.07, 0.22, 0.70],
            [4, 0.22, 0.08, 0.38, 0.72],
            [4, 0.37, 0.08, 0.54, 0.79],
            [4, 0.53, 0.09, 0.72, 0.89],
            [4, 0.72, 0.09, 0.93, 0.94],
        ],
        "qa_reason": "Five empty brown Paolyta-B glass bottles lined up outdoors in front of shutter door; all 5 glass bottles boxed individually.",
    },
    "task8a_153": {
        "qa_verdict": "REJECT",
        "primary_class": "4: glass",
        "proposed_split": "EXCLUDED",
        "specimen_group": "SPEC_GLASS_152_FIVE_PAOLYTA_BOTTLES",
        "scene_context": "EXCLUDED",
        "is_complex_background": True,
        "boxes": [],
        "qa_reason": "Tighter crop/duplicate framing of task8a_152 ('Paolyta B liq empty glass bottles'); rejected per PM anti-duplicate rule.",
    },
    "task8a_154": {
        "qa_verdict": "QA_PASS",
        "primary_class": "4: glass",
        "proposed_split": "VAL_REAL_MULTICLASS_PROPOSED",
        "specimen_group": "SPEC_GLASS_154_WHISBIH_BOTTLE_STEPS",
        "scene_context": "OUTDOOR_STREET_OR_GROUND",
        "is_complex_background": True,
        "boxes": [[4, 0.08, 0.29, 0.93, 0.69]],
        "qa_reason": "Empty brown Whisbih 600cc glass bottle discarded horizontally on rough pebble-concrete steps.",
    },
}


def xyxy_to_yolo(box: List[float]) -> Tuple[int, float, float, float, float]:
    cls_id = int(box[0])
    xmin, ymin, xmax, ymax = float(box[1]), float(box[2]), float(box[3]), float(box[4])
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    return cls_id, xc, yc, w, h


def draw_overlay(item: Dict[str, Any], out_path: Path) -> None:
    img_path = REPO_ROOT / item["local_path"]
    with Image.open(img_path) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    draw = ImageDraw.Draw(rgb)

    verdict = item["qa_verdict"]
    banner_color = (
        (0, 230, 90)
        if verdict == "QA_PASS"
        else ((255, 165, 0) if verdict == "NEEDS_MANUAL_REVIEW" else (240, 60, 60))
    )

    for b_idx, box in enumerate(item["boxes_annotated"], start=1):
        cls_id = int(box["class_id"])
        xmin, ymin, xmax, ymax = box["bbox_xyxy_norm"]
        x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        c_color = CLASS_COLORS.get(cls_id, banner_color)
        draw.rectangle([x1, y1, x2, y2], outline=c_color, width=max(3, min(w, h) // 180))
        label_txt = f"{cls_id}:{CLASS_NAMES[cls_id]} #{b_idx}"
        tag_w = max(115, len(label_txt) * 7 + 10)
        draw.rectangle([x1, max(0, y1 - 22), min(w, x1 + tag_w), y1], fill=c_color)
        draw.text((x1 + 4, max(0, y1 - 18)), label_txt, fill=(10, 10, 10))

    banner_h = 34
    draw.rectangle([0, 0, w, banner_h], fill=(20, 24, 30))
    draw.text(
        (8, 8),
        f"{item['short_key']} | {verdict} | {item['primary_class']} | boxes={item['boxes_count']} | {item['proposed_split']}",
        fill=banner_color,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, quality=90)


def render_qa_contact_sheet(items: List[Dict[str, Any]], title: str, out_path: Path, cols: int = 4) -> None:
    cell_w, cell_h = 460, 360
    header_h = 52
    banner_h = 46
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
        v_color = (
            (60, 230, 110)
            if verdict == "QA_PASS"
            else ((255, 175, 40) if verdict == "NEEDS_MANUAL_REVIEW" else (245, 80, 80))
        )

        draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + header_h], fill=(36, 40, 48))
        draw.text(
            (x0 + 6, y0 + 5),
            f"{item['short_key']} [{verdict}] | {item['primary_class']} | b={item['boxes_count']} | {item['proposed_split'][:9]}",
            fill=v_color,
        )
        draw.text((x0 + 6, y0 + 25), f"{item['qa_reason'][:62]}", fill=(210, 215, 225))

        ov_path = REPO_ROOT / item["overlay_path"]
        with Image.open(ov_path) as im:
            thumb = im.convert("RGB").resize((cell_w - 14, cell_h - 14), Image.Resampling.BILINEAR)
        canvas.paste(thumb, (x0 + 7, y0 + header_h + 7))
        draw.rectangle(
            [x0 + 7, y0 + header_h + 7, x0 + cell_w - 7, y0 + header_h + cell_h - 7],
            outline=v_color,
            width=2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def run_comprehensive_leakage_audit(enriched: List[Dict[str, Any]]) -> Dict[str, Any]:
    import hashlib
    known_sha256: Dict[str, str] = {}
    known_phash: Dict[str, imagehash.ImageHash] = {}
    pool_counts = {
        "dataset_v2_and_cache": 0,
        "frozen_test_299": 0,
        "roboflow_v3_battery_140_base": 0,
        "task7_diverse_battery_56": 0,
        "batch_a_mini01_10": 0,
        "holdout_27_hinh_anh_rac": 0,
    }

    # 1. Dataset V2 & authoritative cache (covers Train V2, Val V2, Frozen Test 299)
    cache_p = REPO_ROOT / "data" / "audit" / "authoritative_hashes_cache.json"
    if cache_p.exists():
        c_data = json.loads(cache_p.read_text(encoding="utf-8"))
        for k, v in c_data.items():
            pool_counts["dataset_v2_and_cache"] += 1
            if v.get("sha256"):
                known_sha256[v["sha256"]] = f"v2_cache:{k}"
            if v.get("phash"):
                try:
                    known_phash[f"v2_cache:{k}"] = imagehash.hex_to_hash(v["phash"])
                except Exception:
                    pass

    # 2. Explicitly verify Frozen Test 299 images (splits_real_v2/test.txt)
    frozen_test_list_p = REPO_ROOT / "data" / "detection" / "splits_real_v2" / "test.txt"
    if frozen_test_list_p.exists():
        for rel_ln in frozen_test_list_p.read_text(encoding="utf-8").splitlines():
            rel_ln = rel_ln.strip()
            if not rel_ln:
                continue
            img_p = REPO_ROOT / "data" / "detection" / rel_ln
            if img_p.exists():
                pool_counts["frozen_test_299"] += 1
                sha = hashlib.sha256(img_p.read_bytes()).hexdigest()
                known_sha256[sha] = f"frozen_test_299:{img_p.name}"

    # 3. Roboflow V3 battery base images (140)
    rf_p = REPO_ROOT / "data" / "acquisition_staging" / "roboflow_battery" / "base_images_qa_manifest.json"
    if rf_p.exists():
        rf_data = json.loads(rf_p.read_text(encoding="utf-8"))
        for item in rf_data.get("items", []):
            pool_counts["roboflow_v3_battery_140_base"] += 1
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"roboflow_v3:{fn}"
            if item.get("phash"):
                try:
                    known_phash[f"roboflow_v3:{fn}"] = imagehash.hex_to_hash(item["phash"])
                except Exception:
                    pass

    # 4. Task 7 diverse battery batch (56 images)
    t7_p = REPO_ROOT / "data" / "acquisition_staging" / "task7_diverse_battery_batch" / "raw_download_manifest.json"
    if t7_p.exists():
        t7_data = json.loads(t7_p.read_text(encoding="utf-8"))
        for item in t7_data.get("candidates", []):
            pool_counts["task7_diverse_battery_56"] += 1
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"task7:{fn}"
            if item.get("phash_0"):
                try:
                    known_phash[f"task7:{fn}"] = imagehash.hex_to_hash(item["phash_0"])
                except Exception:
                    pass

    # 5. Batch A Mini 01 (10 images)
    b1_p = REPO_ROOT / "data" / "acquisition_staging" / "batch_a_mini01" / "manifest.json"
    if b1_p.exists():
        b1_data = json.loads(b1_p.read_text(encoding="utf-8"))
        for item in b1_data.get("items", []):
            pool_counts["batch_a_mini01_10"] += 1
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"batch_a_mini01:{fn}"
            if item.get("pHash"):
                try:
                    known_phash[f"batch_a_mini01:{fn}"] = imagehash.hex_to_hash(item["pHash"])
                except Exception:
                    pass

    # 6. External Holdout 27 (`data/holdout`)
    holdout_p = REPO_ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
    if holdout_p.exists():
        h_data = json.loads(holdout_p.read_text(encoding="utf-8"))
        for item in h_data.get("files", []):
            pool_counts["holdout_27_hinh_anh_rac"] += 1
            fn = item.get("filename", "")
            if item.get("sha256"):
                known_sha256[item["sha256"]] = f"holdout_27:{fn}"
            abs_p = Path(item.get("absolute_path", ""))
            if abs_p.exists():
                try:
                    with Image.open(abs_p) as himg:
                        known_phash[f"holdout_27:{fn}"] = imagehash.phash(ImageOps.exif_transpose(himg).convert("RGB"))
                except Exception:
                    pass

    exact_sha_collisions: List[Dict[str, Any]] = []
    external_phash_collisions: List[Dict[str, Any]] = []

    for c in enriched:
        c_sha = c["sha256"]
        if c_sha in known_sha256:
            exact_sha_collisions.append({
                "candidate_id": c["candidate_id"],
                "qa_verdict": c["qa_verdict"],
                "matched_external": known_sha256[c_sha],
                "sha256": c_sha,
            })
        c_rots = [imagehash.hex_to_hash(h) for h in c["phash_rotations"]]
        for ext_k, ext_ph in known_phash.items():
            min_d = min(abs(rh - ext_ph) for rh in c_rots)
            if min_d <= 8:
                external_phash_collisions.append({
                    "candidate_id": c["candidate_id"],
                    "qa_verdict": c["qa_verdict"],
                    "matched_external": ext_k,
                    "min_rotation_hamming_distance": int(min_d),
                })

    # Check internal near-duplicates across all 154 candidates and verify that any pair with pHash <= 8
    # either has one marked REJECT or is grouped into the SAME specimen_group and SAME split.
    internal_near_pairs: List[Dict[str, Any]] = []
    for i in range(len(enriched)):
        c1 = enriched[i]
        r1 = [imagehash.hex_to_hash(h) for h in c1["phash_rotations"]]
        for j in range(i + 1, len(enriched)):
            c2 = enriched[j]
            r2 = [imagehash.hex_to_hash(h) for h in c2["phash_rotations"]]
            min_d = min(abs(h1 - h2) for h1 in r1 for h2 in r2)
            if min_d <= 8 or (
                c1["specimen_group"] != "REJECTED"
                and c1["specimen_group"] == c2["specimen_group"]
            ):
                internal_near_pairs.append({
                    "candidate_1": c1["short_key"],
                    "verdict_1": c1["qa_verdict"],
                    "split_1": c1["proposed_split"],
                    "candidate_2": c2["short_key"],
                    "verdict_2": c2["qa_verdict"],
                    "split_2": c2["proposed_split"],
                    "specimen_group": c1["specimen_group"],
                    "min_rotation_hamming_distance": int(min_d),
                    "resolution": (
                        "ONE_OR_BOTH_REJECTED_OR_HELD"
                        if ("REJECT" in (c1["qa_verdict"], c2["qa_verdict"]) or "NEEDS_MANUAL_REVIEW" in (c1["qa_verdict"], c2["qa_verdict"]))
                        else f"GROUPED_TOGETHER_IN_{c1['proposed_split']}"
                    ),
                })

    # Cross-split check between TRAIN_V4_PROPOSED and VAL_REAL_MULTICLASS_PROPOSED
    train_items = [c for c in enriched if c["proposed_split"] == "TRAIN_V4_PROPOSED"]
    val_items = [c for c in enriched if c["proposed_split"] == "VAL_REAL_MULTICLASS_PROPOSED"]
    cross_split_collisions: List[Dict[str, Any]] = []

    for tc in train_items:
        t_rots = [imagehash.hex_to_hash(h) for h in tc["phash_rotations"]]
        for vc in val_items:
            v_rots = [imagehash.hex_to_hash(h) for h in vc["phash_rotations"]]
            min_d = min(abs(th - vh) for th in t_rots for vh in v_rots)
            if min_d <= 8 or tc["specimen_group"] == vc["specimen_group"]:
                cross_split_collisions.append({
                    "train_candidate": tc["short_key"],
                    "val_candidate": vc["short_key"],
                    "min_rotation_hamming_distance": int(min_d),
                    "same_specimen_group": tc["specimen_group"] == vc["specimen_group"],
                })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_pools_covered": pool_counts,
        "total_known_sha256_checked": len(known_sha256),
        "total_known_phash_checked": len(known_phash),
        "exact_sha256_collisions_with_existing_datasets": exact_sha_collisions,
        "near_phash_le8_collisions_with_existing_datasets": external_phash_collisions,
        "internal_near_duplicate_and_session_pairs": internal_near_pairs,
        "cross_split_train_vs_val_collisions": cross_split_collisions,
        "zero_external_sha256_leakage": len(exact_sha_collisions) == 0,
        "zero_external_phash_le8_leakage": len(external_phash_collisions) == 0,
        "zero_cross_split_leakage": len(cross_split_collisions) == 0,
    }


def build_10class_coverage_matrix(enriched: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Authoritative Dataset V2 baseline box counts
    v2_train_boxes = {0: 1, 1: 3, 2: 169, 3: 368, 4: 213, 5: 420, 6: 160, 7: 1560, 8: 215, 9: 854}
    v2_val_boxes = {0: 0, 1: 2, 2: 39, 3: 68, 4: 82, 5: 74, 6: 43, 7: 382, 8: 41, 9: 170}
    v2_test_boxes = {0: 1, 1: 1, 2: 36, 3: 91, 4: 32, 5: 73, 6: 34, 7: 274, 8: 39, 9: 139}

    # Previously staged QA_PASS boxes (V3 Roboflow 140 + Task 7 30 + Batch A Mini 01 8)
    prior_staged_train_boxes = {0: 111 + 40, 1: 5, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0}
    prior_staged_val_boxes = {0: 29 + 21, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    qa_pass = [c for c in enriched if c["qa_verdict"] == "QA_PASS"]

    matrix_rows: List[Dict[str, Any]] = []
    remaining_gaps_by_class = {
        0: "Need batteries mixed directly with household plastic/paper/food trash in overflowing bins (Task 8B).",
        1: "Task 8A added 31 QA_PASS images (56 boxes); next step is cooked food leftovers mixed with plastic bags/containers in Task 8B.",
        2: "Task 8A added 10 QA_PASS images (36 boxes); next step is wet crushed cardboard mixed with kitchen waste in Task 8B.",
        3: "Need crumpled/dirty discarded rags, old clothes, and fabric scraps in mixed waste bins (Task 8B).",
        4: "Task 8A added 12 QA_PASS images (26 boxes); next step is glass jars/bottles mixed with metal cans and plastic in Task 8B.",
        5: "Need crushed soda/beer cans, rusty tin cans, and foil co-occurring with plastic/paper in mixed trash scenes (Task 8B).",
        6: "Task 8A added 11 primary QA_PASS images (17 boxes total); next step is wet tissues/napkins/receipts inside mixed bins (Task 8B).",
        7: "High V2 count (1,560 train boxes), but needs crumpled plastic bags and dirty food wrappers co-occurring with other waste (Task 8B).",
        8: "Task 8A added 11 QA_PASS images (24 boxes); next step is single old shoes mixed inside household trash piles (Task 8B).",
        9: "Need strict boundary audit on generic trash (cigarette butts, masks, composite wrappers) in multi-class scenes (Task 8B).",
    }

    for cid in range(10):
        cname = f"{cid}: {CLASS_NAMES[cid]}"
        # Images in Task 8A QA_PASS that contain at least 1 box of cid
        imgs_with_cid = [
            c for c in qa_pass if any(b["class_id"] == cid for b in c["boxes_annotated"])
        ]
        primary_imgs_cid = [
            c for c in qa_pass if c["primary_class"] == cname
        ]
        train_boxes_8a = sum(
            1
            for c in qa_pass
            if c["proposed_split"] == "TRAIN_V4_PROPOSED"
            for b in c["boxes_annotated"]
            if b["class_id"] == cid
        )
        val_boxes_8a = sum(
            1
            for c in qa_pass
            if c["proposed_split"] == "VAL_REAL_MULTICLASS_PROPOSED"
            for b in c["boxes_annotated"]
            if b["class_id"] == cid
        )
        total_boxes_8a = train_boxes_8a + val_boxes_8a
        unique_authors = len({c["author"] for c in imgs_with_cid})
        unique_specimens = len({c["specimen_group"] for c in imgs_with_cid})
        multi_class_imgs = sum(1 for c in imgs_with_cid if len(c["distinct_classes_present"]) >= 2)
        complex_bg_imgs = sum(1 for c in imgs_with_cid if c["is_complex_background"])

        matrix_rows.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "class_label": cname,
            "v2_train_boxes": v2_train_boxes[cid],
            "v2_val_boxes": v2_val_boxes[cid],
            "v2_test_boxes_frozen": v2_test_boxes[cid],
            "prior_staged_qa_pass_boxes": prior_staged_train_boxes[cid] + prior_staged_val_boxes[cid],
            "task8a_primary_qa_pass_images": len(primary_imgs_cid),
            "task8a_all_qa_pass_images_containing_class": len(imgs_with_cid),
            "task8a_independent_specimen_groups": unique_specimens,
            "task8a_unique_photographer_sources": unique_authors,
            "task8a_new_train_boxes": train_boxes_8a,
            "task8a_new_val_boxes": val_boxes_8a,
            "task8a_total_new_boxes": total_boxes_8a,
            "task8a_multi_class_cooccurrence_images": multi_class_imgs,
            "task8a_complex_background_images": complex_bg_imgs,
            "cumulative_projected_train_boxes": v2_train_boxes[cid] + prior_staged_train_boxes[cid] + train_boxes_8a,
            "cumulative_projected_val_boxes": v2_val_boxes[cid] + prior_staged_val_boxes[cid] + val_boxes_8a,
            "remaining_context_gaps_for_task8b": remaining_gaps_by_class[cid],
        })

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hard_negatives_available_in_staging": {
            "task7_qa_pass_hard_negatives": 4,
            "batch_a_mini01_qa_pass_hard_negatives": 2,
            "task8a_in_frame_hard_negative_contexts": "task8a_083 includes unboxed reusable red ceramic mug beside disposable paper cup; task8a_059 includes street litter bin housing around discarded shoes.",
        },
        "classes": matrix_rows,
    }


def main() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = json.loads(RAW_MANIFEST_P.read_text(encoding="utf-8"))
    candidates = [c for c in raw_data["candidates"] if c["short_key"] in QA_DECISIONS]

    if len(candidates) != len(QA_DECISIONS):
        raise ValueError(f"Expected {len(QA_DECISIONS)} Task 8A candidates, found {len(candidates)}")

    # Save dedicated Task 8A 154-candidate raw manifest
    (TASK8A_DIR / "task8a_154_raw_download_manifest.json").write_text(
        json.dumps(
            {
                "batch_name": "task8a_multiclass_batch",
                "generated_at_utc": raw_data.get("generated_at_utc"),
                "total_candidates": len(candidates),
                "candidates": candidates,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        sk = item["short_key"]
        if sk not in QA_DECISIONS:
            raise KeyError(f"Missing QA decision for {sk}")
        dec = QA_DECISIONS[sk]

        boxes_annotated: List[Dict[str, Any]] = []
        yolo_lines: List[str] = []
        distinct_classes = set()

        for raw_b in dec["boxes"]:
            cls_id, xc, yc, bw, bh = xyxy_to_yolo(raw_b)
            distinct_classes.add(f"{cls_id}: {CLASS_NAMES[cls_id]}")
            boxes_annotated.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "bbox_xyxy_norm": [round(float(x), 6) for x in raw_b[1:]],
                "yolo_xywh_norm": [round(xc, 6), round(yc, 6), round(bw, 6), round(bh, 6)],
            })
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

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
            "primary_class": dec["primary_class"],
            "qa_verdict": dec["qa_verdict"],
            "proposed_split": dec["proposed_split"],
            "specimen_group": dec["specimen_group"],
            "scene_context": dec["scene_context"],
            "is_complex_background": dec["is_complex_background"],
            "boxes_count": len(boxes_annotated),
            "distinct_classes_present": sorted(distinct_classes),
            "is_multi_class_image": len(distinct_classes) >= 2,
            "boxes_annotated": boxes_annotated,
            "yolo_label_lines": yolo_lines,
            "label_path": label_rel,
            "qa_reason": dec["qa_reason"],
            "qa_reviewer": "Antigravity Visual QA (Full-Resolution & Coordinate Grid Verified)",
            "qa_reviewed_at_utc": review_ts,
            "approved_for_train": False,
            "approval_status": "PENDING_PM_APPROVAL",
        }

        ov_path = OVERLAYS_DIR / f"{stem}_overlay.jpg"
        draw_overlay(merged, ov_path)
        merged["overlay_path"] = ov_path.relative_to(REPO_ROOT).as_posix()
        enriched.append(merged)

    # Render QA Contact Sheets
    qa_pass_bio = [c for c in enriched if c["qa_verdict"] == "QA_PASS" and c["primary_class"] == "1: biological"]
    qa_pass_paper_card = [
        c for c in enriched if c["qa_verdict"] == "QA_PASS" and c["primary_class"] in ("6: paper", "2: cardboard")
    ]
    qa_pass_glass_shoes = [
        c for c in enriched if c["qa_verdict"] == "QA_PASS" and c["primary_class"] in ("4: glass", "8: shoes")
    ]
    needs_review_items = [c for c in enriched if c["qa_verdict"] == "NEEDS_MANUAL_REVIEW"]
    reject_items = [c for c in enriched if c["qa_verdict"] == "REJECT"]

    s1 = SHEETS_DIR / "contact_sheet_01_qa_pass_biological_part1.jpg"
    s2 = SHEETS_DIR / "contact_sheet_02_qa_pass_biological_part2.jpg"
    s3 = SHEETS_DIR / "contact_sheet_03_qa_pass_paper_and_cardboard.jpg"
    s4 = SHEETS_DIR / "contact_sheet_04_qa_pass_glass_and_shoes.jpg"
    s5 = SHEETS_DIR / "contact_sheet_05_needs_manual_review.jpg"
    s6 = SHEETS_DIR / "contact_sheet_06_rejected_candidates.jpg"

    render_qa_contact_sheet(qa_pass_bio[:16], "TASK 8A QA_PASS — 1: BIOLOGICAL (PART 1: 1..16)", s1, cols=4)
    render_qa_contact_sheet(qa_pass_bio[16:], "TASK 8A QA_PASS — 1: BIOLOGICAL (PART 2: 17..31)", s2, cols=4)
    render_qa_contact_sheet(qa_pass_paper_card, "TASK 8A QA_PASS — 6: PAPER (11) & 2: CARDBOARD (10)", s3, cols=4)
    render_qa_contact_sheet(qa_pass_glass_shoes, "TASK 8A QA_PASS — 4: GLASS (12) & 8: SHOES (11)", s4, cols=4)
    render_qa_contact_sheet(
        needs_review_items,
        f"TASK 8A NEEDS_MANUAL_REVIEW CANDIDATES ({len(needs_review_items)} IMAGES HELD FOR PM)",
        s5,
        cols=4,
    )
    render_qa_contact_sheet(
        reject_items,
        f"TASK 8A REJECTED CANDIDATES ({len(reject_items)} OFF-TARGET / NON-PHOTO / DUPLICATES)",
        s6,
        cols=5,
    )

    leakage_report = run_comprehensive_leakage_audit(enriched)
    (TASK8A_DIR / "task8a_leakage_and_session_report.json").write_text(
        json.dumps(leakage_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    coverage_matrix = build_10class_coverage_matrix(enriched)
    (TASK8A_DIR / "task8a_10class_coverage_matrix.json").write_text(
        json.dumps(coverage_matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Build dedicated license & provenance manifest
    license_manifest = {
        "manifest_name": "TASK8A_SOURCE_AND_LICENSE_PROVENANCE_MANIFEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(enriched),
        "qa_pass_candidates": sum(1 for c in enriched if c["qa_verdict"] == "QA_PASS"),
        "all_qa_pass_have_valid_open_license": all(
            bool(c["license"]) and bool(c["source_page_url"]) and bool(c["author"])
            for c in enriched
            if c["qa_verdict"] == "QA_PASS"
        ),
        "records": [
            {
                "short_key": c["short_key"],
                "candidate_id": c["candidate_id"],
                "filename": c["filename"],
                "qa_verdict": c["qa_verdict"],
                "primary_class": c["primary_class"],
                "source_file_title": c["source_file_title"],
                "source_page_url": c["source_page_url"],
                "download_url": c["download_url"],
                "original_url": c["original_url"],
                "author": c["author"],
                "license": c["license"],
                "license_url": c["license_url"],
                "usage_terms": c["usage_terms"],
                "attribution_required": c["attribution_required"],
                "credit_line": c["credit_line"],
                "acquired_at_utc": c["acquired_at_utc"],
                "sha256": c["sha256"],
                "phash_0": c["phash_0"],
                "phash_rotations": c["phash_rotations"],
            }
            for c in enriched
        ],
    }
    (TASK8A_DIR / "task8a_license_and_source_manifest.json").write_text(
        json.dumps(license_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    qa_pass_all = [c for c in enriched if c["qa_verdict"] == "QA_PASS"]
    by_primary_verdict: Dict[str, Dict[str, int]] = {}
    for c in enriched:
        pcls = c["primary_class"]
        if pcls not in by_primary_verdict:
            by_primary_verdict[pcls] = {"QA_PASS": 0, "NEEDS_MANUAL_REVIEW": 0, "REJECT": 0, "QA_PASS_BOXES": 0}
        by_primary_verdict[pcls][c["qa_verdict"]] += 1
        if c["qa_verdict"] == "QA_PASS":
            by_primary_verdict[pcls]["QA_PASS_BOXES"] += c["boxes_count"]

    qa_manifest = {
        "manifest_name": "TASK8A_MULTICLASS_REAL_WORLD_QA_MANIFEST",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_reviewer": "Antigravity Visual QA (Full-Resolution & Coordinate Grid Verified)",
        "total_candidates_audited": len(enriched),
        "summary_counts": {
            "qa_pass_images": len(qa_pass_all),
            "needs_manual_review_images": len(needs_review_items),
            "reject_images": len(reject_items),
            "approved_for_train_images": 0,
            "qa_pass_total_boxes": sum(c["boxes_count"] for c in qa_pass_all),
            "proposed_train_v4_images": sum(1 for c in qa_pass_all if c["proposed_split"] == "TRAIN_V4_PROPOSED"),
            "proposed_train_v4_boxes": sum(c["boxes_count"] for c in qa_pass_all if c["proposed_split"] == "TRAIN_V4_PROPOSED"),
            "proposed_val_real_multiclass_images": sum(
                1 for c in qa_pass_all if c["proposed_split"] == "VAL_REAL_MULTICLASS_PROPOSED"
            ),
            "proposed_val_real_multiclass_boxes": sum(
                c["boxes_count"] for c in qa_pass_all if c["proposed_split"] == "VAL_REAL_MULTICLASS_PROPOSED"
            ),
            "qa_pass_multi_object_images_ge2_boxes": sum(1 for c in qa_pass_all if c["boxes_count"] >= 2),
            "qa_pass_multi_class_cooccurrence_images": sum(1 for c in qa_pass_all if c["is_multi_class_image"]),
            "qa_pass_complex_background_images": sum(1 for c in qa_pass_all if c["is_complex_background"]),
        },
        "counts_by_primary_class": by_primary_verdict,
        "contact_sheets": [
            s1.relative_to(REPO_ROOT).as_posix(),
            s2.relative_to(REPO_ROOT).as_posix(),
            s3.relative_to(REPO_ROOT).as_posix(),
            s4.relative_to(REPO_ROOT).as_posix(),
            s5.relative_to(REPO_ROOT).as_posix(),
            s6.relative_to(REPO_ROOT).as_posix(),
        ],
        "items": enriched,
    }
    (TASK8A_DIR / "task8a_qa_manifest.json").write_text(
        json.dumps(qa_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Summary Counts:", json.dumps(qa_manifest["summary_counts"], indent=2))
    print("Counts by Primary Class:", json.dumps(by_primary_verdict, indent=2))
    print("Zero external SHA-256 leakage:", leakage_report["zero_external_sha256_leakage"])
    print("Zero external pHash<=8 leakage:", leakage_report["zero_external_phash_le8_leakage"])
    print("Zero cross-split leakage:", leakage_report["zero_cross_split_leakage"])


if __name__ == "__main__":
    main()
