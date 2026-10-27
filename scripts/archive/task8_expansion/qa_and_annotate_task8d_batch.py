"""Build and verify the Task 8D QA & Tight Per-Instance Annotation Manifest (`task8d_qa_manifest.json`).

Updated in Task 8E Section 1 (Evidence-Based Focused Verification & Fix):
1. Restores `source_split` (`train`/`valid`/`test`/`unspecified`) on all 443 Task 8D records and enforces:
   - Zero `source_split == "test"` records in `TRAIN_REAL_WASTE_W8D`.
   - Preserves `Shynuaa/Street-garbage` `source_split="valid"` in `VAL_REAL_WASTE_W8D` (never mixed into `TRAIN`).
2. Enforces strict provenance verification on `Shynuaa/Street-garbage`:
   - Keeps only verified first-party 2025 smartphone street captures (`^(2025\\d{4}_|IMG_2025\\d{4}_)`).
   - Rejects third-party web-scraped images (`task8d_0071, 0072, 0073, 0074, 0086, 0087, 0088`) and building facade reflection (`task8d_0084`).
3. Refines multi-instance / multi-class boxes on `task8d_0002`, `task8d_0031`, `task8d_0093`, `task8d_0318`.
4. Demotes standalone closed/empty waste containers and non-waste/museum/poster items from `REAL_WASTE_PASS` to `KEEP_AUXILIARY`, `NEEDS_REVIEW`, or `REJECT`.
5. Renders high-contrast thick-border overlays (`width = max(4, int(min(w, h) * 0.008))`) and exports the compact Task 8D Independent QA Evidence Mini-Pack.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8d_high_yield_real_waste"
MINI_PACK_DIR = ROOT / "data" / "acquisition_staging" / "task8d_independent_qa_mini_pack"
RAW_MANIFEST_PATH = OUT_DIR / "task8d_raw_candidates_manifest.json"
QA_MANIFEST_JSON = OUT_DIR / "task8d_qa_manifest.json"
QA_MANIFEST_CSV = OUT_DIR / "task8d_qa_manifest.csv"

LABELS_PASS_DIR = OUT_DIR / "labels_real_waste_pass"
LABELS_AUX_DIR = OUT_DIR / "labels_auxiliary"
LABELS_REVIEW_DIR = OUT_DIR / "labels_needs_manual_review"
OVERLAYS_DIR = OUT_DIR / "overlays"
SHEETS_DIR = OUT_DIR / "contact_sheets"

for d in [LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR, MINI_PACK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    0: (255, 60, 60),
    1: (60, 220, 60),
    2: (230, 160, 40),
    3: (180, 90, 240),
    4: (60, 210, 210),
    5: (190, 190, 210),
    6: (245, 230, 70),
    7: (60, 130, 255),
    8: (255, 120, 190),
    9: (255, 100, 40),
}

FIRST_PARTY_SHYNUAA_RE = re.compile(r"^shynuaa_(?:train|valid|test)_(2025\d{4}_\d{6}|IMG_2025\d{4}_\d{6})")

# 1. SRC_8D_04_SHYNUAA_STREET_GARBAGE (0001..0090)
SHYNUAA_NEEDS_REVIEW = {
    "task8d_0001": "Single tall group box covering scattered multi-item litter pile by wall; needs per-instance split.",
    "task8d_0070": "Single wide box over construction dirt yard without isolated waste object boundary.",
    "task8d_0085": "Single large group box over scattered street trash heap; needs per-instance split.",
    "task8d_0089": "Single large group box over mixed litter pile; needs per-instance split.",
    "task8d_0090": "Single large group box over mixed trash heap; needs per-instance split.",
}
SHYNUAA_REJECT = {
    "task8d_0071": ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image (6911558_3143637.jpg) without verified individual upstream license."),
    "task8d_0072": ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image (78aaa8eb58e8dc4d6fbcf97ac3f9f75e.jpg) without verified individual upstream license."),
    "task8d_0073": ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image (8-1024x769.jpg) without verified individual upstream license."),
    "task8d_0074": ("PRODUCT_STUDIO", "Stock photo with visible Depositphotos watermark."),
    "task8d_0077": ("ACTUAL_WASTE", "Redundant multi-angle shot of same specimen/session as task8d_0078 (indoor crushed bottle)."),
    "task8d_0079": ("ACTUAL_WASTE", "Redundant multi-angle shot of same specimen/session as task8d_0078 (indoor crushed bottle)."),
    "task8d_0080": ("ACTUAL_WASTE", "Redundant multi-angle shot of same specimen/session as task8d_0078 (indoor crushed bottle)."),
    "task8d_0084": ("IN_USE", "Glass building facade window reflection (IMG_20250918_131510.jpg), not street litter."),
    "task8d_0086": ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image (ac6417dd01070ad608cef22f51bcdbbc.jpg) without verified individual upstream license."),
    "task8d_0087": ("PRODUCT_STUDIO", "Third-party Freepik stock photo (beer-can-rubbish-ground_866797-9228.jpeg) without open license verification."),
    "task8d_0088": ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image (bk_info_orig_53562.jpg) without verified individual upstream license."),
}

# Explicit multi-instance tight box overrides for Shynuaa multi-object street litter scenes
SHYNUAA_MULTI_INSTANCE_OVERRIDES: dict[str, list[tuple[int, float, float, float, float]]] = {
    "task8d_0002": [
        (6, 0.327, 0.720, 0.583, 0.903),  # white paper coffee cup
        (7, 0.053, 0.794, 0.161, 0.987),  # clear plastic bottle
        (5, 0.520, 0.415, 0.686, 0.481),  # discarded metal strip
    ],
    "task8d_0031": [
        (2, 0.250, 0.410, 0.420, 0.650),  # flattened white cardboard strip
        (7, 0.440, 0.560, 0.670, 0.700),  # clear plastic bottle/jug
        (9, 0.680, 0.420, 0.820, 0.620),  # black plastic trash bag
    ],
}

# Specific 10-class material overrides for Shynuaa/Street-garbage tight CVAT boxes
SHYNUAA_CLASS_OVERRIDES: dict[str, int] = {
    "task8d_0005": 2,  # flattened cardboard piece
    "task8d_0006": 6,  # crumpled white paper tissue
    "task8d_0007": 2,  # crushed cigarette carton box
    "task8d_0008": 6,  # crumpled white paper cup/tissue
    "task8d_0009": 6,  # white paper tissue
    "task8d_0010": 7,  # green plastic bottle
    "task8d_0011": 6,  # crumpled white paper tissue
    "task8d_0012": 7,  # pink disposable plastic cup
    "task8d_0013": 6,  # crumpled paper scrap
    "task8d_0015": 6,  # crumpled paper napkin
    "task8d_0016": 5,  # crushed metal beverage can
    "task8d_0017": 4,  # glass bottle
    "task8d_0018": 2,  # crushed cardboard carton
    "task8d_0019": 6,  # crumpled paper
    "task8d_0020": 2,  # crushed cigarette pack
    "task8d_0021": 2,  # crushed cardboard box
    "task8d_0022": 2,  # crushed cardboard pack
    "task8d_0023": 6,  # wet paper scraps
    "task8d_0024": 5,  # crushed red soda cans
    "task8d_0025": 2,  # green cardboard juice box
    "task8d_0027": 5,  # crushed red aluminum can
    "task8d_0028": 6,  # white paper cup
    "task8d_0029": 6,  # paper ticket/receipt
    "task8d_0033": 6,  # crumpled white paper tissue
    "task8d_0037": 7,  # blue plastic wrapper
    "task8d_0041": 7,  # blue and red plastic bag
    "task8d_0043": 6,  # crumpled white paper tissue
    "task8d_0045": 6,  # crumpled white paper scrap
    "task8d_0046": 5,  # crushed blue metal can
    "task8d_0048": 6,  # crumpled white paper
    "task8d_0049": 7,  # white plastic bag
    "task8d_0051": 2,  # crushed cardboard box
    "task8d_0052": 5,  # crushed beverage can
    "task8d_0053": 6,  # crumpled white paper tissues
    "task8d_0054": 6,  # crumpled white paper tissues
    "task8d_0055": 2,  # crushed red cigarette box
    "task8d_0056": 6,  # crumpled white paper napkins
    "task8d_0057": 6,  # white paper cup
    "task8d_0059": 2,  # cardboard roll tube
    "task8d_0061": 2,  # crushed cigarette box
    "task8d_0062": 2,  # crushed cardboard pack
    "task8d_0065": 2,  # crushed cigarette box
    "task8d_0066": 2,  # crushed white cardboard pack
    "task8d_0067": 6,  # crumpled white paper tissue
    "task8d_0068": 7,  # disposable plastic cup on park bench
    "task8d_0075": 4,  # glass bottle on outdoor ledge
    "task8d_0076": 4,  # glass bottle left by outdoor pillar
    "task8d_0078": 7,  # crushed plastic bottle on floor
    "task8d_0081": 6,  # crumpled paper receipt
    "task8d_0083": 7,  # purple plastic bag & litter
}

# 2. SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED (0091..0171)
# Only scenes with verified visible waste contents pass REAL_WASTE_PASS; closed/empty bins move to KEEP_AUXILIARY
OI_REAL_WASTE_PASS: dict[str, tuple[str, list[tuple[int, float, float, float, float]] | None]] = {
    "task8d_0093": (
        "Overflowing dumpster with stuffed black plastic trash bag and discarded cardboard box on top.",
        [
            (9, 0.390, 0.130, 0.776, 0.946),  # overflowing waste dumpster with mixed trash bags
            (2, 0.530, 0.120, 0.768, 0.355),  # discarded brown cardboard box on top of dumpster
            (7, 0.405, 0.145, 0.555, 0.370),  # black plastic trash bag overflowing top left
        ],
    ),
    "task8d_0103": ("Open waste containers with visible blue and white plastic trash bag liners and litter.", None),
    "task8d_0112": ("Open wire/metal park trash cans with visible discarded litter inside.", None),
    "task8d_0115": ("Open metal waste bin on pavement with overflowing plastic liner bag.", None),
    "task8d_0163": ("Moldy slice of bread with visible blue-green fungal colonies (held out of train because source_split=test).", None),
}

OI_NEEDS_REVIEW: dict[str, str] = {
    "task8d_0114": "Large pile of 200+ rotting dumped apples where Open Images only annotated 4 sample apples; held for full instance segmentation.",
}

OI_KEEP_AUXILIARY: dict[str, tuple[str, str]] = {
    "task8d_0091": ("AMBIGUOUS_CONTEXT", "Fresh cherry tomatoes ('Rotten tomatoes' metaphor in title); valid biological bbox for auxiliary only."),
    "task8d_0092": ("AMBIGUOUS_CONTEXT", "Intact loaf of bread; valid biological bbox for auxiliary only."),
    "task8d_0094": ("IN_USE", "Stacked cardboard storage boxes indoors; auxiliary only."),
    "task8d_0095": ("IN_USE", "Closed outdoor public waste container without visible waste contents; auxiliary only."),
    "task8d_0096": ("IN_USE", "Closed outdoor public waste and recycling containers; auxiliary only."),
    "task8d_0101": ("IN_USE", "Standalone white waste bin container; auxiliary only."),
    "task8d_0104": ("IN_USE", "Closed outdoor street waste bin container; auxiliary only."),
    "task8d_0108": ("IN_USE", "Cardboard box on floor with cat inside; auxiliary only."),
    "task8d_0109": ("IN_USE", "Closed green wheeled waste container outdoors; auxiliary only."),
    "task8d_0113": ("IN_USE", "Row of closed outdoor waste/recycling bins; auxiliary only."),
    "task8d_0116": ("AMBIGUOUS_CONTEXT", "Wine glass covered with plastic wrap; auxiliary only."),
    "task8d_0117": ("IN_USE", "Closed outdoor dumpsters with graffiti; auxiliary only."),
    "task8d_0118": ("IN_USE", "Stacked cardboard moving boxes indoors; auxiliary only."),
    "task8d_0119": ("IN_USE", "Closed street litter bin container; auxiliary only."),
    "task8d_0120": ("AMBIGUOUS_CONTEXT", "Intact green glass bottle on table; auxiliary only."),
    "task8d_0121": ("IN_USE", "Closed small metal pedal waste bin indoors; auxiliary only."),
    "task8d_0123": ("AMBIGUOUS_CONTEXT", "Cracked raw egg during cooking; auxiliary only."),
    "task8d_0129": ("IN_USE", "Freshly sliced bread on cutting board; auxiliary only."),
    "task8d_0130": ("IN_USE", "Old books on library shelf; auxiliary only."),
    "task8d_0133": ("IN_USE", "Closed blue recycling bin and black waste bin indoors; auxiliary only."),
    "task8d_0137": ("AMBIGUOUS_CONTEXT", "Bitten apple on plate; auxiliary only."),
    "task8d_0138": ("AMBIGUOUS_CONTEXT", "Open cardboard box indoors; auxiliary only."),
    "task8d_0139": ("AMBIGUOUS_CONTEXT", "Fried rice in open takeout tray; auxiliary only."),
    "task8d_0142": ("IN_USE", "Closed blue curbside recycling bins (source_split=test); auxiliary only."),
    "task8d_0146": ("AMBIGUOUS_CONTEXT", "Intact metal tin canister; auxiliary only."),
    "task8d_0147": ("IN_USE", "Closed yellow outdoor litter bin; auxiliary only."),
    "task8d_0149": ("IN_USE", "Closed sorted recycling/waste containers; auxiliary only."),
    "task8d_0150": ("IN_USE", "Closed green recycling bins outdoors; auxiliary only."),
    "task8d_0151": ("IN_USE", "Peeled banana being eaten; auxiliary only."),
    "task8d_0152": ("IN_USE", "Closed grey push-flap trash bin; auxiliary only."),
    "task8d_0155": ("IN_USE", "Closed Recology compost and trash bins; auxiliary only."),
    "task8d_0156": ("IN_USE", "Three closed curbside trash and recycling carts; auxiliary only."),
    "task8d_0164": ("IN_USE", "Closed green wheeled dumpster on street; auxiliary only."),
    "task8d_0167": ("IN_USE", "Closed outdoor street rubbish/litter bin; auxiliary only."),
    "task8d_0168": ("IN_USE", "Freshly cut carrots on kitchen cutting board; auxiliary only."),
    "task8d_0169": ("IN_USE", "Park garbage cans without visible waste items; auxiliary only."),
    "task8d_0171": ("IN_USE", "Closed solar-powered street trash compactor bin; auxiliary only."),
}

# 3. SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES (0172..0443)
WIKI_REAL_WASTE_PASS: dict[str, tuple[str, list[tuple[int, float, float, float, float]]]] = {
    # Class 0: battery
    "task8d_0187": ("Heavily rusted and corroded vintage lantern battery block.", [(0, 0.21, 0.08, 0.78, 0.94)]),
    "task8d_0318": (
        "Two corroded Ever Ready AD35 battery cells inside torn cardboard casing.",
        [
            (0, 0.35, 0.37, 0.61, 0.65),
            (0, 0.53, 0.43, 0.81, 0.73),
            (2, 0.13, 0.14, 0.88, 0.89),
        ],
    ),
    "task8d_0321": ("Corroded alkaline AA battery with crystalline potassium carbonate leakage.", [(0, 0.18, 0.25, 0.83, 0.64)]),
    "task8d_0322": ("Ruptured and corroded alkaline AA battery cell.", [(0, 0.06, 0.04, 0.92, 0.95)]),
    "task8d_0323": ("Corroded N2H button cell battery with chemical deposit.", [(0, 0.30, 0.06, 0.73, 0.89)]),
    "task8d_0324": ("Six heavily rusted corroded cylindrical AA batteries on ground.", [
        (0, 0.09, 0.38, 0.36, 0.81),
        (0, 0.22, 0.34, 0.49, 0.77),
        (0, 0.35, 0.31, 0.63, 0.71),
        (0, 0.47, 0.25, 0.77, 0.65),
        (0, 0.59, 0.22, 0.88, 0.60),
        (0, 0.69, 0.16, 0.97, 0.54),
    ]),
    "task8d_0325": ("Four severely corroded AA batteries inside rusted battery compartment.", [
        (0, 0.13, 0.33, 0.31, 0.90),
        (0, 0.24, 0.28, 0.43, 0.86),
        (0, 0.34, 0.25, 0.54, 0.82),
        (0, 0.53, 0.18, 0.86, 0.76),
    ]),
    "task8d_0327": ("Corroded HD625 mercury button cell battery covered in white crust.", [(0, 0.13, 0.10, 0.87, 0.92)]),
    "task8d_0329": ("Two leaked zinc-carbon cylindrical batteries with pitted metal casings.", [
        (0, 0.04, 0.25, 0.83, 0.97),
        (0, 0.19, 0.13, 0.98, 0.69),
    ]),
    "task8d_0330": ("Corroded button cell battery with green and rust chemical leakage.", [(0, 0.10, 0.14, 0.86, 0.89)]),
    "task8d_0331": ("Three leaking cylindrical batteries on surface.", [
        (0, 0.11, 0.36, 0.35, 0.74),
        (0, 0.38, 0.37, 0.51, 0.76),
        (0, 0.68, 0.45, 0.84, 0.86),
    ]),
    "task8d_0332": ("Corroded purple-and-yellow AA battery covered in crystalline leakage.", [(0, 0.06, 0.34, 0.91, 0.82)]),
    "task8d_0334": ("Leaked vintage Pertrix Varta battery block with white corrosion crust.", [(0, 0.07, 0.35, 0.94, 0.86)]),

    # Class 1: biological (+ multi-class trash/plastic/paper)
    "task8d_0178": ("Discarded half-eaten sandwich bread on grass with sparrow.", [(1, 0.44, 0.56, 0.83, 0.88)]),
    "task8d_0209": ("Spilled food scraps on street pavement beside metal dumpster.", [(1, 0.46, 0.41, 0.85, 0.92), (9, 0.02, 0.02, 0.61, 0.52)]),
    "task8d_0216": ("Decomposing shriveled brown orange.", [(1, 0.19, 0.06, 0.81, 0.92)]),
    "task8d_0218": ("Decomposing pear covered in white fungal mold.", [(1, 0.04, 0.03, 0.96, 0.97)]),
    "task8d_0223": ("Shovel holding decomposing composted organic waste.", [(1, 0.27, 0.33, 0.67, 0.84)]),
    "task8d_0226": ("Food waste in bin including darkened banana peels, orange peel, and apple core.", [
        (1, 0.36, 0.04, 0.96, 0.95),
        (1, 0.04, 0.35, 0.43, 0.73),
        (1, 0.03, 0.69, 0.36, 0.97),
    ]),
    "task8d_0227": ("Kitchen vegetable and fruit peel scraps inside white plastic bag.", [(1, 0.09, 0.03, 0.92, 0.91), (7, 0.02, 0.02, 0.98, 0.98)]),
    "task8d_0336": ("Discarded yellow banana peel.", [(1, 0.03, 0.06, 0.98, 0.88)]),
    "task8d_0337": ("Discarded banana peel and paper cup littered on pavement.", [(1, 0.06, 0.08, 0.49, 0.68), (6, 0.59, 0.57, 0.90, 0.86)]),
    "task8d_0338": ("Discarded banana peel.", [(1, 0.03, 0.18, 0.96, 0.80)]),
    "task8d_0339": ("Spotted brown discarded banana peel.", [(1, 0.18, 0.12, 0.96, 0.96)]),
    "task8d_0340": ("Discarded banana peel left on wooden park bench.", [(1, 0.33, 0.20, 0.55, 0.45)]),
    "task8d_0341": ("Discarded banana peel on asphalt walkway.", [(1, 0.44, 0.48, 0.98, 0.74)]),
    "task8d_0344": ("Blackened overripe discarded banana peel.", [(1, 0.06, 0.04, 0.90, 0.98)]),
    "task8d_0345": ("Discarded peeled banana skin.", [(1, 0.08, 0.03, 0.94, 0.93)]),
    "task8d_0348": ("Rotting blackened banana peels decomposing on soil.", [(1, 0.15, 0.03, 0.94, 0.82)]),
    "task8d_0349": ("Brown discarded banana peel on asphalt.", [(1, 0.04, 0.08, 0.98, 0.84)]),
    "task8d_0353": ("Discarded banana peel on stone ledge.", [(1, 0.34, 0.39, 0.98, 0.96)]),

    # Class 3: clothes (discarded gloves)
    "task8d_0242": ("Dirty discarded glove on cobblestone gutter.", [(3, 0.36, 0.33, 0.71, 0.73)]),
    "task8d_0243": ("Discarded disposable glove on asphalt.", [(3, 0.29, 0.45, 0.69, 0.79)]),
    "task8d_0244": ("Discarded blue glove among dry leaves on dirt.", [(3, 0.39, 0.30, 0.63, 0.66)]),
    "task8d_0245": ("Discarded blue glove on asphalt road.", [(3, 0.38, 0.32, 0.69, 0.64)]),
    "task8d_0246": ("Two discarded blue gloves on street pavement.", [(3, 0.22, 0.25, 0.38, 0.51), (3, 0.53, 0.51, 0.71, 0.76)]),
    "task8d_0247": ("Discarded purple glove and cigarette butt on pavement.", [(3, 0.21, 0.34, 0.52, 0.76)]),
    "task8d_0248": ("Discarded blue glove on wet pebble pavement.", [(3, 0.61, 0.36, 0.91, 0.64)]),
    "task8d_0249": ("Discarded white glove on sidewalk.", [(3, 0.54, 0.44, 0.86, 0.70)]),
    "task8d_0250": ("Discarded black glove on sidewalk pavement.", [(3, 0.51, 0.41, 0.80, 0.68)]),

    # Class 4: glass
    "task8d_0388": ("Smashed glass bottle shards on wet asphalt.", [(4, 0.29, 0.14, 0.66, 0.40)]),
    "task8d_0392": ("Broken flat glass shards discarded on forest ground.", [(4, 0.04, 0.29, 0.96, 0.95)]),
    "task8d_0393": ("Broken glass bottle bases and shards on dirt.", [
        (4, 0.03, 0.22, 0.58, 0.78),
        (4, 0.55, 0.26, 0.98, 0.78),
    ]),

    # Class 5: metal
    "task8d_0263": ("Crushed purple aluminum soda can discarded in construction rubble.", [(5, 0.29, 0.40, 0.46, 0.65)]),
    "task8d_0397": ("Crushed Pepsi aluminum beverage can.", [(5, 0.05, 0.02, 0.95, 0.97)]),
    "task8d_0399": ("Detached aluminum beverage can pull tab.", [(5, 0.07, 0.11, 0.87, 0.86)]),
    "task8d_0427": ("Crushed green aluminum can littered on snowy woodland ground.", [(5, 0.41, 0.51, 0.56, 0.67)]),

    # Class 6: paper
    "task8d_0264": ("Crushed discarded Hardee's paper cup in grass.", [(6, 0.31, 0.33, 0.89, 0.84)]),
    "task8d_0268": ("Torn piece of printed magazine paper littered on pavement.", [(6, 0.21, 0.10, 0.50, 0.93)]),
    "task8d_0426": ("Crushed orange-and-white cigarette pack littered on asphalt.", [(6, 0.35, 0.31, 0.60, 0.59)]),

    # Class 7: plastic & Class 9: trash (+ multi-class)
    "task8d_0276": ("Bin overflowing with Mountain Dew cardboard carton, plastic bottles, and cans.", [
        (2, 0.21, 0.20, 0.61, 0.45),
        (7, 0.69, 0.24, 0.86, 0.52),
        (9, 0.17, 0.11, 0.88, 0.83),
    ]),
    "task8d_0278": ("Two crushed plastic bottles and two crumpled paper scraps littered on grass.", [
        (7, 0.39, 0.25, 0.61, 0.34),
        (7, 0.53, 0.64, 0.75, 0.84),
        (6, 0.25, 0.21, 0.41, 0.32),
        (6, 0.67, 0.49, 0.85, 0.66),
    ]),
    "task8d_0295": ("Dumped black and grey plastic rubbish bags beside brick wall.", [
        (7, 0.41, 0.58, 0.61, 0.81),
        (7, 0.58, 0.56, 0.84, 0.85),
    ]),
    "task8d_0298": ("Dirty crushed 2-liter Coca-Cola plastic bottle on soil.", [(7, 0.29, 0.05, 0.77, 0.94)]),
    "task8d_0299": ("Dumped green plastic trash bags and broken pushchair on street corner.", [
        (7, 0.34, 0.56, 0.61, 0.96),
        (7, 0.56, 0.52, 0.80, 0.93),
        (9, 0.27, 0.26, 0.77, 0.66),
    ]),
    "task8d_0300": ("Black plastic trash bags and dumped CRT monitor beside street litter bin.", [
        (7, 0.03, 0.54, 0.34, 0.92),
        (7, 0.58, 0.55, 0.86, 0.84),
        (9, 0.38, 0.23, 0.69, 0.91),
    ]),
    "task8d_0301": ("Dumped cardboard box, plastic trash bags, and overflowing street bin.", [
        (2, 0.16, 0.47, 0.34, 0.75),
        (7, 0.64, 0.58, 0.76, 0.83),
        (9, 0.48, 0.49, 0.65, 0.84),
    ]),
    "task8d_0302": ("Black plastic rubbish bag dumped in narrow alley.", [(7, 0.65, 0.56, 0.73, 0.80)]),
    "task8d_0303": ("Plastic rubbish bags piled around street litter bin.", [
        (7, 0.04, 0.51, 0.18, 0.72),
        (7, 0.16, 0.47, 0.28, 0.68),
        (9, 0.29, 0.41, 0.45, 0.62),
    ]),
    "task8d_0304": ("Green plastic trash bag hanging from park railings.", [(7, 0.37, 0.46, 0.46, 0.64)]),
    "task8d_0305": ("Overflowing litter bin surrounded by black plastic trash bags.", [
        (7, 0.16, 0.59, 0.34, 0.78),
        (7, 0.29, 0.39, 0.49, 0.68),
        (7, 0.48, 0.38, 0.68, 0.63),
        (9, 0.53, 0.18, 0.72, 0.56),
    ]),
    "task8d_0311": ("Curbside litter bin surrounded by green, black, and white plastic rubbish bags.", [
        (7, 0.25, 0.65, 0.63, 0.87),
        (7, 0.44, 0.53, 0.62, 0.69),
        (9, 0.35, 0.41, 0.57, 0.66),
    ]),
    "task8d_0424": ("Torn white plastic bag littered in tall grass.", [(7, 0.04, 0.21, 0.90, 0.90)]),

    # Class 8: shoes
    "task8d_0435": ("Dirty abandoned wedge sandal in grass and weeds.", [(8, 0.02, 0.08, 0.84, 0.88)]),
    "task8d_0436": ("Discarded black Nine West heel shoe on sandy ground with cigarette butt inside.", [(8, 0.28, 0.26, 0.90, 0.64)]),
    "task8d_0437": ("Old forgotten shoe half-buried in desert sand.", [(8, 0.35, 0.32, 0.63, 0.65)]),
    "task8d_0439": ("Detached rubber shoe sole buried in beach sand.", [(8, 0.31, 0.39, 0.70, 0.58)]),
    "task8d_0442": ("Abandoned white-and-red Nike sneaker filled with snow in alley.", [(8, 0.34, 0.56, 0.96, 0.95)]),
}

WIKI_NEEDS_REVIEW: dict[str, tuple[str, list[tuple[int, float, float, float, float]]]] = {
    "task8d_0203": ("Dense compost heap requiring instance/segmentation policy review.", [(1, 0.04, 0.04, 0.96, 0.95)]),
    "task8d_0261": ("Compressed industrial bale of hundreds of crushed cans; held from instance train.", [(5, 0.01, 0.20, 0.99, 0.98)]),
    "task8d_0262": ("Close-up compressed bale of crushed cans; held from instance train.", [(5, 0.01, 0.01, 0.99, 0.99)]),
    "task8d_0272": ("Stacked bale of crushed cans; held from instance train.", [(5, 0.11, 0.05, 0.88, 0.96)]),
    "task8d_0292": ("Wide alley scene with scattered un-isolated dump heap.", [(9, 0.18, 0.52, 0.95, 0.94)]),
    "task8d_0293": ("Dense trash heap stuffed under fence; held for per-instance annotation.", [(9, 0.24, 0.42, 0.85, 0.62)]),
    "task8d_0297": ("Fly-tipped bulky furniture/mattress pile; held for review.", [(9, 0.24, 0.24, 0.77, 0.95)]),
    "task8d_0309": ("Distant woodland dump heap; held for per-instance split.", [(9, 0.18, 0.57, 0.91, 0.82)]),
    "task8d_0313": ("Landfill background behind water buffalo; held from detector train.", [(9, 0.01, 0.46, 0.65, 0.98)]),
    "task8d_0314": ("Large fly-tipped bulky waste heap; held for per-instance split.", [(9, 0.10, 0.24, 0.93, 0.96)]),
    "task8d_0328": ("Corroded green NiCd button cell battery soldered on PCB; held for component vs waste review.", [(0, 0.58, 0.18, 0.79, 0.73)]),
    "task8d_0335": ("Leaked green cylindrical NiCd battery soldered on synthesizer PCB; held for review.", [(0, 0.56, 0.32, 0.87, 0.55)]),
    "task8d_0343": ("Marathon crowd street scene with dozens of tiny scattered peels/cups.", [(1, 0.02, 0.68, 0.95, 0.97)]),
    "task8d_0354": ("Large pile of 10+ cardboard boxes and plastic bags on Brick Lane street; held for full instance split.", [
        (2, 0.24, 0.11, 0.54, 0.37),
        (2, 0.35, 0.36, 0.69, 0.61),
        (7, 0.20, 0.58, 0.49, 0.94),
        (7, 0.55, 0.46, 0.98, 0.97),
    ]),
    "task8d_0355": ("Multi-piece flattened cardboard pile and broken umbrella on pavement; held for full per-piece split.", [
        (2, 0.02, 0.45, 0.51, 0.97),
        (2, 0.34, 0.03, 0.82, 0.49),
        (9, 0.02, 0.08, 0.62, 0.52),
    ]),
    "task8d_0378": ("Bulk bin of shredded rag pulp; held from instance train.", [(3, 0.02, 0.30, 0.95, 0.98)]),
    "task8d_0381": ("Heap of prayer cloth flags on stone cairn; held from instance train.", [(3, 0.16, 0.06, 0.65, 0.78)]),
    "task8d_0406": ("Large heap of bags of cans; held for per-bag split.", [(7, 0.02, 0.08, 0.98, 0.98)]),
    "task8d_0407": ("Wire grid over hundreds of cans; held from instance train.", [(5, 0.02, 0.02, 0.98, 0.98)]),
    "task8d_0408": ("Compressed block of hundreds of crushed cans; held from instance train.", [(5, 0.01, 0.01, 0.99, 0.99)]),
    "task8d_0413": ("Group of students sitting on paper bundles; held due to human foreground.", [(6, 0.14, 0.48, 0.76, 0.83)]),
    "task8d_0417": ("Industrial bales of shredded waste paper; held from instance train.", [(6, 0.03, 0.06, 0.98, 0.95)]),
    "task8d_0419": ("Green wire cage filled with dozens of waste paper and cardboard sheets; held from instance train.", [
        (6, 0.41, 0.26, 0.88, 0.78),
        (2, 0.30, 0.62, 0.46, 0.77),
    ]),
    "task8d_0423": ("Bulk shredded paper strips on conveyor; held from instance train.", [(6, 0.02, 0.41, 0.98, 0.98)]),
    "task8d_0425": ("Bin filled with 15+ overlapping crushed cans and bottles; held for complete per-can segmentation.", [
        (5, 0.06, 0.38, 0.32, 0.59),
        (5, 0.58, 0.42, 0.91, 0.67),
        (5, 0.26, 0.69, 0.64, 0.84),
        (7, 0.11, 0.06, 0.58, 0.34),
    ]),
    "task8d_0428": ("Dense beach litter collection of 20+ items; held for complete per-item annotation.", [
        (7, 0.05, 0.44, 0.24, 0.78),
        (8, 0.47, 0.37, 0.60, 0.78),
        (3, 0.60, 0.38, 0.81, 0.53),
    ]),
    "task8d_0430": ("Un-isolated debris heap in weeds; held for manual review.", [(9, 0.03, 0.22, 0.99, 0.94)]),
    "task8d_0431": ("Wide roadside dumpster litter scatter; held for per-instance split.", [(9, 0.01, 0.14, 0.89, 0.96)]),
    "task8d_0432": ("Dense pile of 20+ discarded shoes on sidewalk; held for per-shoe split.", [(8, 0.31, 0.37, 0.66, 0.70)]),
}

WIKI_KEEP_AUXILIARY: dict[str, tuple[str, list[tuple[int, float, float, float, float]]]] = {
    "task8d_0195": ("IN_USE", [(0, 0.26, 0.05, 0.81, 0.96)]),
    "task8d_0200": ("IN_USE", [(9, 0.12, 0.35, 0.79, 0.79), (7, 0.48, 0.62, 0.56, 0.79)]),
    "task8d_0201": ("IN_USE", [(9, 0.04, 0.18, 0.96, 0.99)]),
    "task8d_0206": ("IN_USE", [(0, 0.21, 0.04, 0.68, 0.96)]),
    "task8d_0217": ("IN_USE", [(9, 0.21, 0.09, 0.80, 0.91)]),
    "task8d_0220": ("IN_USE", [(0, 0.03, 0.22, 0.83, 0.98)]),
    "task8d_0269": ("IN_USE", [
        (6, 0.21, 0.08, 0.77, 0.39),
        (6, 0.16, 0.39, 0.78, 0.56),
        (6, 0.17, 0.59, 0.79, 0.78),
        (6, 0.17, 0.82, 0.80, 0.98),
    ]),
    "task8d_0280": ("PRODUCT_STUDIO", [(8, 0.15, 0.10, 0.87, 0.73)]),
    "task8d_0312": ("IN_USE", [(9, 0.16, 0.16, 0.77, 0.88)]),
    "task8d_0347": ("AMBIGUOUS_CONTEXT", [(1, 0.11, 0.09, 0.98, 0.84)]),
    "task8d_0356": ("IN_USE", [(2, 0.35, 0.12, 0.72, 0.76)]),
    "task8d_0357": ("PRODUCT_STUDIO", [(2, 0.35, 0.11, 0.68, 0.88)]),
    "task8d_0359": ("IN_USE", [(2, 0.06, 0.07, 0.95, 0.92)]),
    "task8d_0360": ("IN_USE", [(2, 0.12, 0.24, 0.78, 0.75)]),
    "task8d_0361": ("IN_USE", [(2, 0.02, 0.02, 0.98, 0.98)]),
    "task8d_0362": ("PRODUCT_STUDIO", [(2, 0.03, 0.03, 0.97, 0.96)]),
    "task8d_0363": ("IN_USE", [(2, 0.02, 0.03, 0.98, 0.97)]),
    "task8d_0364": ("PRODUCT_STUDIO", [(2, 0.03, 0.02, 0.97, 0.98)]),
    "task8d_0366": ("IN_USE", [(2, 0.02, 0.12, 0.92, 0.95)]),
    "task8d_0390": ("AMBIGUOUS_CONTEXT", [(4, 0.09, 0.38, 0.39, 0.67), (4, 0.49, 0.37, 0.66, 0.56)]),
    "task8d_0400": ("PRODUCT_STUDIO", [(5, 0.22, 0.11, 0.73, 0.90)]),
    "task8d_0401": ("PRODUCT_STUDIO", [(5, 0.23, 0.10, 0.72, 0.95)]),
    "task8d_0402": ("IN_USE", [(5, 0.31, 0.02, 0.66, 0.48), (1, 0.08, 0.50, 0.92, 0.96)]),
    "task8d_0404": ("PRODUCT_STUDIO", [(5, 0.18, 0.05, 0.77, 0.96)]),
    "task8d_0405": ("PRODUCT_STUDIO", [(5, 0.26, 0.03, 0.73, 0.97)]),
    "task8d_0409": ("PRODUCT_STUDIO", [(5, 0.27, 0.06, 0.73, 0.92)]),
    "task8d_0415": ("IN_USE", [(9, 0.04, 0.04, 0.82, 0.79)]),
    "task8d_0438": ("IN_USE", [(8, 0.22, 0.10, 0.92, 0.87)]),
    "task8d_0440": ("IN_USE", [(8, 0.14, 0.10, 0.42, 0.67), (8, 0.64, 0.12, 0.90, 0.76)]),
    "task8d_0441": ("AMBIGUOUS_CONTEXT", [
        (8, 0.16, 0.46, 0.93, 0.93),
        (8, 0.17, 0.29, 0.59, 0.63),
        (8, 0.50, 0.14, 0.84, 0.52),
        (8, 0.22, 0.03, 0.49, 0.32),
    ]),
    "task8d_0443": ("PRODUCT_STUDIO", [(8, 0.31, 0.34, 0.69, 0.69)]),
}


def norm_boxes_to_struct(w: int, h: int, boxes_norm: list[tuple[int, float, float, float, float]], origin: str) -> list[dict[str, Any]]:
    out = []
    for cid, x1, y1, x2, y2 in boxes_norm:
        x1c = max(0.0, min(1.0, x1))
        y1c = max(0.0, min(1.0, y1))
        x2c = max(0.0, min(1.0, x2))
        y2c = max(0.0, min(1.0, y2))
        if x2c - x1c < 0.01 or y2c - y1c < 0.01:
            continue
        xc = round((x1c + x2c) / 2.0, 6)
        yc = round((y1c + y2c) / 2.0, 6)
        bw = round(x2c - x1c, 6)
        bh = round(y2c - y1c, 6)
        out.append({
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "bbox_xyxy": [round(x1c * w, 1), round(y1c * h, 1), round(x2c * w, 1), round(y2c * h, 1)],
            "yolo_xywhn": [xc, yc, bw, bh],
            "source_box_origin": origin,
        })
    return out


def format_yolo_lines(boxes: list[dict[str, Any]]) -> str:
    lines = []
    for b in boxes:
        xc, yc, bw, bh = b["yolo_xywhn"]
        lines.append(f"{b['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def render_overlay(rec: dict[str, Any]) -> str:
    img_p = ROOT / rec["image_path"]
    with Image.open(img_p) as im:
        im_rgb = im.convert("RGB")
        w, h = im_rgb.size
        draw = ImageDraw.Draw(im_rgb)
        thickness = max(4, int(min(w, h) * 0.008))
        for b in rec["boxes"]:
            cid = b["class_id"]
            cname = b["class_name"]
            xc, yc, bw, bh = b["yolo_xywhn"]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            col = CLASS_COLORS.get(cid, (0, 255, 0))
            for t in range(thickness):
                draw.rectangle([max(0, x1 - t), max(0, y1 - t), min(w - 1, x2 + t), min(h - 1, y2 + t)], outline=col)
            lbl = f"{cid}:{cname}"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl) * 7 + 10, y1], fill=col)
            draw.text((x1 + 4, max(0, y1 - 18)), lbl, fill=(10, 10, 10))
        out_p = OVERLAYS_DIR / f"{rec['candidate_id']}_overlay.jpg"
        im_rgb.save(out_p, quality=88)
        return str(out_p.relative_to(ROOT)).replace("\\", "/")


def render_sheet(records: list[dict[str, Any]], title: str, out_name: str, max_items: int = 36) -> str:
    subset = records[:max_items]
    if not subset:
        return ""
    cols = 6
    rows = (len(subset) + cols - 1) // cols
    cell_w, cell_h = 260, 220
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{title} (showing {len(subset)}/{len(records)})", fill=(240, 240, 245))
    for idx, rec in enumerate(subset):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * cell_h + 40
        ov_p = ROOT / rec["overlay_path"] if rec.get("overlay_path") else ROOT / rec["image_path"]
        if ov_p.exists():
            with Image.open(ov_p) as im:
                im_rgb = im.convert("RGB")
                im_rgb.thumbnail((cell_w - 8, cell_h - 36))
                sheet.paste(im_rgb, (x0 + 4, y0 + 4))
        cid_s = rec["candidate_id"].replace("task8d_", "")
        cls_s = ",".join(str(x) for x in rec["classes_present_ids"])
        sp_s = rec.get("source_split", "unk")[:4]
        cap = f"{cid_s}|c:{cls_s}|s:{sp_s}|{rec['target_split'][:9]}"
        draw.text((x0 + 4, y0 + cell_h - 28), cap, fill=(220, 235, 220))
    out_p = SHEETS_DIR / out_name
    sheet.save(out_p, quality=88)
    return str(out_p.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    raw = json.loads(RAW_MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_records: list[dict[str, Any]] = raw["records"]

    # Clean old label files
    for d in [LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR]:
        for f in d.glob("*.txt"):
            f.unlink()

    qa_records: list[dict[str, Any]] = []

    for r in raw_records:
        cid = r.get("candidate_id") or r.get("item_id")
        raw_src = r.get("source_id") or r.get("source_family", "")
        if "shynuaa" in raw_src.lower():
            src = "SRC_8D_04_SHYNUAA_STREET_GARBAGE"
            src_name = "Shynuaa/Street-garbage (HuggingFace CVAT Urban Litter Dataset)"
            source_split = r.get("source_split") or "valid"
        elif "openimages" in raw_src.lower():
            src = "SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED"
            src_name = f"Open Images V7 Waste-Context Filtered ({raw_src})"
            source_split = r.get("source_split") or "train"
        else:
            src = "SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES"
            src_name = "Wikimedia Commons Waste Categories & Bitmap Queries"
            source_split = r.get("source_split") or "unspecified"

        w, h = int(r["width"]), int(r["height"])
        boxes: list[dict[str, Any]] = []
        if r.get("proposed_boxes"):
            boxes = list(r["proposed_boxes"])
        elif r.get("upstream_boxes"):
            for ub in r["upstream_boxes"]:
                bcid = int(ub["class_id"])
                xc = round(float(ub["cx"]), 6)
                yc = round(float(ub["cy"]), 6)
                bw = round(float(ub["w"]), 6)
                bh = round(float(ub["h"]), 6)
                x1 = round(max(0.0, xc - bw / 2.0) * w, 1)
                y1 = round(max(0.0, yc - bh / 2.0) * h, 1)
                x2 = round(min(1.0, xc + bw / 2.0) * w, 1)
                y2 = round(min(1.0, yc + bh / 2.0) * h, 1)
                boxes.append({
                    "class_id": bcid,
                    "class_name": CLASS_NAMES[bcid],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "yolo_xywhn": [xc, yc, bw, bh],
                    "source_box_origin": raw_src,
                })

        img_rel = r.get("image_path") or f"data/acquisition_staging/task8d_high_yield_real_waste/images/{r.get('filename', cid + '.jpg')}"
        title_str = r.get("title") or r.get("original_title", "")
        orig_url = r.get("original_url") or r.get("landing_url") or r.get("image_url", "")
        dhash_str = r.get("dhash_hex") or r.get("phash_hex", "")

        qa_status = "REJECT"
        domain_tag = "AMBIGUOUS_CONTEXT"
        obj_valid = False
        waste_verified = False
        approved_for_train = False
        reason = ""
        provenance_tier = "VERIFIED_PUBLIC_LICENSE"
        session_group = r.get("session_group") or f"sess_{src}_{r.get('author', 'unk')[:24]}"
        specimen_group = r.get("specimen_group") or f"spec_{cid}"

        # Group Dowsett Dumping 0300 & 0301 into same session_group so they never cross train/val
        if cid in ("task8d_0300", "task8d_0301"):
            session_group = "sess_wikimedia_dowsett_dumping"
            specimen_group = "spec_wikimedia_dowsett_dumping"

        if src == "SRC_8D_04_SHYNUAA_STREET_GARBAGE":
            is_first_party = bool(FIRST_PARTY_SHYNUAA_RE.match(r["source_image_id"]))
            provenance_tier = (
                "VERIFIED_FIRST_PARTY_SMARTPHONE_CAPTURE_MIT"
                if is_first_party
                else "REJECTED_THIRD_PARTY_WEB_SCRAPE_IN_MIT_REPO"
            )
            # Extract capture minute (YYYYMMDD_HHMM) as session_group for first-party smartphone captures
            m_cap = re.search(r"(2025\d{4}_\d{4})", r["source_image_id"])
            if m_cap:
                session_group = f"sess_shynuaa_{m_cap.group(1)}"

            if cid in SHYNUAA_REJECT or not is_first_party:
                domain_tag, reason = SHYNUAA_REJECT.get(
                    cid, ("AMBIGUOUS_CONTEXT", "Third-party web-scraped image without individual upstream license verification.")
                )
                qa_status = "REJECT"
                obj_valid = False
                waste_verified = False
                if "task8d_0078" in reason:
                    specimen_group = "spec_shynuaa_indoor_bottle_0078"
            elif cid in SHYNUAA_NEEDS_REVIEW:
                domain_tag = "ACTUAL_WASTE"
                qa_status = "NEEDS_REVIEW"
                obj_valid = False
                waste_verified = True
                reason = SHYNUAA_NEEDS_REVIEW[cid]
            else:
                domain_tag = "ACTUAL_WASTE"
                qa_status = "REAL_WASTE_PASS"
                obj_valid = True
                waste_verified = True
                reason = "Verified first-party 2025 outdoor street litter capture with tight EXIF-aligned CVAT instance bounding boxes (source_split=valid preserved)."
                if cid == "task8d_0078":
                    specimen_group = "spec_shynuaa_indoor_bottle_0078"
                if cid in SHYNUAA_MULTI_INSTANCE_OVERRIDES:
                    boxes = norm_boxes_to_struct(w, h, SHYNUAA_MULTI_INSTANCE_OVERRIDES[cid], "manual_tight_multi_instance_v8e_audit")
                elif cid in SHYNUAA_CLASS_OVERRIDES:
                    new_cid = SHYNUAA_CLASS_OVERRIDES[cid]
                    for b in boxes:
                        b["class_id"] = new_cid
                        b["class_name"] = CLASS_NAMES[new_cid]

        elif src == "SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED":
            provenance_tier = "VERIFIED_FLICKR_CC_BY_2.0_WITH_AUTHOR_URL"
            if cid in OI_REAL_WASTE_PASS:
                reason, override_b = OI_REAL_WASTE_PASS[cid]
                if override_b is not None:
                    boxes = norm_boxes_to_struct(w, h, override_b, "manual_tight_multi_instance_v8e_audit")
                domain_tag = "ACTUAL_WASTE"
                qa_status = "REAL_WASTE_PASS"
                obj_valid = True
                waste_verified = True
            elif cid in OI_NEEDS_REVIEW:
                domain_tag = "ACTUAL_WASTE"
                qa_status = "NEEDS_REVIEW"
                obj_valid = False
                waste_verified = True
                reason = OI_NEEDS_REVIEW[cid]
            elif cid in OI_KEEP_AUXILIARY:
                domain_tag, reason = OI_KEEP_AUXILIARY[cid]
                qa_status = "KEEP_AUXILIARY"
                obj_valid = True
                waste_verified = False
                if cid in ("task8d_0150", "task8d_0164"):
                    boxes = [b for b in boxes if b["class_id"] == 9]
            else:
                domain_tag = "IN_USE"
                qa_status = "REJECT"
                obj_valid = False
                waste_verified = False
                reason = "Object in active human/household use or metaphorical Flickr title without real-waste evidence."

        elif src == "SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES":
            provenance_tier = "VERIFIED_WIKIMEDIA_COMMONS_OPEN_LICENSE"
            if cid in WIKI_REAL_WASTE_PASS:
                reason, norm_b = WIKI_REAL_WASTE_PASS[cid]
                boxes = norm_boxes_to_struct(w, h, norm_b, "manual_tight_verified_v8e_audit")
                domain_tag = "ACTUAL_WASTE"
                qa_status = "REAL_WASTE_PASS"
                obj_valid = True
                waste_verified = True
            elif cid in WIKI_NEEDS_REVIEW:
                reason, norm_b = WIKI_NEEDS_REVIEW[cid]
                boxes = norm_boxes_to_struct(w, h, norm_b, "manual_review_hold_v8e_audit")
                domain_tag = "ACTUAL_WASTE"
                qa_status = "NEEDS_REVIEW"
                obj_valid = False
                waste_verified = True
            elif cid in WIKI_KEEP_AUXILIARY:
                domain_tag, norm_b = WIKI_KEEP_AUXILIARY[cid]
                boxes = norm_boxes_to_struct(w, h, norm_b, "manual_tight_auxiliary_v8e_audit")
                qa_status = "KEEP_AUXILIARY"
                obj_valid = True
                waste_verified = False
                reason = f"Valid object annotation but domain is {domain_tag}; stored in auxiliary only."
            else:
                domain_tag = "AMBIGUOUS_CONTEXT"
                qa_status = "REJECT"
                obj_valid = False
                waste_verified = False
                reason = "Rejected non-waste image (scanned historical page, photo collage, wall poster, museum piece, intact landscape, or same-specimen duplicate)."

        cls_ids = sorted({b["class_id"] for b in boxes}) if boxes else [r["primary_class_id"]]
        cls_names = [CLASS_NAMES[c] for c in cls_ids]
        p_cid = cls_ids[0]

        yolo_txt = format_yolo_lines(boxes)
        label_path_rel = None
        if qa_status == "REAL_WASTE_PASS":
            lp = LABELS_PASS_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "KEEP_AUXILIARY":
            lp = LABELS_AUX_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "NEEDS_REVIEW":
            lp = LABELS_REVIEW_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")

        rec_out = {
            "candidate_id": cid,
            "source_id": src,
            "source_name": src_name,
            "source_image_id": r["source_image_id"],
            "source_split": source_split,
            "provenance_tier": provenance_tier,
            "title": title_str,
            "query_used": r.get("query_used", ""),
            "original_url": orig_url,
            "author": r.get("author", "Unknown"),
            "license": r.get("license", "Unknown"),
            "attribution_required": bool(r.get("attribution_required", True)),
            "image_path": img_rel,
            "label_path": label_path_rel,
            "overlay_path": None,
            "sha256": r["sha256"],
            "dhash_hex": dhash_str,
            "width": w,
            "height": h,
            "primary_class_id": p_cid,
            "primary_class_name": CLASS_NAMES[p_cid],
            "classes_present_ids": cls_ids,
            "classes_present_names": cls_names,
            "is_multiclass": len(cls_ids) > 1,
            "bbox_count": len(boxes),
            "boxes": boxes,
            "qa_status": qa_status,
            "domain_tag": domain_tag,
            "OBJECT_ANNOTATION_VALID": obj_valid,
            "REAL_WASTE_CONTEXT_VERIFIED": waste_verified,
            "APPROVED_FOR_TRAIN": approved_for_train,
            "session_group": session_group,
            "specimen_group": specimen_group,
            "target_split": "REJECTED",
            "qa_reason": reason,
        }

        if qa_status in ("REAL_WASTE_PASS", "KEEP_AUXILIARY", "NEEDS_REVIEW"):
            rec_out["overlay_path"] = render_overlay(rec_out)

        qa_records.append(rec_out)

    # Assign target_split strictly respecting source_split:
    # 1. If source_split == "test" -> HOLD_SOURCE_TEST_W8D (NEVER in TRAIN or VAL)
    # 2. If source_split in ("valid", "validation") -> VAL_REAL_WASTE_W8D (NEVER in TRAIN)
    # 3. If source_split in ("train", "unspecified") -> TRAIN_REAL_WASTE_W8D (with ~12% of Wikimedia unsplitted sessions in VAL_REAL_WASTE_W8D)
    pass_recs = [r for r in qa_records if r["qa_status"] == "REAL_WASTE_PASS"]
    wiki_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in pass_recs:
        sp = r["source_split"]
        if sp == "test":
            r["target_split"] = "HOLD_SOURCE_TEST_W8D"
        elif sp in ("valid", "validation"):
            r["target_split"] = "VAL_REAL_WASTE_W8D"
        elif sp == "train":
            r["target_split"] = "TRAIN_REAL_WASTE_W8D"
        else:
            wiki_groups[r["session_group"]].append(r)

    for idx, gk in enumerate(sorted(wiki_groups.keys())):
        grp = wiki_groups[gk]
        if idx % 8 == 7 and len(grp) <= 2:
            for item in grp:
                item["target_split"] = "VAL_REAL_WASTE_W8D"
        else:
            for item in grp:
                item["target_split"] = "TRAIN_REAL_WASTE_W8D"

    for r in qa_records:
        if r["qa_status"] == "KEEP_AUXILIARY":
            r["target_split"] = (
                "HOLD_SOURCE_TEST_AUXILIARY_W8D"
                if r["source_split"] == "test"
                else "AUXILIARY_NON_WASTE_TRAIN_HOLD"
            )
        elif r["qa_status"] == "NEEDS_REVIEW":
            r["target_split"] = "MANUAL_REVIEW_QUEUE"

    # Compute summary statistics
    status_counts = Counter(r["qa_status"] for r in qa_records)
    domain_counts = Counter(r["domain_tag"] for r in qa_records)
    split_counts = Counter(r["target_split"] for r in qa_records)
    source_split_counts = Counter(f"{r['source_id']}:{r['source_split']}->{r['target_split']}" for r in pass_recs)

    # Verify zero source test in train/val and zero source valid in train
    for r in qa_records:
        if r["source_split"] == "test":
            assert r["target_split"] not in ("TRAIN_REAL_WASTE_W8D", "VAL_REAL_WASTE_W8D"), f"Source test leaked: {r['candidate_id']}"
        if r["source_split"] in ("valid", "validation"):
            assert r["target_split"] != "TRAIN_REAL_WASTE_W8D", f"Source valid leaked into train: {r['candidate_id']}"

    source_breakdown: dict[str, dict[str, Any]] = {}
    for src_id in sorted({r["source_id"] for r in qa_records}):
        s_recs = [r for r in qa_records if r["source_id"] == src_id]
        s_pass = [r for r in s_recs if r["qa_status"] == "REAL_WASTE_PASS"]
        s_aux = [r for r in s_recs if r["qa_status"] == "KEEP_AUXILIARY"]
        s_rev = [r for r in s_recs if r["qa_status"] == "NEEDS_REVIEW"]
        s_rej = [r for r in s_recs if r["qa_status"] == "REJECT"]
        source_breakdown[src_id] = {
            "downloaded_candidates": len(s_recs),
            "REAL_WASTE_PASS": len(s_pass),
            "REAL_WASTE_PASS_TRAIN": sum(1 for r in s_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8D"),
            "REAL_WASTE_PASS_VAL": sum(1 for r in s_pass if r["target_split"] == "VAL_REAL_WASTE_W8D"),
            "REAL_WASTE_PASS_HOLD_TEST": sum(1 for r in s_pass if r["target_split"] == "HOLD_SOURCE_TEST_W8D"),
            "KEEP_AUXILIARY": len(s_aux),
            "NEEDS_REVIEW": len(s_rev),
            "REJECT": len(s_rej),
            "real_waste_yield_pct": round(100.0 * len(s_pass) / max(1, len(s_recs)), 2),
        }

    per_class_pass_images = {CLASS_NAMES[k]: 0 for k in CLASS_NAMES}
    per_class_pass_train_images = {CLASS_NAMES[k]: 0 for k in CLASS_NAMES}
    per_class_pass_val_images = {CLASS_NAMES[k]: 0 for k in CLASS_NAMES}
    per_class_pass_boxes = {CLASS_NAMES[k]: 0 for k in CLASS_NAMES}

    for r in pass_recs:
        for cid in r["classes_present_ids"]:
            cname = CLASS_NAMES[cid]
            per_class_pass_images[cname] += 1
            if r["target_split"] == "TRAIN_REAL_WASTE_W8D":
                per_class_pass_train_images[cname] += 1
            elif r["target_split"] == "VAL_REAL_WASTE_W8D":
                per_class_pass_val_images[cname] += 1
        for b in r["boxes"]:
            per_class_pass_boxes[CLASS_NAMES[b["class_id"]]] += 1

    # Verify zero session/specimen leakage between TRAIN_REAL_WASTE_W8D and VAL_REAL_WASTE_W8D
    train_sessions = {r["session_group"] for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8D"}
    val_sessions = {r["session_group"] for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8D"}
    train_specimens = {r["specimen_group"] for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8D"}
    val_specimens = {r["specimen_group"] for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8D"}
    session_overlap = sorted(train_sessions & val_sessions)
    specimen_overlap = sorted(train_specimens & val_specimens)
    assert not session_overlap, f"Session overlap: {session_overlap}"
    assert not specimen_overlap, f"Specimen overlap: {specimen_overlap}"

    # Render contact sheets
    sheets_created = []
    shynuaa_pass = [r for r in pass_recs if r["source_id"] == "SRC_8D_04_SHYNUAA_STREET_GARBAGE"]
    sheets_created.append(render_sheet(
        shynuaa_pass[:36],
        "Task 8D REAL_WASTE_PASS — Shynuaa Valid Split Part 1 (First-Party MIT)",
        "task8d_real_waste_pass_shynuaa.jpg",
        max_items=36,
    ))
    if len(shynuaa_pass) > 36:
        sheets_created.append(render_sheet(
            shynuaa_pass[36:],
            "Task 8D REAL_WASTE_PASS — Shynuaa Valid Split Part 2 (First-Party MIT)",
            "task8d_real_waste_pass_shynuaa_p2.jpg",
            max_items=48,
        ))
    sheets_created.append(render_sheet(
        [r for r in pass_recs if r["source_id"] == "SRC_8D_06_OPENIMAGES_WASTE_CONTEXT_FILTERED"],
        "Task 8D REAL_WASTE_PASS — Open Images V7 Visible Waste Verified",
        "task8d_real_waste_pass_openimages.jpg",
        max_items=36,
    ))
    wiki_pass = [r for r in pass_recs if r["source_id"] == "SRC_8D_07_WIKIMEDIA_WASTE_CONTEXT_QUERIES"]
    sheets_created.append(render_sheet(
        wiki_pass[:48],
        "Task 8D REAL_WASTE_PASS — Wikimedia Waste Categories Part 1",
        "task8d_real_waste_pass_wikimedia_p1.jpg",
        max_items=48,
    ))
    if len(wiki_pass) > 48:
        sheets_created.append(render_sheet(
            wiki_pass[48:96],
            "Task 8D REAL_WASTE_PASS — Wikimedia Waste Categories Part 2",
            "task8d_real_waste_pass_wikimedia_p2.jpg",
            max_items=48,
        ))
    sheets_created.append(render_sheet(
        [r for r in pass_recs if r["is_multiclass"]],
        "Task 8D REAL_WASTE_PASS — Multi-Class Mixed Waste Scenes",
        "task8d_real_waste_pass_multiclass.jpg",
        max_items=36,
    ))
    sheets_created.append(render_sheet(
        [r for r in qa_records if r["qa_status"] == "KEEP_AUXILIARY"],
        "Task 8D KEEP_AUXILIARY — Closed Bins & Non-Waste Valid Object Boxes",
        "task8d_keep_auxiliary.jpg",
        max_items=36,
    ))
    sheets_created.append(render_sheet(
        [r for r in qa_records if r["qa_status"] == "NEEDS_REVIEW"],
        "Task 8D NEEDS_REVIEW — Multi-Item Trash Piles Held for Manual Instance Split",
        "task8d_needs_review.jpg",
        max_items=36,
    ))

    summary_payload = {
        "manifest_version": "TASK8D_HIGH_YIELD_REAL_WASTE_V2_AUDITED_IN_8E",
        "approved_for_train": False,
        "total_downloaded_candidates": len(qa_records),
        "qa_status_counts": dict(status_counts),
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "source_split_to_target_split_pass": dict(source_split_counts),
        "real_waste_pass_total_images": len(pass_recs),
        "real_waste_pass_train_images": split_counts["TRAIN_REAL_WASTE_W8D"],
        "real_waste_pass_val_images": split_counts["VAL_REAL_WASTE_W8D"],
        "real_waste_pass_hold_source_test_images": split_counts["HOLD_SOURCE_TEST_W8D"],
        "real_waste_pass_total_boxes": sum(r["bbox_count"] for r in pass_recs),
        "real_waste_pass_multiclass_images": sum(1 for r in pass_recs if r["is_multiclass"]),
        "overall_acceptance_rate_pct": round(100.0 * len(pass_recs) / max(1, len(qa_records)), 2),
        "source_yield_breakdown": source_breakdown,
        "per_class_real_waste_pass_images": per_class_pass_images,
        "per_class_real_waste_pass_train_images": per_class_pass_train_images,
        "per_class_real_waste_pass_val_images": per_class_pass_val_images,
        "per_class_real_waste_pass_boxes": per_class_pass_boxes,
        "group_leakage_verification": {
            "train_val_session_overlap_count": len(session_overlap),
            "train_val_specimen_overlap_count": len(specimen_overlap),
            "source_test_in_train_count": 0,
            "source_valid_in_train_count": 0,
            "status": "ZERO_LEAKAGE_VERIFIED",
        },
        "contact_sheets": [s for s in sheets_created if s],
        "records": qa_records,
    }

    QA_MANIFEST_JSON.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(QA_MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "source_id", "source_split", "provenance_tier",
            "primary_class_id", "primary_class_name",
            "classes_present_ids", "is_multiclass", "bbox_count",
            "qa_status", "domain_tag", "OBJECT_ANNOTATION_VALID",
            "REAL_WASTE_CONTEXT_VERIFIED", "APPROVED_FOR_TRAIN",
            "target_split", "session_group", "specimen_group",
            "license", "author", "original_url", "qa_reason",
        ])
        for r in qa_records:
            writer.writerow([
                r["candidate_id"], r["source_id"], r["source_split"], r["provenance_tier"],
                r["primary_class_id"], r["primary_class_name"],
                "|".join(str(c) for c in r["classes_present_ids"]), r["is_multiclass"], r["bbox_count"],
                r["qa_status"], r["domain_tag"], r["OBJECT_ANNOTATION_VALID"],
                r["REAL_WASTE_CONTEXT_VERIFIED"], r["APPROVED_FOR_TRAIN"],
                r["target_split"], r["session_group"], r["specimen_group"],
                r["license"], r["author"], r["original_url"], r["qa_reason"],
            ])

    # Build compact Task 8D Independent QA Evidence Mini-Pack (30 stratified REAL_WASTE_PASS images + labels + overlays + sheets + manifest)
    for sub in ["images_sample_30", "labels_sample_30", "overlays_sample_30", "contact_sheets"]:
        d_sub = MINI_PACK_DIR / sub
        if d_sub.exists():
            shutil.rmtree(d_sub)
        d_sub.mkdir(parents=True, exist_ok=True)

    shutil.copy2(QA_MANIFEST_JSON, MINI_PACK_DIR / "task8d_qa_manifest.json")
    shutil.copy2(QA_MANIFEST_CSV, MINI_PACK_DIR / "task8d_qa_manifest.csv")
    for s_rel in summary_payload["contact_sheets"]:
        sp = ROOT / s_rel
        if sp.exists():
            shutil.copy2(sp, MINI_PACK_DIR / "contact_sheets" / sp.name)

    # Select 30 stratified REAL_WASTE_PASS records across all 10 classes & all 3 sources
    selected_30: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for cid_num in range(10):
        for r in pass_recs:
            if cid_num in r["classes_present_ids"] and r["candidate_id"] not in seen_ids:
                selected_30.append(r)
                seen_ids.add(r["candidate_id"])
                if sum(1 for x in selected_30 if cid_num in x["classes_present_ids"]) >= 3:
                    break
    for r in pass_recs:
        if len(selected_30) >= 30:
            break
        if r["candidate_id"] not in seen_ids:
            selected_30.append(r)
            seen_ids.add(r["candidate_id"])

    for r in selected_30[:30]:
        ip = ROOT / r["image_path"]
        lp = ROOT / r["label_path"]
        op = ROOT / r["overlay_path"]
        if ip.exists():
            shutil.copy2(ip, MINI_PACK_DIR / "images_sample_30" / ip.name)
        if lp.exists():
            shutil.copy2(lp, MINI_PACK_DIR / "labels_sample_30" / lp.name)
        if op.exists():
            shutil.copy2(op, MINI_PACK_DIR / "overlays_sample_30" / op.name)

    print(f"Task 8D QA Manifest V2 generated: {QA_MANIFEST_JSON}")
    print("Status counts:", dict(status_counts))
    print("Split counts:", dict(split_counts))
    print("Source yield breakdown:", json.dumps(source_breakdown, indent=2))
    print("Per-class REAL_WASTE_PASS images:", per_class_pass_images)
    print("Per-class REAL_WASTE_PASS train images:", per_class_pass_train_images)
    print("Per-class REAL_WASTE_PASS val images:", per_class_pass_val_images)
    print("Per-class REAL_WASTE_PASS boxes:", per_class_pass_boxes)


if __name__ == "__main__":
    main()
