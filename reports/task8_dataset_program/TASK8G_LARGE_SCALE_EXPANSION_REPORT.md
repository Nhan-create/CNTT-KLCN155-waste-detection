# TASK 8G — LARGE-SCALE REAL-WASTE DATA EXPANSION REPORT

## 1. Task 0 — Regression Case 3 Discrepancy Fix
- Fixed the Regression Case 3 coordinate/IoU reporting in `TASK8F_FINAL_SAFETY_SIGN_OFF_REPORT.{md,json}` and `tests/detection/test_task8_manifest_report_consistency.py` without modifying the Task 8F dataset:
  - **Case 3a (Contained cross-class overlap in `task8f_0694`):** `(2:cardboard, 0.210, 0.030, 0.830, 0.940)` and `(1:biological, 0.390, 0.560, 0.540, 0.720)` -> exact `IoU = 0.042538`, `VALID_CROSS_CLASS_OVERLAP`, `keep_both = True`.
  - **Case 3b (High-IoU cross-class overlap):** `(4:glass, 0.200, 0.200, 0.700, 0.700)` and `(6:paper, 0.210, 0.210, 0.690, 0.690)` -> exact `IoU = 0.921600`, `VALID_CROSS_CLASS_OVERLAP`, `keep_both = True`.

## 2. Task 1 & Task 2 — Large-Scale Checkpointed Acquisition & Source Pilots
- **Checkpointed Raw Pool:** `920` brand-new unique deduplicated candidates (`0` SHA-256 or dHash collisions with Baseline V2/V3 or Waves 1, 2, 8D, 8E, 8F).
- **Verified `REAL_WASTE_PASS` Yield:** **`171`** images (**`160`** `TRAIN_REAL_WASTE_W8G`, **`11`** `VAL_REAL_WASTE_W8G`, **`291`** tight boxes, **`52`** multi-class images).
- **Independent Capture Diversity (`REAL_WASTE_PASS`):** `171` unique specimens across `71` independent capture sessions (`160` train specimens, `62` train sessions).

### Source Pilot Decisions (`30` samples/source before scale-up)
| Pilot ID | Source | Sampled | Pilot Pass Rate | Decision | Reason |
|---|---|---:|---:|---|---|
| `PILOT_8G_01` | `Shynuaa/Street-garbage Remaining First-Party 2025 Captures (`yolo/valid` + `yolo/train`)` | `30` | `93.3%` | **`EXPAND_APPROVED`** | Exhausts all remaining first-party 2025 street-litter captures with multi-object/multi-class annotations. |
| `PILOT_8G_02` | `dmedhi/garbage-image-classification-detection (HuggingFace / Roboflow Universe `student-utr07/garbage-classifier-oehkt`)` | `30` | `26.7%` | **`STOPPED_AFTER_PILOT`** | Stopped after 30-row pilot: rows 10 and 13 have `[0.0, 0.0, 640.0, 640.0]` full-frame boxes and isolated studio backgrounds. |
| `PILOT_8G_03` | `Akhila-9849/trash_detection (HuggingFace / Roboflow Universe `trash-detection-6r6qe`)` | `30` | `50.0%` | **`STOPPED_AFTER_PILOT`** | Stopped after 30-image pilot: missing bounding-box labels and unverified upstream provenance. |
| `PILOT_8G_04` | `Voxel51/food-waste-dataset (HuggingFace FiftyOne Meal Nutrition Dataset)` | `30` | `20.0%` | **`STOPPED_AFTER_PILOT`** | Stopped after 30-image pilot: predominantly served cafeteria meals on trays (`IN_USE`) with machine-generated labels. |
| `PILOT_8G_05` | `Wikimedia Commons Large-Scale Scarce-Class & Multi-Class Real-Waste Categories (80+ Queries)` | `40` | `62.5%` | **`EXPAND_APPROVED_WITH_VISUAL_QA_GATE`** | High-yield source for scarce classes `0: battery`, `1: biological`, `3: clothes`, `8: shoes`, `4: glass`, `2: cardboard`, `6: paper`, `5: metal` and multi-class street/beach litter. |
| `PILOT_8G_06` | `Open Images V7 Scarce-Class & Multi-Class `train` / `validation` Expansion (`0` from `test`)` | `40` | `47.5%` | **`EXPAND_APPROVED_WITH_VISUAL_DOMAIN_GATE`** | Provides human-verified multi-instance and multi-class ground-truth bounding boxes across all 8 scarce classes. |

### Source-by-Source Acquisition & QA Breakdown
| Source ID | Downloaded | `REAL_WASTE_PASS` | Train (`W8G`) | Val (`W8G`) | Boxes (`PASS`) | Multi-Class (`PASS`) | `KEEP_AUXILIARY` | `NEEDS_REVIEW` | `REJECT` | Licenses |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SRC_8G_01_SHYNUAA_REMAINING_FIRST_PARTY` | `168` | **`137`** | `137` | `0` | `251` | `52` | `0` | `0` | `31` | `MIT` |
| `SRC_8G_02_WIKIMEDIA_LARGE_SCALE_REAL_WASTE` | `425` | **`34`** | `23` | `11` | `40` | `0` | `247` | `68` | `76` | `CC BY 2.0, CC BY 2.5, CC BY 3.0` |
| `SRC_8G_03_OPENIMAGES_SCARCE_AND_MULTICLASS` | `327` | **`0`** | `0` | `0` | `0` | `0` | `279` | `39` | `9` | `https://creativecommons.org/licenses/by/2.0/` |

## 3. Task 3 — Quality Control & Safety Gates
- **All records `APPROVED_FOR_TRAIN = False`:** `True` (`920/920`)
- **Zero unannotated waste images used as hard negatives:** `True`
- **Remaining `DUPLICATE_ANNOTATION` pairs (`IoU >= 0.90` & `max_edge_delta <= 0.045`):** `0`
- **Preserved `VALID_SAME_CLASS_OVERLAP` pairs (never blindly deleted):** `12`
- **Preserved `VALID_CROSS_CLASS_OVERLAP` pairs (`agnostic_nms=False`):** `157`
- **Records with Label/Class/Bbox Refinements:** `217` (`23.6%`, `317` total refinements)
- **`ACTUAL_WASTE` Domain Rate:** `25.1%` (`231/920`)
- **Train/Val Session & Specimen Leakage:** `0`

## 4. Task 4 — Per-Class Milestone Tracking (`M1 = 500` & `M4 = 5,000`)
| Class ID | Class Name | Wave 8G `PASS` Total | Wave 8G Train (`+`) | Wave 8G Val (`+`) | Wave 8G Boxes | Cumulative Verified RW Train | Cumulative RW Boxes | Unique Specimens | Unique Sessions | Verified Val Images | Gap to `M1=500` | Gap to `M4=5,000` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `battery` | `2` | **`+1`** | `+1` | `5` | **`52`** | `120` | `48` | `49` | `23` | `448` | `4948` |
| `1` | `biological` | `19` | **`+16`** | `+3` | `26` | **`90`** | `114` | `83` | `75` | `20` | `410` | `4910` |
| `2` | `cardboard` | `20` | **`+20`** | `+0` | `23` | **`185`** | `253` | `185` | `58` | `44` | `315` | `4815` |
| `3` | `clothes` | `3` | **`+3`** | `+0` | `4` | **`34`** | `36` | `34` | `19` | `1` | `466` | `4966` |
| `4` | `glass` | `11` | **`+10`** | `+1` | `12` | **`90`** | `143` | `90` | `35` | `27` | `410` | `4910` |
| `5` | `metal` | `59` | **`+57`** | `+2` | `80` | **`301`** | `466` | `301` | `89` | `56` | `199` | `4699` |
| `6` | `paper` | `36` | **`+35`** | `+1` | `46` | **`201`** | `250` | `200` | `71` | `47` | `299` | `4799` |
| `7` | `plastic` | `73` | **`+71`** | `+2` | `78` | **`985`** | `1602` | `984` | `159` | `177` | `0` | `4015` |
| `8` | `shoes` | `4` | **`+3`** | `+1` | `4` | **`55`** | `84` | `55` | `46` | `11` | `445` | `4945` |
| `9` | `trash` | `10` | **`+10`** | `+0` | `13` | **`361`** | `655` | `360` | `96` | `72` | `139` | `4639` |
