# TASK 8C — REAL-WASTE COVERAGE (`REAL_WASTE_PASS`) & STANDARDIZED VALIDATION REPORT

## 1. Separation of Annotation Quality & Real-Waste Context (Task 2)

Every record in both Wave 1 (`221` records) and Wave 2 (`1,196` records) now carries three independent boolean gates:
1. `OBJECT_ANNOTATION_VALID`: True iff all target physical instances in the image have tight, verified bounding boxes and accurate 10-class taxonomy IDs.
2. `REAL_WASTE_CONTEXT_VERIFIED`: True iff `OBJECT_ANNOTATION_VALID == True` AND the image shows verified physical evidence of waste/disposal (`domain_tag == 'ACTUAL_WASTE'`).
3. `APPROVED_FOR_TRAIN`: Strictly `False` across all `1,417` Wave 1 + Wave 2 records pending PM/User approval.

| Batch | Total Evaluated | `REAL_WASTE_PASS` (`ACTUAL_WASTE`) | `REAL_WASTE_PASS` Train | `REAL_WASTE_PASS` Val (`VAL_REAL_WASTE`) | `KEEP_AUXILIARY` (`PRODUCT_STUDIO` / `IN_USE` / `AMBIGUOUS`) | `NEEDS_REVIEW` Held | `REJECT` Excluded |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Wave 1 (`task8a_multiclass_batch`)** | 221 | **59** (`93` boxes) | **43** (`TRAIN_REAL_WASTE_W1`) | **16** (`VAL_REAL_WASTE_W1`) | **45** (`77` boxes: `38` `AUXILIARY_TRAIN_W1` + `7` `AUXILIARY_HOLD_W1`) | 30 | 87 |
| **Wave 2 (`task8_wave2_large_batch`)** | 1,196 | **41** (`78` boxes) | **30** (`TRAIN_REAL_WASTE_W2`) | **11** (`VAL_REAL_WASTE_W2`) | **166** (`270` boxes: `AUXILIARY_TRAIN_W2`) | 925 | 64 |
| **Combined Wave 1 + Wave 2** | **1,417** | **100** (`171` boxes) | **73** (`123` boxes) | **27** (`48` boxes) | **211** (`347` boxes) | **955** | **151** |

---

## 2. Per-Class Verified Real-Waste (`REAL_WASTE_PASS`) vs Auxiliary (`KEEP_AUXILIARY`) Coverage

> **Strict Accounting Rule:** `KEEP_AUXILIARY` images (studio product shots, intact objects, or `AMBIGUOUS_CONTEXT` items) are stored separately in `labels_auxiliary/` and are **NEVER** counted as real-waste coverage or admitted into `VAL_REAL_WASTE`.

| Class ID & Name | W1 Real-Waste Train | W1 Real-Waste Val | W1 Real-Waste Boxes | W2 Real-Waste Train | W2 Real-Waste Val | W2 Real-Waste Boxes | **Total New Real-Waste Train** | **Total New Real-Waste Val** | **Total New Real-Waste Boxes** | `KEEP_AUXILIARY` Train (Separate) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | 0 | 0 | 0 | **4** | **3** | **15** | **4** | **3** | **15** | 30 (`2` W1 + `28` W2) |
| `1: biological` | 20 | 6 | 31 | **9** | **2** | **13** | **29** | **8** | **44** | 10 (`1` W1 + `9` W2) |
| `2: cardboard` | 1 | 1 | 2 | **3** | **1** | **6** | **4** | **2** | **8** | 34 (`10` W1 + `24` W2) |
| `3: clothes` | 1 | 0 | 1 | **1** | **0** | **1** | **2** | **0** | **2** | 11 (`4` W1 + `7` W2) |
| `4: glass` | 6 | 5 | 16 | **2** | **1** | **3** | **8** | **6** | **19** | 21 (`9` W1 + `12` W2) |
| `5: metal` | 0 | 1 | 5 | **6** | **2** | **8** | **6** | **3** | **13** | 54 (`1` W1 + `53` W2) |
| `6: paper` | 3 | 3 | 8 | **0** | **2** | **2** | **3** | **5** | **10** | 29 (`4` W1 + `25` W2) |
| `7: plastic` | 5 | 3 | 10 | **2** | **0** | **3** | **7** | **3** | **13** | 3 (`3` W1 + `0` W2) |
| `8: shoes` | 7 | 2 | 18 | **4** | **1** | **10** | **11** | **3** | **28** | 18 (`6` W1 + `12` W2) |
| `9: trash` | 2 | 0 | 2 | **5** | **1** | **10** | **7** | **1** | **12** | 0 (`0` W1 + `0` W2) |

---

## 3. Standardized Real-Waste Validation Policy & Open Images Split Hygiene (Task 5)

1. **100% Real-Waste Validation (`VAL_REAL_WASTE_W1` + `VAL_REAL_WASTE_W2` = `27` unique images, `48` boxes):**
   - Every single image in `VAL_REAL_WASTE_W1` (`16` images) and `VAL_REAL_WASTE_W2` (`11` images) has `OBJECT_ANNOTATION_VALID = True`, `REAL_WASTE_CONTEXT_VERIFIED = True`, and `domain_tag = ACTUAL_WASTE`.
   - The `7` non-waste Wave 1 images (`5` `PRODUCT_STUDIO` + `2` `AMBIGUOUS_CONTEXT`) and all `35` non-waste Wave 2 images formerly in `VAL_REAL_WAVE2` have been removed from validation and placed in `AUXILIARY_HOLD_W1` / `AUXILIARY_TRAIN_W2`.
2. **Zero Session & Specimen Leakage:**
   - `VAL_REAL_WASTE_W1` (`16` images) shares **`0`** `specimen_group` IDs with `TRAIN_REAL_WASTE_W1` / `AUXILIARY_TRAIN_W1`.
   - `VAL_REAL_WASTE_W2` (`11` images) shares **`0`** `session_group` and **`0`** `specimen_group` IDs with `TRAIN_REAL_WASTE_W2` / `AUXILIARY_TRAIN_W2`.
3. **Explicit Policy for Open Images V7 Test & Validation Split Origins:**
   - Images originating from `openimages_v7_test` or `openimages_v7_validation` are **preferentially routed to `VAL_REAL_WASTE_W2`** when verified as `REAL_WASTE_PASS` (`ACTUAL_WASTE`), unless their photographer `session_group` already has images in train (in which case the entire session group is kept together to prevent cross-split photographer leakage). They are never mixed across splits and never touch the existing frozen project benchmarks.
4. **Frozen Test & Holdout Isolation:**
   - Existing frozen test sets (`299` pre-audit / `274` sanitized candidate in `data/Unified_Dataset/test`) and the 27-image external holdout (`data/holdout`) remain 100% untouched and are never used to tune data selection or thresholds.

---

## 4. Complete List of All 41 Wave 2 `REAL_WASTE_PASS` Images

- **Visual Contact Sheet:** [`task8c_wave2_all_real_waste_pass_after.jpg`](../../data/acquisition_staging/task8_wave2_large_batch/contact_sheets_task8c/task8c_wave2_all_real_waste_pass_after.jpg)

| Item ID | Source Family | Verified Classes | Boxes | Split | Session Group | Visual Evidence of Real Waste |
|---|---|---|---:|---|---|---|
| `task8w2_0046` | `openimages_v7_train` | `1:biological` | 2 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_10417E6E` | Salvaged from NEEDS_REVIEW: 2 blackened, overripe decaying bananas on wooden surface (ACTUAL_WASTE). |
| `task8w2_0071` | `openimages_v7_train` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_14A9F494` | task8c_reaudit_175_real_waste_verified: Discarded peeled banana skin with wine cork on dark surface (ACTUAL_WASTE). |
| `task8w2_0083` | `openimages_v7_train` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_1E50BC2D` | Salvaged from NEEDS_REVIEW: Single oxidized, browned carved banana (ACTUAL_WASTE). |
| `task8w2_0110` | `openimages_v7_train` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_61874BB6` | task8c_reaudit_175_real_waste_verified: Overripe blackened decaying banana (ACTUAL_WASTE). |
| `task8w2_0271` | `openimages_v7_train` | `2:cardboard` | 3 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_C5E83438` | task8c_reaudit_175_real_waste_verified: 3 discarded empty/flattened corrugated shipping boxes stacked in pile (ACTUAL_WASTE). |
| `task8w2_0381` | `openimages_v7_train` | `3:clothes` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_E8700D16` | task8c_reaudit_175_real_waste_verified: Single lost/discarded black winter glove outdoors (ACTUAL_WASTE). |
| `task8w2_0560` | `openimages_v7_train` | `5:metal` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_7428D585` | task8c_reaudit_175_real_waste_verified: Crushed Coca-Cola aluminum can lying on concrete pavement (ACTUAL_WASTE). |
| `task8w2_0584` | `openimages_v7_train` | `5:metal` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_EF8107AC` | task8c_reaudit_175_real_waste_verified: Rusted, weathered Castrol metal oil tin lying on dirt/grass (ACTUAL_WASTE). |
| `task8w2_0609` | `openimages_v7_train` | `5:metal` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_88A14ACB` | task8c_reaudit_175_real_waste_verified: Crushed, dented Diet Coke aluminum can on ground (ACTUAL_WASTE). |
| `task8w2_0663` | `openimages_v7_train` | `4:glass,5:metal` | 2 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_683F62BD` | Salvaged from NEEDS_REVIEW: Multi-class outdoor street litter with crushed Budweiser can (5: metal) and discarded Smirnoff Ice bottle (4: glass). |
| `task8w2_0893` | `openimages_v7_train` | `8:shoes` | 2 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_1A2A7489` | task8c_reaudit_175_real_waste_verified: Heavily worn, dirty old Converse sneakers on floor (ACTUAL_WASTE). |
| `task8w2_0908` | `openimages_v7_train` | `8:shoes` | 2 | `VAL_REAL_WASTE_W2` | `SESSION_W2_1F0B3475` | task8c_reaudit_175_real_waste_verified: Worn-out, scuffed old black combat boots on floor (ACTUAL_WASTE). |
| `task8w2_0931` | `openimages_v7_train` | `8:shoes` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_08C8F3A9` | task8c_reaudit_175_real_waste_verified: Old muddy rubber boot hanging on weathered shed door (ACTUAL_WASTE). |
| `task8w2_1179` | `wikimedia_commons_wave2` | `0:battery` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_16F8A443` | Salvaged from NEEDS_REVIEW: Single worn, scratched Varta AA battery cell standing upright (ACTUAL_WASTE). |
| `task8w2_1198` | `wikimedia_commons_wave2` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_9BBFC90F` | Salvaged from NEEDS_REVIEW: Abandoned rotting whole cauliflower head lying on street pavement (ACTUAL_WASTE). |
| `task8w2_1203` | `wikimedia_commons_wave2` | `1:biological` | 1 | `VAL_REAL_WASTE_W2` | `SESSION_W2_7B125166` | Salvaged from NEEDS_REVIEW: Discarded orange/citrus peel lying on dead twigs outdoors (ACTUAL_WASTE). |
| `task8w2_1218` | `wikimedia_commons_wave2` | `2:cardboard` | 1 | `VAL_REAL_WASTE_W2` | `SESSION_W2_FE870ECA` | Salvaged from NEEDS_REVIEW: Used empty corrugated banana produce box (2: cardboard, ACTUAL_WASTE). |
| `task8w2_1220` | `wikimedia_commons_wave2` | `2:cardboard` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_1885B1A1` | Salvaged from NEEDS_REVIEW: Flattened corrugated cardboard book packaging mailer on floor (2: cardboard, ACTUAL_WASTE). |
| `task8w2_1246` | `wikimedia_commons_wave2` | `6:paper` | 1 | `VAL_REAL_WASTE_W2` | `SESSION_W2_9B9B8817` | Salvaged from NEEDS_REVIEW: Stack of discarded newspapers/waste paper (Altpapier) in recycling container (6: paper, ACTUAL_WASTE). |
| `task8w2_1247` | `wikimedia_commons_wave2` | `1:biological,6:paper` | 2 | `VAL_REAL_WASTE_W2` | `SESSION_W2_11217B89` | Salvaged from NEEDS_REVIEW: Multi-class discarded paper food tray (6: paper) with leftover food scraps (1: biological) on grass. |
| `task8w2_1252` | `wikimedia_commons_wave2` | `8:shoes` | 4 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_FC2FD1DB` | Salvaged from NEEDS_REVIEW: 4 old abandoned hiking shoes/boots left on forest ground, all 4 instances boxed (8: shoes, ACTUAL_WASTE). |
| `task8w2_1253` | `wikimedia_commons_wave2` | `1:biological,4:glass,7:plastic,8:shoes` | 4 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_2149D350` | Salvaged from NEEDS_REVIEW: 4-class outdoor bus-stop litter scene with green glass bottle (4: glass), plastic bowl (7: plastic), food waste (1: biological), and abandoned slipper (8: shoes). |
| `task8w2_1256` | `wikimedia_commons_wave2` | `9:trash` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_912C00F9` | Salvaged from NEEDS_REVIEW: Dirty discarded trash bag being picked up outdoors (9: trash, ACTUAL_WASTE). |
| `task8w2_1268` | `wikimedia_commons_wave2` | `9:trash` | 3 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_1ACA382F` | Salvaged from NEEDS_REVIEW: 3 used cigarette butts standing in ash, all 3 instances tightly boxed (9: trash, ACTUAL_WASTE). |
| `task8w2_1281` | `wikimedia_commons_wave2` | `0:battery` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_B633BB7B` | Salvaged from NEEDS_REVIEW: Weathered, worn vintage Eveready No. 460 45V battery (ACTUAL_WASTE). |
| `task8w2_1294` | `wikimedia_commons_wave2` | `0:battery` | 4 | `VAL_REAL_WASTE_W2` | `SESSION_W2_EB871385` | Salvaged from NEEDS_REVIEW: 4 spent AA alkaline batteries lined up for disposal, all 4 instances boxed (ACTUAL_WASTE). |
| `task8w2_1298` | `wikimedia_commons_wave2` | `0:battery` | 2 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_D643C0F0` | Salvaged from NEEDS_REVIEW: 2 used Varta Longlife AAA batteries on desk, both instances tightly boxed (ACTUAL_WASTE). |
| `task8w2_1310` | `wikimedia_commons_wave2` | `0:battery` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_426DAC2A` | Salvaged from NEEDS_REVIEW: Single used, surface-scratched button cell battery on wooden table (ACTUAL_WASTE). |
| `task8w2_1327` | `wikimedia_commons_wave2` | `0:battery` | 3 | `VAL_REAL_WASTE_W2` | `SESSION_W2_0753BB19` | Salvaged from NEEDS_REVIEW: 3 used AAA alkaline batteries lying on grid mat, all 3 instances boxed (ACTUAL_WASTE). |
| `task8w2_1328` | `wikimedia_commons_wave2` | `0:battery` | 3 | `VAL_REAL_WASTE_W2` | `SESSION_W2_266F21E6` | task8c_reaudit_175_real_waste_verified: 3 corroded, cut-open Universal Alkaline batteries showing internal chemical decay (ACTUAL_WASTE). |
| `task8w2_1335` | `wikimedia_commons_wave2` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_227F8293` | task8c_reaudit_175_real_waste_verified: Discarded half loaf of bread sitting on top of a public street waste bin in Meidling (ACTUAL_WASTE). |
| `task8w2_1336` | `wikimedia_commons_wave2` | `1:biological,7:plastic` | 4 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_1FB27726` | Salvaged from NEEDS_REVIEW: Multi-class discarded bread loaves (1: biological) inside 2 plastic bags (7: plastic) hung on street fence. |
| `task8w2_1339` | `wikimedia_commons_wave2` | `1:biological` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_9502F579` | Salvaged from NEEDS_REVIEW: Discarded half-eaten bread roll (Bulka) lying on outdoor rock (ACTUAL_WASTE). |
| `task8w2_1353` | `wikimedia_commons_wave2` | `2:cardboard,9:trash` | 2 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_3524A727` | Salvaged from NEEDS_REVIEW: Multi-class street fly-tipping scene with discarded cardboard box (2: cardboard) and dumped stained mattress (9: trash). |
| `task8w2_1381` | `wikimedia_commons_wave2` | `5:metal` | 1 | `VAL_REAL_WASTE_W2` | `SESSION_W2_195E799B` | Salvaged from NEEDS_REVIEW: Pinched, dented Pepsi aluminum beverage can (5: metal, ACTUAL_WASTE). |
| `task8w2_1382` | `wikimedia_commons_wave2` | `5:metal` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_ED2CF132` | Salvaged from NEEDS_REVIEW: Crushed/squeezed aluminum beverage can (5: metal, ACTUAL_WASTE). |
| `task8w2_1384` | `wikimedia_commons_wave2` | `5:metal` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_D9FD9966` | Salvaged from NEEDS_REVIEW: Shredded, torn aluminum can litter lying on dirt and twigs outdoors (5: metal, ACTUAL_WASTE). |
| `task8w2_1400` | `wikimedia_commons_wave2` | `5:metal` | 1 | `VAL_REAL_WASTE_W2` | `SESSION_W2_4F6C9A6A` | Salvaged from NEEDS_REVIEW: Crumpled ball of discarded aluminum foil on gravel, reclassified from 9:trash to 5:metal (ACTUAL_WASTE). |
| `task8w2_1401` | `wikimedia_commons_wave2` | `9:trash` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_4AF6C804` | Salvaged from NEEDS_REVIEW: Shredded burst latex balloon litter lying on asphalt (9: trash, ACTUAL_WASTE). |
| `task8w2_1404` | `wikimedia_commons_wave2` | `4:glass,9:trash` | 4 | `VAL_REAL_WASTE_W2` | `SESSION_W2_D10C8234` | Salvaged from NEEDS_REVIEW: Multi-class glass ashtray (4: glass) containing 3 used cigarette butts (9: trash), all 4 instances boxed. |
| `task8w2_1405` | `wikimedia_commons_wave2` | `9:trash` | 1 | `TRAIN_REAL_WASTE_W2` | `SESSION_W2_0F91FAE4` | Salvaged from NEEDS_REVIEW: Single discarded cigarette butt lying on outdoor gravel (9: trash, ACTUAL_WASTE). |
