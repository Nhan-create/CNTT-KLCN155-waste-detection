# TASK 8 — WAVE 1 QA CORRECTION & FORENSIC AUDIT REPORT (`WAVE1_QA_CORRECTION_REPORT.md`)

- **Repository:** `.`
- **Source of Truth Manifest:** [`task8_wave1_qa_manifest.json`](../../data/acquisition_staging/task8a_multiclass_batch/task8_wave1_qa_manifest.json)
- **Task 8C Consistency Fix Applied:** Reconciled all table cells and bounding-box totals directly from `task8_wave1_qa_manifest.json` (**`104` `QA_PASS` unique images**, **`170` verified bounding boxes**: `93` boxes in `59` `REAL_WASTE_PASS` images + `77` boxes in `45` `KEEP_AUXILIARY` images).
- **Training Approval Gate:** `APPROVED_FOR_TRAIN = False` on all `221` records.

---

## 1. A1 — Forensic Re-Audit of Flagged Wave 1 Images (`task8a_082`, `104`, `116`, `109`, `110`, `111`, `112`)

| Item ID | Original Source Title | Manifest Status (`qa_decision`) | Proposed Split | `domain_tag` | Boxes | Forensic Finding & Action in Manifest |
|---|---|---|---|---|---:|---|
| `task8a_082` | `File:Moldy tomatoes.jpg` | `NEEDS_REVIEW` (`NEEDS_REVIEW`) | `MANUAL_REVIEW_HOLD` | `AMBIGUOUS_CONTEXT` | `4` | A1 QA Fix: Moved from QA_PASS to NEEDS_REVIEW — crate has ~25 tomatoes where only 3 show clear mold while others appear intact; single crate-wide biological box replaced with 3 candidate moldy-tomato boxes pending human review. |
| `task8a_104` | `File:2021-07-23 biowaste 01.jpg` | `NEEDS_REVIEW` (`NEEDS_REVIEW`) | `MANUAL_REVIEW_HOLD` | `ACTUAL_WASTE` | `3` | A1 QA Fix: Moved from QA_PASS to NEEDS_REVIEW — single full-image box [0.01,0.01,0.99,0.99] over a dense multi-item compost pile violates instance detection rules. |
| `task8a_116` | `File:10292023 pizza options at Wikidata Day 2023` | `NEEDS_REVIEW` (`NEEDS_REVIEW`) | `MANUAL_REVIEW_HOLD` | `IN_USE` | `3` | A1 QA Fix: Moved from QA_PASS to NEEDS_REVIEW and stripped biological boxes — provenance confirms 'pizza options at Wikidata Day 2023 at Butler Library' (catered event food IN_USE, not discarded biological waste). |
| `task8a_109` | `File:-Canada150 Coffee Cup (36449417690).jpg` | `REJECT` (`REJECT`) | `EXCLUDED` | `AMBIGUOUS_CONTEXT` | `0` | A1 QA Fix: Same photographer (Coastal Elite) & same Tim Hortons #Canada150 cup specimen as task8a_110/111; rejected to prevent specimen duplication and cross-split session leakage. |
| `task8a_110` | `File:-Canada150 Coffee Cup (36706385981).jpg` | `NEEDS_REVIEW` (`NEEDS_REVIEW`) | `MANUAL_REVIEW_HOLD` | `AMBIGUOUS_CONTEXT` | `3` | A1 QA Fix: Grouped into SPEC_W1_CANADA150_COFFEE_CUP_SESSION_109_110_111_112 and moved to NEEDS_REVIEW — upright cup on restaurant table (AMBIGUOUS_CONTEXT) & part of same Flickr upload batch as 109/111/112. |
| `task8a_111` | `File:-Canada150 Coffee Cup (36706387821).jpg` | `REJECT` (`REJECT`) | `EXCLUDED` | `AMBIGUOUS_CONTEXT` | `0` | A1 QA Fix: Same photographer & same Tim Hortons #Canada150 cup specimen as task8a_110. |
| `task8a_112` | `File:-Canada150 Coffee Cup (36706388951).jpg` | `NEEDS_REVIEW` (`NEEDS_REVIEW`) | `MANUAL_REVIEW_HOLD` | `AMBIGUOUS_CONTEXT` | `6` | A1 QA Fix: Grouped into SPEC_W1_CANADA150_COFFEE_CUP_SESSION_109_110_111_112 and moved to NEEDS_REVIEW — same photographer/series (#Canada150 Coffee Cup by Coastal Elite) and upright café table context. |

---

## 2. A2 & Task 8C Task 2 — Domain Breakdown & 3-Criteria Separation (`REAL_WASTE_PASS` vs `KEEP_AUXILIARY`)

| Wave 1 Subset | Unique Images | Total Verified Boxes | `ACTUAL_WASTE` (`REAL_WASTE_PASS`) | `PRODUCT_STUDIO` (`KEEP_AUXILIARY`) | `AMBIGUOUS_CONTEXT` (`KEEP_AUXILIARY`) | `IN_USE` |
|---|---:|---:|---:|---:|---:|---:|
| **`QA_PASS` (`OBJECT_ANNOTATION_VALID = True`)** | **104** | **170** | **59** (`93` boxes) | **31** | **14** | **0** |
| — `TRAIN_WAVE1` (`QA_PASS`) | **81** | **126** | **43** (`TRAIN_REAL_WASTE_W1`) | **26** (`AUXILIARY_TRAIN_W1`) | **12** (`AUXILIARY_TRAIN_W1`) | **0** |
| — `VAL_REAL_WAVE1` (`QA_PASS`) | **23** | **44** | **16** (`VAL_REAL_WASTE_W1`) | **5** (`AUXILIARY_HOLD_W1`) | **2** (`AUXILIARY_HOLD_W1`) | **0** |
| **`NEEDS_REVIEW` (Held)** | **30** | `46` *(unverified)* | 19 | 0 | 7 | 4 |
| **`REJECT` (Excluded)** | **87** | `0` | 30 | 2 | 32 | 23 |
| **All Evaluated Wave 1 (`221`)** | **221** | `216` | **108** | **33** | **53** | **27** |

---

## 3. A3 — License & Provenance Verification

- **Open Images V7 records in Wave 1:** `42`
- **Verified original Flickr Author resolved:** `42`
- **Upstream Author blank on Flickr but `OriginalLandingURL` preserved:** `0`
- **Synthetic contributor strings remaining:** **`0`**

---

## 4. A4 — Per-Class Wave 1 Breakdown (Directly Reconciled with `task8_wave1_qa_manifest.json`)

| Class ID & Name | `QA_PASS` Unique Images | `QA_PASS` Unique Specimens | `TRAIN_WAVE1` Images | `VAL_REAL_WAVE1` Images | **`QA_PASS` Boxes** | **`REAL_WASTE_PASS` (`ACTUAL_WASTE`)** | `PRODUCT_STUDIO` | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W1` | `NEEDS_REVIEW` Held |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | 3 | 2 | **2** | 1 | **5** | **0** (`0` tr / `0` val) | 3 | 0 | 2 | 0 |
| `1: biological` | 27 | 26 | **21** | 6 | **32** | **26** (`20` tr / `6` val) | 0 | 1 | 1 | 4 |
| `2: cardboard` | 13 | 13 | **11** | 2 | **20** | **2** (`1` tr / `1` val) | 5 | 6 | 10 | 10 |
| `3: clothes` | 6 | 6 | **5** | 1 | **9** | **1** (`1` tr / `0` val) | 5 | 0 | 4 | 1 |
| `4: glass` | 20 | 20 | **15** | 5 | **37** | **11** (`6` tr / `5` val) | 9 | 0 | 9 | 5 |
| `5: metal` | 2 | 2 | **1** | 1 | **6** | **1** (`0` tr / `1` val) | 1 | 0 | 1 | 2 |
| `6: paper` | 10 | 9 | **7** | 3 | **12** | **6** (`3` tr / `3` val) | 1 | 3 | 4 | 5 |
| `7: plastic` | 12 | 11 | **8** | 4 | **16** | **8** (`5` tr / `3` val) | 0 | 4 | 3 | 3 |
| `8: shoes` | 17 | 17 | **13** | 4 | **31** | **9** (`7` tr / `2` val) | 8 | 0 | 6 | 5 |
| `9: trash` | 2 | 2 | **2** | 0 | **2** | **2** (`2` tr / `0` val) | 0 | 0 | 0 | 0 |
| **Unique Image / Box Total** | **104** | **101** | **81** *(sum=85)* | **23** *(sum=27)* | **170** | **59** (`43` tr / `16` val) | **31** | **14** | **38** | **30** |
