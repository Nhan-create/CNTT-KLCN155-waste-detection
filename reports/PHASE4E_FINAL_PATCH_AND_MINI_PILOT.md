# ANTIGRAVITY — PHASE 4E: FINAL PATCH & REAL IMAGE MINI-PILOT REPORT

**Repository:** `D:\CNTT-KLCN155-waste-detection`  
**Execution Timestamp:** 2026-10-12  
**Status:** TASK 1 COMPLETED (100% REGRESSION TESTS PASSED) | TASK 2 MINI-PILOT EXECUTED & DATA QA COMPLETED

---

## 1. EXECUTIVE SUMMARY & THREE STRICT METRICS

In strict accordance with reviewer directives for Phase 4E:
- **Zero synthetic/placeholder data:** No fake Roboflow ZIP export was manufactured or ingested.
- **Fail-closed execution:** Exactly the 4 shortlisted Wikimedia Commons candidates were downloaded under user approval within the 20 MiB budget (10.22 MiB used).
- **Physical inspection before annotation:** No candidate was auto-approved. 100% full-frame inspection was performed on physical pixels.
- **Zero premature label files:** Zero 0-byte label files were written to disk prior to human label review.

```text
======================================================================
THREE MANDATORY METRICS:
- REAL NEW IMAGES DOWNLOADED = 4 (10,714,787 bytes ~ 10.22 MiB <= 20 MiB cap)
- VISUALLY QA PASSED         = 2 (wiki_bio_banana_peel_ground, wiki_neg_junction_boxes)
- APPROVED FOR TRAIN         = 0 (STRICT ZERO AUTO-APPROVAL; awaiting human review)
======================================================================
```

---

## 2. TASK 1: ROBOFLOW INTAKE REFINEMENTS (`scripts/ingest_roboflow_battery_export.py`)

### 2.1 Issues Identified & Resolved

| Bug / Vulnerability | Previous Behavior | Patched Behavior (Phase 4E Final Patch) | Verification Test |
|---|---|---|---|
| **Broad Substring Matching** (`button_shirt`, `coin_purse`, `rectangular_box`, `lithium_flashlight`) | Token matching on `"button"`, `"coin"`, `"rectangular"`, `"lithium"` incorrectly mapped non-battery items to `0: battery`. | Checks `NON_BATTERY_NOUNS` (shirt, purse, box, flashlight, torch, etc.). Non-battery compounds are mapped to `QUARANTINE_UNVERIFIED` (`REQUIRES_APPROVAL`). Standalone form factors are restricted to validated single tokens (`Button`, `Cylindrical`, etc.) or explicit compounds (`button_battery`, `coin_cell`). | `test_evaluate_source_class_mapping_non_battery_compounds` (PASSED) |
| **Numeric Class Name** (`names: ["0"]`) | Hardcoded override auto-accepted `names: ["0"]` as `0: battery`. | Removed hardcoded block. Numeric strings without explicit battery model codes (e.g. "18650") are flagged as `UNVERIFIED_NUMERIC_CLASS`. `inspect_roboflow_data_yaml` fails closed with `ValueError` ("do not contain any vetted battery class"). | `test_evaluate_source_class_mapping_numeric_zero_unverified_and_rejected` (PASSED) |
| **Non-Continuous Dict Keys in `data.yaml`** | Dict keys like `{0: 'battery', 2: 'cardboard'}` were sorted blindly without checking continuity. | Validates that `sorted(int_keys) == list(range(len(int_keys)))`. Fails closed if keys are non-continuous or do not start at 0. | `test_inspect_roboflow_data_yaml_non_continuous_dict_keys` (PASSED) |
| **Basename Collisions Across Subdirectories** | If `train/images/img1.jpg` and `val/images/img1.jpg` had identical basenames, destination `battery_curated/images/img1.jpg` was overwritten. | Added `resolve_unique_curated_path()`. Checks destination file existence and SHA-256; disambiguates colliding basenames with parent folder prefix and content SHA (`val_img1_{sha[:8]}.jpg`). | `test_resolve_unique_curated_path_prevents_basename_collision` (PASSED) |
| **Dataset Ingest Claim Integrity** | Script could theoretically be called without a real ZIP export. | Fails closed: exits with `No Roboflow ZIP found` without claiming ingestion. Ingest status: `AWAITING_EXPORT_ZIP`. | Code inspection & CLI contract check |

---

## 3. TASK 2: WIKIMEDIA MINI-PILOT EXECUTION & PHYSICAL DATA AUDIT

### 3.1 Downloaded Candidates (Physical Bytes, Hashes & Licenses)

All 4 shortlisted images were streamed and downloaded under the 20 MiB safety cap:

| Candidate ID | Target Category | Resolution | File Size | SHA-256 | pHash | License & Author |
|---|---|---|---|---|---|---|
| `wiki_bio_banana_peel_ground` | biological | 5184x3456 | 4,573,954 B | `086eae0e94991f0261c53e1891f568a14918ab2499a6f8f03c7ca253a63d57e1` | `87878797b33330b8` | CC BY-SA 4.0 (Vladiator360) |
| `wiki_bio_street_food_remains` | biological | 2048x1365 | 294,623 B | `05a0cf4609a96f6c7aba3a1963350a45c5c105f12bb75d222d688b91360a9d56` | `dc4c0eeea9b30c53` | CC BY-SA 3.0 (Andrew Currie) |
| `wiki_neg_junction_boxes` | hard-negative | 480x640 | 110,051 B | `bbf480384cc91e0550d85f2cfe4b2aa4632385ca64143f129a53037e85ce5221` | `b7e9e1158f94c0b1` | CC BY-SA 2.0 (Thomas Nugent) |
| `wiki_neg_utility_box_leeds` | hard-negative | 5152x3864 | 5,736,159 B | `c3af6456ce23ca287034d0bb2cfd11063b4ebad7970094da148ed9dccf6cb9f5` | `8787365a7870696d` | CC BY-SA 4.0 (Mtaylor848) |
| **Total** | — | — | **10,714,787 B (~10.22 MiB)** | — | — | **$\le$ 20 MiB Budget** |

### 3.2 Split Leakage Verification
- **Authoritative Hashes Loaded:** 3,842 hashes from `v2` (train/val/test), `v3` candidate splits, and `holdout` (27 images).
- **Exact Collision Audit:** 0 SHA-256 collisions against any train/val/test/holdout image.
- **Perceptual Hash Audit:** 0 near-duplicate matches (Hamming distance $\le 8$).
- **Conclusion:** Dataset splits remain 100% clean and uncompromised.

---

## 4. DETAILED VISUAL QA INSPECTION & BBOX MEASUREMENTS

Every image was decoded to physical pixels and inspected at 100% crop level.

### 4.1 Candidate 1: `wiki_bio_banana_peel_ground.jpg` (Status: PROPOSED_PENDING_USER_APPROVAL)
- **Visual Description:** Authentic discarded banana peel lying on dark outdoor asphalt. Natural degradation, curled skin strips, and prominent stalk. Direct remedy for biological waste recall deficits.
- **Full Resolution Re-measurement ($5184 \times 3456$):**
  - **Previous Box:** `[2150, 1300, 5010, 2110]` was clipped at $x=5010$ due to an intermediate crop boundary.
  - **Reviewer Finding Verified:** The banana peel strip extends all the way to the right border of the photograph (touches $x = 5184$).
  - **Right Edge Verification:** At $x = 5183$, peel skin is actively present between $y = 1764$ and $y = 2294$ ($W=5184$).
  - **Corrected Pixel Box ($x_1, y_1, x_2, y_2$):** `[2100, 1280, 5184, 2300]` (Width = 3084 px, Height = 1020 px)
  - **YOLO Normalized ($cx, cy, w, h$):** `1 0.702546 0.517940 0.594907 0.295139`
- **Verdict:** **PROPOSED_PENDING_USER_APPROVAL**. Perfect biological waste sample, completely enclosed.

### 4.2 Candidate 2: `wiki_bio_street_food_remains.jpg` (Status: REJECTED_TAXONOMY_CONFLICT)
- **Visual Description:** Leftovers on a public ledge. However, the dominant physical objects are **expanded polystyrene (EPS) styrofoam bowls/takeaway containers** with wooden skewers and sauce residue.
- **Taxonomy Conflict:** Styrofoam bowls belong to `7: plastic` (or general trash `9: trash`), NOT `1: biological`. Labeling these large takeaway containers as `1: biological` would introduce label noise and teach the model to classify styrofoam bowls as food waste.
- **Verdict:** **REJECTED_TAXONOMY_CONFLICT**. Excluded to prevent taxonomy contamination.

### 4.3 Candidate 3: `wiki_neg_junction_boxes.jpg` (Status: CLEAN_NEGATIVE_CANDIDATE)
- **Visual Description:** Wall-mounted electrical junction boxes and a green cast iron pillar on clean pavement.
- **Full-Frame Inspection:** 100% of the $480 \times 640$ frame was scanned at full resolution. Zero litter or discarded objects are present.
- **Verdict:** **CLEAN_NEGATIVE_CANDIDATE**. Authentic clean hard-negative to suppress false positive detections on electrical junction equipment. Staged for review; zero premature label files written to disk.

### 4.4 Candidate 4: `wiki_neg_utility_box_leeds.jpg` (Status: PENDING_OBJECT_IDENTITY_REVIEW)
- **Visual Description:** Green telecommunications cabinet on sidewalk against brick wall.
- **Full-Frame Inspection:** Discovered an unidentified metallic cylindrical object in the grass at the cabinet base ($x \approx 3580\text{--}4100$, $y \approx 2800\text{--}3100$).
- **Reviewer Directive Applied:** Insufficient proof to confirm definitely as a metal can. Cannot be treated as a pure negative (risk of unannotated litter) and cannot be labeled as metal without proof.
- **Verdict:** **PENDING_OBJECT_IDENTITY_REVIEW**. Blocked until verified.

---

## 5. GENERATED ARTIFACTS & MANIFEST SYNCHRONIZATION

1. **Contact Sheet:**
   - Path: `data/acquisition_staging/phase4e_staging/contact_sheet/mini_pilot_contact_sheet.jpg`
   - Description: 2x2 grid showing all 4 candidates with candidate ID, category, dimensions, SHA-256, pHash, and byte size.
2. **Visual QA Overlays:**
   - `overlay_wiki_bio_banana_peel_ground.jpg`: Corrected green bounding box enclosing banana peel to right image edge (`[2100, 1280, 5184, 2300]`).
   - `overlay_wiki_bio_street_food_remains.jpg`: Highlights EPS styrofoam bowls with rejection notice.
   - `overlay_wiki_neg_junction_boxes.jpg`: Confirms clean negative status with 0 boxes.
   - `overlay_wiki_neg_utility_box_leeds.jpg`: Orange box highlighting unconfirmed cylindrical metallic object (`PENDING_OBJECT_IDENTITY_REVIEW`).
3. **Decoupled QA Manifests:**
   - `data/acquisition_staging/manifests/mini_pilot_qa_manifest.json`
   - `data/acquisition_staging/phase4e_staging/mini_pilot_staging_summary.json`
   - Strictly separates `technical_qa_result`, `annotation_review_status`, and `user_train_approval`.
4. **Staged Training Expansion Candidate Manifest:**
   - `data/acquisition_staging/manifests/train_expansion_candidate_manifest.json`
   - Records the 2 staged candidates (`wiki_bio_banana_peel_ground`, `wiki_neg_junction_boxes`), exact physical image and proposed label SHA-256 hashes, licenses, and attributions.

---

## 6. REGRESSION TEST VERIFICATION

The updated intake safety test suite in `tests/qa/test_phase4e_intake_safety.py` contains 26 comprehensive tests, all passing:

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\CNTT-KLCN155-waste-detection, configfile: pytest.ini
collected 26 items

tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_valid_cases PASSED [  3%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_token_count_strictly_five PASSED [  7%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_class_id_validation PASSED [ 11%]
tests/qa/test_phase4e_intake_safety.py::test_validate_yolo_bbox_strictly_within_image_bounds PASSED [ 15%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_nc_mismatch PASSED [ 19%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_identifies_battery_classes PASSED [ 23%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_zip_slip_protection PASSED [ 26%]
tests/qa/test_phase4e_intake_safety.py::test_safe_extract_untrusted_zip_bomb_protection PASSED [ 30%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_fails_closed_on_missing_files PASSED [ 34%]
tests/qa/test_phase4e_intake_safety.py::test_load_authoritative_split_hashes_verifies_holdout_jpg_jpeg_png PASSED [ 38%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_whitelist_does_not_contain_pending_annotation_comments PASSED [ 42%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_mini_pilot_shortlist_structure PASSED [ 46%]
tests/qa/test_phase4e_intake_safety.py::test_stage_phase4e_unapproved_run_fails_closed PASSED [ 50%]
tests/qa/test_phase4e_intake_safety.py::test_quarantine_non_battery_source_class PASSED [ 53%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_vetted_battery_types PASSED [ 57%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_accessories_excluded PASSED [ 61%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_ambiguous_quarantined PASSED [ 65%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_with_vetted_battery_form_factors PASSED [ 69%]
tests/qa/test_phase4e_intake_safety.py::test_cache_key_uniqueness_for_different_paths PASSED [ 73%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_non_battery_compounds PASSED [ 76%]
tests/qa/test_phase4e_intake_safety.py::test_evaluate_source_class_mapping_numeric_zero_unverified_and_rejected PASSED [ 80%]
tests/qa/test_phase4e_intake_safety.py::test_inspect_roboflow_data_yaml_non_continuous_dict_keys PASSED [ 84%]
tests/qa/test_phase4e_intake_safety.py::test_resolve_unique_curated_path_prevents_basename_collision PASSED [ 88%]
tests/qa/test_phase4e_intake_safety.py::test_mini_pilot_qa_manifest_contract PASSED [ 92%]
tests/qa/test_phase4e_intake_safety.py::test_train_expansion_candidate_manifest_contract PASSED [ 96%]
tests/qa/test_phase4e_intake_safety.py::test_mini_pilot_contact_sheet_and_overlays_exist PASSED [100%]

============================= 26 passed in 1.71s ==============================
```

Full repository detection, web, and QA regression suites: **417 passed, 1 skipped, 0 failed in 99.78s**.

---

## 7. NEXT STEPS & TRAINING HANDOFF

- **Current Staging State:** 2 candidates staged in `train_expansion_candidate_manifest.json` (`APPROVED_FOR_TRAIN = 0`).
- **User Approval Gate:** Once user confirms approval for the 2 candidates (`wiki_bio_banana_peel_ground` and `wiki_neg_junction_boxes`), transition directly to the dataset expansion plan for waste batteries and organic waste fine-tuning, without additional audit cycles.
