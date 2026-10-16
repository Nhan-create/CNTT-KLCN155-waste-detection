# TASK 8 — TEST SET GOVERNANCE & 4-WAY LEAKAGE AUDIT REPORT (`TEST_SETS_AND_LEAKAGE_AUDIT_REPORT.md`)

- **Generated At:** `2026-10-14`
- **Scope:** Disambiguation of `299` vs `274` vs `27` test/holdout sets and 4-way deduplication/leakage verification across Wave 1 (`221` images) and Wave 2 (`1196` images).

---

## 1. Official Evaluation Splits Disambiguation (`299` vs `274` vs `27`)

| Split Identifier | Manifest / File Path | Unique Images | Exact Purpose & Rules |
|---|---|---:|---|
| **`real_v2/test` (Frozen Official Test)** | `data/detection/splits/real_v2/test.txt` | **299** | Primary immutable benchmark used for `E1`, `E2`, `E3` historical comparisons. Never modified or re-split. |
| **`v3_candidate/test` (Leakage-Cleansed Test)** | `data/detection_v3_candidate/splits/test.txt` | **274** | Strict subset of `real_v2/test` (`299 - 25 = 274`) with 25 legacy train-test near-duplicate images removed. Used to verify leakage-free generalization. |
| **`holdout_27` (`data/holdout`)** | `data/holdout` (`reports/qa_acceptance/`) | **27** | Independent physical Vietnamese waste holdout set managed by QA Agent D. Zero overlap with any training/staging data. |
| **`VAL_REAL_WAVE1` + `VAL_REAL_WAVE2`** | Staging QA manifests | **23 + 40 = 63** | Photographer-session-disjoint validation splits from Wave 1 and Wave 2. |

---

## 2. 4-Way Leakage & Deduplication Verification Results

| Check Dimension | Wave 1 (`task8a_multiclass_batch`) | Wave 2 (`task8_wave2_large_batch`) | Verdict |
|---|---:|---:|---|
| **1. Exact `SHA-256` overlap with `real_v2`, `v3_candidate`, `holdout_27`** | `0` | `0` | **PASS (`0` exact duplicates)** |
| **2. Perceptual `dHash/pHash` (`Hamming <= 6`) overlap with `real_v2/test (299)`, `v3_candidate/test (274)`, `holdout_27 (27)`** | `0` | `0` | **PASS (`0` near-duplicate test leakage)** |
| **3. Upstream `ImageID` overlap with `real_v2` (`1,500` TACO + `527` Open Images)** | `0` | `0` | **PASS (`0` shared source IDs)** |
| **4. Photographer / Session (`AuthorProfileURL` / `session_group`) overlap between `TRAIN_WAVE*` and `VAL_REAL_WAVE*` or `real_v2`** | `0` (Fixed `#Canada150` `109/110/111/112`) | `0` (`session_group` hash-partitioned) | **PASS (`0` cross-split session leakage)** |
