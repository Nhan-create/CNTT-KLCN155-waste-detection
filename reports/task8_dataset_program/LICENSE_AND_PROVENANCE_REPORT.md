# TASK 8 — LICENSE & PROVENANCE AUDIT REPORT (`LICENSE_AND_PROVENANCE_REPORT.md`)

- **Generated At:** `2026-10-14`
- **Core Governance Rule (A3 & B1):**
  1. Every Open Images V7 image must carry its real original Flickr `Author`, `AuthorProfileURL`, `OriginalLandingURL`, `Title`, `OriginalMD5`, and `CC-BY-2.0` license URL from the official Google Open Images metadata CSVs (`train-images-boxable-with-rotation.csv`, `validation-images-with-rotation.csv`, `test-images-with-rotation.csv`).
  2. Zero synthetic contributor strings (`"Open Images V7 Contributor (<id>)"`) are permitted.
  3. Verified on-disk image counts are strictly separated from upstream theoretical pool estimates.

---

## 1. Verified On-Disk Provenance Summary Across Task 8 (Wave 1 + Wave 2)

| Source Family | Wave | Verified On-Disk Images | `QA_PASS` Images | License Family | Real Author / Landing URL Verified | Synthetic Contributor Placeholders |
|---|---|---:|---:|---|---:|---:|
| `openimages_v7_validation` (Wave 1) | Wave 1 | **42** | 37 | `CC-BY-2.0` (`creativecommons.org/licenses/by/2.0/`) | **42 / 42 (`100%`)** | **0** |
| `wikimedia_commons` (Wave 1) | Wave 1 | **179** | 67 | `CC0 / CC-BY / CC-BY-SA` | **179 / 179 (`100%`)** | **0** |
| `openimages_v7_train` (Wave 2 Priority #1) | Wave 2 | **936** | 157 | `CC-BY-2.0` (`creativecommons.org/licenses/by/2.0/`) | **936 / 936 (`100%`)** | **0** |
| `openimages_v7_test` (Wave 2 Supplement) | Wave 2 | **38** | 6 | `CC-BY-2.0` (`creativecommons.org/licenses/by/2.0/`) | **38 / 38 (`100%`)** | **0** |
| `openimages_v7_validation` (Wave 2 Supplement) | Wave 2 | **3** | 0 | `CC-BY-2.0` (`creativecommons.org/licenses/by/2.0/`) | **3 / 3 (`100%`)** | **0** |
| `wikimedia_commons_wave2` (Wave 2 Waste & Battery) | Wave 2 | **219** | 12 | `CC0 / CC-BY / CC-BY-SA` | **219 / 219 (`100%`)** | **0** |
| **TOTAL TASK 8 ON DISK** | **W1 + W2** | **1417** | **279** | **100% Open / CC Licensed** | **1417 / 1417 (`100%`)** | **0** |

---

## 2. Upstream Candidate Pools: Verified Local Metadata vs Theoretical Estimates

| Upstream Source | License | Already Used in `real_v2` | Verified Metadata Rows Cached Locally (`data/external/waste-training/`) | Actually Downloaded & Audited in W1+W2 | Remaining Unaudited Pool Status |
|---|---|---:|---:|---:|---|
| **TACO Official v1.0 (`pedropro/TACO`)** | `CC-BY-4.0` | **`1,500 / 1,500` (`100%`)** (`1,490` active + `10` EXIF-quarantined in `real_v2_source_manifest.csv`) | `1,500` images (`4,784` boxes in `taco_official_annotations.json`) | `0` in W1/W2 (intentionally skipped to prevent 100% duplication/leakage with `real_v2`!) | **Exhausted (`0` unused images remaining in official TACO 1.0)** |
| **Open Images V7 Train** | `CC-BY-2.0` (Flickr attribution) | `0` | `6,133` non-human, non-rotated images (`17,681` boxes) cached in `openimages-train-slice-meta.csv` (`5,422` author-disjoint from `real_v2`) | **936** downloaded in Wave 2 | ~`4,486` cached author-disjoint candidates remaining locally + full 9M Open Images Train CSV streamable |
| **Open Images V7 Validation** | `CC-BY-2.0` (Flickr attribution) | `527` | `41,620` image metadata rows (`286,206` boxes) in `validation-images-with-rotation.csv` | **45** (`42` in W1 + `3` in W2) | ~`1,800` target-MID candidates remaining before QA filtering |
| **Open Images V7 Test** | `CC-BY-2.0` (Flickr attribution) | `0` | `125,436` image metadata rows (`937,327` boxes) in `test-images-with-rotation.csv` | **38** downloaded in Wave 2 | ~`5,200` target-MID candidates remaining before QA filtering |
| **ZeroWaste (`bashkirova/ZeroWaste`)** | `CC-BY-NC-4.0` (Non-Commercial) | `0` | Metadata surveyed | `0` | Eligible for academic thesis research (`KLCN155`), requires sorting-conveyor domain flag (`ACTUAL_WASTE`) |
