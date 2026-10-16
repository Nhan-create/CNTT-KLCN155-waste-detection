# TASK 8C — INDEPENDENT IMAGE, SPECIMEN & SESSION ACCOUNTING REPORT

| Dataset / Split | Unique Images | Unique Specimens | Unique Photographer Sessions | Verified Boxes | `REAL_WASTE_PASS` (`ACTUAL_WASTE`) | `KEEP_AUXILIARY` | Multi-Class Images |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Existing Clean Baseline Train** (`data/Unified_Dataset/train`) | **2,428** | **2,428** | N/A | **3,965** | N/A (Baseline) | N/A | **38** |
| **Wave 1 `REAL_WASTE_PASS`** (`TRAIN_REAL_WASTE_W1` + `VAL_REAL_WASTE_W1`) | **59** (`43` tr / `16` val) | **57** | **54** | **93** | **59** (100%) | **0** | **13** |
| **Wave 1 `KEEP_AUXILIARY`** (`AUXILIARY_TRAIN_W1` + `AUXILIARY_HOLD_W1`) | **45** (`38` tr / `7` hold) | **44** | **42** | **77** | **0** | **45** (`31` studio + `14` ambig) | **2** |
| **Wave 2 `REAL_WASTE_PASS`** (`TRAIN_REAL_WASTE_W2` + `VAL_REAL_WASTE_W2`) | **41** (`30` tr / `11` val) | **41** | **39** | **78** | **41** (100%) | **0** | **11** |
| **Wave 2 `KEEP_AUXILIARY`** (`AUXILIARY_TRAIN_W2`) | **166** | **164** | **156** | **270** | **0** | **166** (`39` studio + `1` in-use + `126` ambig) | **3** |
| **Wave 2 `NEEDS_REVIEW` (Held)** | **925** | **904** | **825** | *(unverified)* | — | — | — |
| **Wave 2 `REJECT` (Excluded)** | **64** | — | — | **0** | — | — | — |
