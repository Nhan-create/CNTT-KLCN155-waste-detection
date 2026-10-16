# TASK 8C — 10-CLASS PROGRESS & MILESTONE GAP REPORT (`REAL_WASTE_PASS` vs `KEEP_AUXILIARY`)

- **Milestone 1 Target:** `500` independent real-waste train images / class
- **Milestone 2 Target:** `1,500` independent real-waste train images / class
- **Milestone 3 Target:** `3,000` independent real-waste train images / class
- **Milestone 4 Target:** `5,000+` independent real-waste train images / class
- **Strict Accounting Rule (Task 2):** Only `REAL_WASTE_PASS` (`ACTUAL_WASTE`) train images are added to the cumulative real-waste train total and used to compute milestone gaps. `KEEP_AUXILIARY` images are reported in a separate column and do not reduce the real-waste milestone gap.

| Class ID & Name | Baseline Clean Train | W1 `REAL_WASTE_PASS` Train | W2 `REAL_WASTE_PASS` Train | **Cumulative Real-Waste Train** | Separate `KEEP_AUXILIARY` Train (W1+W2) | W2 `NEEDS_REVIEW` Queue | Gap to M1 (`500`) | Gap to M2 (`1,500`) | Gap to M3 (`3,000`) | Gap to M4 (`5,000`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0: battery` | 16 | +0 | +4 | **20** | +30 (`2` W1 + `28` W2) | 46 | **480** | **1,480** | **2,980** | **4,980** |
| `1: biological` | 16 | +20 | +9 | **45** | +10 (`1` W1 + `9` W2) | 168 | **455** | **1,455** | **2,955** | **4,955** |
| `2: cardboard` | 97 | +1 | +3 | **101** | +34 (`10` W1 + `24` W2) | 131 | **399** | **1,399** | **2,899** | **4,899** |
| `3: clothes` | 409 | +1 | +1 | **411** | +11 (`4` W1 + `7` W2) | 128 | **89** | **1,089** | **2,589** | **4,589** |
| `4: glass` | 115 | +6 | +2 | **123** | +21 (`9` W1 + `12` W2) | 152 | **377** | **1,377** | **2,877** | **4,877** |
| `5: metal` | 182 | +0 | +6 | **188** | +54 (`1` W1 + `53` W2) | 72 | **312** | **1,312** | **2,812** | **4,812** |
| `6: paper` | 67 | +3 | +0 | **70** | +29 (`4` W1 + `25` W2) | 115 | **430** | **1,430** | **2,930** | **4,930** |
| `7: plastic` | 1,245 | +5 | +2 | **1,252** | +3 (`3` W1 + `0` W2) | 0 | **0** | **248** | **1,748** | **3,748** |
| `8: shoes` | 113 | +7 | +4 | **124** | +18 (`6` W1 + `12` W2) | 133 | **376** | **1,376** | **2,876** | **4,876** |
| `9: trash` | 210 | +2 | +5 | **217** | +0 (`0` W1 + `0` W2) | 59 | **283** | **1,283** | **2,783** | **4,783** |
