# REAL-WASTE 10-CLASS VALIDATION PLAN (V4 — POST-TASK 8H SCARCE-CLASS SCALING)

- **Plan Version:** `REAL_WASTE_10CLASS_VALIDATION_PLAN_V4_TASK8H_SCARCE_SCALING`
- **Total Unique Verified Real-Waste Validation Images:** `403` (`+ 39` from Wave 8H)
- **Frozen Test (`299`), Candidate Test (`274`), External Holdout (`27`):** `UNTOUCHED (True)`
- **Zero Train/Val Leakage Verified:** `True`

| Class ID | Class Name | V3 TACO Val | Pre-8H Staged Val | Wave 8H Val (`+`) | Combined Val Images | Combined Val Boxes | Gap to 50 Instances | Gap to 100 Instances |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | `battery` | `0` (`0` bx) | `23` | **`+8`** (`10` bx) | **`31`** | **`57`** | `0` | `43` |
| `1` | `biological` | `1` (`2` bx) | `19` | **`+20`** (`21` bx) | **`40`** | **`43`** | `7` | `57` |
| `2` | `cardboard` | `24` (`30` bx) | `20` | **`+0`** (`0` bx) | **`44`** | **`61`** | `0` | `39` |
| `3` | `clothes` | `0` (`0` bx) | `1` | **`+3`** (`5` bx) | **`4`** | **`7`** | `43` | `93` |
| `4` | `glass` | `12` (`15` bx) | `15` | **`+5`** (`8` bx) | **`32`** | **`43`** | `7` | `57` |
| `5` | `metal` | `40` (`68` bx) | `16` | **`+1`** (`2` bx) | **`57`** | **`94`** | `0` | `6` |
| `6` | `paper` | `19` (`28` bx) | `28` | **`+0`** (`0` bx) | **`47`** | **`71`** | `0` | `29` |
| `7` | `plastic` | `159` (`335` bx) | `18` | **`+0`** (`0` bx) | **`177`** | **`367`** | `0` | `0` |
| `8` | `shoes` | `1` (`2` bx) | `10` | **`+4`** (`4` bx) | **`15`** | **`21`** | `29` | `79` |
| `9` | `trash` | `47` (`95` bx) | `25` | **`+0`** (`0` bx) | **`72`** | **`151`** | `0` | `0` |
