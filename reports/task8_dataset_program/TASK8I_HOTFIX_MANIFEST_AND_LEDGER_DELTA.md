# TASK 8I FINAL HOTFIX — BẢNG ĐỐI CHIẾU THAY ĐỔI MANIFEST & LEDGER DELTA

## 1. Thay đổi trên `task8i_inaturalist_107_verified_manifest.json`

| Chỉ số | Trước Hotfix (Số thực trong Manifest) | Sau Hotfix (Task 2 + Task 3) | Chênh lệch (`Delta`) | Giải thích |
|---|---:|---:|---:|---|
| `REAL_WASTE_PASS` (Tổng) | `33` | **`30`** | `-3` | Hạ `task8h_0085, 0092, 0137` (trái cây mốc cầm tay không có bằng chứng vứt bỏ) sang `KEEP_AUXILIARY` |
| `REAL_WASTE_PASS` (`Train`) | `27` | **`24`** | `-3` | Cả 3 ảnh `0085, 0092, 0137` thuộc tập `TRAIN_REAL_WASTE_W8H` |
| `REAL_WASTE_PASS` (`Val`) | `6` | **`6`** | `0` | Không thay đổi |
| `KEEP_AUXILIARY` | `29` | **`37`** | `+8` | `+3` ca cầm tay (`0085, 0092, 0137`) + `+5` ca giấy phép ảnh mở nhưng ngữ cảnh phụ trợ (`0072, 0102, 0114, 0141, 0145`) |
| `REJECT` | `45` | **`40`** | `-5` | `5` ca (`0072, 0102, 0114, 0141, 0145`) có `photo_license` hợp lệ chuyển từ `REJECT` sang `KEEP_AUXILIARY` |

## 2. Thay đổi trên `task8h_qa_manifest.json` & `CANONICAL_5000_DATASET_LEDGER.json` (V6)

| Chỉ số | Trước Hotfix | Sau Hotfix | Chênh lệch (`Delta`) |
|---|---:|---:|---:|
| Wave 8H `REAL_WASTE_PASS` (Tổng) | `158` | **`155`** | `-3` |
| Wave 8H `REAL_WASTE_PASS` (`Train`) | `130` | **`127`** | `-3` |
| Wave 8H `REAL_WASTE_PASS` (`Val`) | `28` | **`28`** | `0` |
| Lớp `1: biological` Verified RW Train (Toàn chương trình) | `122` | **`119`** | `-3` |
| Tổng Unique Verified RW Train Images (`Tier 2`) | `1811` | **`1808`** | `-3` |
| 9 lớp còn lại (`0, 2, 3, 4, 5, 6, 7, 8, 9`) | Giữ nguyên | **Giữ nguyên** | `0` |
