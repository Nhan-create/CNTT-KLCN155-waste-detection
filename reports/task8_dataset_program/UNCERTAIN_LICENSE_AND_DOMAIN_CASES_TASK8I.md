# TASK 8I FINAL HOTFIX — DANH SÁCH CA LICENSE & DOMAIN CHƯA CHẮC CHẮN / CẦN LƯU Ý

## Phần A — 11 Ca Chênh lệch Giấy phép Ảnh (`Photo License`) và Giấy phép Quan sát (`Observation License`)

Theo tài liệu chính thức của iNaturalist, `photo.license_code` quy định quyền bản quyền đối với **bức ảnh**, còn `observation.license_code` quy định quyền đối với **bản ghi dữ liệu quan sát** (tọa độ GPS, ngày giờ, loài). Trong cả 11 ca dưới đây, bức ảnh có giấy phép mở (`cc0`, `cc-by`, `cc-by-sa`), nên không bị cấm sử dụng về mặt bản quyền ảnh nếu không tái phân phối metadata quan sát. Tuy nhiên, khi đánh giá độc lập về **ngữ cảnh rác thật (`ACTUAL_WASTE`)**, không có ca nào đủ điều kiện nâng lên `REAL_WASTE_PASS` (`5` ca chuyển sang `KEEP_AUXILIARY`, `6` ca giữ ở `REJECT` do lỗi hiển vi/sai lớp):

| ID | Obs ID | Photo ID | Lớp | Photo License | Obs License | Đánh giá ngữ cảnh & hình ảnh | Trạng thái Trước Hotfix | Trạng thái Sau Hotfix |
|---|---:|---:|---|---|---|---|---|---|
| `task8h_0072` | `393573256` | `721408118` | `biological` | `cc0` | `all-rights-reserved` | `NATURAL_ORCHARD_PLANT_DECAY_OBS_RESTRICTED` | `REJECT` | **`KEEP_AUXILIARY`** |
| `task8h_0087` | `381065135` | `697219293` | `biological` | `cc-by` | `cc-by-nc` | `MACRO_MOLD_COLONY_CLOSEUP` | `REJECT` | **`REJECT`** |
| `task8h_0102` | `341439711` | `621573592` | `biological` | `cc-by-sa` | `cc-by-nc` | `ISOLATED_SPECIMEN_UNVERIFIED_DISPOSAL_OBS_RESTRICTED` | `REJECT` | **`KEEP_AUXILIARY`** |
| `task8h_0114` | `385680403` | `706092727` | `biological` | `cc-by` | `cc-by-nc` | `ISOLATED_SPECIMEN_UNVERIFIED_DISPOSAL_OBS_RESTRICTED` | `REJECT` | **`KEEP_AUXILIARY`** |
| `task8h_0118` | `381065874` | `697221122` | `biological` | `cc-by` | `cc-by-nc` | `MACRO_MOLD_COLONY_CLOSEUP` | `REJECT` | **`REJECT`** |
| `task8h_0141` | `206259626` | `364791252` | `biological` | `cc-by` | `cc-by-nc` | `ISOLATED_SPECIMEN_UNVERIFIED_DISPOSAL_OBS_RESTRICTED` | `REJECT` | **`KEEP_AUXILIARY`** |
| `task8h_0145` | `65374647` | `105210260` | `biological` | `cc-by` | `cc-by-nc` | `ISOLATED_SPECIMEN_UNVERIFIED_DISPOSAL_OBS_RESTRICTED` | `REJECT` | **`KEEP_AUXILIARY`** |
| `task8h_0149` | `191131338` | `335224816` | `glass` | `cc-by` | `cc-by-nc` | `NON_WASTE_WILDLIFE_ON_BOTTLE_NECK` | `REJECT` | **`REJECT`** |
| `task8h_0151` | `37766401` | `59937445` | `glass` | `cc0` | `cc-by-nc` | `TAXONOMY_MISMATCH_PLASTIC_IN_GLASS_QUERY` | `REJECT` | **`REJECT`** |
| `task8h_0152` | `37765976` | `59936631` | `glass` | `cc0` | `cc-by-nc` | `TAXONOMY_MISMATCH_PLASTIC_IN_GLASS_QUERY` | `REJECT` | **`REJECT`** |
| `task8h_0155` | `343390748` | `625508924` | `glass` | `cc-by` | `all-rights-reserved` | `NON_WASTE_WILDLIFE_ON_BOTTLE_NECK` | `REJECT` | **`REJECT`** |

## Phần B — Các Ca Trái Cây Cầm Tay (`Task 2`) & Ca Biên Ngữ Cảnh (`Domain Watchlist`)

| ID | Hành động trong Hotfix | Lý do / Bằng chứng | Trạng thái Sau Hotfix |
|---|---|---|---|
| `task8h_0085` | `DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY` | Moldy lemon held in hand indoors; empty observation description; no trash bin or disposal context. | **`KEEP_AUXILIARY`** |
| `task8h_0092` | `DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY` | Moldy mandarin held in hand indoors; None observation description; no visual disposal proof. | **`KEEP_AUXILIARY`** |
| `task8h_0137` | `DEMOTED_FROM_REAL_WASTE_PASS_TO_KEEP_AUXILIARY` | Moldy mandarin held in hand; None observation description; no visual disposal proof. | **`KEEP_AUXILIARY`** |
| `task8h_0094` | `FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE` | Close-up of moldy tangerine held in fingers; unambiguous spoiled food specimen, flagged if PM prefers zero handheld shots in train. | **`REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)`** |
| `task8h_0101` | `FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE` | Cut half of spent moldy lemon held in hand over kitchen box ('This lemon lived a good life...'); flagged for PM visibility. | **`REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)`** |
| `task8h_0142` | `FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE` | Moldy citrus held over outdoor foliage/bin; flagged for PM visibility. | **`REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)`** |
| `task8h_0150` | `FLAGGED_ON_DOMAIN_WATCHLIST_UNCHANGED_PER_SCOPE` | Clear bottle base buried in forest litter; observer note mentions 'plastic drink bottle' though rim appears rigid. | **`REAL_WASTE_PASS (APPROVED_FOR_TRAIN=False)`** |
