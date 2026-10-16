# TASK 8C — TASK 4: SALVAGE AUDIT OF PRIORITIZED `NEEDS_REVIEW` CANDIDATES

- **Prioritized High-Probability Candidates Visually Inspected:** **`209`** (focused on `0: battery`, `1: biological`, `2: cardboard`, `4: glass`, `5: metal`, `6: paper`, `8: shoes`, `9: trash`)
- **Salvaged to `REAL_WASTE_PASS` (`OBJECT_ANNOTATION_VALID=True`, `REAL_WASTE_CONTEXT_VERIFIED=True`, `ACTUAL_WASTE`):** **`29` unique images**
- **Salvaged to `KEEP_AUXILIARY` (`OBJECT_ANNOTATION_VALID=True`, `REAL_WASTE_CONTEXT_VERIFIED=False`):** **`32` unique images**
- **Reclassified to `REJECT` (non-waste posters/signs/drawings/non-target materials):** **`33` unique images**
- **Returned to / Kept in `NEEDS_REVIEW` (dense indistinguishable heaps or unannotated secondary objects):** **`115` unique images**
- **BEFORE / AFTER Contact Sheets:**
  - [`task8c_salvaged_23_real_waste_before.jpg`](../../data/acquisition_staging/task8_wave2_large_batch/contact_sheets_task8c/task8c_salvaged_23_real_waste_before.jpg) -> [`task8c_salvaged_23_real_waste_after.jpg`](../../data/acquisition_staging/task8_wave2_large_batch/contact_sheets_task8c/task8c_salvaged_23_real_waste_after.jpg)
  - [`task8c_salvaged_27_auxiliary_after.jpg`](../../data/acquisition_staging/task8_wave2_large_batch/contact_sheets_task8c/task8c_salvaged_27_auxiliary_after.jpg)

---

## 1. 29 Images Salvaged to `REAL_WASTE_PASS` (`ACTUAL_WASTE`)

| Item ID | Source Family | Classes Annotated (`0..9`) | Verified Boxes | Domain Tag | Assigned Split | Manual Instance Annotation & Visual Verification |
|---|---|---|---:|---|---|---|
| `task8w2_0046` | `openimages_v7_train` | `1,1` | 2 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 2 blackened, overripe decaying bananas on wooden surface (ACTUAL_WASTE). |
| `task8w2_0083` | `openimages_v7_train` | `1` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Single oxidized, browned carved banana (ACTUAL_WASTE). |
| `task8w2_0663` | `openimages_v7_train` | `5,4` | 2 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Multi-class outdoor street litter with crushed Budweiser can (5: metal) and discarded Smirnoff Ice bottle (4: glass). |
| `task8w2_1179` | `wikimedia_commons_wave2` | `0` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Single worn, scratched Varta AA battery cell standing upright (ACTUAL_WASTE). |
| `task8w2_1198` | `wikimedia_commons_wave2` | `1` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Abandoned rotting whole cauliflower head lying on street pavement (ACTUAL_WASTE). |
| `task8w2_1203` | `wikimedia_commons_wave2` | `1` | 1 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Discarded orange/citrus peel lying on dead twigs outdoors (ACTUAL_WASTE). |
| `task8w2_1218` | `wikimedia_commons_wave2` | `2` | 1 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Used empty corrugated banana produce box (2: cardboard, ACTUAL_WASTE). |
| `task8w2_1220` | `wikimedia_commons_wave2` | `2` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Flattened corrugated cardboard book packaging mailer on floor (2: cardboard, ACTUAL_WASTE). |
| `task8w2_1246` | `wikimedia_commons_wave2` | `6` | 1 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Stack of discarded newspapers/waste paper (Altpapier) in recycling container (6: paper, ACTUAL_WASTE). |
| `task8w2_1247` | `wikimedia_commons_wave2` | `6,1` | 2 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Multi-class discarded paper food tray (6: paper) with leftover food scraps (1: biological) on grass. |
| `task8w2_1252` | `wikimedia_commons_wave2` | `8,8,8,8` | 4 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 4 old abandoned hiking shoes/boots left on forest ground, all 4 instances boxed (8: shoes, ACTUAL_WASTE). |
| `task8w2_1253` | `wikimedia_commons_wave2` | `4,7,1,8` | 4 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 4-class outdoor bus-stop litter scene with green glass bottle (4: glass), plastic bowl (7: plastic), food waste (1: biological), and abandoned slipper (8: shoes). |
| `task8w2_1256` | `wikimedia_commons_wave2` | `9` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Dirty discarded trash bag being picked up outdoors (9: trash, ACTUAL_WASTE). |
| `task8w2_1268` | `wikimedia_commons_wave2` | `9,9,9` | 3 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 3 used cigarette butts standing in ash, all 3 instances tightly boxed (9: trash, ACTUAL_WASTE). |
| `task8w2_1281` | `wikimedia_commons_wave2` | `0` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Weathered, worn vintage Eveready No. 460 45V battery (ACTUAL_WASTE). |
| `task8w2_1294` | `wikimedia_commons_wave2` | `0,0,0,0` | 4 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 4 spent AA alkaline batteries lined up for disposal, all 4 instances boxed (ACTUAL_WASTE). |
| `task8w2_1298` | `wikimedia_commons_wave2` | `0,0` | 2 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 2 used Varta Longlife AAA batteries on desk, both instances tightly boxed (ACTUAL_WASTE). |
| `task8w2_1310` | `wikimedia_commons_wave2` | `0` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Single used, surface-scratched button cell battery on wooden table (ACTUAL_WASTE). |
| `task8w2_1327` | `wikimedia_commons_wave2` | `0,0,0` | 3 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: 3 used AAA alkaline batteries lying on grid mat, all 3 instances boxed (ACTUAL_WASTE). |
| `task8w2_1336` | `wikimedia_commons_wave2` | `7,7,1,1` | 4 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Multi-class discarded bread loaves (1: biological) inside 2 plastic bags (7: plastic) hung on street fence. |
| `task8w2_1339` | `wikimedia_commons_wave2` | `1` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Discarded half-eaten bread roll (Bulka) lying on outdoor rock (ACTUAL_WASTE). |
| `task8w2_1353` | `wikimedia_commons_wave2` | `2,9` | 2 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Multi-class street fly-tipping scene with discarded cardboard box (2: cardboard) and dumped stained mattress (9: trash). |
| `task8w2_1381` | `wikimedia_commons_wave2` | `5` | 1 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Pinched, dented Pepsi aluminum beverage can (5: metal, ACTUAL_WASTE). |
| `task8w2_1382` | `wikimedia_commons_wave2` | `5` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Crushed/squeezed aluminum beverage can (5: metal, ACTUAL_WASTE). |
| `task8w2_1384` | `wikimedia_commons_wave2` | `5` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Shredded, torn aluminum can litter lying on dirt and twigs outdoors (5: metal, ACTUAL_WASTE). |
| `task8w2_1400` | `wikimedia_commons_wave2` | `5` | 1 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Crumpled ball of discarded aluminum foil on gravel, reclassified from 9:trash to 5:metal (ACTUAL_WASTE). |
| `task8w2_1401` | `wikimedia_commons_wave2` | `9` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Shredded burst latex balloon litter lying on asphalt (9: trash, ACTUAL_WASTE). |
| `task8w2_1404` | `wikimedia_commons_wave2` | `4,9,9,9` | 4 | `ACTUAL_WASTE` | `VAL_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Multi-class glass ashtray (4: glass) containing 3 used cigarette butts (9: trash), all 4 instances boxed. |
| `task8w2_1405` | `wikimedia_commons_wave2` | `9` | 1 | `ACTUAL_WASTE` | `TRAIN_REAL_WASTE_W2` | Salvaged from NEEDS_REVIEW: Single discarded cigarette butt lying on outdoor gravel (9: trash, ACTUAL_WASTE). |

---

## 2. 32 Images Salvaged to `KEEP_AUXILIARY` (Tight Per-Instance Boxes, Non-Waste Context)

| Item ID | Source Family | Classes Annotated (`0..9`) | Verified Boxes | Domain Tag | Assigned Split | Manual Instance Annotation & Rationale |
|---|---|---|---:|---|---|---|
| `task8w2_0298` | `openimages_v7_train` | `2,2` | 2 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact cardboard tissue boxes on wooden floor (2: cardboard). |
| `task8w2_0548` | `openimages_v7_train` | `5,1,1` | 3 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Multi-class Fanta metal can (5: metal) with orange and lemon (1: biological), all 3 boxed. |
| `task8w2_0728` | `openimages_v7_train` | `6` | 1 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single vintage postal envelope (6: paper), tight box. |
| `task8w2_0745` | `openimages_v7_train` | `6` | 1 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single open manuscript book (6: paper), tight box. |
| `task8w2_1177` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (AA + 10180) on studio background, both instances boxed. |
| `task8w2_1178` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (AA + 2/3A) on studio background, both instances boxed. |
| `task8w2_1180` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (AA + 22.5V Eveready) on studio background, both instances boxed. |
| `task8w2_1181` | `wikimedia_commons_wave2` | `0` | 1 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single GP 27A 12V battery above ruler, tight instance box. |
| `task8w2_1182` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (4.5V flat-pack + AA) on studio background, both instances boxed. |
| `task8w2_1183` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (AA + 4/5A) on studio background, both instances boxed. |
| `task8w2_1184` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Retail blister card of 4 AA batteries, tight box. |
| `task8w2_1185` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single intact Panasonic 9V battery, tight box. |
| `task8w2_1186` | `wikimedia_commons_wave2` | `0` | 1 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single Varta 9V battery posed next to matchstick, tight box. |
| `task8w2_1211` | `wikimedia_commons_wave2` | `4,2` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Multi-class glass jar (4: glass) sitting on cardboard box (2: cardboard). |
| `task8w2_1235` | `wikimedia_commons_wave2` | `5` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Top-down macro photo of intact 12-oz aluminum soda can (5: metal). |
| `task8w2_1237` | `wikimedia_commons_wave2` | `5,5` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact Coca-Cola Zero Sugar aluminum cans on white background, both boxed. |
| `task8w2_1285` | `wikimedia_commons_wave2` | `0,0,0` | 3 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 3 intact batteries (18650, AA, AAA) standing upright, all 3 instances boxed. |
| `task8w2_1290` | `wikimedia_commons_wave2` | `0,0,0,0,0` | 5 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 5 intact cylindrical batteries on grid, all 5 instances boxed. |
| `task8w2_1291` | `wikimedia_commons_wave2` | `0,0,0` | 3 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 3 horizontal batteries (A23, AAA, AA), all 3 instances boxed. |
| `task8w2_1292` | `wikimedia_commons_wave2` | `0,0,0,0` | 4 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 4 vertical batteries (AA, AAA, AAAA, A23), all 4 instances boxed. |
| `task8w2_1295` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 Super-cell AAA batteries on white background, both instances separately boxed. |
| `task8w2_1296` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 4 AAA heavy-duty batteries in shrink-wrap pack, tight box. |
| `task8w2_1297` | `wikimedia_commons_wave2` | `0,0,0,0` | 4 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 4 batteries (9V, AA, AAA, AAAA) on grid, all 4 instances boxed. |
| `task8w2_1299` | `wikimedia_commons_wave2` | `0,0,0,0,0` | 5 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 5 alkaline batteries on white background, all 5 instances boxed. |
| `task8w2_1303` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Single intact 625U-LR9 button cell battery, tight box. |
| `task8w2_1305` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Duracell 625A alkaline button cell in retail blister card. |
| `task8w2_1309` | `wikimedia_commons_wave2` | `0` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Retail pack of 2 CR2032 3V button batteries. |
| `task8w2_1319` | `wikimedia_commons_wave2` | `0,0,0` | 3 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 3 cylindrical batteries on grid, all 3 instances boxed. |
| `task8w2_1321` | `wikimedia_commons_wave2` | `0,0` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: 2 intact batteries (AA + A23) on grid, both instances boxed. |
| `task8w2_1354` | `wikimedia_commons_wave2` | `2` | 1 | `AMBIGUOUS_CONTEXT` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Close-up of sealed corrugated cardboard shipping box (2: cardboard). |
| `task8w2_1357` | `wikimedia_commons_wave2` | `2` | 1 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Closed corrugated cardboard book mailer on white background (2: cardboard). |
| `task8w2_1358` | `wikimedia_commons_wave2` | `2,6` | 2 | `PRODUCT_STUDIO` | `AUXILIARY_TRAIN_W2` | Salvaged to KEEP_AUXILIARY: Multi-class unfolded cardboard book mailer (2: cardboard) holding a book (6: paper). |

---

## 3. 33 Candidates Reclassified from `NEEDS_REVIEW` to `REJECT`

| Item ID | Source Family | Domain Tag | Outcome | Rejection Reason |
|---|---|---|---|---|
| `task8w2_0172` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Red cast-iron wall postbox, not 2: cardboard. |
| `task8w2_0186` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Slatted wooden crate on pavement, not 2: cardboard. |
| `task8w2_0194` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Blue metal Domino tin box, not 2: cardboard. |
| `task8w2_0202` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Polished wooden jewelry/cigar box, not 2: cardboard. |
| `task8w2_0203` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Plywood storage crate ('Magic Box'), not 2: cardboard. |
| `task8w2_0212` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Wooden fish-balls shipping crate ('King Oscar'), not 2: cardboard. |
| `task8w2_0217` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Open wooden crate on floor, not 2: cardboard. |
| `task8w2_0230` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Wooden ballot box with slot, not 2: cardboard. |
| `task8w2_0240` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Museum wooden ballot box on pedestal, not 2: cardboard. |
| `task8w2_0241` | `openimages_v7_train` | `AMBIGUOUS_CONTEXT` | **`REJECT`** | Decorated metal lunchbox tin, not 2: cardboard. |
| `task8w2_1190` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | WWII vintage propaganda poster illustration ('Don't be finicky!'), not a physical waste photo. |
| `task8w2_1191` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | WWII vintage propaganda poster illustration ('Don't bite off more...'), not a physical waste photo. |
| `task8w2_1192` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | WWII vintage propaganda poster illustration ('How's your bunker?'), not a physical waste photo. |
| `task8w2_1193` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | WWII vintage propaganda poster ('Join the Ranks - Fight Food Waste'), not a physical waste photo. |
| `task8w2_1213` | `wikimedia_commons_wave2` | `PRODUCT_STUDIO` | **`REJECT`** | 1887 black-and-white historical engraving/illustration (FOURNEL), not a photograph. |
| `task8w2_1225` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Cracked iPod screen (consumer electronics device, not 4: glass packaging/container waste). |
| `task8w2_1226` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Shattered public bus windshield (vehicle infrastructure, not 4: glass waste item). |
| `task8w2_1269` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Cast-iron street manhole cover in Istanbul (municipal infrastructure, not 9: trash). |
| `task8w2_1270` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Broken concrete sidewalk curb (civil infrastructure, not 9: trash). |
| `task8w2_1271` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Stage drum set at jazz concert ('Conversations About Leading'), unrelated to 0: battery. |
| `task8w2_1272` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Digital fantasy illustration by David Revoy ('electron-donor'), not a photograph. |
| `task8w2_1278` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Stone wall window in Cachi ('Cachi 1994 12'), unrelated to 0: battery. |
| `task8w2_1331` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Chinese anti-food-waste wall plaque/sign, not biological waste. |
| `task8w2_1340` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Metal street fence sign ('Public Food Scrap Drop-Off Site'), not biological waste. |
| `task8w2_1345` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Aerial landscape view of agricultural fields, no waste instances. |
| `task8w2_1346` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Metal pipe mounted on wooden wall, no biological waste instances. |
| `task8w2_1360` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Technical cross-section diagram of cardboard flute types with text labels. |
| `task8w2_1362` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Technical diagram of corrugated fiberboard flute profiles with text labels. |
| `task8w2_1378` | `wikimedia_commons_wave2` | `IN_USE` | **`REJECT`** | Japanese technical diagram of an aluminum beverage can. |
| `task8w2_1392` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Dutch vector infographic chart ('Duurzaam oud papier'), not a photograph. |
| `task8w2_1395` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Park directional signpost on pole, not 9: trash. |
| `task8w2_1397` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Bank ATM machine built into wall, not 9: trash. |
| `task8w2_1410` | `wikimedia_commons_wave2` | `ACTUAL_WASTE` | **`REJECT`** | Painted sticker/graffiti on metal roller shutter, not 9: trash. |
