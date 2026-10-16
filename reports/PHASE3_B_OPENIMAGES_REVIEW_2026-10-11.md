# Phase 3B — Open Images provenance and mapping review

Scope: nine active real_v2 Open Images samples identified by the reproducible audit. No label, mapping, manifest, split, or source image was changed. Machine-readable evidence is in `reports/PHASE3_B_DATA_P0_AUDIT_2026-10-11.json`.

## Four active train images missing from the v2 provenance manifest

| Image ID | Stored labels | Exact source coordinate match | Local visual review | Disposition |
|---|---:|---|---|---|
| `19c5c4dae0da0f77` | 1 shoes box | `/m/01b638` Boot | Close-up boot on decking; no discarded-waste context | `BLOCK_AND_REVIEW` |
| `2e62e7ce1ce182b5` | 2 shoes boxes | 2 × `/m/01b638` Boot | Pair of boots photographed against a studio background | `BLOCK_AND_REVIEW` |
| `48fe726552b07707` | 1 shoes box | `/m/06k2mb` High heels | Product-style high heel on a white background | `BLOCK_AND_REVIEW` |
| `57f0572aa1a3cba5` | 2 shoes boxes | 2 × `/m/01b638` Boot | Pair of shoes photographed against a white background | `BLOCK_AND_REVIEW` |

The stored YOLO coordinates match the cited rows in the cached official Open Images V7 validation CSV to within `5e-8`. This recovers annotation provenance, but it does not approve the class decision: neither `Boot` nor `High heels` has an entry in `configs/detection_source_mapping.yaml`, and none of the four images has a row in `data/detection/real_v2_source_manifest.csv`.

## Five active images with box-count disagreement

| Image ID | Split | Stored / current-mapping source boxes | Evidence | Disposition |
|---|---|---:|---|---|
| `004dd2ac922baf98` | test | 1 / 3 | One jacket is described by overlapping Shirt, Jeans, and Jacket source labels; the stored label keeps one box. Image is apparel laid out indoors, with no waste context. | Object-level hierarchy and scope review |
| `17135174fbbf3301` | train | 2 / 1 | Stored labels contain Wine glass plus a Bottle provisionally assigned to glass; Bottle remains material-ambiguous in config and its queue item is pending. Image shows a wine bottle and filled glass on a table. | Material and scope review |
| `1a04c113301b09f8` | train | 3 / 2 | Stored labels contain two Wine glass boxes plus a Bottle provisionally assigned to glass; Bottle remains material-ambiguous. Image shows a wine bottle and two glasses indoors. | Material and scope review |
| `1a94b67907646757` | val | 2 / 3 | Source hierarchy has Clothing, Coat, and Jacket labels on two physical regions; stored labels collapse one duplicate. Image shows tagged apparel on a hanger. | Object-level hierarchy and scope review |
| `48a211ad0792aa28` | train | 2 / 3 | Source hierarchy has Shirt, Coat, and Jacket labels on the same garment; stored labels collapse one duplicate. Image shows apparel on a mannequin/display. | Object-level hierarchy and scope review |

Adding every source row would create duplicate boxes for the same physical garment. Removing the provisional Bottle rows would encode an unapproved material decision. These five files therefore remain unchanged.

## Mapping readiness

- Open Images contributes all `528` clothes boxes and `288/295` shoes boxes in active real_v2 splits.
- The configured clothing and footwear MIDs have status `REQUIRES_REVIEW`.
- `scripts/collect_real_detection_v2.py` currently selects every mapping with a non-null `class_id`; it does not require `status: CONFIRMED`.
- Image-level absence of a Person annotation is not evidence that apparel or footwear is discarded waste. The nine reviewed cases above are out-of-domain candidates despite having usable object boxes.

Training readiness for Open Images clothes/shoes remains **BLOCKED** until scope decisions and object-level reviews are recorded. No relabeling should be performed to improve metrics.

The v2 source manifest schema contains a generic image URL but no per-image license or attribution fields. The configuration's dataset-level `CC BY 4.0 / CC BY 2.0` statement does not by itself establish the license and attribution record for each downloaded image. Legal/provenance readiness therefore remains blocked as well.
