# Phase 3 — Ground Truth v2 approval package

**State:** `PROPOSED_ONLY_NOT_FROZEN`  
**Built:** 2026-10-11 23:19 (Asia/Saigon)  
**Protected inputs:** GT v1 and GT v2 were read only. No annotation source file was changed.

This package covers all six active disputed objects, all ten zero-active filenames, and the two remaining quarantined objects in mixed images. The ten zero-active filenames represent only nine unique image contents because `images (13).jpg` and `images (7).jpg` have the same SHA-256.

## Quick visual review

Current v2 proposal:

![Current GT contact sheet](contact_sheet_current.jpg)

Recommended decisions:

![Proposed decision contact sheet](contact_sheet_proposed.jpg)

The individual source image, current overlay, and proposed overlay paths are recorded in [review_manifest.json](review_manifest.json). The proposed drawings are review aids; they are not a modified Ground Truth file.

## Approval set A — six active disputed objects

| Image / object | Current GT | Recommendation | Reason | Evaluation impact if approved |
|---|---|---|---|---|
| `images (10).jpg` / `img04_obj01` | clothes, `[0,160,640,398]`, active, overlapping | `EXCLUDE_IMAGE_PENDING_INSTANCE_REANNOTATION` | Two same-class boxes describe overlapping parts of one clothes dump. The image does not support two defensible instance boundaries. | Exclude the whole image: remove `img04_obj01` and accepted companion `img04_obj02` from the main denominator. All three stored pipelines have 0 predictions at confidence 0.25 on this image, so this removes 2 FN. |
| `images (12).jpg` / `img06_obj01` | cardboard, `[32,36,535,519]`, active, container nesting | `KEEP_AND_MARK_ACCEPTED` | The cardboard container and paper contents are visibly different target materials. Cross-class containment is valid and is not a duplicate by itself. | No count or class change. This object is FN in all three stored runs. |
| `OIP (1).webp` / `img20_obj01` | clothes, `[0,95,313,237]`, active, overlapping | `EXCLUDE_IMAGE_PENDING_INSTANCE_REANNOTATION` | Low-resolution image with two same-class regions that overlap and do not form defensible independent instances. | Exclude the whole image: remove `img20_obj01` and accepted companion `img20_obj02`. All three pipelines have 0 predictions at confidence 0.25, so this removes 2 FN. |
| `Phan-Loai-Rac-Tai-Ng-01.jpg` / `img22_obj01` | plastic, `[230,220,460,550]`, active | `RELABEL_TO_TRASH` | The bag contains mixed waste. This follows the existing v2 guideline that bagged heterogeneous waste is `trash`; an empty discarded bag remains `plastic`. | Total GT stays constant; one object moves plastic → trash. It is TP for both stored SSDLite runs under the current plastic label. With the proposed trash label it becomes FN and the plastic prediction becomes FP at the frozen threshold. YOLO remains FN. |
| `Phan-Loai-Rac-Tai-Ng-01.jpg` / `img22_obj02` | plastic, `[340,445,675,705]`, active | `RELABEL_TO_TRASH` | Same bagged heterogeneous-waste rule. | Total GT stays constant; one FN moves from plastic → trash in all three stored runs. |
| `Phan-Loai-Rac-Tai-Ng-01.jpg` / `img22_obj04` | cardboard, `[680,570,765,650]`, active | `KEEP_CARDBOARD_AND_MARK_ACCEPTED` | The fixed ten-class taxonomy has no beverage-carton class. Cardboard is the closest existing class for this rigid paper/carton composite package; there is no evidence supporting another current class. | No count or class change. This object is FN in all three stored runs. |

If all six recommendations are approved, the main proposed GT count becomes **41**, down from 45: four objects leave with the two excluded images, while the two bag objects change class without changing total count. This is a review simulation only.

Individual visuals:

- `images (10).jpg`: [original](originals/01_images__10_.jpg), [current](overlays_current/01_images__10_.jpg.png), [proposed](overlays_proposed/01_images__10_.jpg.png)
- `images (12).jpg`: [original](originals/02_images__12_.jpg), [current](overlays_current/02_images__12_.jpg.png), [proposed](overlays_proposed/02_images__12_.jpg.png)
- `OIP (1).webp`: [original](originals/03_OIP__1_.webp), [current](overlays_current/03_OIP__1_.webp.png), [proposed](overlays_proposed/03_OIP__1_.webp.png)
- `Phan-Loai-Rac-Tai-Ng-01.jpg`: [original](originals/04_Phan-Loai-Rac-Tai-Ng-01.jpg), [current](overlays_current/04_Phan-Loai-Rac-Tai-Ng-01.jpg.png), [proposed](overlays_proposed/04_Phan-Loai-Rac-Tai-Ng-01.jpg.png)

## Approval set B — ten zero-active filenames

These images must not silently become hard negatives. Eight contain unsegregated/full-frame piles. Two show utility receptacles. Until an explicit decision is approved, the safe action is to remove both their GT and their predictions from the main instance metric.

| Filename / object | Current inactive state | Recommendation | Stored predictions ≥0.25 counted as FP by current evaluator (before / after / YOLO) |
|---|---|---|---:|
| `images (13).jpg` / `img07_obj01` | plastic pile | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 2 |
| `images (18).jpg` / `img11_obj01` | utility trash bin | Quarantine; only accept as negative after exhaustive whole-image review | 1 / 0 / 2 |
| `images (19).jpg` / `img12_obj01` | biological pile | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 1 |
| `images (6).jpg` / `img16_obj01` | biological pile | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 0 |
| `images (7).jpg` / `img17_obj01` | plastic pile; exact duplicate of `images (13).jpg` | Exclude and deduplicate before any future split/evaluation | 0 / 0 / 2 |
| `images (8).jpg` / `img18_obj01` | utility office basket | Quarantine; only accept as negative after exhaustive whole-image review | 1 / 0 / 1 |
| `images (9).jpg` / `img19_obj01` | cardboard pile/full frame | Exclude from main instance evaluation; optional separate pile task | 1 / 1 / 0 |
| `OIP.webp` / `img21_obj01` | clothes pile/full frame | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 0 |
| `photo-5-1642556540067542343380.webp` / `img23_obj01` | clothes pile | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 1 |
| `rac-thai-huu-co-1.jpg` / `img26_obj01` | biological pile/full frame | Exclude from main instance evaluation; optional separate pile task | 0 / 0 / 0 |

The existing comparison evaluator filters these GT objects but still evaluates predictions on their images. It therefore counts **3 / 1 / 9** predictions as FP for SSDLite-before / SSDLite-after / YOLO E1 respectively. That behavior is only valid if the images are approved as exhaustively labeled hard negatives. They are not.

## Approval set C — two quarantined objects in mixed images

These two images still contain accepted discrete objects and should remain in the instance benchmark. Approval removes only the invalid area-level duplicate annotation.

| Image / object | Recommendation | Reason |
|---|---|---|
| `base64-1717231151649602366964.jpeg` / `img01_obj07` | `REMOVE_INVALID_TEXTURE_CARPET_GT_ONLY` | The 85.38% glass-fragment carpet overlaps four accepted discrete glass objects. Keep the image and discrete objects; do not count the carpet as a fifth glass instance. |
| `download (2).jpg` / `img03_obj03` | `REMOVE_DUPLICATE_FULL_CANVAS_GT_ONLY` | The full-canvas plastic box duplicates two accepted bottles. Keep the image and the two bottle objects. |

Visuals:

- `base64-1717231151649602366964.jpeg`: [original](originals/15_base64-1717231151649602366964.jpeg), [current](overlays_current/15_base64-1717231151649602366964.jpeg.png), [proposed](overlays_proposed/15_base64-1717231151649602366964.jpeg.png)
- `download (2).jpg`: [original](originals/16_download__2_.jpg), [current](overlays_current/16_download__2_.jpg.png), [proposed](overlays_proposed/16_download__2_.jpg.png)

## Explicit evaluation policy proposed for approval

1. **Main task:** ten-class discrete waste instance detection with the fixed class order already defined by the project.
2. **Excluded image:** remove both GT and predictions from the main metric. It contributes no TP, FP, or FN.
3. **Mixed image with ambiguous pile:** keep accepted discrete GT. An approved ignore region may suppress predictions whose center lies inside it; ignored predictions are neither TP nor FP, and the ignore region is not a GT object. The ignore region must be drawn and approved before use.
4. **Zero-active pile image:** exclude from main instance evaluation until exhaustive discrete-instance annotation exists. Never interpret it as a clean negative merely because active GT is empty.
5. **Utility receptacle:** quarantine. It becomes a hard-negative image only after exhaustive whole-image review confirms that no object from the ten target classes is present.
6. **Pile objective:** if pile detection is required, create a separate benchmark and annotation policy. Do not silently mix pile areas into the fixed ten-class instance benchmark.
7. **Duplicates:** exact-content duplicates contribute once per evaluation cohort and must remain within one split if retained elsewhere.

## Metric simulation — evidence, not an official result

Protocol: confidence ≥0.25, IoU ≥0.50, one-to-one greedy matching by descending confidence, same class. These are operating-threshold metrics, not AP/mAP.

| Pipeline | Current evaluator behavior (GT / Pred / TP / FP / FN / P / R / F1) | Explicitly exclude zero-active images | All recommendations approved |
|---|---|---|---|
| SSDLite before quick fix | 45 / 21 / 10 / 11 / 35 / 47.62 / 22.22 / 30.30 | 45 / 18 / 10 / 8 / 35 / 55.56 / 22.22 / 31.75 | 41 / 18 / 9 / 9 / 32 / 50.00 / 21.95 / 30.51 |
| SSDLite after quick fix | 45 / 11 / 7 / 4 / 38 / 63.64 / 15.56 / 25.00 | 45 / 10 / 7 / 3 / 38 / 70.00 / 15.56 / 25.45 | 41 / 10 / 6 / 4 / 35 / 60.00 / 14.63 / 23.53 |
| YOLOv8n E1 | 45 / 32 / 5 / 27 / 40 / 15.62 / 11.11 / 12.99 | 45 / 23 / 5 / 18 / 40 / 21.74 / 11.11 / 14.71 | 41 / 23 / 5 / 18 / 36 / 21.74 / 12.20 / 15.62 |

The simulation demonstrates why an explicit exclusion policy is required. It must not be used to choose the policy that gives the best metric.

## Evidence and integrity

- Machine-readable decisions and simulation: [review_manifest.json](review_manifest.json)
- Build validation: [validation_results.json](validation_results.json)
- Generation log: [generation_log.json](generation_log.json)
- SHA-256 inventory: [ARTIFACT_SHA256.txt](ARTIFACT_SHA256.txt)
- Reproducible builder: [build_phase3_gt_review.py](build_phase3_gt_review.py)

Input hashes are embedded in the manifest. The builder verifies all copied source-image hashes against GT v2 and asserts exactly 6 active disputed objects and 10 zero-active filenames. GT v2 remains a proposal and is not frozen by this package.
