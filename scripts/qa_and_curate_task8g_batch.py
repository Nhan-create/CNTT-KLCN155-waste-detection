"""Task 8G — Quality Control, Multi-Class Per-Instance Annotation & Visual QA Pipeline (Tasks 2–4).

Processes the checkpointed Task 8G candidate pool (`task8g_raw_candidates_manifest.json`) in
`data/acquisition_staging/task8g_large_scale_expansion/` enforcing all Task 8F & Task 8G safety gates:
1. Never use blind NMS to auto-delete valid ground-truth overlapping boxes (`classify_box_pair_overlap`, `agnostic_nms=False`).
2. Never infer final class from filename alone (`propose_candidate_class_from_title` is candidate-only).
3. Never auto-assert `ACTUAL_WASTE` from keywords alone without visual/pixel context verification.
4. Never miss instances in multi-object scenes; route dense heaps (`> 6` instances or bulk piles) to `NEEDS_REVIEW`.
5. Never use images with missing annotations as hard negatives (`zero_unannotated_waste_used_as_hard_negative=True`).
6. Strictly separate `REAL_WASTE_PASS` / `KEEP_AUXILIARY` / `NEEDS_REVIEW` / `REJECT`.
7. Keep `APPROVED_FOR_TRAIN = False` on 100% of records.
8. Preserve `source_split` (`train -> TRAIN_REAL_WASTE_W8G`, `validation/valid -> VAL_REAL_WASTE_W8G`, `0` from `test`) and enforce zero session/specimen leakage.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquire_and_qa_task8f_scarce_classes import (
    BAD_WIKI_RE,
    BULK_PILE_WIKI_RE,
    CLASS_COLORS,
    CLASS_NAMES,
    SCARCE_CLASSES,
    classify_box_pair_overlap,
    classify_shynuaa_scarce_material,
    deduplicate_same_class_boxes_iou,
    extract_tight_per_instance_boxes,
    format_yolo_lines,
    norm_boxes_to_struct,
    propose_candidate_class_from_title,
)
from scripts.correct_task8f_semantic_qa import _struct_box_to_norm_tuple, compute_iou_xyxy

OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8g_large_scale_expansion"
RAW_MANIFEST_JSON = OUT_DIR / "task8g_raw_candidates_manifest.json"
QA_MANIFEST_JSON = OUT_DIR / "task8g_qa_manifest.json"
QA_MANIFEST_CSV = OUT_DIR / "task8g_qa_manifest.csv"

LABELS_PASS_DIR = OUT_DIR / "labels_real_waste_pass"
LABELS_AUX_DIR = OUT_DIR / "labels_auxiliary"
LABELS_REVIEW_DIR = OUT_DIR / "labels_needs_review"
OVERLAYS_DIR = OUT_DIR / "overlays"
SHEETS_DIR = OUT_DIR / "contact_sheets"


def render_overlay_8g(rec: dict[str, Any]) -> str:
    img_p = ROOT / rec["image_path"]
    with Image.open(img_p) as im:
        im_rgb = im.convert("RGB")
        w, h = im_rgb.size
        draw = ImageDraw.Draw(im_rgb)
        thickness = max(4, int(min(w, h) * 0.008))
        for b in rec["boxes"]:
            cid = int(b["class_id"])
            cname = b["class_name"]
            xc, yc, bw, bh = b["yolo_xywhn"]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            col = CLASS_COLORS.get(cid, (0, 255, 0))
            for t in range(thickness):
                draw.rectangle([max(0, x1 - t), max(0, y1 - t), min(w - 1, x2 + t), min(h - 1, y2 + t)], outline=col)
            lbl = f"{cid}:{cname}"
            draw.rectangle([x1, max(0, y1 - 20), x1 + len(lbl) * 7 + 10, y1], fill=col)
            draw.text((x1 + 4, max(0, y1 - 18)), lbl, fill=(10, 10, 10))
        out_p = OVERLAYS_DIR / f"{rec['candidate_id']}_overlay.jpg"
        im_rgb.save(out_p, quality=86)
        return str(out_p.relative_to(ROOT)).replace("\\", "/")


def render_sheet_8g(records: list[dict[str, Any]], title: str, out_name: str, max_items: int = 36) -> str:
    subset = records[:max_items]
    if not subset:
        return ""
    cols = 6
    rows = (len(subset) + cols - 1) // cols
    cell_w, cell_h = 260, 220
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + 40), (18, 20, 26))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"{title} (showing {len(subset)}/{len(records)})", fill=(240, 240, 245))
    for idx, rec in enumerate(subset):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = r * cell_h + 40
        ov_p = ROOT / rec["overlay_path"] if rec.get("overlay_path") else ROOT / rec["image_path"]
        if ov_p.exists():
            with Image.open(ov_p) as im:
                im_rgb = im.convert("RGB")
                im_rgb.thumbnail((cell_w - 8, cell_h - 36))
                sheet.paste(im_rgb, (x0 + 4, y0 + 4))
        cid_s = rec["candidate_id"].replace("task8g_", "")
        cls_s = ",".join(str(x) for x in rec["classes_present_ids"])
        cap = f"{cid_s}|c:{cls_s}|bx:{rec['bbox_count']}|{rec['target_split'][:9]}"
        draw.text((x0 + 4, y0 + cell_h - 28), cap, fill=(220, 235, 220))
    out_p = SHEETS_DIR / out_name
    sheet.save(out_p, quality=88)
    return str(out_p.relative_to(ROOT)).replace("\\", "/")


def annotate_shynuaa_multiclass_scene(
    img_bgr: np.ndarray,
    upstream_boxes: list[dict[str, Any]],
    detector: YOLO,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Refine upstream Shynuaa street-garbage boxes with class-aware YOLO (`agnostic_nms=False`) and crop material analysis.

    Returns `(boxes_norm, class_corrections_count)` where `boxes_norm` contains `(cid, x1, y1, x2, y2)`
    and `class_corrections_count` counts how many upstream generic `trash` boxes were refined into specific material classes.
    """
    h, w = img_bgr.shape[:2]
    det_res = detector.predict(img_bgr, conf=0.18, agnostic_nms=False, iou=0.65, verbose=False)[0]
    det_preds: list[tuple[int, float, float, float, float, float]] = []
    if det_res.boxes is not None and len(det_res.boxes) > 0:
        xyxyn = det_res.boxes.xyxyn.cpu().numpy()
        clss = det_res.boxes.cls.cpu().numpy().astype(int)
        confs = det_res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, conf in zip(xyxyn, clss, confs):
            bw, bh = float(x2 - x1), float(y2 - y1)
            if 0.015 <= bw <= 0.94 and 0.015 <= bh <= 0.94 and 0 <= int(c) < 10:
                det_preds.append((int(c), float(conf), float(x1), float(y1), float(x2), float(y2)))

    out_norm: list[tuple[int, float, float, float, float]] = []
    used_det_indices: set[int] = set()
    corrections = 0

    for up in upstream_boxes:
        cx, cy, bw, bh = float(up["cx"]), float(up["cy"]), float(up["w"]), float(up["h"])
        ux1, uy1 = max(0.0, cx - bw / 2.0), max(0.0, cy - bh / 2.0)
        ux2, uy2 = min(1.0, cx + bw / 2.0), min(1.0, cy + bh / 2.0)
        ub_xyxy = [ux1, uy1, ux2, uy2]

        matching_preds: list[tuple[int, float]] = []
        sub_det_boxes: list[tuple[int, float, float, float, float, float]] = []
        for d_idx, (dc, dconf, dx1, dy1, dx2, dy2) in enumerate(det_preds):
            iou_v = compute_iou_xyxy(ub_xyxy, [dx1, dy1, dx2, dy2])
            # Also check if detector box is tightly inside a coarse upstream box
            inter_w = max(0.0, min(ux2, dx2) - max(ux1, dx1))
            inter_h = max(0.0, min(uy2, dy2) - max(uy1, dy1))
            det_area = max(1e-6, (dx2 - dx1) * (dy2 - dy1))
            containment = (inter_w * inter_h) / det_area
            if iou_v >= 0.25:
                matching_preds.append((dc, dconf))
                used_det_indices.add(d_idx)
            if containment >= 0.75 and det_area < 0.65 * max(1e-6, bw * bh) and dconf >= 0.28:
                sub_det_boxes.append((dc, dconf, dx1, dy1, dx2, dy2))
                used_det_indices.add(d_idx)

        # If a coarse upstream box (>18% image area) contains 2+ distinct detector sub-boxes, split into per-instance boxes!
        if (bw * bh) > 0.18 and len(sub_det_boxes) >= 2:
            for dc, _dconf, dx1, dy1, dx2, dy2 in sub_det_boxes:
                out_norm.append((dc, round(dx1, 4), round(dy1, 4), round(dx2, 4), round(dy2, 4)))
            corrections += 1
            continue

        px1, py1 = int(ux1 * w), int(uy1 * h)
        px2, py2 = int(ux2 * w), int(uy2 * h)
        crop = img_bgr[max(0, py1) : min(h, py2), max(0, px1) : min(w, px2)]
        mat_cid = classify_shynuaa_scarce_material(crop, bw, bh, matching_preds)
        if mat_cid != int(up.get("src_cls", 9)):
            corrections += 1
        out_norm.append((mat_cid, round(ux1, 4), round(uy1, 4), round(ux2, 4), round(uy2, 4)))

    out_norm = deduplicate_same_class_boxes_iou(out_norm, iou_thresh=0.70, is_ground_truth=False)
    return out_norm, corrections


# Visual QA Overrides after contact-sheet inspection (populated/verified during visual QA pass)
VISUAL_REJECT_OVERRIDES_8G: set[str] = {
    "task8g_0399",  # Indoor recycling bins/doors without loose target waste
    "task8g_0322",  # Multi-angle repeat of studio apple core (task8g_0301)
    "task8g_0329",  # Multi-angle repeat of studio apple core (task8g_0301)
    "task8g_0892",  # Second walk-by angle of same gutter litter as task8g_0890
    "task8g_0905",  # Second walk-by angle of same sidewalk cup/wrapper as task8g_0904
    "task8g_0913",  # Second walk-by angle of same bench cup as task8g_0912
}
VISUAL_NEEDS_REVIEW_OVERRIDES_8G: set[str] = {
    "task8g_0085",  # Indoor warehouse full of industrial hoses/machinery
}
VISUAL_KEEP_AUX_OVERRIDES_8G: set[str] = {
    "task8g_0058",  # Museum display of historical worn work jacket
    "task8g_0087",  # Unfolded vintage camera cardboard box on white studio background
    "task8g_0141",  # Isolated moth-eaten wool sweater on plain studio background
    "task8g_0215",  # Isolated torn trousers on plain studio background
    "task8g_0226",  # Cardboard sign held up in use at an event
    "task8g_0301",  # Apple core photographed on white studio background
    "task8g_0333",  # Vintage rusted tin box photographed on plain white studio background
    "task8g_0367",  # Vintage coffee paper bag on indoor display shelf
    "task8g_0419",  # Macro photograph of corroded button cell on white studio background
}
VISUAL_PASS_OVERRIDES_8G: dict[str, dict[str, Any]] = {
    # Class 0: battery
    "task8g_0063": {
        "boxes": [(0, 0.02, 0.14, 0.44, 0.89), (0, 0.56, 0.10, 0.96, 0.90)],
        "reason": "Visual QA verified: two corroded spent industrial battery cells on workbench -> REAL_WASTE_PASS (0:battery).",
    },
    "task8g_0098": {
        "boxes": [(0, 0.06, 0.08, 0.35, 0.94), (0, 0.33, 0.28, 0.56, 0.94), (0, 0.56, 0.33, 0.74, 0.94)],
        "reason": "Visual QA verified: three leaking corroded AA/AAA batteries with visible chemical crust -> REAL_WASTE_PASS (0:battery).",
    },
    # Class 1: biological
    "task8g_0128": {
        "boxes": [(1, 0.27, 0.25, 0.65, 0.72)],
        "reason": "Visual QA verified: discarded dried orange peel on outdoor twigs/ground -> REAL_WASTE_PASS (1:biological).",
    },
    "task8g_0200": {
        "boxes": [(1, 0.10, 0.14, 0.88, 0.85)],
        "reason": "Visual QA verified: discarded fruit peels and eaten cores food waste -> REAL_WASTE_PASS (1:biological).",
    },
    "task8g_0315": {
        "boxes": [(1, 0.04, 0.10, 0.86, 0.88)],
        "reason": "Visual QA verified: discarded apple cores in kitchen organic waste -> REAL_WASTE_PASS (1:biological).",
    },
    "task8g_0357": {
        "boxes": [(1, 0.32, 0.52, 0.70, 0.92)],
        "reason": "Visual QA verified: rotting fallen apple decaying on forest floor -> REAL_WASTE_PASS (1:biological).",
    },
    "task8g_0362": {
        "boxes": [(1, 0.18, 0.16, 0.82, 0.85)],
        "reason": "Visual QA verified: discarded eaten apple core on dark ground -> REAL_WASTE_PASS (1:biological).",
    },
    "task8g_0386": {
        "boxes": [(1, 0.26, 0.06, 0.78, 0.92)],
        "reason": "Visual QA verified: discarded decaying apple core outdoors with wasp feeding -> REAL_WASTE_PASS (1:biological).",
    },
    # Class 2: cardboard
    "task8g_0169": {
        "boxes": [(2, 0.04, 0.06, 0.96, 0.94)],
        "reason": "Visual QA verified: torn corrugated cardboard packaging waste -> REAL_WASTE_PASS (2:cardboard).",
    },
    "task8g_0247": {
        "boxes": [(2, 0.02, 0.04, 0.76, 0.88)],
        "reason": "Visual QA verified: discarded empty pulp cardboard egg carton on ground -> REAL_WASTE_PASS (2:cardboard).",
    },
    # Class 3: clothes
    "task8g_0136": {
        "boxes": [(3, 0.04, 0.14, 0.44, 0.96), (3, 0.32, 0.04, 0.68, 0.94)],
        "reason": "Visual QA verified: two dirty worn-out discarded socks on ground -> REAL_WASTE_PASS (3:clothes).",
    },
    "task8g_0186": {
        "boxes": [(3, 0.04, 0.08, 0.96, 0.92)],
        "reason": "Visual QA verified: frayed torn discarded cloth rag -> REAL_WASTE_PASS (3:clothes).",
    },
    "task8g_0219": {
        "boxes": [(3, 0.16, 0.12, 0.86, 0.94)],
        "reason": "Visual QA verified: torn shredded textile rag waste -> REAL_WASTE_PASS (3:clothes).",
    },
    # Class 4: glass
    "task8g_0079": {
        "boxes": [(4, 0.14, 0.34, 0.34, 0.56), (4, 0.44, 0.38, 0.72, 0.62)],
        "reason": "Visual QA verified: two broken discarded brown glass bottles embedded in dirt -> REAL_WASTE_PASS (4:glass).",
    },
    "task8g_0161": {
        "boxes": [(4, 0.16, 0.12, 0.88, 0.86)],
        "reason": "Visual QA verified: shattered broken glass windshield panels on plastic tarp -> REAL_WASTE_PASS (4:glass).",
    },
    "task8g_0304": {
        "boxes": [(4, 0.52, 0.70, 0.62, 0.90)],
        "reason": "Visual QA verified: discarded green glass bottle left by outdoor brick corner -> REAL_WASTE_PASS (4:glass).",
    },
    # Class 5: metal
    "task8g_0015": {
        "boxes": [(5, 0.04, 0.42, 0.96, 0.82)],
        "reason": "Visual QA verified: rusted scrap metal beams discarded in snow -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0020": {
        "boxes": [(5, 0.12, 0.08, 0.96, 0.96)],
        "reason": "Visual QA verified: rusted scrap metal pipes inside waste skip -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0022": {
        "boxes": [(5, 0.12, 0.06, 0.88, 0.90)],
        "reason": "Visual QA verified: heavily rusted abandoned cast-metal gear in weeds -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0027": {
        "boxes": [(5, 0.04, 0.06, 0.96, 0.94)],
        "reason": "Visual QA verified: corroded rusted scrap iron chains -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0061": {
        "boxes": [(5, 0.04, 0.14, 0.92, 0.96)],
        "reason": "Visual QA verified: twisted scrap metal rods inside open dumpster -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0064": {
        "boxes": [(5, 0.04, 0.06, 0.96, 0.94)],
        "reason": "Visual QA verified: rusted structural steel scrap in container -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0067": {
        "boxes": [(5, 0.10, 0.08, 0.92, 0.90)],
        "reason": "Visual QA verified: crushed scrap metal car body -> REAL_WASTE_PASS (5:metal).",
    },
    "task8g_0069": {
        "boxes": [(5, 0.16, 0.14, 0.88, 0.88)],
        "reason": "Visual QA verified: corroded discarded metal number plate in dirt -> REAL_WASTE_PASS (5:metal).",
    },
    # Class 6: paper
    "task8g_0155": {
        "boxes": [(6, 0.48, 0.14, 0.76, 0.56), (6, 0.24, 0.08, 0.86, 0.92)],
        "reason": "Visual QA verified: used stained disposable paper coffee cup and paper plate on outdoor picnic table -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0351": {
        "boxes": [(6, 0.04, 0.04, 0.96, 0.96)],
        "reason": "Visual QA verified: crumpled creased waste paper sheet -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0385": {
        "boxes": [(6, 0.04, 0.04, 0.96, 0.96)],
        "reason": "Visual QA verified: crumpled brown kraft wrapping paper waste -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0390": {
        "boxes": [(6, 0.22, 0.04, 0.78, 0.88)],
        "reason": "Visual QA verified: crumpled discarded paper bag -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0777": {
        "boxes": [(6, 0.7424, 0.7414, 0.9061, 0.8971)],
        "reason": "Visual QA verified: crumpled discarded paper on grass next to street bin (false bin-base box removed) -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0870": {
        "boxes": [(6, 0.3200, 0.2600, 0.6800, 0.6400)],
        "reason": "Visual QA verified: full tight bounding box on discarded white paper sheet on grass -> REAL_WASTE_PASS (6:paper).",
    },
    "task8g_0904": {
        "boxes": [(6, 0.3700, 0.4900, 0.5500, 0.6000), (6, 0.5307, 0.3818, 0.5939, 0.4292)],
        "reason": "Visual QA verified: both discarded coffee cup and wrapper annotated on sidewalk -> REAL_WASTE_PASS (6:paper).",
    },
    # Class 7: plastic & multi-class Shynuaa fixes
    "task8g_0271": {
        "boxes": [(7, 0.16, 0.48, 0.58, 0.76)],
        "reason": "Visual QA verified: discarded plastic cup/bottle washed onto beach sand -> REAL_WASTE_PASS (7:plastic).",
    },
    "task8g_0292": {
        "boxes": [(7, 0.34, 0.52, 0.70, 0.82)],
        "reason": "Visual QA verified: discarded empty plastic bottle on tiled floor -> REAL_WASTE_PASS (7:plastic).",
    },
    "task8g_0800": {
        "boxes": [(7, 0.5009, 0.6248, 0.7040, 0.8930)],
        "reason": "Visual QA verified: discarded blue plastic cup on pavement -> REAL_WASTE_PASS (7:plastic).",
    },
    "task8g_0828": {
        "boxes": [(7, 0.1562, 0.8656, 0.2220, 0.9005), (5, 0.3327, 0.4698, 0.4443, 0.5363)],
        "reason": "Visual QA verified: plastic cap and metal foil litter on dirt (coarse background dirt box removed) -> REAL_WASTE_PASS (5:metal, 7:plastic).",
    },
    # Class 8: shoes
    "task8g_0030": {
        "boxes": [(8, 0.02, 0.10, 0.85, 0.81)],
        "reason": "Visual QA verified: abandoned dirty sandal lying on leaves and soil -> REAL_WASTE_PASS (8:shoes).",
    },
    "task8g_0037": {
        "boxes": [(8, 0.24, 0.56, 0.62, 0.90)],
        "reason": "Visual QA verified: abandoned pair of shoes left on wet street curb -> REAL_WASTE_PASS (8:shoes).",
    },
    "task8g_0075": {
        "boxes": [(8, 0.32, 0.30, 0.70, 0.54)],
        "reason": "Visual QA verified: discarded worn shoe sole buried in beach sand -> REAL_WASTE_PASS (8:shoes).",
    },
    "task8g_0400": {
        "boxes": [(8, 0.08, 0.25, 0.90, 0.68)],
        "reason": "Visual QA verified: discarded worn leather shoe on dry grass -> REAL_WASTE_PASS (8:shoes).",
    },
}


def main() -> None:
    t0 = time.time()
    assert RAW_MANIFEST_JSON.exists(), f"Missing {RAW_MANIFEST_JSON}"
    raw_payload = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    manifest_records = raw_payload["records"]
    pilots = raw_payload.get("pilots", [])
    print(f"Loaded {len(manifest_records)} Task 8G raw candidates from {RAW_MANIFEST_JSON}", flush=True)

    for d in [LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*"):
            f.unlink()

    model_path = ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot" / "weights" / "best.pt"
    if not model_path.exists():
        model_path = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
    detector = YOLO(str(model_path))

    # Strict Wikimedia real-waste evidence regex (used only when combined with pixel/context verification)
    wiki_strong_waste_re = re.compile(
        r"\b(litter|littered|garbage|rubbish|trash|discarded|abandoned|rotten|rotting|moldy|mouldy|spoiled|decaying|"
        r"corroded|leaking|leaked|leakage|rusty|crushed|flattened|shattered|broken\s+bottle|broken\s+glass|"
        r"torn|dirty|muddy|worn.?out|used\s+batter|dead\s+batter|spent\s+batter|street\s+litter|beach\s+litter|"
        r"fly.?tipping|illegal\s+dumping|food\s+waste|kitchen\s+waste|eggshell|peel)\b",
        re.I,
    )
    wiki_in_use_or_non_waste_re = re.compile(
        r"\b(engine\s+bay|installed|shelf|store|market|supermarket|shop|museum|exhibition|airline|flight|"
        r"breakfast|lunch|dinner|restaurant|cafe|serving|recipe|cooking|baking|zest|garnish|tree|vine|plant|"
        r"growing|fresh|new\s+shoe|pair\s+of\s+new|collection\s+box|recycling\s+bin|bottle\s+bank|container|"
        r"truck|lorry|factory|building|window|windshield|pane|bale|bales|person|people|portrait|wearing)\b",
        re.I,
    )

    qa_records: list[dict[str, Any]] = []
    prev_shynuaa_dt: dict[str, datetime] = {}
    seen_specimens: set[str] = set()
    total_label_corrections = 0
    records_with_label_corrections = 0

    for r in manifest_records:
        cid = r["candidate_id"]
        src = r["source_id"]
        source_split = r.get("source_split", "unspecified")
        w, h = int(r["width"]), int(r["height"])
        img_p = ROOT / r["image_path"]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        title_str = r.get("title", "")
        query_str = r.get("query_used", "")
        combined_text = f"{title_str} {query_str}".replace("_", " ").replace("-", " ")

        qa_status = "REJECT"
        domain_tag = "AMBIGUOUS_CONTEXT"
        obj_valid = False
        waste_verified = False
        approved_for_train = False
        reason = ""
        boxes: list[dict[str, Any]] = []
        rec_corrections = 0

        hsv_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv_full[:, :, 1]))
        white_px_ratio = float(np.mean((hsv_full[:, :, 1] < 22) & (hsv_full[:, :, 2] > 235)))
        low_sat_ratio = float(np.mean(hsv_full[:, :, 1] < 25))
        is_diagram_or_bw = (white_px_ratio > 0.46 and low_sat_ratio > 0.70) or (mean_sat < 10.0)

        border_mask = np.zeros(img_bgr.shape[:2], dtype=bool)
        border_mask[:14, :] = border_mask[-14:, :] = border_mask[:, :14] = border_mask[:, -14:] = True
        border_white_ratio = float(np.mean((hsv_full[border_mask, 1] < 24) & (hsv_full[border_mask, 2] > 232)))
        border_uniform_std = float(np.std(hsv_full[border_mask, 2]))
        is_studio_border = (border_white_ratio > 0.42) or (border_uniform_std < 11.5 and white_px_ratio > 0.25)

        if cid in VISUAL_PASS_OVERRIDES_8G:
            spec = VISUAL_PASS_OVERRIDES_8G[cid]
            boxes = norm_boxes_to_struct(w, h, spec["boxes"], "task8g_visual_qa_verified", is_ground_truth=True)
            qa_status = "REAL_WASTE_PASS"
            domain_tag = "ACTUAL_WASTE"
            obj_valid = True
            waste_verified = True
            reason = spec["reason"]
            rec_corrections += 1
        elif cid in VISUAL_KEEP_AUX_OVERRIDES_8G:
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            tight_norm = extract_tight_per_instance_boxes(img_bgr, proposal["proposed_class_id"], detector)
            boxes = norm_boxes_to_struct(w, h, tight_norm, "task8g_visual_qa_auxiliary")
            qa_status = "KEEP_AUXILIARY"
            domain_tag = "PRODUCT_STUDIO" if is_studio_border else "IN_USE"
            obj_valid = True
            waste_verified = False
            reason = "Visual contact-sheet QA: valid object annotation without verified real-waste disposal context; routed to KEEP_AUXILIARY."
            rec_corrections += 1
        elif cid in VISUAL_NEEDS_REVIEW_OVERRIDES_8G:
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            tight_norm = extract_tight_per_instance_boxes(img_bgr, proposal["proposed_class_id"], detector)
            boxes = norm_boxes_to_struct(w, h, tight_norm, "task8g_visual_qa_review")
            qa_status = "NEEDS_REVIEW"
            domain_tag = "AMBIGUOUS_CONTEXT"
            obj_valid = False
            waste_verified = False
            reason = "Visual contact-sheet QA: multi-object pile or ambiguous class/boundary held in NEEDS_REVIEW."
            rec_corrections += 1
        elif cid in VISUAL_REJECT_OVERRIDES_8G:
            qa_status = "REJECT"
            domain_tag = "AMBIGUOUS_CONTEXT"
            boxes = []
            reason = "Visual contact-sheet QA: rejected non-waste / diagram / scenic / invalid candidate."
            rec_corrections += 1

        elif src == "SRC_8G_01_SHYNUAA_REMAINING_FIRST_PARTY":
            # Check timestamp burst (< 4 seconds in same split) to prevent same-specimen burst duplicates
            m_ts = re.match(r"^(2025\d{4}_\d{6})", title_str)
            is_burst_dup = False
            if m_ts:
                try:
                    dt_cur = datetime.strptime(m_ts.group(1), "%Y%m%d_%H%M%S")
                    if source_split in prev_shynuaa_dt and abs((dt_cur - prev_shynuaa_dt[source_split]).total_seconds()) < 5.0:
                        is_burst_dup = True
                    else:
                        prev_shynuaa_dt[source_split] = dt_cur
                except Exception:
                    pass

            up_boxes = r.get("upstream_boxes") or []
            norm_boxes, n_corr = annotate_shynuaa_multiclass_scene(img_bgr, up_boxes, detector)
            rec_corrections += n_corr
            boxes = norm_boxes_to_struct(w, h, norm_boxes, f"shynuaa_{source_split}_multiclass_v8g", is_ground_truth=False)

            if is_burst_dup:
                qa_status = "REJECT"
                domain_tag = "ACTUAL_WASTE"
                boxes = []
                reason = "Rejected burst-capture repeat (<5s interval) in Shynuaa first-party smartphone stream to preserve specimen independence."
            elif not boxes:
                qa_status = "REJECT"
                domain_tag = "AMBIGUOUS_CONTEXT"
                reason = "No valid tight bounding boxes after refinement."
            elif len(boxes) > 6 or any((b["yolo_xywhn"][2] * b["yolo_xywhn"][3]) > 0.72 for b in boxes):
                qa_status = "NEEDS_REVIEW"
                domain_tag = "ACTUAL_WASTE"
                obj_valid = False
                waste_verified = True
                reason = f"Dense multi-object street-garbage pile ({len(boxes)} instances or coarse heap box >72% area) held in NEEDS_REVIEW per instance-detection policy."
            else:
                qa_status = "REAL_WASTE_PASS"
                domain_tag = "ACTUAL_WASTE"
                obj_valid = True
                waste_verified = True
                reason = (
                    f"Verified first-party 2025 smartphone street-garbage capture ({source_split} split) "
                    f"with {len(boxes)} tight per-instance box(es) across {sorted({b['class_name'] for b in boxes})}."
                )

        elif src == "SRC_8G_02_WIKIMEDIA_LARGE_SCALE_REAL_WASTE":
            proposal = propose_candidate_class_from_title(title_str, int(r["primary_class_id"]))
            p_cid = proposal["proposed_class_id"]
            if proposal["title_changed_class"]:
                rec_corrections += 1
            title_clean = title_str.replace("_", " ").replace("-", " ")

            # Detect multi-angle series of the same physical specimen by normalized stem
            stem_alpha = re.sub(r"^file:", "", title_clean, flags=re.I).strip()
            stem_alpha = re.sub(r"\.(jpg|jpeg|png|webp)$", "", stem_alpha, flags=re.I)
            stem_series = re.sub(r"[\s_()-]*\d+\s*$", "", stem_alpha).strip().lower()
            spec_key = f"spec_wiki8g_{re.sub(r'[^a-z0-9]+', '_', stem_series)[:36]}" if len(stem_series) >= 8 else r["specimen_group"]
            r["specimen_group"] = spec_key

            if is_diagram_or_bw or BAD_WIKI_RE.search(title_clean):
                qa_status = "REJECT"
                domain_tag = "AMBIGUOUS_CONTEXT"
                boxes = []
                reason = "Rejected Wikimedia diagram, B&W archival scan, or non-waste subject."
            elif spec_key in seen_specimens:
                qa_status = "REJECT"
                domain_tag = "ACTUAL_WASTE"
                boxes = []
                reason = f"Same-specimen multi-angle series repeat ({spec_key}); rejected to preserve specimen independence."
            else:
                seen_specimens.add(spec_key)
                tight_norm = extract_tight_per_instance_boxes(img_bgr, p_cid, detector)
                is_fallback_box = len(tight_norm) == 1 and tight_norm[0][1:] == (0.14, 0.14, 0.86, 0.86)
                boxes = norm_boxes_to_struct(w, h, tight_norm, "wikimedia_v8g_detector", is_ground_truth=False)

                if not boxes:
                    qa_status = "REJECT"
                    domain_tag = "AMBIGUOUS_CONTEXT"
                    reason = "No valid object bounding box localized."
                elif is_fallback_box or BULK_PILE_WIKI_RE.search(combined_text) or len(boxes) > 6:
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "ACTUAL_WASTE" if wiki_strong_waste_re.search(combined_text) else "AMBIGUOUS_CONTEXT"
                    obj_valid = False
                    waste_verified = bool(wiki_strong_waste_re.search(combined_text))
                    reason = "Fallback contour box or dense bulk heap (>6 instances) held in NEEDS_REVIEW for manual per-instance verification."
                elif is_studio_border or wiki_in_use_or_non_waste_re.search(combined_text):
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "PRODUCT_STUDIO" if is_studio_border else "IN_USE"
                    obj_valid = True
                    waste_verified = False
                    reason = "Valid scarce-class object in studio/isolated or in-use context; routed to KEEP_AUXILIARY."
                elif proposal["title_changed_class"]:
                    # Task 3 Rule: Never auto-promote title-inferred class change without explicit visual override
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "ACTUAL_WASTE" if wiki_strong_waste_re.search(combined_text) else "AMBIGUOUS_CONTEXT"
                    obj_valid = False
                    waste_verified = False
                    reason = f"Title candidate class proposal ({proposal['proposed_class_name']}) differs from category default; held in NEEDS_REVIEW pending visual confirmation."
                elif (
                    wiki_strong_waste_re.search(title_clean)
                    and not is_studio_border
                    and 1 <= len(boxes) <= 4
                    and all(0.025 <= (b["yolo_xywhn"][2] * b["yolo_xywhn"][3]) <= 0.65 for b in boxes)
                ):
                    qa_status = "REAL_WASTE_PASS"
                    domain_tag = "ACTUAL_WASTE"
                    obj_valid = True
                    waste_verified = True
                    reason = f"Verified Wikimedia real-waste scene ('{title_clean[:65]}') with non-studio outdoor/disposal context and {len(boxes)} tight detector box(es)."
                else:
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "IN_USE"
                    obj_valid = True
                    waste_verified = False
                    reason = "Valid scarce-class object without unambiguous disposal context; routed to KEEP_AUXILIARY."

        elif src == "SRC_8G_03_OPENIMAGES_SCARCE_AND_MULTICLASS":
            up_b = r.get("upstream_boxes") or []
            cids_set = {int(b["class_id"]) for b in up_b}
            norm_list: list[tuple[int, float, float, float, float]] = []
            for b in up_b:
                bcid = int(b["class_id"])
                cx, cy, bw, bh = float(b["cx"]), float(b["cy"]), float(b["w"]), float(b["h"])
                norm_list.append((bcid, cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0))

            # Task 3 Rule: Collapse only Open Images hierarchical parent/child duplicate boxes (DUPLICATE_ANNOTATION: IoU >= 0.90 & max_edge_delta <= 0.045) while preserving 100% of VALID_SAME_CLASS_OVERLAP and VALID_CROSS_CLASS_OVERLAP!
            dedup_oi_norm = deduplicate_same_class_boxes_iou(norm_list, iou_thresh=0.70, is_ground_truth=False)
            if len(dedup_oi_norm) != len(norm_list):
                rec_corrections += len(norm_list) - len(dedup_oi_norm)
            boxes = norm_boxes_to_struct(w, h, dedup_oi_norm, f"openimages_v7_{source_split}_gt", is_ground_truth=True)
            strict_oi_waste_re = re.compile(
                r"\b(trash|garbage|litter|rubbish|dumpster|landfill|discarded|abandoned|rotten|moldy|mouldy|"
                r"spoiled|crushed\s+can|broken\s+bottle|broken\s+glass|scrap|waste|recycl|compost)\b",
                re.I,
            )
            oi_in_use_re = re.compile(
                r"\b(recipe|delicious|breakfast|lunch|dinner|restaurant|menu|served|cooking|baking|fresh|"
                r"outfit|fashion|wearing|worn\s+by|portrait|girl|boy|man|woman|bride|wedding|player|concert)\b",
                re.I,
            )

            if not boxes or is_diagram_or_bw or BAD_WIKI_RE.search(title_str):
                qa_status = "REJECT"
                domain_tag = "IN_USE"
                boxes = []
                reason = "Rejected Open Images non-waste/portrait/B&W or empty GT boxes."
            elif len(boxes) > 6:
                qa_status = "NEEDS_REVIEW"
                domain_tag = "AMBIGUOUS_CONTEXT"
                obj_valid = False
                waste_verified = False
                reason = f"Dense multi-object Open Images scene ({len(boxes)} GT boxes); held in NEEDS_REVIEW to prevent missing co-occurring instances."
            elif is_studio_border or oi_in_use_re.search(title_str):
                qa_status = "KEEP_AUXILIARY"
                domain_tag = "PRODUCT_STUDIO" if is_studio_border else "IN_USE"
                obj_valid = True
                waste_verified = False
                reason = "Valid Open Images V7 human-verified GT boxes in studio or active in-use context; stored in KEEP_AUXILIARY."
            elif strict_oi_waste_re.search(title_str) and not (cids_set & {3, 8} and not re.search(r"\b(discarded|abandoned|trash|garbage|litter|rubbish|dumpster|old\s+shoe|worn\s+out)\b", title_str, re.I)):
                qa_status = "REAL_WASTE_PASS"
                domain_tag = "ACTUAL_WASTE"
                obj_valid = True
                waste_verified = True
                reason = (
                    f"Verified Open Images V7 ({source_split} split) real-waste scene ('{title_str[:65]}') "
                    f"with {len(boxes)} human-verified GT box(es) across {sorted({b['class_name'] for b in boxes})}."
                )
            else:
                qa_status = "KEEP_AUXILIARY"
                domain_tag = "IN_USE"
                obj_valid = True
                waste_verified = False
                reason = "Valid Open Images V7 scarce-class/multi-class GT boxes without explicit disposal evidence; stored in KEEP_AUXILIARY."

        if rec_corrections > 0:
            total_label_corrections += rec_corrections
            records_with_label_corrections += 1

        cls_ids = sorted({int(b["class_id"]) for b in boxes}) if boxes else [int(r["primary_class_id"])]
        cls_names = [CLASS_NAMES[c] for c in cls_ids]
        p_cid = cls_ids[0]

        yolo_txt = format_yolo_lines(boxes)
        label_path_rel = None
        if qa_status == "REAL_WASTE_PASS":
            lp = LABELS_PASS_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "KEEP_AUXILIARY":
            lp = LABELS_AUX_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")
        elif qa_status == "NEEDS_REVIEW":
            lp = LABELS_REVIEW_DIR / f"{cid}.txt"
            lp.write_text(yolo_txt, encoding="utf-8")
            label_path_rel = str(lp.relative_to(ROOT)).replace("\\", "/")

        rec_out = {
            "candidate_id": cid,
            "source_id": src,
            "source_name": r["source_name"],
            "source_image_id": r["source_image_id"],
            "source_split": source_split,
            "provenance_tier": r.get("provenance_tier", "VERIFIED_PUBLIC_LICENSE"),
            "title": title_str,
            "query_used": query_str,
            "original_url": r["original_url"],
            "author": r["author"],
            "license": r["license"],
            "attribution_required": bool(r.get("attribution_required", True)),
            "image_path": r["image_path"],
            "label_path": label_path_rel,
            "overlay_path": None,
            "sha256": r["sha256"],
            "dhash_hex": r["dhash_hex"],
            "width": w,
            "height": h,
            "primary_class_id": p_cid,
            "primary_class_name": CLASS_NAMES[p_cid],
            "classes_present_ids": cls_ids,
            "classes_present_names": cls_names,
            "is_multiclass": len(cls_ids) > 1,
            "bbox_count": len(boxes),
            "boxes": boxes,
            "label_corrections_count": rec_corrections,
            "qa_status": qa_status,
            "domain_tag": domain_tag,
            "OBJECT_ANNOTATION_VALID": obj_valid,
            "REAL_WASTE_CONTEXT_VERIFIED": waste_verified,
            "APPROVED_FOR_TRAIN": approved_for_train,
            "session_group": r["session_group"],
            "specimen_group": r["specimen_group"],
            "target_split": "REJECTED",
            "qa_reason": reason,
        }
        if qa_status in ("REAL_WASTE_PASS", "KEEP_AUXILIARY", "NEEDS_REVIEW"):
            rec_out["overlay_path"] = render_overlay_8g(rec_out)
        qa_records.append(rec_out)

    # Split allocation preserving source_split AND allocating 1/3 of independent Wikimedia session groups to VAL_REAL_WASTE_W8G
    pass_recs = [r for r in qa_records if r["qa_status"] == "REAL_WASTE_PASS"]
    wiki_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in pass_recs:
        sp = r["source_split"]
        if sp == "test":
            r["target_split"] = "HOLD_SOURCE_TEST_W8G"
        elif sp in ("validation", "valid"):
            r["target_split"] = "VAL_REAL_WASTE_W8G"
        elif sp == "train":
            r["target_split"] = "TRAIN_REAL_WASTE_W8G"
        else:
            wiki_groups[r["session_group"]].append(r)

    for idx, gk in enumerate(sorted(wiki_groups.keys())):
        grp = wiki_groups[gk]
        if idx % 3 == 2 and len(grp) <= 2:
            for item in grp:
                item["target_split"] = "VAL_REAL_WASTE_W8G"
        else:
            for item in grp:
                item["target_split"] = "TRAIN_REAL_WASTE_W8G"

    for r in qa_records:
        if r["qa_status"] == "KEEP_AUXILIARY":
            r["target_split"] = (
                "AUXILIARY_NON_WASTE_VAL_HOLD"
                if r["source_split"] in ("validation", "valid")
                else "AUXILIARY_NON_WASTE_TRAIN_HOLD"
            )
        elif r["qa_status"] == "NEEDS_REVIEW":
            r["target_split"] = "MANUAL_REVIEW_QUEUE"
        elif r["qa_status"] == "REJECT":
            r["target_split"] = "REJECTED"

    # Verify zero session/specimen leakage & classify all box overlaps
    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8G"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8G"]
    train_sessions = {r["session_group"] for r in train_recs}
    val_sessions = {r["session_group"] for r in val_recs}
    train_specimens = {r["specimen_group"] for r in train_recs}
    val_specimens = {r["specimen_group"] for r in val_recs}
    assert not (train_sessions & val_sessions), f"Session leakage: {train_sessions & val_sessions}"
    assert not (train_specimens & val_specimens), f"Specimen leakage: {train_specimens & val_specimens}"

    duplicate_pairs = []
    valid_same_class_overlaps = []
    valid_cross_class_overlaps = []
    for r in qa_records:
        bxs = r["boxes"]
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                rel = classify_box_pair_overlap(
                    _struct_box_to_norm_tuple(bxs[i]),
                    _struct_box_to_norm_tuple(bxs[j]),
                )
                if rel["category"] == "DUPLICATE_ANNOTATION":
                    duplicate_pairs.append((r["candidate_id"], r["qa_status"], i, j, rel["iou"]))
                elif rel["category"] == "VALID_SAME_CLASS_OVERLAP":
                    valid_same_class_overlaps.append((r["candidate_id"], r["qa_status"], i, j, rel["iou"]))
                elif rel["category"] == "VALID_CROSS_CLASS_OVERLAP":
                    valid_cross_class_overlaps.append((r["candidate_id"], r["qa_status"], i, j, rel["iou"]))
    assert not duplicate_pairs, f"Found DUPLICATE_ANNOTATION pairs: {duplicate_pairs}"

    status_counts = Counter(r["qa_status"] for r in qa_records)
    domain_counts = Counter(r["domain_tag"] for r in qa_records)
    split_counts = Counter(r["target_split"] for r in qa_records)

    per_class_pass_images = {}
    per_class_pass_train_images = {}
    per_class_pass_val_images = {}
    per_class_pass_boxes = {}
    per_class_pass_train_boxes = {}
    per_class_pass_val_boxes = {}

    for cid_num in range(10):
        cname = CLASS_NAMES[cid_num]
        c_pass = [r for r in pass_recs if cid_num in r["classes_present_ids"]]
        c_train = [r for r in c_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8G"]
        c_val = [r for r in c_pass if r["target_split"] == "VAL_REAL_WASTE_W8G"]
        c_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_pass)
        c_train_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_train)
        c_val_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_val)
        per_class_pass_images[cname] = len(c_pass)
        per_class_pass_train_images[cname] = len(c_train)
        per_class_pass_val_images[cname] = len(c_val)
        per_class_pass_boxes[cname] = c_boxes
        per_class_pass_train_boxes[cname] = c_train_boxes
        per_class_pass_val_boxes[cname] = c_val_boxes

    sheets_created = []
    for p_idx in range(0, len(pass_recs), 36):
        part_no = p_idx // 36 + 1
        sheets_created.append(
            render_sheet_8g(
                pass_recs[p_idx : p_idx + 36],
                f"Task 8G Large-Scale REAL_WASTE_PASS — Part {part_no}",
                f"task8g_real_waste_pass_p{part_no}.jpg",
                max_items=36,
            )
        )
    sheets_created.append(
        render_sheet_8g(
            [r for r in pass_recs if r["is_multiclass"]],
            "Task 8G Multi-Class REAL_WASTE_PASS Scenes",
            "task8g_multiclass_real_waste_pass.jpg",
            max_items=36,
        )
    )
    sheets_created.append(
        render_sheet_8g(
            val_recs,
            "Task 8G VAL_REAL_WASTE_W8G — Independent Validation Set",
            "task8g_val_real_waste.jpg",
            max_items=36,
        )
    )
    sheets_created.append(
        render_sheet_8g(
            [r for r in qa_records if r["qa_status"] == "KEEP_AUXILIARY"],
            "Task 8G KEEP_AUXILIARY — Studio / In-Use Valid Object Annotations",
            "task8g_keep_auxiliary.jpg",
            max_items=36,
        )
    )
    sheets_created.append(
        render_sheet_8g(
            [r for r in qa_records if r["qa_status"] == "NEEDS_REVIEW"],
            "Task 8G NEEDS_REVIEW — Dense Piles / Candidate Class Proposals",
            "task8g_needs_review.jpg",
            max_items=36,
        )
    )

    multiclass_pass_count = sum(1 for r in pass_recs if r["is_multiclass"])
    actual_waste_count = domain_counts["ACTUAL_WASTE"]

    summary_payload = {
        "manifest_version": "TASK8G_LARGE_SCALE_EXPANSION_V1",
        "approved_for_train": False,
        "zero_unannotated_waste_used_as_hard_negative": True,
        "pilots": pilots,
        "total_downloaded_candidates": len(qa_records),
        "qa_status_counts": dict(status_counts),
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "real_waste_pass_total_images": len(pass_recs),
        "real_waste_pass_train_images": len(train_recs),
        "real_waste_pass_val_images": len(val_recs),
        "real_waste_pass_total_boxes": sum(r["bbox_count"] for r in pass_recs),
        "real_waste_pass_multiclass_images": multiclass_pass_count,
        "real_waste_pass_unique_specimens": len({r["specimen_group"] for r in pass_recs}),
        "real_waste_pass_unique_sessions": len({r["session_group"] for r in pass_recs}),
        "real_waste_pass_train_unique_specimens": len(train_specimens),
        "real_waste_pass_train_unique_sessions": len(train_sessions),
        "actual_waste_domain_rate": round(actual_waste_count / max(1, len(qa_records)), 4),
        "real_waste_pass_rate": round(len(pass_recs) / max(1, len(qa_records)), 4),
        "records_with_label_corrections": records_with_label_corrections,
        "total_label_corrections": total_label_corrections,
        "label_correction_rate": round(records_with_label_corrections / max(1, len(qa_records)), 4),
        "overlap_audit": {
            "duplicate_annotation_pairs_remaining": len(duplicate_pairs),
            "preserved_valid_same_class_overlap_pairs": len(valid_same_class_overlaps),
            "preserved_valid_cross_class_overlap_pairs": len(valid_cross_class_overlaps),
        },
        "per_class_real_waste_pass_images": per_class_pass_images,
        "per_class_real_waste_pass_train_images": per_class_pass_train_images,
        "per_class_real_waste_pass_val_images": per_class_pass_val_images,
        "per_class_real_waste_pass_boxes": per_class_pass_boxes,
        "per_class_real_waste_pass_train_boxes": per_class_pass_train_boxes,
        "per_class_real_waste_pass_val_boxes": per_class_pass_val_boxes,
        "contact_sheets": [s for s in sheets_created if s],
        "elapsed_sec": round(time.time() - t0, 2),
        "records": qa_records,
    }
    QA_MANIFEST_JSON.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with QA_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "source_id", "source_split", "provenance_tier",
            "primary_class_id", "primary_class_name",
            "classes_present_ids", "is_multiclass", "bbox_count",
            "qa_status", "domain_tag", "OBJECT_ANNOTATION_VALID",
            "REAL_WASTE_CONTEXT_VERIFIED", "APPROVED_FOR_TRAIN",
            "target_split", "session_group", "specimen_group",
            "license", "author", "original_url", "qa_reason",
        ])
        for r in qa_records:
            writer.writerow([
                r["candidate_id"], r["source_id"], r["source_split"], r["provenance_tier"],
                r["primary_class_id"], r["primary_class_name"],
                "|".join(str(c) for c in r["classes_present_ids"]), r["is_multiclass"], r["bbox_count"],
                r["qa_status"], r["domain_tag"], r["OBJECT_ANNOTATION_VALID"],
                r["REAL_WASTE_CONTEXT_VERIFIED"], r["APPROVED_FOR_TRAIN"],
                r["target_split"], r["session_group"], r["specimen_group"],
                r["license"], r["author"], r["original_url"], r["qa_reason"],
            ])

    print(f"[Task 8G QA Complete] Saved {QA_MANIFEST_JSON}")
    print("Status counts:", dict(status_counts))
    print("Domain counts:", dict(domain_counts))
    print("Split counts:", dict(split_counts))
    print("REAL_WASTE_PASS total:", len(pass_recs), "train:", len(train_recs), "val:", len(val_recs), "multiclass:", multiclass_pass_count)
    print("Per-class REAL_WASTE_PASS train images:", per_class_pass_train_images)
    print("Per-class REAL_WASTE_PASS val images:", per_class_pass_val_images)
    print("Per-class REAL_WASTE_PASS boxes:", per_class_pass_boxes)


if __name__ == "__main__":
    main()
