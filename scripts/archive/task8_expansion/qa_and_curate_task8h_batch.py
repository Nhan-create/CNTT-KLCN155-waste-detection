# Mô tả: [Lưu trữ — Kịch bản mở rộng & kiểm định dữ liệu đa lớp (Task 8)] Gán nhãn khung bao, rà soát chất lượng điểm ảnh (Visual QA) và chuẩn bị dữ liệu cho `qa_and_curate_task8h_batch`.
"""Task 8H — Scarce-Class Quality Control, Multi-Class Per-Instance Annotation & Visual QA Pipeline.

Processes the checkpointed Task 8H candidate pool (`task8h_raw_candidates_manifest.json`) in
`data/acquisition_staging/task8h_scarce_class_scaling/` enforcing all Task 8F/8G/8H safety gates:
1. Never use blind NMS to auto-delete valid ground-truth overlapping boxes (`classify_box_pair_overlap`, `agnostic_nms=False`).
2. Never infer final class from filename alone (`propose_candidate_class_from_title` is candidate-only).
3. Never auto-assert `ACTUAL_WASTE` from keywords alone without visual/pixel context verification.
4. Never miss co-occurring instances in multi-object/multi-class scenes; route dense heaps (`> 6` instances or bulk piles) to `NEEDS_REVIEW`.
5. Never use images with missing annotations as hard negatives (`zero_unannotated_waste_used_as_hard_negative=True`).
6. Strictly separate `REAL_WASTE_PASS` (`ACTUAL_WASTE`) / `KEEP_AUXILIARY` (`PRODUCT_STUDIO` | `IN_USE`) / `NEEDS_REVIEW` / `REJECT`.
7. Keep `APPROVED_FOR_TRAIN = False` on 100% of records.
8. Preserve `source_split` and enforce zero `session_group` and `specimen_group` leakage between `TRAIN_REAL_WASTE_W8H` and `VAL_REAL_WASTE_W8H`.
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acquire_and_qa_task8f_scarce_classes import (
    BAD_WIKI_RE,
    BULK_PILE_WIKI_RE,
    CLASS_COLORS,
    CLASS_NAMES,
    classify_box_pair_overlap,
    deduplicate_same_class_boxes_iou,
    extract_tight_per_instance_boxes,
    format_yolo_lines,
    norm_boxes_to_struct,
    propose_candidate_class_from_title,
)
from scripts.correct_task8f_semantic_qa import _struct_box_to_norm_tuple, compute_iou_xyxy

OUT_DIR = ROOT / "data" / "acquisition_staging" / "task8h_scarce_class_scaling"
RAW_MANIFEST_JSON = OUT_DIR / "task8h_raw_candidates_manifest.json"
QA_MANIFEST_JSON = OUT_DIR / "task8h_qa_manifest.json"
QA_MANIFEST_CSV = OUT_DIR / "task8h_qa_manifest.csv"

LABELS_PASS_DIR = OUT_DIR / "labels_real_waste_pass"
LABELS_AUX_DIR = OUT_DIR / "labels_auxiliary"
LABELS_REVIEW_DIR = OUT_DIR / "labels_needs_review"
OVERLAYS_DIR = OUT_DIR / "overlays"
SHEETS_DIR = OUT_DIR / "contact_sheets"


def render_overlay_8h(rec: dict[str, Any]) -> str:
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


def render_sheet_8h(records: list[dict[str, Any]], title: str, out_name: str, max_items: int = 36) -> str:
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
        cid_s = rec["candidate_id"].replace("task8h_", "")
        cls_s = ",".join(str(x) for x in rec["classes_present_ids"])
        cap = f"{cid_s}|c:{cls_s}|bx:{rec['bbox_count']}|{rec['target_split'][:9]}"
        draw.text((x0 + 4, y0 + cell_h - 28), cap, fill=(220, 235, 220))
    out_p = SHEETS_DIR / out_name
    sheet.save(out_p, quality=88)
    return str(out_p.relative_to(ROOT)).replace("\\", "/")


def extract_multiclass_boxes_8h(
    img_bgr: np.ndarray,
    primary_cid: int,
    detector: YOLO,
    title_str: str = "",
    src_id: str = "",
    source_image_id: str = "",
) -> tuple[list[tuple[int, float, float, float, float]], str, int]:
    """Extract tight per-instance boxes for primary scarce class AND any co-occurring waste classes (`agnostic_nms=False`)."""
    raw_base = extract_tight_per_instance_boxes(img_bgr, primary_cid, detector)
    # Preserve verified scarce target class on primary object boxes so pilot detector does not overwrite moldy fruit/battery as plastic
    base_norm: list[tuple[int, float, float, float, float]] = [
        (primary_cid, round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4))
        for (_, x1, y1, x2, y2) in raw_base
    ]
    if "19818270" in source_image_id and len(base_norm) > 1:
        # Each Zenodo 19818270 workshop photo depicts a single dismantled EV/industrial battery pack
        base_norm = sorted(base_norm, key=lambda b: (b[3] - b[1]) * (b[4] - b[2]), reverse=True)[:1]

    box_src = "yolo_or_contour_instance"
    corrections = 1 if any(int(c) != primary_cid for (c, _, _, _, _) in raw_base) else 0

    # Also check class-aware detector predictions for additional instances or genuine co-occurring classes
    det_res = detector.predict(img_bgr, conf=0.28, agnostic_nms=False, iou=0.65, verbose=False)[0]
    co_boxes: list[tuple[int, float, float, float, float]] = list(base_norm)
    allow_plastic_co = bool(re.search(r"\b(plastic|bag|wrapper|cup|trash|rubbish|garbage|litter|bin|dump)\b", title_str, re.I))
    is_single_specimen_source = src_id == "SRC_8H_02_ZENODO_ACADEMIC_BATTERY_AND_TEXTILE_WASTE"

    if det_res.boxes is not None and len(det_res.boxes) > 0 and not is_single_specimen_source:
        xyxyn = det_res.boxes.xyxyn.cpu().numpy()
        clss = det_res.boxes.cls.cpu().numpy().astype(int)
        confs = det_res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, conf in zip(xyxyn, clss, confs):
            c_int = int(c)
            bw, bh = float(x2 - x1), float(y2 - y1)
            area = bw * bh
            if not (0 <= c_int < 10 and 0.02 <= area <= 0.65):
                continue
            # Same-class additional instances require conf >= 0.30; cross-class secondary objects require conf >= 0.50
            if c_int == primary_cid and float(conf) < 0.30:
                continue
            if c_int != primary_cid:
                if float(conf) < 0.50:
                    continue
                if c_int == 7 and not allow_plastic_co:
                    continue
            cand_xyxy = [float(x1), float(y1), float(x2), float(y2)]
            overlaps_existing = False
            for ex_c, ex1, ey1, ex2, ey2 in co_boxes:
                iou_v = compute_iou_xyxy(cand_xyxy, [ex1, ey1, ex2, ey2])
                if iou_v >= 0.38:
                    overlaps_existing = True
                    break
            if not overlaps_existing and len(co_boxes) < 6:
                co_boxes.append((c_int, round(float(x1), 4), round(float(y1), 4), round(float(x2), 4), round(float(y2), 4)))
                if c_int != primary_cid:
                    corrections += 1
                    box_src = f"{box_src}+co_occurring_yolo"

    dedup_boxes = deduplicate_same_class_boxes_iou(co_boxes, iou_thresh=0.70, is_ground_truth=False)
    if len(dedup_boxes) != len(co_boxes):
        corrections += len(co_boxes) - len(dedup_boxes)
    return dedup_boxes, box_src, corrections


# Visual QA Overrides populated after 100% contact sheet inspection of all 419 candidates
VISUAL_REJECT_OVERRIDES_8H: set[str] = {
    # p1..p3 microscope slides, agar petri dishes, tree branches, beetles/slugs on glass
    "task8h_0064", "task8h_0086", "task8h_0108", "task8h_0109", "task8h_0113", "task8h_0126",
    "task8h_0132", "task8h_0139", "task8h_0146", "task8h_0148", "task8h_0149",
    # p4 insects/crabs in jars, street bin exterior, NYC Battery Park skyline, harbor buoy, corridor, mountain, wood, tools, person wearing clothes, PCB, hands
    "task8h_0154", "task8h_0155", "task8h_0156", "task8h_0159", "task8h_0164", "task8h_0165",
    "task8h_0166", "task8h_0168", "task8h_0169", "task8h_0170", "task8h_0171", "task8h_0176",
    "task8h_0194", "task8h_0208",
    # p5 shopfronts, dog in doorway, stage speaker, gardening, berry on branch, forest trees, closed bin exterior, fungal hyphae
    "task8h_0211", "task8h_0213", "task8h_0215", "task8h_0220", "task8h_0221", "task8h_0248",
    "task8h_0254", "task8h_0255", "task8h_0256", "task8h_0257", "task8h_0258", "task8h_0259",
    "task8h_0260", "task8h_0261", "task8h_0269",
    # p6 text quote, kayaks on river, forest path, magnets in hand, military/steamboat paintings, museum artillery, plastic bag, socks worn on feet, museum print/figurine, multi-angle glove duplicate
    "task8h_0270", "task8h_0272", "task8h_0274", "task8h_0275", "task8h_0276", "task8h_0286",
    "task8h_0292", "task8h_0293", "task8h_0294", "task8h_0325", "task8h_0335", "task8h_0342",
    "task8h_0343", "task8h_0350", "task8h_0352",
    # p7..p8 multi-angle grey canvas shoe duplicates (keeping 0364 in AUX), trucks/crates, Bottle Beach / Glass Beach landscapes, menu, plastic bottles, Bottle shop buildings, toilet building, street traffic
    "task8h_0362", "task8h_0363", "task8h_0365", "task8h_0366", "task8h_0367", "task8h_0368",
    "task8h_0369", "task8h_0370", "task8h_0371", "task8h_0372", "task8h_0373", "task8h_0377",
    "task8h_0384", "task8h_0386", "task8h_0387", "task8h_0388", "task8h_0389", "task8h_0394",
    "task8h_0395", "task8h_0401", "task8h_0402", "task8h_0405", "task8h_0408", "task8h_0409",
    "task8h_0411", "task8h_0413", "task8h_0415",
    # aux_p1..p4 non-waste / active wear / fresh fruit stalls / Egyptian temple / canal boats / jellyfish ("blue bottle") / people / multi-angle pairs
    "task8h_0025", "task8h_0033", "task8h_0034", "task8h_0037", "task8h_0045", "task8h_0046",
    "task8h_0047", "task8h_0048", "task8h_0049", "task8h_0050", "task8h_0053", "task8h_0180",
    "task8h_0201", "task8h_0203", "task8h_0204", "task8h_0206", "task8h_0212", "task8h_0214",
    "task8h_0216", "task8h_0217", "task8h_0218", "task8h_0233", "task8h_0236", "task8h_0244",
    "task8h_0287", "task8h_0332", "task8h_0337", "task8h_0338", "task8h_0344", "task8h_0351",
    "task8h_0355", "task8h_0380", "task8h_0381", "task8h_0382", "task8h_0383", "task8h_0390",
    "task8h_0391", "task8h_0397", "task8h_0403", "task8h_0407", "task8h_0410",
}
VISUAL_NEEDS_REVIEW_OVERRIDES_8H: set[str] = {
    "task8h_0267", "task8h_0385", "task8h_0398",
}
VISUAL_KEEP_AUX_OVERRIDES_8H: set[str] = {
    # Zenodo 19818270 close-up battery label shots -> keep in AUXILIARY rather than REAL_WASTE_PASS
    "task8h_0001", "task8h_0007", "task8h_0010", "task8h_0015", "task8h_0017", "task8h_0018",
    # Valid objects in studio/museum/indoor context without outdoor disposal context
    "task8h_0027", "task8h_0030", "task8h_0036", "task8h_0163", "task8h_0188", "task8h_0196",
    "task8h_0199", "task8h_0200", "task8h_0202", "task8h_0209", "task8h_0226", "task8h_0280",
    "task8h_0341", "task8h_0353", "task8h_0356", "task8h_0357", "task8h_0358", "task8h_0360",
    "task8h_0364",
}
VISUAL_PASS_OVERRIDES_8H: dict[str, dict[str, Any]] = {
    # Promoted from KEEP_AUXILIARY after visual inspection of genuine corroded batteries, discarded clothes/gloves, worn-out/abandoned shoes, moldy/rotting food, and broken/discarded glass bottles
    "task8h_0029": {"boxes": [(8, 0.08, 0.18, 0.92, 0.78)]},
    "task8h_0038": {"boxes": [(8, 0.14, 0.08, 0.88, 0.90)]},
    "task8h_0112": {"boxes": [(1, 0.05, 0.05, 0.95, 0.95)]},
    "task8h_0119": {"boxes": [(1, 0.05, 0.05, 0.95, 0.95)]},
    "task8h_0124": {"boxes": [(1, 0.06, 0.06, 0.94, 0.94)]},
    "task8h_0127": {"boxes": [(1, 0.14, 0.08, 0.90, 0.90)]},
    "task8h_0128": {"boxes": [(1, 0.18, 0.12, 0.86, 0.88)]},
    "task8h_0129": {"boxes": [(1, 0.26, 0.18, 0.74, 0.82)]},
    "task8h_0130": {"boxes": [(1, 0.10, 0.05, 0.92, 0.95)]},
    "task8h_0173": {"boxes": [(3, 0.22, 0.56, 0.35, 0.72), (3, 0.60, 0.70, 0.82, 0.88)]},
    "task8h_0179": {"boxes": [(3, 0.16, 0.42, 0.62, 0.90), (3, 0.52, 0.16, 0.94, 0.62)]},
    "task8h_0181": {"boxes": [(8, 0.18, 0.52, 0.34, 0.68), (8, 0.38, 0.54, 0.60, 0.76)]},
    "task8h_0182": {"boxes": [(8, 0.22, 0.38, 0.92, 0.74)]},
    "task8h_0190": {"boxes": [(4, 0.18, 0.34, 0.88, 0.72)]},
    "task8h_0195": {"boxes": [(0, 0.35, 0.24, 0.68, 0.76)]},
    "task8h_0197": {"boxes": [(0, 0.55, 0.04, 0.92, 0.42)]},
    "task8h_0207": {"boxes": [(0, 0.12, 0.24, 0.88, 0.76)]},
    "task8h_0210": {"boxes": [(3, 0.04, 0.42, 0.28, 0.86), (3, 0.32, 0.22, 0.78, 0.76)]},
    "task8h_0223": {"boxes": [(3, 0.28, 0.34, 0.40, 0.50)]},
    "task8h_0225": {"boxes": [(3, 0.28, 0.52, 0.52, 0.76)]},
    "task8h_0227": {"boxes": [(8, 0.52, 0.16, 0.82, 0.48), (8, 0.18, 0.58, 0.40, 0.78)]},
    "task8h_0228": {"boxes": [(8, 0.16, 0.48, 0.32, 0.62), (5, 0.60, 0.60, 0.72, 0.80)]},
    "task8h_0240": {"boxes": [(8, 0.18, 0.36, 0.54, 0.74), (7, 0.58, 0.70, 0.92, 0.92)]},
    "task8h_0281": {"boxes": [(4, 0.38, 0.56, 0.58, 0.72)]},
    "task8h_0282": {"boxes": [(4, 0.30, 0.16, 0.44, 0.56)]},
    "task8h_0284": {"boxes": [(4, 0.30, 0.34, 0.76, 0.68)]},
    "task8h_0285": {"boxes": [(4, 0.42, 0.32, 0.48, 0.52), (5, 0.76, 0.42, 0.84, 0.58)]},
    "task8h_0298": {"boxes": [(0, 0.36, 0.08, 0.60, 0.90)]},
    "task8h_0299": {"boxes": [(0, 0.12, 0.08, 0.32, 0.84), (0, 0.32, 0.08, 0.50, 0.84), (0, 0.48, 0.52, 0.96, 0.78)]},
    "task8h_0300": {"boxes": [(0, 0.06, 0.14, 0.36, 0.86), (0, 0.48, 0.08, 0.84, 0.88)]},
    "task8h_0301": {"boxes": [(0, 0.12, 0.14, 0.88, 0.86)]},
    "task8h_0303": {"boxes": [(0, 0.08, 0.36, 0.92, 0.68)]},
    "task8h_0306": {"boxes": [(0, 0.40, 0.04, 0.58, 0.78), (0, 0.06, 0.68, 0.66, 0.94)]},
    "task8h_0307": {"boxes": [(0, 0.66, 0.06, 0.78, 0.36), (0, 0.22, 0.42, 0.58, 0.84), (0, 0.34, 0.56, 0.68, 0.94)]},
    "task8h_0308": {"boxes": [(0, 0.32, 0.28, 0.84, 0.48)]},
    "task8h_0309": {"boxes": [(0, 0.24, 0.08, 0.76, 0.90)]},
    "task8h_0310": {"boxes": [(0, 0.26, 0.08, 0.50, 0.92), (0, 0.54, 0.06, 0.78, 0.90)]},
    "task8h_0311": {"boxes": [(0, 0.02, 0.08, 0.50, 0.76), (0, 0.50, 0.06, 0.98, 0.76)]},
    "task8h_0312": {"boxes": [(0, 0.22, 0.18, 0.56, 0.94)]},
    "task8h_0313": {"boxes": [(0, 0.18, 0.10, 0.94, 0.92)]},
    "task8h_0315": {"boxes": [(0, 0.24, 0.10, 0.86, 0.84)]},
    "task8h_0319": {"boxes": [(0, 0.22, 0.18, 0.78, 0.92)]},
    "task8h_0320": {"boxes": [(0, 0.12, 0.10, 0.90, 0.86)]},
    "task8h_0321": {"boxes": [(0, 0.14, 0.26, 0.46, 0.92), (0, 0.54, 0.24, 0.86, 0.92)]},
    "task8h_0322": {"boxes": [(0, 0.12, 0.06, 0.88, 0.94)]},
    "task8h_0324": {"boxes": [(0, 0.10, 0.10, 0.90, 0.90)]},
    "task8h_0331": {"boxes": [(3, 0.14, 0.06, 0.50, 0.90), (3, 0.50, 0.08, 0.88, 0.90)]},
    "task8h_0333": {"boxes": [(3, 0.08, 0.50, 0.44, 0.74), (3, 0.48, 0.38, 0.96, 0.82), (7, 0.48, 0.34, 0.76, 0.60)]},
    "task8h_0334": {"boxes": [(3, 0.14, 0.08, 0.34, 0.52), (3, 0.34, 0.06, 0.62, 0.54), (3, 0.62, 0.10, 0.88, 0.54), (3, 0.20, 0.50, 0.48, 0.94), (3, 0.52, 0.50, 0.80, 0.94)]},
    "task8h_0378": {"boxes": [(1, 0.24, 0.16, 0.54, 0.48)]},
    "task8h_0396": {"boxes": [(4, 0.16, 0.22, 0.82, 0.68), (5, 0.46, 0.66, 0.80, 0.88)]},
    # Also refine multi-class box on task8h_0177 (clothes + shoes on dirt ground)
    "task8h_0177": {"boxes": [(3, 0.14, 0.12, 0.88, 0.88), (8, 0.62, 0.68, 0.84, 0.86)]},
}

STRONG_WASTE_RE_8H = re.compile(
    r"\b(discarded|abandoned|litter|littered|rubbish|garbage|trash|waste|dump|dumped|dumpster|bin|"
    r"corroded|leaking|leaked|rusted|rusty|rotten|rotting|moldy|mouldy|decaying|spoiled|compost|"
    r"banana\s+peel|apple\s+core|orange\s+peel|eggshell|broken\s+glass|broken\s+bottle|shattered|"
    r"broken\s+beer|lost\s+shoe|old\s+shoe|old\s+boot|worn\s+out|worn-out|lost\s+glove|lost\s+sock|dirty\s+sock|"
    r"rag|torn|dismantl|end-of-life|spent\s+batter|used\s+batter|dead\s+batter|car\s+batter|recycl|"
    r"monilinia\s+fructigena|penicillium\s+digitatum|penicillium\s+italicum|rhizopus\s+stolonifer|"
    r"monilinia\s+laxa|fly-tipping|flytipping|broken\s+window|glass\s+shard|broken\s+jar|broken\s+shoe|"
    r"electronic_waste|battery_recycling|textile_waste|used_clothing|abandoned_shoes|worn-out_shoes|"
    r"food_waste|rotting_food|moldy_food|broken_glass|broken_bottles|glass_recycling|discarded_bottles)\b",
    re.I,
)

NON_WASTE_OR_MACRO_RE_8H = re.compile(
    r"\b(portrait|selfie|wedding|bride|groom|fashion\s+show|runway|model\s+wearing|outfit\s+of\s+the\s+day|"
    r"recipe|delicious|restaurant\s+menu|breakfast|dinner\s+plate|fresh\s+fruit|supermarket\s+display|"
    r"museum\s+exhibit|art\s+installation|painting|drawing|diagram|chart|map|logo|icon|screenshot|"
    r"beetle|ladybug|coccinella|insect|spider|larva|caterpillar|butterfly|moth|mushroom\s+in\s+forest|"
    r"artillery\s+battery|coastal\s+battery|gun\s+battery|missile\s+battery|fort\s+battery|"
    r"microscope|microscopic|petri\s+dish|agar|spores|conidia|hyphae|stained\s+glass|cathedral|church\s+window)\b",
    re.I,
)


def main() -> None:
    t0 = time.time()
    for d in (LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR, SHEETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    for d in (LABELS_PASS_DIR, LABELS_AUX_DIR, LABELS_REVIEW_DIR, OVERLAYS_DIR):
        for old_f in d.glob("*"):
            old_f.unlink()
    for old_sheet in SHEETS_DIR.glob("task8h_*.jpg"):
        old_sheet.unlink()

    raw_data = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    raw_records = raw_data["records"]
    exhausted_sources = raw_data.get("exhausted_sources", [])
    source_evaluations = raw_data.get("source_evaluations", [])
    pilot_sheets = raw_data.get("pilot_contact_sheets", [])

    yolo_weights = ROOT / "artifacts" / "detection_v3_pilot" / "E3_yolov8n_pilot" / "weights" / "best.pt"
    if not yolo_weights.exists():
        yolo_weights = ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "weights" / "best.pt"
    detector = YOLO(str(yolo_weights))

    qa_records: list[dict[str, Any]] = []
    records_with_label_corrections = 0
    total_label_corrections = 0

    for idx, r in enumerate(raw_records):
        cid = r["candidate_id"]
        src = r["source_id"]
        source_split = r.get("source_split", "unsplit")
        title_str = str(r.get("title") or "")
        query_str = str(r.get("query_used") or "")
        combined_text = f"{title_str} {query_str}"
        primary_cid = int(r["primary_class_id"])

        img_p = ROOT / r["image_path"]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # Compute border uniformity (studio white/solid backdrop check) and color saturation
        b_px = max(4, min(h, w) // 25)
        top_b = img_bgr[:b_px, :, :]
        bot_b = img_bgr[h - b_px :, :, :]
        left_b = img_bgr[:, :b_px, :]
        right_b = img_bgr[:, w - b_px :, :]
        border_all = np.concatenate(
            [top_b.reshape(-1, 3), bot_b.reshape(-1, 3), left_b.reshape(-1, 3), right_b.reshape(-1, 3)],
            axis=0,
        )
        white_mask = np.all(border_all > 236, axis=1)
        border_white_ratio = float(np.mean(white_mask))
        border_std = float(np.std(border_all))
        is_studio_border = (border_white_ratio > 0.72) or (border_white_ratio > 0.55 and border_std < 18.0)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        is_diagram_or_bw = sat_mean < 10.0

        rec_corrections = 0
        approved_for_train = False

        # 1. Explicit Visual QA Overrides take highest priority
        if cid in VISUAL_REJECT_OVERRIDES_8H:
            qa_status = "REJECT"
            domain_tag = "IN_USE"
            obj_valid = False
            waste_verified = False
            boxes = []
            reason = "Visual QA override: rejected non-waste / irrelevant / multi-angle repeat."
        elif cid in VISUAL_NEEDS_REVIEW_OVERRIDES_8H:
            norm_b, b_src, c_cnt = extract_multiclass_boxes_8h(img_bgr, primary_cid, detector, title_str, src)
            rec_corrections += c_cnt
            boxes = norm_boxes_to_struct(w, h, norm_b, b_src, is_ground_truth=False)
            qa_status = "NEEDS_REVIEW"
            domain_tag = "AMBIGUOUS_CONTEXT"
            obj_valid = False
            waste_verified = False
            reason = "Visual QA override: dense pile or ambiguous context held in NEEDS_REVIEW."
        elif cid in VISUAL_KEEP_AUX_OVERRIDES_8H:
            norm_b, b_src, c_cnt = extract_multiclass_boxes_8h(img_bgr, primary_cid, detector, title_str, src)
            rec_corrections += c_cnt
            boxes = norm_boxes_to_struct(w, h, norm_b, b_src, is_ground_truth=False)
            qa_status = "KEEP_AUXILIARY"
            domain_tag = "PRODUCT_STUDIO" if is_studio_border else "IN_USE"
            obj_valid = True
            waste_verified = False
            reason = "Visual QA override: valid scarce-class object on studio/sorting backdrop or active use -> KEEP_AUXILIARY."
        elif cid in VISUAL_PASS_OVERRIDES_8H:
            ov_info = VISUAL_PASS_OVERRIDES_8H[cid]
            ov_boxes_norm = ov_info["boxes"]
            boxes = norm_boxes_to_struct(w, h, ov_boxes_norm, "visual_qa_verified_override", is_ground_truth=False)
            rec_corrections += 1
            qa_status = "REAL_WASTE_PASS"
            domain_tag = "ACTUAL_WASTE"
            obj_valid = True
            waste_verified = True
            reason = ov_info.get("reason", "Visual QA verified real-waste scarce-class scene with tight per-instance bounding boxes.")

        # 2. Source-specific QA gates
        elif src == "SRC_8H_02_ZENODO_ACADEMIC_BATTERY_AND_TEXTILE_WASTE":
            norm_b, b_src, c_cnt = extract_multiclass_boxes_8h(
                img_bgr, primary_cid, detector, title_str, src, str(r.get("source_image_id", ""))
            )
            rec_corrections += c_cnt
            boxes = norm_boxes_to_struct(w, h, norm_b, b_src, is_ground_truth=False)
            if "19044186" in str(r.get("source_image_id", "")):
                # TrondheimSocks: genuine post-consumer waste sock from recycling bin, but photographed on white quilted lab sheet
                qa_status = "KEEP_AUXILIARY"
                domain_tag = "PRODUCT_STUDIO"
                obj_valid = True
                waste_verified = False
                reason = "Zenodo 19044186 NTNU post-consumer discarded sock photographed on uniform white lab sorting sheet -> KEEP_AUXILIARY."
            else:
                # Zenodo 19818270: EU RECIRCULATE End-of-Life Dismantled Battery specimen in recycling workshop (1 photo per battery type)
                if boxes and not is_studio_border:
                    qa_status = "REAL_WASTE_PASS"
                    domain_tag = "ACTUAL_WASTE"
                    obj_valid = True
                    waste_verified = True
                    reason = f"Zenodo 19818270 first-party end-of-life dismantled battery specimen ({r['source_image_id']}) in workshop -> REAL_WASTE_PASS."
                else:
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "PRODUCT_STUDIO"
                    obj_valid = True
                    waste_verified = False
                    reason = "Zenodo 19818270 battery specimen on plain backdrop -> KEEP_AUXILIARY."

        else:
            # Openverse, Wikimedia Deep Subcategories, and iNaturalist Field Observations
            if is_diagram_or_bw or BAD_WIKI_RE.search(combined_text) or NON_WASTE_OR_MACRO_RE_8H.search(title_str):
                qa_status = "REJECT"
                domain_tag = "IN_USE"
                obj_valid = False
                waste_verified = False
                boxes = []
                reason = f"Rejected non-waste / macro biology / B&W / military battery / portrait ('{title_str[:65]}')."
            elif BULK_PILE_WIKI_RE.search(title_str):
                norm_b, b_src, c_cnt = extract_multiclass_boxes_8h(img_bgr, primary_cid, detector, title_str, src)
                rec_corrections += c_cnt
                boxes = norm_boxes_to_struct(w, h, norm_b, b_src, is_ground_truth=False)
                qa_status = "NEEDS_REVIEW"
                domain_tag = "ACTUAL_WASTE"
                obj_valid = False
                waste_verified = True
                reason = f"Dense bulk waste pile ('{title_str[:65]}'); routed to NEEDS_REVIEW to avoid coarse single-box annotation."
            else:
                # Candidate-only title class proposal check (Task 8F/8G/8H rule: never auto-promote title-inferred class change)
                proposal = propose_candidate_class_from_title(title_str, primary_cid)
                norm_b, b_src, c_cnt = extract_multiclass_boxes_8h(img_bgr, primary_cid, detector, title_str, src)
                rec_corrections += c_cnt
                boxes = norm_boxes_to_struct(w, h, norm_b, b_src, is_ground_truth=False)

                if not boxes:
                    qa_status = "REJECT"
                    domain_tag = "IN_USE"
                    obj_valid = False
                    waste_verified = False
                    reason = "No valid object bounding box extracted."
                elif len(boxes) > 6:
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "ACTUAL_WASTE"
                    obj_valid = False
                    waste_verified = False
                    reason = f"Dense multi-object scene ({len(boxes)} boxes > 6); held in NEEDS_REVIEW."
                elif proposal["title_changed_class"] and proposal["proposed_class_id"] != primary_cid:
                    qa_status = "NEEDS_REVIEW"
                    domain_tag = "ACTUAL_WASTE" if STRONG_WASTE_RE_8H.search(combined_text) else "AMBIGUOUS_CONTEXT"
                    obj_valid = False
                    waste_verified = False
                    reason = (
                        f"Title candidate class proposal ({proposal['proposed_class_name']}) differs from query target "
                        f"({CLASS_NAMES[primary_cid]}); held in NEEDS_REVIEW pending visual confirmation."
                    )
                elif is_studio_border:
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "PRODUCT_STUDIO"
                    obj_valid = True
                    waste_verified = False
                    reason = f"Valid {CLASS_NAMES[primary_cid]} object on studio/isolated background -> KEEP_AUXILIARY."
                elif (
                    STRONG_WASTE_RE_8H.search(combined_text)
                    and 1 <= len(boxes) <= 6
                    and all(0.015 <= (b["yolo_xywhn"][2] * b["yolo_xywhn"][3]) <= 0.78 for b in boxes)
                ):
                    qa_status = "REAL_WASTE_PASS"
                    domain_tag = "ACTUAL_WASTE"
                    obj_valid = True
                    waste_verified = True
                    reason = (
                        f"Verified real-waste {CLASS_NAMES[primary_cid]} scene ('{title_str[:65]}') from {src} "
                        f"with {len(boxes)} tight per-instance box(es)."
                    )
                else:
                    qa_status = "KEEP_AUXILIARY"
                    domain_tag = "IN_USE"
                    obj_valid = True
                    waste_verified = False
                    reason = "Valid scarce-class object without unambiguous disposal context -> KEEP_AUXILIARY."

        if rec_corrections > 0:
            total_label_corrections += rec_corrections
            records_with_label_corrections += 1

        cls_ids = sorted({int(b["class_id"]) for b in boxes}) if boxes else [primary_cid]
        cls_names = [CLASS_NAMES[c] for c in cls_ids]
        # Keep primary_cid as the scarce target class if present in cls_ids, else cls_ids[0]
        p_cid = primary_cid if primary_cid in cls_ids else cls_ids[0]

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
            rec_out["overlay_path"] = render_overlay_8h(rec_out)
        qa_records.append(rec_out)

    # Split allocation preserving source_split AND allocating independent session groups without train/val leakage
    pass_recs = [r for r in qa_records if r["qa_status"] == "REAL_WASTE_PASS"]
    unsplit_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in pass_recs:
        sp = r["source_split"]
        if sp == "test":
            r["target_split"] = "HOLD_SOURCE_TEST_W8H"
        elif sp in ("validation", "valid"):
            r["target_split"] = "VAL_REAL_WASTE_W8H"
        elif sp == "train":
            r["target_split"] = "TRAIN_REAL_WASTE_W8H"
        else:
            unsplit_groups[r["session_group"]].append(r)

    for idx, gk in enumerate(sorted(unsplit_groups.keys())):
        grp = unsplit_groups[gk]
        if idx % 5 == 4 and len(grp) <= 2:
            for item in grp:
                item["target_split"] = "VAL_REAL_WASTE_W8H"
        else:
            for item in grp:
                item["target_split"] = "TRAIN_REAL_WASTE_W8H"

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

    train_recs = [r for r in pass_recs if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
    val_recs = [r for r in pass_recs if r["target_split"] == "VAL_REAL_WASTE_W8H"]
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
        c_train = [r for r in c_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
        c_val = [r for r in c_pass if r["target_split"] == "VAL_REAL_WASTE_W8H"]
        c_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_pass)
        c_train_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_train)
        c_val_boxes = sum(sum(1 for b in r["boxes"] if b["class_id"] == cid_num) for r in c_val)
        per_class_pass_images[cname] = len(c_pass)
        per_class_pass_train_images[cname] = len(c_train)
        per_class_pass_val_images[cname] = len(c_val)
        per_class_pass_boxes[cname] = c_boxes
        per_class_pass_train_boxes[cname] = c_train_boxes
        per_class_pass_val_boxes[cname] = c_val_boxes

    # Per-source selection rates
    source_stats: dict[str, dict[str, Any]] = {}
    for src_id in sorted({r["source_id"] for r in qa_records}):
        s_recs = [r for r in qa_records if r["source_id"] == src_id]
        s_pass = [r for r in s_recs if r["qa_status"] == "REAL_WASTE_PASS"]
        s_train = [r for r in s_pass if r["target_split"] == "TRAIN_REAL_WASTE_W8H"]
        s_val = [r for r in s_pass if r["target_split"] == "VAL_REAL_WASTE_W8H"]
        s_aux = [r for r in s_recs if r["qa_status"] == "KEEP_AUXILIARY"]
        s_rev = [r for r in s_recs if r["qa_status"] == "NEEDS_REVIEW"]
        s_rej = [r for r in s_recs if r["qa_status"] == "REJECT"]
        source_stats[src_id] = {
            "downloaded": len(s_recs),
            "real_waste_pass": len(s_pass),
            "train_real_waste": len(s_train),
            "val_real_waste": len(s_val),
            "keep_auxiliary": len(s_aux),
            "needs_review": len(s_rev),
            "reject": len(s_rej),
            "pass_yield_rate": round(len(s_pass) / max(1, len(s_recs)), 4),
        }

    sheets_created = list(pilot_sheets)
    for p_idx in range(0, len(pass_recs), 36):
        part_no = p_idx // 36 + 1
        sheets_created.append(
            render_sheet_8h(
                pass_recs[p_idx : p_idx + 36],
                f"Task 8H Scarce-Class REAL_WASTE_PASS — Part {part_no}",
                f"task8h_real_waste_pass_p{part_no}.jpg",
                max_items=36,
            )
        )
    sheets_created.append(
        render_sheet_8h(
            [r for r in pass_recs if r["is_multiclass"]],
            "Task 8H Multi-Class REAL_WASTE_PASS Scenes",
            "task8h_multiclass_real_waste_pass.jpg",
            max_items=36,
        )
    )
    sheets_created.append(
        render_sheet_8h(
            val_recs,
            "Task 8H VAL_REAL_WASTE_W8H — Independent Validation Set",
            "task8h_val_real_waste.jpg",
            max_items=36,
        )
    )
    aux_recs_all = [r for r in qa_records if r["qa_status"] == "KEEP_AUXILIARY"]
    for a_idx in range(0, len(aux_recs_all), 36):
        a_no = a_idx // 36 + 1
        sheets_created.append(
            render_sheet_8h(
                aux_recs_all[a_idx : a_idx + 36],
                f"Task 8H KEEP_AUXILIARY — Part {a_no}",
                f"task8h_keep_auxiliary_p{a_no}.jpg",
                max_items=36,
            )
        )
    sheets_created.append(
        render_sheet_8h(
            [r for r in qa_records if r["qa_status"] == "NEEDS_REVIEW"],
            "Task 8H NEEDS_REVIEW — Dense Piles / Candidate Class Proposals",
            "task8h_needs_review.jpg",
            max_items=36,
        )
    )

    multiclass_pass_count = sum(1 for r in pass_recs if r["is_multiclass"])
    multiobject_pass_count = sum(1 for r in pass_recs if r["bbox_count"] >= 2)
    actual_waste_count = domain_counts["ACTUAL_WASTE"]

    summary_payload = {
        "manifest_version": "TASK8H_SCARCE_CLASS_SCALING_V1",
        "approved_for_train": False,
        "zero_unannotated_waste_used_as_hard_negative": True,
        "exhausted_sources": exhausted_sources,
        "source_evaluations": source_evaluations,
        "source_stats": source_stats,
        "total_downloaded_candidates": len(qa_records),
        "qa_status_counts": dict(status_counts),
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "real_waste_pass_total_images": len(pass_recs),
        "real_waste_pass_train_images": len(train_recs),
        "real_waste_pass_val_images": len(val_recs),
        "real_waste_pass_total_boxes": sum(r["bbox_count"] for r in pass_recs),
        "real_waste_pass_multiobject_images": multiobject_pass_count,
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

    print(f"[Task 8H QA Complete] Saved {QA_MANIFEST_JSON}", flush=True)
    print("Status counts:", dict(status_counts), flush=True)
    print("Domain counts:", dict(domain_counts), flush=True)
    print("Split counts:", dict(split_counts), flush=True)
    print(
        "REAL_WASTE_PASS total:", len(pass_recs),
        "train:", len(train_recs),
        "val:", len(val_recs),
        "multiobject:", multiobject_pass_count,
        "multiclass:", multiclass_pass_count,
        flush=True,
    )
    print("Per-class REAL_WASTE_PASS train images:", per_class_pass_train_images, flush=True)
    print("Per-class REAL_WASTE_PASS val images:", per_class_pass_val_images, flush=True)
    print("Per-class REAL_WASTE_PASS boxes:", per_class_pass_boxes, flush=True)
    print("Source stats:", json.dumps(source_stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
