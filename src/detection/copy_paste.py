"""Copy-Paste Data Augmentation Module for Waste Detection (Phase 2).

Implements Simple Copy-Paste (Ghiasi et al., 2021) [5] using verified instance
segmentation masks from training data with rigorous pixel-level mask occlusion handling.

Core Academic Principles:
1. Strict Split Isolation: Donor objects are extracted EXCLUSIVELY from the training split.
   Validation and test splits are NEVER touched or used as donors or backgrounds.
2. Verified Masks Only: Uses hand-annotated polygon segmentation masks from TACO raw annotations.
   Rectangular bounding-box crops are NOT used as fake masks.
3. True Mask Occlusion & Union Tracking:
   - Occlusion is calculated from actual binary/alpha masks of donors, NOT rectangular bounding boxes.
   - Mask union (cumulative mask union) is maintained so overlapping donors do not double-count occlusion.
   - Donor-on-donor occlusion: Subsequent donors track and occlude previous donors, updating tight bounding boxes.
4. Conservative Bounding-Box Heuristic:
   - When background objects have instance masks, true pixel-level remaining visibility is calculated.
   - When background objects only have bounding boxes, the donor mask intersection ratio with the box is used.
     Conservative policy: boxes are dropped ONLY if mask occlusion ratio >= 0.80 (remaining visibility < 0.20).
     Bounding box overlap alone is NEVER treated as sufficient evidence to delete labels.
5. Synchronized Output & Comprehensive Telemetry:
   - Returns transformed RGB image, updated YOLO-format bounding boxes [cx, cy, w, h], and class IDs.
   - Detailed telemetry logs donor IDs, provenance, positions, scales, occlusion ratios, and explicit drop reasons.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml


@dataclass
class DonorObject:
    """Foreground object extracted with pixel-level mask from training data."""
    image_rgba: np.ndarray  # H x W x 4 (RGB + Alpha mask uint8)
    binary_mask: np.ndarray  # H x W bool
    class_id: int
    class_name: str
    source_image_id: str
    original_bbox: list[float]  # [x, y, w, h] in pixels
    donor_id: str = ""
    source_annotation_id: str = ""

    def __post_init__(self):
        if not self.donor_id:
            self.donor_id = f"donor_{self.source_image_id}_{self.class_name}"


class DonorBank:
    """Thread-safe in-memory cache of verified foreground donor objects from train split."""

    def __init__(self, donors: list[DonorObject] | None = None):
        self.donors = donors or []
        self._by_class: dict[int, list[DonorObject]] = {}
        for d in self.donors:
            self._by_class.setdefault(d.class_id, []).append(d)

    def __len__(self) -> int:
        return len(self.donors)

    def get_random_donor(self, class_id: int | None = None) -> DonorObject | None:
        if not self.donors:
            return None
        if class_id is not None:
            candidates = self._by_class.get(class_id, [])
            return random.choice(candidates) if candidates else None
        return random.choice(self.donors)

    @classmethod
    def from_dataset_root(
        cls,
        root_dir: str | Path,
        max_donors_per_image: int = 10,
    ) -> DonorBank:
        """Extract verified donor objects from train split records only."""
        root = Path(root_dir)
        manifest_path = root / "data" / "detection" / "manifest.jsonl"
        taco_raw_path = root / "data" / "audit" / "taco_annotations_raw.json"
        mapping_path = root / "configs" / "detection_source_mapping.yaml"

        if not manifest_path.is_file() or not taco_raw_path.is_file() or not mapping_path.is_file():
            return cls([])

        # 1. Load category mapping from TACO to 10 classes
        with open(mapping_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        taco_cat_map = cfg.get("sources", {}).get("taco", {}).get("category_mapping", {})
        # Map taco_cat_id -> (class_id, class_name)
        cat_id_to_project = {}
        for k, v in taco_cat_map.items():
            if v and v.get("class_id") is not None:
                cat_id_to_project[int(k)] = (int(v["class_id"]), str(v["class_name"]))

        # 2. Get verified train TACO image IDs (Strict Train Split Isolation)
        train_taco_ids: set[int] = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("split") == "train" and record.get("is_real", False):
                    try:
                        train_taco_ids.add(int(record["image_id"]))
                    except (ValueError, KeyError):
                        pass

        # 3. Load TACO raw annotations
        with open(taco_raw_path, "r", encoding="utf-8") as f:
            taco = json.load(f)

        img_id_to_meta = {img["id"]: img for img in taco.get("images", [])}
        anns_by_img: dict[int, list[dict]] = {}
        for a in taco.get("annotations", []):
            if a.get("image_id") in train_taco_ids and a.get("segmentation"):
                anns_by_img.setdefault(a["image_id"], []).append(a)

        donors: list[DonorObject] = []
        for img_id in sorted(train_taco_ids):
            img_meta = img_id_to_meta.get(img_id)
            if not img_meta:
                continue
            img_file = root / "data" / "detection" / "images" / "real" / f"taco_{img_id:04d}.jpg"
            if not img_file.is_file():
                continue

            try:
                with Image.open(img_file) as opened:
                    base_img = opened.convert("RGB")
            except Exception:
                continue

            w_img, h_img = base_img.size
            raw_w = float(img_meta.get("width", w_img))
            raw_h = float(img_meta.get("height", h_img))
            sx = w_img / raw_w if raw_w > 0 else 1.0
            sy = h_img / raw_h if raw_h > 0 else 1.0

            anns = anns_by_img.get(img_id, [])[:max_donors_per_image]
            for ann_idx, ann in enumerate(anns):
                cat_info = cat_id_to_project.get(ann.get("category_id"))
                if not cat_info:
                    continue
                class_id, class_name = cat_info
                segs = ann.get("segmentation")
                if not segs or not isinstance(segs, list) or not segs[0]:
                    continue

                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue
                raw_bx, raw_by, raw_bw, raw_bh = bbox
                bx = raw_bx * sx
                by = raw_by * sy
                bw = raw_bw * sx
                bh = raw_bh * sy

                if bw < 5 or bh < 5 or bx < 0 or by < 0:
                    continue

                # Crop bounding box with margin
                x1 = max(0, int(bx))
                y1 = max(0, int(by))
                x2 = min(w_img, int(bx + bw))
                y2 = min(h_img, int(by + bh))

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < 5 or crop_h < 5:
                    continue

                # Draw polygon mask on crop
                mask_crop = Image.new("L", (crop_w, crop_h), 0)
                draw = ImageDraw.Draw(mask_crop)
                for poly in segs:
                    if len(poly) >= 6:
                        poly_rel = [((poly[i] * sx) - x1, (poly[i + 1] * sy) - y1) for i in range(0, len(poly), 2)]
                        draw.polygon(poly_rel, fill=255)

                mask_arr = np.array(mask_crop) > 128
                if not np.any(mask_arr):
                    continue

                crop_rgb = np.array(base_img.crop((x1, y1, x2, y2)))
                alpha_ch = (mask_arr.astype(np.uint8) * 255)[:, :, np.newaxis]
                crop_rgba = np.concatenate([crop_rgb, alpha_ch], axis=2)

                ann_id = str(ann.get("id", f"idx{ann_idx}"))
                donor_id = f"donor_taco_{img_id:04d}_ann_{ann_id}_{class_name}"

                donors.append(
                    DonorObject(
                        donor_id=donor_id,
                        image_rgba=crop_rgba,
                        binary_mask=mask_arr,
                        class_id=class_id,
                        class_name=class_name,
                        source_image_id=str(img_id),
                        original_bbox=[float(bx), float(by), float(bw), float(bh)],
                        source_annotation_id=ann_id,
                    )
                )

        return cls(donors)


def apply_copy_paste(
    image: np.ndarray,
    boxes: list[list[float]],
    category_ids: list[int],
    donor_bank: DonorBank,
    background_masks: list[np.ndarray] | None = None,
    p: float = 0.5,
    max_paste: int = 2,
    force_apply: bool = False,
    min_visibility: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[float]], list[int], dict[str, Any]]:
    """Apply Simple Copy-Paste augmentation with exact mask occlusion handling.

    Args:
        image: np.ndarray (H, W, 3) in uint8 RGB.
        boxes: list of YOLO normalized boxes [cx, cy, w, h] in range [0, 1].
        category_ids: list of integer class IDs.
        donor_bank: DonorBank instance containing verified training masks.
        background_masks: Optional list of binary masks (H, W bool) for background objects.
        p: Execution probability (ignored if force_apply is True).
        max_paste: Maximum number of objects to paste (1..max_paste).
        force_apply: If True, forces execution (for verification tests).
        min_visibility: Minimum visibility threshold to retain occluded boxes (default: 0.2).
        seed: Optional random seed for deterministic augmentation.

    Returns:
        transformed_image, transformed_boxes, transformed_category_ids, telemetry.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    telemetry: dict[str, Any] = {
        "operation": "copy_paste",
        "applied": False,
        "forced": force_apply,
        "p": p,
        "donors_available": len(donor_bank),
        "pasted_objects_attempted": 0,
        "pasted_objects_kept": 0,
        "pasted_objects_dropped": 0,
        "boxes_before": len(boxes),
        "boxes_after": len(boxes),
        "background_boxes_kept": len(boxes),
        "background_boxes_dropped": 0,
        "dropped_occluded_count": 0,
        "drop_reasons": [],
        "donor_telemetry": [],
        "background_telemetry": [],
    }

    if not force_apply and random.random() > p:
        return image.copy(), list(boxes), list(category_ids), telemetry

    if len(donor_bank) == 0:
        telemetry["drop_reasons"].append("DonorBank is empty: no valid training masks available")
        return image.copy(), list(boxes), list(category_ids), telemetry

    h_img, w_img = image.shape[:2]
    num_to_paste = random.randint(1, max(1, max_paste))
    out_img = image.copy()

    # Cumulative boolean mask union of all pasted donor masks on canvas
    cumulative_donor_mask = np.zeros((h_img, w_img), dtype=bool)

    # 1. Parse and record initial background objects
    bg_objects: list[dict[str, Any]] = []
    for idx, (b, c) in enumerate(zip(boxes, category_ids)):
        cx, cy, bw, bh = b
        px1 = max(0.0, (cx - bw / 2.0) * w_img)
        py1 = max(0.0, (cy - bh / 2.0) * h_img)
        px2 = min(float(w_img), (cx + bw / 2.0) * w_img)
        py2 = min(float(h_img), (cy + bh / 2.0) * h_img)
        mask = background_masks[idx] if background_masks and idx < len(background_masks) else None

        bg_objects.append({
            "index": idx,
            "box": [px1, py1, px2, py2],
            "class_id": int(c),
            "mask": mask,
            "box_area": max(1.0, (px2 - px1) * (py2 - py1)),
            "status": "KEPT",
            "occlusion_ratio": 0.0,
            "drop_reason": "",
            "final_box_pixel": [px1, py1, px2, py2],
        })

    # 2. Paste donors sequentially with mask union and donor-on-donor occlusion tracking
    pasted_records: list[dict[str, Any]] = []

    for paste_idx in range(num_to_paste):
        donor = donor_bank.get_random_donor()
        if not donor:
            continue

        donor_rgba = donor.image_rgba
        dh, dw = donor_rgba.shape[:2]

        scale = random.uniform(0.6, 1.2)
        target_w = max(8, int(dw * scale))
        target_h = max(8, int(dh * scale))

        if target_w >= w_img or target_h >= h_img:
            fit_scale = min((w_img * 0.5) / dw, (h_img * 0.5) / dh)
            target_w = max(8, int(dw * fit_scale))
            target_h = max(8, int(dh * fit_scale))

        # Scale donor RGB and Alpha mask
        donor_pil = Image.fromarray(donor_rgba).resize((target_w, target_h), Image.Resampling.BILINEAR)
        scaled_rgba = np.array(donor_pil)
        scaled_rgb = scaled_rgba[:, :, :3]
        scaled_alpha = scaled_rgba[:, :, 3] > 128

        if not np.any(scaled_alpha):
            telemetry["drop_reasons"].append(f"Donor {donor.donor_id} alpha mask became empty after scaling")
            continue

        # Choose placement within boundaries
        max_x = max(0, w_img - target_w)
        max_y = max(0, h_img - target_h)
        paste_x = random.randint(0, max_x)
        paste_y = random.randint(0, max_y)

        # Handle canvas placement with safe clipping
        x1_clip = max(0, paste_x)
        y1_clip = max(0, paste_y)
        x2_clip = min(w_img, paste_x + target_w)
        y2_clip = min(h_img, paste_y + target_h)

        dx1 = x1_clip - paste_x
        dy1 = y1_clip - paste_y
        dx2 = dx1 + (x2_clip - x1_clip)
        dy2 = dy1 + (y2_clip - y1_clip)

        donor_crop_alpha = scaled_alpha[dy1:dy2, dx1:dx2]
        donor_crop_rgb = scaled_rgb[dy1:dy2, dx1:dx2]

        if not np.any(donor_crop_alpha):
            telemetry["drop_reasons"].append(f"Donor {donor.donor_id} clipped completely outside visible canvas")
            continue

        # Create canvas-level boolean mask for this donor
        canvas_donor_mask = np.zeros((h_img, w_img), dtype=bool)
        canvas_donor_mask[y1_clip:y2_clip, x1_clip:x2_clip] = donor_crop_alpha

        # Paste onto image using alpha mask
        paste_roi = out_img[y1_clip:y2_clip, x1_clip:x2_clip]
        paste_roi[donor_crop_alpha] = donor_crop_rgb[donor_crop_alpha]
        out_img[y1_clip:y2_clip, x1_clip:x2_clip] = paste_roi

        # Occlude previously pasted donors (donor-on-donor occlusion tracking)
        for prev_d in pasted_records:
            prev_d["visible_mask"] = prev_d["visible_mask"] & (~canvas_donor_mask)

        # Register current donor
        initial_area = int(np.count_nonzero(canvas_donor_mask))
        pasted_records.append({
            "donor_id": donor.donor_id,
            "class_id": donor.class_id,
            "class_name": donor.class_name,
            "source_image_id": donor.source_image_id,
            "scale": float(scale),
            "paste_pos": [int(paste_x), int(paste_y)],
            "canvas_mask": canvas_donor_mask,
            "visible_mask": canvas_donor_mask.copy(),
            "initial_mask_area": initial_area,
            "remaining_mask_area": initial_area,
            "occlusion_ratio": 0.0,
            "status": "KEPT",
            "drop_reason": "",
            "final_box_pixel": None,
        })

        # Update cumulative mask union (prevents double-counting overlapping donor areas)
        cumulative_donor_mask = cumulative_donor_mask | canvas_donor_mask

    # 3. Evaluate and finalize all pasted donors
    for d in pasted_records:
        vis_mask = d["visible_mask"]
        vis_area = int(np.count_nonzero(vis_mask))
        init_area = d["initial_mask_area"]
        vis_ratio = vis_area / max(1, init_area)
        occl_ratio = 1.0 - vis_ratio
        d["remaining_mask_area"] = vis_area
        d["occlusion_ratio"] = occl_ratio

        if vis_ratio < min_visibility:
            d["status"] = "DROPPED"
            d["drop_reason"] = (
                f"Pasted donor {d['donor_id']} occluded by subsequent donors "
                f"(remaining visibility {vis_ratio:.2f} < {min_visibility:.2f})"
            )
        else:
            # Compute tight bounding box strictly from visible binary mask
            rows, cols = np.where(vis_mask)
            if len(rows) == 0:
                d["status"] = "DROPPED"
                d["drop_reason"] = f"Pasted donor {d['donor_id']} has 0 visible pixels remaining"
            else:
                bx1 = float(cols.min())
                bx2 = float(cols.max() + 1)
                by1 = float(rows.min())
                by2 = float(rows.max() + 1)
                if (bx2 - bx1) < 4.0 or (by2 - by1) < 4.0:
                    d["status"] = "DROPPED"
                    d["drop_reason"] = f"Pasted donor {d['donor_id']} visible fragment too small (< 4px)"
                else:
                    d["status"] = "KEPT"
                    d["final_box_pixel"] = [bx1, by1, bx2, by2]

    # 4. Evaluate and finalize background objects
    for bg in bg_objects:
        bx1, by1, bx2, by2 = bg["box"]

        if bg["mask"] is not None:
            # Case A: Background has verified instance mask
            bg_mask = bg["mask"]
            orig_area = int(np.count_nonzero(bg_mask))
            vis_mask = bg_mask & (~cumulative_donor_mask)
            vis_area = int(np.count_nonzero(vis_mask))
            vis_ratio = vis_area / max(1, orig_area)
            occl_ratio = 1.0 - vis_ratio
            bg["occlusion_ratio"] = occl_ratio

            if vis_ratio < min_visibility:
                bg["status"] = "DROPPED"
                bg["drop_reason"] = (
                    f"Background mask class={bg['class_id']} occluded by donor masks "
                    f"(remaining visibility {vis_ratio:.2f} < {min_visibility:.2f})"
                )
            else:
                bg["status"] = "KEPT"
                rows, cols = np.where(vis_mask)
                if len(rows) > 0:
                    nbx1, nbx2 = float(cols.min()), float(cols.max() + 1)
                    nby1, nby2 = float(rows.min()), float(rows.max() + 1)
                    if (nbx2 - nbx1) >= 4.0 and (nby2 - nby1) >= 4.0:
                        bg["final_box_pixel"] = [nbx1, nby1, nbx2, nby2]
        else:
            # Case B: Background only has bounding box -> Conservative Heuristic
            ix1 = max(0, int(bx1))
            iy1 = max(0, int(by1))
            ix2 = min(w_img, int(bx2))
            iy2 = min(h_img, int(by2))

            if ix2 > ix1 and iy2 > iy1:
                # Count actual donor mask pixels covering the background bounding box
                sub_donor_mask = cumulative_donor_mask[iy1:iy2, ix1:ix2]
                covered_pixels = int(np.count_nonzero(sub_donor_mask))
            else:
                covered_pixels = 0

            box_area = bg["box_area"]
            occl_ratio = covered_pixels / box_area
            bg["occlusion_ratio"] = occl_ratio

            # Conservative policy: drop only if covered by actual mask >= (1 - min_visibility)
            if occl_ratio >= (1.0 - min_visibility):
                bg["status"] = "DROPPED"
                bg["drop_reason"] = (
                    f"Background bbox class={bg['class_id']} occluded by donor masks "
                    f"(covered {occl_ratio:.2f} >= threshold {1.0 - min_visibility:.2f})"
                )
            else:
                bg["status"] = "KEPT"
                # Box overlap alone does NOT shrink bbox without mask guidance
                bg["final_box_pixel"] = [bx1, by1, bx2, by2]

    # 5. Assemble surviving boxes and category IDs in YOLO normalized format [cx, cy, w, h]
    surviving_boxes: list[list[float]] = []
    surviving_cats: list[int] = []

    for bg in bg_objects:
        if bg["status"] == "KEPT":
            px1, py1, px2, py2 = bg["final_box_pixel"]
            cx = ((px1 + px2) / 2.0) / w_img
            cy = ((py1 + py2) / 2.0) / h_img
            bw = (px2 - px1) / w_img
            bh = (py2 - py1) / h_img
            surviving_boxes.append([
                max(0.0, min(1.0, float(cx))),
                max(0.0, min(1.0, float(cy))),
                max(0.001, min(1.0, float(bw))),
                max(0.001, min(1.0, float(bh))),
            ])
            surviving_cats.append(bg["class_id"])
        else:
            telemetry["drop_reasons"].append(bg["drop_reason"])

    for d in pasted_records:
        if d["status"] == "KEPT":
            px1, py1, px2, py2 = d["final_box_pixel"]
            cx = ((px1 + px2) / 2.0) / w_img
            cy = ((py1 + py2) / 2.0) / h_img
            bw = (px2 - px1) / w_img
            bh = (py2 - py1) / h_img
            surviving_boxes.append([
                max(0.0, min(1.0, float(cx))),
                max(0.0, min(1.0, float(cy))),
                max(0.001, min(1.0, float(bw))),
                max(0.001, min(1.0, float(bh))),
            ])
            surviving_cats.append(d["class_id"])
        else:
            telemetry["drop_reasons"].append(d["drop_reason"])

    # Update telemetry
    pasted_kept = sum(1 for d in pasted_records if d["status"] == "KEPT")
    pasted_dropped = sum(1 for d in pasted_records if d["status"] == "DROPPED")
    bg_kept = sum(1 for b in bg_objects if b["status"] == "KEPT")
    bg_dropped = sum(1 for b in bg_objects if b["status"] == "DROPPED")

    telemetry.update({
        "applied": True,
        "pasted_objects": pasted_kept,
        "pasted_objects_attempted": len(pasted_records),
        "pasted_objects_kept": pasted_kept,
        "pasted_objects_dropped": pasted_dropped,
        "boxes_after": len(surviving_boxes),
        "background_boxes_kept": bg_kept,
        "background_boxes_dropped": bg_dropped,
        "dropped_occluded_count": bg_dropped + pasted_dropped,
        "donor_telemetry": [
            {
                "donor_id": d["donor_id"],
                "class_id": d["class_id"],
                "class_name": d["class_name"],
                "source_image_id": d["source_image_id"],
                "scale": d["scale"],
                "paste_pos": d["paste_pos"],
                "initial_mask_area": d["initial_mask_area"],
                "remaining_mask_area": d["remaining_mask_area"],
                "occlusion_ratio": float(d["occlusion_ratio"]),
                "status": d["status"],
                "drop_reason": d["drop_reason"],
                "tight_bbox": d.get("final_box_pixel"),
            }
            for d in pasted_records
        ],
        "background_telemetry": [
            {
                "index": b["index"],
                "class_id": b["class_id"],
                "box": b["box"],
                "has_mask": b["mask"] is not None,
                "occlusion_ratio": float(b["occlusion_ratio"]),
                "status": b["status"],
                "drop_reason": b["drop_reason"],
            }
            for b in bg_objects
        ],
    })

    return out_img, surviving_boxes, surviving_cats, telemetry
