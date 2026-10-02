"""Copy-Paste Data Augmentation Module for Waste Detection (Phase 2).

Implements Simple Copy-Paste (Ghiasi et al., 2021) [5] using verified instance
segmentation masks from training data.

Core Academic Principles:
1. Strict Split Isolation: Donor objects are extracted EXCLUSIVELY from the training split.
   Validation and test splits are NEVER touched or used as donors or backgrounds.
2. Verified Masks Only: Uses hand-annotated polygon segmentation masks from TACO raw annotations.
   Rectangular bounding-box crops are NOT used as fake masks.
3. Occlusion Handling: When a donor object occludes an existing background object:
   - If occlusion ratio >= 0.8 (remaining visibility < 0.2), the background box is removed.
   - If occlusion ratio < 0.8, the background box is retained.
4. Synchronized Output: Returns transformed RGB image, updated YOLO-format bounding boxes,
   and corresponding integer category IDs.
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
    image_rgba: np.ndarray  # H x W x 4 (RGB + Alpha mask)
    binary_mask: np.ndarray  # H x W bool
    class_id: int
    class_name: str
    source_image_id: str
    original_bbox: list[float]  # [x, y, w, h] in pixels


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

        # 2. Get verified train TACO image IDs
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
        for img_id in train_taco_ids:
            img_meta = img_id_to_meta.get(img_id)
            if not img_meta:
                continue
            # TACO images are stored in data/detection/images/real/taco_{id:04d}.jpg
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
            for ann in anns:
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

                # Draw mask on crop
                mask_crop = Image.new("L", (crop_w, crop_h), 0)
                draw = ImageDraw.Draw(mask_crop)
                for poly in segs:
                    if len(poly) >= 6:
                        # Translate and scale poly to crop coords
                        poly_rel = [((poly[i] * sx) - x1, (poly[i + 1] * sy) - y1) for i in range(0, len(poly), 2)]
                        draw.polygon(poly_rel, fill=255)

                mask_arr = np.array(mask_crop) > 128
                if not np.any(mask_arr):
                    continue

                crop_rgb = np.array(base_img.crop((x1, y1, x2, y2)))
                alpha_ch = (mask_arr.astype(np.uint8) * 255)[:, :, np.newaxis]
                crop_rgba = np.concatenate([crop_rgb, alpha_ch], axis=2)

                donors.append(
                    DonorObject(
                        image_rgba=crop_rgba,
                        binary_mask=mask_arr,
                        class_id=class_id,
                        class_name=class_name,
                        source_image_id=str(img_id),
                        original_bbox=[float(bx), float(by), float(bw), float(bh)],
                    )
                )

        return cls(donors)


def apply_copy_paste(
    image: np.ndarray,
    boxes: list[list[float]],
    category_ids: list[int],
    donor_bank: DonorBank,
    p: float = 0.5,
    max_paste: int = 2,
    force_apply: bool = False,
    min_visibility: float = 0.2,
) -> tuple[np.ndarray, list[list[float]], list[int], dict[str, Any]]:
    """Apply Simple Copy-Paste augmentation with occlusion handling.

    Args:
        image: np.ndarray (H, W, 3) in uint8 RGB.
        boxes: list of YOLO normalized boxes [cx, cy, w, h] in range [0, 1].
        category_ids: list of integer class IDs.
        donor_bank: DonorBank instance containing verified training masks.
        p: Execution probability (ignored if force_apply is True).
        max_paste: Maximum number of objects to paste (1..max_paste).
        force_apply: If True, forces execution (for verification tests).
        min_visibility: Minimum visibility threshold to retain occluded boxes.

    Returns:
        transformed_image, transformed_boxes, transformed_category_ids, telemetry.
    """
    telemetry: dict[str, Any] = {
        "operation": "copy_paste",
        "applied": False,
        "forced": force_apply,
        "p": p,
        "donors_available": len(donor_bank),
        "pasted_objects": 0,
        "boxes_before": len(boxes),
        "boxes_after": len(boxes),
        "dropped_occluded_count": 0,
        "drop_reasons": [],
    }

    if not force_apply and random.random() > p:
        return image.copy(), list(boxes), list(category_ids), telemetry

    if len(donor_bank) == 0:
        telemetry["drop_reasons"].append("DonorBank is empty: no valid training masks available")
        return image.copy(), list(boxes), list(category_ids), telemetry

    h_img, w_img = image.shape[:2]
    num_to_paste = random.randint(1, max(1, max_paste))
    out_img = image.copy()

    # Convert YOLO boxes to pixel coordinates [x1, y1, x2, y2]
    bg_pixel_boxes: list[dict[str, Any]] = []
    for b, c in zip(boxes, category_ids):
        cx, cy, bw, bh = b
        px1 = (cx - bw / 2.0) * w_img
        py1 = (cy - bh / 2.0) * h_img
        px2 = (cx + bw / 2.0) * w_img
        py2 = (cy + bh / 2.0) * h_img
        bg_pixel_boxes.append({
            "box": [px1, py1, px2, py2],
            "class_id": int(c),
            "area": max(1.0, (px2 - px1) * (py2 - py1)),
            "occluded_area": 0.0,
        })

    new_boxes: list[dict[str, Any]] = []
    pasted_count = 0

    for _ in range(num_to_paste):
        donor = donor_bank.get_random_donor()
        if not donor:
            continue

        donor_rgba = donor.image_rgba
        dh, dw = donor_rgba.shape[:2]

        # Random scaling between 0.6 and 1.2, keeping donor within image bounds
        scale = random.uniform(0.6, 1.2)
        target_w = max(10, int(dw * scale))
        target_h = max(10, int(dh * scale))

        if target_w >= w_img or target_h >= h_img:
            # Downscale to fit inside background with margin
            fit_scale = min((w_img * 0.5) / dw, (h_img * 0.5) / dh)
            target_w = max(10, int(dw * fit_scale))
            target_h = max(10, int(dh * fit_scale))

        donor_pil = Image.fromarray(donor_rgba).resize((target_w, target_h), Image.Resampling.BILINEAR)
        scaled_rgba = np.array(donor_pil)
        scaled_rgb = scaled_rgba[:, :, :3]
        scaled_alpha = scaled_rgba[:, :, 3] > 128

        # Random paste placement
        max_x = w_img - target_w
        max_y = h_img - target_h
        paste_x = random.randint(0, max(0, max_x))
        paste_y = random.randint(0, max(0, max_y))

        # Paste onto background using alpha mask
        paste_roi = out_img[paste_y : paste_y + target_h, paste_x : paste_x + target_w]
        paste_roi[scaled_alpha] = scaled_rgb[scaled_alpha]
        out_img[paste_y : paste_y + target_h, paste_x : paste_x + target_w] = paste_roi

        # Calculate bounding box for pasted donor
        donor_box_x1 = float(paste_x)
        donor_box_y1 = float(paste_y)
        donor_box_x2 = float(paste_x + target_w)
        donor_box_y2 = float(paste_y + target_h)

        new_boxes.append({
            "box": [donor_box_x1, donor_box_y1, donor_box_x2, donor_box_y2],
            "class_id": donor.class_id,
            "source": f"donor_{donor.class_name}_{donor.source_image_id}",
        })
        pasted_count += 1

        # Check occlusion of existing background boxes
        for bg in bg_pixel_boxes:
            bx1, by1, bx2, by2 = bg["box"]
            # Intersection of donor box and background box
            ix1 = max(bx1, donor_box_x1)
            iy1 = max(by1, donor_box_y1)
            ix2 = min(bx2, donor_box_x2)
            iy2 = min(by2, donor_box_y2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                bg["occluded_area"] += inter_area

    # Filter background boxes based on occlusion ratio
    surviving_boxes: list[list[float]] = []
    surviving_cats: list[int] = []
    dropped_count = 0

    for bg in bg_pixel_boxes:
        occlusion_ratio = bg["occluded_area"] / bg["area"]
        if occlusion_ratio > (1.0 - min_visibility):
            # Dropped because occluded
            dropped_count += 1
            telemetry["drop_reasons"].append(
                f"Background box class={bg['class_id']} occluded={occlusion_ratio:.2f} > threshold={1.0-min_visibility:.2f}"
            )
        else:
            # Retain box
            bx1, by1, bx2, by2 = bg["box"]
            cx = ((bx1 + bx2) / 2.0) / w_img
            cy = ((by1 + by2) / 2.0) / h_img
            bw = (bx2 - bx1) / w_img
            bh = (by2 - by1) / h_img
            surviving_boxes.append([
                max(0.0, min(1.0, cx)),
                max(0.0, min(1.0, cy)),
                max(0.001, min(1.0, bw)),
                max(0.001, min(1.0, bh)),
            ])
            surviving_cats.append(bg["class_id"])

    # Add pasted new boxes
    for nb in new_boxes:
        bx1, by1, bx2, by2 = nb["box"]
        cx = ((bx1 + bx2) / 2.0) / w_img
        cy = ((by1 + by2) / 2.0) / h_img
        bw = (bx2 - bx1) / w_img
        bh = (by2 - by1) / h_img
        surviving_boxes.append([
            max(0.0, min(1.0, cx)),
            max(0.0, min(1.0, cy)),
            max(0.001, min(1.0, bw)),
            max(0.001, min(1.0, bh)),
        ])
        surviving_cats.append(nb["class_id"])

    telemetry.update({
        "applied": True,
        "pasted_objects": pasted_count,
        "boxes_after": len(surviving_boxes),
        "dropped_occluded_count": dropped_count,
    })

    return out_img, surviving_boxes, surviving_cats, telemetry
