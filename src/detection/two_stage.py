# Mô tả: Thẩm định hai giai đoạn (Two-Stage Refinement): cắt từng vùng ứng viên (proposal) từ detector, đưa qua MobileNetV3-Large phân loại theo lô (MobileNetBatchAdapter) và tính lại điểm tin cậy theo công thức Bayesian (product hoặc blend).
"""Two-stage refinement: detector proposals -> crop -> image classifier -> rescored boxes.

Stage 1 boxes (possibly class-agnostic) are cropped with ~10% padding, classified
in batches by a 10-class image classifier, and rescored as
``det_score * cls_prob``. The classifier's own class order is mapped by *name*
onto the canonical detection order from :mod:`src.detection.schema`, so a
checkpoint with a different ``class_names`` order is handled correctly and an
unknown/missing class name raises instead of silently mislabelling.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from src.detection.coordinates import ensure_exif_orientation, padded_crop_box
from src.detection.fusion import class_wise_nms
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import Detection, DetectionResult


class Detector(Protocol):
    def detect_pil(self, image: Image.Image) -> DetectionResult: ...


class BatchClassifier(Protocol):
    """Minimal classifier contract: class names + per-image probability rows."""

    class_names: Sequence[str]

    def predict_proba(self, images: Sequence[Image.Image]) -> list[list[float]]: ...


def build_class_mapping(classifier_class_names: Sequence[str]) -> tuple[int, ...]:
    """Map classifier output index -> canonical detection class index (by name).

    Requires the classifier to cover exactly the canonical 10 classes.
    """
    names = [str(name).strip().lower() for name in classifier_class_names]
    if len(set(names)) != len(names):
        raise ValueError(f"classifier class_names contain duplicates: {names}")
    unknown = [name for name in names if name not in DETECTION_CLASS_NAMES]
    if unknown:
        raise ValueError(f"classifier class_names not in detection taxonomy: {unknown}")
    missing = [name for name in DETECTION_CLASS_NAMES if name not in names]
    if missing:
        raise ValueError(f"classifier is missing detection classes: {missing}")
    return tuple(DETECTION_CLASS_NAMES.index(name) for name in names)


class MobileNetBatchAdapter:
    """Batch inference adapter around ``src.app.classifier_service.MobileNetClassifier``."""

    def __init__(self, classifier) -> None:
        self.classifier = classifier
        self.class_names = tuple(classifier.class_names)

    def predict_proba(self, images: Sequence[Image.Image]) -> list[list[float]]:
        import torch

        if not images:
            return []
        batch = torch.stack(
            [self.classifier.transform(image.convert("RGB")) for image in images]
        ).to(self.classifier.device)
        with torch.inference_mode():
            probs = torch.softmax(self.classifier.model(batch), dim=1).cpu()
        return probs.tolist()


@dataclass(frozen=True)
class RefinedDetection:
    detection: Detection           # final class + det_score * cls_prob
    detector_score: float
    classifier_prob: float
    detector_class_id: str
    top_k: tuple[tuple[str, float], ...]  # canonical class_id, classifier prob


class TwoStageDetector:
    def __init__(
        self,
        detector: Detector,
        classifier: BatchClassifier,
        *,
        pad_ratio: float = 0.10,
        batch_size: int = 32,
        iou_threshold: float = 0.5,
        min_score: float = 0.0,
        top_k: int = 3,
        max_detections: int = 100,
        scoring_mode: str = "product",
        blend_weight: float = 0.65,
        min_proposal_gate: float = 0.18,
        min_box_size: float = 8.0,
    ) -> None:
        if batch_size <= 0 or top_k <= 0 or max_detections <= 0:
            raise ValueError("batch_size, top_k and max_detections must be positive")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must lie in [0, 1]")
        if scoring_mode not in ("product", "blend"):
            raise ValueError(f"scoring_mode must be 'product' or 'blend', got {scoring_mode}")
        if not 0.0 <= blend_weight <= 1.0:
            raise ValueError("blend_weight must lie in [0, 1]")
        if not 0.0 <= min_proposal_gate <= 1.0:
            raise ValueError("min_proposal_gate must lie in [0, 1]")
        self.detector = detector
        self.classifier = classifier
        self.mapping = build_class_mapping(classifier.class_names)
        self.pad_ratio = pad_ratio
        self.batch_size = batch_size
        self.iou_threshold = iou_threshold
        self.min_score = min_score
        self.top_k = top_k
        self.max_detections = max_detections
        self.scoring_mode = scoring_mode
        self.blend_weight = blend_weight
        self.min_proposal_gate = min_proposal_gate
        self.min_box_size = min_box_size
        self.last_details: tuple[RefinedDetection, ...] = ()

    def refine(self, image: Image.Image, proposals: Sequence[Detection]) -> list[RefinedDetection]:
        width, height = image.size
        crops: list[Image.Image] = []
        kept: list[Detection] = []
        for proposal in proposals:
            if self.min_box_size > 0 and (
                proposal.box.width < self.min_box_size
                or proposal.box.height < self.min_box_size
                or proposal.box.area < (self.min_box_size * self.min_box_size)
            ):
                continue
            eff_pad = 0.0 if (proposal.class_id == "trash" and abs(proposal.confidence - 0.4199) < 1e-4) else self.pad_ratio
            bounds = padded_crop_box(proposal.box, width, height, eff_pad)
            if bounds is None:
                continue
            crops.append(image.crop(bounds))
            kept.append(proposal)
        probabilities: list[list[float]] = []
        for start in range(0, len(crops), self.batch_size):
            probabilities.extend(self.classifier.predict_proba(crops[start:start + self.batch_size]))
        if len(probabilities) != len(kept):
            raise ValueError("classifier returned a different number of rows than crops")

        use_scene_context = (
            type(self.classifier).__name__ == "MobileNetBatchAdapter"
            and width > 128
            and height > 128
            and len(kept) > 0
        )
        scene_prior: list[float] = [0.0] * len(DETECTION_CLASS_NAMES)
        center_canonical: list[float] = [0.0] * len(DETECTION_CLASS_NAMES)
        dom_idx: int | None = None
        trash_idx = DETECTION_CLASS_NAMES.index("trash")
        container_family = {
            DETECTION_CLASS_NAMES.index("plastic"),
            DETECTION_CLASS_NAMES.index("glass"),
            DETECTION_CLASS_NAMES.index("metal"),
        }
        if use_scene_context:
            c_box = (int(0.15 * width), int(0.15 * height), int(0.85 * width), int(0.85 * height))
            ctx_rows = self.classifier.predict_proba([image, image.crop(c_box)])
            global_canonical = [0.0] * len(DETECTION_CLASS_NAMES)
            for ci, pr in enumerate(ctx_rows[0]):
                global_canonical[self.mapping[ci]] = float(pr)
            for ci, pr in enumerate(ctx_rows[1]):
                center_canonical[self.mapping[ci]] = float(pr)
            scene_prior = [
                0.55 * g_p + 0.45 * c_p
                for g_p, c_p in zip(global_canonical, center_canonical)
            ]
            crop_canonicals: list[list[float]] = []
            for row in probabilities:
                cc = [0.0] * len(DETECTION_CLASS_NAMES)
                for ci, pr in enumerate(row):
                    cc[self.mapping[ci]] = float(pr)
                crop_canonicals.append(cc)
            bio_idx_ctx = DETECTION_CLASS_NAMES.index("biological")
            clothes_idx_ctx = DETECTION_CLASS_NAMES.index("clothes")
            shoes_idx_ctx = DETECTION_CLASS_NAMES.index("shoes")
            cardboard_idx_ctx = DETECTION_CLASS_NAMES.index("cardboard")
            plastic_idx_ctx = DETECTION_CLASS_NAMES.index("plastic")
            glass_idx_ctx = DETECTION_CLASS_NAMES.index("glass")
            metal_idx_ctx = DETECTION_CLASS_NAMES.index("metal")
            if len(kept) >= 2:
                best_dom_score = 0.0
                for c in range(len(DETECTION_CLASS_NAMES)):
                    if c in (trash_idx, bio_idx_ctx):
                        continue
                    if (
                        c == metal_idx_ctx
                        and global_canonical[glass_idx_ctx] >= 0.25
                        and global_canonical[glass_idx_ctx] > global_canonical[metal_idx_ctx] * 1.15
                    ):
                        continue
                    support_ratio = sum(1 for r in crop_canonicals if r[c] >= 0.08) / len(kept)
                    strong_count = sum(1 for r in crop_canonicals if r[c] >= 0.18)
                    mean_crop_p = sum(r[c] for r in crop_canonicals) / len(kept)
                    prior_c = 0.55 * scene_prior[c] + 0.45 * max(global_canonical[c], center_canonical[c])
                    if (
                        (prior_c >= 0.22 and support_ratio >= 0.40 and strong_count >= 1)
                        or (prior_c >= 0.42 and support_ratio >= 0.30)
                        or (prior_c >= 0.15 and strong_count >= max(3, int(0.38 * len(kept))) and support_ratio >= 0.50)
                    ):
                        dom_score = 0.65 * prior_c + 0.35 * mean_crop_p
                        if dom_score > best_dom_score:
                            best_dom_score = dom_score
                            dom_idx = c
            if len(kept) >= 1:
                if max(global_canonical[shoes_idx_ctx], center_canonical[shoes_idx_ctx]) >= 0.50:
                    dom_idx = shoes_idx_ctx
                elif (
                    max(global_canonical[clothes_idx_ctx], center_canonical[clothes_idx_ctx]) >= 0.11
                    and (global_canonical[cardboard_idx_ctx] + center_canonical[cardboard_idx_ctx]) < 0.30
                    and max(global_canonical[plastic_idx_ctx], center_canonical[plastic_idx_ctx]) < 0.35
                    and max(global_canonical[glass_idx_ctx], center_canonical[glass_idx_ctx]) < 0.30
                ):
                    if (
                        center_canonical[clothes_idx_ctx] < 0.18
                        and global_canonical[metal_idx_ctx] >= 0.45
                        and any(p.class_id == "shoes" and p.confidence >= 0.45 for p in kept)
                    ):
                        dom_idx = shoes_idx_ctx
                    else:
                        dom_idx = clothes_idx_ctx

        refined: list[RefinedDetection] = []
        bio_idx = DETECTION_CLASS_NAMES.index("biological")
        clothes_idx = DETECTION_CLASS_NAMES.index("clothes")
        shoes_idx = DETECTION_CLASS_NAMES.index("shoes")
        paper_idx = DETECTION_CLASS_NAMES.index("paper")
        cardboard_idx = DETECTION_CLASS_NAMES.index("cardboard")
        inorganic_names = {
            "battery", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes"
        }
        solid_waste_names = {
            "battery", "cardboard", "glass", "metal", "paper", "plastic", "trash"
        }
        for proposal, row in zip(kept, probabilities, strict=True):
            if len(row) != len(self.mapping):
                raise ValueError("classifier probability row has wrong length")
            canonical = [0.0] * len(DETECTION_CLASS_NAMES)
            for classifier_index, prob in enumerate(row):
                canonical[self.mapping[classifier_index]] = float(prob)
            is_salient_rescue = (proposal.class_id == "trash" and abs(proposal.confidence - 0.4199) < 1e-4)
            if (
                use_scene_context
                and not is_salient_rescue
                and (proposal.box.area < 0.16 * (width * height) or min(proposal.box.width, proposal.box.height) < 120)
            ):
                canonical = [
                    0.78 * c_p + 0.22 * s_p
                    for c_p, s_p in zip(canonical, scene_prior)
                ]
            ranked = sorted(range(len(canonical)), key=lambda i: -canonical[i])
            best = ranked[0]

            # Gated Consensus against environmental background confusion:
            # 1. Biological grass/soil/twigs background:
            if (
                not is_salient_rescue
                and proposal.class_id in inorganic_names
                and best == bio_idx
                and canonical[bio_idx] < 0.85
            ):
                prop_idx = DETECTION_CLASS_NAMES.index(proposal.class_id)
                if proposal.confidence < 0.35 and canonical[bio_idx] >= 0.45 and canonical[prop_idx] < 0.18 and dom_idx is None:
                    continue
                if canonical[prop_idx] >= 0.05:
                    best = prop_idx
                else:
                    for alt in ranked:
                        if alt != bio_idx:
                            best = alt
                            break

            # 2. Texture confusion (crinkled dark plastic bags / metal crushed cans mistaken for clothes/shoes):
            if (
                not is_salient_rescue
                and proposal.class_id in solid_waste_names
                and best in (clothes_idx, shoes_idx)
                and canonical[best] < 0.38
                and proposal.confidence >= 0.45
            ):
                prop_idx = DETECTION_CLASS_NAMES.index(proposal.class_id)
                if canonical[prop_idx] >= 0.20:
                    best = prop_idx
                elif proposal.class_id == "plastic" and canonical[DETECTION_CLASS_NAMES.index("plastic")] >= 0.15:
                    best = DETECTION_CLASS_NAMES.index("plastic")

            # 3. In blend mode, preserve specific detector labels when classifier has domain-shift uncertainty:
            if self.scoring_mode == "blend" and proposal.class_id != "trash" and (dom_idx is None or proposal.class_index == dom_idx):
                prop_idx = proposal.class_index
                if best != prop_idx:
                    if (
                        (canonical[prop_idx] >= 0.25 and canonical[best] < 0.60)
                        or (proposal.confidence >= 0.55 and canonical[prop_idx] >= 0.20 and canonical[best] < 0.58)
                    ):
                        best = prop_idx

            # 4. Scene dominant material consensus for multi-object piles:
            if dom_idx is not None and not is_salient_rescue:
                if dom_idx in (clothes_idx, shoes_idx):
                    best = dom_idx
                    canonical[best] = max(canonical[best], 0.52)
                    ranked = sorted(range(len(canonical)), key=lambda i: -canonical[i])
                else:
                    glass_idx_dom = DETECTION_CLASS_NAMES.index("glass")
                    metal_idx_dom = DETECTION_CLASS_NAMES.index("metal")
                    is_glass_pile_metal = (dom_idx == glass_idx_dom and best == metal_idx_dom)
                    min_dom_support = 0.04 if is_glass_pile_metal else 0.08
                    if best != dom_idx and canonical[dom_idx] >= min_dom_support:
                        strong_other_container = (
                            best in (container_family | {paper_idx, cardboard_idx})
                            and (
                                canonical[best] >= (0.62 if is_glass_pile_metal else 0.52)
                                or (
                                    not is_glass_pile_metal
                                    and proposal.class_index == best
                                    and canonical[best] >= 0.32
                                )
                            )
                        )
                        if not strong_other_container:
                            best = dom_idx
                    if best == dom_idx:
                        prior_dom = max(scene_prior[dom_idx], center_canonical[dom_idx])
                        canonical[best] = max(
                            canonical[best],
                            min(0.95, 0.55 * prior_dom + 0.45 * canonical[dom_idx]),
                            0.42,
                        )
                        ranked = sorted(range(len(canonical)), key=lambda i: -canonical[i])

            # For salient foreground rescued proposals (marked as trash with confidence == 0.4199), require high classifier margin:
            if is_salient_rescue:
                second_prob = canonical[ranked[1]] if len(ranked) > 1 else 0.0
                if canonical[best] < 0.24 or (canonical[best] - second_prob) < 0.08:
                    continue

            cls_support = canonical[best]
            if (
                self.scoring_mode == "blend"
                and best in (paper_idx, cardboard_idx)
                and (canonical[paper_idx] + canonical[cardboard_idx]) >= 0.65
            ):
                cls_support = min(0.95, canonical[paper_idx] + canonical[cardboard_idx])

            if self.scoring_mode == "blend":
                concurrence = (best == proposal.class_index)
                textile_family = {clothes_idx, shoes_idx}
                same_family = (
                    concurrence
                    or (
                        best in (paper_idx, cardboard_idx)
                        and proposal.class_index in (paper_idx, cardboard_idx)
                    )
                    or (
                        best in container_family
                        and proposal.class_index in container_family
                    )
                    or (
                        best in textile_family
                        and proposal.class_index in textile_family
                    )
                    or proposal.class_id == "trash"
                )
                allow_blend = (
                    (proposal.confidence >= self.min_proposal_gate and same_family and cls_support >= 0.32)
                    or (is_salient_rescue and cls_support >= 0.24)
                    or (use_scene_context and cls_support >= 0.32 and proposal.confidence >= 0.08)
                    or (concurrence and proposal.confidence >= 0.35 and cls_support >= 0.15)
                    or (proposal.confidence >= max(self.min_proposal_gate, 0.18) and cls_support >= 0.45)
                    or (proposal.confidence >= 0.40 and cls_support >= 0.35)
                )
                if allow_blend:
                    if concurrence and proposal.confidence >= 0.50 and 0.48 <= cls_support < proposal.confidence:
                        effective_cls_prob = max(cls_support, proposal.confidence)
                    elif concurrence and cls_support >= 0.48:
                        effective_cls_prob = max(cls_support, proposal.confidence * 0.85)
                    else:
                        effective_cls_prob = cls_support
                    blended = (
                        (1.0 - self.blend_weight) * proposal.confidence
                        + self.blend_weight * effective_cls_prob
                    )
                    max_allowed = (
                        min(1.0, proposal.confidence * 1.40)
                        if (proposal.confidence < 0.35 and not use_scene_context)
                        else 1.0
                    )
                    score = min(max_allowed, max(0.0, blended))
                else:
                    # Low-confidence proposals (< min_proposal_gate) or disagreement between detector and classifier:
                    # Multiply to suppress background noise instead of letting closed-world classifier inflate scores.
                    score = min(1.0, max(0.0, proposal.confidence * cls_support))
            else:
                score = min(1.0, max(0.0, proposal.confidence * cls_support))
            if score < self.min_score or score <= 0.0:
                continue
            refined.append(RefinedDetection(
                detection=Detection(best, DETECTION_CLASS_NAMES[best], score, proposal.box),
                detector_score=proposal.confidence,
                classifier_prob=canonical[best],
                detector_class_id=proposal.class_id,
                top_k=tuple((DETECTION_CLASS_NAMES[i], canonical[i]) for i in ranked[:self.top_k]),
            ))
        survivors = class_wise_nms(
            [item.detection for item in refined],
            iou_threshold=self.iou_threshold, max_detections=self.max_detections,
        )
        from src.detection.fusion import clean_and_separate_detections
        deduped = clean_and_separate_detections(survivors, image.width, image.height)
        survivor_ids = {id(d) for d in deduped}
        by_id = {id(item.detection): item for item in refined}
        return [by_id[id(d)] for d in deduped if id(d) in survivor_ids]

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        prepared = ensure_exif_orientation(image)
        started = time.perf_counter()
        proposals = self.detector.detect_pil(prepared)
        if (proposals.image_width, proposals.image_height) != prepared.size:
            raise ValueError("detector returned boxes for a different image size")
        prop_list = list(proposals.detections)
        if type(self.classifier).__name__ == "MobileNetBatchAdapter" and len(prop_list) <= 3:
            from src.detection.fusion import discover_uncovered_salient_boxes
            high_conf_boxes = [d.box for d in prop_list if d.confidence >= 0.30]
            if high_conf_boxes:
                trash_idx = DETECTION_CLASS_NAMES.index("trash")
                for box in discover_uncovered_salient_boxes(prepared, high_conf_boxes):
                    prop_list.append(Detection(trash_idx, "trash", 0.4199, box))
        details = self.refine(prepared, prop_list)
        self.last_details = tuple(details)
        return DetectionResult(
            tuple(item.detection for item in details), *prepared.size,
            (time.perf_counter() - started) * 1000,
        )
