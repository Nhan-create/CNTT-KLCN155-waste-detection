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
            bounds = padded_crop_box(proposal.box, width, height, self.pad_ratio)
            if bounds is None:
                continue
            crops.append(image.crop(bounds))
            kept.append(proposal)
        probabilities: list[list[float]] = []
        for start in range(0, len(crops), self.batch_size):
            probabilities.extend(self.classifier.predict_proba(crops[start:start + self.batch_size]))
        if len(probabilities) != len(kept):
            raise ValueError("classifier returned a different number of rows than crops")

        refined: list[RefinedDetection] = []
        bio_idx = DETECTION_CLASS_NAMES.index("biological")
        clothes_idx = DETECTION_CLASS_NAMES.index("clothes")
        shoes_idx = DETECTION_CLASS_NAMES.index("shoes")
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
            ranked = sorted(range(len(canonical)), key=lambda i: -canonical[i])
            best = ranked[0]

            # Gated Consensus against environmental background confusion:
            # 1. Biological grass/soil background:
            if (
                proposal.class_id in inorganic_names
                and best == bio_idx
                and canonical[bio_idx] < 0.85
            ):
                prop_idx = DETECTION_CLASS_NAMES.index(proposal.class_id)
                if canonical[prop_idx] >= 0.05:
                    best = prop_idx
                else:
                    for alt in ranked:
                        if alt != bio_idx:
                            best = alt
                            break

            # 2. Texture confusion (crinkled dark plastic bags / metal crushed cans mistaken for clothes/shoes):
            if (
                proposal.class_id in solid_waste_names
                and best in (clothes_idx, shoes_idx)
                and canonical[best] < 0.80
            ):
                prop_idx = DETECTION_CLASS_NAMES.index(proposal.class_id)
                if canonical[prop_idx] >= 0.05:
                    best = prop_idx
                elif proposal.class_id == "plastic" and canonical[DETECTION_CLASS_NAMES.index("plastic")] >= 0.03:
                    best = DETECTION_CLASS_NAMES.index("plastic")
                else:
                    for alt in ranked:
                        if alt not in (clothes_idx, shoes_idx, bio_idx):
                            best = alt
                            break

            cls_support = canonical[best]
            if self.scoring_mode == "blend":
                concurrence = (best == proposal.class_index)
                allow_blend = (
                    (proposal.confidence >= max(self.min_proposal_gate, 0.28) and concurrence)
                    or proposal.confidence >= 0.38
                )
                if allow_blend:
                    effective_cls_prob = (
                        max(cls_support, proposal.confidence * 0.8)
                        if concurrence
                        else cls_support
                    )
                    blended = (
                        (1.0 - self.blend_weight) * proposal.confidence
                        + self.blend_weight * effective_cls_prob
                    )
                    max_allowed = (
                        min(1.0, proposal.confidence * 1.4)
                        if proposal.confidence < 0.35
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
        survivor_ids = {id(d) for d in survivors}
        by_id = {id(item.detection): item for item in refined}
        return [by_id[id(d)] for d in survivors if id(d) in survivor_ids]

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        prepared = ensure_exif_orientation(image)
        started = time.perf_counter()
        proposals = self.detector.detect_pil(prepared)
        if (proposals.image_width, proposals.image_height) != prepared.size:
            raise ValueError("detector returned boxes for a different image size")
        details = self.refine(prepared, proposals.detections)
        self.last_details = tuple(details)
        return DetectionResult(
            tuple(item.detection for item in details), *prepared.size,
            (time.perf_counter() - started) * 1000,
        )
