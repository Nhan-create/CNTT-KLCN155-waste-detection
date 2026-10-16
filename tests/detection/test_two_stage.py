"""Two-stage refinement: padded crops, classifier->canonical class mapping, rescoring, NMS."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from src.detection.coordinates import padded_crop_box
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.two_stage import TwoStageDetector, build_class_mapping
from src.detection.types import BoundingBox, Detection, DetectionResult

CLASSIFIER_CHECKPOINT = Path("artifacts/official_run/best_model.pt")


# ---------- padded crops ----------

def test_padded_crop_interior_box_expands_by_ratio() -> None:
    assert padded_crop_box((100, 200, 200, 400), 1000, 1000, 0.10) == (90, 180, 210, 420)


@pytest.mark.parametrize(
    ("box", "size", "expected"),
    [
        ((0, 0, 50, 40), (300, 200), (0, 0, 55, 44)),               # top-left corner
        ((260, 170, 300, 200), (300, 200), (256, 167, 300, 200)),    # bottom-right corner
        ((-30, -10, 120, 90), (100, 80), (0, 0, 100, 80)),           # partly outside
        ((10.4, 20.6, 30.2, 40.9), (64, 64), (8, 18, 33, 43)),       # fractional -> floor/ceil
    ],
)
def test_padded_crop_clips_to_image(box, size, expected) -> None:
    crop = padded_crop_box(box, *size, pad_ratio=0.10)
    assert crop == expected
    left, top, right, bottom = crop
    assert 0 <= left < right <= size[0] and 0 <= top < bottom <= size[1]


def test_padded_crop_rejects_degenerate_or_outside() -> None:
    assert padded_crop_box((50, 50, 50, 60), 100, 100) is None
    assert padded_crop_box((120, 10, 150, 20), 100, 100) is None
    with pytest.raises(ValueError):
        padded_crop_box((1, 1, 5, 5), 10, 10, pad_ratio=-0.1)


# ---------- class mapping ----------

def test_mapping_from_real_classifier_checkpoint_metadata() -> None:
    if not CLASSIFIER_CHECKPOINT.is_file():
        pytest.skip("classifier checkpoint not available")
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=False)
    names = checkpoint["class_names"]
    assert checkpoint["num_classes"] == len(names) == 10
    assert checkpoint["state_dict"]["classifier.3.weight"].shape[0] == len(names)
    mapping = build_class_mapping(names)
    for classifier_index, canonical_index in enumerate(mapping):
        assert DETECTION_CLASS_NAMES[canonical_index] == names[classifier_index]


def test_mapping_handles_permuted_order() -> None:
    permuted = list(reversed(DETECTION_CLASS_NAMES))
    mapping = build_class_mapping(permuted)
    assert mapping == tuple(range(9, -1, -1))


@pytest.mark.parametrize(
    "names",
    [
        list(DETECTION_CLASS_NAMES[:9]),                       # missing class
        [*DETECTION_CLASS_NAMES[:9], "styrofoam"],             # unknown class
        [*DETECTION_CLASS_NAMES[:9], "battery"],               # duplicate
    ],
)
def test_mapping_rejects_inconsistent_names(names) -> None:
    with pytest.raises(ValueError):
        build_class_mapping(names)


# ---------- end-to-end refinement with fakes ----------

PALETTE = {"metal": (200, 0, 0), "paper": (0, 200, 0), "glass": (0, 0, 200)}


class FixedDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        return DetectionResult(tuple(self.detections), *image.size)


class ColourClassifier:
    """Fake classifier with a *reversed* class order; votes by dominant crop colour."""

    def __init__(self) -> None:
        self.class_names = list(reversed(DETECTION_CLASS_NAMES))
        self.batch_sizes: list[int] = []

    def predict_proba(self, images):
        self.batch_sizes.append(len(images))
        rows = []
        for image in images:
            colours = image.convert("RGB").getcolors(maxcolors=1 << 16)
            dominant = max((c for c in colours if c[1] != (0, 0, 0)), default=(0, (0, 0, 0)))[1]
            label = next((name for name, rgb in PALETTE.items() if rgb == dominant), "trash")
            second = "plastic" if label != "plastic" else "trash"
            probs = [0.0] * 10
            probs[self.class_names.index(label)] = 0.8
            probs[self.class_names.index(second)] = 0.15
            probs[self.class_names.index("clothes")] = 0.05
            rows.append(probs)
        return rows


def _det(box, score=0.9, class_index=9) -> Detection:
    return Detection(class_index, DETECTION_CLASS_NAMES[class_index], score, BoundingBox(*box))


def test_two_stage_relabels_rescores_and_keeps_top3() -> None:
    image = Image.new("RGB", (400, 300), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 109, 109), fill=PALETTE["metal"])
    draw.rectangle((200, 50, 299, 199), fill=PALETTE["paper"])
    draw.rectangle((330, 230, 399, 299), fill=PALETTE["glass"])  # touches image corner
    proposals = [
        _det((10, 10, 110, 110), 0.9),
        _det((12, 12, 108, 108), 0.7),     # duplicate of the metal box -> NMS after relabel
        _det((200, 50, 300, 200), 0.6),
        _det((330, 230, 400, 300), 0.5),
        _det((205, 55, 295, 195), 0.4),    # duplicate of paper box
    ]
    classifier = ColourClassifier()
    two_stage = TwoStageDetector(FixedDetector(proposals), classifier, batch_size=2)
    result = two_stage.detect_pil(image)

    assert classifier.batch_sizes == [2, 2, 1]
    got = [(d.class_id, round(d.confidence, 4), (d.box.x1, d.box.y1, d.box.x2, d.box.y2))
           for d in result.detections]
    assert got == [
        ("metal", round(0.9 * 0.8, 4), (10, 10, 110, 110)),
        ("paper", round(0.6 * 0.8, 4), (200, 50, 300, 200)),
        ("glass", round(0.5 * 0.8, 4), (330, 230, 400, 300)),
    ]
    details = two_stage.last_details
    assert len(details) == 3
    assert details[0].top_k == (("metal", 0.8), ("plastic", 0.15), ("clothes", 0.05))
    assert details[0].detector_score == 0.9 and details[0].detector_class_id == "trash"


def test_two_stage_min_score_filters() -> None:
    image = Image.new("RGB", (100, 100), PALETTE["metal"])
    two_stage = TwoStageDetector(
        FixedDetector([_det((10, 10, 90, 90), 0.5)]), ColourClassifier(), min_score=0.5
    )
    assert two_stage.detect_pil(image).detections == ()


def test_real_classifier_adapter_runs_on_cpu_batch() -> None:
    if not CLASSIFIER_CHECKPOINT.is_file():
        pytest.skip("classifier checkpoint not available")
    from src.detection.two_stage import MobileNetBatchAdapter
    from src.web.classifier_service import MobileNetClassifier

    adapter = MobileNetBatchAdapter(MobileNetClassifier(CLASSIFIER_CHECKPOINT, device="cpu"))
    rows = adapter.predict_proba([Image.new("RGB", (50, 80), (120, 90, 30)),
                                  Image.new("RGB", (300, 40), (10, 10, 10))])
    assert len(rows) == 2 and all(len(r) == 10 for r in rows)
    assert all(abs(sum(r) - 1.0) < 1e-4 for r in rows)


def test_two_stage_blend_mode_computes_weighted_score() -> None:
    image = Image.new("RGB", (100, 100), PALETTE["metal"])
    # Proposal: score = 0.40; Classifier will output metal = 0.80.
    # Blend: 0.35 * 0.40 + 0.65 * 0.80 = 0.14 + 0.52 = 0.66
    two_stage = TwoStageDetector(
        FixedDetector([_det((10, 10, 90, 90), score=0.40)]),
        ColourClassifier(),
        scoring_mode="blend",
        blend_weight=0.65,
    )
    result = two_stage.detect_pil(image)
    assert len(result.detections) == 1
    assert result.detections[0].class_id == "metal"
    assert round(result.detections[0].confidence, 4) == round(0.35 * 0.40 + 0.65 * 0.80, 4)


def test_two_stage_low_proposal_suppressed_by_gate() -> None:
    image = Image.new("RGB", (100, 100), PALETTE["metal"])
    # Proposal has low confidence (0.10 < gate 0.18).
    # Instead of blending (0.35 * 0.10 + 0.65 * 0.80 = 0.555),
    # gate enforces multiplication: 0.10 * 0.80 = 0.08.
    # With min_score=0.20, it is filtered out completely!
    two_stage = TwoStageDetector(
        FixedDetector([_det((10, 10, 90, 90), score=0.10)]),
        ColourClassifier(),
        scoring_mode="blend",
        min_proposal_gate=0.18,
        min_score=0.20,
    )
    result = two_stage.detect_pil(image)
    assert len(result.detections) == 0


def test_gated_consensus_protects_inorganic_from_biological_grass() -> None:
    """When detector predicts glass (43.7%) and classifier votes biological (59.5%) due to grass,

    gated consensus prevents the inorganic bottle from being flipped to biological.
    """
    class ConfusedGrassClassifier:
        class_names = list(DETECTION_CLASS_NAMES)
        def predict_proba(self, images):
            rows = []
            for _ in images:
                probs = [0.01] * 10
                probs[DETECTION_CLASS_NAMES.index("biological")] = 0.595  # grass in background
                probs[DETECTION_CLASS_NAMES.index("glass")] = 0.151       # actual bottle
                probs[DETECTION_CLASS_NAMES.index("metal")] = 0.100
                rows.append(probs)
            return rows

    # Detector saw glass
    glass_idx = DETECTION_CLASS_NAMES.index("glass")
    proposal = Detection(glass_idx, "glass", 0.437, BoundingBox(10, 10, 90, 90))
    image = Image.new("RGB", (100, 100), (0, 150, 0))  # green grass image

    two_stage = TwoStageDetector(
        FixedDetector([proposal]),
        ConfusedGrassClassifier(),
        scoring_mode="blend",
        min_score=0.20,
    )
    result = two_stage.detect_pil(image)
    assert len(result.detections) == 1
    # MUST be preserved as glass, NOT flipped to biological
    assert result.detections[0].class_id == "glass"
    assert result.detections[0].confidence >= 0.25


