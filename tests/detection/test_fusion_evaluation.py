from pathlib import Path

import pytest

from src.detection.evaluate import coco_metrics, operating_metrics
from src.detection.fusion import fuse_detections
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult


def detection(index=0, confidence=0.9, x=10):
    return Detection(index, DETECTION_CLASS_NAMES[index], confidence, BoundingBox(x, 10, x + 20, 30))


def result(*rows):
    return DetectionResult(tuple(rows), 100, 80)


def test_wbf_weighted_coordinates_and_class_isolation():
    fused = fuse_detections((result(detection(x=10), detection(index=1)), result(detection(confidence=.6, x=12))),
                            weights=(2, 1), iou_threshold=.5)
    assert len(fused.detections) == 2
    fused_first = next(d for d in fused.detections if d.class_index == 0)
    assert fused_first.box.x1 == pytest.approx((10 * 1.8 + 12 * .6) / 2.4, abs=1e-5)
    assert fused_first.class_id == DETECTION_CLASS_NAMES[0]


def test_wbf_empty_output_threshold_and_invalid_shape():
    assert not fuse_detections((result(), result())).detections
    assert not fuse_detections((result(detection()), result()), output_threshold=.8).detections
    with pytest.raises(ValueError, match="dimensions"):
        fuse_detections((result(), DetectionResult((), 20, 30)))
    with pytest.raises(ValueError, match="positive"):
        fuse_detections((result(), result()), weights=(1, 0))
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(float("nan"), 0, 20, 20)


def test_coco_perfect_absent_class_and_empty_predictions():
    truth = [[detection(confidence=1.0)]]
    perfect = coco_metrics([result(detection())], truth)
    assert perfect["map50_95"] == pytest.approx(1.0)
    assert perfect["ap_by_class"][DETECTION_CLASS_NAMES[-1]]["ap50_95"] is None
    assert coco_metrics([result()], truth)["map50"] == 0.0


def test_operating_duplicate_is_fp_and_wrong_class_does_not_match():
    metrics = operating_metrics([result(detection(), detection(confidence=.8), detection(index=1))], [[detection()]])
    assert metrics["micro"]["tp"] == 1
    assert metrics["micro"]["fp"] == 2
    assert metrics["micro"]["fn"] == 0


def test_tune_uses_validation_only_and_serializes_selection(tmp_path, monkeypatch):
    import src.detection.tune as module
    requested = []
    monkeypatch.setattr(module, "load_evaluation_split", lambda path, split: (requested.append(split) or [{"image_id": "a"}], "valhash"))
    monkeypatch.setattr(module, "create_detector", lambda *a, **k: object())
    monkeypatch.setattr(module, "collect_predictions", lambda *a, **k: ([result(detection())], [[detection()]], {}))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test checkpoint hash only")
    selection = module.tune(Path("unused.yaml"), "ssdlite", checkpoint, None,
                            Path("configs/detection_fusion.yaml"), tmp_path / "selected.json")
    assert requested == ["val"]
    assert selection["selected_on"] == "val"
    assert selection["validation_fingerprint"] == "valhash"
    assert selection["best_validation_metrics"]["map50_95"] == pytest.approx(1.0)
