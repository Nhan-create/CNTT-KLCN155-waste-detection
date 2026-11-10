from pathlib import Path

import pytest

from src.detection.evaluation import ImagePrediction, calibrate_thresholds, summarize_predictions
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.types import BoundingBox, Detection, DetectionResult


def detection(name: str, score: float, box: tuple = (0, 0, 10, 10)) -> Detection:
    return Detection(DETECTION_CLASS_NAMES.index(name), name, score, BoundingBox(*box))


def test_evaluation_counts_duplicate_and_wrong_class_boxes_as_false_positives() -> None:
    sample = ImagePrediction(
        image=Path("bottles.jpg"), source="field",
        ground_truth=(detection("plastic", 1.0),),
        result=DetectionResult((
            detection("plastic", .9), detection("plastic", .8), detection("glass", .7),
        ), 20, 20),
    )

    report = summarize_predictions([sample], default_threshold=.29)

    assert report["true_positives"] == 1
    assert report["false_positives"] == 2
    assert report["precision"] == pytest.approx(1 / 3)
    assert report["recall"] == 1.0
    assert report["per_class"]["glass"]["false_positives"] == 1
    assert report["per_source"]["field"]["objects"] == 1


def test_calibration_uses_validation_and_reports_unattainable_precision() -> None:
    sample = ImagePrediction(
        image=Path("bottles.jpg"), source="field",
        ground_truth=(detection("plastic", 1.0), detection("battery", 1.0, (10, 10, 20, 20))),
        result=DetectionResult((
            detection("plastic", .9), detection("plastic", .3, (10, 0, 20, 10)),
            detection("battery", .8, (0, 10, 10, 20)),
        ), 20, 20),
    )

    report = calibrate_thresholds([sample], split="val", min_precision=.8)

    assert report["per_class_thresholds"]["plastic"] == pytest.approx(.9)
    assert report["classes"]["battery"]["status"] == "precision_unattainable"
    assert report["classes"]["glass"]["status"] == "no_validation_targets"
    assert "battery" in report["disabled_classes"]
    with pytest.raises(ValueError, match="validation"):
        calibrate_thresholds([sample], split="test", min_precision=.8)


def test_disabled_class_is_excluded_even_at_a_saturated_score() -> None:
    sample = ImagePrediction(
        Path("no_battery.jpg"), "field", (),
        DetectionResult((detection("battery", 1.0),), 20, 20),
    )

    report = summarize_predictions([sample], disabled_classes=("battery",))

    assert report["predictions"] == 0
    assert report["false_positives"] == 0
