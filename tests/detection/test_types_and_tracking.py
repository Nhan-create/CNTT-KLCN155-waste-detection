from src.detection.tracking import TemporalDetectionSmoother, box_iou
from src.detection.types import BoundingBox, Detection, DetectionResult


def detection(class_index: int, class_id: str, box: BoundingBox) -> Detection:
    return Detection(class_index, class_id, 0.8, box)


def test_iou_and_temporal_smoothing_keep_new_classes_responsive() -> None:
    first_box = BoundingBox(10, 10, 50, 50)
    moved_box = BoundingBox(14, 10, 54, 50)
    assert 0.0 < box_iou(first_box, moved_box) < 1.0

    smoother = TemporalDetectionSmoother(
        iou_threshold=0.3, smoothing_alpha=0.5, max_missed=0
    )
    smoother.update(DetectionResult((detection(7, "plastic", first_box),), 100, 100))
    result = smoother.update(
        DetectionResult(
            (
                detection(7, "plastic", moved_box),
                detection(5, "metal", BoundingBox(60, 20, 90, 70)),
            ),
            100,
            100,
        )
    )

    assert {row.class_id for row in result.detections} == {"plastic", "metal"}
    plastic = next(row for row in result.detections if row.class_id == "plastic")
    assert plastic.box.x1 == 12


def test_stale_box_expires_after_configured_misses() -> None:
    smoother = TemporalDetectionSmoother(max_missed=1)
    original = DetectionResult(
        (detection(6, "paper", BoundingBox(1, 1, 20, 20)),), 30, 30
    )
    assert smoother.update(original).detections
    assert smoother.update(DetectionResult((), 30, 30)).detections
    assert not smoother.update(DetectionResult((), 30, 30)).detections


def test_overlapping_new_class_replaces_old_class_without_ghost_box() -> None:
    smoother = TemporalDetectionSmoother(max_missed=2)
    box = BoundingBox(5, 5, 25, 25)
    smoother.update(DetectionResult((detection(7, "plastic", box),), 30, 30))

    result = smoother.update(
        DetectionResult((detection(5, "metal", box),), 30, 30)
    )

    assert [row.class_id for row in result.detections] == ["metal"]
