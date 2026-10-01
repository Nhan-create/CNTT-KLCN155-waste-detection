from types import SimpleNamespace

from src.detection.train import _best_f1_confidence, _serialize_test_metrics


def test_serializes_global_and_per_class_test_metrics() -> None:
    metrics = SimpleNamespace(
        results_dict={"metrics/mAP50(B)": 0.75},
        names={0: "battery", 1: "biological"},
        box=SimpleNamespace(
            p=[0.8, 0.6],
            r=[0.7, 0.5],
            f1=[0.7467, 0.5455],
            ap50=[0.9, 0.6],
            maps=[0.7, 0.4],
        ),
    )

    result = _serialize_test_metrics(metrics)

    assert result["overall"] == {"metrics/mAP50(B)": 0.75}
    assert result["per_class"]["battery"]["precision"] == 0.8
    assert result["per_class"]["biological"]["map50_95"] == 0.4


def test_selects_confidence_with_highest_validation_mean_f1() -> None:
    metrics = SimpleNamespace(
        box=SimpleNamespace(
            curves_results=[
                (
                    [0.0, 0.5, 1.0],
                    [[0.0, 0.8, 0.0], [0.0, 0.6, 0.0]],
                    "Confidence",
                    "F1",
                )
            ]
        )
    )

    result = _best_f1_confidence(metrics)

    assert result == {"confidence": 0.5, "mean_f1": 0.7}
