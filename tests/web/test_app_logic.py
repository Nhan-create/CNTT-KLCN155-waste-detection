from src.inference.predict import Prediction, ScoredClass
from src.web.app_logic import prediction_view


def test_prediction_view_contains_text_probability_and_no_icon() -> None:
    topk = (
        ScoredClass(4, "glass", 0.62),
        ScoredClass(7, "plastic", 0.23),
        ScoredClass(5, "metal", 0.10),
    )
    view = prediction_view(Prediction(topk[0], topk, False))

    assert view.display_name == "Thủy tinh"
    assert view.probability == 0.62
    assert [row.display_name for row in view.topk] == [
        "Thủy tinh",
        "Nhựa",
        "Kim loại",
    ]
    assert not hasattr(view, "icon")
    assert all(not hasattr(row, "icon") for row in view.topk)
