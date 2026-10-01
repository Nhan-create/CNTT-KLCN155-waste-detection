"""Phase-two detection labels, deliberately separate from classification labels."""

from __future__ import annotations

DETECTION_CLASS_NAMES = ("plastic", "paper", "metal", "glass", "organic", "hazardous")
CLASS_NAMES = DETECTION_CLASS_NAMES
DETECTION_CLASS_NAMES_VI = ("Nhựa", "Giấy / bìa carton", "Kim loại", "Thủy tinh", "Hữu cơ", "Nguy hại")
SPLITS = ("train", "val", "test")
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
HARD_CONDITIONS = ("low_light", "backlight", "complex_background", "small_objects", "occlusion", "overlap")
SOURCE_LABEL_MAPPING = {
    "garbage_v2": {
        "plastic": "plastic", "paper": "paper", "cardboard": "paper",
        "metal": "metal", "glass": "glass", "biological": "organic",
        "battery": "hazardous",
    },
    "vn_trash": {"organic": "organic", "medical": "hazardous"},
}


class DetectionDatasetError(RuntimeError):
    """Detection data is incomplete, inconsistent, or unsafe to use."""


def map_source_label(source_dataset: str, original_label: str, *, manual_class: str = "", reviewer: str = "") -> str:
    """Map material semantics only; this never creates bounding boxes."""
    source = source_dataset.strip().lower().replace("-", "_")
    label = original_label.strip().lower().replace("-", "_")
    source = {"vn_trash_classification": "vn_trash", "garbage_classification_v2": "garbage_v2"}.get(source, source)
    automatic = SOURCE_LABEL_MAPPING.get(source, {}).get(label)
    if automatic is not None:
        if manual_class and manual_class != automatic:
            raise DetectionDatasetError(f"Mapping for {source}/{label} must be {automatic}")
        return automatic
    ambiguous = (source == "vn_trash" and label == "inorganic") or (source == "garbage_v2" and label in {"shoes", "clothes", "trash"})
    if ambiguous and manual_class in CLASS_NAMES and reviewer.strip():
        return manual_class
    raise DetectionDatasetError(f"Exclude or manually review {source}/{label}; a material class and reviewer are required")
