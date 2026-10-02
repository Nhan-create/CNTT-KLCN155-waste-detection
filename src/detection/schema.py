"""Phase-two detection labels, deliberately separate from classification labels."""

from __future__ import annotations

DETECTION_CLASS_NAMES = (
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
)
CLASS_NAMES = DETECTION_CLASS_NAMES
DETECTION_CLASS_NAMES_VI = (
    "Pin / Ắc quy",
    "Rác sinh học / Hữu cơ",
    "Bìa carton",
    "Quần áo cũ",
    "Thủy tinh",
    "Kim loại",
    "Giấy",
    "Nhựa",
    "Giày dép",
    "Rác khác",
)
SPLITS = ("train", "val", "test")
SPLIT_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)
HARD_CONDITIONS = ("low_light", "backlight", "complex_background", "small_objects", "occlusion", "overlap")
SOURCE_LABEL_MAPPING = {
    "garbage_v2": {
        "battery": "battery",
        "biological": "biological",
        "cardboard": "cardboard",
        "clothes": "clothes",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "shoes": "shoes",
        "trash": "trash",
    },
    "vn_trash": {
        "organic": "biological",
        "medical": "battery",
    },
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
    ambiguous = (source == "vn_trash" and label == "inorganic")
    if ambiguous and manual_class in CLASS_NAMES and reviewer.strip():
        return manual_class
    raise DetectionDatasetError(f"Exclude or manually review {source}/{label}; a material class and reviewer are required")

