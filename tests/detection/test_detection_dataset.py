from pathlib import Path

from PIL import Image

from src.data.schema import CLASS_NAMES
from src.detection.dataset import validate_detection_dataset


def test_validates_complete_yolo_dataset(tmp_path: Path) -> None:
    lines = [
        "path: dataset",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for split in ("train", "val", "test"):
        for index, class_name in enumerate(CLASS_NAMES):
            image_path = tmp_path / "dataset" / "images" / split / f"{class_name}.jpg"
            label_path = tmp_path / "dataset" / "labels" / split / f"{class_name}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), "white").save(image_path)
            label_path.write_text(
                f"{index} 0.5 0.5 0.8 0.8\n", encoding="utf-8"
            )

    report = validate_detection_dataset(yaml_path)

    assert report.images_by_split == {"train": 10, "val": 10, "test": 10}
    assert report.boxes_by_split == {"train": 10, "val": 10, "test": 10}
    assert set(report.boxes_by_class) == set(CLASS_NAMES)
