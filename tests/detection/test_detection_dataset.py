import json
from pathlib import Path

from PIL import Image

from src.detection.schema import CLASS_NAMES
from src.detection.dataset import validate_detection_dataset


def test_validates_complete_yolo_dataset(tmp_path: Path) -> None:
    manifest_records = []
    lines = [
        "path: dataset",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "manifest: manifest.jsonl",
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
            color_val = {"train": 0, "val": 50, "test": 100}[split] + index * 5
            Image.new("RGB", (32, 32), color=(color_val, color_val, color_val)).save(image_path)
            label_path.write_text(
                f"{index} 0.5 0.5 0.8 0.8\n", encoding="utf-8"
            )
            manifest_records.append({
                "image_id": f"{split}_{class_name}",
                "image_path": f"images/{split}/{class_name}.jpg",
                "label_path": f"labels/{split}/{class_name}.txt",
                "source_dataset": "local_real",
                "scene_id": f"scene_{split}_{class_name}",
                "is_real": True,
                "reviewed": True,
                "reviewer": "reviewer1",
                "split": split,
            })

    (tmp_path / "dataset" / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest_records), encoding="utf-8"
    )

    report = validate_detection_dataset(yaml_path)

    assert report.images_by_split == {"train": len(CLASS_NAMES), "val": len(CLASS_NAMES), "test": len(CLASS_NAMES)}
    assert report.boxes_by_split == {"train": len(CLASS_NAMES), "val": len(CLASS_NAMES), "test": len(CLASS_NAMES)}
    assert set(report.boxes_by_class) == set(CLASS_NAMES)
