"""Export a trained YOLO detector for mobile or accelerated inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--format",
        choices=("litert", "onnx", "openvino", "coreml", "engine"),
        default="litert",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument(
        "--data", type=Path, default=Path("configs/detection_dataset.yaml")
    )
    arguments = parser.parse_args()
    if not arguments.model.is_file():
        parser.error(f"Detector not found: {arguments.model}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Export failed: install dependencies with pip install -r requirements.txt"
        )
        return 2
    model = YOLO(str(arguments.model), task="detect")
    options: dict[str, object] = {
        "format": arguments.format,
        "imgsz": arguments.imgsz,
        "device": arguments.device,
    }
    if arguments.int8:
        options.update(quantize=8, data=str(arguments.data))
    exported = model.export(**options)
    print(json.dumps({"exported_model": str(exported)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
