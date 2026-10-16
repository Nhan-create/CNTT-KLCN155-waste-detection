"""CPU end-to-end smoke run of the v2 inference pipeline in 4 modes.

Runs the real app flow (create_detector -> build_detection_pipeline ->
DetectionService -> analyze_image_file -> draw_detections) and records box
counts, timings and coordinate sanity checks. Not a quality evaluation.

Usage:
    .venv\\Scripts\\python.exe reports/pipeline_v2/run_pipeline_v2.py [image]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.detection.factory import create_detector  # noqa: E402
from src.web.classifier_service import MobileNetClassifier  # noqa: E402
from src.web.detection_logic import analyze_image_file, build_detection_pipeline  # noqa: E402
from src.web.detection_service import DetectionService  # noqa: E402
from src.web.detection_settings import detector_input_threshold, parse_detection_settings  # noqa: E402

IMAGE = Path(sys.argv[1] if len(sys.argv) > 1 else "data/holdout/base64-1717231151649602366964.jpeg")
OUT = ROOT / "reports" / "pipeline_v2"
CHECKPOINT = ROOT / "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = IMAGE.read_bytes()
    conf = os.environ.get("PIPELINE_V2_CONF")
    suffix = f"_conf{conf}" if conf else ""
    base_settings = parse_detection_settings(
        ["--backend", "ssdlite", "--detector", str(CHECKPOINT), "--device", "cpu",
         *(["--confidence-threshold", conf] if conf else [])], {}
    )
    classifier = MobileNetClassifier(base_settings.classifier_path, device="cpu")
    modes = {
        "normal": dict(enable_tiling=False, enable_two_stage=False),
        "tiling": dict(enable_tiling=True, enable_two_stage=False),
        "two_stage": dict(enable_tiling=False, enable_two_stage=True),
        "tiling_two_stage": dict(enable_tiling=True, enable_two_stage=True),
    }
    report = {
        "image": str(IMAGE), "image_sha256": sha256(IMAGE),
        "detector_checkpoint": str(CHECKPOINT.relative_to(ROOT)), "detector_sha256": sha256(CHECKPOINT),
        "classifier_checkpoint": str(base_settings.classifier_path),
        "classifier_sha256": sha256(ROOT / base_settings.classifier_path),
        "classifier_class_names": list(classifier.class_names),
        "device": "cpu", "operating_confidence": base_settings.confidence_threshold, "modes": {},
    }
    for name, flags in modes.items():
        settings = replace(base_settings, **flags)
        base = create_detector(
            "ssdlite", settings.model_path, device="cpu",
            confidence_threshold=detector_input_threshold(settings),
            iou_threshold=settings.iou_threshold, max_detections=settings.max_detections,
        )
        pipeline = build_detection_pipeline(base, settings, classifier=classifier)
        service = DetectionService(pipeline)
        analyze_image_file(IMAGE.name, payload, service)  # warm-up (model load)
        started = time.perf_counter()
        output = analyze_image_file(IMAGE.name, payload, service)
        wall_ms = (time.perf_counter() - started) * 1000
        result = output.result
        overlay = OUT / f"{IMAGE.stem}_{name}{suffix}.png"
        overlay.write_bytes(output.annotated_png)
        in_bounds = all(
            0 <= d.box.x1 < d.box.x2 <= result.image_width
            and 0 <= d.box.y1 < d.box.y2 <= result.image_height
            for d in result.detections
        )
        tiled = getattr(getattr(pipeline, "detector", None), "detector", None)
        tiler = next(
            (obj for obj in (pipeline, getattr(pipeline, "detector", None), tiled)
             if obj is not None and hasattr(obj, "last_tile_count")), None,
        )
        report["modes"][name] = {
            **flags,
            "detector_input_threshold": detector_input_threshold(settings),
            "num_boxes": len(result.detections),
            "image_size": [result.image_width, result.image_height],
            "tiles": tiler.last_tile_count if tiler else 0,
            "pipeline_ms": round(result.inference_ms or 0.0, 1),
            "wall_ms_incl_decode_draw": round(wall_ms, 1),
            "all_boxes_in_bounds": in_bounds,
            "overlay": str(overlay.relative_to(ROOT)),
            "detections": [
                {
                    "class_id": d.class_id, "confidence": round(d.confidence, 4),
                    "box_xyxy": [round(v, 1) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                    **({"top3": [[c, round(p, 4)] for c, p in output.details[i].top_k],
                        "detector_score": round(output.details[i].detector_score, 4),
                        "detector_class_id": output.details[i].detector_class_id}
                       if output.details else {}),
                }
                for i, d in enumerate(result.detections)
            ],
        }
        print(f"{name:18s} boxes={len(result.detections):3d} "
              f"pipeline_ms={report['modes'][name]['pipeline_ms']:8.1f} in_bounds={in_bounds}")
    (OUT / f"pipeline_v2_run{suffix}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(f"report: {OUT / f'pipeline_v2_run{suffix}.json'}")


if __name__ == "__main__":
    main()

