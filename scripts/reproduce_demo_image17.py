import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hashlib
import json
import time
from PIL import Image
import torch

from src.detection.factory import create_detector
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.tiling import TiledDetector
from src.detection.two_stage import MobileNetBatchAdapter, TwoStageDetector
from src.detection.visualization import draw_detections
from src.web.classifier_service import MobileNetClassifier
from src.web.detection_logic import PerClassThresholdDetector, apply_class_thresholds
from src.web.detection_settings import (
    DEFAULT_CLASSIFIER_PATH,
    DEFAULT_SSDLITE_PATH,
    DEFAULT_SSDLITE_V2_PATH,
    DEFAULT_YOLO_PATH,
    DEFAULT_YOLO_V2_PATH,
    load_per_class_thresholds,
)

def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_experiment():
    img_path = (Path(__import__("os").environ.get("WASTE_HOLDOUT_DIR", str(Path(__file__).resolve().parents[1] / "data" / "holdout"))) / "images (17).jpg")
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    print(f"Target Image: {img_path} ({w}x{h})")

    # Load thresholds
    calibrated_path = Path("configs/detection_thresholds_calibrated.json")
    base_thresholds = load_per_class_thresholds(calibrated_path)
    # UI logic at confidence = 0.25:
    scale = 0.25 / 0.25 # 1.0
    active_per_class = tuple(
        (name, min(1.0, max(round(val * scale, 3), round(val * scale, 3))))
        for name, val in base_thresholds
    )
    print("Calibrated Per-Class Thresholds (at conf=0.25):", dict(active_per_class))

    # Shared classifier for Two-Stage
    classifier = MobileNetClassifier(DEFAULT_CLASSIFIER_PATH)
    batch_classifier = MobileNetBatchAdapter(classifier)

    # Models to test
    models = {
        "SSDLite_v2": DEFAULT_SSDLITE_V2_PATH,
        "SSDLite_v1": Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"),
        "YOLOv8n_v2": DEFAULT_YOLO_V2_PATH,
        "YOLOv8n_v1": Path("artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"),
    }

    results = {}

    for m_key, m_path in models.items():
        if not m_path.exists():
            print(f"Skipping {m_key}: file not found {m_path}")
            continue
        m_hash = sha256_of_file(m_path)
        backend = "yolov8n" if "YOLO" in m_key else "ssdlite"
        print(f"\n==========================================")
        print(f"Testing Model: {m_key} ({m_path})")
        print(f"SHA-256: {m_hash}")

        # 1. Raw Detector at very low threshold (0.01) to see ALL proposals
        raw_detector_low = create_detector(
            backend=backend,
            model_path=str(m_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.01,
            iou_threshold=0.55,
        )
        t0 = time.perf_counter()
        raw_low_res = raw_detector_low.detect_pil(img)
        raw_low_time = (time.perf_counter() - t0) * 1000
        print(f"Raw Detector (conf >= 0.01): {len(raw_low_res.detections)} detections ({raw_low_time:.1f} ms)")
        for d in raw_low_res.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

        # 2. Raw Detector at standard 0.25 threshold
        raw_detector_025 = create_detector(
            backend=backend,
            model_path=str(m_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.25,
            iou_threshold=0.55,
        )
        t0 = time.perf_counter()
        raw_025_res = raw_detector_025.detect_pil(img)
        raw_025_time = (time.perf_counter() - t0) * 1000
        print(f"Raw Detector (conf >= 0.25): {len(raw_025_res.detections)} detections ({raw_025_time:.1f} ms)")
        for d in raw_025_res.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

        # 3. Mode A: Detector + Tiling + Two-Stage (UI Exact setup)
        # Note: In UI, detector is created with detector_input_threshold which is 0.15 (two_stage_proposal_threshold)
        detector_stage1 = create_detector(
            backend=backend,
            model_path=str(m_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.15,
            iou_threshold=0.55,
        )
        tiled_detector = TiledDetector(
            detector_stage1,
            tile_size=640,
            overlap=0.2,
            min_side_to_tile=800, # UI sets 800
            iou_threshold=0.55,
            max_detections=100,
        )
        two_stage_detector = TwoStageDetector(
            tiled_detector,
            batch_classifier,
            pad_ratio=0.10,
            iou_threshold=0.55,
            scoring_mode="product",
            min_proposal_gate=0.28,
            min_score=0.0,
        )
        ui_calibrated_filter = PerClassThresholdDetector(
            two_stage_detector,
            default_threshold=0.25,
            per_class_thresholds=active_per_class,
        )
        flat_025_filter = PerClassThresholdDetector(
            two_stage_detector,
            default_threshold=0.25,
            per_class_thresholds=(), # Pure 0.25
        )

        t0 = time.perf_counter()
        res_A_ui = ui_calibrated_filter.detect_pil(img)
        time_A = (time.perf_counter() - t0) * 1000
        print(f"\n--- Config A (Detector + Tiling + Two-Stage + UI Calibrated Thresholds) ---")
        print(f"Boxes: {len(res_A_ui.detections)}, Processing Time: {time_A:.1f} ms")
        for d in res_A_ui.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} ({d.confidence:.1%}) box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

        # Also check with pure 0.25 filter:
        res_A_pure025 = flat_025_filter.detect_pil(img)
        print(f"Config A with Pure 0.25 Threshold (no calibrated discounts): {len(res_A_pure025.detections)} boxes")
        for d in res_A_pure025.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} ({d.confidence:.1%}) box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

        # 4. Mode B: Detector + Tiling (No Two-Stage)
        # Here base detector is at 0.25
        detector_b = create_detector(
            backend=backend,
            model_path=str(m_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.25,
            iou_threshold=0.55,
        )
        tiled_b = TiledDetector(
            detector_b,
            tile_size=640,
            overlap=0.2,
            min_side_to_tile=800,
            iou_threshold=0.55,
            max_detections=100,
        )
        t0 = time.perf_counter()
        res_b = tiled_b.detect_pil(img)
        time_b = (time.perf_counter() - t0) * 1000
        print(f"\n--- Config B (Detector + Tiling, No Two-Stage, conf=0.25) ---")
        print(f"Boxes: {len(res_b.detections)}, Processing Time: {time_b:.1f} ms")
        for d in res_b.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} ({d.confidence:.1%}) box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

        # Also check what happens if min_side_to_tile=400 (so tiling actually slices the 547x365 image!)
        tiled_b_400 = TiledDetector(
            detector_b,
            tile_size=640,
            overlap=0.2,
            min_side_to_tile=400,
            iou_threshold=0.55,
            max_detections=100,
        )
        res_b_400 = tiled_b_400.detect_pil(img)
        print(f"Config B with min_side_to_tile=400 (actual tiling executed): {len(res_b_400.detections)} boxes")
        for d in res_b_400.detections:
            print(f"  - [{d.class_id}] conf={d.confidence:.4f} ({d.confidence:.1%}) box=({d.box.x1:.1f}, {d.box.y1:.1f}, {d.box.x2:.1f}, {d.box.y2:.1f})")

if __name__ == "__main__":
    run_experiment()
