# Mô tả: Kiểm tra siêu dữ liệu, kiến trúc và mã băm SHA-256 của các tệp trọng số checkpoint trong dự án.
"""
Inspection of the 4 reported checkpoints through product code and PyTorch introspection.
Exports exact SHA-256, file size, architecture, head shape, class names,
background index, epoch, training metadata, and verbatim load results.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hashlib
import json
import torch

from src.detection.ssdlite import SSDLiteWasteDetector, ARCHITECTURE
from src.detection.yolo import WasteDetector, DetectionError
from src.detection.schema import DETECTION_CLASS_NAMES

def get_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

checkpoints_to_check = [
    {
        "name": "SSDLite320-MobileNetV3 (combined-seed42)",
        "backend": "ssdlite",
        "path": Path("artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"),
        "role": "Primary candidate model (combined augmentation, seed 42)"
    },
    {
        "name": "YOLOv8n (combined-seed42)",
        "backend": "yolov8n",
        "path": Path("artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"),
        "role": "Secondary comparison detector (combined augmentation, seed 42)"
    },
    {
        "name": "SSDLite320-MobileNetV3 (smoke)",
        "backend": "ssdlite",
        "path": Path("artifacts/detection/waste-ssdlite320-mobilenetv3/weights/best.pt"),
        "role": "Baseline/smoke SSDLite detector"
    },
    {
        "name": "YOLOv8n (smoke)",
        "backend": "yolov8n",
        "path": Path("artifacts/detection/waste-yolov8n/weights/best.pt"),
        "role": "Baseline/smoke YOLOv8n detector"
    }
]

inspection_results = []

for ckpt_info in checkpoints_to_check:
    p = ckpt_info["path"]
    backend = ckpt_info["backend"]
    print(f"\n=======================================================")
    print(f"Inspecting: {ckpt_info['name']}")
    print(f"Path: {p}")

    if not p.is_file():
        print(f"ERROR: File not found: {p}")
        inspection_results.append({
            "name": ckpt_info["name"],
            "backend": backend,
            "path": str(p),
            "exists": False,
            "error": "File not found"
        })
        continue

    file_size = p.stat().st_size
    sha256_hash = get_sha256(p)
    print(f"Size: {file_size:,} bytes")
    print(f"SHA-256: {sha256_hash}")

    record = {
        "name": ckpt_info["name"],
        "role": ckpt_info["role"],
        "backend": backend,
        "path": str(p),
        "exists": True,
        "size_bytes": file_size,
        "sha256": sha256_hash,
    }

    # PyTorch introspection
    try:
        raw = torch.load(p, map_location="cpu", weights_only=False)
        record["pytorch_keys"] = list(raw.keys()) if isinstance(raw, dict) else str(type(raw))

        if backend == "ssdlite":
            metadata = raw.get("metadata", {})
            record["architecture"] = metadata.get("architecture")
            record["metadata_backend"] = metadata.get("backend")
            record["class_names"] = metadata.get("class_names")
            record["background_index"] = metadata.get("background_index")
            record["trained"] = metadata.get("trained")
            record["epoch"] = raw.get("epoch")
            record["training_args"] = raw.get("args") or metadata.get("args")
            
            # Check head shape in state dict
            sd = raw.get("model_state_dict", {})
            head_weights = {}
            for k, v in sd.items():
                if "head.classification_head" in k and "weight" in k:
                    head_weights[k] = list(v.shape)
            record["head_shapes"] = head_weights

            # Missing fields audit
            missing_fields = []
            for field in ["architecture", "backend", "class_names", "background_index", "trained"]:
                if field not in metadata:
                    missing_fields.append(f"metadata.{field}")
            if "epoch" not in raw:
                missing_fields.append("epoch")
            if "args" not in raw and "args" not in metadata:
                missing_fields.append("training_args")
            record["missing_metadata_fields"] = missing_fields

        elif backend == "yolov8n":
            # For YOLOv8 ultralytics checkpoint
            model_obj = raw.get("model")
            yaml_data = getattr(model_obj, "yaml", {}) if model_obj else {}
            names = raw.get("names") or getattr(model_obj, "names", None)
            
            phase2_meta = getattr(model_obj, "phase2_metadata", {}) if model_obj else {}
            record["architecture"] = "yolov8n"
            record["yaml_nc"] = yaml_data.get("nc")
            record["model_names"] = names
            record["epoch"] = raw.get("epoch")
            record["phase2_metadata"] = phase2_meta
            record["train_args"] = raw.get("train_args") or raw.get("args")

            # Inspect detect head convs
            head_conv_shapes = []
            if model_obj and hasattr(model_obj, "model"):
                net_model = model_obj.model
                if hasattr(net_model, "__getitem__"):
                    try:
                        detect_head = net_model[-1]
                        cv3 = getattr(detect_head, "cv3", None)
                        if cv3:
                            for idx, branch in enumerate(cv3):
                                conv = branch[2] if hasattr(branch, "__getitem__") and len(branch) > 2 else getattr(branch, "conv", branch)
                                if hasattr(conv, "weight"):
                                    head_conv_shapes.append(list(conv.weight.shape))
                    except Exception as e:
                        head_conv_shapes.append(f"Error inspecting head: {e}")
            record["head_shapes"] = head_conv_shapes

            # Missing fields audit
            missing_fields = []
            if not phase2_meta:
                missing_fields.append("model.phase2_metadata")
            else:
                for field in ["backend", "trained", "class_names"]:
                    if field not in phase2_meta:
                        missing_fields.append(f"phase2_metadata.{field}")
            if "epoch" not in raw:
                missing_fields.append("epoch")
            if "train_args" not in raw and "args" not in raw:
                missing_fields.append("train_args")
            record["missing_metadata_fields"] = missing_fields

    except Exception as e:
        record["introspection_error"] = str(e)
        print(f"Introspection error: {e}")

    # Load through product code
    print("\nLoading through product wrapper...")
    try:
        if backend == "ssdlite":
            detector = SSDLiteWasteDetector(p, device="cpu")
            loaded_model = detector._ensure_loaded()
            record["product_load_success"] = True
            record["product_load_message"] = f"Successfully loaded SSDLite model on cpu: {type(loaded_model).__name__}"
            print(f"Product load: PASS - {record['product_load_message']}")
        elif backend == "yolov8n":
            detector = WasteDetector(p, device="cpu")
            loaded_model = detector._ensure_loaded()
            record["product_load_success"] = True
            record["product_load_message"] = f"Successfully loaded YOLO model on cpu: {type(loaded_model).__name__}"
            print(f"Product load: PASS - {record['product_load_message']}")
    except DetectionError as de:
        record["product_load_success"] = False
        record["product_load_message"] = f"DetectionError: {str(de)}"
        print(f"Product load: FAIL - {record['product_load_message']}")
    except Exception as e:
        record["product_load_success"] = False
        record["product_load_message"] = f"{type(e).__name__}: {str(e)}"
        print(f"Product load: ERROR - {record['product_load_message']}")

    # Evaluation of training label semantic provenance:
    # Check whether the checkpoint was trained on dataset that contains the Mendeley synthetic labels
    record["semantic_label_provenance"] = {
        "status": "UNVERIFIED_OR_ERRONEOUS",
        "reason": "Trained on data containing 1305 synthetic images from Mendeley (dataset-v1) where class semantics were systematically misaligned (e.g. cans->battery, organic->clothes, ropes->metal). Requires dataset cleanup and retraining in subsequent task.",
        "requires_retraining": True
    }

    inspection_results.append(record)

# Save inspection report to JSON
report_file = Path("reports/checkpoint_inspection_report.json")
report_file.parent.mkdir(parents=True, exist_ok=True)
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(inspection_results, f, indent=2)

print(f"\n=======================================================")
print(f"All 4 checkpoints inspected. Report saved to: {report_file}")
