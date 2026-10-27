import hashlib
import json
import os
import sys
from pathlib import Path
import torch

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

checkpoints = [
    ("SSDLite Combined Seed42", "artifacts/detection/waste-ssdlite320-mobilenetv3-combined-seed42/weights/best.pt"),
    ("YOLOv8n Combined Seed42", "artifacts/detection/waste-yolov8n-combined-seed42/weights/best.pt"),
    ("SSDLite Baseline/Raw", "artifacts/detection/waste-ssdlite320-mobilenetv3/weights/best.pt"),
    ("YOLOv8n Baseline/Raw", "artifacts/detection/waste-yolov8n/weights/best.pt"),
]

results = []

for label, ckpt_str in checkpoints:
    p = Path(ckpt_str)
    print(f"\n==========================================")
    print(f"ANALYZING: {label} ({p})")
    print(f"==========================================")
    if not p.is_file():
        print(f"ERROR: File not found: {p}")
        continue
    
    file_bytes = p.stat().st_size
    file_sha256 = sha256_file(p)
    print(f"Size: {file_bytes} bytes")
    print(f"SHA-256: {file_sha256}")
    
    # Inspect checkpoint content
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        print(f"Type of loaded object: {type(data)}")
        if isinstance(data, dict):
            print(f"Top-level keys: {list(data.keys())}")
            if "metadata" in data:
                print(f"Metadata:")
                print(json.dumps({k: v for k, v in data['metadata'].items() if k != 'class_names'}, indent=2))
                print(f"Class names: {data['metadata'].get('class_names')}")
            if "epoch" in data:
                print(f"Checkpoint epoch field: {data['epoch']}")
            if "validation" in data:
                print(f"Validation metrics field in checkpoint: {data['validation']}")
            
            # Inspect classification head for SSDLite
            if "model_state_dict" in data:
                sd = data["model_state_dict"]
                cls_keys = [k for k in sd.keys() if "classification_head" in k]
                print(f"Classification head keys count: {len(cls_keys)}")
                for k in cls_keys:
                    t = sd[k]
                    print(f"  {k}: shape={list(t.shape)}")
            
            # If YOLO dict (Ultralytics checkpoint format)
            if "model" in data:
                yolo_model = data["model"]
                print(f"YOLO model object type: {type(yolo_model)}")
                if hasattr(yolo_model, "names"):
                    print(f"YOLO model.names: {yolo_model.names}")
                if hasattr(yolo_model, "nc"):
                    print(f"YOLO model.nc: {yolo_model.nc}")
                if hasattr(yolo_model, "yaml"):
                    print(f"YOLO model.yaml nc: {yolo_model.yaml.get('nc') if isinstance(yolo_model.yaml, dict) else None}")
                
                # Check detection head cv3
                if hasattr(yolo_model, "model") and hasattr(yolo_model.model, "__len__"):
                    head = yolo_model.model[-1]
                    print(f"Head type: {type(head).__name__}, head.nc: {getattr(head, 'nc', None)}")
                    if hasattr(head, "cv3"):
                        for i, cv_branch in enumerate(head.cv3):
                            out_conv = cv_branch[2] if hasattr(cv_branch, "__getitem__") and len(cv_branch) > 2 else getattr(cv_branch, "conv", cv_branch)
                            w = getattr(out_conv, "weight", None)
                            b = getattr(out_conv, "bias", None)
                            print(f"  cv3[{i}] weight shape: {list(w.shape) if w is not None else None}, bias shape: {list(b.shape) if b is not None else None}")
                
                if hasattr(yolo_model, "phase2_metadata"):
                    print(f"YOLO phase2_metadata: {yolo_model.phase2_metadata}")

            if "train_args" in data:
                print(f"YOLO train_args exists. Epochs: {data['train_args'].get('epochs')}")
            if "date" in data:
                print(f"YOLO checkpoint date: {data.get('date')}")
            if "best_fitness" in data:
                print(f"YOLO checkpoint best_fitness: {data.get('best_fitness')}")
        else:
            print(f"Loaded object is not a dict: {type(data)}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()

    # Check sibling training files in directory
    parent_dir = p.parent.parent
    print(f"\nChecking sibling logs in {parent_dir}:")
    for fname in ["run_metadata.json", "training_summary.json", "history.json", "effective_train_args.json", "results.csv", "args.yaml"]:
        fpath = parent_dir / fname
        if fpath.is_file():
            print(f"  Found {fname} ({fpath.stat().st_size} bytes)")
        else:
            print(f"  MISSING {fname}")
