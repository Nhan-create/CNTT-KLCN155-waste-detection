"""Analyze layer correspondence, shape matching, and stride/dilation between
Torchvision MobileNetV3-Large (Classifier) and SSDLite320-MobileNetV3 (Detector).
"""

from __future__ import annotations
import json
import torchvision.models as models
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

def main():
    cls_model = models.mobilenet_v3_large(weights=None)
    ssd_model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=7)

    cls_state = cls_model.state_dict()
    ssd_state = ssd_model.state_dict()

    print(f"Total keys in Classifier: {len(cls_state)}")
    print(f"Total keys in SSDLite: {len(ssd_state)}")

    # Examine classification head
    cls_head_keys = [k for k in cls_state if k.startswith("classifier.")]
    print(f"\nClassifier head keys (NON-TRANSFERABLE): {len(cls_head_keys)}")
    for k in cls_head_keys:
        print(f"  {k}: {list(cls_state[k].shape)}")

    # Examine SSDLite heads
    ssd_head_keys = [k for k in ssd_state if k.startswith("head.")]
    print(f"\nSSDLite Detection Head keys (NON-TRANSFERABLE, newly trained): {len(ssd_head_keys)}")

    # Examine SSDLite Extra layers
    ssd_extra_keys = [k for k in ssd_state if k.startswith("backbone.extra.")]
    print(f"\nSSDLite Extra Feature Pyramid layers (NON-TRANSFERABLE, newly trained): {len(ssd_extra_keys)}")

    # Examine Backbone features correspondence
    # SSDLite groups MobileNetV3 features into 2 feature extractors:
    # Feature 0: C4 (stride 16) - outputs from intermediate block
    # Feature 1: C5 (stride 32) - outputs from final block
    # Let's inspect backbone.features submodules
    print("\nSSDLite backbone.features submodules:")
    for i, sub in enumerate(ssd_model.backbone.features):
        print(f"  backbone.features[{i}]: {type(sub).__name__} with {len(sub)} children")
        for j, child in enumerate(sub):
            print(f"    [{j}]: {type(child).__name__}")

    # Map keys directly between cls_model.features and ssd_model.backbone.features
    print("\nClass-level mapping test:")
    # In SSDLite, backbone.features[0] contains features[0..13] of MobileNetV3
    # and backbone.features[1] contains features[14..16] of MobileNetV3!
    # Let's verify by matching parameter shapes
    mapped_count = 0
    unmapped_cls_feature_keys = []
    
    # Check shape matching
    ssd_feature_keys = [k for k in ssd_state if k.startswith("backbone.features.")]
    print(f"Total keys in ssd_model.backbone.features: {len(ssd_feature_keys)}")

    # Attempt exact parameter name mapping:
    # Notice: cls features[i] maps to:
    # if i <= 13: backbone.features.0.{i}...
    # if i >= 14: backbone.features.1.{i-14}...
    mapping_dict = {}
    for k in cls_state:
        if k.startswith("features."):
            parts = k.split(".")
            block_idx = int(parts[1])
            rest = ".".join(parts[2:])
            if block_idx <= 13:
                target_k = f"backbone.features.0.{block_idx}.{rest}"
            else:
                target_k = f"backbone.features.1.{block_idx - 14}.{rest}"
            
            if target_k in ssd_state:
                if cls_state[k].shape == ssd_state[target_k].shape:
                    mapping_dict[k] = target_k
                    mapped_count += 1
                else:
                    print(f"Shape mismatch: {k} {cls_state[k].shape} vs {target_k} {ssd_state[target_k].shape}")
            else:
                unmapped_cls_feature_keys.append((k, target_k))

    print(f"\nSuccessfully mapped feature keys: {mapped_count} / {len([k for k in cls_state if k.startswith('features.')])}")
    print(f"Unmapped feature keys: {len(unmapped_cls_feature_keys)}")
    if unmapped_cls_feature_keys:
        print("Sample unmapped keys:")
        for k, tk in unmapped_cls_feature_keys[:5]:
            print(f"  {k} -> expected {tk} (not found in SSD)")

    # Save mapping report
    report = {
        "classifier_total_keys": len(cls_state),
        "ssd_total_keys": len(ssd_state),
        "transferable_backbone_keys": mapped_count,
        "total_backbone_keys": len([k for k in cls_state if k.startswith("features.")]),
        "classifier_head_keys_excluded": len(cls_head_keys),
        "ssd_detection_head_keys_to_train": len(ssd_head_keys),
        "ssd_extra_pyramid_keys_to_train": len(ssd_extra_keys),
        "mapping_ratio": f"{mapped_count}/{len([k for k in cls_state if k.startswith('features.')])}",
    }
    with open("D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved report to D:/CNTT-KLCN155-waste-detection/data/audit/backbone_transfer_audit.json")

if __name__ == "__main__":
    main()
