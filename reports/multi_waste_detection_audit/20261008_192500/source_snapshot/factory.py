"""Single backend selection contract shared by the demo and evaluation CLI."""

from pathlib import Path

from src.detection.fusion import FusionDetector


def create_detector(
    backend: str, model_path: Path | str, *, secondary_model_path: Path | str | None = None,
    device: str = "auto", confidence_threshold: float = 0.35,
    iou_threshold: float = 0.55, image_size: int = 640, max_detections: int = 100,
    fusion_weights: tuple[float, float] = (1.0, 1.0), fusion_iou_threshold: float = 0.55,
    fusion_input_threshold: float | None = None,
):
    from src.detection.ssdlite import SSDLiteWasteDetector
    from src.detection.yolo import WasteDetector

    options = dict(device=device, confidence_threshold=confidence_threshold,
                   iou_threshold=iou_threshold, max_detections=max_detections)
    if backend == "ssdlite":
        return SSDLiteWasteDetector(model_path, image_size=320, **options)
    if backend == "yolov8n":
        return WasteDetector(model_path, image_size=image_size, **options)
    if backend != "wbf":
        raise ValueError(f"Unknown detector backend: {backend}")
    if secondary_model_path is None:
        raise ValueError("WBF requires both SSDLite and YOLOv8n checkpoints")
    input_threshold = confidence_threshold if fusion_input_threshold is None else fusion_input_threshold
    options["confidence_threshold"] = input_threshold
    return FusionDetector(
        (SSDLiteWasteDetector(model_path, image_size=320, **options),
         WasteDetector(secondary_model_path, image_size=image_size, **options)),
        weights=fusion_weights, iou_threshold=fusion_iou_threshold,
        confidence_threshold=input_threshold, output_threshold=confidence_threshold,
        max_detections=max_detections,
    )
