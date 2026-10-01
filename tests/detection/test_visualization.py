from PIL import Image

from src.detection.types import BoundingBox, Detection, DetectionResult
from src.detection.visualization import draw_detections


def test_draw_detections_returns_copy_with_box_pixels_changed() -> None:
    image = Image.new("RGB", (160, 120), "white")
    result = DetectionResult(
        (
            Detection(
                7,
                "plastic",
                0.92,
                BoundingBox(20, 30, 120, 100),
            ),
        ),
        160,
        120,
    )

    annotated = draw_detections(image, result)

    assert image.getpixel((20, 30)) == (255, 255, 255)
    assert annotated.getpixel((20, 30)) != image.getpixel((20, 30))
    assert annotated.size == image.size
