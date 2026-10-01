"""Draw Vietnamese class labels and confidence onto detection boxes."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from src.detection.types import DetectionResult
from src.ui.presentation import COLOR_BY_CLASS, present_label

DETECTION_DISPLAY_NAMES = {
    "plastic": "Nhựa",
    "paper": "Giấy",
    "metal": "Kim loại",
    "glass": "Thủy tinh",
    "organic": "Rác hữu cơ",
    "hazardous": "Rác nguy hại",
}
DETECTION_COLORS = {
    **COLOR_BY_CLASS,
    "organic": "#15803D",
    "hazardous": "#B91C1C",
}


def detection_label(class_id: str) -> str:
    """Use the phase-2 labels without changing phase-1 classification labels."""
    return DETECTION_DISPLAY_NAMES.get(class_id, present_label(class_id).display_name)


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # TrueType fonts cover Vietnamese diacritics on common Windows/Linux hosts.
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _rgb_color(class_id: str) -> tuple[int, int, int]:
    value = DETECTION_COLORS.get(class_id, "#64748B").lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def draw_detections(image: Image.Image, result: DetectionResult) -> Image.Image:
    """Return a copy with a readable box and label for every detection."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line_width = max(2, round(min(canvas.size) / 240))
    font_size = max(14, round(min(canvas.size) / 35))
    font = _label_font(font_size)
    padding = max(3, line_width)

    for detection in result.detections:
        box = detection.box.clamp(*canvas.size)
        color = _rgb_color(detection.class_id)
        coordinates = (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        draw.rectangle(coordinates, outline=color, width=line_width)
        label = detection_label(detection.class_id)
        text = f"{label} {detection.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
        label_x = max(0, min(coordinates[0], canvas.width - text_width - padding * 2))
        label_y = max(0, coordinates[1] - text_height - padding * 2)
        background = (
            label_x,
            label_y,
            min(canvas.width, label_x + text_width + padding * 2),
            min(canvas.height, label_y + text_height + padding * 2),
        )
        draw.rectangle(background, fill=color)
        draw.text(
            (label_x + padding, label_y + padding - top),
            text,
            fill=(255, 255, 255),
            font=font,
        )
    return canvas
