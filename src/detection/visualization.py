"""Draw Vietnamese class labels and confidence onto detection boxes."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.detection.types import DetectionResult
from src.app.presentation import COLOR_BY_CLASS, present_label

DETECTION_CLASS_NAMES_VI = (
    "Pin",
    "Rác hữu cơ",
    "Bìa carton",
    "Quần áo",
    "Thủy tinh",
    "Kim loại",
    "Giấy",
    "Nhựa",
    "Giày dép",
    "Rác khác",
)

DETECTION_DISPLAY_NAMES = {
    "battery": "Pin",
    "biological": "Rác hữu cơ",
    "cardboard": "Bìa carton",
    "clothes": "Quần áo",
    "glass": "Thủy tinh",
    "metal": "Kim loại",
    "paper": "Giấy",
    "plastic": "Nhựa",
    "shoes": "Giày dép",
    "trash": "Rác khác",
    # Legacy aliases
    "organic": "Rác hữu cơ",
    "hazardous": "Rác nguy hại",
}

DETECTION_COLORS = {
    **COLOR_BY_CLASS,
    "battery": "#DC2626",      # Red
    "biological": "#16A34A",   # Green
    "cardboard": "#D97706",    # Amber/Brown
    "clothes": "#9333EA",      # Purple
    "glass": "#0284C7",        # Sky Blue
    "metal": "#475569",        # Slate
    "paper": "#2563EB",        # Blue
    "plastic": "#EA580C",      # Orange
    "shoes": "#0D9488",        # Teal
    "trash": "#64748B",        # Gray
    "organic": "#15803D",
    "hazardous": "#B91C1C",
}

CLASS_INDEX_MAP = {
    0: "battery",
    1: "biological",
    2: "cardboard",
    3: "clothes",
    4: "glass",
    5: "metal",
    6: "paper",
    7: "plastic",
    8: "shoes",
    9: "trash",
}


def detection_label(class_id: str | int) -> str:
    """Use the 10-class Vietnamese labels without changing phase-1 classification labels.

    Supports both class name strings and integer indices (0..9).
    """
    if isinstance(class_id, int):
        if 0 <= class_id < len(DETECTION_CLASS_NAMES_VI):
            return DETECTION_CLASS_NAMES_VI[class_id]
        return f"Lớp {class_id}"
    return DETECTION_DISPLAY_NAMES.get(class_id, present_label(class_id).display_name)


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # TrueType fonts cover Vietnamese diacritics on common Windows/Linux hosts.
    for name in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _rgb_color(class_id: str | int) -> tuple[int, int, int]:
    if isinstance(class_id, int):
        class_id = CLASS_INDEX_MAP.get(class_id, "trash")
    value = DETECTION_COLORS.get(class_id, "#64748B").lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def draw_detections(
    image: Image.Image, result: DetectionResult, style: str = "box"
) -> Image.Image:
    """Return a copy with readable annotations for every detection.

    Draws sharp, rectangular bounding boxes and clear Vietnamese label tags.
    Does not draw obscuring contour masks over objects.
    """
    canvas = ImageOps.exif_transpose(image).convert("RGB")
    line_width = max(2, round(min(canvas.size) / 240))
    font_size = max(14, round(min(canvas.size) / 35))
    font = _label_font(font_size)
    padding = max(3, line_width)

    # Optional subtle fill for highlight style, without obscuring the object
    if style in ("highlight", "contour"):
        np_canvas = np.array(canvas)
        overlay = np_canvas.copy()
        for detection in result.detections:
            box = detection.box.clip(*canvas.size) or detection.box.clamp(*canvas.size)
            x1, y1, x2, y2 = round(box.x1), round(box.y1), round(box.x2), round(box.y2)
            color = _rgb_color(detection.class_id)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # Very light 15% tint so underlying object details remain completely clear
        cv2.addWeighted(overlay, 0.15, np_canvas, 0.85, 0, np_canvas)
        canvas = Image.fromarray(np_canvas)

    draw = ImageDraw.Draw(canvas)
    for detection in result.detections:
        box = detection.box.clip(*canvas.size) or detection.box.clamp(*canvas.size)
        color = _rgb_color(detection.class_id)
        coordinates = (round(box.x1), round(box.y1), round(box.x2), round(box.y2))

        # Always draw crisp rectangular bounding box border (no contour obscuring the box)
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
