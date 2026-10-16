"""Coordinate conversions, EXIF orientation handling, letterboxing, and box clipping."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageOps

from src.detection.types import BoundingBox


def ensure_exif_orientation(image: Image.Image) -> Image.Image:
    """Normalize image orientation based on EXIF tag and convert to RGB.

    Ensures that phone or camera captures containing orientation tags (1-8)
    are transposed to standard upright orientation before model inference and drawing.
    """
    return ImageOps.exif_transpose(image).convert("RGB")


def letterbox_image(
    image: Image.Image,
    target_size: int | tuple[int, int] = 640,
    fill_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, tuple[float, float]]:
    """Letterbox an image into target_size while strictly preserving aspect ratio.

    Returns:
        (letterboxed_image, scale_factor, (offset_dx, offset_dy))
    """
    orig_w, orig_h = image.size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

    target_w, target_h = (
        (target_size, target_size) if isinstance(target_size, int) else target_size
    )
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"Invalid target dimensions: {target_w}x{target_h}")

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Offset to center the resized image inside target dimensions
    offset_dx = (target_w - new_w) / 2.0
    offset_dy = (target_h - new_h) / 2.0

    canvas = Image.new("RGB", (target_w, target_h), fill_color)
    paste_x = round(offset_dx)
    paste_y = round(offset_dy)
    canvas.paste(resized, (paste_x, paste_y))

    return canvas, scale, (float(paste_x), float(paste_y))


def scale_coords_back(
    boxes: Sequence[tuple[float, float, float, float] | BoundingBox],
    scale: float,
    pad: tuple[float, float],
    orig_size: tuple[int, int],
) -> list[BoundingBox]:
    """Inverse-transform bounding boxes from letterbox space back to original image space.

    Formula:
        x_orig = (x_padded - pad_x) / scale
        y_orig = (y_padded - pad_y) / scale

    Clips strictly to [0, orig_w] and [0, orig_h]. Completely filters out:
        - Non-finite coordinates (NaN, Inf)
        - Empty or inverted boxes (x1 >= x2 or y1 >= y2)
        - Boxes completely outside original image bounds
    """
    orig_w, orig_h = orig_size
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid original image dimensions: {orig_w}x{orig_h}")
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError(f"Invalid scale factor: {scale}")

    pad_x, pad_y = pad
    if not (math.isfinite(pad_x) and math.isfinite(pad_y)):
        raise ValueError(f"Invalid padding offsets: {pad}")

    restored: list[BoundingBox] = []
    for item in boxes:
        if isinstance(item, BoundingBox):
            x1, y1, x2, y2 = item.x1, item.y1, item.x2, item.y2
        else:
            x1, y1, x2, y2 = (float(v) for v in item)

        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue

        # Invert padding offset and scale
        ox1 = (x1 - pad_x) / scale
        oy1 = (y1 - pad_y) / scale
        ox2 = (x2 - pad_x) / scale
        oy2 = (y2 - pad_y) / scale

        # Eliminate inverted or degenerate box before clipping
        if ox1 >= ox2 or oy1 >= oy2:
            continue

        # Clip strictly to original image pixel boundaries: 0 <= x1 < x2 <= orig_w
        cx1 = max(0.0, min(ox1, float(orig_w)))
        cy1 = max(0.0, min(oy1, float(orig_h)))
        cx2 = max(0.0, min(ox2, float(orig_w)))
        cy2 = max(0.0, min(oy2, float(orig_h)))

        # Eliminate boxes that have zero or negative area after clipping
        if cx1 >= cx2 or cy1 >= cy2:
            continue

        restored.append(BoundingBox(cx1, cy1, cx2, cy2))

    return restored


def scale_coords_direct(
    boxes: Sequence[tuple[float, float, float, float] | BoundingBox],
    input_size: tuple[int, int],
    orig_size: tuple[int, int],
) -> list[BoundingBox]:
    """Inverse-transform bounding boxes from directly resized (stretched) space back to original image space.

    Formula:
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        x_orig = x * scale_x
        y_orig = y * scale_y
    """
    orig_w, orig_h = orig_size
    input_w, input_h = input_size
    if orig_w <= 0 or orig_h <= 0 or input_w <= 0 or input_h <= 0:
        raise ValueError("Dimensions must be positive")

    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    restored: list[BoundingBox] = []
    for item in boxes:
        if isinstance(item, BoundingBox):
            x1, y1, x2, y2 = item.x1, item.y1, item.x2, item.y2
        else:
            x1, y1, x2, y2 = (float(v) for v in item)

        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue

        ox1 = x1 * scale_x
        oy1 = y1 * scale_y
        ox2 = x2 * scale_x
        oy2 = y2 * scale_y

        if ox1 >= ox2 or oy1 >= oy2:
            continue

        cx1 = max(0.0, min(ox1, float(orig_w)))
        cy1 = max(0.0, min(oy1, float(orig_h)))
        cx2 = max(0.0, min(ox2, float(orig_w)))
        cy2 = max(0.0, min(oy2, float(orig_h)))

        if cx1 >= cx2 or cy1 >= cy2:
            continue

        restored.append(BoundingBox(cx1, cy1, cx2, cy2))

    return restored


def clip_and_validate_box(
    box: tuple[float, float, float, float] | BoundingBox,
    image_width: int,
    image_height: int,
) -> BoundingBox | None:
    """Strictly clip and validate a bounding box against image boundaries.

    Returns a valid BoundingBox satisfying:
        0 <= x1 < x2 <= image_width
        0 <= y1 < y2 <= image_height
    Returns None if:
        - Coordinates contain NaN / Inf
        - Box is empty or inverted (x1 >= x2 or y1 >= y2)
        - Box lies entirely outside the image
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    if isinstance(box, BoundingBox):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    else:
        x1, y1, x2, y2 = (float(v) for v in box)

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None

    if x1 >= x2 or y1 >= y2:
        return None

    cx1 = max(0.0, min(x1, float(image_width)))
    cy1 = max(0.0, min(y1, float(image_height)))
    cx2 = max(0.0, min(x2, float(image_width)))
    cy2 = max(0.0, min(y2, float(image_height)))

    if cx1 >= cx2 or cy1 >= cy2:
        return None

    return BoundingBox(cx1, cy1, cx2, cy2)


def tile_to_image(
    box: tuple[float, float, float, float] | BoundingBox,
    offset: tuple[float, float],
    image_size: tuple[int, int],
) -> BoundingBox | None:
    """Translate a box from tile-local pixels to original-image pixels, then clip.

    ``offset`` is the (x0, y0) of the tile's top-left corner in the original image.
    """
    if isinstance(box, BoundingBox):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    else:
        x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (float(v) for v in offset)
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError(f"Invalid tile offset: {offset}")
    return clip_and_validate_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), *image_size)


def image_to_tile(
    box: tuple[float, float, float, float] | BoundingBox,
    offset: tuple[float, float],
    tile_size: tuple[int, int],
) -> BoundingBox | None:
    """Inverse of :func:`tile_to_image`: original-image pixels to tile-local pixels (clipped)."""
    if isinstance(box, BoundingBox):
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    else:
        x1, y1, x2, y2 = (float(v) for v in box)
    dx, dy = (float(v) for v in offset)
    return clip_and_validate_box((x1 - dx, y1 - dy, x2 - dx, y2 - dy), *tile_size)


def padded_crop_box(
    box: tuple[float, float, float, float] | BoundingBox,
    image_width: int,
    image_height: int,
    pad_ratio: float = 0.10,
) -> tuple[int, int, int, int] | None:
    """Expand a box by ``pad_ratio`` of its width/height on each side and clip.

    Returns integer pixel bounds ``(left, top, right, bottom)`` suitable for
    ``Image.crop`` with ``0 <= left < right <= W`` and ``0 <= top < bottom <= H``.
    """
    if pad_ratio < 0 or not math.isfinite(pad_ratio):
        raise ValueError("pad_ratio must be finite and non-negative")
    valid = clip_and_validate_box(box, image_width, image_height)
    if valid is None:
        return None
    pad_x = valid.width * pad_ratio
    pad_y = valid.height * pad_ratio
    left = max(0, math.floor(valid.x1 - pad_x))
    top = max(0, math.floor(valid.y1 - pad_y))
    right = min(image_width, math.ceil(valid.x2 + pad_x))
    bottom = min(image_height, math.ceil(valid.y2 + pad_y))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
