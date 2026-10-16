"""Tiled (SAHI-style) inference: tile geometry, coordinate round-trips and merging.

Uses a real pixel-scanning fake detector (finds solid-colour rectangles in
whatever crop it receives), so tile->image coordinate mapping is verified end
to end without any knowledge of tile offsets inside the detector.
"""

from __future__ import annotations

import io
import random

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.detection.coordinates import image_to_tile, tile_to_image
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.tiling import TiledDetector, compute_tiles
from src.detection.types import BoundingBox, Detection, DetectionResult


def _color(class_index: int) -> tuple[int, int, int]:
    return (20 * (class_index + 1), 200, 255 - 20 * class_index)


class ColorRectDetector:
    """Detect solid rectangles of known colours in the given image (any size)."""

    def __init__(self, confidence: float = 0.9) -> None:
        self.calls: list[tuple[int, int]] = []
        self.confidence = confidence

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        array = np.asarray(image.convert("RGB"))
        self.calls.append(image.size)
        rows = []
        for index, name in enumerate(DETECTION_CLASS_NAMES):
            mask = np.all(array == np.array(_color(index), dtype=np.uint8), axis=-1)
            if not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            box = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            rows.append(Detection(index, name, self.confidence, box))
        return DetectionResult(tuple(rows), *image.size)


def _scene(size: tuple[int, int], objects: list[tuple[int, tuple[int, int, int, int]]]) -> Image.Image:
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for class_index, (x1, y1, x2, y2) in objects:
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=_color(class_index))
    return image


def _as_set(result: DetectionResult) -> set[tuple[int, tuple[float, ...]]]:
    return {
        (d.class_index, (round(d.box.x1), round(d.box.y1), round(d.box.x2), round(d.box.y2)))
        for d in result.detections
    }


@pytest.mark.parametrize(
    "size", [(3000, 1700), (1500, 2600), (1281, 1281), (800, 600), (640, 640), (4032, 3024)]
)
def test_tiles_cover_image_with_overlap_and_flush_edges(size) -> None:
    width, height = size
    tiles = compute_tiles(width, height, tile_size=640, overlap=0.2)
    covered = np.zeros((height, width), dtype=bool)
    for tile in tiles:
        assert 0 <= tile.x0 < tile.x1 <= width and 0 <= tile.y0 < tile.y1 <= height
        assert tile.width == min(640, width) and tile.height == min(640, height)
        covered[tile.y0:tile.y1, tile.x0:tile.x1] = True
    assert covered.all()
    assert max(t.x1 for t in tiles) == width and max(t.y1 for t in tiles) == height
    for starts in (sorted({t.x0 for t in tiles}), sorted({t.y0 for t in tiles})):
        for a, b in zip(starts, starts[1:]):
            assert b - a <= 640 * 0.8 + 1  # neighbours overlap by >= 20%


@pytest.mark.parametrize("seed", range(5))
def test_tile_image_round_trip(seed: int) -> None:
    rng = random.Random(seed)
    width, height = rng.choice([(3000, 1700), (1500, 2600), (4032, 3024)])
    for tile in compute_tiles(width, height, 640, 0.2):
        x1 = rng.uniform(0, tile.width - 2)
        y1 = rng.uniform(0, tile.height - 2)
        local = BoundingBox(x1, y1, rng.uniform(x1 + 1, tile.width), rng.uniform(y1 + 1, tile.height))
        mapped = tile_to_image(local, (tile.x0, tile.y0), (width, height))
        assert mapped is not None
        assert 0 <= mapped.x1 < mapped.x2 <= width and 0 <= mapped.y1 < mapped.y2 <= height
        back = image_to_tile(mapped, (tile.x0, tile.y0), (tile.width, tile.height))
        assert (back.x1, back.y1, back.x2, back.y2) == pytest.approx(
            (local.x1, local.y1, local.x2, local.y2)
        )


def test_tile_to_image_clips_to_image_and_rejects_outside() -> None:
    clipped = tile_to_image((500, 500, 700, 700), (2400, 1100), (3000, 1700))
    assert clipped == BoundingBox(2900.0, 1600.0, 3000.0, 1700.0)
    assert tile_to_image((700, 700, 800, 800), (2400, 1100), (3000, 1700)) is None


@pytest.mark.parametrize("size", [(3000, 1700), (1500, 2600)])
def test_tiled_detection_restores_original_coordinates_and_dedupes(size) -> None:
    width, height = size
    objects = [
        (0, (10, 12, 60, 70)),                          # top-left image corner
        (4, (width - 80, height - 50, width, height)),  # flush bottom-right image corner
        (6, (500, 500, 760, 690)),                      # straddles tile borders (stride 512)
        (7, (530, 40, 620, 120)),                       # inside x-overlap of two tiles
        (9, (100, 1000, 1400, 1200)),                   # larger than one tile -> cut in every tile
    ]
    image = _scene(size, objects)
    detector = ColorRectDetector()
    tiled = TiledDetector(detector, tile_size=640, overlap=0.2, min_side_to_tile=1280)
    result = tiled.detect_pil(image)

    assert (result.image_width, result.image_height) == size
    assert tiled.last_tile_count == len(compute_tiles(width, height, 640, 0.2)) > 1
    assert detector.calls[0] == size  # full-image pass first
    assert _as_set(result) == set(objects)
    assert len(result.detections) == len(objects)


def test_tile_only_mode_merges_duplicates_from_overlap() -> None:
    objects = [(7, (530, 40, 620, 120)), (2, (1700, 1300, 1790, 1380))]
    image = _scene((3000, 1700), objects)
    tiled = TiledDetector(ColorRectDetector(), include_full_image=False)
    result = tiled.detect_pil(image)
    assert _as_set(result) == set(objects)


def test_wbf_merge_matches_ground_truth() -> None:
    pytest.importorskip("ensemble_boxes")
    objects = [(7, (530, 40, 620, 120)), (6, (500, 500, 760, 690))]
    tiled = TiledDetector(ColorRectDetector(), merge="wbf")
    result = tiled.detect_pil(_scene((3000, 1700), objects))
    assert _as_set(result) == set(objects)


def test_small_images_bypass_tiling() -> None:
    detector = ColorRectDetector()
    tiled = TiledDetector(detector, min_side_to_tile=1280)
    result = tiled.detect_pil(_scene((1280, 720), [(1, (5, 5, 50, 50))]))
    assert detector.calls == [(1280, 720)]
    assert tiled.last_tile_count == 0
    assert _as_set(result) == {(1, (5, 5, 50, 50))}


def test_exif_orientation_is_applied_before_tiling() -> None:
    upright_objects = [(3, (100, 2000, 300, 2300)), (5, (1300, 50, 1500, 200))]
    upright = _scene((1500, 2600), upright_objects)
    # exif_transpose applies ROTATE_270 for orientation 6, so store the 90-rotated pixels.
    stored = upright.transpose(Image.Transpose.ROTATE_90)
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    stored.save(buffer, format="PNG", exif=exif.tobytes())
    reopened = Image.open(io.BytesIO(buffer.getvalue()))
    assert reopened.size == (2600, 1500)

    result = TiledDetector(ColorRectDetector()).detect_pil(reopened)
    assert (result.image_width, result.image_height) == (1500, 2600)
    assert _as_set(result) == set(upright_objects)


def test_compute_adaptive_tile_size_and_adaptive_tiling() -> None:
    from src.detection.tiling import compute_adaptive_tile_size, compute_tiles
    # Image 532x299 should scale down from 640 to ~179px to create overlapping tiles
    tile_size = compute_adaptive_tile_size(532, 299, base_tile_size=640, min_tile_size=160)
    assert 160 <= tile_size <= 200
    tiles = compute_tiles(532, 299, tile_size, overlap=0.25)
    assert len(tiles) >= 4  # multiple overlapping tiles instead of 1 placebo tile

    # Large image preserves base_tile_size
    large_tile = compute_adaptive_tile_size(3000, 2000, base_tile_size=640)
    assert large_tile == 640
