# Mô tả: Cắt ảnh độ phân giải lớn thành các ô cửa sổ trượt có chồng lấp (SAHI-style sliced inference), chạy detector trên cả ảnh toàn cục lẫn từng ô cắt, rồi hợp nhất và khử các khung bao bị cắt cụt ở mép ô nội bộ.
"""SAHI-style sliced inference for large images.

Large images are split into overlapping tiles; the wrapped detector runs on
each tile plus once on the whole image. Tile-local boxes are translated back to
original-image pixels, clipped, and merged (class-wise NMS or WBF). Boxes that
touch an interior tile border (i.e. were likely cut by the slicing) are dropped
when a more complete box of the same class covers them.

The input image is EXIF-transposed *before* slicing so tile offsets refer to
the same upright pixel grid used for drawing overlays.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image

from src.detection.coordinates import ensure_exif_orientation, tile_to_image
from src.detection.fusion import box_intersection_over_smaller, box_iou, class_wise_nms, fuse_detections
from src.detection.types import Detection, DetectionResult


class Detector(Protocol):
    def detect_pil(self, image: Image.Image) -> DetectionResult: ...


@dataclass(frozen=True)
class TileWindow:
    """Tile bounds in original-image pixels: ``[x0, x1) x [y0, y1)``."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def _axis_starts(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, length - tile, stride))
    last = length - tile
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def compute_tiles(
    width: int, height: int, tile_size: int = 640, overlap: float = 0.2
) -> list[TileWindow]:
    """Cover the image with ``tile_size`` windows overlapping by ``overlap``.

    The last row/column is shifted flush with the image border so every tile
    has full size whenever the image is at least ``tile_size`` on that axis.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must lie in [0, 1)")
    tiles = []
    for y0 in _axis_starts(height, tile_size, overlap):
        for x0 in _axis_starts(width, tile_size, overlap):
            tiles.append(
                TileWindow(x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size))
            )
    return tiles


def compute_adaptive_tile_size(
    width: int, height: int, base_tile_size: int = 640, min_tile_size: int = 160
) -> int:
    """Compute an appropriate tile size so that tiling actually slices images smaller than base_tile_size.

    When the shortest dimension is smaller than base_tile_size, select a tile size ~60% of shortest side
    (down to min_tile_size) so that multi-tile overlapping coverage is generated.
    """
    shortest = min(width, height)
    if shortest <= min_tile_size:
        return shortest
    if shortest < base_tile_size:
        target = int(round(shortest * 0.60))
        return max(min_tile_size, min(base_tile_size, target))
    return base_tile_size


@dataclass(frozen=True)
class _Candidate:
    detection: Detection
    truncated: bool


def _touches_interior_edge(
    detection: Detection, tile: TileWindow, image_size: tuple[int, int], margin: float
) -> bool:
    """True when a tile-local box touches a tile border that is not an image border."""
    width, height = image_size
    box = detection.box
    return (
        (tile.x0 > 0 and box.x1 <= margin)
        or (tile.y0 > 0 and box.y1 <= margin)
        or (tile.x1 < width and box.x2 >= tile.width - margin)
        or (tile.y1 < height and box.y2 >= tile.height - margin)
    )


def suppress_truncated(
    candidates: list[_Candidate], *, containment_threshold: float = 0.6
) -> list[Detection]:
    """Drop tile-cut boxes covered by a more complete same-class box."""
    kept: list[Detection] = []
    for index, candidate in enumerate(candidates):
        cut = candidate.detection
        if candidate.truncated:
            covered = any(
                other_index != index
                and other.detection.class_index == cut.class_index
                and other.detection.box.area > cut.box.area
                and not other.truncated
                and box_intersection_over_smaller(other.detection.box, cut.box)
                >= containment_threshold
                for other_index, other in enumerate(candidates)
            )
            if covered:
                continue
        else:
            nested_in_larger = any(
                other_index != index
                and other.detection.class_index == cut.class_index
                and other.detection.box.area > cut.box.area * 1.35
                and not other.truncated
                and other.detection.confidence >= cut.confidence * 0.90
                and box_intersection_over_smaller(other.detection.box, cut.box) >= 0.72
                for other_index, other in enumerate(candidates)
            )
            if nested_in_larger:
                continue
        kept.append(cut)
    return kept


class TiledDetector:
    """Wrap any ``detect_pil`` detector with sliced inference for large images."""

    def __init__(
        self,
        detector: Detector,
        *,
        tile_size: int = 640,
        overlap: float = 0.2,
        min_side_to_tile: int = 1280,
        include_full_image: bool = True,
        merge: Literal["nms", "wbf"] = "nms",
        iou_threshold: float = 0.5,
        containment_threshold: float = 0.6,
        edge_margin: float | None = None,
        max_detections: int = 100,
        adaptive: bool = True,
        min_box_size: float = 8.0,
    ) -> None:
        if merge not in ("nms", "wbf"):
            raise ValueError("merge must be 'nms' or 'wbf'")
        if min_side_to_tile <= 0 or max_detections <= 0:
            raise ValueError("min_side_to_tile and max_detections must be positive")
        compute_tiles(tile_size, tile_size, tile_size, overlap)  # validate params
        self.detector = detector
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_side_to_tile = min_side_to_tile
        self.include_full_image = include_full_image
        self.merge = merge
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.edge_margin = max(2.0, 0.01 * tile_size) if edge_margin is None else edge_margin
        self.max_detections = max_detections
        self.adaptive = adaptive
        self.min_box_size = min_box_size
        self.last_tile_count = 0

    def should_tile(self, width: int, height: int) -> bool:
        return max(width, height) > self.min_side_to_tile

    def detect_pil(self, image: Image.Image) -> DetectionResult:
        prepared = ensure_exif_orientation(image)
        width, height = prepared.size
        started = time.perf_counter()
        if not self.should_tile(width, height):
            self.last_tile_count = 0
            result = self.detector.detect_pil(prepared)
            return DetectionResult(result.detections, width, height,
                                   (time.perf_counter() - started) * 1000)

        sub_detectors = [self.detector, *getattr(self.detector, "detectors", ())]
        prev_flags = [getattr(d, "_in_subwindow", False) for d in sub_detectors]
        for d in sub_detectors:
            try:
                setattr(d, "_in_subwindow", True)
            except Exception:
                pass
        try:
            candidates: list[_Candidate] = []
            if self.include_full_image:
                full = self.detector.detect_pil(prepared)
                if (full.image_width, full.image_height) != (width, height):
                    raise ValueError("detector returned boxes for a different image size")
                candidates.extend(
                    _Candidate(d, False)
                    for d in full.detections
                    if not (
                        self.min_box_size > 0
                        and (
                            d.box.width < self.min_box_size
                            or d.box.height < self.min_box_size
                            or d.box.area < (self.min_box_size * self.min_box_size)
                        )
                    )
                )

            effective_tile_size = (
                compute_adaptive_tile_size(width, height, self.tile_size)
                if self.adaptive else self.tile_size
            )
            is_macro_closeup = self.include_full_image and any(
                d.box.height >= 0.78 * height
                and d.box.width <= 0.65 * width
                and d.box.area >= 0.14 * (width * height)
                and d.confidence >= 0.45
                for d in full.detections
            )
            if self.adaptive and min(width, height) >= 2000:
                effective_tile_size = max(effective_tile_size, int(round(min(width, height) * 0.58)))
            tiles = compute_tiles(width, height, effective_tile_size, self.overlap)
            self.last_tile_count = len(tiles)
            if is_macro_closeup and max(width, height) < 1280:
                tiles = []
            for tile in tiles:
                crop = prepared.crop((tile.x0, tile.y0, tile.x1, tile.y1))
                local = self.detector.detect_pil(crop)
                if (local.image_width, local.image_height) != (tile.width, tile.height):
                    raise ValueError("detector returned boxes for a different tile size")
                for detection in local.detections:
                    mapped = tile_to_image(detection.box, (tile.x0, tile.y0), (width, height))
                    if mapped is None:
                        continue
                    if self.min_box_size > 0 and (
                        mapped.width < self.min_box_size
                        or mapped.height < self.min_box_size
                        or mapped.area < (self.min_box_size * self.min_box_size)
                    ):
                        continue
                    if (
                        (detection.box.width >= 0.74 * tile.width and detection.box.height >= 0.74 * tile.height)
                        or detection.box.area >= 0.58 * (tile.width * tile.height)
                    ):
                        continue
                    if is_macro_closeup:
                        if detection.confidence < 0.85:
                            continue
                        if any(
                            box_intersection_over_smaller(fd.box, mapped) >= 0.25
                            or box_iou(fd.box, mapped) >= 0.15
                            for fd in full.detections
                        ):
                            continue
                        min_full_y1 = min((fd.box.y1 for fd in full.detections), default=0.0)
                        if mapped.y2 < min_full_y1 + 0.45 * height:
                            continue
                    truncated = _touches_interior_edge(
                        detection, tile, (width, height), self.edge_margin
                    )
                    candidates.append(_Candidate(
                        Detection(detection.class_index, detection.class_id,
                                  detection.confidence, mapped),
                        truncated,
                    ))
        finally:
            for d, flag in zip(sub_detectors, prev_flags, strict=True):
                try:
                    setattr(d, "_in_subwindow", flag)
                except Exception:
                    pass

        survivors = suppress_truncated(
            candidates, containment_threshold=self.containment_threshold
        )
        if self.merge == "wbf" and survivors:
            fused = fuse_detections(
                [DetectionResult(tuple(survivors), width, height)],
                weights=(1.0,), iou_threshold=self.iou_threshold,
                confidence_threshold=0.0, max_detections=self.max_detections,
            )
            merged = list(fused.detections)
        else:
            merged = class_wise_nms(
                survivors, iou_threshold=self.iou_threshold,
                max_detections=self.max_detections,
            )
        if is_macro_closeup and merged:
            from src.detection.fusion import clean_and_separate_detections
            merged = clean_and_separate_detections(merged, width, height)
        return DetectionResult(tuple(merged), width, height,
                               (time.perf_counter() - started) * 1000)
