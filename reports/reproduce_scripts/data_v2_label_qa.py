"""Label QA for real detection images (data/detection/images/real + labels/real).

Checks: label file exists, empty files, class ids in 0..9, normalized box validity,
placeholder boxes (0.5 0.5 0.4 0.4 fallback emitted by collect_real_detection_data.py),
class distribution per source. Optionally draws random overlays.

Usage: python reports/reproduce_scripts/data_v2_label_qa.py [--overlays N] [--out reports/data_v2]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "data" / "detection"
NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
COLORS = ["#ff0000", "#8b4513", "#d2b48c", "#ff00ff", "#00ffff", "#808080", "#ffffff", "#ffff00", "#0000ff", "#000000"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def source_of(name: str) -> str:
    if name.startswith("taco_"):
        return "taco"
    if name.startswith("oi_"):
        return "openimages_v7"
    return "other"


def parse_label(path: Path):
    boxes, errors = [], []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"line {ln}: expected 5 fields, got {len(parts)}")
            continue
        try:
            c = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"line {ln}: non-numeric")
            continue
        if not 0 <= c <= 9:
            errors.append(f"line {ln}: class {c} out of range")
        if w <= 0 or h <= 0:
            errors.append(f"line {ln}: non-positive size")
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
        if x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
            errors.append(f"line {ln}: box out of bounds ({x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f})")
        boxes.append((c, xc, yc, w, h))
    return boxes, errors


def is_placeholder(b) -> bool:
    _, xc, yc, w, h = b
    return abs(xc - 0.5) < 1e-6 and abs(yc - 0.5) < 1e-6 and abs(w - 0.4) < 1e-6 and abs(h - 0.4) < 1e-6


def run_qa(img_dir: Path, lbl_dir: Path):
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    img_stems = {p.stem for p in images}
    label_stems = {p.stem for p in lbl_dir.glob("*.txt")}
    res = {
        "num_images": len(images),
        "num_label_files": len(label_stems),
        "orphan_labels": sorted(label_stems - img_stems),
        "missing_labels": sorted(img_stems - label_stems),
        "empty_labels": [],
        "invalid": {},
        "placeholder_box_images": [],
        "per_source": defaultdict(lambda: {"images": 0, "boxes": Counter(), "image_presence": Counter()}),
        "unreadable_images": [],
        "tiny_boxes": [],
    }
    for p in images:
        src = source_of(p.name)
        res["per_source"][src]["images"] += 1
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:  # noqa: BLE001
            res["unreadable_images"].append(f"{p.name}: {exc}")
        lp = lbl_dir / f"{p.stem}.txt"
        if not lp.exists():
            continue
        boxes, errs = parse_label(lp)
        if errs:
            res["invalid"][p.name] = errs
        if not boxes:
            res["empty_labels"].append(p.name)
        if any(is_placeholder(b) for b in boxes):
            res["placeholder_box_images"].append(p.name)
        for b in boxes:
            res["per_source"][src]["boxes"][NAMES[b[0]] if 0 <= b[0] <= 9 else str(b[0])] += 1
            if b[3] * b[4] < 1e-4:
                res["tiny_boxes"].append(p.name)
        for c in {b[0] for b in boxes}:
            if 0 <= c <= 9:
                res["per_source"][src]["image_presence"][NAMES[c]] += 1
    res["per_source"] = {k: {"images": v["images"], "boxes": dict(v["boxes"]), "image_presence": dict(v["image_presence"])} for k, v in res["per_source"].items()}
    res["tiny_boxes"] = sorted(set(res["tiny_boxes"]))
    return res


def draw_overlay(img_path: Path, lbl_path: Path, out_path: Path):
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    boxes, _ = parse_label(lbl_path) if lbl_path.exists() else ([], [])
    lw = max(2, W // 300)
    for c, xc, yc, w, h in boxes:
        x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
        col = COLORS[c] if 0 <= c <= 9 else "#00ff00"
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        d.rectangle([x1, max(0, y1 - 14), x1 + 8 * len(NAMES[c]) + 6, y1], fill=col)
        d.text((x1 + 3, max(0, y1 - 13)), NAMES[c], fill="#000000" if c not in (9, 8, 1) else "#ffffff")
    im.thumbnail((1024, 1024))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=str(DET / "images" / "real"))
    ap.add_argument("--lbl-dir", default=str(DET / "labels" / "real"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "data_v2"))
    ap.add_argument("--overlays", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="label_qa")
    args = ap.parse_args()
    img_dir, lbl_dir, out = Path(args.img_dir), Path(args.lbl_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = run_qa(img_dir, lbl_dir)
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: (len(v) if isinstance(v, (list, dict)) and k != "per_source" else v) for k, v in res.items()}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.overlays > 0:
        rng = random.Random(args.seed)
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        for p in rng.sample(imgs, min(args.overlays, len(imgs))):
            draw_overlay(p, lbl_dir / f"{p.stem}.txt", out / "overlays" / f"{p.stem}_overlay.jpg")
        print(f"overlays written to {out / 'overlays'}")


if __name__ == "__main__":
    main()
