# Mô tả: Phiên bản v2 của công cụ thu thập và chuẩn hóa ảnh phát hiện rác thực tế.
"""scripts/collect_real_detection_v2.py
-------------------------------------
Data v2 expansion (Agent A). Non-destructive: never overwrites an existing image or label.

1. TACO (official annotations.json, 1500 images) -> images/real/taco_XXXX.jpg + labels/real/taco_XXXX.txt
   mapped with configs/detection_source_mapping.yaml::sources.taco.category_mapping.
   - Downloads the Flickr 640px rendition (fallback: original URL).
   - Orientation guard: if downloaded aspect ratio is transposed w.r.t. annotation width/height,
     the image is NOT labeled and is recorded as orientation_mismatch (excluded downstream).
   - Ambiguous/unmapped TACO categories (class_id null) are not written to the 10-class labels;
     they are kept in data/detection/real_v2_ambiguous_boxes.json (used for class-agnostic labels).
2. OpenImages V7 validation supplement for under-represented classes, using only labels present in
   sources.openimages_v7.class_mapping with a non-null class_id. Image-level filters:
   - no human annotation anywhere in the image (Person/Man/Woman/Boy/Girl/Human body parts):
     enforces acceptance_criteria.exclude_in_use_person_clothing (worn shoes/clothes, held cans).
   - no mapped box flagged IsDepiction or IsGroupOf.
   - capped per class (default 300), seed 42.
3. Writes data/detection/real_v2_source_manifest.csv (per-image provenance).

Usage:
  python scripts/collect_real_detection_v2.py [--workers 16] [--oi-cap 300] [--skip-taco] [--skip-oi]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "data" / "detection"
IMG_DIR = DET / "images" / "real"
LBL_DIR = DET / "labels" / "real"
TACO_ANN = ROOT / "data" / "audit" / "taco_annotations_raw.json"
OI_BOXES = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training"))) / "openimages-validation-boxes.csv")
AMBIG_OUT = DET / "real_v2_ambiguous_boxes.json"
MANIFEST_OUT = DET / "real_v2_source_manifest.csv"

HUMAN_LABELS = {
    "/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655", "/m/02p0tk3", "/m/035r7c",
    "/m/031n1", "/m/0dzf4", "/m/0k65p", "/m/04hgtk", "/m/0dzct", "/m/03q69",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch(url: str, timeout: int = 30, retries: int = 2) -> bytes | None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                data = r.read()
            if len(data) > 1024:
                return data
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return None


def coco_to_yolo(bbox, W, H):
    x, y, w, h = bbox
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def write_label(path: Path, boxes):
    path.write_text("".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in boxes), encoding="utf-8")


def save_image(data: bytes, dest: Path):
    """Validate decode, then write bytes as-is (no re-encode)."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        size = im.size
    dest.write_bytes(data)
    return size


# ----------------------------------------------------------------------------- TACO
def collect_taco(cfg, workers: int, records: list, ambiguous: dict):
    taco = json.loads(TACO_ANN.read_text(encoding="utf-8"))
    cmap = {int(k): v for k, v in cfg["sources"]["taco"]["category_mapping"].items()}
    cat_names = {c["id"]: c["name"] for c in taco["categories"]}
    anns = {}
    for a in taco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    def job(meta):
        iid = meta["id"]
        fn = f"taco_{iid:04d}.jpg"
        dst, lbl = IMG_DIR / fn, LBL_DIR / f"taco_{iid:04d}.txt"
        W, H = meta["width"], meta["height"]
        rec = {"filename": fn, "source": "taco", "source_image_id": iid, "source_file": meta["file_name"],
               "url": meta.get("flickr_640_url") or meta.get("flickr_url"), "status": "", "num_boxes": 0,
               "num_ambiguous": 0}
        boxes, amb = [], []
        for a in anns.get(iid, []):
            yb = coco_to_yolo(a["bbox"], W, H)
            if yb is None:
                continue
            m = cmap.get(a["category_id"])
            if m and m.get("class_id") is not None and m.get("status") == "CONFIRMED":
                boxes.append((int(m["class_id"]), *yb))
            else:
                amb.append({"annotation_id": a["id"], "category_id": a["category_id"],
                            "category_name": cat_names.get(a["category_id"]), "yolo_bbox": [round(v, 6) for v in yb]})
        existed = dst.exists()
        if not existed:
            data = None
            for url in (meta.get("flickr_640_url"), meta.get("flickr_url")):
                if url:
                    data = fetch(url)
                    if data:
                        rec["url"] = url
                        break
            if data is None:
                rec["status"] = "download_failed"
                return rec, amb
            try:
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001
                rec["status"] = "decode_failed"
                return rec, amb
            if abs(w / h - W / H) > 0.03 * (W / H):
                # Transposed / cropped rendition: labels would be misaligned -> do not keep.
                rec["status"] = f"orientation_mismatch({w}x{h} vs ann {W}x{H})"
                return rec, amb
            save_image(data, dst)
        else:
            with Image.open(dst) as im:
                w, h = im.size
            if abs(w / h - W / H) > 0.03 * (W / H):
                rec["status"] = f"existing_orientation_mismatch({w}x{h} vs ann {W}x{H})"
                return rec, amb
        if lbl.exists():
            rec["status"] = "existing_kept" if existed else "label_existing_kept"
            rec["num_boxes"] = sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip())
        else:
            write_label(lbl, boxes)
            rec["status"] = "new"
            rec["num_boxes"] = len(boxes)
        rec["num_ambiguous"] = len(amb)
        return rec, amb

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, m) for m in taco["images"]]
        for i, f in enumerate(as_completed(futs), 1):
            rec, amb = f.result()
            records.append(rec)
            if amb and not rec["status"].startswith(("download", "decode", "orientation", "existing_orient")):
                ambiguous[rec["filename"]] = amb
            if i % 100 == 0:
                print(f"[TACO] {i}/{len(futs)} ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- OpenImages
def collect_openimages(cfg, workers: int, cap: int, seed: int, records: list):
    import pandas as pd

    oi = cfg["sources"]["openimages_v7"]
    mapped = {k: int(v["class_id"]) for k, v in oi["class_mapping"].items() if v.get("class_id") is not None}
    df = pd.read_csv(OI_BOXES)
    human_imgs = set(df.loc[df.LabelName.isin(HUMAN_LABELS), "ImageID"])
    t = df[df.LabelName.isin(mapped)].copy()
    t["cls"] = t.LabelName.map(mapped)
    bad = set(t.loc[(t.IsDepiction == 1) | (t.IsGroupOf == 1), "ImageID"])
    t = t[~t.ImageID.isin(human_imgs | bad)]
    existing = {p.stem[3:] for p in IMG_DIR.glob("oi_*.jpg")}

    # Per-class pools, rarest class first, so an image is counted for its rarest class.
    rng = random.Random(seed)
    counts = t.groupby("cls").ImageID.nunique().sort_values()
    chosen: dict[str, int] = {}
    for cls in counts.index:
        ids = sorted(set(t.loc[t.cls == cls, "ImageID"]) - set(chosen))
        rng.shuffle(ids)
        already = sum(1 for i, c in chosen.items() if c == cls)
        for iid in ids[: max(0, cap - already)]:
            chosen[iid] = int(cls)
    print(f"[OI] pool after filters: {t.ImageID.nunique()} images; selected {len(chosen)}; "
          f"per stratum {dict(sorted(__import__('collections').Counter(chosen.values()).items()))}")
    tmpl = oi["s3_image_url_template"]
    boxes_by_img = {iid: g for iid, g in t[t.ImageID.isin(chosen)].groupby("ImageID")}

    def job(iid):
        fn = f"oi_{iid}.jpg"
        dst, lbl = IMG_DIR / fn, LBL_DIR / f"oi_{iid}.txt"
        rec = {"filename": fn, "source": "openimages_v7", "source_image_id": iid, "source_file": "validation",
               "url": tmpl.format(image_id=iid), "status": "", "num_boxes": 0, "num_ambiguous": 0}
        if iid in existing or dst.exists():
            rec["status"] = "existing_kept"
            return rec
        data = fetch(rec["url"])
        if data is None:
            rec["status"] = "download_failed"
            return rec
        try:
            save_image(data, dst)
        except Exception:  # noqa: BLE001
            rec["status"] = "decode_failed"
            return rec
        boxes = []
        for _, r in boxes_by_img[iid].iterrows():
            x1, x2 = max(0.0, r.XMin), min(1.0, r.XMax)
            y1, y2 = max(0.0, r.YMin), min(1.0, r.YMax)
            if x2 - x1 > 0.002 and y2 - y1 > 0.002:
                boxes.append((int(r.cls), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
        if not lbl.exists():
            write_label(lbl, boxes)
        rec["status"] = "new"
        rec["num_boxes"] = len(boxes)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(job, iid) for iid in chosen]), 1):
            records.append(f.result())
            if i % 100 == 0:
                print(f"[OI] {i}/{len(chosen)}", flush=True)


def recover_taco_orientation(cfg, workers: int, records: list, ambiguous: dict):
    """For TACO images rejected as orientation_mismatch (640px rendition is EXIF-stripped/transposed):
    download the original (flickr_url, carries EXIF), apply EXIF transpose, accept only if the result matches the
    annotation width/height aspect, then save an upright JPEG (max side 1280, q=92) with NO EXIF orientation tag,
    so loaders that ignore EXIF (YoloBoxDataset) and those that apply it (Ultralytics) see the same pixels."""
    from PIL import ImageOps

    prev = {}
    if MANIFEST_OUT.exists():
        with MANIFEST_OUT.open(encoding="utf-8") as f:
            prev = {r["filename"]: r for r in csv.DictReader(f)}
    targets = {int(r["source_image_id"]) for r in prev.values()
               if r["source"] == "taco" and r["status"].startswith("orientation_mismatch")}
    taco = json.loads(TACO_ANN.read_text(encoding="utf-8"))
    cmap = {int(k): v for k, v in cfg["sources"]["taco"]["category_mapping"].items()}
    cat_names = {c["id"]: c["name"] for c in taco["categories"]}
    anns = {}
    for a in taco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    metas = [m for m in taco["images"] if m["id"] in targets]

    def job(meta):
        iid = meta["id"]
        fn = f"taco_{iid:04d}.jpg"
        dst, lbl = IMG_DIR / fn, LBL_DIR / f"taco_{iid:04d}.txt"
        W, H = meta["width"], meta["height"]
        rec = {"filename": fn, "source": "taco", "source_image_id": iid, "source_file": meta["file_name"],
               "url": meta.get("flickr_url"), "status": "", "num_boxes": 0, "num_ambiguous": 0}
        if dst.exists() or lbl.exists():
            rec["status"] = "existing_kept"
            return rec, []
        data = fetch(meta["flickr_url"], timeout=90) if meta.get("flickr_url") else None
        if data is None:
            rec["status"] = "orientation_mismatch_recover_download_failed"
            return rec, []
        with Image.open(io.BytesIO(data)) as im:
            up = ImageOps.exif_transpose(im).convert("RGB")
        w, h = up.size
        if abs(w / h - W / H) > 0.03 * (W / H):
            rec["status"] = f"orientation_mismatch_unrecoverable({w}x{h} vs ann {W}x{H})"
            return rec, []
        up.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        up.save(dst, "JPEG", quality=92)
        boxes, amb = [], []
        for a in anns.get(iid, []):
            yb = coco_to_yolo(a["bbox"], W, H)
            if yb is None:
                continue
            m = cmap.get(a["category_id"])
            if m and m.get("class_id") is not None and m.get("status") == "CONFIRMED":
                boxes.append((int(m["class_id"]), *yb))
            else:
                amb.append({"annotation_id": a["id"], "category_id": a["category_id"],
                            "category_name": cat_names.get(a["category_id"]), "yolo_bbox": [round(v, 6) for v in yb]})
        write_label(lbl, boxes)
        rec.update(status="new_recovered_exif_transposed", num_boxes=len(boxes), num_ambiguous=len(amb))
        return rec, amb

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(job, m) for m in metas]), 1):
            rec, amb = f.result()
            records.append(rec)
            if amb:
                ambiguous[rec["filename"]] = amb
            if i % 20 == 0:
                print(f"[TACO-recover] {i}/{len(metas)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--oi-cap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-taco", action="store_true")
    ap.add_argument("--skip-oi", action="store_true")
    ap.add_argument("--recover-orientation", action="store_true",
                    help="only run the TACO orientation recovery pass (original Flickr image + EXIF transpose)")
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs" / "detection_source_mapping.yaml").read_text(encoding="utf-8"))
    records, ambiguous = [], {}
    if args.recover_orientation:
        args.skip_taco = args.skip_oi = True
        if AMBIG_OUT.exists():
            ambiguous = json.loads(AMBIG_OUT.read_text(encoding="utf-8"))
        recover_taco_orientation(cfg, args.workers, records, ambiguous)
        AMBIG_OUT.write_text(json.dumps(ambiguous, indent=1), encoding="utf-8")
    if not args.skip_taco:
        collect_taco(cfg, args.workers, records, ambiguous)
        AMBIG_OUT.write_text(json.dumps(ambiguous, indent=1), encoding="utf-8")
    if not args.skip_oi:
        collect_openimages(cfg, args.workers, args.oi_cap, args.seed, records)
    old = []
    if MANIFEST_OUT.exists():
        with MANIFEST_OUT.open(encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f) if r["filename"] not in {x["filename"] for x in records}]
    fields = ["filename", "source", "source_image_id", "source_file", "url", "status", "num_boxes", "num_ambiguous"]
    with MANIFEST_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(old + records, key=lambda r: r["filename"]):
            w.writerow({k: r.get(k, "") for k in fields})
    from collections import Counter
    print("status:", dict(Counter((r["source"], r["status"].split("(")[0]) for r in records)))


if __name__ == "__main__":
    sys.exit(main())
