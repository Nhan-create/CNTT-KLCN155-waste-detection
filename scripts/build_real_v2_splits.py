"""scripts/build_real_v2_splits.py
---------------------------------
Build the real-only (no Mendeley/synthetic) detection split v2 + class-agnostic Stage-1 labels.

Outputs (all new files, existing splits/configs untouched):
  data/detection/splits_real_v2/{train,val,test}.txt        lines: images/real/<file>   (dataset-root relative;
                                                            consumed by src/detection YoloBoxDataset/split_images)
  data/detection/splits_real_v2/{train,val,test}_ultra.txt  lines: ./../images/real/<file> (Ultralytics resolves "./"
                                                            relative to the txt file; bare relative lines are CWD-relative)
  data/detection/quarantine_real_v2.csv                     excluded images + reason (raw files are NOT deleted)
  data/detection/real_v2_split_manifest.csv                 per-image split, group, phash, sha256, stratum
  data/detection/labels_agnostic/real/*.txt                 1-class ('waste'=0) labels
  data/detection_agnostic/{images/real (hardlinks), labels/real, splits_real_v2}
  configs/detection_dataset_real_v2.yaml                    10-class (SSDLite loader + Ultralytics-compatible layout)
  configs/detection_dataset_real_v2_ultralytics.yaml        10-class, *_ultra.txt lists
  configs/detection_dataset_real_v2_agnostic.yaml           1-class, Ultralytics

Usage: python scripts/build_real_v2_splits.py [--seed 42] [--phash-thr 8]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parent.parent
DET = ROOT / "data" / "detection"
AGN = ROOT / "data" / "detection_agnostic"
IMG_DIR, LBL_DIR = DET / "images" / "real", DET / "labels" / "real"
SPLIT_DIR = DET / "splits_real_v2"
NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
HOLDOUT_MANIFEST = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"
OI_BOXES = (Path(__import__("os").environ.get("WASTE_TRAINING_DIR", str(Path(__file__).resolve().parents[1] / "data" / "external" / "waste-training"))) / "openimages-validation-boxes.csv")
HUMAN_LABELS = {
    "/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655", "/m/02p0tk3", "/m/035r7c",
    "/m/031n1", "/m/0dzf4", "/m/0k65p", "/m/04hgtk", "/m/0dzct", "/m/03q69",
}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(p: Path) -> int:
    with Image.open(p) as im:
        g = ImageOps.exif_transpose(im).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    d = dctn(np.asarray(g, dtype=np.float64), norm="ortho")[:8, :8].flatten()
    med = np.median(d[1:])
    bits = d > med
    return int("".join("1" if b else "0" for b in bits), 2)


def read_boxes(p: Path):
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            out.append((int(float(parts[0])), *map(float, parts[1:5])))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def validate(boxes):
    for c, xc, yc, w, h in boxes:
        if not 0 <= c <= 9:
            return f"class_out_of_range({c})"
        if w <= 0 or h <= 0 or xc - w / 2 < -0.01 or yc - h / 2 < -0.01 or xc + w / 2 > 1.01 or yc + h / 2 > 1.01:
            return "invalid_geometry"
        if abs(xc - 0.5) < 1e-6 and abs(yc - 0.5) < 1e-6 and abs(w - 0.4) < 1e-6 and abs(h - 0.4) < 1e-6:
            return "placeholder_box_0.5_0.5_0.4_0.4"
    return None


def load_exclusion_sources():
    review = {}
    man = DET / "manifest_detection_v1.csv"
    if man.exists():
        with man.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                review[r["filename"]] = r.get("review_status", "")
    holdout = {f["sha256"] for f in json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))["files"]}
    collect_status = {}
    cm = DET / "real_v2_source_manifest.csv"
    if cm.exists():
        with cm.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                collect_status[r["filename"]] = r["status"]
    oi_human, oi_bad = set(), set()
    if OI_BOXES.exists():
        import pandas as pd

        df = pd.read_csv(OI_BOXES, usecols=["ImageID", "LabelName", "IsGroupOf", "IsDepiction"])
        oi_human = set(df.loc[df.LabelName.isin(HUMAN_LABELS), "ImageID"])
        mapping = yaml.safe_load((ROOT / "configs" / "detection_source_mapping.yaml").read_text(encoding="utf-8"))
        mapped = {k for k, v in mapping["sources"]["openimages_v7"]["class_mapping"].items() if v.get("class_id") is not None}
        sub = df[df.LabelName.isin(mapped)]
        oi_bad = set(sub.loc[(sub.IsGroupOf == 1) | (sub.IsDepiction == 1), "ImageID"])
    return review, holdout, collect_status, oi_human, oi_bad


def union_find_groups(names, hashes, thr):
    n = len(names)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    arr = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(n):
        x = np.bitwise_xor(arr[i + 1:], arr[i])
        dist = np.array([bin(int(v)).count("1") for v in x]) if len(x) else np.array([])
        for j in np.nonzero(dist <= thr)[0]:
            a, b = find(i), find(i + 1 + int(j))
            pairs.append((names[i], names[i + 1 + int(j)], int(dist[j])))
            if a != b:
                parent[b] = a
    return [find(i) for i in range(n)], pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phash-thr", type=int, default=8)
    args = ap.parse_args()
    review, holdout, collect_status, oi_human, oi_bad = load_exclusion_sources()

    kept, quarantine, seen_sha = {}, [], {}
    for img in sorted(IMG_DIR.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        fn = img.name
        lbl = LBL_DIR / f"{img.stem}.txt"
        reason = None
        if not (fn.startswith("taco_") or fn.startswith("oi_")):
            reason = "unknown_source_prefix"
        elif not lbl.exists():
            reason = "missing_label"
        elif review.get(fn) in {"REJECTED", "NEEDS_RELABEL"}:
            reason = f"review_status_{review[fn]}"
        elif "orientation_mismatch" in collect_status.get(fn, ""):
            reason = "taco_orientation_mismatch"
        elif fn.startswith("oi_") and img.stem[3:] in oi_human:
            reason = "oi_human_present_in_use_object"
        elif fn.startswith("oi_") and img.stem[3:] in oi_bad:
            reason = "oi_groupof_or_depiction_box"
        digest = sha256(img)
        boxes = []
        if reason is None:
            if digest in holdout:
                reason = "holdout_sha256_match"
            elif digest in seen_sha:
                reason = f"exact_duplicate_of_{seen_sha[digest]}"
        if reason is None:
            boxes = read_boxes(lbl)
            reason = "empty_label" if not boxes else validate(boxes)
        if reason is None:
            with Image.open(img) as im:
                orient = im.getexif().get(274, 1)
            if orient not in (0, 1):
                # TACO boxes are in the EXIF-applied frame; YoloBoxDataset ignores EXIF -> misaligned.
                reason = f"exif_orientation_{orient}"
        if reason:
            quarantine.append({"filename": fn, "reason": reason, "sha256": digest})
            continue
        seen_sha[digest] = fn
        kept[fn] = {"boxes": boxes, "sha256": digest, "source": "taco" if fn.startswith("taco_") else "openimages_v7"}

    names = sorted(kept)
    hashes = [phash(IMG_DIR / n) for n in names]
    roots, pairs = union_find_groups(names, hashes, args.phash_thr)
    for n, h, r in zip(names, hashes, roots):
        kept[n]["phash"] = f"{h:016x}"
        kept[n]["group"] = f"g{r:05d}"

    # Stratum = source + rarest class present (global image frequency), so rare classes reach val/test.
    img_freq = Counter(c for n in names for c in {b[0] for b in kept[n]["boxes"]})
    for n in names:
        present = {b[0] for b in kept[n]["boxes"]}
        rarest = min(present, key=lambda c: (img_freq[c], c))
        kept[n]["stratum"] = f"{kept[n]['source']}|{NAMES[rarest]}"

    groups = defaultdict(list)
    for n in names:
        groups[kept[n]["group"]].append(n)
    strata = defaultdict(list)
    for g, members in groups.items():
        # Group stratum = rarest stratum among members.
        st = min((kept[m]["stratum"] for m in members), key=lambda s: (img_freq[NAMES.index(s.split("|")[1])], s))
        strata[st].append(g)

    rng = random.Random(args.seed)
    assignment = {}
    for st in sorted(strata):
        gs = sorted(strata[st])
        rng.shuffle(gs)
        n = len(gs)
        if n == 1:
            n_val, n_test = 0, 0
        elif n == 2:
            n_val, n_test = 0, 1
        else:
            n_val = max(1, round(n * RATIOS["val"]))
            n_test = max(1, round(n * RATIOS["test"]))
        for i, g in enumerate(gs):
            split = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
            for m in groups[g]:
                assignment[m] = split

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for split in RATIOS:
        members = sorted(n for n in names if assignment[n] == split)
        p = SPLIT_DIR / f"{split}.txt"
        p.write_text("".join(f"images/real/{m}\n" for m in members), encoding="utf-8", newline="\n")
        (SPLIT_DIR / f"{split}_ultra.txt").write_text("".join(f"./../images/real/{m}\n" for m in members),
                                                      encoding="utf-8", newline="\n")
        split_files[split] = members

    with (DET / "quarantine_real_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "reason", "sha256"])
        w.writeheader()
        w.writerows(quarantine)
    with (DET / "real_v2_split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "source", "group", "stratum", "phash", "sha256", "num_boxes", "classes"])
        for n in names:
            k = kept[n]
            w.writerow([n, assignment[n], k["source"], k["group"], k["stratum"], k["phash"], k["sha256"],
                        len(k["boxes"]), " ".join(sorted({NAMES[b[0]] for b in k["boxes"]}))])

    # ---------------------------------------------------------------- stats
    stats = {}
    for split, members in split_files.items():
        box_c, img_c, src_c = Counter(), Counter(), Counter()
        for m in members:
            src_c[kept[m]["source"]] += 1
            for b in kept[m]["boxes"]:
                box_c[NAMES[b[0]]] += 1
            for c in {b[0] for b in kept[m]["boxes"]}:
                img_c[NAMES[c]] += 1
        stats[split] = {"images": len(members), "boxes": sum(box_c.values()),
                        "boxes_per_class": {c: box_c.get(c, 0) for c in NAMES},
                        "images_per_class": {c: img_c.get(c, 0) for c in NAMES}, "images_per_source": dict(src_c)}
    fp = {s: sha256(SPLIT_DIR / f"{s}.txt") for s in RATIOS}
    fp_ultra = {s: sha256(SPLIT_DIR / f"{s}_ultra.txt") for s in RATIOS}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    near_dup_cross = [p for p in pairs if assignment[p[0]] != assignment[p[1]]]

    def dataset_yaml(train, val, test, extra_note, fps):
        return {
            "path": "../data/detection", "train": train, "val": val, "test": test, "nc": 10,
            "names": {i: n for i, n in enumerate(NAMES)},
            "metadata": {
                "version": "v2.0-real-only-no-mendeley", "created_utc": now, "seed": args.seed,
                "generator": "scripts/build_real_v2_splits.py", "sources": ["taco", "openimages_v7"],
                "synthetic_or_mendeley_images": 0, "phash_near_dup_threshold": args.phash_thr,
                "stratification": "source + rarest class present; pHash near-duplicate groups kept in one split",
                "total_images": len(names), "train_images": stats["train"]["images"],
                "val_images": stats["val"]["images"], "test_images": stats["test"]["images"],
                "train_boxes": stats["train"]["boxes"], "val_boxes": stats["val"]["boxes"],
                "test_boxes": stats["test"]["boxes"], "quarantined_images": len(quarantine),
                "split_sha256": fps, "note": extra_note,
            },
        }

    cfg_dir = ROOT / "configs"
    with (cfg_dir / "detection_dataset_real_v2.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml("splits_real_v2/train.txt", "splits_real_v2/val.txt", "splits_real_v2/test.txt",
                                    "Lines are dataset-root relative (images/real/...): used by src/detection split_images/"
                                    "YoloBoxDataset. For Ultralytics use detection_dataset_real_v2_ultralytics.yaml.", fp),
                       f, sort_keys=False, allow_unicode=True)
    with (cfg_dir / "detection_dataset_real_v2_ultralytics.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml("splits_real_v2/train_ultra.txt", "splits_real_v2/val_ultra.txt",
                                    "splits_real_v2/test_ultra.txt",
                                    "Same image sets as detection_dataset_real_v2.yaml; lines start with './' so Ultralytics "
                                    "resolves them relative to the list file.", fp_ultra), f, sort_keys=False, allow_unicode=True)

    # ---------------------------------------------------------------- class-agnostic labels
    amb_path = DET / "real_v2_ambiguous_boxes.json"
    ambiguous = json.loads(amb_path.read_text(encoding="utf-8")) if amb_path.exists() else {}
    agn_lbl = DET / "labels_agnostic" / "real"
    agn_lbl.mkdir(parents=True, exist_ok=True)
    (AGN / "images" / "real").mkdir(parents=True, exist_ok=True)
    (AGN / "labels" / "real").mkdir(parents=True, exist_ok=True)
    (AGN / "splits_real_v2").mkdir(parents=True, exist_ok=True)
    added_amb = 0
    agn_boxes = Counter()
    for n in names:
        bx = [b[1:] for b in kept[n]["boxes"]]
        for a in ambiguous.get(n, []):
            cand = tuple(a["yolo_bbox"])
            if all(iou(cand, e) < 0.7 for e in bx):
                bx.append(cand)
                added_amb += 1
        text = "".join(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for xc, yc, w, h in bx)
        stem = Path(n).stem
        (agn_lbl / f"{stem}.txt").write_text(text, encoding="utf-8")
        (AGN / "labels" / "real" / f"{stem}.txt").write_text(text, encoding="utf-8")
        agn_boxes[assignment[n]] += len(bx)
        dst = AGN / "images" / "real" / n
        if not dst.exists():
            try:
                os.link(IMG_DIR / n, dst)
            except OSError:
                shutil.copy2(IMG_DIR / n, dst)
    for f in SPLIT_DIR.glob("*.txt"):
        shutil.copy2(f, AGN / "splits_real_v2" / f.name)
    agn_yaml = {
        "path": "../data/detection_agnostic", "train": "splits_real_v2/train_ultra.txt",
        "val": "splits_real_v2/val_ultra.txt", "test": "splits_real_v2/test_ultra.txt", "nc": 1, "names": {0: "waste"},
        "metadata": {
            "version": "v2.0-real-only-agnostic", "created_utc": now, "seed": args.seed,
            "generator": "scripts/build_real_v2_splits.py", "same_images_as": "configs/detection_dataset_real_v2.yaml",
            "train_images": stats["train"]["images"], "val_images": stats["val"]["images"],
            "test_images": stats["test"]["images"], "boxes_per_split": dict(agn_boxes),
            "taco_ambiguous_boxes_added": added_amb,
            "label_rule": "every 10-class box -> 0, plus TACO REQUIRES_REVIEW/unmapped boxes (IoU<0.7 with existing)",
            "layout": "images/real are hardlinks to data/detection/images/real; labels/real mirror "
                      "data/detection/labels_agnostic/real (Ultralytics maps /images/ -> /labels/)",
            "relative_lists": "splits_real_v2/{train,val,test}.txt (images/real/..) for root-relative loaders",
            "ssdlite_note": "src/detection/dataset.py load_detection_dataset_yaml enforces nc=10 + 10 names; "
                            "SSDLite stage-1 needs a loader change (Agent B scope).",
            "split_sha256": {s: sha256(AGN / "splits_real_v2" / f"{s}_ultra.txt") for s in RATIOS},
        },
    }
    with (cfg_dir / "detection_dataset_real_v2_agnostic.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(agn_yaml, f, sort_keys=False, allow_unicode=True)

    summary = {"stats": stats, "quarantine_reasons": dict(Counter(q["reason"].split("_of_")[0] for q in quarantine)),
               "near_dup_pairs": len(pairs), "near_dup_pairs_cross_split": len(near_dup_cross),
               "agnostic_boxes": dict(agn_boxes), "taco_ambiguous_boxes_added": added_amb, "split_sha256": fp}
    out = ROOT / "reports" / "data_v2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_real_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "near_duplicate_pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
