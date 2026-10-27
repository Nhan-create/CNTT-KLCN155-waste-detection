# Mô tả: Kiểm thử tính toàn vẹn, không rò rỉ SHA-256/pHash và phân bố lớp của bộ chia tập phát hiện thực tế splits_real_v2.
"""Regression tests for the real-only detection split v2 (Agent A, data v2).

Guarantees:
- no image leakage between train/val/test (filename, SHA-256, and pHash near-duplicate groups)
- no synthetic / Mendeley image in any split
- every image exists and has a valid YOLO label (class ids 0..9, normalized geometry, non-empty)
- no image matches the acceptance holdout data/holdout (SHA-256 vs reports/qa_acceptance manifest)
- dataset YAML fingerprints match the split files; class order unchanged
- class-agnostic labels mirror the split images with class 0 only
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "data" / "detection"
SPLIT_DIR = DET / "splits_real_v2"
SPLITS = ("train", "val", "test")
NAMES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
CFG = ROOT / "configs" / "detection_dataset_real_v2.yaml"
CFG_ULTRA = ROOT / "configs" / "detection_dataset_real_v2_ultralytics.yaml"
CFG_AGN = ROOT / "configs" / "detection_dataset_real_v2_agnostic.yaml"
HOLDOUT = ROOT / "reports" / "qa_acceptance" / "manifest_hinh_anh_rac.json"

pytestmark = pytest.mark.skipif(not (SPLIT_DIR / "train.txt").exists(), reason="split v2 not built")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def splits() -> dict[str, list[str]]:
    return {s: [ln.strip() for ln in (SPLIT_DIR / f"{s}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
            for s in SPLITS}


@pytest.fixture(scope="module")
def image_sha(splits) -> dict[str, str]:
    return {rel: _sha(DET / rel) for s in SPLITS for rel in splits[s]}


def test_splits_non_empty_and_ratio(splits):
    n = {s: len(v) for s, v in splits.items()}
    total = sum(n.values())
    assert min(n.values()) > 0
    assert 0.60 <= n["train"] / total <= 0.80
    assert 0.10 <= n["val"] / total <= 0.20
    assert 0.10 <= n["test"] / total <= 0.20


def test_no_duplicates_within_split(splits):
    for s, items in splits.items():
        assert len(items) == len(set(items)), s


def test_no_filename_leakage(splits):
    sets = {s: set(v) for s, v in splits.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_no_sha256_leakage(splits, image_sha):
    owner = {}
    for s in SPLITS:
        for rel in splits[s]:
            h = image_sha[rel]
            assert owner.get(h, s) == s, f"{rel} content also in split {owner[h]}"
            owner[h] = s


def test_no_near_duplicate_group_leakage(splits):
    rows = list(csv.DictReader((DET / "real_v2_split_manifest.csv").open(encoding="utf-8")))
    split_of = {}
    for s in SPLITS:
        for rel in splits[s]:
            split_of[Path(rel).name] = s
    group_splits: dict[str, set[str]] = {}
    for r in rows:
        assert split_of[r["filename"]] == r["split"]
        group_splits.setdefault(r["group"], set()).add(r["split"])
    assert all(len(v) == 1 for v in group_splits.values())


def test_no_near_duplicate_pairs_across_splits(splits):
    pairs = json.loads((ROOT / "reports" / "data_v2" / "near_duplicate_pairs.json").read_text(encoding="utf-8"))
    split_of = {Path(rel).name: s for s in SPLITS for rel in splits[s]}
    for a, b, _ in pairs:
        assert split_of[a] == split_of[b], (a, b)


def test_no_synthetic_or_mendeley(splits):
    for s in SPLITS:
        for rel in splits[s]:
            low = rel.lower()
            assert low.startswith("images/real/"), rel
            assert "synthetic" not in low and "mendeley" not in low and "syn_" not in low, rel
            assert Path(rel).name.startswith(("taco_", "oi_")), rel


def test_every_image_has_valid_label(splits):
    for s in SPLITS:
        for rel in splits[s]:
            img = DET / rel
            assert img.is_file(), rel
            lbl = DET / "labels" / "real" / (img.stem + ".txt")
            assert lbl.is_file(), rel
            lines = [ln.split() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert lines, f"empty label {lbl}"
            for parts in lines:
                assert len(parts) == 5
                c = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                assert 0 <= c <= 9
                assert w > 0 and h > 0
                assert xc - w / 2 >= -0.01 and yc - h / 2 >= -0.01
                assert xc + w / 2 <= 1.01 and yc + h / 2 <= 1.01
                assert not (abs(xc - 0.5) < 1e-6 and abs(yc - 0.5) < 1e-6 and abs(w - 0.4) < 1e-6
                            and abs(h - 0.4) < 1e-6), f"placeholder box in {lbl}"


def test_no_holdout_images(image_sha):
    files = json.loads(HOLDOUT.read_text(encoding="utf-8"))["files"]
    holdout = {f["sha256"] for f in files}
    # Compare against the unique SHA-256 set (manifest lists 27 files, 2 are byte-identical -> 26 hashes).
    assert holdout
    assert not holdout & set(image_sha.values())
    for rel in image_sha:
        assert "hinh_anh_rac" not in rel.lower()


def test_quarantined_images_not_in_splits(splits):
    q = {r["filename"] for r in csv.DictReader((DET / "quarantine_real_v2.csv").open(encoding="utf-8"))}
    used = {Path(rel).name for s in SPLITS for rel in splits[s]}
    assert not q & used


def test_yaml_configs_and_fingerprints():
    for cfg_path, suffix in ((CFG, ""), (CFG_ULTRA, "_ultra")):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert cfg["path"] == "../data/detection"
        assert cfg["nc"] == 10
        assert [cfg["names"][i] for i in range(10)] == NAMES
        for s in SPLITS:
            assert cfg[s] == f"splits_real_v2/{s}{suffix}.txt"
            assert cfg["metadata"]["split_sha256"][s] == _sha(SPLIT_DIR / f"{s}{suffix}.txt")
        assert cfg["metadata"]["synthetic_or_mendeley_images"] == 0


def test_ultra_lists_match_canonical(splits):
    for s in SPLITS:
        ultra = [ln.strip() for ln in (SPLIT_DIR / f"{s}_ultra.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert [u.replace("./../", "", 1) for u in ultra] == splits[s]


def test_project_loader_reads_v2(splits):
    from src.detection.dataset import _label_path
    from src.detection.training_common import split_images

    for s in SPLITS:
        images, root = split_images(CFG, s)
        assert len(images) == len(splits[s])
        for img in images[:20]:
            assert _label_path(img, root).is_file()


def test_agnostic_labels(splits):
    cfg = yaml.safe_load(CFG_AGN.read_text(encoding="utf-8"))
    assert cfg["nc"] == 1 and cfg["names"] == {0: "waste"}
    agn_root = (CFG_AGN.parent / cfg["path"]).resolve()
    for s in SPLITS:
        lines = [ln.strip() for ln in (agn_root / cfg[s]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert [ln.replace("./../", "", 1) for ln in lines] == splits[s]
        for rel in splits[s]:
            stem = Path(rel).stem
            assert (agn_root / rel).is_file()
            mirror = agn_root / "labels" / "real" / f"{stem}.txt"
            src = DET / "labels_agnostic" / "real" / f"{stem}.txt"
            assert mirror.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
            agn = [ln.split() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
            ten = [ln for ln in (DET / "labels" / "real" / f"{stem}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
            assert all(p[0] == "0" for p in agn)
            assert len(agn) >= len(ten)
