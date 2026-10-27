"""
Script to download and verify Mini-Batch 01 (10 Wikimedia images).
Enforces:
- Strict 50 MiB cumulative download limit.
- Official Wikimedia User-Agent.
- Stream downloading with byte counting.
- SHA-256 verification and pHash computation.
- Leakage/duplicate detection against authoritative_hashes_cache.json.
- Preservation of original image bytes.
"""

import sys
import json
import hashlib
from pathlib import Path
import urllib.request
import urllib.parse
from PIL import Image
import imagehash

USER_AGENT = "WasteDetectionResearchBot/1.0 (academic research; contact: student_research@university.edu)"
MAX_TOTAL_BYTES = 50 * 1024 * 1024  # 50 MiB limit

TARGET_CANDIDATE_IDS = [
    "wiki_bat_corroded_screwdriver",
    "wiki_bio_apple_core_01",
    "wiki_bio_banana_peel_bucharest_02",
    "wiki_bio_banana_peel_sidewalk_06",
    "wiki_bio_banana_peel_bus_stop_03",
    "wiki_bio_banana_peel_oak_hill_05",
    "wiki_bio_food_scraps_bin_09",
    "wiki_neg_utility_box_corner_05",
    "wiki_neg_brick_wall_clean",
    "wiki_multi_banana_and_cup_01"
]

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(__file__).resolve().parents[3]
    manifest_path = root / "data/acquisition_staging/manifests/batch_a_candidate_manifest.json"
    cache_path = root / "data/audit/authoritative_hashes_cache.json"
    
    out_dir = root / "data/acquisition_staging/batch_a_mini01"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    candidates_by_id = {c["candidate_id"]: c for c in manifest_data["candidates"]}
    
    # Load authoritative hash cache for leakage detection
    known_hashes = {}
    known_phashes = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            for k, v in cache_data.items():
                if "sha256" in v:
                    known_hashes[v["sha256"]] = k
                if "phash" in v and v["phash"]:
                    try:
                        known_phashes[k] = imagehash.hex_to_hash(v["phash"])
                    except Exception:
                        pass
        print(f"Loaded {len(known_hashes)} known SHA-256 and {len(known_phashes)} known pHashes from cache.")
    
    headers = {"User-Agent": USER_AGENT}
    total_bytes_downloaded = 0
    results = []
    
    print(f"\n--- STARTING MINI-BATCH 01 DOWNLOAD ({len(TARGET_CANDIDATE_IDS)} images, limit {MAX_TOTAL_BYTES / (1024*1024)} MiB) ---")
    
    for cid in TARGET_CANDIDATE_IDS:
        if cid not in candidates_by_id:
            print(f"ERROR: Candidate {cid} not found in manifest!")
            continue
        c = candidates_by_id[cid]
        url = c["direct_download_url"]
        exp_bytes = c.get("file_size_bytes", 0)
        
        # Determine local filename
        # Use candidate_id + original extension from URL
        ext = Path(url.split("?")[0]).suffix.lower()
        if not ext or ext not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"
        dest_filename = f"{cid}{ext}"
        dest_path = images_dir / dest_filename
        
        print(f"\nFetching: {cid} ({exp_bytes:,} bytes expected)")
        print(f"  URL: {url}")
        
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                status_code = resp.status
                if status_code != 200:
                    print(f"  FAILED: HTTP {status_code}")
                    results.append({
                        "candidate_id": cid,
                        "status": f"HTTP_ERROR_{status_code}",
                        "dest_file": None
                    })
                    continue
                bytes_for_this_file = 0
                with open(dest_path, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        total_bytes_downloaded += len(chunk)
                        bytes_for_this_file += len(chunk)
                        if total_bytes_downloaded > MAX_TOTAL_BYTES:
                            raise RuntimeError(f"Exceeded hard limit of {MAX_TOTAL_BYTES} bytes! Aborting.")
                        f.write(chunk)
        except Exception as e:
            print(f"  NETWORK/HTTP ERROR: {e}")
            results.append({
                "candidate_id": cid,
                "status": f"NETWORK_ERROR: {e}",
                "dest_file": None
            })
            continue
                    
        print(f"  Downloaded: {bytes_for_this_file:,} bytes. Cumulative: {total_bytes_downloaded:,} / {MAX_TOTAL_BYTES:,} bytes ({total_bytes_downloaded / (1024*1024):.2f} MiB)")
        
        # Verify decode
        try:
            with Image.open(dest_path) as img:
                w, h = img.size
                fmt = img.format
                mode = img.mode
                # compute phash
                ph = imagehash.phash(img)
            decode_ok = True
        except Exception as e:
            decode_ok = False
            w, h, fmt, mode, ph = 0, 0, None, None, None
            print(f"  DECODE ERROR: {e}")
            
        file_sha256 = compute_sha256(dest_path)
        
        # Check leakage against known dataset
        sha_leak = known_hashes.get(file_sha256)
        phash_matches = []
        if ph is not None:
            for k, known_ph in known_phashes.items():
                dist = ph - known_ph
                if dist <= 5:  # High similarity threshold
                    phash_matches.append({"match_key": k, "hamming_distance": dist})
                    
        rec = {
            "candidate_id": cid,
            "filename": dest_filename,
            "file_path": str(dest_path.relative_to(root)),
            "category": c.get("category"),
            "target_class": c.get("target_class"),
            "camera_domain": c.get("camera_domain"),
            "expected_bytes": exp_bytes,
            "downloaded_bytes": bytes_for_this_file,
            "bytes_match": bytes_for_this_file == exp_bytes,
            "decode_ok": decode_ok,
            "resolution": [w, h],
            "format": fmt,
            "color_mode": mode,
            "sha256": file_sha256,
            "phash": str(ph) if ph else None,
            "sha256_leak_detected": sha_leak,
            "near_duplicates_dataset": phash_matches,
            "license": c.get("license"),
            "author": c.get("author")
        }
        results.append(rec)
        print(f"  Resolution: {w}x{h}, Mode: {mode}, SHA-256: {file_sha256[:16]}..., pHash: {ph}")
        print(f"  Leakage SHA match: {sha_leak if sha_leak else 'NONE (CLEAN)'}")
        print(f"  Dataset near-dup (dist<=5): {len(phash_matches)} found")

    # Check internal pairwise pHash distances among the 10 downloaded images
    print("\n--- CHECKING INTRA-BATCH PAIRWISE PHASH DISTANCES ---")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            r1 = results[i]
            r2 = results[j]
            if r1.get("phash") and r2.get("phash"):
                h1 = imagehash.hex_to_hash(r1["phash"])
                h2 = imagehash.hex_to_hash(r2["phash"])
                dist = h1 - h2
                if dist <= 8:
                    print(f"  Warning: Intra-batch high similarity between {r1['candidate_id']} and {r2['candidate_id']}: dist={dist}")
                else:
                    print(f"  {r1['candidate_id']} vs {r2['candidate_id']}: dist={dist}")
                    
    # Save log and manifest
    out_log = root / "reports/source_verification/minibatch01_download_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump({
            "total_candidates": len(TARGET_CANDIDATE_IDS),
            "total_downloaded_bytes": total_bytes_downloaded,
            "max_allowed_bytes": MAX_TOTAL_BYTES,
            "results": results
        }, f, indent=2)
    print(f"\nSaved download log to: {out_log}")

if __name__ == "__main__":
    main()
