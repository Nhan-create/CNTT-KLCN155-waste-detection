"""Generate visual comparison and quantitative metrics for the 18 cross-split pHash candidate pairs."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def main():
    csv_path = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\all_cross_split_phash_candidates.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    base_dir = Path(r"D:\HK7\Đồ án khóa luận")

    vis_dir = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\visual_phash_inspection")
    vis_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, row in df.iterrows():
        p1 = base_dir / row["path1"]
        p2 = base_dir / row["path2"]

        if not p1.exists() or not p2.exists():
            print(f"Missing file for pair {idx+1}: {p1} or {p2}")
            continue

        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size

        im1_r = img1.resize((128, 128))
        im2_r = img2.resize((128, 128))
        arr1 = np.array(im1_r, dtype=np.float32)
        arr2 = np.array(im2_r, dtype=np.float32)
        mae = float(np.mean(np.abs(arr1 - arr2)))

        target_h = 280
        w1_scaled = int(w1 * target_h / h1)
        w2_scaled = int(w2 * target_h / h2)

        comp = Image.new("RGB", (w1_scaled + w2_scaled + 20, target_h + 80), (255, 255, 255))
        comp.paste(img1.resize((w1_scaled, target_h)), (0, 75))
        comp.paste(img2.resize((w2_scaled, target_h)), (w1_scaled + 20, 75))

        draw = ImageDraw.Draw(comp)
        h_dist = row["hamming_dist"]
        c1 = row["class1"]
        c2 = row["class2"]
        s1 = row["split1"]
        s2 = row["split2"]
        f1 = row["file1"]
        f2 = row["file2"]

        title = f"Pair #{idx+1} | pHash Hamming Dist: {h_dist} | Resized MAE: {mae:.1f} | Class: {c1} vs {c2}"
        sub1 = f"LEFT: [{s1.upper()}] {f1} ({w1}x{h1})"
        sub2 = f"RIGHT: [{s2.upper()}] {f2} ({w2}x{h2})"
        draw.text((10, 5), title, fill=(0, 0, 0))
        draw.text((10, 25), sub1, fill=(180, 0, 0) if s1 == "test" else (0, 0, 160))
        draw.text((10, 45), sub2, fill=(180, 0, 0) if s2 == "test" else (0, 0, 160))

        comp_filename = f"pair_{idx+1:02d}_{c1}_{s1}_vs_{c2}_{s2}_dist{h_dist}.jpg"
        comp_path = vis_dir / comp_filename
        comp.save(comp_path, quality=85)

        # Precise classification of the pair:
        if c1 != c2:
            verdict = "CROSS_CLASS_COINCIDENCE"
            explanation = f"Different classes ({c1} vs {c2}) sharing low-frequency frequency patterns (plain background / simple contour)."
        elif mae < 20.0:
            verdict = "BURST_SHOT_SAME_OBJECT"
            explanation = "Near-identical or burst shot of the same physical object with identical background and lighting."
        elif mae < 38.0:
            verdict = "SAME_OBJECT_ROTATED_OR_PERSPECTIVE"
            explanation = "Same physical object captured with different camera angle, rotation, or slight shift."
        else:
            verdict = "VISUAL_COINCIDENCE_GENERIC_SHAPE"
            explanation = "Different physical objects that happen to have similar low-frequency DCT coefficients."

        results.append({
            "pair_id": idx + 1,
            "class1": c1,
            "class2": c2,
            "split1": s1,
            "file1": f1,
            "path1": str(p1),
            "split2": s2,
            "file2": f2,
            "path2": str(p2),
            "hamming_dist": h_dist,
            "pixel_mae": round(mae, 2),
            "size1": f"{w1}x{h1}",
            "size2": f"{w2}x{h2}",
            "verdict": verdict,
            "explanation": explanation,
            "comparison_image": comp_filename,
        })

    df_res = pd.DataFrame(results)
    out_csv = Path(r"D:\CNTT-KLCN155-waste-detection\data\audit\cross_split_phash_verdicts.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"Saved {len(df_res)} verdicts to {out_csv}")
    print(f"Generated {len(df_res)} comparison images in {vis_dir}")

    # Summary of verdicts
    print("\nSummary of Verdicts across 18 candidate pairs:")
    print(df_res["verdict"].value_counts())
    print("\nDetailed list:")
    for _, r in df_res.iterrows():
        print(f"Pair #{r['pair_id']:02d}: [{r['split1']} vs {r['split2']}] {r['class1']} ({r['file1']}) <-> {r['class2']} ({r['file2']}) | Dist={r['hamming_dist']} | MAE={r['pixel_mae']} | {r['verdict']}")

if __name__ == "__main__":
    main()
