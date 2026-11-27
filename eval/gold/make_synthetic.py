"""Generates SYNTHETIC gold fixtures (random, seeded). For testing the code only, not real labels.
    python -m eval.gold.make_synthetic
Writes eval/gold/synthetic/gold_people_synthetic.csv and gold_frames_synthetic.csv.
"""
import csv
import random
from pathlib import Path

VIDEO = "synthetic_video"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white",
          "gray", "beige", "multicolor", "unknown"]
TYPES = ["sweater", "hoodie", "t-shirt", "shirt", "jacket", "coat", "dress", "pants_only", "vest",
         "uniform", "other", "unknown"]
VISIBLE = ["да", "спиной", "частично", "не видно"]
OUT = Path(__file__).parent / "synthetic"


def main(n_crops=60, n_frames=22, seed=7):
    rnd = random.Random(seed)
    OUT.mkdir(exist_ok=True)
    rows = []
    for i in range(n_crops):
        snap, idx = f"sn_{i // 6:04d}", i % 6
        color = rnd.choices(COLORS, weights=[3, 1, 1, 2, 4, 1, 1, 2, 9, 3, 4, 2, 1, 2])[0]
        typ = rnd.choice(TYPES)
        vis = rnd.choices(VISIBLE, weights=[6, 3, 2, 1])[0]
        h = rnd.choice([50, 70, 90, 120, 160, 220])
        raters = ["synth_a", "synth_b"] + (["synth_c"] if i < 50 else [])
        for r in raters:
            c = color if (r == "synth_a" or rnd.random() < 0.8) else rnd.choice(COLORS)
            t = typ if (r == "synth_a" or rnd.random() < 0.85) else rnd.choice(TYPES)
            v = vis if rnd.random() < 0.9 else rnd.choice(VISIBLE)
            rows.append([VIDEO, snap, idx, c, "", t, v, h, r])
    with open(OUT / "gold_people_synthetic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snapshot_id", "person_idx", "color", "color_2", "type", "visible", "box_h", "labeler"])
        w.writerows(rows)
    frows = []
    for i in range(n_frames):
        true = rnd.randint(3, 18)
        frows.append([VIDEO, f"sn_{i:04d}", true, "synth_a"])
        frows.append([VIDEO, f"sn_{i:04d}", max(0, true + rnd.choice([0, 0, 0, 1, -1])), "synth_b"])
    with open(OUT / "gold_frames_synthetic.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "snapshot_id", "true_people_count", "labeler"])
        w.writerows(frows)
    print(f"wrote {len(rows)} people rows, {len(frows)} frame rows to {OUT}")


if __name__ == "__main__":
    main()
