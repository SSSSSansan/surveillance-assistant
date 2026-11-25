"""Быстрая проверка M1/M2/M3 БЕЗ эталона. Это НЕ точность: VLM тоже ошибается (на этапе 2 путала
бежевую куртку). Скрипт ловит поломки (перекос по цветам, расхождение вариантов), показывает, где метод и
VLM спорят чаще всего, и умеет собрать КАРТИНКУ с кропами из спорной пары, чтобы посмотреть глазами, кто прав.

    python -m eval.quick_agreement --pred results/hist__crop__A.jsonl results/siglip__crop__A.jsonl --vlm vlm/snapshots_vlm.json

    # контактный лист: 12 кропов, где VLM сказала black, а ПЕРВЫЙ метод из --pred сказал blue
    python -m eval.quick_agreement --pred results/hist__crop__A.jsonl --vlm vlm/snapshots_vlm.json \
        --index data/video1/crops/crops_index.json --show black:blue --n 12 --sheet results/sheet_black_blue.jpg

Сравнение с VLM идёт только по видео, которое есть в --vlm (то есть по видео этапа 2).
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from eval.score_methods import load_preds


def load_paths(index_paths):
    import glob
    import json
    out = {}
    for pat in index_paths:
        for p in glob.glob(pat):
            for c in json.loads(Path(p).read_text(encoding="utf-8")):
                if c.get("kept") and c.get("path"):
                    out[(c["video_id"], c["snapshot_id"], c["person_idx"])] = c["path"]
    return out


def contact_sheet(items, paths, out, per_row=6, h=200):
    """items: [(key, vlm_color, method_color)]. Подпись под каждым кропом: id, VLM, метод."""
    tiles = []
    for key, v, m in items:
        img = cv2.imread(paths[key]) if key in paths else None
        if img is None:
            continue
        w = max(40, int(img.shape[1] * h / img.shape[0]))
        tile = cv2.resize(img, (w, h))
        cap = np.full((46, w, 3), 255, np.uint8)
        cv2.putText(cap, f"{key[1]}_p{key[2]:02d}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(cap, f"VLM: {v}", (3, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 0, 0), 1)
        cv2.putText(cap, f"method: {m}", (3, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 160), 1)
        tiles.append(np.vstack([tile, cap]))
    if not tiles:
        return False
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        rows.append(np.hstack(row))
    Wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, Wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--show", help="пара ЦВЕТ_VLM:ЦВЕТ_МЕТОДА, например black:blue")
    ap.add_argument("--index", nargs="*", default=[], help="crops_index.json (нужен для --sheet)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--sheet", help="куда сохранить картинку с кропами")
    a = ap.parse_args()

    vlm = load_preds(a.vlm)
    runs = {p: load_preds(p) for p in a.pred}

    for p, preds in runs.items():
        cnt = Counter(r["color"] for r in preds.values())
        n = len(preds)
        print(f"\n=== {p}  ({n} людей)")
        print("распределение цветов: " + ", ".join(f"{c} {k / n:.0%}" for c, k in cnt.most_common()))
        both = [k for k in preds if k in vlm]
        if both:
            vc = Counter(vlm[k]["color"] for k in both)
            print("для сравнения у VLM:   " + ", ".join(f"{c} {k / len(both):.0%}" for c, k in vc.most_common()))
            pc = Counter(preds[k]["color"] for k in both)
            for c in set(vc) | set(pc):
                d_ = (pc[c] - vc[c]) / len(both)
                if abs(d_) > 0.20:
                    print(f"  ! доля '{c}' у метода отличается от VLM на {d_:+.0%} - проверь правила/баланс белого")
            agree = sum(preds[k]["color"] == vlm[k]["color"] for k in both)
            agree2 = sum(vlm[k]["color"] in (preds[k]["color"], preds[k].get("color_2")) for k in both)
            print(f"согласие с VLM по цвету: {agree}/{len(both)} = {agree / len(both):.0%} "
                  f"(с учётом второго цвета метода: {agree2 / len(both):.0%})")
            pairs = Counter((vlm[k]["color"], preds[k]["color"]) for k in both if preds[k]["color"] != vlm[k]["color"])
            print("чаще всего спорят (VLM -> метод):")
            for (v, m), c in pairs.most_common(a.top):
                print(f"   {v:>9} -> {m:<9} {c}")

    if len(runs) >= 2:
        (pa, ra), (pb, rb) = list(runs.items())[:2]
        keys = [k for k in ra if k in rb]
        diff = [k for k in keys if ra[k]["color"] != rb[k]["color"]]
        print(f"\nварианты расходятся: {len(diff)}/{len(keys)} = {len(diff) / max(1, len(keys)):.0%}  ({pa} vs {pb})")
        pairs = Counter((ra[k]["color"], rb[k]["color"]) for k in diff)
        print("   " + ", ".join(f"{x}->{y} {c}" for (x, y), c in pairs.most_common(6)))

    if a.show:
        want_v, want_m = a.show.split(":")
        first = runs[a.pred[0]]
        hits = [(k, vlm[k]["color"], first[k]["color"]) for k in first
                if k in vlm and vlm[k]["color"] == want_v and first[k]["color"] == want_m]
        print(f"\n--show {a.show}: найдено {len(hits)} кропов в {a.pred[0]}")
        for k, v, m in hits[:a.n]:
            print(f"   {k[1]}_p{k[2]:02d}   VLM={v}  метод={m}")
        if a.sheet:
            paths = load_paths(a.index)
            if not paths:
                print("  ! для --sheet нужен --index (crops_index.json)")
            elif contact_sheet(hits[:a.n], paths, a.sheet):
                print(f"картинка сохранена: {a.sheet}")


if __name__ == "__main__":
    main()
