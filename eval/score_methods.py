"""Сравнение методов с эталоном + T33 (счёт людей YOLO против эталона).

Ожидаемые CSV (экспорт из labels_gold_people / labels_gold_frames; если у Айгерим
другие названия колонок - переименуй в шапке CSV):
  gold_people.csv : video_id,snapshot_id,person_idx,color,type,visible,box_h,labeler
                    (snapshot_id строкой: sn_0003; вместо color/type принимаются clothing_color/clothing_type)
  gold_frames.csv : video_id,snapshot_id,true_people_count,labeler
Предсказания (--pred): results/*.jsonl от eval.run_method ИЛИ готовый snapshots_vlm.json Панды -
тогда он попадёт в таблицу как строка "VLM" (так сразу видно, лучше ли M1/M2/M3 неё).

Одежда:
    python -m eval.score_methods people --gold gold_people.csv \
        --pred results/hist__crop__A.jsonl results/siglip__crop__A.jsonl results/par__crop__A.jsonl \
        --out-md results/clothing.md --out-dir results/confusion

Счёт людей (T33):
    python -m eval.score_methods count --gold gold_frames.csv \
        --frames-index data/meva_school_01/frames_index.json data/meva_school_02/frames_index.json

Если один объект размечали двое+, берётся большинство; при ничьей объект пропускается
(и считается в "конфликтах" - это и есть потолок точности для любой модели).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

BUCKETS = [("<80", 0, 80), ("80-150", 80, 150), (">150", 150, 1e9)]


def _majority(rows, field):
    """rows: список dict одного объекта от разных разметчиков -> (значение, конфликт?)."""
    vals = [r[field] for r in rows if r.get(field) not in (None, "")]
    if not vals:
        return None, False
    c = Counter(vals).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None, True
    return c[0][0], False


def _alias(rows, field, alt):
    """Колонка могла называться clothing_color вместо color: копируем значение в field."""
    return [{**r, field: r.get(field) or r.get(alt)} for r in rows]


def load_gold_people(path, skip_visible=("не видно", "not_visible")):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[(r["video_id"], str(r["snapshot_id"]).strip(), int(r["person_idx"]))].append(r)
    gold, conflicts = {}, 0
    for k, rs in groups.items():
        vis, _ = _majority(rs, "visible")
        if vis in skip_visible:
            continue
        col, c1 = _majority(_alias(rs, "color", "clothing_color"), "color")
        typ, c2 = _majority(_alias(rs, "type", "clothing_type"), "type")
        conflicts += int(c1)
        if col:
            gold[k] = {"color": col, "type": typ, "box_h": rs[0].get("box_h")}
    return gold, conflicts, len(groups)


def load_preds(path):
    """-> {(video_id, snapshot_id, person_idx): строка}. Понимает:
    * .jsonl от eval.run_method;
    * snapshots_vlm.json (формат Панды: snapshots[].people[].clothing_color/clothing_type)."""
    out = {}
    if str(path).endswith(".json"):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for snap in d["snapshots"]:
            for p in snap.get("people", []):
                box = p.get("box")
                out[(d["video_id"], snap["snapshot_id"], int(p["person_idx"]))] = {
                    "video_id": d["video_id"], "snapshot_id": snap["snapshot_id"],
                    "person_idx": int(p["person_idx"]), "color": p.get("clothing_color", "unknown"),
                    "color_2": None, "type": p.get("clothing_type", "unknown"), "confidence": None,
                    "sec": None, "box_h": (box[3] - box[1]) if box else None,
                    "method": "vlm:" + str(d.get("vlm_model", "qwen3-vl-4b"))}
        return out
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[(r["video_id"], str(r["snapshot_id"]), int(r["person_idx"]))] = r
    return out


def bucket(h):
    try:
        h = float(h)
    except (TypeError, ValueError):
        return None
    for name, lo, hi in BUCKETS:
        if lo <= h < hi:
            return name
    return None


def score_field(pairs, field):
    """pairs: [(gold_dict, pred_dict)]. Для type метод без типа (все unknown) пропускаем."""
    from sklearn.metrics import f1_score

    pairs = [(g, p) for g, p in pairs if g.get(field)]
    if not pairs:
        return None
    y, yhat = [g[field] for g, _ in pairs], [p[field] for _, p in pairs]
    if field == "type" and set(yhat) <= {"unknown"}:
        return None
    n = len(pairs)
    res = {
        "n": n,
        "accuracy": sum(a == b for a, b in zip(y, yhat)) / n,
        "macro_f1": float(f1_score(y, yhat, labels=sorted(set(y)), average="macro", zero_division=0)),
        "unknown_share": sum(b == "unknown" for b in yhat) / n,
    }
    if field == "color":
        res["top2"] = sum(g[field] == p[field] or g[field] == p.get("color_2") for g, p in pairs) / n
        res["black_pred"] = sum(b == "black" for b in yhat) / n
        res["black_gold"] = sum(a == "black" for a in y) / n
    return res


def confusion_csv(pairs, field, path):
    labels = sorted({g[field] for g, _ in pairs if g.get(field)} | {p[field] for _, p in pairs})
    cnt = Counter((g[field], p[field]) for g, p in pairs if g.get(field))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold\\pred"] + labels)
        for a in labels:
            w.writerow([a] + [cnt.get((a, b), 0) for b in labels])


def cmd_people(a):
    gold, conflicts, n_obj = load_gold_people(a.gold)
    print(f"эталон: объектов {n_obj}, годных {len(gold)}, конфликтов разметчиков {conflicts}\n")
    lines = ["| метод | n | accuracy | macro-F1 | top-2 | unknown | black pred/gold | сек/кроп |",
             "|---|---|---|---|---|---|---|---|"]
    detail = []
    for pat in a.pred:
        for pth in glob.glob(pat):
            preds = load_preds(pth)
            name = Path(pth).stem
            pairs = [(g, preds[k]) for k, g in gold.items() if k in preds]
            if not pairs:
                print(f"{name}: нет пересечения с эталоном (проверь video_id/snapshot_id/person_idx)")
                continue
            m = score_field(pairs, "color")
            secs = [p["sec"] for _, p in pairs if p.get("sec") is not None]
            sec = sum(secs) / len(secs) if secs else float("nan")
            lines.append(f"| {name} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                         f"{m['top2']:.3f} | {m['unknown_share']:.3f} | "
                         f"{m['black_pred']:.2f}/{m['black_gold']:.2f} | {sec:.4f} |")
            # по высоте кропа
            rows = []
            for bn, lo, hi in BUCKETS:
                sub = [(g, p) for g, p in pairs if bucket(p.get("box_h")) == bn]
                if sub:
                    rows.append(f"{bn}px: {sum(g['color'] == p['color'] for g, p in sub) / len(sub):.3f} (n={len(sub)})")
            detail.append(f"- **{name}** точность цвета по высоте: " + "; ".join(rows))
            t = score_field(pairs, "type")
            if t:
                detail.append(f"- **{name}** тип: accuracy {t['accuracy']:.3f}, macro-F1 {t['macro_f1']:.3f}, n={t['n']}")
            if a.out_dir:
                Path(a.out_dir).mkdir(parents=True, exist_ok=True)
                confusion_csv(pairs, "color", Path(a.out_dir) / f"{name}_color.csv")
    md = "## Одежда: цвет\n\n" + "\n".join(lines) + "\n\n### Детали\n" + "\n".join(detail) + "\n"
    print(md)
    if a.out_md:
        Path(a.out_md).write_text(md, encoding="utf-8")


def _count_stats(errs):
    n = len(errs)
    return (f"MAE: {sum(abs(e) for e in errs) / n:.2f}  |  точных: {sum(e == 0 for e in errs) / n:.0%}  |  "
            f"недосчёт: {sum(e < 0 for e in errs)}  |  пересчёт: {sum(e > 0 for e in errs)}  |  "
            f"средняя ошибка со знаком: {sum(errs) / n:+.2f}")


def cmd_count(a):
    """T33. Сравнивает с человеком ДВА числа: сырой people_count детектора (то, что сейчас в базе)
    и число людей после удаления дублей рамок (cv.crop_people)."""
    gold = defaultdict(list)
    with open(a.gold, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gold[(r["video_id"], str(r["snapshot_id"]).strip())].append(int(r["true_people_count"]))
    pred = {}
    for pat in a.frames_index:
        for p in glob.glob(pat):
            for fr in json.loads(Path(p).read_text(encoding="utf-8")):
                pred[(fr["video_id"], fr["snapshot_id"])] = (fr["people_count"], fr["people_after_dedup"])
    raw_e, dd_e, rows = [], [], []
    for k, counts in sorted(gold.items()):
        if k not in pred:
            continue
        truth = sorted(counts)[(len(counts) - 1) // 2]  # несколько разметчиков: медиана (нижняя)
        raw, dd = pred[k]
        raw_e.append(raw - truth)
        dd_e.append(dd - truth)
        rows.append((k, truth, raw, dd))
    if not raw_e:
        raise SystemExit("нет общих кадров между эталоном и frames_index.json (проверь video_id и sn_XXXX)")
    md = ["## Счёт людей: детектор против человека", "", f"кадров: {len(raw_e)}", "",
          "**Сырой people_count (как сейчас в базе):** " + _count_stats(raw_e), "",
          "**После удаления дублей рамок:** " + _count_stats(dd_e), "",
          "минус = детектор теряет людей, плюс = лишние рамки", "",
          "| видео | срез | эталон | сырой | без дублей |", "|---|---|---|---|---|"]
    md += [f"| {k[0]} | {k[1]} | {t} | {r:d} ({r - t:+d}) | {d:d} ({d - t:+d}) |" for k, t, r, d in rows]
    out = "\n".join(md)
    print(out)
    if a.out_md:
        Path(a.out_md).write_text(out, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("people")
    p.add_argument("--gold", required=True)
    p.add_argument("--pred", nargs="+", required=True)
    p.add_argument("--out-md")
    p.add_argument("--out-dir")
    c = sub.add_parser("count")
    c.add_argument("--gold", required=True)
    c.add_argument("--frames-index", nargs="+", required=True)
    c.add_argument("--out-md")
    a = ap.parse_args()
    {"people": cmd_people, "count": cmd_count}[a.cmd](a)


if __name__ == "__main__":
    main()
