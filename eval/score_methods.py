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
        --out-md results/clothing.md --out-dir results/confusion [--exclude-file eval/gold_real/fewshot_examples.json]
    Таблица: accuracy / macro-F1 цвета и типа, top-2 цвета, доля unknown, доля black (модель / эталон), сек на кроп,
    accuracy цвета по высоте кропа box_h (<80 / 80-150 / >150 px). Если в эталоне 2+ разметчика (несколько CSV или
    колонка labeler), ниже идёт их согласие попарно: % совпадений и каппа Коэна по color и type.
    --exclude-file: json с кропами, которые не оцениваются ни у одного метода (few-shot примеры варианта C):
    список или {"examples": [...]}, у каждого video_id, snapshot_id, person_idx (или вложенный "source" с ними).

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


def _expand(paths):
    """--gold может быть несколькими файлами или маской: eval/gold/gold_frames_*.csv (по файлу на разметчика)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        hit = sorted(glob.glob(str(p)))
        out += hit if hit else [str(p)]
    return out


def _alias(rows, field, alt):
    """Колонка могла называться clothing_color вместо color: копируем значение в field."""
    return [{**r, field: r.get(field) or r.get(alt)} for r in rows]


def load_gold_people(path, skip_visible=("не видно", "not_visible")):
    groups = defaultdict(list)
    for one in _expand(path):
        with open(one, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not (r.get("color") or r.get("clothing_color")):
                    continue  # строка ещё не размечена
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


def load_exclude(paths):
    """--exclude-file -> set of (video_id, snapshot_id, person_idx). Understands eval/gold_real/fewshot_examples.json
    (flat keys) and vlm/fewshot.json (keys inside "source")."""
    out = set()
    for one in _expand(paths or []):
        d = json.loads(Path(one).read_text(encoding="utf-8"))
        items = d.get("examples", []) if isinstance(d, dict) else d
        for e in items:
            s = e.get("source") if isinstance(e.get("source"), dict) else e
            out.add((s["video_id"], str(s["snapshot_id"]).strip(), int(s["person_idx"])))
    return out


def labeler_agreement(path, exclude=frozenset(), fields=("color", "type")):
    """Согласие разметчиков попарно (общие объекты пары): [{pair, field, n, agree, kappa}]. [] если разметчик один."""
    from eval.agreement import pairwise  # % совпадений + eval.metrics.cohen_kappa, та же функция, что в eval.agreement
    rows = []
    for one in _expand(path):
        with open(one, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                r = {**r, "color": r.get("color") or r.get("clothing_color"),
                     "type": r.get("type") or r.get("clothing_type"),
                     "labeler": r.get("labeler") or Path(one).stem, "snapshot_id": str(r["snapshot_id"]).strip()}
                if r["color"] and (r["video_id"], r["snapshot_id"], int(r["person_idx"])) not in exclude:
                    rows.append(r)
    if len({r["labeler"] for r in rows}) < 2:
        return []
    return [x for x in pairwise(rows, fields) if x["n"]]


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


def _bucket_cells(pairs):
    cells = []
    for bn, _, _ in BUCKETS:
        sub = [(g, p) for g, p in pairs if bucket(p.get("box_h")) == bn]
        cells.append(f"{sum(g['color'] == p['color'] for g, p in sub) / len(sub):.3f} (n={len(sub)})" if sub else "-")
    return cells


def cmd_people(a):
    gold, conflicts, n_obj = load_gold_people(a.gold)
    exclude = load_exclude(a.exclude_file)
    if exclude:
        before = len(gold)
        gold = {k: v for k, v in gold.items() if k not in exclude}
        print(f"исключено из оценки (--exclude-file): {before - len(gold)} из {len(exclude)} кропов")
    print(f"эталон: объектов {n_obj}, годных {len(gold)}, конфликтов разметчиков {conflicts}\n")
    lines = ["| метод | n | accuracy | macro-F1 | top-2 | unknown | black pred/gold | сек/кроп | "
             "тип: accuracy | тип: macro-F1 | тип: unknown | цвет <80px | цвет 80-150px | цвет >150px |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
            t = score_field(pairs, "type")
            tcells = [f"{t['accuracy']:.3f}", f"{t['macro_f1']:.3f}", f"{t['unknown_share']:.3f}"] if t else ["-"] * 3
            lines.append(f"| {name} | {m['n']} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
                         f"{m['top2']:.3f} | {m['unknown_share']:.3f} | "
                         f"{m['black_pred']:.2f}/{m['black_gold']:.2f} | {sec:.4f} | "
                         + " | ".join(tcells + _bucket_cells(pairs)) + " |")
            if a.out_dir:
                Path(a.out_dir).mkdir(parents=True, exist_ok=True)
                confusion_csv(pairs, "color", Path(a.out_dir) / f"{name}_color.csv")
    md = ("## Одежда: цвет и тип\n\n" + "\n".join(lines) + "\n\n"
          "accuracy / macro-F1 / top-2 / unknown / black без пометки - по цвету. Тип \"-\": метод не даёт тип "
          "(все unknown). Цвет по высоте кропа box_h: accuracy цвета (n кропов в бакете).\n")
    agr = labeler_agreement(a.gold, exclude)
    if agr:
        md += ("\n### Согласие разметчиков (общие объекты пары)\n\n| пара | поле | n | совпадений | каппа Коэна |\n"
               "|---|---|---|---|---|\n" + "\n".join(f"| {x['pair']} | {x['field']} | {x['n']} | {x['agree']:.1%} | "
                                                       f"{x['kappa']:.3f} |" for x in agr) + "\n")
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
    gold, labelers = defaultdict(list), defaultdict(int)
    for one in _expand(a.gold):
        with open(one, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if str(r.get("true_people_count", "")).strip() == "":
                    continue  # ещё не заполнено
                gold[(r["video_id"], str(r["snapshot_id"]).strip())].append(int(r["true_people_count"]))
                labelers[r.get("labeler", "?")] += 1
    if not gold:
        raise SystemExit("в эталоне нет ни одной заполненной строки true_people_count")
    print("разметчики: " + ", ".join(f"{k} - {v} кадров" for k, v in labelers.items()))
    spread = {k: v for k, v in gold.items() if len(set(v)) > 1}
    if spread:
        print(f"разметчики разошлись на {len(spread)} из {len(gold)} кадров: " +
              "; ".join(f"{k[1]}={v}" for k, v in sorted(spread.items())))
    pred = {}
    for pat in a.frames_index:
        for p in glob.glob(pat):
            for fr in json.loads(Path(p).read_text(encoding="utf-8")):
                pred[(fr["video_id"], fr["snapshot_id"])] = (fr["people_count"], fr["people_after_dedup"])
    vlm, vlm_skipped = {}, {}
    for pat in a.vlm_counts:
        for pth in sorted(glob.glob(pat)):
            name, used, skipped = Path(pth).stem, {}, 0
            for line in open(pth, encoding="utf-8"):
                r = json.loads(line)
                if r.get("status") != "ok" or r.get("suspicious") or r.get("people_count") is None:
                    skipped += 1
                    continue
                used[(r["video_id"], r["snapshot_id"])] = int(r["people_count"])
            vlm[name], vlm_skipped[name] = used, skipped
    raw_e, dd_e, rows = [], [], []
    vlm_e = {n: [] for n in vlm}
    for k, counts in sorted(gold.items()):
        if k not in pred:
            continue
        truth = sorted(counts)[(len(counts) - 1) // 2]  # несколько разметчиков: медиана (нижняя)
        raw, dd = pred[k]
        raw_e.append(raw - truth)
        dd_e.append(dd - truth)
        extra = {}
        for n, used in vlm.items():
            if k in used:
                vlm_e[n].append(used[k] - truth)
                extra[n] = used[k] - truth, used[k]
        rows.append((k, truth, raw, dd, extra))
    if not raw_e:
        raise SystemExit("нет общих кадров между эталоном и frames_index.json (проверь video_id и sn_XXXX)")
    md = ["## Счёт людей: детектор и VLM против человека", "", f"кадров с эталоном: {len(raw_e)}", "",
          "**YOLO, сырой people_count (как сейчас в базе):** " + _count_stats(raw_e), "",
          "**YOLO после удаления дублей рамок:** " + _count_stats(dd_e), ""]
    for n, errs in vlm_e.items():
        if errs:
            md += [f"**{n}** (кадров {len(errs)}, не учтено из-за status/suspicious: {vlm_skipped[n]}): "
                   + _count_stats(errs), ""]
        else:
            md += [f"**{n}**: нет ни одного годного кадра (не учтено {vlm_skipped[n]})", ""]
    md += ["минус = теряет людей, плюс = лишние", ""]
    head = "| видео | срез | эталон | YOLO сырой | YOLO без дублей |" + "".join(f" {n} |" for n in vlm)
    md += [head, "|" + "---|" * (5 + len(vlm))]
    for k, t, r, d, extra in rows:
        cells = "".join(f" {extra[n][1]} ({extra[n][0]:+d}) |" if n in extra else " - |" for n in vlm)
        md.append(f"| {k[0]} | {k[1]} | {t} | {r:d} ({r - t:+d}) | {d:d} ({d - t:+d}) |{cells}")
    out = "\n".join(md)
    print(out)
    if a.out_md:
        Path(a.out_md).write_text(out, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("people")
    p.add_argument("--gold", required=True, nargs="+")
    p.add_argument("--pred", nargs="+", required=True)
    p.add_argument("--out-md")
    p.add_argument("--out-dir")
    p.add_argument("--exclude-file", nargs="*", default=[],
                   help="json с кропами (video_id, snapshot_id, person_idx), которые не оцениваются: few-shot примеры")
    c = sub.add_parser("count")
    c.add_argument("--gold", required=True, nargs="+")
    c.add_argument("--frames-index", nargs="+", required=True)
    c.add_argument("--vlm-counts", nargs="*", default=[],
                   help="count_*.jsonl от VLM (R2/R3/R4); строки со status!=ok или suspicious=true не учитываются")
    c.add_argument("--out-md")
    a = ap.parse_args()
    {"people": cmd_people, "count": cmd_count}[a.cmd](a)


if __name__ == "__main__":
    main()
