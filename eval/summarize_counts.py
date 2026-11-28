"""Counting people per frame: what eval.score_methods count does NOT show. The metrics themselves are not re-invented:
the gold rule and the line filter are the same as in score_methods.cmd_count (see below).

    python -m eval.summarize_counts --gold eval/gold/gold_frames_*.csv \
        --frames-index arai_data/video*/video*_frames_index.json \
        --vlm-counts "server_runs2/count_*.jsonl" --out-md server_runs2/count_summary.md

What it prints (markdown):
  a) per count file: frames total, ok, truncated, error (everything else), suspicious, frames used for metrics;
  b) per method (YOLO raw, YOLO without duplicate boxes, every count file) and per video + all together:
     frames, MAE, share of exact answers, mean signed error (minus = misses people, plus = extra people);
  c) ceiling: agreement between labelers on their common frames (MAE, exact share, per video and together) and the
     list of frames where they differ, WITH the video (score_methods prints only sn_XXXX, and sn_0000 is in both
     videos);
  d) every file <base>_<tag>.jsonl next to <base>.jsonl (here: count_<model>_R3_rep1.1 vs count_<model>_R3):
     ok share in both, and on frames ok in both the mean shift of people_count (tag minus base).

Same rules as score_methods.cmd_count:
  * gold: every eval/gold/gold_frames_*.csv row with a filled true_people_count; key (video_id, snapshot_id);
    several labelers on one frame -> lower median (with two labelers: the smaller number);
  * only frames that are in a frames_index.json are scored (that is where YOLO numbers come from);
  * a VLM line counts only if status == ok, suspicious is not true and people_count is not empty.
score_methods keeps these rules inline in cmd_count (no separate function to import), so the two lines that copy
them live in gold_truth() and usable_line(); a test checks that both scripts give the same MAE on the same files.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from eval.score_methods import _expand  # same mask expansion as score_methods (--gold eval/gold/gold_frames_*.csv)


def gold_truth(counts):
    """score_methods.cmd_count: several labelers -> median (lower)."""
    return sorted(counts)[(len(counts) - 1) // 2]


def usable_line(r):
    """score_methods.cmd_count: lines with status != ok, suspicious or without people_count are not scored."""
    return not (r.get("status") != "ok" or r.get("suspicious") or r.get("people_count") is None)


def load_gold(paths):
    """-> (by_frame {(video_id, snapshot_id): {labeler: count}}). Empty true_people_count rows are skipped."""
    by_frame = defaultdict(dict)
    for one in _expand(paths):
        with open(one, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if str(r.get("true_people_count", "")).strip() == "":
                    continue
                key = (r["video_id"], str(r["snapshot_id"]).strip())
                lab = r.get("labeler") or Path(one).stem
                if lab in by_frame[key]:
                    raise SystemExit(f"{one}: разметчик {lab} дважды размечал {key}")
                by_frame[key][lab] = int(r["true_people_count"])
    return dict(by_frame)


def load_frames_index(paths):
    """-> ({(video_id, snapshot_id): (people_count, people_after_dedup)}, {video_id: short name})."""
    pred, names = {}, {}
    for pat in paths:
        for p in sorted(glob.glob(str(pat))):
            short = Path(p).stem.replace("_frames_index", "").replace("frames_index", "") or Path(p).parent.name
            for fr in json.loads(Path(p).read_text(encoding="utf-8")):
                pred[(fr["video_id"], fr["snapshot_id"])] = (fr["people_count"], fr["people_after_dedup"])
                names.setdefault(fr["video_id"], short)
    return pred, names


def load_count_file(path):
    """-> (status counts dict, {key: people_count} of usable lines, duplicate keys)."""
    st = {"frames": 0, "ok": 0, "truncated": 0, "error": 0, "suspicious": 0}
    used, seen, dups = {}, set(), []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r.get("video_id"), r.get("snapshot_id"))
        if key in seen:
            dups.append(key)
        seen.add(key)
        st["frames"] += 1
        s = r.get("status")
        st["ok" if s == "ok" else "truncated" if s == "truncated" else "error"] += 1
        st["suspicious"] += int(r.get("suspicious") is True)
        if usable_line(r):
            used[key] = int(r["people_count"])
    st["used"] = len(used)
    st["duplicate_keys"] = len(dups)
    return st, used, dups


def stats(errs):
    """-> {n, mae, exact, bias} or None for an empty list."""
    if not errs:
        return None
    n = len(errs)
    return {"n": n, "mae": sum(abs(e) for e in errs) / n, "exact": sum(e == 0 for e in errs) / n,
            "bias": sum(errs) / n}


def errors_by_video(values, truth):
    """values: {key: predicted}; truth: {key: gold}. -> {video_id: [errors]} on the common keys."""
    out = defaultdict(list)
    for k in sorted(truth):
        if k in values:
            out[k[0]].append(values[k] - truth[k])
    return out


def labeler_agreement(gold):
    """-> list of {a, b, per_video {video: stats}, all stats, diffs [(video, snapshot, ca, cb)]}."""
    labelers = sorted({lab for v in gold.values() for lab in v})
    res = []
    for a, b in combinations(labelers, 2):
        per, diffs = defaultdict(list), []
        for k in sorted(gold):
            v = gold[k]
            if a in v and b in v:
                per[k[0]].append(v[b] - v[a])
                if v[a] != v[b]:
                    diffs.append((k[0], k[1], v[a], v[b]))
        if per:
            res.append({"a": a, "b": b, "per_video": {vid: stats(e) for vid, e in per.items()},
                        "all": stats([e for e in per.values() for e in e]), "diffs": diffs})
    return res


def tagged_pairs(names):
    """File stems -> [(base, tagged)] where tagged == base + '_' + tag and base is present too."""
    s = set(names)
    out = []
    for n in sorted(names):
        for b in sorted(s, key=len, reverse=True):
            if n != b and n.startswith(b + "_") and not re.search(r"_repeat\d+$", n):
                out.append((b, n))
                break
    return out


def _f(x, pct=False, sign=False):
    if x is None:
        return "-"
    if pct:
        return f"{x:.0%}"
    return f"{x:+.2f}" if sign else f"{x:.2f}"


def build(gold_paths, frames_index, vlm_counts):
    gold_raw = load_gold(gold_paths)
    if not gold_raw:
        raise SystemExit("в эталоне нет ни одной заполненной строки true_people_count")
    pred, vnames = load_frames_index(frames_index)
    truth = {k: gold_truth(list(v.values())) for k, v in gold_raw.items() if k in pred}
    if not truth:
        raise SystemExit("нет общих кадров между эталоном и frames_index.json (проверь video_id и sn_XXXX)")
    videos = sorted({k[0] for k in truth}, key=lambda v: (vnames.get(v, v), v))

    def vname(v):
        return vnames.get(v, v)

    files = {}
    for pat in vlm_counts:
        for p in sorted(glob.glob(str(pat))):
            files[Path(p).stem] = load_count_file(p)

    md = ["## Счёт людей: дополнение к score_methods count", "",
          "Эталон: " + ", ".join(f"{vname(v)} = {v}" for v in videos) + f"; кадров с эталоном и в frames_index: "
          f"{len(truth)}. Правило эталона как в score_methods: при нескольких разметчиках нижняя медиана "
          "(при двух - меньшее число). VLM-строка учитывается, только если status=ok, не suspicious и "
          "people_count не пустой.", ""]
    gold_only = sorted(k for k in gold_raw if k not in pred)
    if gold_only:
        md += [f"Кадры эталона без frames_index (не учтены): {len(gold_only)}", ""]

    # a) statuses
    if files:
        md += ["### а) Статусы ответов по файлам", "",
               "| файл | кадров | ok | truncated | error | suspicious | учтено |", "|---|---|---|---|---|---|---|"]
        for n, (st, _, dups) in files.items():
            md.append(f"| {n} | {st['frames']} | {st['ok']} | {st['truncated']} | {st['error']} | "
                      f"{st['suspicious']} | {st['used']} |")
        for n, (st, _, dups) in files.items():
            if dups:
                md.append(f"\nВНИМАНИЕ: в {n} повторяются ключи (берётся последняя строка): {dups}")
        md.append("")

    # b) metrics per video and together
    methods = [("YOLO сырой", {k: v[0] for k, v in pred.items()}),
               ("YOLO без дублей", {k: v[1] for k, v in pred.items()})]
    methods += [(n, used) for n, (_, used, _) in files.items()]
    cols = videos + [None]
    md += ["### б) По видео и вместе: MAE / точных / смещение со знаком (n кадров)", "",
           "| метод | " + " | ".join(vname(v) if v else "вместе" for v in cols) + " |",
           "|---|" + "---|" * len(cols)]
    table_b = {}
    for name, values in methods:
        per = errors_by_video(values, truth)
        cells = []
        for v in cols:
            s = stats([e for vid, es in per.items() for e in es] if v is None else per.get(v, []))
            table_b[(name, v)] = s
            cells.append("-" if s is None else
                         f"{_f(s['mae'])} / {_f(s['exact'], pct=True)} / {_f(s['bias'], sign=True)} ({s['n']})")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md += ["", "Минус = модель теряет людей, плюс = лишние. n меньше числа кадров, если часть ответов "
           "truncated / error / suspicious: такие модели сравнивать по MAE с остальными нельзя.", ""]

    # c) ceiling
    agree = labeler_agreement({k: v for k, v in gold_raw.items() if k in pred})
    md += ["### в) Потолок: насколько расходятся разметчики", ""]
    if not agree:
        md += ["Общих кадров у двух и более разметчиков нет.", ""]
    for g in agree:
        cells = [f"{vname(v)}: MAE {_f(s['mae'])}, совпало {_f(s['exact'], pct=True)}, n={s['n']}"
                 for v, s in sorted(g["per_video"].items(), key=lambda x: vname(x[0]))]
        s = g["all"]
        md += [f"**{g['a']} и {g['b']}**: " + "; ".join(cells) +
               f"; вместе: MAE {_f(s['mae'])}, совпало {_f(s['exact'], pct=True)}, "
               f"разошлись на {len(g['diffs'])} из {s['n']} кадров, смещение {g['b']} относительно {g['a']}: "
               f"{_f(s['bias'], sign=True)}", ""]
        if g["diffs"]:
            md += [f"| видео | кадр | {g['a']} | {g['b']} | эталон (нижняя медиана) |", "|---|---|---|---|---|"]
            md += [f"| {vname(v)} | {sn} | {ca} | {cb} | {truth.get((v, sn), '-')} |" for v, sn, ca, cb in g["diffs"]]
            md.append("")
    table_c = agree

    # d) tagged runs
    pairs = tagged_pairs(list(files))
    table_d = []
    if pairs:
        md += ["### г) Прогон с меткой против базового (одни и те же кадры)", "",
               "| база | с меткой | ok база | ok метка | кадров ok в обоих | ср. people база | ср. people метка | "
               "сдвиг (метка - база) | MAE база | MAE метка |", "|---|---|---|---|---|---|---|---|---|---|"]
        for b, t in pairs:
            sb, ub, _ = files[b]
            stt, ut, _ = files[t]
            common = sorted(set(ub) & set(ut))
            shift = stats([ut[k] - ub[k] for k in common])
            mb = sum(ub[k] for k in common) / len(common) if common else None
            mt = sum(ut[k] for k in common) / len(common) if common else None
            mae_b, mae_t = table_b.get((b, None)), table_b.get((t, None))
            row = {"base": b, "tagged": t, "ok_base": sb["ok"] / sb["frames"] if sb["frames"] else None,
                   "ok_tagged": stt["ok"] / stt["frames"] if stt["frames"] else None, "common": len(common),
                   "shift": shift["bias"] if shift else None}
            table_d.append(row)
            md.append(f"| {b} | {t} | {_f(row['ok_base'], pct=True)} | {_f(row['ok_tagged'], pct=True)} | "
                      f"{len(common)} | {_f(mb)} | {_f(mt)} | {_f(row['shift'], sign=True)} | "
                      f"{_f(mae_b and mae_b['mae'])} ({mae_b['n'] if mae_b else 0}) | "
                      f"{_f(mae_t and mae_t['mae'])} ({mae_t['n'] if mae_t else 0}) |")
        md += ["", "Сдвиг и средние посчитаны только на кадрах, где оба прогона дали годный ответ. MAE - на своих "
               "годных кадрах (в скобках n), поэтому при разном n они не вполне сравнимы.", ""]
    return "\n".join(md) + "\n", {"files": {n: v[0] for n, v in files.items()}, "b": table_b, "c": table_c,
                                  "d": table_d, "truth": truth}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, nargs="+")
    ap.add_argument("--frames-index", required=True, nargs="+")
    ap.add_argument("--vlm-counts", nargs="*", default=[])
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    md, _ = build(a.gold, a.frames_index, a.vlm_counts)
    print(md)
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md, encoding="utf-8")
        print(f"записано: {a.out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
