"""Summary of the server prediction files (run_predictions.py output) WITHOUT the gold standard:
how many answers came back, how fast, how stable. Accuracy is scored separately by bench_clothing / bench_count.

    python -m eval.summarize_runs --dir server_runs [--out server_runs/summary_runs.csv]

Reads every *.jsonl in --dir except runs.jsonl (logs and other files are ignored). File names:
    pred_clothing_<model>_R1_<A-D>[_repeat2][_<tag>].jsonl      R1 (crops)
    count_<model>_<R2|R3|R4>[_<tag>].jsonl                        R2 / R3 / R4
<tag> (e.g. rep1.1 = a run with a repetition penalty) makes its own row. Unknown names: model / mode / variant
are taken from the lines themselves, tag = file name. A *_repeat2 file is not a row of its own: it is only used for
the repeatability of its base file.

One row per model x mode x variant x tag x video_id. A file of the index mode (crops_index.json of two videos)
holds several videos with the SAME snapshot_id / crop_id (sn_0000 is in both): it is split by video_id, and the
repeatability pairs are matched on (video_id, snapshot_id, person_idx), never on snapshot_id alone.
    rows, ok / truncated / error shares (error = everything that is neither ok nor truncated: error, skipped;
    the three shares add up to 1), mean and p95 of seconds (lines with seconds only),
    R1: black and unknown shares of color (among ok lines), repeatability vs *_repeat2 (share of the same color and
        the same type, among people that are ok in both passes),
    count: mean, standard deviation, min and max of people_count (ok lines), number of suspicious lines.
Old files (before raw_response / finish_reason / suspicious existed) are read without errors: format = old,
suspicious = empty (unknown, NOT 0). In old files a cut answer was recorded as error, so truncated = 0 there
does not mean "nothing was cut".
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

MODE_CODE = {"crop": "R1", "frame-count": "R2", "frame-list": "R3", "boxes": "R4",
             "R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4"}
RE_CROPS = re.compile(r"^pred_clothing_(?P<model>.+)_R1_(?P<variant>[A-D])(?P<rest>.*)\.jsonl$")
RE_COUNT = re.compile(r"^count_(?P<model>.+)_(?P<mode>R[234])(?P<rest>.*)\.jsonl$")
RE_REPEAT = re.compile(r"_repeat(\d+)")

COLUMNS = ["model", "mode", "variant", "tag", "video_id", "file", "format", "rows", "ok_share", "truncated_share", "error_share",
           "sec_mean", "sec_p95", "black_share", "unknown_share", "repeat_pairs", "repeat_color_same",
           "repeat_type_same", "count_mean", "count_std", "count_min", "count_max", "suspicious", "bad_lines"]
HEADERS = {"model": "модель", "mode": "режим", "variant": "вариант", "tag": "метка", "video_id": "видео", "file": "файл",
           "format": "формат", "rows": "строк", "ok_share": "ok", "truncated_share": "truncated",
           "error_share": "ошибка", "sec_mean": "сек ср.", "sec_p95": "сек p95", "black_share": "black",
           "unknown_share": "unknown", "repeat_pairs": "пар повтора", "repeat_color_same": "повтор цвет",
           "repeat_type_same": "повтор тип", "count_mean": "people ср.", "count_std": "people СКО",
           "count_min": "people мин", "count_max": "people макс", "suspicious": "suspicious",
           "bad_lines": "битых строк"}


def read_jsonl(path):
    """-> (rows, number of unreadable lines). Never raises on a bad line."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(r, dict):
                rows.append(r)
            else:
                bad += 1
    return rows, bad


def p95(values):
    """Nearest-rank 95th percentile (no interpolation)."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def share(n, d):
    return round(n / d, 4) if d else None


def describe_file(path, rows):
    """-> {model, mode, variant, tag, repeat(int|None)} from the name, falling back to the lines."""
    name = path.name
    m = RE_CROPS.match(name) or RE_COUNT.match(name)
    info = {"model": None, "mode": None, "variant": None, "tag": "", "repeat": None}
    if m:
        rest = m.group("rest")
        rep = RE_REPEAT.search(rest)
        if rep:
            info["repeat"] = int(rep.group(1))
            rest = rest[:rep.start()] + rest[rep.end():]
        info.update(model=m.group("model"), mode=m.groupdict().get("mode") or "R1",
                    variant=m.groupdict().get("variant") or "A", tag=rest.strip("_"))
        return info
    first = next((r for r in rows if r.get("status") == "ok"), rows[0] if rows else {})
    rec = first.get("record") if isinstance(first.get("record"), dict) else {}
    method = first.get("method") or rec.get("method") or ""
    info["model"] = first.get("model") or (method[4:] if method.startswith("vlm:") else method) or "?"
    info["mode"] = MODE_CODE.get(first.get("mode") or rec.get("mode") or ("crop" if "person_idx" in first else ""), "?")
    info["variant"] = rec.get("prompt_variant") or "A"
    info["tag"] = path.stem
    return info


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def summarize(rows, info, repeat_rows=None):
    n = len(rows)
    st = [r.get("status") for r in rows]
    new_format = any("finish_reason" in r or "raw_response" in r for r in rows)
    secs = [float(r["seconds"]) for r in rows if _num(r.get("seconds"))]
    vids = sorted({str(r.get("video_id")) for r in rows if r.get("video_id") is not None})
    out = {"model": info["model"], "mode": info["mode"], "variant": info["variant"], "tag": info["tag"],
           "video_id": ", ".join(vids),
           "format": "new" if new_format else "old", "rows": n,
           "ok_share": share(st.count("ok"), n), "truncated_share": share(st.count("truncated"), n),
           "error_share": share(sum(1 for s in st if s not in ("ok", "truncated")), n),
           "sec_mean": round(statistics.fmean(secs), 3) if secs else None,
           "sec_p95": round(p95(secs), 3) if secs else None}
    if info["mode"] == "R1":
        ok = [r["record"] for r in rows if r.get("status") == "ok" and isinstance(r.get("record"), dict)]
        colors = [str(rec.get("color") or "").lower() for rec in ok]
        out["black_share"] = share(colors.count("black"), len(colors))
        out["unknown_share"] = share(colors.count("unknown"), len(colors))
        if repeat_rows is not None:
            def by_key(rs):
                return {(r.get("video_id"), r.get("snapshot_id"), r.get("person_idx")): r["record"] for r in rs
                        if r.get("status") == "ok" and isinstance(r.get("record"), dict)}
            a, b = by_key(rows), by_key(repeat_rows)
            keys = sorted(set(a) & set(b), key=str)
            out["repeat_pairs"] = len(keys)
            out["repeat_color_same"] = share(sum(a[k].get("color") == b[k].get("color") for k in keys), len(keys))
            out["repeat_type_same"] = share(sum(a[k].get("type") == b[k].get("type") for k in keys), len(keys))
    else:
        counts = [r["people_count"] for r in rows if r.get("status") == "ok" and _num(r.get("people_count"))]
        if counts:
            out.update(count_mean=round(statistics.fmean(counts), 3),
                       count_std=round(statistics.pstdev(counts), 3), count_min=min(counts), count_max=max(counts))
        if any("suspicious" in r for r in rows):
            out["suspicious"] = sum(1 for r in rows if r.get("suspicious") is True)
    return out


def collect(folder):
    """-> list of summary rows, sorted by model, mode, variant, tag, video_id."""
    files = sorted(p for p in Path(folder).glob("*.jsonl") if p.name != "runs.jsonl")
    loaded = {}
    for p in files:
        rows, bad = read_jsonl(p)
        loaded[p] = (rows, bad, describe_file(p, rows))
    bases = {p: v for p, v in loaded.items() if v[2]["repeat"] is None}
    result = []
    for p, (rows, bad, info) in bases.items():
        rep = None
        if info["mode"] == "R1":
            for q, (qrows, _, qinfo) in loaded.items():
                if qinfo["repeat"] is not None and all(qinfo[k] == info[k] for k in ("model", "mode", "variant", "tag")):
                    rep = qrows
                    break
        for vid in sorted({r.get("video_id") for r in rows} or {None}, key=str):  # one row per video
            vrows = [r for r in rows if r.get("video_id") == vid]
            vrep = None if rep is None else [r for r in rep if r.get("video_id") == vid]
            s = summarize(vrows, info, vrep)
            s.update(file=p.name, bad_lines=bad)
            result.append(s)
    orphans = [p.name for p, v in loaded.items() if v[2]["repeat"] is not None and not any(
        all(v[2][k] == b[2][k] for k in ("model", "mode", "variant", "tag")) for b in bases.values())]
    for name in orphans:
        print(f"повтор без основного файла, пропущен: {name}", file=sys.stderr)
    result.sort(key=lambda r: (r["model"] or "", r["mode"] or "", r["variant"] or "", r["tag"] or "", r["video_id"] or ""))
    return result


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def to_markdown(rows):
    r1 = [r for r in rows if r["mode"] == "R1"]
    cnt = [r for r in rows if r["mode"] != "R1"]
    base = ["model", "mode", "variant", "tag", "video_id", "format", "rows", "ok_share", "truncated_share", "error_share",
            "sec_mean", "sec_p95"]
    parts = []
    for title, rs, extra in (("Одежда (R1)", r1, ["black_share", "unknown_share", "repeat_pairs",
                                                   "repeat_color_same", "repeat_type_same"]),
                             ("Счёт людей (R2/R3/R4)", cnt, ["count_mean", "count_std", "count_min", "count_max",
                                                             "suspicious"])):
        if not rs:
            continue
        cols = base + extra + (["bad_lines"] if any(r.get("bad_lines") for r in rs) else [])
        lines = [f"### {title}", "", "| " + " | ".join(HEADERS[c] for c in cols) + " |",
                 "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rs]
        parts.append("\n".join(lines))
    parts.append("Доли ok / truncated / ошибка в сумме дают 1. Секунды — на один запрос при параллельных запросах "
                 "(--workers), на общей карте. Пустая клетка: величина не считается для этого режима или "
                 "её нет в старом формате файла.")
    return "\n\n".join(parts) + "\n"


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in COLUMNS})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="server_runs", help="folder with the server *.jsonl files")
    ap.add_argument("--out", default=None, help="CSV path (default: <dir>/summary_runs.csv)")
    a = ap.parse_args(argv)
    if not Path(a.dir).is_dir():
        print(f"нет папки {a.dir}", file=sys.stderr)
        return 2
    rows = collect(a.dir)
    if not rows:
        print(f"в {a.dir} нет файлов предсказаний (*.jsonl кроме runs.jsonl)", file=sys.stderr)
        return 1
    out = a.out or str(Path(a.dir) / "summary_runs.csv")
    write_csv(rows, out)
    print(to_markdown(rows))
    print(f"CSV: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
