"""Shared helpers for bench_clothing / bench_count / agreement: gold loading, snapshots, runs.jsonl, tables."""
import csv
import datetime
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "vlm")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULTS_DIR = ROOT / "eval" / "results"
GOLD_DIR = ROOT / "eval" / "gold"
SYNTH_BANNER = ("> **ВНИМАНИЕ: SYNTHETIC.** В данных есть video_id с 'synthetic'. Это тестовые фикстуры, "
                "числа ниже НЕ результаты эксперимента.\n")


def is_synthetic(*video_ids) -> bool:
    return any("synthetic" in str(v).lower() for v in video_ids if v)


def read_csv(path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, header: list, rows: list) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def append_csv_row(path, header: list, row: list) -> None:
    path = Path(path)
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def md_table(headers: list, rows: list) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "n/a" if v != v else f"{v:.3f}"
        return "" if v is None else str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def majority(values: list):
    """Most common non-empty value; None on empty input or a tie for first place."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return None
    top = Counter(vals).most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def people_key(row) -> tuple:
    return (row["video_id"], row["snapshot_id"], int(row["person_idx"]))


def load_gold_people(path, video_id=None, exclude_visible=()) -> tuple:
    """Gold CSV -> (consensus dict, info).
    consensus[key] = {color, color_2, type, visible, box_h, labelers}; a field with a tie is None.
    info = {"n_objects", "n_labelers", "n_color_ties", "n_type_ties"}"""
    by_key = defaultdict(list)
    for r in read_csv(path):
        if video_id and r["video_id"] != video_id:
            continue
        by_key[people_key(r)].append(r)
    gold, color_ties, type_ties, labelers = {}, 0, 0, set()
    for k, rows in by_key.items():
        labelers.update(r["labeler"] for r in rows)
        vis = majority([r.get("visible") for r in rows])
        if vis in exclude_visible:
            continue
        color, typ = majority([r["color"] for r in rows]), majority([r["type"] for r in rows])
        color_ties += color is None
        type_ties += typ is None
        hs = [h for h in (_to_float(r.get("box_h")) for r in rows) if h is not None]
        gold[k] = {"color": color, "color_2": majority([r.get("color_2") for r in rows]), "type": typ,
                   "visible": vis, "box_h": hs[0] if hs else None,
                   "labelers": sorted({r["labeler"] for r in rows})}
    return gold, {"n_objects": len(gold), "n_labelers": len(labelers),
                  "n_color_ties": color_ties, "n_type_ties": type_ties}


def load_gold_frames(path, video_id=None) -> tuple:
    by_key = defaultdict(list)
    for r in read_csv(path):
        if video_id and r["video_id"] != video_id:
            continue
        by_key[(r["video_id"], r["snapshot_id"])].append(r)
    gold, ties = {}, 0
    for k, rows in by_key.items():
        v = majority([r["true_people_count"] for r in rows])
        if v is None:
            ties += 1
            continue
        gold[k] = int(float(v))
    return gold, {"n_frames": len(gold), "n_ties": ties}


def load_snapshots(path) -> tuple:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("video_id", "unknown_video"), data.get("snapshots", []), data


def find_frame(frame_path: str, frames_dir, json_dir):
    rel = Path(str(frame_path).replace("\\", "/"))
    for cand in (Path(frames_dir) / rel, Path(frames_dir) / rel.name, Path(json_dir) / rel):
        if cand.is_file():
            return cand
    return None


def load_crops_index(path, video_id: str) -> dict:
    """crops_index.json from Arai -> {(video_id, snapshot_id, person_idx): {"crop_path", "box_h"}}.
    FORMAT ASSUMED (NOT VERIFIED, the file does not exist yet): a list, or {"crops": [...]}, of objects with
    snapshot_id, person_idx, crop_path (or path) and optionally box_h / video_id. Relative paths are
    resolved against the index file's folder."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("crops", []) if isinstance(data, dict) else data
    out = {}
    for it in items:
        cp = it.get("crop_path") or it.get("path")
        if cp is None or "snapshot_id" not in it or "person_idx" not in it:
            continue
        cp = Path(cp)
        out[(it.get("video_id", video_id), it["snapshot_id"], int(it["person_idx"]))] = {
            "crop_path": str(cp if cp.is_absolute() else p.parent / cp), "box_h": it.get("box_h")}
    return out


def new_run_id(prefix="run") -> str:
    return f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def append_run(record: dict, results_dir=None) -> None:
    """One line per run in eval/results/runs.jsonl (material for the `runs` table of the database).
    Keys: run_id, video_id, method, model, mode, prompt_variant, weights, started_at, duration_sec, params."""
    keys = ("run_id", "video_id", "method", "model", "mode", "prompt_variant", "weights", "started_at",
            "duration_sec", "params")
    rec = {k: record.get(k) for k in keys}
    d = Path(results_dir) if results_dir else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
