"""People-count benchmark: YOLO (people_count in snapshots.json) vs VLM modes R2/R3/R4 vs the gold standard.

    python -m eval.bench_count --gold eval/gold/gold_frames.csv --snapshots snapshots.json --frames-dir . \
        --model qwen3-vl-4b --api-url http://127.0.0.1:8765/v1/chat/completions --methods yolo,R2,R3,R4

yolo     the detector count already stored in snapshots.json (no request). --yolo-weights is only a label for runs.jsonl.
R2       whole frame, "how many people" (no detector at all)
R3       whole frame, "list the people" -> count = length of the list
R4       frame + the detector boxes as text; the model's OWN people_count is used (NOT the number of boxes)

Metrics per method over gold frames: MAE, exact-match share, undercount / overcount share, mean signed error
(prediction - gold; negative = misses people). A failed request is reported (n_failed), not scored.
Outputs (eval/results/): count_<run_id>.md / _frames.csv, rows appended to count_summary.csv, lines to runs.jsonl.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import metrics as M
from eval.bench_common import (RESULTS_DIR, SYNTH_BANNER, append_csv_row, append_run, find_frame, is_synthetic,
                               load_gold_frames, load_snapshots, md_table, new_run_id, now_iso, write_csv)

SUMMARY_HEADER = ["run_id", "video_id", "method", "model", "n", "n_failed", "mae", "exact", "under", "over", "bias",
                  "sec_per_frame", "synthetic"]


def yolo_counts(snaps: list) -> dict:
    return {s["snapshot_id"]: ({"count": s["people_count"], "seconds": None}
                               if isinstance(s.get("people_count"), int) else None) for s in snaps}


def vlm_counts(describer, snaps: list, frames_dir, json_dir, workers=1) -> dict:
    def one(s):
        fp = find_frame(s.get("frame_path", ""), frames_dir, json_dir)
        if fp is None:
            return s["snapshot_id"], None
        r = describer.describe_frame(str(fp), s.get("boxes") or [])
        if r["status"] != "ok" or r["people_count"] is None:
            return s["snapshot_id"], None
        return s["snapshot_id"], {"count": r["people_count"], "seconds": r["seconds"]}
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            return dict(ex.map(one, snaps))
    return dict(map(one, snaps))


def load_counts_file(path, video_id=None) -> dict:
    """JSONL from vlm/run_predictions.py count -> {snapshot_id: {"count", "seconds"} | None}, plus the method label.
    Only the lines of video_id are read (a file of the index mode holds several videos with the same sn_XXXX);
    without video_id a file with several videos is refused."""
    import json
    out, label, model, seen = {}, None, None, set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if video_id and r.get("video_id") != video_id:
                continue
            if not video_id:
                seen.add(r.get("video_id"))
                if len(seen) > 1:  # sn_XXXX repeat in every video: keyed by snapshot_id they would overwrite
                    raise SystemExit(f"{path}: several videos {sorted(map(str, seen))}; pass video_id")
            label, model = r.get("method"), r.get("model")
            ok = r.get("status") == "ok" and isinstance(r.get("people_count"), int)
            out[r["snapshot_id"]] = {"count": r["people_count"], "seconds": r.get("seconds")} if ok else None
    return out, label, model


def score_method(gold: dict, video_id: str, counts: dict) -> dict:
    ks = sorted(k for k in gold if k[0] == video_id)
    true = [gold[k] for k in ks]
    pred = [counts.get(k[1]) and counts[k[1]]["count"] for k in ks]
    pred = [p if p is not None else None for p in pred]
    m = M.count_metrics(true, pred)
    secs = [counts[k[1]]["seconds"] for k in ks if counts.get(k[1]) and counts[k[1]]["seconds"] is not None]
    m["sec_per_frame"] = sum(secs) / len(secs) if secs else float("nan")
    m["keys"], m["true"], m["pred"] = ks, true, pred
    return m


def report_md(video_id: str, results: dict, synthetic: bool) -> str:
    parts = [f"# Счёт людей · {video_id}\n"]
    if synthetic:
        parts.append(SYNTH_BANNER)
    parts.append(md_table(
        ["метод", "кадров", "без ответа", "MAE", "точное совпадение", "недосчёт", "пересчёт", "смещение", "сек/кадр"],
        [[name, m["n"], m["n_failed"], m["mae"], m["exact"], m["under"], m["over"], m["bias"], m["sec_per_frame"]]
         for name, m in results.items()]))
    parts.append("\nСмещение = среднее (метод − эталон); минус = метод теряет людей. "
                 "`yolo` — счёт детектора из snapshots.json, `R4` — собственный счёт модели, а не число рамок.")
    return "\n".join(parts) + "\n"


def run_count_benchmark(gold: dict, video_id: str, snaps: list, methods: list, describer_factory=None,
                        frames_dir=".", json_dir=".", workers=1, model=None, yolo_weights=None, out_dir=None,
                        precomputed=None) -> dict:
    """methods: subset of yolo,R2,R3,R4. describer_factory(mode_code) -> describer. Returns {name: metrics}.
    precomputed: {name: (counts, method_label, model)} from load_counts_file: no request is made and no runs.jsonl
    line is written (the server's run_predictions.py already wrote it)."""
    d = Path(out_dir) if out_dir else RESULTS_DIR
    results, rid_group = {}, new_run_id("cnt")
    for name in methods:
        t0, started = time.perf_counter(), now_iso()
        if precomputed and name in precomputed:
            counts, method_label, mdl = precomputed[name]
        elif name == "yolo":
            counts, method_label, mdl = yolo_counts(snaps), "yolo", None
        else:
            describer = describer_factory(name)
            counts = vlm_counts(describer, snaps, frames_dir, json_dir, workers)
            method_label, mdl = describer.method, describer.model
        results[name] = score_method(gold, video_id, counts)
        if precomputed and name in precomputed:
            continue
        append_run({"run_id": f"{rid_group}_{name}", "video_id": video_id, "method": method_label, "model": mdl,
                    "mode": {"yolo": "detector", "R2": "frame-count", "R3": "frame-list", "R4": "boxes"}[name],
                    "prompt_variant": None if name == "yolo" else "A",
                    "weights": yolo_weights if name == "yolo" else None, "started_at": started,
                    "duration_sec": round(time.perf_counter() - t0, 2),
                    "params": {"bench": "count", "workers": workers}}, d)
    synth = is_synthetic(video_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"count_{rid_group}.md").write_text(report_md(video_id, results, synth), encoding="utf-8")
    write_csv(d / f"count_{rid_group}_frames.csv", ["snapshot_id", "gold"] + list(results), [
        [k[1], t] + [results[n]["pred"][i] if results[n]["pred"][i] is not None else "" for n in results]
        for i, (k, t) in enumerate(zip(results[methods[0]]["keys"], results[methods[0]]["true"]))])
    for n, m in results.items():
        append_csv_row(d / "count_summary.csv", SUMMARY_HEADER,
                       [f"{rid_group}_{n}", video_id, n, model if n != "yolo" else "", m["n"], m["n_failed"], m["mae"],
                        m["exact"], m["under"], m["over"], m["bias"], m["sec_per_frame"], synth])
    results["_report"] = d / f"count_{rid_group}.md"
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--frames-dir", default=".")
    ap.add_argument("--methods", default=None, help="default: yolo,R2,R3,R4 (or yolo + every --counts name)")
    ap.add_argument("--counts", action="append", default=[], metavar="NAME=FILE",
                    help="ready counts from the server, e.g. R2=count_qwen3-vl-4b_R2.jsonl (repeatable; no model needed)")
    ap.add_argument("--model", default="qwen3-vl-4b")
    ap.add_argument("--api-url", default="http://127.0.0.1:8765/v1/chat/completions")
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    vid, snaps, _ = load_snapshots(a.snapshots)
    gold, info = load_gold_frames(a.gold, vid)
    if not gold:
        raise SystemExit(f"no gold frames for video_id {vid!r}")
    pre = {}
    for spec in a.counts:
        name, _, path = spec.partition("=")
        if name not in ("R2", "R3", "R4") or not path:
            raise SystemExit(f"--counts expects R2|R3|R4=FILE, got {spec!r}")
        pre[name] = load_counts_file(path, vid)
    methods = [m for m in (a.methods or ("yolo," + ",".join(pre) if pre else "yolo,R2,R3,R4")).split(",") if m]

    def factory(mode):
        from vlm.describers.vlm_api import VlmApiDescriber
        return VlmApiDescriber(a.model, a.api_url, mode, "A", a.disable_thinking)
    res = run_count_benchmark(
        gold, vid, snaps, methods, factory,
        a.frames_dir, Path(a.snapshots).parent, a.workers, a.model, a.yolo_weights,
        Path(a.out_dir) if a.out_dir else None, precomputed=pre)
    print(res["_report"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
