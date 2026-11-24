"""Server-side runner: ONLY requests to the model -> JSONL predictions. Metrics are computed later, on the Mac.
Needs only the files in vlm/ (no eval/), the standard library and OpenCV or Pillow (imgutil).

  crops  R1, one request per box of every snapshot -> predictions for eval/bench_clothing.py
      python3 run_predictions.py crops --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames \
          --model qwen3-vl-4b --variant A --out ~/work/runs/pred_clothing_qwen3-vl-4b_R1_A.jsonl [--repeat 2]
  count  R2 / R3 / R4, one request per frame -> counts for eval/bench_count.py
      python3 run_predictions.py count --mode R2 --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames \
          --model qwen3-vl-4b --out ~/work/runs/count_qwen3-vl-4b_R2.jsonl

crops line : {"video_id","snapshot_id","person_idx","box_h","record":{contract}|null,"seconds","status","error"}
count line : {"video_id","snapshot_id","mode","method","model","people_count"|null,"seconds","status","error"}
--repeat 2 (crops) writes the second pass to <out stem>_repeat2.jsonl (for the repeatability metric).
Every run appends one line to runs.jsonl next to --out (run_id, video_id, method, model, mode, prompt_variant,
weights, started_at, duration_sec, params). NOT VERIFIED on the real server.
"""
import argparse
import datetime
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgutil  # noqa: E402
from describers.vlm_api import VlmApiDescriber  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8765/v1/chat/completions"


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def find_frame(frame_path, frames_dir):
    rel = Path(str(frame_path).replace("\\", "/"))
    for cand in (Path(frames_dir) / rel, Path(frames_dir) / rel.name):
        if cand.is_file():
            return cand
    return None


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_run(out, run):
    with open(Path(out).parent / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run, ensure_ascii=False) + "\n")


def build_items(snaps, video_id, frames_dir, pad, min_side, min_h):
    items, rows_fail = [], []
    for s in snaps:
        fp = find_frame(s.get("frame_path", ""), frames_dir)
        img = imgutil.load_image(str(fp)) if fp else None
        for idx, box in enumerate(s.get("boxes") or []):
            key = {"video_id": video_id, "snapshot_id": s["snapshot_id"], "person_idx": idx}
            try:
                box_h = round(float(box[3]) - float(box[1]), 1)
            except (TypeError, ValueError, IndexError):
                box_h = None
            err = None
            if img is None:
                err = f"frame not found: {s.get('frame_path')!r}"
            elif box_h is None or box_h < min_h:
                err = "box too small / invalid (not sent)"
            url = None if err else imgutil.crop_data_url(img, box, pad, min_side)
            if not err and url is None:
                err = "invalid box"
            if err:
                rows_fail.append({**key, "box_h": box_h, "record": None, "seconds": None, "status": "skipped", "error": err})
            else:
                items.append({**key, "box_h": box_h, "url": url})
    return items, rows_fail


def one_pass(describer, items, workers):
    done = [0]

    def one(it):
        r = describer.describe_crop(it["url"], it["person_idx"])
        done[0] += 1
        if done[0] % 25 == 0:
            log(f"  {done[0]}/{len(items)}")
        return {"video_id": it["video_id"], "snapshot_id": it["snapshot_id"], "person_idx": it["person_idx"],
                "box_h": it["box_h"], "record": r["record"], "seconds": r["seconds"], "status": r["status"],
                "error": r["error"]}
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, items))
    return [one(i) for i in items]


def cmd_crops(a, snaps, video_id):
    run_id, started, t0 = f"clo_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}", \
        datetime.datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    items, fails = build_items(snaps, video_id, a.frames_dir, a.crop_pad, a.min_crop_side, a.min_box_height)
    log(f"{len(items)} crops to send, {len(fails)} not sent")
    d = VlmApiDescriber(a.model, a.api_url, "R1", a.variant, a.disable_thinking, a.timeout)
    for n in range(1, a.repeat + 1):
        rows = fails + one_pass(d, items, a.workers)
        rows.sort(key=lambda r: (r["snapshot_id"], r["person_idx"]))
        out = Path(a.out) if n == 1 else Path(a.out).with_name(Path(a.out).stem + f"_repeat{n}.jsonl")
        write_jsonl(out, rows)
        log(f"pass {n}: wrote {out} ({sum(1 for r in rows if r['status'] == 'ok')} ok, "
            f"{sum(1 for r in rows if r['status'] != 'ok')} not ok)")
    append_run(a.out, {"run_id": run_id, "video_id": video_id, "method": d.method, "model": a.model, "mode": "crop",
                       "prompt_variant": a.variant, "weights": None, "started_at": started,
                       "duration_sec": round(time.perf_counter() - t0, 2),
                       "params": {"kind": "clothing_predictions", "api_url": a.api_url, "repeat": a.repeat,
                                  "workers": a.workers, "disable_thinking": a.disable_thinking, "out": str(a.out)}})


def cmd_count(a, snaps, video_id):
    run_id, started, t0 = f"cnt_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}", \
        datetime.datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    d = VlmApiDescriber(a.model, a.api_url, a.mode, "A", a.disable_thinking, a.timeout)

    def one(s):
        base = {"video_id": video_id, "snapshot_id": s["snapshot_id"], "mode": d.mode, "method": d.method, "model": a.model}
        fp = find_frame(s.get("frame_path", ""), a.frames_dir)
        if fp is None:
            return {**base, "people_count": None, "seconds": None, "status": "error",
                    "error": f"frame not found: {s.get('frame_path')!r}"}
        r = d.describe_frame(str(fp), s.get("boxes") or [])
        ok = r["status"] == "ok" and r["people_count"] is not None
        return {**base, "people_count": r["people_count"] if ok else None, "seconds": r["seconds"],
                "status": "ok" if ok else "error", "error": r["error"] or (None if ok else "no people_count in the answer")}
    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, snaps))
    else:
        rows = [one(s) for s in snaps]
    write_jsonl(a.out, rows)
    log(f"wrote {a.out} ({sum(1 for r in rows if r['status'] == 'ok')}/{len(rows)} ok)")
    append_run(a.out, {"run_id": run_id, "video_id": video_id, "method": d.method, "model": a.model, "mode": d.mode,
                       "prompt_variant": "A", "weights": None, "started_at": started,
                       "duration_sec": round(time.perf_counter() - t0, 2),
                       "params": {"kind": "count_predictions", "api_url": a.api_url, "workers": a.workers,
                                  "disable_thinking": a.disable_thinking, "out": str(a.out)}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["crops", "count"])
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--model", required=True, help="= --served-model-name of the running vLLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--api-url", default=DEFAULT_URL)
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "D"], help="crops only")
    ap.add_argument("--mode", default="R2", choices=["R2", "R3", "R4"], help="count only")
    ap.add_argument("--repeat", type=int, default=1, help="crops only: 2 = second pass for repeatability")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--crop-pad", type=float, default=0.12)
    ap.add_argument("--min-crop-side", type=int, default=336)
    ap.add_argument("--min-box-height", type=float, default=0, help="0 = send every box (the gold decides what is scored)")
    a = ap.parse_args()
    data = json.loads(Path(a.snapshots).read_text(encoding="utf-8"))
    video_id, snaps = data.get("video_id", "unknown_video"), data.get("snapshots", [])
    (cmd_crops if a.kind == "crops" else cmd_count)(a, snaps, video_id)


if __name__ == "__main__":
    main()
