"""Server-side runner: ONLY requests to the model -> JSONL predictions. Metrics are computed later, on the Mac.
Needs only the files in vlm/ (no eval/), the standard library and OpenCV or Pillow (imgutil).

  crops  R1, one request per box of every snapshot -> predictions for eval/bench_clothing.py
      python3 run_predictions.py crops --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames \
          --model qwen3-vl-4b --variant A --out ~/work/runs/pred_clothing_qwen3-vl-4b_R1_A.jsonl [--repeat 2]
  count  R2 / R3 / R4, one request per frame -> counts for eval/bench_count.py
      python3 run_predictions.py count --mode R2 --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames \
          --model qwen3-vl-4b --out ~/work/runs/count_qwen3-vl-4b_R2.jsonl

Index mode (Arai's cv/crop_people.py output; one crops_index.json / frames_index.json per video):
  crops  --crops-index v1/crops_index.json v2/crops_index.json [--crops-root v1 v2]
         only kept=true; the ready crop <crops-root>/full/<crop_id>.jpg is sent (no new cropping).
         the file is <crops-root>/full/<crop_id>.jpg or <crops-root>/<crop_id>.jpg (Arai's video1/ layout).
         --crops-root defaults to the folder of each index; if given, one per index (same order).
         snapshot_id and crop_id repeat in every video: the key is always (video_id, snapshot_id, person_idx).
  count  --frames-index v1/frames_index.json v2/frames_index.json --frames-dir F1 F2 [--crops-index ...]
         frame = <frames-dir>/<frame_path>, one --frames-dir per frames index (same order). A video whose frames folder
         does not exist is skipped with a message (ПРОПУСК ...); the run stops only if no video is left.
         R4 needs --crops-index: only kept=true boxes are put in the prompt (box_person_idx in the line).
Few-shot, prompt variant C (crops only):
  crops --variant C --fewshot-examples eval/gold_real/fewshot_examples.json [--limit-images 4]
         the request is: 3 x (example crop + its correct JSON answer as the assistant turn), then the target crop
         (vlm/describers/vlm_api.py). Examples: list or {"examples": [...]}, each with video_id, snapshot_id,
         person_idx, color, type (+ crop_path, description). The image is looked up in --crops-index by the key first
         (works on the server), then crop_path (absolute, or relative to the json / current folder / repo root).
         Examples are upscaled like the target crops. examples + 1 must fit --limit-images (= LIMIT_IMAGES of the
         vLLM server, vlm/server/_serve_common.sh: start it with LIMIT_IMAGES=4). The example crops are predicted
         like all others: drop them from scoring with score_methods --exclude-file.
Index lines are deduplicated on (video_id, snapshot_id, person_idx) / (video_id, snapshot_id), as eval/run_method.py.

crops line : {"video_id","snapshot_id","person_idx","crop_id","box","box_h","record":{contract}|null,"seconds",
              "status","error","raw_response","prompt_tokens","completion_tokens","finish_reason","suspicious":false}
flat crops line (Arai's results/<method>__crop__<variant>.jsonl, read by `python -m eval.score_methods people`),
written to --out-flat (default <out folder>/flat/<out stem>.jsonl, so summarize_runs does not see it):
             {"video_id","snapshot_id","person_idx","crop_id","box_h","box","sec","color","color_2","type",
              "confidence","description","method","mode","prompt_variant","status","finish_reason","error","raw_response"}
             a line that is NOT ok (truncated, error, skipped) has color = type = "unknown": score_methods then
             counts it as a miss (it skips only people with no line at all) and it shows in the unknown share.
count line : {"video_id","snapshot_id","mode","method","model","people_count"|null,"seconds","status","error",
              "raw_response","prompt_tokens","completion_tokens","finish_reason","suspicious"}
status: ok | truncated (finish_reason == "length": answer cut at max_tokens, result null, nothing repaired)
        | error | skipped (crops: box not sent).
raw_response is the model text cut to 2000 characters. suspicious (count only): status ok, people_count == 0,
but the frame has boxes that pass --min-box-height (snapshots.json; index mode: kept boxes of --crops-index,
or n_kept of frames_index.json when --crops-index is not given). Always false for crops.
--frequency-penalty / --repetition-penalty / --max-tokens are NOT sent unless given (old runs stay reproducible);
their values (or null) go to params in runs.jsonl.
--repeat 2 (crops) writes the second pass to <out stem>_repeat2.jsonl (for the repeatability metric).
Every run appends one line to runs.jsonl next to --out (run_id, video_id, method, model, mode, prompt_variant,
weights, started_at, duration_sec, params). NOT VERIFIED on the real server.
"""
import argparse
import datetime
import glob
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
RAW_LIMIT = 2000
NO_ANSWER = {"raw_response": "", "prompt_tokens": None, "completion_tokens": None, "finish_reason": None}


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


def answer_fields(r):
    """raw_response (cut to RAW_LIMIT chars), tokens and finish_reason from a describer result."""
    return {"raw_response": (r.get("raw_response") or "")[:RAW_LIMIT], "prompt_tokens": r.get("prompt_tokens"),
            "completion_tokens": r.get("completion_tokens"), "finish_reason": r.get("finish_reason")}


def box_height(box):
    try:
        return round(float(box[3]) - float(box[1]), 1)
    except (TypeError, ValueError, IndexError):
        return None


def sampling_params(a):
    return {"frequency_penalty": a.frequency_penalty, "repetition_penalty": a.repetition_penalty,
            "max_tokens": a.max_tokens}


KEY_FIELDS = ("video_id", "snapshot_id", "person_idx", "crop_id", "box", "box_h")
CONTRACT = ("color", "color_2", "type", "confidence", "description", "method", "mode", "prompt_variant")


def expand(paths):
    """Paths or masks (in quotes) -> sorted list of files; a mask that matches nothing stays as it is."""
    out = []
    for p in paths or []:
        hit = sorted(glob.glob(str(p)))
        out += hit if hit else [str(p)]
    return out


def pick(values, i, n, what):
    """One value per index file. A single value for several files is refused: snapshot_id / crop_id repeat in every
    video (sn_0000_p00 exists in both), so one shared folder would mix the videos up."""
    if not values:
        return None
    if len(values) == n:
        return values[i]
    if n > 1 and len(values) == 1:
        raise SystemExit(f"{what}: {n} index files but one folder. sn_XXXX names repeat in every video, "
                         f"give one folder per index file, in the same order")
    raise SystemExit(f"{what}: give one value per index file ({n}), got {len(values)}")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no file {path}")


def load_crops_index(paths, roots):
    """All entries of the crops_index.json files (kept or not), first one wins on a repeated key.
    Every entry gets _root: the folder with full/<crop_id>.jpg."""
    files = expand(paths)
    out, seen = [], set()
    for i, p in enumerate(files):
        root = pick(roots, i, len(files), "--crops-root") or str(Path(p).parent)
        for c in read_json(p):
            key = (c["video_id"], c["snapshot_id"], int(c["person_idx"]))
            if key not in seen:
                seen.add(key)
                out.append({**c, "_root": root})
    return out


def find_crop(c):
    root = Path(c["_root"])
    cands = [root / "full" / f"{c['crop_id']}.jpg", root / f"{c['crop_id']}.jpg"]
    if c.get("full_path"):
        fp = Path(str(c["full_path"]).replace("\\", "/"))
        cands += [fp, root / "full" / fp.name]
    return next((p for p in cands if p.is_file()), None)


def flat_row(r, method, variant):
    """Our crops line -> Arai's flat line. Not ok -> unknown answer (a miss for score_methods)."""
    rec = r.get("record") if r.get("status") == "ok" and isinstance(r.get("record"), dict) else None
    ans = {k: rec.get(k) for k in CONTRACT} if rec else {
        "color": "unknown", "color_2": None, "type": "unknown", "confidence": None, "description": "",
        "method": method, "mode": "crop", "prompt_variant": variant}
    return {**{k: r.get(k) for k in KEY_FIELDS}, "sec": r.get("seconds"), **ans, "status": r.get("status"),
            "finish_reason": r.get("finish_reason"), "error": r.get("error"), "raw_response": r.get("raw_response")}


def make_describer(a, mode, variant):
    return VlmApiDescriber(a.model, a.api_url, mode, variant, a.disable_thinking, a.timeout,
                           fewshot_path=getattr(a, "fewshot", None), **sampling_params(a))


def load_fewshot_examples(path, entries, min_side, limit_images):
    """--fewshot-examples -> ({"examples": [{image: data URL, clothing_color, clothing_type, description}]}, keys).
    Stops with a message if an example is incomplete, its crop is not found or the image limit is exceeded."""
    data = read_json(path)
    items = data.get("examples", []) if isinstance(data, dict) else data
    if not items:
        raise SystemExit(f"{path}: no examples")
    if len(items) + 1 > limit_images:
        raise SystemExit(f"{path}: {len(items)} examples + the target crop = {len(items) + 1} images per request, "
                         f"but --limit-images is {limit_images} (vLLM --limit-mm-per-prompt, LIMIT_IMAGES)")
    by_key = {(c["video_id"], c["snapshot_id"], int(c["person_idx"])): c for c in entries or []}
    repo = Path(__file__).resolve().parents[1]
    out, keys = [], []
    for i, e in enumerate(items):
        miss = [k for k in ("video_id", "snapshot_id", "person_idx", "color", "type") if e.get(k) in (None, "")]
        if miss:
            raise SystemExit(f"{path}: example {i} has no {', '.join(miss)}")
        key = (e["video_id"], str(e["snapshot_id"]), int(e["person_idx"]))
        fp = find_crop(by_key[key]) if key in by_key else None
        if fp is None and e.get("crop_path"):
            cp = Path(str(e["crop_path"]).replace("\\", "/"))
            cands = [cp] if cp.is_absolute() else [Path(path).parent / cp, Path.cwd() / cp, repo / cp]
            fp = next((c for c in cands if c.is_file()), None)
        if fp is None:
            raise SystemExit(f"{path}: example {i} {key}: crop not found (not in --crops-index, crop_path "
                             f"{e.get('crop_path')!r} does not exist)")
        img = imgutil.load_image(str(fp))
        w, h = imgutil.image_size(img)
        url = imgutil.crop_data_url(img, [0, 0, w, h], 0.0, min_side)
        if url is None:
            raise SystemExit(f"{path}: example {i}: empty crop file {fp}")
        out.append({"image": url, "clothing_color": e["color"], "clothing_type": e["type"],
                    "description": e.get("description") or f"Person wearing a {e['color']} {e['type']}."})
        keys.append(list(key))
        log(f"few-shot example {i}: {key} {e['color']} {e['type']} <- {fp}")
    return {"examples": out}, keys


def build_items(snaps, video_id, frames_dir, pad, min_side, min_h):
    items, rows_fail = [], []
    for s in snaps:
        fp = find_frame(s.get("frame_path", ""), frames_dir)
        img = imgutil.load_image(str(fp)) if fp else None
        for idx, box in enumerate(s.get("boxes") or []):
            key = {"video_id": video_id, "snapshot_id": s["snapshot_id"], "person_idx": idx,
                   "crop_id": f"{s['snapshot_id']}_p{idx:02d}", "box": box}
            box_h = box_height(box)
            err = None
            if img is None:
                err = f"frame not found: {s.get('frame_path')!r}"
            elif box_h is None or box_h < min_h:
                err = "box too small / invalid (not sent)"
            url = None if err else imgutil.crop_data_url(img, box, pad, min_side)
            if not err and url is None:
                err = "invalid box"
            if err:
                rows_fail.append({**key, "box_h": box_h, "record": None, "seconds": None, "status": "skipped", "error": err,
                                  **NO_ANSWER, "suspicious": False})
            else:
                items.append({**key, "box_h": box_h, "url": url})
    return items, rows_fail


def build_items_index(entries, min_side, min_h):
    """Index mode: only kept=true, the ready crop file is read as it is (upscaled like crop_data_url)."""
    items, rows_fail = [], []
    for c in entries:
        if not c.get("kept"):
            continue
        box = c.get("box")
        key = {"video_id": c["video_id"], "snapshot_id": c["snapshot_id"], "person_idx": int(c["person_idx"]),
               "crop_id": c.get("crop_id"), "box": box}
        box_h = c.get("box_h") if c.get("box_h") is not None else box_height(box)
        fp, err, url = find_crop(c), None, None
        if fp is None:
            err = f"crop file not found: {c.get('crop_id')}.jpg in {c['_root']}/full"
        elif box_h is None or box_h < min_h:
            err = "box too small / invalid (not sent)"
        else:
            try:
                img = imgutil.load_image(str(fp))
                w, h = imgutil.image_size(img)
                url = imgutil.crop_data_url(img, [0, 0, w, h], 0.0, min_side)
            except OSError as e:
                err = f"crop not readable: {e}"
            if not err and url is None:
                err = "empty crop file"
        if err:
            rows_fail.append({**key, "box_h": box_h, "record": None, "seconds": None, "status": "skipped",
                              "error": err, **NO_ANSWER, "suspicious": False})
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
        return {**{k: it[k] for k in KEY_FIELDS}, "record": r["record"], "seconds": r["seconds"], "status": r["status"],
                "error": r["error"], **answer_fields(r), "suspicious": False}
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, items))
    return [one(i) for i in items]


def run_lines(run_id, started, t0, method, model, mode, variant, video_ids, params):
    """One runs.jsonl line per video (as eval/run_method.py writes one runs.json entry per video)."""
    dur = round(time.perf_counter() - t0, 2)
    return [{"run_id": run_id, "video_id": v, "method": method, "model": model, "mode": mode,
             "prompt_variant": variant, "weights": None, "started_at": started, "duration_sec": dur,
             "params": params} for v in video_ids]


def flat_path(a, out):
    """Flat file next to `out`: --out-flat for the first pass, <out folder>/flat/<stem>.jsonl by default."""
    if a.out_flat and Path(out) == Path(a.out):
        return Path(a.out_flat)
    if a.out_flat:
        return Path(a.out_flat).with_name(Path(a.out_flat).stem + Path(out).stem[len(Path(a.out).stem):] + ".jsonl")
    return Path(out).parent / "flat" / Path(out).name


def cmd_crops(a, items, fails, video_ids, source):
    run_id, started, t0 = f"clo_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}", \
        datetime.datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    log(f"{len(items)} crops to send, {len(fails)} not sent")
    d = make_describer(a, "R1", a.variant)
    for n in range(1, a.repeat + 1):
        rows = fails + one_pass(d, items, a.workers)
        rows.sort(key=lambda r: (r["video_id"], r["snapshot_id"], r["person_idx"]))
        out = Path(a.out) if n == 1 else Path(a.out).with_name(Path(a.out).stem + f"_repeat{n}.jsonl")
        write_jsonl(out, rows)
        fout = flat_path(a, out)
        write_jsonl(fout, [flat_row(r, d.method, a.variant) for r in rows])
        log(f"pass {n}: wrote {out} and {fout} ({sum(1 for r in rows if r['status'] == 'ok')} ok, "
            f"{sum(1 for r in rows if r['status'] != 'ok')} not ok)")
    params = {"kind": "clothing_predictions", "source": source, "api_url": a.api_url, "repeat": a.repeat,
              "workers": a.workers, "disable_thinking": a.disable_thinking, "out": str(a.out),
              "out_flat": str(flat_path(a, a.out)), **sampling_params(a)}
    if a.variant == "C":
        params.update(fewshot_examples=a.fewshot_examples, fewshot_keys=getattr(a, "fewshot_keys", None))
    for run in run_lines(run_id, started, t0, d.method, a.model, "crop", a.variant, video_ids, params):
        append_run(a.out, run)


def cmd_count(a, frames, video_ids, source):
    """frames: [{video_id, snapshot_id, frame_path, boxes, _frames_dir, n_boxes|None, box_person_idx|None}]."""
    run_id, started, t0 = f"cnt_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}", \
        datetime.datetime.now().isoformat(timespec="seconds"), time.perf_counter()
    d = make_describer(a, a.mode, "A")

    def one(s):
        base = {"video_id": s["video_id"], "snapshot_id": s["snapshot_id"], "mode": d.mode, "method": d.method,
                "model": a.model}
        if s.get("box_person_idx") is not None and a.mode == "R4":
            base["box_person_idx"] = s["box_person_idx"]
        fp = find_frame(s.get("frame_path", ""), s["_frames_dir"])
        if fp is None:
            return {**base, "people_count": None, "seconds": None, "status": "error",
                    "error": f"frame not found: {s.get('frame_path')!r}", **NO_ANSWER, "suspicious": False}
        r = d.describe_frame(str(fp), s.get("boxes") or [])
        ok = r["status"] == "ok" and r["people_count"] is not None
        status = "ok" if ok else ("truncated" if r["status"] == "truncated" else "error")
        n_boxes = s.get("n_boxes")
        if n_boxes is None:
            n_boxes = sum(1 for b in (s.get("boxes") or []) if (h := box_height(b)) is not None and h >= a.min_box_height)
        return {**base, "people_count": r["people_count"] if ok else None, "seconds": r["seconds"], "status": status,
                "error": r["error"] or (None if ok else "no people_count in the answer"), **answer_fields(r),
                "suspicious": ok and r["people_count"] == 0 and n_boxes > 0}
    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(one, frames))
    else:
        rows = [one(s) for s in frames]
    write_jsonl(a.out, rows)
    log(f"wrote {a.out} ({sum(1 for r in rows if r['status'] == 'ok')}/{len(rows)} ok)")
    params = {"kind": "count_predictions", "source": source, "api_url": a.api_url, "workers": a.workers,
              "disable_thinking": a.disable_thinking, "min_box_height": a.min_box_height, "out": str(a.out),
              **sampling_params(a)}
    for run in run_lines(run_id, started, t0, d.method, a.model, d.mode, "A", video_ids, params):
        append_run(a.out, run)


def frames_from_index(a):
    """frames_index.json (+ kept boxes from --crops-index) -> frame dicts for cmd_count."""
    files = expand(a.frames_index)
    kept = {}
    if a.crops_index:
        for c in load_crops_index(a.crops_index, a.crops_root):
            if c.get("kept"):
                kept.setdefault((c["video_id"], c["snapshot_id"]), []).append(c)
    frames, seen = [], set()
    for i, p in enumerate(files):
        fdir = pick(a.frames_dir, i, len(files), "--frames-dir")
        rows = read_json(p)
        if not Path(fdir).is_dir():
            vids = sorted({fr["video_id"] for fr in rows})
            log(f"ПРОПУСК {', '.join(vids)} ({p}): нет папки с кадрами {fdir}. {a.mode} для этого видео не запускается; "
                f"положи кадры (frame_path из индекса, например {rows[0].get('frame_path') if rows else '?'}) и повтори.")
            continue
        for fr in rows:
            key = (fr["video_id"], fr["snapshot_id"])
            if key in seen:
                continue
            seen.add(key)
            f = {"video_id": fr["video_id"], "snapshot_id": fr["snapshot_id"], "frame_path": fr.get("frame_path", ""),
                 "_frames_dir": fdir, "boxes": [], "n_boxes": None, "box_person_idx": None}
            if a.crops_index:
                cs = sorted(kept.get(key, []), key=lambda c: int(c["person_idx"]))
                f["boxes"] = [c["box"] for c in cs]
                f["box_person_idx"] = [int(c["person_idx"]) for c in cs]
                f["n_boxes"] = sum(1 for c in cs if (c.get("box_h") or box_height(c["box"]) or 0) >= a.min_box_height)
            else:
                f["n_boxes"] = fr.get("n_kept")
            frames.append(f)
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["crops", "count"])
    ap.add_argument("--snapshots", help="old mode: snapshots.json with boxes (cropping done here)")
    ap.add_argument("--crops-index", nargs="+", help="index mode: crops_index.json of cv/crop_people.py, one per video")
    ap.add_argument("--crops-root", nargs="+", help="folder with <crop_id>.jpg or full/<crop_id>.jpg, one per index (default: folder of each index)")
    ap.add_argument("--frames-index", nargs="+", help="count, index mode: frames_index.json, one per video")
    ap.add_argument("--frames-dir", nargs="+", help="frames folder: one for --snapshots, one per --frames-index file")
    ap.add_argument("--model", required=True, help="= --served-model-name of the running vLLM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-flat", help="crops: Arai's flat format (default <out folder>/flat/<out stem>.jsonl)")
    ap.add_argument("--api-url", default=DEFAULT_URL)
    ap.add_argument("--variant", default="A", choices=["A", "B", "C", "D"], help="crops only")
    ap.add_argument("--fewshot-examples", help="crops, --variant C: json with the few-shot examples "
                                                "(eval/gold_real/fewshot_examples.json)")
    ap.add_argument("--limit-images", type=int, default=4,
                    help="images allowed per request by the server (LIMIT_IMAGES / --limit-mm-per-prompt); "
                         "variant C needs examples + 1")
    ap.add_argument("--mode", default="R2", choices=["R2", "R3", "R4"], help="count only")
    ap.add_argument("--repeat", type=int, default=1, help="crops only: 2 = second pass for repeatability")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--crop-pad", type=float, default=0.12, help="old mode only (index crops already have it)")
    ap.add_argument("--min-crop-side", type=int, default=336)
    ap.add_argument("--min-box-height", type=float, default=0,
                    help="crops: 0 = send every box (the gold decides what is scored); count: boxes counted for 'suspicious'")
    ap.add_argument("--frequency-penalty", type=float, default=None, help="not sent unless given")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="not sent unless given (vLLM extension)")
    ap.add_argument("--max-tokens", type=int, default=None, help="not sent unless given: replaces the per-mode limit")
    a = ap.parse_args(argv)

    if a.fewshot_examples and (a.kind != "crops" or a.variant != "C"):
        ap.error("--fewshot-examples is for crops --variant C only")
    if a.kind == "crops":
        if bool(a.snapshots) == bool(a.crops_index):
            ap.error("crops: give --snapshots (old mode) OR --crops-index (index mode)")
        if a.variant == "C" and a.fewshot_examples:
            entries_fs = load_crops_index(a.crops_index, a.crops_root) if a.crops_index else []
            a.fewshot, a.fewshot_keys = load_fewshot_examples(a.fewshot_examples, entries_fs, a.min_crop_side,
                                                              a.limit_images)
        if a.snapshots:
            if not a.frames_dir or len(a.frames_dir) != 1:
                ap.error("--snapshots needs one --frames-dir")
            data = read_json(a.snapshots)
            video_id = data.get("video_id", "unknown_video")
            items, fails = build_items(data.get("snapshots", []), video_id, a.frames_dir[0], a.crop_pad,
                                       a.min_crop_side, a.min_box_height)
            return cmd_crops(a, items, fails, [video_id], "snapshots")
        entries = load_crops_index(a.crops_index, a.crops_root)
        if not any(c.get("kept") for c in entries):
            raise SystemExit("no kept=true crops in --crops-index")
        items, fails = build_items_index(entries, a.min_crop_side, a.min_box_height)
        videos = sorted({c["video_id"] for c in entries if c.get("kept")})
        return cmd_crops(a, items, fails, videos, "crops_index")

    if bool(a.snapshots) == bool(a.frames_index):
        ap.error("count: give --snapshots (old mode) OR --frames-index (index mode)")
    if not a.frames_dir:
        ap.error("count needs --frames-dir")
    if a.snapshots:
        if len(a.frames_dir) != 1:
            ap.error("--snapshots needs one --frames-dir")
        data = read_json(a.snapshots)
        video_id = data.get("video_id", "unknown_video")
        frames = [{"video_id": video_id, "snapshot_id": s["snapshot_id"], "frame_path": s.get("frame_path", ""),
                   "boxes": s.get("boxes") or [], "_frames_dir": a.frames_dir[0]} for s in data.get("snapshots", [])]
        return cmd_count(a, frames, [video_id], "snapshots")
    if a.mode == "R4" and not a.crops_index:
        ap.error("R4 with --frames-index needs --crops-index (the kept=true boxes go into the prompt)")
    frames = frames_from_index(a)
    if not frames:
        raise SystemExit("нет ни одного видео с папкой кадров: запускать нечего (см. ПРОПУСК выше)")
    return cmd_count(a, frames, sorted({f["video_id"] for f in frames}), "frames_index")


if __name__ == "__main__":
    main()
