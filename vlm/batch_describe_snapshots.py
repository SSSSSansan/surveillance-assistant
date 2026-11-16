"""Fill people[] (clothing color / type / description) in snapshots.json with Qwen3-VL.

Usage:
    python vlm/batch_describe_snapshots.py SNAPSHOTS_JSON FRAMES_DIR OUTPUT_JSON
        [--mode {crops,frame}] [--only sn_0000,sn_0003] [--workers N]
        [--min-box-height 60] [--min-box-width 30] [--crop-pad 0.12] [--min-crop-side 336]
        [--api-url URL] [--model NAME] [--timeout SEC]

SNAPSHOTS_JSON  file from Arai (cv/sample_snapshots.py): snapshots with "boxes" (person boxes
                [x1,y1,x2,y2] in frame pixels, sorted left to right) and "people_count".
FRAMES_DIR      folder with the frames. A frame_path ("frames/sn_0000.jpg") is looked up as
                FRAMES_DIR/<frame_path>, then FRAMES_DIR/<file name>, then next to the input JSON.
OUTPUT_JSON     result. The input file is never overwritten.

Modes
  crops (default)  one request per box (crop = box + padding, inside the frame, tiny crops are
                   upscaled). person_idx = index of the box in "boxes"; every person gets
                   "box": [x1,y1,x2,y2]. A box smaller than --min-box-height/--min-box-width is
                   NOT sent: that person gets unknown/unknown. A failed crop (request or parse
                   error after one retry) also gets unknown/unknown, the other crops go on.
  frame            whole frame in one request; the model lists the people itself
                   (person_idx = its order; no "box"). Poor on crowded scenes.

Per snapshot
  vlm_status        "ok" | "partial" (some crops failed; those people are unknown) | "error"
  people            filled on ok/partial. On "error" the key is REMOVED: the loader
                    (database.snapshots_loader) leaves people of a snapshot without the "people"
                    key untouched, while "people": [] would erase them in the DB.
  people_count      Arai's detector count stays the main one. If it is empty (null),
                    len(people) is used. A different VLM count is logged.
  vlm_people_count  how many people the VLM side produced (crops: number of boxes; null on error)
  vlm_crops_skipped / vlm_crop_errors   (crops mode) indices of too-small boxes / failed crops
Other keys are kept as they are. Root gets "vlm_mode" and "vlm_model". Progress goes to stderr.
"""
import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imgutil
from describe_snapshot import (API_URL, MODEL, TIMEOUT_SEC, describe_person_crop,
                               describe_snapshot)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def find_frame(frame_path: str, frames_dir: Path, json_dir: Path):
    rel = Path(str(frame_path).replace("\\", "/"))
    for cand in (frames_dir / rel, frames_dir / rel.name, json_dir / rel):
        if cand.is_file():
            return cand
    return None


def _rounded(box):
    return [round(float(v), 1) for v in box]


def describe_boxes(frame: Path, boxes: list, a, tag: str) -> dict:
    """Crops mode for one snapshot. Returns people[], skipped[], errors[], warnings, tokens."""
    img = imgutil.load_image(str(frame))
    n = len(boxes)

    def one(idx):
        box = boxes[idx]
        label = f"{tag} #{idx} "
        base = {"person_idx": idx, "clothing_color": "unknown", "clothing_type": "unknown"}
        try:
            bw, bh = float(box[2]) - float(box[0]), float(box[3]) - float(box[1])
            box_out = _rounded(box)
        except (TypeError, ValueError, IndexError):
            return ({**base, "description": "invalid box", "box": box}, "error", ["invalid box"], 0, False)
        if bh < a.min_box_height or bw < a.min_box_width:
            return ({**base, "description": "too small to describe", "box": box_out},
                    "skipped", [], 0, False)
        url = imgutil.crop_data_url(img, box, a.crop_pad, a.min_crop_side)
        if url is None:
            return ({**base, "description": "box outside the frame", "box": box_out},
                    "error", ["box is empty after clamping to the frame"], 0, False)
        res = describe_person_crop(url, idx, a.api_url, a.model, a.timeout, label=label)
        toks = (res["prompt_tokens"] or 0) + (res["completion_tokens"] or 0)
        warns = [f"#{idx}: {w}" for w in res["warnings"]]
        if res["status"] != "ok":
            return ({**base, "description": "not described (VLM error)", "box": box_out},
                    "error", warns + [f"#{idx}: {res['error']} | raw: {res['raw_response']!r}"], toks, True)
        person = dict(res["person"])
        person["box"] = box_out
        return person, "ok", warns, toks, True

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        results = list(pool.map(one, range(n)))
    people = [r[0] for r in results]
    return {
        "people": people,
        "skipped": [i for i, r in enumerate(results) if r[1] == "skipped"],
        "errors": [i for i, r in enumerate(results) if r[1] == "error"],
        "sent": sum(1 for r in results if r[4]),
        "n_ok": sum(1 for r in results if r[1] == "ok"),
        "warnings": [w for r in results for w in r[2]],
        "tokens": sum(r[3] for r in results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots_json")
    parser.add_argument("frames_dir")
    parser.add_argument("output_json")
    parser.add_argument("--mode", choices=("crops", "frame"), default="crops")
    parser.add_argument("--only", default=None, help="comma-separated snapshot_ids to process; others are dropped from the output")
    parser.add_argument("--workers", type=int, default=1, help="parallel crop requests per snapshot")
    parser.add_argument("--min-box-height", type=float, default=60.0, help="px; smaller boxes are not sent")
    parser.add_argument("--min-box-width", type=float, default=30.0, help="px; smaller boxes are not sent")
    parser.add_argument("--crop-pad", type=float, default=0.12, help="padding around the box, fraction of its size")
    parser.add_argument("--min-crop-side", type=int, default=336, help="upscale crops whose longer side is below this (0 = never)")
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_SEC)
    a = parser.parse_args()

    in_path, out_path = Path(a.snapshots_json), Path(a.output_json)
    if in_path.resolve() == out_path.resolve():
        sys.exit("OUTPUT_JSON must differ from SNAPSHOTS_JSON (the input is never overwritten)")
    if a.mode == "crops" and imgutil.BACKEND is None:
        sys.exit("crops mode needs OpenCV or Pillow in this Python: pip install pillow  (or opencv-python-headless)")

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    result = copy.deepcopy(data)
    result["vlm_mode"], result["vlm_model"] = a.mode, a.model
    frames_dir, json_dir = Path(a.frames_dir), in_path.resolve().parent

    snaps = result.get("snapshots", [])
    if a.only:
        wanted = {s.strip() for s in a.only.split(",") if s.strip()}
        snaps = [s for s in snaps if s.get("snapshot_id") in wanted]
        missing = wanted - {s.get("snapshot_id") for s in snaps}
        if missing:
            log(f"WARN: --only ids not found in the file: {sorted(missing)}")
        if not snaps:
            sys.exit("--only matched no snapshots")
        result["snapshots"] = snaps
    log(f"Snapshots: {len(snaps)}  mode={a.mode}  model={a.model}  {a.api_url}  image backend={imgutil.BACKEND}")
    t0 = time.time()
    n_ok = n_part = n_err = n_mismatch = 0
    n_sent = n_crop_err = n_skipped = tokens = 0

    for i, snap in enumerate(snaps, 1):
        sid = snap.get("snapshot_id", f"#{i}")
        tag = f"[{i}/{len(snaps)}] {sid} t={snap.get('t')}"

        def mark_error(reason: str) -> None:
            nonlocal n_err
            snap["vlm_status"] = "error"
            snap["vlm_people_count"] = None
            snap.pop("people", None)  # see module docstring
            n_err += 1
            log(f"{tag} ERROR: {reason}")

        frame = find_frame(snap.get("frame_path", ""), frames_dir, json_dir)
        if frame is None:
            mark_error(f"frame not found: {snap.get('frame_path')!r} (looked in {frames_dir})")
            continue

        status = "ok"
        if a.mode == "frame":
            try:
                res = describe_snapshot(str(frame), a.api_url, a.model, a.timeout)
            except Exception as e:  # one bad frame must not stop the batch
                mark_error(f"unexpected {type(e).__name__}: {e}")
                continue
            for w in res["warnings"]:
                log(f"{tag} WARN: {w}")
            if res["status"] != "ok":
                mark_error(f"{res['error']} | raw answer: {res['raw_response']!r}")
                continue
            people = res["people"]
            n_vlm = len(people)
        else:
            boxes = snap.get("boxes")
            if not isinstance(boxes, list):
                mark_error("no 'boxes' in this snapshot: crops mode needs Arai's boxes (use --mode frame without them)")
                continue
            try:
                r = describe_boxes(frame, boxes, a, tag)
            except Exception as e:
                mark_error(f"unexpected {type(e).__name__}: {e}")
                continue
            for w in r["warnings"]:
                log(f"{tag} WARN: {w}")
            people, n_vlm = r["people"], len(boxes)
            n_sent += r["sent"]
            n_crop_err += len(r["errors"])
            n_skipped += len(r["skipped"])
            tokens += r["tokens"]
            snap["vlm_crops_skipped"], snap["vlm_crop_errors"] = r["skipped"], r["errors"]
            if r["sent"] and r["n_ok"] == 0:
                mark_error(f"all {r['sent']} sent crops failed")
                continue
            if r["errors"]:
                status = "partial"

        snap["people"] = people
        snap["vlm_people_count"] = n_vlm
        snap["vlm_status"] = status
        n_ok += status == "ok"
        n_part += status == "partial"

        detector = snap.get("people_count")
        if detector is None:
            snap["people_count"] = n_vlm
            note = "people_count was empty -> taken from VLM"
        elif detector != n_vlm:
            n_mismatch += 1
            note = f"MISMATCH detector={detector} vlm={n_vlm} (detector value kept)"
        else:
            note = "count matches detector"
        extra = ""
        if a.mode == "crops":
            extra = f"; skipped(small)={snap['vlm_crops_skipped']} crop_errors={snap['vlm_crop_errors']}"
        log(f"{tag} {status}: {len(people)} people; {note}{extra}")
        for p in people:
            log(f"    #{p['person_idx']:<2} {p['clothing_color']:<10} {p['clothing_type']:<10} | {p['description']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"\nDone in {time.time() - t0:.0f}s: ok={n_ok} partial={n_part} error={n_err} count_mismatch={n_mismatch}"
        + (f" | crops sent={n_sent} failed={n_crop_err} skipped_small={n_skipped} tokens={tokens}" if a.mode == "crops" else ""))
    log(f"Saved: {out_path}")
    if n_err:
        log("Snapshots with vlm_status=error have NO 'people' key: re-run for them (--only) or describe by hand.")


if __name__ == "__main__":
    main()
