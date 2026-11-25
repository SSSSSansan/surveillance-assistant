"""Draw the VLM result on copies of the frames, for a visual check by eye.

Usage:
    python vlm/annotate_snapshots.py SNAPSHOTS_VLM_JSON FRAMES_DIR [--out annotated] [--only sn_0000,sn_0003]

For every snapshot of snapshots_vlm.json (output of batch_describe_snapshots.py):
  * crops mode: a box around each person with the label "idx color type"
      green  = color and type known
      orange = color or type is "unknown"
      gray   = box was too small, not sent to the model
      red    = request/parse failed for this crop
  * frame mode (people have no boxes): the list "idx color type" is printed in the top-left corner.
  * the top line shows: snapshot id, t, detector count, VLM status. A snapshot with
    vlm_status "error" is stamped red (and has no people).
The original frames are never modified. Output: OUT/<snapshot_id>.jpg.
Needs OpenCV or Pillow (whichever is installed).
"""
import argparse
import json
import sys
from pathlib import Path

import imgutil
from batch_describe_snapshots import find_frame


def person_kind(p: dict, skipped: set, errors: set) -> str:
    idx = p.get("person_idx")
    if idx in errors:
        return "error"
    if idx in skipped:
        return "skipped"
    if "unknown" in (p.get("clothing_color"), p.get("clothing_type")):
        return "unknown"
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshots_json")
    parser.add_argument("frames_dir")
    parser.add_argument("--out", default="annotated")
    parser.add_argument("--only", default=None, help="comma-separated snapshot_ids")
    a = parser.parse_args()

    if imgutil.BACKEND is None:
        sys.exit("Need OpenCV or Pillow: pip install pillow  (or opencv-python-headless)")
    with open(a.snapshots_json, encoding="utf-8") as f:
        data = json.load(f)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir, json_dir = Path(a.frames_dir), Path(a.snapshots_json).resolve().parent
    wanted = {s.strip() for s in a.only.split(",")} if a.only else None

    done = failed = 0
    for snap in data.get("snapshots", []):
        sid = snap.get("snapshot_id", "?")
        if wanted and sid not in wanted:
            continue
        frame = find_frame(snap.get("frame_path", ""), frames_dir, json_dir)
        if frame is None:
            print(f"{sid}: frame not found ({snap.get('frame_path')!r})", file=sys.stderr)
            failed += 1
            continue
        try:
            img = imgutil.load_image(str(frame))
            status = snap.get("vlm_status", "?")
            skipped = set(snap.get("vlm_crops_skipped") or [])
            errors = set(snap.get("vlm_crop_errors") or [])
            items, lines = [], []
            for p in snap.get("people", []) or []:
                label = f"{p.get('person_idx')} {p.get('clothing_color')} {p.get('clothing_type')}"
                kind = person_kind(p, skipped, errors)
                if isinstance(p.get("box"), list) and len(p["box"]) == 4:
                    items.append((p["box"], label, kind))
                else:
                    lines.append(label)
            header = (f"{sid} t={snap.get('t')}s  detector={snap.get('people_count')} "
                      f"vlm={snap.get('vlm_people_count')}  status={status}")
            kind = {"ok": "ok", "partial": "unknown"}.get(status, "error")
            if status == "error":
                lines = ["VLM ERROR: no people written for this snapshot"]
            imgutil.annotate_and_save(img, items, header, str(out_dir / f"{sid}.jpg"),
                                      banner_lines=lines, banner_kind=kind)
            print(f"{sid}: saved ({len(items)} boxes, {len(lines)} text lines)", file=sys.stderr)
            done += 1
        except Exception as e:
            print(f"{sid}: ERROR {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
    print(f"Done: {done} saved, {failed} failed -> {out_dir}/", file=sys.stderr)
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
