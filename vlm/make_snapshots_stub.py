"""Fallback when Arai's frames have not arrived yet: cut one frame every N seconds
from a video and write an EMPTY snapshots.json (no counts, no clothing).

Usage:
    python vlm/make_snapshots_stub.py VIDEO OUT_DIR [--camera-id G421] [--source meva] [--interval 30]

Result:
    OUT_DIR/frames/sn_0000.jpg, sn_0001.jpg, ...
    OUT_DIR/snapshots.json     (format of STAGE2_PLAN.md, section 4; people_count = null, no "people" key)

Same frame selection as cv/sample_snapshots.py: t = 0, 30, 60, ... strictly less than
the video duration (5 min -> 10 frames), frames read sequentially.

NOTE: people_count = null is rejected by database.snapshots_loader. Run
batch_describe_snapshots.py first (it fills people_count from the VLM when it is empty)
or take the real count from Arai's file.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2


def guess_camera_id(stem: str) -> str:
    # MEVA name: 2018-03-15.15-00-00.15-05-00.school.G421.r13 -> G421
    return next((p for p in stem.split(".") if p.startswith("G") and p[1:].isdigit()), "unknown")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("out_dir")
    parser.add_argument("--camera-id", default=None, help="default: taken from the file name (G421)")
    parser.add_argument("--source", default="meva")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between snapshots")
    args = parser.parse_args()

    stem = Path(args.video).stem
    camera_id = args.camera_id or guess_camera_id(stem)
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    print(f"video: {stem}, fps={fps:.2f}, frames={n_frames}, duration={duration:.1f}s", file=sys.stderr)

    targets, k = {}, 0
    while k * args.interval < duration:
        targets[int(round(k * args.interval * fps))] = k
        k += 1
    if not targets:
        sys.exit("No snapshots to take (video too short or unreadable)")
    print(f"snapshots: {len(targets)} (every {args.interval}s)", file=sys.stderr)

    snapshots, frame_no, last = [], -1, max(targets)
    while frame_no < last:
        if not cap.grab():
            break
        frame_no += 1
        if frame_no not in targets:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            print(f"  cannot read frame {frame_no}, skipped", file=sys.stderr)
            continue
        sid = f"sn_{targets[frame_no]:04d}"
        cv2.imwrite(str(frames_dir / f"{sid}.jpg"), frame)
        snapshots.append({
            "snapshot_id": sid,
            "t": round(frame_no / fps, 3),
            "frame_path": f"frames/{sid}.jpg",  # always forward slashes
            "people_count": None,
            # NO "people" key on purpose: database.snapshots_loader treats "people": [] as
            # "nobody there" and would erase people already described. batch_describe_snapshots.py adds the key.
        })
        print(f"  {sid}  t={frame_no / fps:6.1f}s", file=sys.stderr)
    cap.release()

    out = {"video_id": stem, "camera_id": camera_id, "source": args.source,
           "interval_sec": args.interval, "snapshots": snapshots}
    path = out_dir / "snapshots.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Done: {len(snapshots)} snapshots -> {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
