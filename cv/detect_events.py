"""T8. События из треков и зон + нарезка клипов -> events.json.

  python cv/detect_events.py data/out/tracks.json cv/zones.json data/meva/video.avi \
      --camera-id G421 --source meva --out cv/events.json --clips-dir clips

Правила (ногами человека считается середина нижней стороны рамки):
  enter / exit              трек начинается / заканчивается в зоне вида entrance
  approach_zone / leave_zone  трек вошёл в зону / вышел из неё
  dwell_in_zone             пробыл в зоне не меньше --dwell секунд
  pickup_or_put_candidate   пробыл в зоне вида object не меньше --dwell секунд, клип 2-8 с -> в VLM
Событие "оставленный предмет" здесь не ищется: это отдельная логика по предметам.
"""
import argparse
import os
import cv2
from events_writer import make_event, save_events
from tracks_io import load_tracks


def inside(pt, poly):
    x, y = pt
    res = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            res = not res
    return res


def segments(points, poly, gap):
    """Непрерывные отрезки времени, когда ноги внутри многоугольника."""
    segs, start, last = [], None, None
    for p in points:
        x1, y1, x2, y2 = p["bbox"]
        if inside(((x1 + x2) / 2, y2), poly):
            if start is None:
                start = p["t"]
            last = p["t"]
        elif start is not None and p["t"] - last > gap:
            segs.append((start, last))
            start = None
    if start is not None:
        segs.append((start, last))
    return segs


def detect(tracks, zones, dwell=3.0, gap=1.0, min_stay=0.5, edge=1.0):
    found = []  # (type, track_id, zone_id, t_start, t_end)
    for tr in tracks:
        pts = sorted(tr["points"], key=lambda p: p["t"])
        if not pts:
            continue
        first_t, last_t = pts[0]["t"], pts[-1]["t"]
        for z in zones:
            for s, e in segments(pts, z["polygon"], gap):
                dur = e - s
                if dur < min_stay:
                    continue
                tid, zid = tr["track_id"], z["zone_id"]
                is_in = z["kind"] == "entrance" and s - first_t <= edge
                is_out = z["kind"] == "entrance" and last_t - e <= edge
                if is_in:
                    found.append(("enter", tid, zid, s, min(s + 1.0, e)))
                else:
                    found.append(("approach_zone", tid, zid, s, min(s + 1.0, e)))
                if is_out:
                    found.append(("exit", tid, zid, max(e - 1.0, s), e))
                else:
                    found.append(("leave_zone", tid, zid, max(e - 1.0, s), e))
                if dur >= dwell:
                    found.append(("dwell_in_zone", tid, zid, s, e))
                    if z["kind"] == "object":
                        found.append(("pickup_or_put_candidate", tid, zid, s, max(min(e, s + 8.0), s + 2.0)))
    found.sort(key=lambda x: x[3])
    return found


def cut_clip(video, t0, t1, path):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t0 * fps))
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(int((t1 - t0) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        out.write(fr)
    out.release()
    cap.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("tracks")
    p.add_argument("zones")
    p.add_argument("video")
    p.add_argument("--camera-id", required=True)
    p.add_argument("--source", default="meva", choices=["own", "meva", "warehouse"])
    p.add_argument("--out", default="cv/events.json")
    p.add_argument("--clips-dir", default="clips")
    p.add_argument("--dwell", type=float, default=3.0, help="сек в зоне для dwell и кандидата")
    p.add_argument("--gap", type=float, default=1.0, help="пропуск, при котором зона не считается покинутой")
    p.add_argument("--min-stay", type=float, default=0.5, help="минимум секунд в зоне, чтобы считать заход")
    p.add_argument("--edge", type=float, default=1.0, help="допуск у начала/конца трека для enter/exit")
    a = p.parse_args()

    import json
    T = load_tracks(a.tracks)
    Z = json.load(open(a.zones, encoding="utf-8"))["zones"]
    os.makedirs(a.clips_dir, exist_ok=True)
    events = []
    for i, (typ, tid, zid, t0, t1) in enumerate(detect(T["tracks"], Z, dwell=a.dwell, gap=a.gap, min_stay=a.min_stay, edge=a.edge)):
        clip = None
        if typ == "pickup_or_put_candidate":
            clip = os.path.join(a.clips_dir, f"cv_{i:04d}.mp4")
            cut_clip(a.video, t0, t1, clip)
        events.append(make_event(i, typ, tid, zid, t0, t1, clip_path=clip,
                                 needs_vlm=typ == "pickup_or_put_candidate"))
    save_events(a.out, T["video_id"], a.camera_id, a.source, T["fps"], events)
    print("событий:", len(events), "->", a.out)