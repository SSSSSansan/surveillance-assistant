"""T32 (опционально). Один человек на разных срезах -> один track_id -> самый частый цвет.

Треки берём из tracks.json этапа 1 (формат cv/tracks_io.py; этап 1 шёл на ТОМ ЖЕ видео
2018-03-15.15-00-00.15-05-00.school.G421, так что трекер заново гнать не надо):

    python -m cv.track_vote --tracks data/out/tracks.json \
        --index data/out_snapshots/crops/crops_index.json \
        --pred results/hist__crop__A.jsonl --out results/hist+vote__crop__vote.jsonl

Если tracks.json нет: добавь --video data/meva/video.avi, ByteTrack прогонится заново (долго, нужен ultralytics).
В --pred можно подать и snapshots_vlm.json Панды: получится "VLM + голосование".

Как работает: у каждого кропа есть время среза t и рамка. В tracks.json ищем точку трека,
ближайшую по времени (не дальше --max-dt секунд), с максимальным IoU рамок (не меньше --min-iou).
Рамки в snapshots.json и в tracks.json получены разными прогонами YOLO, поэтому совпадают приблизительно.
Дальше по каждому треку голосуем (вес = confidence, у VLM его нет - вес 1).

ЖДАТЬ ОТ НЕГО МАЛО. Срезы стоят раз в 30 секунд, и человек, сидящий за столом, попадает в несколько срезов,
но заходящий-выходящий - почти никогда. Скрипт печатает, сколько треков встретились на >=2 срезах и у скольких
из них ответ менялся: это и есть прямая оценка нестабильности (бежевая куртка на этапе 2).
Оценка точности голосования - только по эталону (score_methods people), и только для людей из таких треков.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from cv.crop_people import iou
from eval.score_methods import load_preds


def load_tracks_file(path):
    """tracks.json (tracks_io) -> {track_id: [(t, [x1,y1,x2,y2]), ...]}"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(tr["track_id"]): [(float(p["t"]), p["bbox"]) for p in tr["points"]] for tr in d["tracks"]}


def run_tracker(video, weights="yolo11s.pt", conf=0.3, stride=3, device=None):
    """Запасной путь: ByteTrack заново, как в cv/run_tracking.py."""
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(weights)
    tracks, n = defaultdict(list), -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % stride:
            continue
        r = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=conf,
                        device=device, verbose=False)[0]
        if r.boxes is not None and r.boxes.id is not None:
            for b, tid in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist()):
                tracks[int(tid)].append((n / fps, b))
    cap.release()
    return dict(tracks)


def assign_tracks(crops, tracks, max_dt=0.5, min_iou=0.3):
    """-> {(snapshot_id, person_idx): track_id}"""
    out = {}
    for c in crops:
        best, best_iou = None, min_iou
        for tid, pts in tracks.items():
            t, box = min(pts, key=lambda p: abs(p[0] - c["t"]))
            if abs(t - c["t"]) > max_dt:
                continue
            v = iou(c["box"], box)
            if v > best_iou:
                best, best_iou = tid, v
        if best is not None:
            out[(c["snapshot_id"], c["person_idx"])] = best
    return out


def vote(rows_by_key, key_track):
    """rows_by_key: {(snapshot_id, person_idx): строка с color/type}. -> ({track: (color, type)}, {track: [строки]})"""
    by_track = defaultdict(list)
    for k, tid in key_track.items():
        if k in rows_by_key:
            by_track[tid].append(rows_by_key[k])
    win = {}
    for tid, rs in by_track.items():
        sc, st = Counter(), Counter()
        for r in rs:
            w = r.get("confidence") or 1.0
            if r["color"] != "unknown":
                sc[r["color"]] += w
            if r.get("type") not in (None, "unknown"):
                st[r["type"]] += w
        win[tid] = (sc.most_common(1)[0][0] if sc else "unknown", st.most_common(1)[0][0] if st else "unknown")
    return win, by_track


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--index", required=True, help="crops_index.json")
    ap.add_argument("--pred", required=True, help="results/*.jsonl или snapshots_vlm.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracks", help="tracks.json этапа 1")
    ap.add_argument("--video", help="если нет tracks.json: прогнать ByteTrack по этому видео")
    ap.add_argument("--max-dt", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.3)
    a = ap.parse_args()
    if not (a.tracks or a.video):
        raise SystemExit("нужен --tracks (tracks.json этапа 1) или --video")

    crops = [c for c in json.loads(Path(a.index).read_text(encoding="utf-8")) if c["kept"]]
    preds = load_preds(a.pred)
    tracks = load_tracks_file(a.tracks) if a.tracks else run_tracker(a.video)
    key_track = assign_tracks(crops, tracks, a.max_dt, a.min_iou)
    rows_by_key = {(k[1], k[2]): r for k, r in preds.items()}
    win, by_track = vote(rows_by_key, key_track)

    multi = {t: rs for t, rs in by_track.items() if len(rs) > 1}
    changed = sum(1 for rs in multi.values() if len({r["color"] for r in rs}) > 1)
    print(f"кропов {len(crops)}, привязано к трекам {len(key_track)} ({len(key_track) / max(1, len(crops)):.0%}), "
          f"треков {len(by_track)}; на >=2 срезах: {len(multi)}, из них цвет менялся: {changed}")

    with open(a.out, "w", encoding="utf-8") as f:
        for k, r in preds.items():
            q = dict(r)
            tid = key_track.get((k[1], k[2]))
            if tid is not None and tid in win:
                q["color"], vt = win[tid]
                if vt != "unknown":
                    q["type"] = vt
                q["extras"] = {**(r.get("extras") or {}), "track_id": tid, "n_in_track": len(by_track[tid])}
            q["method"] = r["method"] + "+vote"
            q["prompt_variant"] = "vote"
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("записано:", a.out)


if __name__ == "__main__":
    main()
