"""Общий формат tracks.json. Если у тебя уже есть свой код детекции и трекинга,
достаточно вызвать save_tracks, и остальные скрипты заработают.

tracks: {track_id: [{"t": секунды, "frame": номер, "bbox": [x1,y1,x2,y2], "conf": 0.9}, ...]}
"""
import json


def save_tracks(path, video_id, fps, width, height, tracks):
    data = {"video_id": video_id, "fps": fps, "width": width, "height": height,
            "tracks": [{"track_id": int(k), "points": v} for k, v in sorted(tracks.items())]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_tracks(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)