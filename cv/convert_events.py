"""Конвертер старого events.json (video/type/start_s/end_s/zone) в договорённый формат.

Запуск:
    python cv/convert_events.py cv/events_old.json cv/events.json \
        --video-id <имя файла MEVA> --camera-id <камера> --source meva
"""
import argparse
import json


def convert(old, video_id=None, camera_id="cam01", source="meva"):
    events = []
    for i, e in enumerate(old.get("events", [])):
        events.append({
            "event_id": e.get("event_id", f"cv_{i:04d}"),
            "event_type": e.get("event_type", e.get("type")),
            "track_id": e.get("track_id"),
            "zone_id": e.get("zone_id", e.get("zone")),
            "t_start": e.get("t_start", e.get("start_s")),
            "t_end": e.get("t_end", e.get("end_s")),
            "trigger_score": e.get("trigger_score"),
            "needs_vlm": e.get("needs_vlm", True),
            "clip_path": e.get("clip_path"),
            "detected_by": e.get("detected_by", "cv"),
            "vlm_status": e.get("vlm_status", "not_checked"),
            "vlm_event_type": e.get("vlm_event_type"),
            "vlm_description": e.get("vlm_description"),
            "prompt_variant": e.get("prompt_variant"),
        })
    return {
        "video_id": video_id or old.get("video_id", old.get("video")),
        "camera_id": old.get("camera_id", camera_id),
        "source": source,
        "fps": old.get("fps"),
        "events": events,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--video-id")
    p.add_argument("--camera-id", default="cam01")
    p.add_argument("--source", default="meva", choices=["own", "meva", "warehouse"])
    a = p.parse_args()
    with open(a.src, encoding="utf-8") as f:
        old = json.load(f)
    with open(a.dst, "w", encoding="utf-8") as f:
        json.dump(convert(old, a.video_id, a.camera_id, a.source), f,
                  ensure_ascii=False, indent=2)
    print("готово:", a.dst)