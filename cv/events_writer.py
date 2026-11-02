"""Запись событий детектора в договорённый формат events.json.

Использование в твоём пайплайне (вместо ручной сборки словаря):

    from events_writer import make_event, save_events

    events = [make_event(0, "pickup_or_put_candidate", track_id=274,
                         zone_id="table", t_start=22.9, t_end=30.9,
                         clip_path="clips/cv_0000.mp4")]
    save_events("cv/events.json", video_id="<имя файла MEVA без расширения>",
                camera_id="<камера MEVA, например G421>", source="meva",
                fps=30.0, events=events)
"""
import json

ALLOWED_SOURCES = {"own", "meva", "warehouse"}
ALLOWED_VLM_STATUS = {"not_checked", "confirmed", "rejected", "error"}
ALLOWED_VLM_TYPES = {None, "pick_up", "put_down", "none"}


def make_event(index, event_type, track_id, zone_id, t_start, t_end,
               clip_path=None, trigger_score=None, needs_vlm=True):
    """Одно событие. Поля VLM остаются пустыми, их заполняет Панда."""
    if t_end < t_start:
        raise ValueError(f"t_end < t_start в событии {index}")
    return {
        "event_id": f"cv_{index:04d}",
        "event_type": event_type,
        "track_id": track_id,
        "zone_id": zone_id,
        "t_start": round(float(t_start), 2),
        "t_end": round(float(t_end), 2),
        "trigger_score": trigger_score,
        "needs_vlm": needs_vlm,
        "clip_path": clip_path,
        "detected_by": "cv",
        "vlm_status": "not_checked",
        "vlm_event_type": None,
        "vlm_description": None,
        "prompt_variant": None,
    }


def save_events(path, video_id, camera_id, source, fps, events):
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"source должен быть один из {ALLOWED_SOURCES}")
    for e in events:
        if e["vlm_status"] not in ALLOWED_VLM_STATUS:
            raise ValueError(f"плохой vlm_status в {e['event_id']}")
        if e["vlm_event_type"] not in ALLOWED_VLM_TYPES:
            raise ValueError(f"плохой vlm_event_type в {e['event_id']}")
    data = {"video_id": video_id, "camera_id": camera_id, "source": source,
            "fps": fps, "events": events}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)