"""Чистые функции для страницы Streamlit (без самого Streamlit, чтобы их можно было тестировать)."""
from pathlib import Path


def collect_events(steps):
    """События из результатов вызовов функций, без повторов (по video_id + event_id), в порядке появления."""
    seen, out = set(), []
    for step in steps:
        res = step.result if hasattr(step, "result") else step
        if not isinstance(res, dict):
            continue
        items = []
        for key in ("events", "results"):
            if isinstance(res.get(key), list):
                items.extend(res[key])
        if isinstance(res.get("event"), dict):
            items.append(res["event"])
        for ev in items:
            if not isinstance(ev, dict) or "event_id" not in ev:
                continue
            key = (ev.get("video_id"), ev["event_id"])
            if key not in seen:
                seen.add(key)
                out.append(ev)
    return out


def format_time(ev):
    a, b = ev.get("t_start"), ev.get("t_end")
    if a is None:
        return ""
    return f"{a:g} с" if b is None or b == a else f"{a:g}–{b:g} с"


def event_row(ev):
    """Строка таблицы событий для интерфейса."""
    return {"event": ev.get("event_id"), "video": ev.get("video_id"), "тип": ev.get("event_type"),
            "человек": ev.get("track_id"), "зона": ev.get("zone_id"), "время": format_time(ev),
            "описание VLM": ev.get("vlm_description") or ""}


def safe_clip_path(clip_path, root):
    """Путь к клипу из базы -> существующий файл ВНУТРИ root, иначе None.
    Значение из базы не доверяется: '../' и абсолютные пути за пределами root отклоняются."""
    if not clip_path or not isinstance(clip_path, str):
        return None
    try:
        root = Path(root).resolve()
        path = (root / clip_path).resolve()
        path.relative_to(root)
    except (ValueError, OSError):
        return None
    return path if path.is_file() else None
