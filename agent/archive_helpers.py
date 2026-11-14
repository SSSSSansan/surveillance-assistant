"""Чистые функции для страницы «Архив событий» и показа кадров в чате (без Streamlit: тестируются отдельно).
Текст одежды приходит от VLM и не доверяется: в HTML он экранируется, цвет берётся только из белого списка."""
import html

COLOR_HEX = {"red": "#d64545", "green": "#3d9a55", "blue": "#3d6fd1", "light blue": "#7fb6ea",
             "black": "#222222", "white": "#f2f2f2", "gray": "#8c8c8c", "yellow": "#e8c53a",
             "orange": "#e88a3a", "brown": "#8a5a3c", "pink": "#e58fb0", "purple": "#8a5ac0",
             "beige": "#d8c8a8", "maroon": "#7a2a3a"}
UNKNOWN_HEX = "#c8c8c8"


def fmt_clock(t):
    """90 -> '1:30', 125.5 -> '2:05' (секунды округляются вниз)."""
    s = int(max(float(t), 0))
    return f"{s // 60}:{s % 60:02d}"


def color_hex(name):
    """Цвет точки для названия цвета. Неизвестное значение -> серый; 'dark blue' -> по последнему слову."""
    v = (name or "").strip().lower().replace("grey", "gray")
    if v in COLOR_HEX:
        return COLOR_HEX[v]
    return COLOR_HEX.get(v.split()[-1], UNKNOWN_HEX) if v else UNKNOWN_HEX


def person_label(p):
    color, typ = p.get("clothing_color") or "unknown", p.get("clothing_type") or "unknown"
    if color == "unknown" and typ == "unknown":
        return "не удалось определить"
    return " ".join(x for x in (color, typ) if x != "unknown")


def chip_html(p):
    """Плашка человека: цветная точка + «red sweater». Безопасна для unsafe_allow_html."""
    return (f'<span class="chip"><span class="dot" style="background:{color_hex(p.get("clothing_color"))}"></span>'
            f'{html.escape(person_label(p))}</span>')


def people_html(people):
    return "".join(chip_html(p) for p in people) or '<span class="muted">людей не описано</span>'


def filter_options(snapshots):
    """Какие цвета и типы одежды есть в архиве (без 'unknown'), для фильтров."""
    colors, types = set(), set()
    for s in snapshots:
        for p in s.get("people", []):
            if p.get("clothing_color") not in (None, "unknown"):
                colors.add(p["clothing_color"])
            if p.get("clothing_type") not in (None, "unknown"):
                types.add(p["clothing_type"])
    return sorted(colors), sorted(types)


def match_snapshots(snapshots, colors=(), types=()):
    """Копии срезов с полем matched. Без фильтров matched = False у всех. Цвет И тип (если заданы оба)
    должны относиться к одному и тому же человеку; внутри списка цветов/типов достаточно любого."""
    colors, types = set(colors), set(types)
    out = []
    for s in snapshots:
        matched = False
        if colors or types:
            matched = any((not colors or p.get("clothing_color") in colors) and
                          (not types or p.get("clothing_type") in types) for p in s.get("people", []))
        out.append({**s, "matched": matched})
    return out


def collect_snapshots(steps):
    """Срезы из результатов search_by_clothing (matches) и count_people_at (один срез), без повторов.
    Результат get_snapshot_summary сюда не попадает: весь архив смотрят на странице «Архив»."""
    seen, out = set(), []
    for step in steps:
        res = step.result if hasattr(step, "result") else step
        if not isinstance(res, dict) or not res.get("ok", True):
            continue
        items = list(res["matches"]) if isinstance(res.get("matches"), list) else []
        if res.get("snapshot_id") and res.get("found"):
            items.append({k: res.get(k) for k in ("video_id", "snapshot_id", "t", "frame_path", "people_count")})
        for sn in items:
            if not isinstance(sn, dict) or "snapshot_id" not in sn:
                continue
            key = (sn.get("video_id"), sn["snapshot_id"])
            if key not in seen:
                seen.add(key)
                out.append(sn)
    return out
