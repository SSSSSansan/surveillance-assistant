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


# ---------- карточки архива (вёрстка в HTML, чтобы вид не зависел от внутреннего устройства Streamlit) ----------
import base64  # noqa: E402
import functools  # noqa: E402
import io  # noqa: E402
import os  # noqa: E402


import re  # noqa: E402

_DATA_URI = re.compile(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+")   # строго base64: в атрибут src ничего лишнего не попадёт


@functools.lru_cache(maxsize=512)
def _thumb_cached(path_str, mtime, max_w):
    try:
        from PIL import Image
        with Image.open(path_str) as im:
            im = im.convert("RGB")
            im.thumbnail((max_w, max_w))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001  (нет Pillow, битый файл и т. п.: карточка покажет заглушку)
        return None


def thumb_data_uri(path, max_w=360):
    """Уменьшенный кадр как data-URI для вставки в HTML-карточку. None, если файла нет или он не читается."""
    if not path:
        return None
    try:
        return _thumb_cached(str(path), os.path.getmtime(path), max_w)
    except OSError:
        return None


def snapshot_text(s):
    """Текст карточки: описания людей одной строкой (чистый текст, в HTML экранируется при выводе)."""
    descs = [(p.get("description") or person_label(p)) for p in s.get("people", [])]
    if descs:
        return "; ".join(descs)
    return "В кадре никого нет." if s.get("people_count") == 0 else "Одежда ещё не описана (VLM)."


def search_snapshots(snapshots, query):
    """Поиск по описанию: подстрока без учёта регистра в тексте, цвете, типе и номере среза."""
    q = (query or "").strip().lower()
    if not q:
        return list(snapshots)
    out = []
    for s in snapshots:
        hay = " ".join([snapshot_text(s), str(s.get("snapshot_id", "")), fmt_clock(s["t"])] +
                       [f"{p.get('clothing_color', '')} {p.get('clothing_type', '')}" for p in s.get("people", [])])
        if q in hay.lower():
            out.append(s)
    return out


def sort_snapshots(snapshots, newest_first=True):
    return sorted(snapshots, key=lambda s: s["t"], reverse=bool(newest_first))


def card_html(s, thumb_uri=None, matched=False, missing_text="кадр не найден"):
    """HTML одной карточки среза. Всё, что пришло из базы (описания от VLM, пути), экранируется;
    картинка вставляется только как data:image/..., созданная thumb_data_uri."""
    if isinstance(thumb_uri, str) and _DATA_URI.fullmatch(thumb_uri):
        img = f'<img src="{thumb_uri}" alt="">'
    else:
        img = f'<div class="noimg">{html.escape(missing_text)}</div>'
    head = (f'<span class="ts">{fmt_clock(s["t"])}</span>'
            f'<span class="meta">{html.escape(str(s.get("snapshot_id", "")))} · {s["t"]:g} с</span>'
            f'<span class="badge">людей: {int(s.get("people_count", 0))}</span>')
    if matched:
        head += '<span class="badge hit">совпадение</span>'
    chips = people_html(s["people"]) if s.get("people") else ""
    return (f'<div class="card{" hit" if matched else ""}"><div class="card-img">{img}</div><div class="card-body">'
            f'<div class="card-head">{head}</div><div class="card-text">{html.escape(snapshot_text(s))}</div>'
            f'<div>{chips}</div></div></div>')
