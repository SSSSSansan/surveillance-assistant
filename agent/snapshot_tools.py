"""Этап 2: функции агента по архиву срезов (snapshots / snapshot_people).

  search_by_clothing(color, clothing_type, video_id, t_from, t_to)  -> был ли человек в такой одежде, когда
  count_people_at(t, video_id)                                      -> сколько людей на ближайшем к t срезе
  get_snapshot_summary(video_id)                                    -> весь архив видео + максимум/среднее людей

Важно для ответов:
  * Срезы не связаны трекингом: person_idx это НЕ track_id. По срезам нельзя узнать одежду
    «человека 7» из событий; можно ответить только «был ли кто-то в красной кофте» и «когда».
  * Человек с цветом/типом 'unknown' никогда не совпадает с запросом. Чтобы отрицательный ответ
    был честным, в результате считаются uncertain_people и snapshots_not_described.
  * Срезы не подчиняются правилу needs_vlm/confirmed из событий: это отдельный слой.
Проверялось только на SQLite. Значения цветов и типов (русский -> английский, группы синонимов)
собраны в таблицах ниже и ещё не согласованы с Пандой: словарь VLM должен быть фиксированным.
"""
import functools
import re

from agent import tools

MAX_MATCHES = 50


class _InputError(ValueError):
    pass


# ---------- нормализация цвета и типа одежды ----------

_RU_COLORS = [("красн", "red"), ("зелен", "green"), ("зелён", "green"), ("голуб", "light blue"),
              ("син", "blue"), ("желт", "yellow"), ("жёлт", "yellow"), ("оранж", "orange"),
              ("черн", "black"), ("чёрн", "black"), ("бел", "white"), ("сер", "gray"),
              ("коричнев", "brown"), ("розов", "pink"), ("фиолет", "purple"), ("лилов", "purple"),
              ("беж", "beige"), ("бордов", "maroon")]
_COLOR_ALIASES = {"grey": "gray"}

# каноническое название -> синонимы. Запрос и тип человека сравниваются по каноническому названию.
TYPE_GROUPS = {
    "sweater": ["sweater", "jumper", "cardigan", "hoodie", "sweatshirt", "pullover"],
    "t-shirt": ["t-shirt", "tshirt", "t shirt", "tee", "tee shirt"],
    "shirt": ["shirt", "blouse", "button-up"],
    "jacket": ["jacket", "windbreaker", "parka", "bomber", "puffer", "down jacket"],
    "coat": ["coat", "overcoat", "trench coat", "raincoat"],
    "dress": ["dress"],
}
_RU_TYPES = [("кофт", "sweater"), ("свитер", "sweater"), ("толстовк", "sweater"), ("худи", "sweater"),
             ("джемпер", "sweater"), ("кардиган", "sweater"), ("футболк", "t-shirt"), ("майк", "t-shirt"),
             ("рубашк", "shirt"), ("блуз", "shirt"), ("куртк", "jacket"), ("ветровк", "jacket"),
             ("пуховик", "jacket"), ("пальт", "coat"), ("плащ", "coat"), ("плать", "dress")]
_CANON = {syn: name for name, syns in TYPE_GROUPS.items() for syn in syns}
_CYR = re.compile(r"[а-яё]", re.I)


def normalize_color(value):
    v = " ".join(value.lower().replace("-", " ").split())
    if _CYR.search(v):
        for stem, en in _RU_COLORS:
            if v.startswith(stem):
                return en
        return v
    return " ".join(_COLOR_ALIASES.get(w, w) for w in v.split())


def canonical_type(value):
    v = " ".join(value.lower().replace("_", " ").split())
    if _CYR.search(v):
        for stem, en in _RU_TYPES:
            if v.startswith(stem):
                return en
        return v
    return _CANON.get(v, _CANON.get(v.replace(" ", "-"), v))


def color_matches(wanted, person_color):
    """'blue' совпадает с 'dark blue' (слова запроса входят в цвет человека); 'unknown' не совпадает."""
    have = normalize_color(person_color or "unknown")
    if have == "unknown":
        return False
    return set(wanted.split()) <= set(have.split())


def type_matches(wanted, person_type):
    have = canonical_type(person_type or "unknown")
    return have != "unknown" and have == wanted


# ---------- проверка аргументов (пустая строка = аргумента нет: слабые модели так делают) ----------

def _opt_str(name, v):
    if v is None:
        return None
    if not isinstance(v, str):
        raise _InputError(f"{name}: нужна строка, получено {v!r}")
    return v.strip() or None


def _opt_num(name, v, minimum=None):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise _InputError(f"{name}: нужно число, получено {v!r}")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise _InputError(f"{name}: нужно число, получено {v!r}") from None
    if x != x or x in (float("inf"), float("-inf")):
        raise _InputError(f"{name}: нужно конечное число, получено {v!r}")
    if minimum is not None and x < minimum:
        raise _InputError(f"{name}: должно быть >= {minimum}, получено {v!r}")
    return x


def _safe(fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except _InputError as e:
            return {"ok": False, "error": str(e)}
    return wrapper


# ---------- работа с базой ----------

def _rows(conn, sql, params=()):
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, list(params))
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        if own:
            conn.close()


def _where(video_id, t_from, t_to):
    parts, params = [], []
    if video_id:
        parts.append("s.video_id = %s")
        params.append(video_id)
    if t_from is not None:
        parts.append("s.t >= %s")
        params.append(t_from)
    if t_to is not None:
        parts.append("s.t <= %s")
        params.append(t_to)
    return (" WHERE " + " AND ".join(parts)) if parts else "", params


def _load(conn, video_id=None, t_from=None, t_to=None):
    """Срезы с людьми: список словарей в порядке (video_id, t)."""
    where, params = _where(video_id, t_from, t_to)
    rows = _rows(conn,
                 "SELECT s.video_id, s.snapshot_id, s.t, s.frame_path, s.people_count, p.person_idx, "
                 "p.clothing_color, p.clothing_type, p.description "
                 "FROM snapshots s LEFT JOIN snapshot_people p "
                 "ON p.video_id = s.video_id AND p.snapshot_id = s.snapshot_id" + where +
                 " ORDER BY s.video_id, s.t, s.snapshot_id, p.person_idx", params)
    out, index = [], {}
    for vid, sid, t, fp, cnt, idx, color, typ, desc in rows:
        key = (vid, sid)
        if key not in index:
            index[key] = {"video_id": vid, "snapshot_id": sid, "t": t, "frame_path": fp,
                          "people_count": cnt, "people": []}
            out.append(index[key])
        if idx is not None:
            index[key]["people"].append({"person_idx": idx, "clothing_color": color,
                                         "clothing_type": typ, "description": desc})
    return out


def _videos(conn):
    return [r[0] for r in _rows(conn, "SELECT DISTINCT video_id FROM snapshots ORDER BY video_id")]


def _pick_video(conn, video_id):
    """Возвращает (video_id, ошибка). Без video_id допустимо только одно видео с срезами."""
    if video_id:
        return video_id, None
    vids = _videos(conn)
    if len(vids) == 1:
        return vids[0], None
    if not vids:
        return None, None
    return None, f"в архиве несколько видео ({', '.join(vids)}); укажите video_id"


# ---------- функции агента ----------

@_safe
def search_by_clothing(color=None, clothing_type=None, video_id=None, t_from=None, t_to=None, conn=None):
    """Есть ли на срезах человек с такой одеждой, и на каких. Нужен color или clothing_type."""
    color, clothing_type, video_id = (_opt_str("color", color), _opt_str("clothing_type", clothing_type),
                                      _opt_str("video_id", video_id))
    t_from, t_to = _opt_num("t_from", t_from, 0), _opt_num("t_to", t_to, 0)
    if color is None and clothing_type is None:
        raise _InputError("нужен хотя бы один из аргументов: color или clothing_type")
    if t_from is not None and t_to is not None and t_from > t_to:
        raise _InputError("t_from не может быть больше t_to")
    want_color = normalize_color(color) if color else None
    want_type = canonical_type(clothing_type) if clothing_type else None

    snaps = _load(conn, video_id, t_from, t_to)
    matches, uncertain, undescribed, people_matched = [], 0, 0, 0
    for s in snaps:
        if s["people_count"] > 0 and not s["people"]:
            undescribed += 1
            continue
        hit = []
        for p in s["people"]:
            color_unknown = want_color and normalize_color(p["clothing_color"] or "unknown") == "unknown"
            type_unknown = want_type and canonical_type(p["clothing_type"] or "unknown") == "unknown"
            ok_c = (not want_color) or color_matches(want_color, p["clothing_color"])
            ok_t = (not want_type) or type_matches(want_type, p["clothing_type"])
            if ok_c and ok_t:
                hit.append(p)
            elif color_unknown or type_unknown:
                uncertain += 1
        if hit:
            people_matched += len(hit)
            matches.append({"video_id": s["video_id"], "snapshot_id": s["snapshot_id"], "t": s["t"],
                            "frame_path": s["frame_path"], "people_count": s["people_count"], "people": hit})
    res = {"ok": True, "found": bool(matches), "count": len(matches), "people_matched": people_matched,
           "matches": matches[:MAX_MATCHES], "snapshots_checked": len(snaps),
           "uncertain_people": uncertain, "snapshots_not_described": undescribed,
           "query": {"color": want_color, "clothing_type": want_type}}
    if not snaps:
        res["note"] = "no snapshots in the archive for this filter: the answer is 'not found'"
    elif not matches and (uncertain or undescribed):
        res["note"] = (f"not found among described people; {uncertain} people had unknown clothing and "
                       f"{undescribed} snapshots were not described yet, so 'No' is not certain")
    return res


@_safe
def count_people_at(t=None, video_id=None, conn=None):
    """Сколько людей на срезе, ближайшем к моменту t (секунды). Возвращает и время самого среза."""
    t = _opt_num("t", t, 0)
    if t is None:
        raise _InputError("t: нужно число секунд от начала видео")
    vid, err = _pick_video(conn, _opt_str("video_id", video_id))
    if err:
        return {"ok": False, "error": err}
    snaps = _load(conn, vid) if vid else []
    if not snaps:
        return {"ok": True, "found": False, "requested_t": t, "video_id": vid,
                "note": "no snapshots for this video: the answer is 'not found'"}
    best = min(snaps, key=lambda s: (abs(s["t"] - t), s["t"]))   # при равенстве берём более ранний
    return {"ok": True, "found": True, "video_id": best["video_id"], "requested_t": t,
            "snapshot_id": best["snapshot_id"], "t": best["t"], "distance_sec": round(abs(best["t"] - t), 3),
            "people_count": best["people_count"], "frame_path": best["frame_path"],
            "people": best["people"]}


@_safe
def get_snapshot_summary(video_id=None, conn=None):
    """Весь архив срезов видео: людей на каждом срезе, максимум (и когда), среднее."""
    vid, err = _pick_video(conn, _opt_str("video_id", video_id))
    if err:
        return {"ok": False, "error": err}
    snaps = _load(conn, vid) if vid else []
    if not snaps:
        return {"ok": True, "video_id": vid, "snapshots_count": 0, "snapshots": [],
                "note": "no snapshots for this video"}
    counts = [s["people_count"] for s in snaps]
    top = max(counts)
    items = [{"snapshot_id": s["snapshot_id"], "t": s["t"], "people_count": s["people_count"],
              "frame_path": s["frame_path"], "described": bool(s["people"]) or s["people_count"] == 0,
              "people": s["people"]} for s in snaps]
    return {"ok": True, "video_id": snaps[0]["video_id"], "snapshots_count": len(snaps),
            "max_people": top, "max_people_t": [s["t"] for s in snaps if s["people_count"] == top],
            "avg_people": round(sum(counts) / len(counts), 2), "snapshots": items}


def list_snapshot_videos(conn=None):
    """Видео, у которых есть срезы (для выбора на странице архива). Не функция агента."""
    return _videos(conn)
