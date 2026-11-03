"""T5: функции для чат-ассистента (версия 2): query_events и count_events.

Правило видимости (что ассистент вправе использовать в обычных ответах):
  * needs_vlm = false                       -> событие доступно;
  * needs_vlm = true и vlm_status=confirmed -> доступно;
  * needs_vlm = true и not_checked          -> недоступно (пока не подтверждено);
  * rejected                                -> недоступно.
vlm_status берётся из vlm_results для выбранного варианта подсказки (A или B).
Нет строки в vlm_results = not_checked.

Отладка: debug_query_events / debug_count_events умеют показывать неподтверждённые и
отклонённые события. Их НЕЛЬЗЯ подключать к чат-ассистенту как инструменты.

Вариант подсказки: по умолчанию DEFAULT_VLM_VARIANT ("A"). Это ВРЕМЕННОЕ допущение:
команда ещё должна подтвердить выбор основного варианта.
Параметр vlm_variant нужен разработчику, в схему инструментов для модели его включать не надо.

Окно времени [t_from, t_to]: событие подходит, если его интервал пересекается с окном
(t_end >= t_from и t_start <= t_to). Допуск по времени в функциях не зашит.

Подключение: переменная окружения DATABASE_URL, например postgresql://user:pass@host:5432/db
Проверено только на SQLite (см. eval/test_tools.py), на PostgreSQL не запускалось.
"""
import json
import math
import os
import re

EVENT_TYPES = (
    "enter", "exit", "approach_zone", "leave_zone",
    "dwell_in_zone", "pick_up", "put_down", "abandoned_object",
    "pickup_or_put_candidate",
)
VLM_VARIANTS = ("A", "B")
# ВРЕМЕННОЕ ДОПУЩЕНИЕ: вариант "A" считается основным для ответов ассистента.
# Команда ещё должна подтвердить, какой вариант основной. До подтверждения не менять и не считать решённым.
DEFAULT_VLM_VARIANT = "A"
VIEWS = ("assistant", "all", "unconfirmed", "not_checked", "rejected")
MAX_LIMIT = 200
DEFAULT_LIMIT = 50

_VIEW_CLAUSES = {
    "assistant": "(e.needs_vlm = FALSE OR r.vlm_status = 'confirmed')",
    "all": None,
    "unconfirmed": "(e.needs_vlm = TRUE AND COALESCE(r.vlm_status, 'not_checked') <> 'confirmed')",
    "not_checked": "(e.needs_vlm = TRUE AND COALESCE(r.vlm_status, 'not_checked') = 'not_checked')",
    "rejected": "r.vlm_status = 'rejected'",
}

_SELECT = (
    "SELECT e.video_id, e.event_id, v.camera_id, v.source, e.event_type, e.track_id, "
    "e.zone_id, e.t_start, e.t_end, e.trigger_score, e.needs_vlm, e.clip_path, "
    "COALESCE(r.vlm_status, 'not_checked') AS vlm_status, r.vlm_description, r.prompt_variant "
)
_FROM = (
    "FROM events e JOIN videos v ON v.video_id = e.video_id "
    "LEFT JOIN vlm_results r ON r.video_id = e.video_id AND r.event_id = e.event_id "
    "AND r.prompt_variant = %s "
)


class ToolInputError(ValueError):
    """Неправильный аргумент функции."""


# ---------- проверка аргументов ----------

def _str(name, value, *, lower=False, upper=False):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError(f"{name}: нужна непустая строка, получено {value!r}")
    value = value.strip()
    if lower:
        return value.lower()
    if upper:
        return value.upper()
    return value


def _number(name, value, *, minimum=None):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ToolInputError(f"{name}: нужно число, получено {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ToolInputError(f"{name}: нужно число, получено {value!r}") from None
    if math.isnan(x) or math.isinf(x):
        raise ToolInputError(f"{name}: число должно быть конечным")
    if minimum is not None and x < minimum:
        raise ToolInputError(f"{name}: не может быть меньше {minimum}, получено {x}")
    return x


def _integer(name, value, *, minimum=None):
    x = _number(name, value, minimum=minimum)
    if x is None:
        return None
    if x != int(x):
        raise ToolInputError(f"{name}: нужно целое число, получено {value!r}")
    return int(x)


def _prepare(view, video_id, camera_id, event_type, zone_id, track_id,
             t_from, t_to, vlm_variant):
    """Проверяет аргументы. Возвращает (вариант подсказки, WHERE-строка, параметры)."""
    if view not in VIEWS:
        raise ToolInputError(f"view: допустимо {', '.join(VIEWS)}, получено {view!r}")
    video_id = _str("video_id", video_id)
    camera_id = _str("camera_id", camera_id)
    event_type = _str("event_type", event_type, lower=True)
    zone_id = _str("zone_id", zone_id, lower=True)
    track_id = _integer("track_id", track_id, minimum=0)
    t_from = _number("t_from", t_from, minimum=0)
    t_to = _number("t_to", t_to, minimum=0)
    variant = _str("vlm_variant", vlm_variant if vlm_variant is not None
                   else DEFAULT_VLM_VARIANT, upper=True)

    if variant not in VLM_VARIANTS:
        raise ToolInputError(f"vlm_variant: допустимо A или B, получено {variant!r}")
    if event_type is not None and event_type not in EVENT_TYPES:
        raise ToolInputError(
            f"event_type: неизвестный тип {event_type!r}, допустимы: {', '.join(EVENT_TYPES)}")
    if t_from is not None and t_to is not None and t_from > t_to:
        raise ToolInputError(f"t_from ({t_from}) больше t_to ({t_to})")

    clauses, params = [], []
    if video_id is not None:
        clauses.append("e.video_id = %s"); params.append(video_id)
    if camera_id is not None:
        clauses.append("v.camera_id = %s"); params.append(camera_id)
    if event_type is not None:
        clauses.append("e.event_type = %s"); params.append(event_type)
    if zone_id is not None:
        clauses.append("e.zone_id = %s"); params.append(zone_id)
    if track_id is not None:
        clauses.append("e.track_id = %s"); params.append(track_id)
    if t_from is not None:
        clauses.append("e.t_end >= %s"); params.append(t_from)
    if t_to is not None:
        clauses.append("e.t_start <= %s"); params.append(t_to)
    if _VIEW_CLAUSES[view]:
        clauses.append(_VIEW_CLAUSES[view])
    where = ("WHERE " + " AND ".join(clauses) + " ") if clauses else ""
    return variant, where, params


# ---------- подключение ----------

def _connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Не задана переменная окружения DATABASE_URL")
    try:
        import psycopg2  # импорт здесь, чтобы тесты работали без psycopg2
    except ImportError as e:
        raise RuntimeError("Нужен пакет psycopg2-binary (pip install psycopg2-binary)") from e
    return psycopg2.connect(url)


def _run(sql, params, conn):
    own = conn is None
    if own:
        conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
        cur.close()
        return names, rows
    finally:
        if own:
            conn.close()


# ---------- общая логика ----------

def _query(view, video_id, camera_id, event_type, zone_id, track_id, t_from, t_to,
           order, limit, vlm_variant, conn):
    try:
        variant, where, params = _prepare(view, video_id, camera_id, event_type, zone_id,
                                          track_id, t_from, t_to, vlm_variant)
        order = _str("order", order, lower=True) or "asc"
        if order not in ("asc", "desc"):
            raise ToolInputError("order: допустимо 'asc' или 'desc'")
        limit = _integer("limit", limit, minimum=1)
        if limit is None:
            limit = DEFAULT_LIMIT
        limit = min(limit, MAX_LIMIT)
    except ToolInputError as e:
        return {"ok": False, "error": str(e)}

    d = order.upper()
    sql = (_SELECT + _FROM + where +
           f"ORDER BY e.video_id {d}, e.t_start {d}, e.event_id {d} LIMIT %s")
    names, rows = _run(sql, [variant] + params + [limit], conn)
    events = []
    for row in rows:
        ev = dict(zip(names, row))
        ev["needs_vlm"] = bool(ev["needs_vlm"])
        events.append(ev)
    return {"ok": True, "view": view, "vlm_variant": variant,
            "count": len(events), "events": events}


def _count(view, video_id, camera_id, event_type, zone_id, track_id, t_from, t_to,
           vlm_variant, conn):
    try:
        variant, where, params = _prepare(view, video_id, camera_id, event_type, zone_id,
                                          track_id, t_from, t_to, vlm_variant)
    except ToolInputError as e:
        return {"ok": False, "error": str(e)}
    _, rows = _run("SELECT COUNT(*) " + _FROM + where, [variant] + params, conn)
    return {"ok": True, "view": view, "vlm_variant": variant, "count": int(rows[0][0])}


# ---------- функции для ассистента (только видимые события) ----------

def query_events(video_id=None, camera_id=None, event_type=None, zone_id=None,
                 track_id=None, t_from=None, t_to=None, order="asc",
                 limit=DEFAULT_LIMIT, vlm_variant=None, conn=None):
    """Найти события по видео, камере, типу, зоне, человеку и окну времени (секунды)."""
    return _query("assistant", video_id, camera_id, event_type, zone_id, track_id,
                  t_from, t_to, order, limit, vlm_variant, conn)


def count_events(video_id=None, camera_id=None, event_type=None, zone_id=None,
                 track_id=None, t_from=None, t_to=None, vlm_variant=None, conn=None):
    """Посчитать события с теми же фильтрами, что в query_events."""
    return _count("assistant", video_id, camera_id, event_type, zone_id, track_id,
                  t_from, t_to, vlm_variant, conn)


# ---------- только для отладки (не отдавать ассистенту) ----------

def debug_query_events(view="unconfirmed", video_id=None, camera_id=None, event_type=None,
                       zone_id=None, track_id=None, t_from=None, t_to=None, order="asc",
                       limit=DEFAULT_LIMIT, vlm_variant=None, conn=None):
    """Как query_events, но с выбором view: all, unconfirmed, not_checked, rejected, assistant."""
    return _query(view, video_id, camera_id, event_type, zone_id, track_id,
                  t_from, t_to, order, limit, vlm_variant, conn)


def debug_count_events(view="unconfirmed", video_id=None, camera_id=None, event_type=None,
                       zone_id=None, track_id=None, t_from=None, t_to=None,
                       vlm_variant=None, conn=None):
    """Как count_events, но с выбором view."""
    return _count(view, video_id, camera_id, event_type, zone_id, track_id,
                  t_from, t_to, vlm_variant, conn)


# =====================================================================
# T10: остальные функции ассистента
# Те же правила: видимость по VLM, проверка аргументов, ответ {"ok": False, "error": ...}
# вместо исключения. Все четыре работают только с видимыми событиями.
# =====================================================================

def _add_where(where, cond):
    """Добавляет условие к WHERE-строке из _prepare (она может быть пустой)."""
    return where + ("AND " if where else "WHERE ") + cond + " "


def get_track_timeline(track_id=None, video_id=None, camera_id=None, vlm_variant=None, conn=None):
    """События одного человека по порядку времени. track_id уникален только внутри видео:
    если он встречается в нескольких видео, а video_id не указан, вернётся ошибка."""
    if track_id is None:
        return {"ok": False, "error": "track_id: обязателен"}
    r = _query("assistant", video_id, camera_id, None, None, track_id, None, None,
               "asc", MAX_LIMIT, vlm_variant, conn)
    if not r["ok"]:
        return r
    vids = sorted({e["video_id"] for e in r["events"]})
    if len(vids) > 1:
        return {"ok": False, "error": f"track_id {track_id} встречается в нескольких видео "
                                      f"({', '.join(vids)}); укажите video_id"}
    r["track_id"] = _integer("track_id", track_id)
    r["video_id"] = vids[0] if vids else video_id
    r["truncated"] = r["count"] >= MAX_LIMIT
    return r


def _details(view, event_id, video_id, vlm_variant, conn, with_all_variants):
    try:
        event_id = _str("event_id", event_id)
        if event_id is None:
            raise ToolInputError("event_id: обязателен")
        variant, where, params = _prepare(view, video_id, None, None, None, None, None, None,
                                          vlm_variant)
    except ToolInputError as e:
        return {"ok": False, "error": str(e)}
    where = _add_where(where, "e.event_id = %s")
    names, rows = _run(_SELECT + _FROM + where + "ORDER BY e.video_id LIMIT 3",
                       [variant] + params + [event_id], conn)
    evs = []
    for row in rows:
        ev = dict(zip(names, row))
        ev["needs_vlm"] = bool(ev["needs_vlm"])
        evs.append(ev)
    if not evs:
        return {"ok": True, "found": False, "event_id": event_id,
                "message": "Событие не найдено или не подтверждено"}
    if len(evs) > 1:
        return {"ok": False, "error": f"event_id {event_id!r} есть в нескольких видео "
                                      f"({', '.join(e['video_id'] for e in evs)}); укажите video_id"}
    res = {"ok": True, "found": True, "vlm_variant": variant, "event": evs[0]}
    if with_all_variants:
        _, vr = _run("SELECT prompt_variant, vlm_status, vlm_description FROM vlm_results "
                     "WHERE video_id = %s AND event_id = %s ORDER BY prompt_variant",
                     [evs[0]["video_id"], event_id], conn)
        res["view"] = view
        res["vlm_results"] = [{"prompt_variant": a, "vlm_status": b, "vlm_description": c}
                              for a, b, c in vr]
    return res


def get_event_details(event_id=None, video_id=None, vlm_variant=None, conn=None):
    """Детали одного события (в том числе clip_path и описание VLM). Скрытые события
    (не подтверждённые или отклонённые VLM) не находятся: found = False."""
    return _details("assistant", event_id, video_id, vlm_variant, conn, False)


def debug_get_event_details(event_id=None, video_id=None, view="all", vlm_variant=None, conn=None):
    """Только для отладки: находит и скрытые события, показывает результаты A и B."""
    return _details(view, event_id, video_id, vlm_variant, conn, True)


def list_zones_cameras(camera_id=None, conn=None):
    """Список камер, их видео и зон."""
    try:
        camera_id = _str("camera_id", camera_id)
    except ToolInputError as e:
        return {"ok": False, "error": str(e)}
    cond, p = ("WHERE camera_id = %s ", [camera_id]) if camera_id else ("", [])
    _, vrows = _run("SELECT camera_id, video_id, source FROM videos " + cond +
                    "ORDER BY camera_id, video_id", p, conn)
    _, zrows = _run("SELECT camera_id, zone_id, description FROM zones " + cond +
                    "ORDER BY camera_id, zone_id", p, conn)
    cams = {}
    for cam, vid, src in vrows:
        cams.setdefault(cam, {"camera_id": cam, "videos": [], "zones": []})["videos"].append(
            {"video_id": vid, "source": src})
    for cam, zid, desc in zrows:
        cams.setdefault(cam, {"camera_id": cam, "videos": [], "zones": []})["zones"].append(
            {"zone_id": zid, "description": desc})
    cameras = [cams[k] for k in sorted(cams)]
    return {"ok": True, "count": len(cameras), "cameras": cameras}


SEARCH_MODES = ("keyword", "semantic")


def search_descriptions(query=None, mode="semantic", top_k=5, video_id=None, camera_id=None,
                        event_type=None, vlm_variant=None, min_score=None,
                        embedder=None, model=None, conn=None):
    """Поиск по описаниям VLM среди видимых событий выбранного варианта.

    mode='keyword'  : по словам; score = доля слов запроса, найденных в описании.
    mode='semantic' : по смыслу; score = косинусная близость эмбеддингов.
                      По умолчанию используется предварительный min_score=0.65;
                      порог нужно проверить на размеченных вопросах и при необходимости изменить.
    Эмбеддинги, которых нет или которые устарели, пропускаются; их число в skipped_no_embedding."""
    try:
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError("query: нужна непустая строка")
        mode = _str("mode", mode, lower=True)
        if mode not in SEARCH_MODES:
            raise ToolInputError(f"mode: допустимо {', '.join(SEARCH_MODES)}, получено {mode!r}")
        top_k = _integer("top_k", top_k, minimum=1)
        top_k = min(top_k if top_k is not None else 5, MAX_LIMIT)
        min_score = _number("min_score", min_score)
        variant, where, params = _prepare("assistant", video_id, camera_id, event_type, None, None,
                                          None, None, vlm_variant)
        where = _add_where(where, "r.vlm_description IS NOT NULL")
    except ToolInputError as e:
        return {"ok": False, "error": str(e)}

    def clean(ev):
        ev["needs_vlm"] = bool(ev["needs_vlm"])
        return ev

    if mode == "keyword":
        tokens = sorted(set(t for t in re.findall(r"[^\W_]+", query.lower()) if len(t) >= 2))
        if not tokens:
            return {"ok": False, "error": "query: нет слов для поиска"}
        like = " OR ".join("lower(r.vlm_description) LIKE %s" for _ in tokens)
        names, rows = _run(_SELECT + _FROM + _add_where(where, f"({like})"),
                           [variant] + params + [f"%{t}%" for t in tokens], conn)
        results = []
        for row in rows:
            ev = clean(dict(zip(names, row)))
            low = ev["vlm_description"].lower()
            ev["score"] = sum(t in low for t in tokens) / len(tokens)
            results.append(ev)
        skipped = 0
    else:
        try:
            from database import embeddings
            embedder = embedder or embeddings.get_default_embedder()
            model = model or getattr(embedder, "name", embeddings.DEFAULT_MODEL)
            qvec = embedder.embed([query])[0]
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": f"не удалось посчитать эмбеддинг запроса: {e}"}
        names, rows = _run(
            _SELECT + ", d.text_hash AS _h, d.embedding_json AS _emb " + _FROM +
            "LEFT JOIN description_embeddings d ON d.video_id = e.video_id AND d.event_id = e.event_id "
            "AND d.prompt_variant = r.prompt_variant AND d.model = %s " + where,
            [variant, model] + params, conn)
        results, skipped = [], 0
        for row in rows:
            ev = dict(zip(names, row))
            h, emb = ev.pop("_h"), ev.pop("_emb")
            if emb is None or h != embeddings.text_hash(ev["vlm_description"]):
                skipped += 1
                continue
            ev = clean(ev)
            ev["score"] = embeddings.cosine(qvec, json.loads(emb))
            results.append(ev)

    if min_score is not None:
        results = [r for r in results if r["score"] >= min_score]
    results.sort(key=lambda r: (-r["score"], r["video_id"], r["t_start"], r["event_id"]))
    results = results[:top_k]
    return {"ok": True, "mode": mode, "query": query, "vlm_variant": variant,
            "count": len(results), "results": results, "skipped_no_embedding": skipped}
