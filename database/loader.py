"""T10: загрузчик events.json в Postgres (проверялся только на SQLite).

Два вида файлов:
  * CV-файл (Арай):    load_cv_file   -> videos + events      (VLM-поля из файла игнорируются)
  * VLM-файл (Панда):  load_vlm_file  -> vlm_results          (вариант A или B берётся из файла)

Гарантии:
  * Повторная загрузка того же файла не создаёт дубликатов: везде upsert по ключам схемы.
  * Загрузка CV не удаляет и не меняет vlm_results (events обновляется через upsert,
    а не через DELETE + INSERT, иначе ON DELETE CASCADE стёр бы результаты VLM).
  * Варианты A и B независимы: строка (video_id, event_id, prompt_variant) обновляется
    только своим файлом.
  * Файл сначала проверяется целиком. Если есть хотя бы одна ошибка, в базу ничего не пишется.
  * События, которых нет в новом CV-файле, не удаляются (о них выдаётся предупреждение).

Использование:
    export DATABASE_URL=postgresql://user:pass@host:5432/db
    (команды запускать из корня проекта)
    python -m database.loader cv  database/seed_events.json
    python -m database.loader vlm database/seed_vlm_A.json --variant A
    python -m database.loader vlm database/seed_vlm_B.json --variant B
    python -m database.loader embed          # посчитать эмбеддинги описаний (нужна модель BGE-M3)
"""
import argparse
import json
import math
import sys

from agent import tools
from database.snapshots_loader import load_snapshots_file

VLM_STATUSES = ("not_checked", "confirmed", "rejected")


# ---------- чтение и проверка файла ----------

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None, [f"файл не найден: {path}"]
    except (OSError, ValueError) as e:
        return None, [f"не удалось прочитать {path}: {e}"]
    if not isinstance(data, dict):
        return None, ["корень файла должен быть JSON-объектом"]
    return data, []


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _header(data, errors):
    out = []
    for name in ("video_id", "camera_id", "source"):
        v = data.get(name)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"{name}: нужна непустая строка, получено {v!r}")
            out.append(None)
        else:
            out.append(v.strip())
    events = data.get("events")
    if not isinstance(events, list):
        errors.append("events: нужен список")
        events = []
    return out[0], out[1], out[2], events


def _event(ev, i, kind, errors, notes):
    """Проверяет и нормализует одно событие. kind: 'cv' или 'vlm'."""
    eid = ev.get("event_id") if isinstance(ev, dict) else None
    label = f"events[{i}]" + (f" ({eid})" if isinstance(eid, str) and eid else "")
    if not isinstance(ev, dict):
        errors.append(f"{label}: событие должно быть объектом")
        return None
    local = []
    bad = lambda msg: local.append(f"{label}: {msg}")
    cv = (kind == "cv")

    if not isinstance(eid, str) or not eid.strip():
        bad(f"event_id: нужна непустая строка, получено {eid!r}")
    out = {"event_id": eid.strip() if isinstance(eid, str) else None}

    et = ev.get("event_type")
    if et is None and not cv:
        out["event_type"] = None
    elif et not in tools.EVENT_TYPES:
        bad(f"event_type: допустимы {', '.join(tools.EVENT_TYPES)}, получено {et!r}")
    else:
        out["event_type"] = et

    tr = ev.get("track_id")
    if tr is None:
        out["track_id"] = None
    elif _is_num(tr) and float(tr) == int(tr) and tr >= 0:
        out["track_id"] = int(tr)
    else:
        bad(f"track_id: нужно целое число >= 0 или null, получено {tr!r}")

    z = ev.get("zone_id")
    if z is None:
        out["zone_id"] = None
    elif isinstance(z, str) and z.strip():
        out["zone_id"] = z.strip().lower()  # tools ищет по зоне в нижнем регистре
        if out["zone_id"] != z:
            notes["zone_normalized"] += 1
    else:
        bad(f"zone_id: нужна непустая строка или null, получено {z!r}")

    for name in ("t_start", "t_end"):
        v = ev.get(name)
        if v is None and not cv:
            out[name] = None
        elif not _is_num(v) or v < 0:
            bad(f"{name}: нужно конечное число >= 0, получено {v!r}")
        else:
            out[name] = float(v)
    if out.get("t_start") is not None and out.get("t_end") is not None and out["t_end"] < out["t_start"]:
        bad(f"t_end ({out['t_end']}) меньше t_start ({out['t_start']})")

    ts = ev.get("trigger_score")
    if ts is None:
        out["trigger_score"] = None
    elif _is_num(ts) and 0 <= ts <= 1:
        out["trigger_score"] = float(ts)
    else:
        bad(f"trigger_score: нужно число от 0 до 1 или null, получено {ts!r}")

    nv = ev.get("needs_vlm", False if cv else None)
    if nv is None or isinstance(nv, bool):
        out["needs_vlm"] = nv
    else:
        bad(f"needs_vlm: нужно true или false, получено {nv!r}")

    cp = ev.get("clip_path")
    if cp is None or (isinstance(cp, str) and cp.strip()):
        out["clip_path"] = cp
    else:
        bad(f"clip_path: нужна непустая строка или null, получено {cp!r}")

    st = ev.get("vlm_status", "not_checked")
    if st not in VLM_STATUSES:
        bad(f"vlm_status: допустимы {', '.join(VLM_STATUSES)}, получено {st!r}")
    out["vlm_status"] = st
    d = ev.get("vlm_description")
    if d is not None and not isinstance(d, str):
        bad(f"vlm_description: нужна строка или null, получено {d!r}")
    out["vlm_description"] = d

    pv = ev.get("prompt_variant")
    if isinstance(pv, str):
        pv = pv.strip().upper()
    if pv is None and cv:
        out["prompt_variant"] = None
    elif pv in ("A", "B"):
        out["prompt_variant"] = pv
    else:
        bad(f"prompt_variant: нужно 'A' или 'B', получено {ev.get('prompt_variant')!r}")

    errors.extend(local)
    return None if local else out


def _validate(data, kind, expected_variant=None):
    errors, notes = [], {"zone_normalized": 0}
    vid, cam, src, raw_events = _header(data, errors)
    events, seen = [], set()
    for i, ev in enumerate(raw_events):
        e = _event(ev, i, kind, errors, notes)
        if e is None:
            continue
        key = e["event_id"] if kind == "cv" else (e["event_id"], e["prompt_variant"])
        if key in seen:
            errors.append(f"events[{i}]: повтор {'event_id' if kind == 'cv' else '(event_id, prompt_variant)'} {key!r} в файле")
        seen.add(key)
        if kind == "vlm" and expected_variant and e["prompt_variant"] != expected_variant:
            errors.append(f"events[{i}] ({e['event_id']}): prompt_variant {e['prompt_variant']!r}, "
                          f"а ожидался {expected_variant!r}")
        events.append(e)
    return vid, cam, src, events, errors, notes


# ---------- работа с базой ----------

def _fetch(cur, sql, params=()):
    cur.execute(sql, list(params))
    return cur.fetchall()


def _open(conn):
    return (conn, False) if conn is not None else (tools._connect(), True)


def _report(kind, path):
    return {"ok": False, "kind": kind, "file": path, "errors": [], "warnings": []}


def load_cv_file(path, conn=None):
    """Загружает CV-файл в videos и events. Таблицу vlm_results не трогает."""
    rep = _report("cv", path)
    data, errs = _load_json(path)
    if errs:
        rep["errors"] = errs
        return rep
    vid, cam, src, events, errors, notes = _validate(data, "cv")
    if errors:
        rep["errors"] = errors
        return rep
    if notes["zone_normalized"]:
        rep["warnings"].append(f"zone_id приведён к нижнему регистру у {notes['zone_normalized']} событий")
    ignored = sum(1 for e in events if e["vlm_status"] != "not_checked" or e["vlm_description"] is not None
                  or e["prompt_variant"] is not None)
    if ignored:
        rep["warnings"].append(f"VLM-поля игнорируются в CV-файле (у {ignored} событий); "
                               f"результаты VLM загружаются отдельными файлами")

    conn, own = _open(conn)
    cur = conn.cursor()
    try:
        sources = {r[0] for r in _fetch(cur, "SELECT source FROM sources")}
        if src not in sources:
            rep["errors"].append(f"source {src!r} нет в таблице sources ({', '.join(sorted(sources))}). "
                                 f"Добавьте значение командой INSERT INTO sources после подтверждения командой")
            return rep
        prev = _fetch(cur, "SELECT camera_id, source FROM videos WHERE video_id = %s", [vid])
        if prev and tuple(prev[0]) != (cam, src):
            rep["warnings"].append(f"видео {vid} уже было в базе с camera_id={prev[0][0]!r}, source={prev[0][1]!r}; "
                                   f"заменено на {cam!r}, {src!r}")
        existing = {r[0] for r in _fetch(cur, "SELECT event_id FROM events WHERE video_id = %s", [vid])}
        cur.execute(
            "INSERT INTO videos (video_id, camera_id, source) VALUES (%s, %s, %s) "
            "ON CONFLICT (video_id) DO UPDATE SET camera_id = excluded.camera_id, source = excluded.source",
            [vid, cam, src])
        for e in events:
            cur.execute(
                "INSERT INTO events (video_id, event_id, event_type, track_id, zone_id, t_start, t_end, "
                "trigger_score, needs_vlm, clip_path) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (video_id, event_id) DO UPDATE SET event_type = excluded.event_type, "
                "track_id = excluded.track_id, zone_id = excluded.zone_id, t_start = excluded.t_start, "
                "t_end = excluded.t_end, trigger_score = excluded.trigger_score, "
                "needs_vlm = excluded.needs_vlm, clip_path = excluded.clip_path",
                [vid, e["event_id"], e["event_type"], e["track_id"], e["zone_id"], e["t_start"],
                 e["t_end"], e["trigger_score"], bool(e["needs_vlm"]), e["clip_path"]])
        conn.commit()
        in_file = {e["event_id"] for e in events}
        missing = existing - in_file
        if missing:
            rep["warnings"].append(f"{len(missing)} событий видео {vid} есть в базе, но нет в файле; не удалены")
        rep.update(ok=True, video_id=vid, events_inserted=len(in_file - existing),
                   events_updated=len(in_file & existing))
        return rep
    except Exception as ex:  # noqa: BLE001
        conn.rollback()
        rep["errors"].append(f"ошибка записи в базу, изменения отменены: {ex}")
        return rep
    finally:
        cur.close()
        if own:
            conn.close()


def load_vlm_file(path, expected_variant=None, conn=None):
    """Загружает VLM-файл в vlm_results. Таблицы events и videos не меняет.
    expected_variant ('A' или 'B'): отклонить файл, если в нём другой вариант."""
    rep = _report("vlm", path)
    if expected_variant is not None:
        expected_variant = str(expected_variant).strip().upper()
        if expected_variant not in ("A", "B"):
            rep["errors"].append(f"variant: допустимо A или B, получено {expected_variant!r}")
            return rep
    data, errs = _load_json(path)
    if errs:
        rep["errors"] = errs
        return rep
    # VLM output files may contain the full CV event list. Entries that were
    # never processed by VLM have no prompt variant and no description; they
    # are not rows for vlm_results (prompt_variant is NOT NULL in the schema).
    raw_events = data.get("events", [])
    skipped_unprocessed = 0
    if isinstance(raw_events, list):
        filtered_events = []
        for event in raw_events:
            if (
                isinstance(event, dict)
                and event.get("prompt_variant") is None
                and event.get("vlm_status", "not_checked") == "not_checked"
                and event.get("vlm_description") is None
            ):
                skipped_unprocessed += 1
                continue
            filtered_events.append(event)
        data = dict(data)
        data["events"] = filtered_events

    vid, cam, src, events, errors, notes = _validate(data, "vlm", expected_variant)
    if errors:
        rep["errors"] = errors
        return rep
    if skipped_unprocessed:
        rep["warnings"].append(
            f"Пропущено необработанных событий без prompt_variant: {skipped_unprocessed}"
        )

    conn, own = _open(conn)
    cur = conn.cursor()
    try:
        db_events = {r[0]: r for r in _fetch(
            cur, "SELECT event_id, event_type, track_id, t_start, t_end, needs_vlm FROM events "
                 "WHERE video_id = %s", [vid])}
        missing = sorted({e["event_id"] for e in events} - set(db_events))
        if missing:
            shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
            rep["errors"].append(f"событий нет в базе для видео {vid}: {shown}. Сначала загрузите CV-файл")
            return rep
        vrow = _fetch(cur, "SELECT camera_id, source FROM videos WHERE video_id = %s", [vid])
        if vrow and tuple(vrow[0]) != (cam, src):
            rep["warnings"].append(f"camera_id/source в файле ({cam!r}, {src!r}) не совпадают с базой {tuple(vrow[0])}")
        mism, not_needed = [], []
        for e in events:
            r = db_events[e["event_id"]]
            for name, idx in (("event_type", 1), ("track_id", 2), ("t_start", 3), ("t_end", 4)):
                v = e.get(name)
                if v is not None and (abs(v - r[idx]) > 1e-9 if isinstance(v, float) and r[idx] is not None
                                      else v != r[idx]):
                    mism.append(f"{e['event_id']}.{name}: файл {v!r}, база {r[idx]!r}")
            if not r[5]:
                not_needed.append(e["event_id"])
        if mism:
            rep["warnings"].append(f"CV-поля в VLM-файле расходятся с базой ({len(mism)}): " + "; ".join(mism[:5]))
        if not_needed:
            rep["warnings"].append(f"результаты VLM для событий с needs_vlm=false (в обычных ответах не влияют): "
                                   + ", ".join(sorted(set(not_needed))[:10]))
        existing = {tuple(r) for r in _fetch(
            cur, "SELECT event_id, prompt_variant FROM vlm_results WHERE video_id = %s", [vid])}
        for e in events:
            cur.execute(
                "INSERT INTO vlm_results (video_id, event_id, prompt_variant, vlm_status, vlm_description) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (video_id, event_id, prompt_variant) DO UPDATE SET "
                "vlm_status = excluded.vlm_status, vlm_description = excluded.vlm_description",
                [vid, e["event_id"], e["prompt_variant"], e["vlm_status"], e["vlm_description"]])
        conn.commit()
        keys = {(e["event_id"], e["prompt_variant"]) for e in events}
        rep.update(ok=True, video_id=vid, variants=sorted({k[1] for k in keys}),
                   results_inserted=len(keys - existing), results_updated=len(keys & existing))
        return rep
    except Exception as ex:  # noqa: BLE001
        conn.rollback()
        rep["errors"].append(f"ошибка записи в базу, изменения отменены: {ex}")
        return rep
    finally:
        cur.close()
        if own:
            conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="Загрузка events.json в базу")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cv", help="CV-файл (videos + events)")
    c.add_argument("files", nargs="+")
    v = sub.add_parser("vlm", help="VLM-файл (vlm_results)")
    v.add_argument("files", nargs="+")
    v.add_argument("--variant", choices=["A", "B"], help="ожидаемый вариант подсказки в файле")
    sn = sub.add_parser("snapshots", help="snapshots.json (срезы, этап 2)")
    sn.add_argument("files", nargs="+")
    e = sub.add_parser("embed", help="посчитать эмбеддинги описаний")
    e.add_argument("--model", default=None)
    args = p.parse_args(argv)

    bad = False
    try:
        if args.cmd == "embed":
            from database import embeddings
            conn = tools._connect()
            try:
                res = embeddings.build_embeddings(conn, embeddings.get_default_embedder(), args.model)
            finally:
                conn.close()
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0 if res.get("ok") else 1
        for f in args.files:
            res = (load_cv_file(f) if args.cmd == "cv" else
                   load_snapshots_file(f) if args.cmd == "snapshots" else load_vlm_file(f, args.variant))
            print(json.dumps(res, ensure_ascii=False, indent=2))
            bad = bad or not res["ok"]
    except RuntimeError as e:  # нет DATABASE_URL, нет psycopg2 и т. п.
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
