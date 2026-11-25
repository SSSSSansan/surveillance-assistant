"""Этап 2: загрузчик snapshots.json (срезы каждые N секунд) в таблицы snapshots и snapshot_people.
Проверялся только на SQLite; на PostgreSQL не запускался.

Файл приходит в два приёма, и оба грузятся одной командой:
  * Арай:  кадры + people_count, БЕЗ ключа "people"  -> videos + snapshots
  * Панда: те же срезы с "people": [...] (цвет, тип, описание) -> дозаполняет snapshot_people

Гарантии (по аналогии с database/loader.py):
  * Файл проверяется целиком; при любой ошибке в базу ничего не пишется.
  * Повторная загрузка не создаёт дубликатов (upsert по (video_id, snapshot_id)).
  * Если в срезе НЕТ ключа "people", описания людей этого среза в базе не трогаются
    (повторная загрузка файла Арай не стирает работу Панды).
    Если ключ есть (даже пустой список), люди этого среза заменяются целиком.
  * Срезы, которых нет в новом файле, не удаляются (выдаётся предупреждение).
  * Неизвестный source отклоняется (список значений: таблица sources).
  * people_count (детектор) и число описанных людей (VLM) могут различаться: это предупреждение, не ошибка.

Использование из корня проекта (нужны DATABASE_URL и psycopg2):
    python -m database.snapshots_loader --init-schema            # создать таблицы (безопасно повторять)
    python -m database.snapshots_loader snapshots.json
    python -m database.snapshots_loader snapshots.json --frames-root /путь/к/кадрам   # проверить, что кадры есть
"""
import argparse
import json
import math
import sys
from pathlib import Path

from agent import tools

SCHEMA_FILE = Path(__file__).resolve().parent / "schema_snapshots.sql"


def _is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int_like(x):
    return _is_num(x) and x == int(x)


def _nonempty_str(x):
    return isinstance(x, str) and bool(x.strip())


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


def _norm_label(value):
    """Цвет/тип одежды: нижний регистр, пробелы по краям убраны; пусто -> 'unknown'."""
    if value is None:
        return "unknown", True
    v = value.strip().lower()
    return (v, False) if v else ("unknown", True)


def _validate(data):
    """Возвращает (vid, cam, src, snapshots, errors, notes). snapshots[i]['people'] = None, если ключа нет."""
    errors, notes = [], {"unknown_filled": 0, "backslashes": 0, "count_mismatch": 0}
    head = []
    for name in ("video_id", "camera_id", "source"):
        v = data.get(name)
        if not _nonempty_str(v):
            errors.append(f"{name}: нужна непустая строка, получено {v!r}")
            head.append(None)
        else:
            head.append(v.strip())
    interval = data.get("interval_sec")
    if interval is not None and not (_is_num(interval) and interval > 0):
        errors.append(f"interval_sec: нужно положительное число, получено {interval!r}")
    raw = data.get("snapshots")
    if not isinstance(raw, list) or not raw:
        errors.append("snapshots: нужен непустой список")
        raw = []

    snaps, seen_id, seen_t = [], set(), {}
    for i, s in enumerate(raw):
        where = f"snapshots[{i}]"
        if not isinstance(s, dict):
            errors.append(f"{where}: нужен объект")
            continue
        n_err = len(errors)
        sid = s.get("snapshot_id")
        if not _nonempty_str(sid):
            errors.append(f"{where}: snapshot_id: нужна непустая строка, получено {sid!r}")
        else:
            sid = sid.strip()
            where = f"snapshots[{i}] ({sid})"
            if sid in seen_id:
                errors.append(f"{where}: повтор snapshot_id в файле")
            seen_id.add(sid)
        t = s.get("t")
        if not _is_num(t) or t < 0:
            errors.append(f"{where}: t: нужно число >= 0, получено {t!r}")
        elif t in seen_t:
            errors.append(f"{where}: t={t} уже у среза {seen_t[t]}")
        else:
            seen_t[t] = sid
        fp = s.get("frame_path")
        if fp is not None and not _nonempty_str(fp):
            errors.append(f"{where}: frame_path: строка или null, получено {fp!r}")
            fp = None
        elif fp is not None:
            if "\\" in fp:
                notes["backslashes"] += 1
            fp = fp.strip().replace("\\", "/")
        pc = s.get("people_count")
        if not _int_like(pc) or pc < 0:
            errors.append(f"{where}: people_count: нужно целое число >= 0, получено {pc!r}")
        people = None
        if "people" in s:
            raw_people = s["people"]
            if not isinstance(raw_people, list):
                errors.append(f"{where}: people: нужен список")
            else:
                people, idxs = [], set()
                for j, p in enumerate(raw_people):
                    pw = f"{where}: people[{j}]"
                    if not isinstance(p, dict):
                        errors.append(f"{pw}: нужен объект")
                        continue
                    idx = p.get("person_idx")
                    if not _int_like(idx) or idx < 0:
                        errors.append(f"{pw}: person_idx: нужно целое число >= 0, получено {idx!r}")
                        continue
                    if int(idx) in idxs:
                        errors.append(f"{pw}: повтор person_idx {int(idx)}")
                    idxs.add(int(idx))
                    labels = []
                    for key in ("clothing_color", "clothing_type"):
                        v = p.get(key)
                        if v is not None and not isinstance(v, str):
                            errors.append(f"{pw}: {key}: нужна строка, получено {v!r}")
                            v = None
                        lab, filled = _norm_label(v)
                        notes["unknown_filled"] += filled
                        labels.append(lab)
                    desc = p.get("description")
                    if desc is not None and not isinstance(desc, str):
                        errors.append(f"{pw}: description: строка или null, получено {desc!r}")
                        desc = None
                    people.append({"person_idx": int(idx), "clothing_color": labels[0],
                                   "clothing_type": labels[1], "description": desc})
                if people is not None and _int_like(pc) and len(people) != int(pc):
                    notes["count_mismatch"] += 1
        if len(errors) == n_err:
            snaps.append({"snapshot_id": sid, "t": float(t), "frame_path": fp,
                          "people_count": int(pc), "people": people})
    return head[0], head[1], head[2], snaps, errors, notes


def _fetch(cur, sql, params=()):
    cur.execute(sql, list(params))
    return cur.fetchall()


def apply_schema(conn=None):
    """Создаёт таблицы snapshots и snapshot_people (если их нет)."""
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    try:
        if hasattr(conn, "raw"):            # тестовая прослойка над sqlite3
            conn.raw.executescript(sql)
        else:
            cur = conn.cursor()
            cur.execute(sql)
            cur.close()
        conn.commit()
    finally:
        if own:
            conn.close()


def load_snapshots_file(path, conn=None, frames_root=None):
    """Загружает snapshots.json. Возвращает отчёт {"ok", "errors", "warnings", ...}."""
    rep = {"ok": False, "kind": "snapshots", "file": str(path), "errors": [], "warnings": []}
    data, errs = _load_json(path)
    if errs:
        rep["errors"] = errs
        return rep
    vid, cam, src, snaps, errors, notes = _validate(data)
    if errors:
        rep["errors"] = errors
        return rep
    if notes["unknown_filled"]:
        rep["warnings"].append(f"пустой цвет/тип одежды заменён на 'unknown' ({notes['unknown_filled']} значений)")
    if notes["backslashes"]:
        rep["warnings"].append(f"в frame_path у {notes['backslashes']} срезов обратные слэши заменены на '/'")
    if notes["count_mismatch"]:
        rep["warnings"].append(f"у {notes['count_mismatch']} срезов people_count (детектор) не равен числу "
                               f"описанных людей (VLM); сохранено как есть")
    if frames_root is not None:
        root = Path(frames_root)
        missing = [s["snapshot_id"] for s in snaps if s["frame_path"] and not (root / s["frame_path"]).is_file()]
        if missing:
            rep["warnings"].append(f"нет файла кадра у {len(missing)} срезов в {root} (например {missing[:3]})")
    no_frame = sum(1 for s in snaps if not s["frame_path"])
    if no_frame:
        rep["warnings"].append(f"у {no_frame} срезов нет frame_path: фронт покажет заглушку")

    own = conn is None
    conn = conn if conn is not None else tools._connect()
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
        existing = {r[0] for r in _fetch(cur, "SELECT snapshot_id FROM snapshots WHERE video_id = %s", [vid])}
        cur.execute(
            "INSERT INTO videos (video_id, camera_id, source) VALUES (%s, %s, %s) "
            "ON CONFLICT (video_id) DO UPDATE SET camera_id = excluded.camera_id, source = excluded.source",
            [vid, cam, src])
        people_rows, replaced = 0, 0
        for s in snaps:
            cur.execute(
                "INSERT INTO snapshots (video_id, snapshot_id, t, frame_path, people_count) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (video_id, snapshot_id) DO UPDATE SET t = excluded.t, "
                "frame_path = excluded.frame_path, people_count = excluded.people_count",
                [vid, s["snapshot_id"], s["t"], s["frame_path"], s["people_count"]])
            if s["people"] is None:
                continue
            cur.execute("DELETE FROM snapshot_people WHERE video_id = %s AND snapshot_id = %s",
                        [vid, s["snapshot_id"]])
            replaced += 1
            for p in s["people"]:
                cur.execute(
                    "INSERT INTO snapshot_people (video_id, snapshot_id, person_idx, clothing_color, "
                    "clothing_type, description) VALUES (%s, %s, %s, %s, %s, %s)",
                    [vid, s["snapshot_id"], p["person_idx"], p["clothing_color"], p["clothing_type"],
                     p["description"]])
                people_rows += 1
        conn.commit()
        in_file = {s["snapshot_id"] for s in snaps}
        missing = existing - in_file
        if missing:
            rep["warnings"].append(f"{len(missing)} срезов видео {vid} есть в базе, но нет в файле; не удалены")
        rep.update(ok=True, video_id=vid, snapshots_inserted=len(in_file - existing),
                   snapshots_updated=len(in_file & existing), snapshots_with_people_replaced=replaced,
                   people_rows=people_rows)
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
    p = argparse.ArgumentParser(description="Загрузка snapshots.json в базу (этап 2)")
    p.add_argument("files", nargs="*", help="snapshots.json (можно несколько)")
    p.add_argument("--init-schema", action="store_true", help="создать таблицы snapshots и snapshot_people")
    p.add_argument("--frames-root", default=None, help="папка, относительно которой лежат frame_path")
    a = p.parse_args(argv)
    if not a.files and not a.init_schema:
        p.error("укажите файл или --init-schema")
    bad = False
    try:
        if a.init_schema:
            apply_schema()
            print("таблицы snapshots и snapshot_people готовы")
        for f in a.files:
            res = load_snapshots_file(f, frames_root=a.frames_root)
            print(json.dumps(res, ensure_ascii=False, indent=2))
            bad = bad or not res["ok"]
    except RuntimeError as e:  # нет DATABASE_URL, нет psycopg2
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
