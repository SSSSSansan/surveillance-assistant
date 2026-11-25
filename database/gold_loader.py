"""Этап 3 (T15): таблицы эталона и журнала прогонов, загрузка файлов, запись разметки, экспорт в CSV.
Проверялся только на SQLite; на PostgreSQL не запускался.

Что грузим (команды из корня проекта, нужны DATABASE_URL и psycopg2):
    py -m database.gold_loader --init-schema
    py -m database.gold_loader runs runs.jsonl                       # журнал прогонов (Сания)
    py -m database.gold_loader crops video1_crops_index.json         # кропы для разметки (Арай)
    py -m database.gold_loader frames video1_frames_index.json       # кадры для счёта людей (Арай)
    py -m database.gold_loader import-frames eval/gold/gold_frames_Aigerim.csv   # готовая разметка кадров
    py -m database.gold_loader export eval/gold                      # gold_people.csv и gold_frames.csv

Гарантии: файл проверяется целиком, при ошибке в базу ничего не пишется; повторная загрузка не создаёт
дубликатов (upsert); разметка записывается по ключу с разметчиком, поэтому разметчики не затирают друг друга.
"""
import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from agent import tools
try:
    from vlm.describers.base import COLORS, TYPES
except ImportError:        # в ветке нет vlm/describers (ещё не подтянут PR Арай): та же копия словаря
    COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
              "black", "white", "gray", "beige", "multicolor", "unknown")
    TYPES = ("sweater", "hoodie", "t-shirt", "shirt", "jacket", "coat", "dress",
             "pants_only", "vest", "uniform", "other", "unknown")

SCHEMA_FILE = Path(__file__).resolve().parent / "schema_gold.sql"
VISIBLE = ("да", "спиной", "частично", "не видно")
PEOPLE_COLS = ["video_id", "snapshot_id", "person_idx", "color", "color_2", "type", "visible", "box_h", "labeler"]
FRAMES_COLS = ["video_id", "snapshot_id", "true_people_count", "labeler"]
RUN_FIELDS = ["run_id", "video_id", "method", "model", "mode", "prompt_variant", "weights", "started_at",
              "duration_sec", "params"]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _str(x):
    return isinstance(x, str) and bool(x.strip())


def _now():
    return datetime.now().isoformat(timespec="seconds")


def apply_schema(conn=None):
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


def _fetch(conn, sql, params=()):
    cur = conn.cursor()
    try:
        cur.execute(sql, list(params))
        return cur.fetchall()
    finally:
        cur.close()


def _write(conn, own, rep, statements):
    """statements: [(sql, params)]. Всё или ничего."""
    cur = conn.cursor()
    try:
        for sql, params in statements:
            cur.execute(sql, params)
        conn.commit()
        rep["ok"] = True
    except Exception as ex:  # noqa: BLE001
        conn.rollback()
        rep["errors"].append(f"ошибка записи в базу, изменения отменены: {ex}")
    finally:
        cur.close()
        if own:
            conn.close()
    return rep


def _read_json(path, want):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f), []
    except FileNotFoundError:
        return None, [f"файл не найден: {path}"]
    except (OSError, ValueError) as e:
        return None, [f"не удалось прочитать {path}: {e}"]


# ---------- runs.jsonl ----------

def load_runs(path, conn=None):
    rep = {"ok": False, "kind": "runs", "file": str(path), "errors": [], "warnings": []}
    rows, seen = [], set()
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except (OSError, ValueError) as e:
        rep["errors"].append(f"не удалось прочитать {path}: {e}")
        return rep
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            rep["errors"].append(f"строка {i}: не JSON ({e})")
            continue
        if not isinstance(r, dict):
            rep["errors"].append(f"строка {i}: нужен JSON-объект")
            continue
        for k in ("run_id", "video_id", "method"):
            if not _str(r.get(k)):
                rep["errors"].append(f"строка {i}: {k}: нужна непустая строка, получено {r.get(k)!r}")
        d = r.get("duration_sec")
        if d is not None and not (_num(d) and d >= 0):
            rep["errors"].append(f"строка {i}: duration_sec: число >= 0 или null, получено {d!r}")
        if _str(r.get("run_id")):
            if r["run_id"] in seen:
                rep["warnings"].append(f"строка {i}: повтор run_id {r['run_id']}; берётся последняя строка")
            seen.add(r["run_id"])
        rows.append(r)
    if rep["errors"]:
        return rep
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    st = []
    for r in rows:
        w = r.get("weights")
        vals = [r["run_id"], r["video_id"], r["method"], r.get("model"), r.get("mode"),
                r.get("prompt_variant"), None if w is None else str(w), r.get("started_at"),
                r.get("duration_sec"), json.dumps(r.get("params"), ensure_ascii=False, sort_keys=True)]
        st.append(("INSERT INTO runs (" + ", ".join(RUN_FIELDS) + ") VALUES (" + ", ".join(["%s"] * 10) + ") "
                   "ON CONFLICT (run_id) DO UPDATE SET video_id = excluded.video_id, method = excluded.method, "
                   "model = excluded.model, mode = excluded.mode, prompt_variant = excluded.prompt_variant, "
                   "weights = excluded.weights, started_at = excluded.started_at, "
                   "duration_sec = excluded.duration_sec, params = excluded.params", vals))
    rep["rows"] = len(rows)
    return _write(conn, own, rep, st)


# ---------- crops_index.json / frames_index.json ----------

def load_crops_index(path, conn=None):
    rep = {"ok": False, "kind": "crops", "file": str(path), "errors": [], "warnings": []}
    data, errs = _read_json(path, list)
    if errs:
        rep["errors"] = errs
        return rep
    if not isinstance(data, list) or not data:
        rep["errors"].append("корень файла должен быть непустым списком")
        return rep
    rows, keys = [], set()
    for i, c in enumerate(data):
        w = f"[{i}]"
        if not isinstance(c, dict):
            rep["errors"].append(f"{w}: нужен объект")
            continue
        for k in ("video_id", "snapshot_id", "crop_id"):
            if not _str(c.get(k)):
                rep["errors"].append(f"{w}: {k}: нужна непустая строка, получено {c.get(k)!r}")
        pi = c.get("person_idx")
        if not (_num(pi) and pi == int(pi) and pi >= 0):
            rep["errors"].append(f"{w}: person_idx: целое число >= 0, получено {pi!r}")
        if not isinstance(c.get("kept"), bool):
            rep["errors"].append(f"{w}: kept: нужно true/false, получено {c.get('kept')!r}")
        if rep["errors"]:
            continue
        k = (c["video_id"], c["snapshot_id"], int(pi))
        if k in keys:
            rep["errors"].append(f"{w}: повтор ключа {k}")
        keys.add(k)
        rows.append([c["video_id"], c["snapshot_id"], int(pi), c["crop_id"], c.get("t"), c.get("box_h"),
                     c.get("box_w"), c.get("h_bucket"), c["kept"], c.get("reason"), c.get("frame_path"),
                     c.get("path"), c.get("full_path")])
    if rep["errors"]:
        return rep
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    st = [("INSERT INTO gold_crops (video_id, snapshot_id, person_idx, crop_id, t, box_h, box_w, h_bucket, kept, "
           "reason, frame_path, torso_path, full_path) VALUES (" + ", ".join(["%s"] * 13) + ") "
           "ON CONFLICT (video_id, snapshot_id, person_idx) DO UPDATE SET crop_id = excluded.crop_id, "
           "t = excluded.t, box_h = excluded.box_h, box_w = excluded.box_w, h_bucket = excluded.h_bucket, "
           "kept = excluded.kept, reason = excluded.reason, frame_path = excluded.frame_path, "
           "torso_path = excluded.torso_path, full_path = excluded.full_path", r) for r in rows]
    rep.update(rows=len(rows), kept=sum(1 for r in rows if r[8]), dropped=sum(1 for r in rows if not r[8]))
    return _write(conn, own, rep, st)


def load_frames_index(path, conn=None):
    rep = {"ok": False, "kind": "frames", "file": str(path), "errors": [], "warnings": []}
    data, errs = _read_json(path, list)
    if errs:
        rep["errors"] = errs
        return rep
    if not isinstance(data, list) or not data:
        rep["errors"].append("корень файла должен быть непустым списком")
        return rep
    rows, keys = [], set()
    for i, c in enumerate(data):
        w = f"[{i}]"
        if not isinstance(c, dict):
            rep["errors"].append(f"{w}: нужен объект")
            continue
        for k in ("video_id", "snapshot_id"):
            if not _str(c.get(k)):
                rep["errors"].append(f"{w}: {k}: нужна непустая строка, получено {c.get(k)!r}")
        if rep["errors"]:
            continue
        k = (c["video_id"], c["snapshot_id"])
        if k in keys:
            rep["errors"].append(f"{w}: повтор ключа {k}")
        keys.add(k)
        rows.append([c["video_id"], c["snapshot_id"], c.get("t"), c.get("frame_path")])   # people_count не берём
    if rep["errors"]:
        return rep
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    st = [("INSERT INTO gold_frames_index (video_id, snapshot_id, t, frame_path) VALUES (%s, %s, %s, %s) "
           "ON CONFLICT (video_id, snapshot_id) DO UPDATE SET t = excluded.t, frame_path = excluded.frame_path", r)
          for r in rows]
    rep["rows"] = len(rows)
    return _write(conn, own, rep, st)


# ---------- запись разметки (для страницы T18 и для импорта CSV) ----------

def save_people_label(conn, video_id, snapshot_id, person_idx, labeler, color, type_, visible,
                      color_2=None, box_h=None):
    """Одна разметка кропа. Возвращает (ok, сообщение). Повторная запись того же разметчика заменяет прежнюю."""
    if not _str(labeler):
        return False, "укажите имя разметчика"
    if color not in COLORS:
        return False, f"цвет {color!r} не из словаря"
    if color_2 in ("", "—"):
        color_2 = None
    if color_2 is not None and color_2 not in COLORS:
        return False, f"color_2 {color_2!r} не из словаря"
    if type_ not in TYPES:
        return False, f"тип {type_!r} не из словаря"
    if visible not in VISIBLE:
        return False, f"visible {visible!r}: допустимо {', '.join(VISIBLE)}"
    if box_h is None:
        r = _fetch(conn, "SELECT box_h FROM gold_crops WHERE video_id = %s AND snapshot_id = %s AND person_idx = %s",
                   [video_id, snapshot_id, int(person_idx)])
        box_h = r[0][0] if r else None
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO labels_gold_people (video_id, snapshot_id, person_idx, labeler, color, color_2, type, "
            "visible, box_h, labeled_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (video_id, snapshot_id, person_idx, labeler) DO UPDATE SET color = excluded.color, "
            "color_2 = excluded.color_2, type = excluded.type, visible = excluded.visible, "
            "box_h = excluded.box_h, labeled_at = excluded.labeled_at",
            [video_id, snapshot_id, int(person_idx), labeler.strip(), color, color_2, type_, visible, box_h, _now()])
        conn.commit()
        return True, "сохранено"
    except Exception as ex:  # noqa: BLE001
        conn.rollback()
        return False, f"ошибка записи: {ex}"
    finally:
        cur.close()


def save_frame_label(conn, video_id, snapshot_id, labeler, true_people_count):
    if not _str(labeler):
        return False, "укажите имя разметчика"
    if not (_num(true_people_count) and true_people_count == int(true_people_count) and true_people_count >= 0):
        return False, f"число людей: целое >= 0, получено {true_people_count!r}"
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO labels_gold_frames (video_id, snapshot_id, labeler, true_people_count, labeled_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (video_id, snapshot_id, labeler) DO UPDATE SET "
            "true_people_count = excluded.true_people_count, labeled_at = excluded.labeled_at",
            [video_id, snapshot_id, labeler.strip(), int(true_people_count), _now()])
        conn.commit()
        return True, "сохранено"
    except Exception as ex:  # noqa: BLE001
        conn.rollback()
        return False, f"ошибка записи: {ex}"
    finally:
        cur.close()


def import_frames_csv(path, conn=None):
    """Загружает готовый gold_frames_<имя>.csv. Пустые true_people_count (не размечено) пропускаются."""
    rep = {"ok": False, "kind": "frames_csv", "file": str(path), "errors": [], "warnings": []}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, ValueError) as e:
        rep["errors"].append(f"не удалось прочитать {path}: {e}")
        return rep
    good, skipped = [], 0
    for i, r in enumerate(rows, 2):
        v = (r.get("true_people_count") or "").strip()
        if not v:
            skipped += 1
            continue
        try:
            n = int(v)
            assert n >= 0
        except (ValueError, AssertionError):
            rep["errors"].append(f"строка {i}: true_people_count: целое >= 0, получено {v!r}")
            continue
        if not (_str(r.get("video_id")) and _str(r.get("snapshot_id")) and _str(r.get("labeler"))):
            rep["errors"].append(f"строка {i}: нужны video_id, snapshot_id, labeler")
            continue
        good.append((r["video_id"].strip(), r["snapshot_id"].strip(), r["labeler"].strip(), n))
    if rep["errors"]:
        return rep
    if skipped:
        rep["warnings"].append(f"{skipped} строк без числа пропущено (не размечено)")
    own = conn is None
    conn = conn if conn is not None else tools._connect()
    ts = _now()
    st = [("INSERT INTO labels_gold_frames (video_id, snapshot_id, labeler, true_people_count, labeled_at) "
           "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (video_id, snapshot_id, labeler) DO UPDATE SET "
           "true_people_count = excluded.true_people_count, labeled_at = excluded.labeled_at",
           [v, s, l, n, ts]) for v, s, l, n in good]
    rep["rows"] = len(good)
    return _write(conn, own, rep, st)


# ---------- чтение для страницы и экспорт ----------

def next_crops(conn, labeler, video_id=None, limit=1):
    """Кропы (kept) без разметки этого разметчика. Сначала те, где уже одна чужая оценка (нужна вторая), потом без оценок, в конце с двумя:
    так каждый кроп быстрее получает вторую независимую оценку. Чужие значения НЕ возвращаем."""
    q = ("SELECT c.video_id, c.snapshot_id, c.person_idx, c.crop_id, c.box_h, c.full_path, "
         "(SELECT COUNT(*) FROM labels_gold_people l WHERE l.video_id = c.video_id AND l.snapshot_id = c.snapshot_id "
         " AND l.person_idx = c.person_idx) AS n_labels "
         "FROM gold_crops c WHERE c.kept = %s "
         "AND NOT EXISTS (SELECT 1 FROM labels_gold_people l WHERE l.video_id = c.video_id "
         "AND l.snapshot_id = c.snapshot_id AND l.person_idx = c.person_idx AND l.labeler = %s) ")
    params = [True, labeler]
    if video_id:
        q += "AND c.video_id = %s "
        params.append(video_id)
    q += "ORDER BY c.video_id, c.snapshot_id, c.person_idx"
    rows = _fetch(conn, q, params)
    # порядок: сначала кропы с одной чужой оценкой (им нужна вторая), потом без оценок, в конце с двумя и больше
    rows.sort(key=lambda r: 0 if r[6] == 1 else (1 if r[6] == 0 else 2))
    keys = ["video_id", "snapshot_id", "person_idx", "crop_id", "box_h", "full_path", "n_labels"]
    return [dict(zip(keys, r)) for r in rows[:limit]]


def next_frames(conn, labeler, video_id=None, limit=1):
    """Кадры без счёта этого разметчика, по порядку времени. Счёт модели и чужие числа не возвращаем."""
    q = ("SELECT f.video_id, f.snapshot_id, f.t, f.frame_path FROM gold_frames_index f "
         "WHERE NOT EXISTS (SELECT 1 FROM labels_gold_frames l WHERE l.video_id = f.video_id "
         "AND l.snapshot_id = f.snapshot_id AND l.labeler = %s) ")
    params = [labeler]
    if video_id:
        q += "AND f.video_id = %s "
        params.append(video_id)
    q += "ORDER BY f.video_id, f.t, f.snapshot_id"
    return [dict(zip(["video_id", "snapshot_id", "t", "frame_path"], r)) for r in _fetch(conn, q, params)[:limit]]


def gold_videos(conn):
    """Видео, у которых есть кропы или кадры для разметки."""
    rows = _fetch(conn, "SELECT video_id FROM gold_crops UNION SELECT video_id FROM gold_frames_index ORDER BY 1")
    return [r[0] for r in rows]


def delete_people_label(conn, video_id, snapshot_id, person_idx, labeler):
    """Отмена своей разметки кропа (кнопка Undo). Чужие строки не трогаем."""
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM labels_gold_people WHERE video_id = %s AND snapshot_id = %s AND person_idx = %s "
                    "AND labeler = %s", [video_id, snapshot_id, int(person_idx), labeler])
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
        raise
    finally:
        cur.close()


def delete_frame_label(conn, video_id, snapshot_id, labeler):
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM labels_gold_frames WHERE video_id = %s AND snapshot_id = %s AND labeler = %s",
                    [video_id, snapshot_id, labeler])
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
        raise
    finally:
        cur.close()


def progress(conn, labeler):
    people = _fetch(conn, "SELECT COUNT(*) FROM labels_gold_people WHERE labeler = %s", [labeler])[0][0]
    total = _fetch(conn, "SELECT COUNT(*) FROM gold_crops WHERE kept = %s", [True])[0][0]
    frames = _fetch(conn, "SELECT COUNT(*) FROM labels_gold_frames WHERE labeler = %s", [labeler])[0][0]
    n_frames = _fetch(conn, "SELECT COUNT(*) FROM gold_frames_index")[0][0]
    return {"people_done": people, "people_total": total, "frames_done": frames, "frames_total": n_frames}


def export_csv(conn, out_dir):
    """gold_people.csv и gold_frames.csv в формате eval/score_methods.py. Строк разметки НЕ склеиваем:
    одна строка на (кроп, разметчик), большинство голосов считает score_methods."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    people = _fetch(conn, "SELECT video_id, snapshot_id, person_idx, color, color_2, type, visible, box_h, labeler "
                          "FROM labels_gold_people ORDER BY video_id, snapshot_id, person_idx, labeler")
    frames = _fetch(conn, "SELECT video_id, snapshot_id, true_people_count, labeler FROM labels_gold_frames "
                          "ORDER BY video_id, snapshot_id, labeler")
    for name, cols, rows in (("gold_people.csv", PEOPLE_COLS, people), ("gold_frames.csv", FRAMES_COLS, frames)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow(["" if v is None else v for v in r])
    return {"people_rows": len(people), "frames_rows": len(frames), "dir": str(out)}


def main(argv=None):
    p = argparse.ArgumentParser(description="Эталон разметки и журнал прогонов (этап 3)")
    p.add_argument("--init-schema", action="store_true", help="создать таблицы (безопасно повторять)")
    sub = p.add_subparsers(dest="cmd")
    for name, h in (("runs", "runs.jsonl"), ("crops", "crops_index.json"), ("frames", "frames_index.json"),
                    ("import-frames", "gold_frames_<имя>.csv")):
        sub.add_parser(name, help=h).add_argument("file", nargs="+")
    sub.add_parser("export", help="выгрузить gold_people.csv и gold_frames.csv").add_argument("out_dir")
    a = p.parse_args(argv)
    if not a.cmd and not a.init_schema:
        p.error("укажите команду или --init-schema")
    bad = False
    try:
        if a.init_schema:
            apply_schema()
            print("таблицы эталона готовы")
        fn = {"runs": load_runs, "crops": load_crops_index, "frames": load_frames_index,
              "import-frames": import_frames_csv}
        if a.cmd in fn:
            for f in a.file:
                res = fn[a.cmd](f)
                print(json.dumps(res, ensure_ascii=False, indent=2))
                bad = bad or not res["ok"]
        elif a.cmd == "export":
            conn = tools._connect()
            try:
                print(json.dumps(export_csv(conn, a.out_dir), ensure_ascii=False, indent=2))
            finally:
                conn.close()
    except RuntimeError as e:  # нет DATABASE_URL, нет psycopg2
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())