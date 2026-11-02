"""Тесты tools.py и schema.sql на SQLite (БЕЗ PostgreSQL).
Запуск из корня проекта:  python -m eval.build && python -m eval.test_tools
SQLite проверяет логику функций, правило VLM и ограничения схемы. Диалектные отличия
PostgreSQL этим не проверяются: на настоящей базе тесты не запускались."""
import csv, json, sqlite3, sys
from agent import tools
from eval import testutil

SCHEMA = testutil.read_db("schema.sql")
SEED = testutil.read_db("seed.sql")
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


class Cur:
    def __init__(self, cur): self.cur = cur
    def execute(self, sql, params=()): self.cur.execute(sql.replace("%s", "?"), params); return self
    @property
    def description(self): return self.cur.description
    def fetchall(self): return self.cur.fetchall()
    def close(self): self.cur.close()


class Shim:
    def __init__(self, raw): self.raw = raw
    def cursor(self): return Cur(self.raw.cursor())


def fresh():
    raw = sqlite3.connect(":memory:")
    raw.execute("PRAGMA foreign_keys = ON")
    raw.executescript(SCHEMA)
    raw.executescript(SEED)
    return raw, Shim(raw)


def raises(raw, sql):
    try:
        raw.execute(sql)
        return False
    except sqlite3.IntegrityError:
        return True


def ids(res):
    return {e["event_id"] for e in res["events"]}


raw, conn = fresh()

# ---------- 1. все вопросы из questions.csv (вариант A по умолчанию) ----------
print("== вопросы ==")
skipped = 0
for row in csv.DictReader(open(testutil.eval_path("questions.csv"), encoding="utf-8-sig")):
    tool, args, exp = row["expected_tool"], json.loads(row["expected_args"]), row["expected_answer"]
    if tool == "count_events":
        got = str(tools.count_events(conn=conn, **args)["count"])
    elif tool == "query_events" and row["type"] == "exists":
        got = "да" if tools.query_events(conn=conn, **args)["count"] else "нет"
    elif tool == "query_events" and row["type"] == "time":
        evs = tools.query_events(conn=conn, **args)["events"]
        got = f"{evs[0]['t_start']} с" if evs else "не нашёл"
    elif tool == "query_events" and row["type"] == "list":
        evs = tools.query_events(conn=conn, **args)["events"]
        got = "; ".join(f"человек {e['track_id']} ({e['t_start']} с)" for e in evs)
    else:
        skipped += 1  # get_track_timeline (T10) и вопросы вне базы
        continue
    check(f"{row['id']} {exp}", got == exp, got)
print(f"(пропущено {skipped}: get_track_timeline появится в T10, и вопросы вне базы)")

# ---------- 2. правило видимости ----------
print("== правило VLM ==")
default = tools.query_events(conn=conn, limit=200)
check("по умолчанию 17 событий из 19", default["count"] == 17, default["count"])
check("обычный ответ: только needs_vlm=false или confirmed",
      all((not e["needs_vlm"]) or e["vlm_status"] == "confirmed" for e in default["events"]))
check("rejected (e0018) скрыт", "e0018" not in ids(default))
check("not_checked (e0019) скрыт", "e0019" not in ids(default))
check("confirmed (e0004, e0010, e0013) видны", {"e0004", "e0010", "e0013"} <= ids(default))
check("события без VLM видны (e0001)", "e0001" in ids(default))
check("count_events согласован с query_events",
      tools.count_events(conn=conn)["count"] == default["count"])

print("== отладочные представления ==")
dq = lambda view, **kw: tools.debug_query_events(view=view, limit=200, conn=conn, **kw)
check("all = 19", dq("all")["count"] == 19)
check("unconfirmed = {e0018, e0019}", ids(dq("unconfirmed")) == {"e0018", "e0019"}, ids(dq("unconfirmed")))
check("not_checked = {e0019}", ids(dq("not_checked")) == {"e0019"})
check("rejected = {e0018}", ids(dq("rejected")) == {"e0018"})
check("debug_count_events all = 19", tools.debug_count_events(view="all", conn=conn)["count"] == 19)
check("debug-события несут статус и описание",
      next(e for e in dq("rejected")["events"])["vlm_status"] == "rejected"
      and "no object" in dq("rejected")["events"][0]["vlm_description"])
check("e0019 в отладке: not_checked, описания нет",
      dq("not_checked")["events"][0]["vlm_status"] == "not_checked"
      and dq("not_checked")["events"][0]["vlm_description"] is None)
try:
    tools.query_events(view="all", conn=conn)
    check("query_events(view=...) запрещён", False)
except TypeError:
    check("query_events(view=...) запрещён", True)

print("== варианты A и B ==")
check("put_down: A -> 1 событие (confirmed)",
      tools.query_events(event_type="put_down", vlm_variant="A", conn=conn)["count"] == 1)
check("put_down: B -> 0 (B отклонил e0010, e0019 не проверен)",
      tools.query_events(event_type="put_down", vlm_variant="B", conn=conn)["count"] == 0)
check("abandoned_object: B -> 0 (нет результата B)",
      tools.query_events(event_type="abandoned_object", vlm_variant="B", conn=conn)["count"] == 0)
check("всего видно при B = 15", tools.count_events(vlm_variant="B", conn=conn)["count"] == 15)
check("строчная буква 'b' принимается",
      tools.count_events(vlm_variant="b", conn=conn)["count"] == 15)
b_pick = tools.query_events(event_type="pick_up", vlm_variant="B", conn=conn)["events"][0]
check("описание берётся из выбранного варианта",
      b_pick["prompt_variant"] == "B" and b_pick["vlm_description"].startswith("Person 7"), b_pick)
check("B rejected = {e0010, e0018}",
      ids(tools.debug_query_events(view="rejected", vlm_variant="B", conn=conn)) == {"e0010", "e0018"})

# ---------- 3. схема ----------
print("== схема ==")
raw, conn = fresh()
n_ab = raw.execute("SELECT COUNT(*) FROM vlm_results WHERE video_id='test_video_01' AND event_id='e0004'").fetchone()[0]
check("A и B для одного события лежат отдельно (2 строки)", n_ab == 2, n_ab)
check("дубль (video_id, event_id, 'A') запрещён",
      raises(raw, "INSERT INTO vlm_results VALUES ('test_video_01','e0004','A','confirmed','x')"))
check("дубль (video_id, event_id) в events запрещён",
      raises(raw, "INSERT INTO events (video_id,event_id,event_type,t_start,t_end) VALUES ('test_video_01','e0001','enter',1,2)"))
raw.execute("INSERT INTO videos VALUES ('test_video_02','test_cam02','own')")
ok = not raises(raw, "INSERT INTO events (video_id,event_id,event_type,t_start,t_end) VALUES ('test_video_02','e0001','enter',1,2)")
check("тот же event_id в другом видео разрешён", ok)
check("count по видео разделяет одинаковые event_id",
      tools.count_events(video_id="test_video_02", conn=Shim(raw))["count"] == 1
      and tools.count_events(video_id="test_video_01", conn=Shim(raw))["count"] == 17)
check("фильтр camera_id работает",
      tools.count_events(camera_id="test_cam02", conn=Shim(raw))["count"] == 1
      and tools.count_events(camera_id="test_cam01", conn=Shim(raw))["count"] == 17)
check("vlm_results для несуществующего события запрещён (FK)",
      raises(raw, "INSERT INTO vlm_results VALUES ('test_video_01','e9999','A','confirmed','x')"))
check("видео с неизвестным source запрещено",
      raises(raw, "INSERT INTO videos VALUES ('v3','cam03','unknown_source')"))
raw.execute("INSERT INTO sources VALUES ('new_source')")
check("после INSERT INTO sources новый source допустим",
      not raises(raw, "INSERT INTO videos VALUES ('v3','cam03','new_source')"))
check("prompt_variant 'C' запрещён",
      raises(raw, "INSERT INTO vlm_results VALUES ('test_video_02','e0001','C','confirmed','x')"))
check("неизвестный vlm_status запрещён",
      raises(raw, "INSERT INTO vlm_results VALUES ('test_video_02','e0001','A','maybe','x')"))
check("неизвестный event_type запрещён",
      raises(raw, "INSERT INTO events (video_id,event_id,event_type,t_start,t_end) VALUES ('test_video_02','e2','fly',1,2)"))
check("t_end < t_start запрещён",
      raises(raw, "INSERT INTO events (video_id,event_id,event_type,t_start,t_end) VALUES ('test_video_02','e3','enter',5,2)"))
check("trigger_score > 1 запрещён",
      raises(raw, "INSERT INTO events (video_id,event_id,event_type,t_start,t_end,trigger_score) VALUES ('test_video_02','e4','enter',1,2,1.5)"))
raw.execute("DELETE FROM events WHERE video_id='test_video_01' AND event_id='e0004'")
check("удаление события каскадом удаляет его vlm_results",
      raw.execute("SELECT COUNT(*) FROM vlm_results WHERE event_id='e0004'").fetchone()[0] == 0)

# ---------- 3b. seed.sql безопасен для уже существующих данных ----------
print("== seed.sql: зоны и повторный запуск ==")
check("seed.sql не содержит DELETE FROM zones", "DELETE FROM zones" not in SEED)
check("тестовые id не похожи на реальные: нет 'meva_clip', 'cam01'",
      "meva_clip" not in SEED and "'cam01'" not in SEED and "test_video_01" in SEED and "test_cam01" in SEED)
for fn in ("seed_events.json", "seed_vlm_A.json", "seed_vlm_B.json"):
    d = json.load(open(testutil.db_path(fn), encoding="utf-8"))
    check(f"{fn}: video_id=test_video_01, camera_id=test_cam01",
          d["video_id"] == "test_video_01" and d["camera_id"] == "test_cam01")
    check(f"{fn}: clip_path не содержит meva_clip",
          all("meva_clip" not in (e["clip_path"] or "") for e in d["events"]))
pre = sqlite3.connect(":memory:")
pre.execute("PRAGMA foreign_keys = ON")
pre.executescript(SCHEMA)
pre.execute("INSERT INTO zones VALUES ('test_cam01','gate','Real gate')")      # чужая зона камеры
pre.execute("INSERT INTO zones VALUES ('test_cam01','desk','Real desk')")      # совпадает с тестовой
pre.execute("INSERT INTO zones VALUES ('test_cam01','shelf',NULL)")            # без описания
pre.executescript(SEED)
zs = dict(pre.execute("SELECT zone_id, description FROM zones WHERE camera_id='test_cam01'").fetchall())
check("существующая зона 'gate' сохранена", zs.get("gate") == "Real gate", zs)
check("описание существующей зоны 'desk' не затёрто", zs.get("desk") == "Real desk", zs)
check("недостающая зона 'door' добавлена", zs.get("door") == "Entrance door", zs)
pre.execute("UPDATE zones SET description=NULL WHERE zone_id='desk'")
pre.executescript(SEED)
check("пустое описание заполняется тестовым",
      pre.execute("SELECT description FROM zones WHERE zone_id='desk'").fetchone()[0] == "Work desk")
before = [pre.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ("videos", "zones", "events", "vlm_results")]
pre.executescript(SEED)
after_ = [pre.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
          for t in ("videos", "zones", "events", "vlm_results")]
check("повторный запуск seed.sql не создаёт дубликатов", before == after_, (before, after_))
check("после seed: видео 1, зон 4, событий 19, результатов VLM 7",
      after_ == [1, 4, 19, 7], after_)

# ---------- 4. кривые аргументы ----------
print("== кривые аргументы ==")
raw, conn = fresh()
bad_kwargs = [dict(event_type="fly"), dict(track_id="abc"), dict(track_id=7.5), dict(t_from=-1),
              dict(t_from=50, t_to=10), dict(t_from=float("nan")), dict(zone_id=""),
              dict(camera_id=""), dict(vlm_variant="C"), dict(track_id=True)]
for kw in bad_kwargs:
    r1 = tools.query_events(conn=conn, **kw)
    r2 = tools.count_events(conn=conn, **{k: v for k, v in kw.items()})
    check(f"{kw}", r1["ok"] is False and r2["ok"] is False and bool(r1["error"]))
for kw in [dict(limit=0), dict(order="sideways")]:
    check(f"query_events {kw}", tools.query_events(conn=conn, **kw)["ok"] is False)
check("debug: неизвестный view", tools.debug_query_events(view="everything", conn=conn)["ok"] is False)
check("мягкий ввод ' ENTER ', track_id='7'",
      tools.count_events(conn=conn, event_type=" ENTER ", track_id="7")["count"] == 1)
check("limit 1000 ограничен MAX_LIMIT и не падает",
      tools.query_events(conn=conn, limit=1000)["count"] == 17)

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)
