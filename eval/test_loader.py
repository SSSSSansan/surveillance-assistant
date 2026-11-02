"""Тесты загрузчика (loader.py) на SQLite, БЕЗ PostgreSQL.
Запуск из корня проекта: python -m eval.build && python -m eval.test_loader"""
import json, os, sqlite3, sys, tempfile
from database import loader
from eval import testutil

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


def snap(raw, tables=("videos", "events", "vlm_results")):
    return {t: sorted(raw.execute(f"SELECT * FROM {t}").fetchall(), key=repr) for t in tables}


def counts(raw):
    return [raw.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("videos", "events", "vlm_results")]


tmp = tempfile.TemporaryDirectory()


def write_json(name, data):
    p = os.path.join(tmp.name, name)
    json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    return p


def read_json(name):
    return json.load(open(name, encoding="utf-8"))


CV, A, B = (testutil.db_path(n) for n in ("seed_events.json", "seed_vlm_A.json", "seed_vlm_B.json"))

print("== первая загрузка ==")
raw, conn = testutil.fresh(seed=False)
r = loader.load_cv_file(CV, conn=conn)
check("CV загружен", r["ok"], r)
check("CV: 19 событий добавлено, 0 обновлено", (r["events_inserted"], r["events_updated"]) == (19, 0), r)
check("CV: VLM-таблица пуста", counts(raw) == [1, 19, 0], counts(raw))
rA = loader.load_vlm_file(A, expected_variant="A", conn=conn)
rB = loader.load_vlm_file(B, expected_variant="B", conn=conn)
check("A загружен: 4 строки", rA["ok"] and rA["results_inserted"] == 4 and rA["variants"] == ["A"], rA)
check("B загружен: 3 строки", rB["ok"] and rB["results_inserted"] == 3 and rB["variants"] == ["B"], rB)
check("итого 1 видео, 19 событий, 7 результатов VLM", counts(raw) == [1, 19, 7], counts(raw))

seed_raw, _ = testutil.fresh(seed=True)
check("загрузчик даёт те же данные, что seed.sql (videos, events, vlm_results)",
      snap(raw) == snap(seed_raw))
base = snap(raw)

print("== повторная загрузка ==")
r2 = loader.load_cv_file(CV, conn=conn)
check("CV повторно: 0 добавлено, 19 обновлено", (r2["ok"], r2["events_inserted"], r2["events_updated"]) == (True, 0, 19), r2)
a2 = loader.load_vlm_file(A, conn=conn)
b2 = loader.load_vlm_file(B, conn=conn)
check("A повторно: 0 добавлено, 4 обновлено", (a2["results_inserted"], a2["results_updated"]) == (0, 4), a2)
check("B повторно: 0 добавлено, 3 обновлено", (b2["results_inserted"], b2["results_updated"]) == (0, 3), b2)
check("после повторной загрузки данные не изменились, дубликатов нет", snap(raw) == base and counts(raw) == [1, 19, 7])

print("== CV не трогает VLM ==")
cv = read_json(CV)
cv["events"][3]["t_end"] = 27.5            # e0004 исправлен Арай
cv["events"][0]["trigger_score"] = 0.5
p = write_json("cv_fixed.json", cv)
r3 = loader.load_cv_file(p, conn=conn)
check("исправленный CV загружен", r3["ok"], r3)
check("vlm_results не изменились после перезагрузки CV",
      snap(raw, ("vlm_results",)) == {"vlm_results": base["vlm_results"]} and counts(raw)[2] == 7)
check("событие e0004 обновлено", raw.execute(
    "SELECT t_end FROM events WHERE event_id='e0004'").fetchone()[0] == 27.5)
check("описание A у e0004 на месте", raw.execute(
    "SELECT vlm_description FROM vlm_results WHERE event_id='e0004' AND prompt_variant='A'").fetchone()[0].startswith("A person"))
cvw = read_json(CV)
cvw["events"][3]["vlm_status"] = "rejected"
cvw["events"][3]["prompt_variant"] = "A"
cvw["events"][3]["vlm_description"] = "попытка подменить результат из CV-файла"
r4 = loader.load_cv_file(write_json("cv_with_vlm.json", cvw), conn=conn)
check("VLM-поля в CV-файле игнорируются с предупреждением",
      r4["ok"] and any("VLM-поля игнорируются" in w for w in r4["warnings"]), r4)
check("результат VLM не подменён через CV-файл", raw.execute(
    "SELECT vlm_status FROM vlm_results WHERE event_id='e0004' AND prompt_variant='A'").fetchone()[0] == "confirmed")
loader.load_cv_file(CV, conn=conn)  # вернуть исходные значения

print("== A и B независимы ==")
va = read_json(A)
va["events"][0]["vlm_description"] = "Changed A description."
va["events"][1]["vlm_status"] = "rejected"
rA2 = loader.load_vlm_file(write_json("A2.json", va), expected_variant="A", conn=conn)
check("изменённый A загружен", rA2["ok"], rA2)
b_rows = sorted(raw.execute("SELECT * FROM vlm_results WHERE prompt_variant='B'").fetchall())
check("строки B не изменились после загрузки A", b_rows == sorted(r for r in base["vlm_results"] if r[2] == "B"))
check("строка A обновилась", raw.execute(
    "SELECT vlm_description FROM vlm_results WHERE event_id='e0004' AND prompt_variant='A'").fetchone()[0] == "Changed A description.")
check("статус A обновился", raw.execute(
    "SELECT vlm_status FROM vlm_results WHERE event_id='e0010' AND prompt_variant='A'").fetchone()[0] == "rejected")
loader.load_vlm_file(A, conn=conn)
check("исходный A восстановлен, B и счётчики прежние", snap(raw) == base)
check("A и B одного события лежат отдельными строками",
      raw.execute("SELECT COUNT(*) FROM vlm_results WHERE event_id='e0004'").fetchone()[0] == 2)

print("== ошибки: ничего не пишется ==")
raw, conn = testutil.fresh(seed=False)
before = snap(raw)
rr = loader.load_vlm_file(A, conn=conn)
check("VLM до CV отклонён с понятной ошибкой", not rr["ok"] and "Сначала загрузите CV-файл" in rr["errors"][0], rr)
check("после отклонения база пуста", snap(raw) == before and counts(raw) == [0, 0, 0])

bad = read_json(CV)
bad["source"] = "unknown_src"
rr = loader.load_cv_file(write_json("bad_source.json", bad), conn=conn)
check("неизвестный source отклонён", not rr["ok"] and "sources" in rr["errors"][0], rr)
check("после отклонения source база пуста", counts(raw) == [0, 0, 0])

bad = read_json(CV)
bad["events"][0]["event_type"] = "fly"
bad["events"][1]["t_end"] = 1.0           # t_end < t_start
bad["events"][2]["trigger_score"] = 2
bad["events"][3]["track_id"] = "abc"
bad["events"][4]["event_id"] = "e0006"    # дубль (e0006 стоит на позиции 5)
rr = loader.load_cv_file(write_json("bad_many.json", bad), conn=conn)
check("все ошибки файла собираются сразу (>= 5, включая дубль)", not rr["ok"] and len(rr["errors"]) >= 5
      and any("повтор" in e for e in rr["errors"]), rr["errors"])
check("файл с ошибками не пишет ничего", counts(raw) == [0, 0, 0])

for name, patch in [("нет video_id", lambda d: d.pop("video_id")),
                    ("events не список", lambda d: d.update(events={})),
                    ("needs_vlm строкой", lambda d: d["events"][0].update(needs_vlm="yes")),
                    ("nan во времени", lambda d: d["events"][0].update(t_start=float("nan")))]:
    d = read_json(CV); patch(d)
    p = os.path.join(tmp.name, "x.json")
    open(p, "w", encoding="utf-8").write(json.dumps(d))   # json.dumps пишет NaN как NaN
    rr = loader.load_cv_file(p, conn=conn)
    check(f"CV: {name} -> ошибка", not rr["ok"] and rr["errors"], rr)
check("после всех ошибок база пуста", counts(raw) == [0, 0, 0])
check("несуществующий файл -> ошибка, не исключение", not loader.load_cv_file("no_such.json", conn=conn)["ok"])
open(os.path.join(tmp.name, "broken.json"), "w").write("{не json")
check("битый JSON -> ошибка, не исключение",
      not loader.load_cv_file(os.path.join(tmp.name, "broken.json"), conn=conn)["ok"])

loader.load_cv_file(CV, conn=conn)
rr = loader.load_vlm_file(B, expected_variant="A", conn=conn)
check("файл B при --variant A отклонён", not rr["ok"] and "ожидался 'A'" in rr["errors"][0], rr)
check("после этого vlm_results пуста", counts(raw)[2] == 0)
rr = loader.load_vlm_file(A, expected_variant="C", conn=conn)
check("variant 'C' отклонён", not rr["ok"])
vbad = read_json(A); vbad["events"][0]["prompt_variant"] = None
check("VLM без prompt_variant отклонён", not loader.load_vlm_file(write_json("nov.json", vbad), conn=conn)["ok"])
vdup = read_json(A); vdup["events"].append(dict(vdup["events"][0]))
check("дубль (event_id, variant) в VLM-файле отклонён",
      not loader.load_vlm_file(write_json("dup.json", vdup), conn=conn)["ok"])
vmiss = read_json(A); vmiss["events"][0]["event_id"] = "e9999"
rr = loader.load_vlm_file(write_json("miss.json", vmiss), conn=conn)
check("VLM для неизвестного события отклонён целиком", not rr["ok"] and "e9999" in rr["errors"][0], rr)
check("и ни одна другая строка из того файла не записана", counts(raw)[2] == 0)

print("== предупреждения ==")
vm = read_json(A); vm["events"][0]["t_start"] = 20.0   # в базе 24.0
rr = loader.load_vlm_file(write_json("mism.json", vm), conn=conn)
check("расхождение CV-полей в VLM-файле -> предупреждение, файл загружен",
      rr["ok"] and any("расходятся" in w for w in rr["warnings"]), rr)
vn = read_json(A)
vn["events"][0]["event_id"] = "e0001"       # e0001 - enter, needs_vlm=false
rr = loader.load_vlm_file(write_json("notneeded.json", vn), conn=conn)
check("VLM для события с needs_vlm=false -> предупреждение",
      rr["ok"] and any("needs_vlm=false" in w for w in rr["warnings"]), rr)
cz = read_json(CV); cz["events"][0]["zone_id"] = " Door "
rr = loader.load_cv_file(write_json("zone.json", cz), conn=conn)
check("zone_id приводится к нижнему регистру с предупреждением",
      rr["ok"] and any("нижнему регистру" in w for w in rr["warnings"])
      and raw.execute("SELECT zone_id FROM events WHERE event_id='e0001'").fetchone()[0] == "door", rr)
short = read_json(CV); short["events"] = short["events"][:5]
rr = loader.load_cv_file(write_json("short.json", short), conn=conn)
check("события, отсутствующие в новом CV-файле, не удаляются, есть предупреждение",
      rr["ok"] and counts(raw)[1] == 19 and any("не удалены" in w for w in rr["warnings"]), rr)

print("== другое видео не мешает ==")
raw, conn = testutil.fresh(seed=False)
loader.load_cv_file(CV, conn=conn); loader.load_vlm_file(A, conn=conn); loader.load_vlm_file(B, conn=conn)
other = read_json(CV); other["video_id"] = "test_video_02"; other["camera_id"] = "test_cam02"
check("второе видео с теми же event_id загружается",
      loader.load_cv_file(write_json("v2.json", other), conn=conn)["ok"])
check("первое видео не изменилось, всего 2 видео и 38 событий, 7 результатов VLM",
      counts(raw) == [2, 38, 7], counts(raw))

print(f"\nИТОГО провалов: {FAILS}")
tmp.cleanup()
sys.exit(1 if FAILS else 0)
