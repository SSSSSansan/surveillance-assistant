"""Тесты этапа 2: схема срезов, загрузчик snapshots.json, функции агента по срезам и помощники архива.
Только SQLite и заглушки. PostgreSQL, настоящая LLM и настоящий Streamlit здесь НЕ запускались:
страница проверяется на ЗАГЛУШКЕ модуля streamlit (ловит ошибки нашего кода, не совместимость с Streamlit).
Запуск из корня проекта: python -m eval.test_snapshots"""
import contextlib, copy, inspect, io, json, os, runpy, sqlite3, sys, tempfile
from pathlib import Path
from unittest import mock

from agent import archive_helpers as ah
from agent import runner, snapshot_tools as st_, tool_schemas
from agent.demo_db import DB_DIR, Shim, make_demo_db
from database import snapshots_loader as sl

ROOT = Path(__file__).resolve().parents[1]
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


def base_db(cls=Shim):
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.execute("PRAGMA foreign_keys = ON")
    raw.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    c = cls(raw)
    sl.apply_schema(c)
    return c


def n(conn, sql):
    cur = conn.cursor(); cur.execute(sql); v = cur.fetchall()[0][0]; cur.close(); return v


TMP = tempfile.TemporaryDirectory()


def write(name, obj):
    p = Path(TMP.name) / name
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return str(p)


def person(i, color="red", typ="sweater", desc="d"):
    return {"person_idx": i, "clothing_color": color, "clothing_type": typ, "description": desc}


def snap(sid, t, count, people="ABSENT", fp="frames/x.jpg"):
    s = {"snapshot_id": sid, "t": t, "frame_path": fp, "people_count": count}
    if people != "ABSENT":
        s["people"] = people
    return s


def doc(snaps, **kw):
    d = {"video_id": "v1", "camera_id": "c1", "source": "meva", "interval_sec": 30, "snapshots": snaps}
    d.update(kw)
    return d


SEED = json.loads((DB_DIR / "seed_snapshots.json").read_text(encoding="utf-8"))
SEED_PEOPLE = sum(len(s["people"]) for s in SEED["snapshots"])

# ---------- схема и загрузчик ----------
print("== схема и загрузчик ==")
db = base_db()
rep = sl.load_snapshots_file(DB_DIR / "seed_snapshots.json", conn=db)
check("тестовые срезы загружаются", rep["ok"] and rep["snapshots_inserted"] == 10 and rep["people_rows"] == SEED_PEOPLE, rep)
check("в базе 10 срезов и все люди", n(db, "SELECT COUNT(*) FROM snapshots") == 10
      and n(db, "SELECT COUNT(*) FROM snapshot_people") == SEED_PEOPLE)
check("видео создано из заголовка файла", n(db, "SELECT COUNT(*) FROM videos WHERE video_id='test_video_01'") == 1)
rep2 = sl.load_snapshots_file(DB_DIR / "seed_snapshots.json", conn=db)
check("повторная загрузка: без дублей", rep2["ok"] and rep2["snapshots_inserted"] == 0 and rep2["snapshots_updated"] == 10
      and n(db, "SELECT COUNT(*) FROM snapshots") == 10 and n(db, "SELECT COUNT(*) FROM snapshot_people") == SEED_PEOPLE)

arai = copy.deepcopy(SEED)
for s in arai["snapshots"]:
    s.pop("people")
arai["snapshots"][3]["people_count"] = 5
r = sl.load_snapshots_file(write("arai.json", arai), conn=db)
check("файл Арай (без people) не стирает описания Панды",
      r["ok"] and r["snapshots_with_people_replaced"] == 0 and n(db, "SELECT COUNT(*) FROM snapshot_people") == SEED_PEOPLE, r)
check("people_count при этом обновился", n(db, "SELECT people_count FROM snapshots WHERE snapshot_id='sn_0003'") == 5)

panda = copy.deepcopy(SEED)
panda["snapshots"][2]["people"] = [person(0, "Blue ", "T-Shirt", "new")]
panda["snapshots"][3]["people"] = []
r = sl.load_snapshots_file(write("panda.json", panda), conn=db)
check("people заменяются целиком только у срезов, где ключ есть",
      r["ok"] and n(db, "SELECT COUNT(*) FROM snapshot_people WHERE snapshot_id='sn_0002'") == 1
      and n(db, "SELECT COUNT(*) FROM snapshot_people WHERE snapshot_id='sn_0003'") == 0
      and n(db, "SELECT COUNT(*) FROM snapshot_people WHERE snapshot_id='sn_0004'") == 2, r)
check("цвет и тип приводятся к нижнему регистру без пробелов",
      n(db, "SELECT COUNT(*) FROM snapshot_people WHERE snapshot_id='sn_0002' AND clothing_color='blue' "
            "AND clothing_type='t-shirt'") == 1)
check("повторная загрузка с тем же файлом: прежних людей среза не осталось",
      n(db, "SELECT COUNT(*) FROM snapshot_people WHERE snapshot_id='sn_0002' AND clothing_color='red'") == 0)

BAD = {
    "нет video_id": doc([snap("a", 0, 0)], video_id=""),
    "нет camera_id": {k: v for k, v in doc([snap("a", 0, 0)]).items() if k != "camera_id"},
    "пустой список срезов": doc([]),
    "snapshots не список": doc("x"),
    "interval_sec <= 0": doc([snap("a", 0, 0)], interval_sec=0),
    "повтор snapshot_id": doc([snap("a", 0, 0), snap("a", 30, 0)]),
    "повтор t": doc([snap("a", 30, 0), snap("b", 30, 0)]),
    "t отрицательное": doc([snap("a", -1, 0)]),
    "t строка": doc([snap("a", "5", 0)]),
    "people_count отрицательный": doc([snap("a", 0, -1)]),
    "people_count дробный": doc([snap("a", 0, 2.5)]),
    "people_count bool": doc([snap("a", 0, True)]),
    "people_count нет": doc([{"snapshot_id": "a", "t": 0}]),
    "snapshot_id пустой": doc([snap("", 0, 0)]),
    "people не список": doc([snap("a", 0, 1, {"x": 1})]),
    "повтор person_idx": doc([snap("a", 0, 2, [person(0), person(0, "blue")])]),
    "person_idx отрицательный": doc([snap("a", 0, 1, [person(-1)])]),
    "цвет не строка": doc([snap("a", 0, 1, [person(0, 5)])]),
    "description не строка": doc([snap("a", 0, 1, [person(0, "red", "sweater", 7)])]),
    "frame_path число": doc([snap("a", 0, 0, fp=5)]),
    "одна плохая запись среди хороших": doc([snap("a", 0, 0), snap("b", -5, 0), snap("c", 60, 0)]),
}
for name, d in BAD.items():
    fresh = base_db()
    r = sl.load_snapshots_file(write("bad.json", d), conn=fresh)
    check(f"ошибка «{name}»: отклонено, в базу ничего не записано",
          (not r["ok"]) and r["errors"] and n(fresh, "SELECT COUNT(*) FROM videos") == 0
          and n(fresh, "SELECT COUNT(*) FROM snapshots") == 0, r["errors"][:1])
r = sl.load_snapshots_file(write("bad2.json", doc([snap("a", 0, 0)], source="nowhere")), conn=(f := base_db()))
check("неизвестный source отклонён, видео не создано", not r["ok"] and "sources" in r["errors"][0]
      and n(f, "SELECT COUNT(*) FROM videos") == 0, r)
check("файл не найден", not sl.load_snapshots_file("/нет/такого.json", conn=base_db())["ok"])
Path(TMP.name, "broken.json").write_text("{не json", encoding="utf-8")
check("битый JSON -> ошибка, не исключение", not sl.load_snapshots_file(str(Path(TMP.name) / "broken.json"), conn=base_db())["ok"])
Path(TMP.name, "list.json").write_text("[1]", encoding="utf-8")
check("корень-массив -> ошибка", not sl.load_snapshots_file(str(Path(TMP.name) / "list.json"), conn=base_db())["ok"])

f = base_db()
r = sl.load_snapshots_file(write("w.json", doc([
    snap("a", 0, 2, [person(0, "", None), person(1)], fp="frames\\a.jpg"),
    snap("b", 30, 3, [person(0)]),
    snap("c", 60, 1, fp=None)])), conn=f, frames_root=TMP.name)
w = " ".join(r["warnings"])
check("предупреждения: unknown, слэши, несовпадение числа людей, нет кадров, нет frame_path",
      r["ok"] and "unknown" in w and "слэши" in w and "не равен" in w and "нет файла кадра" in w and "нет frame_path" in w, r)
check("пустой цвет -> unknown, None -> unknown",
      n(f, "SELECT COUNT(*) FROM snapshot_people WHERE clothing_color='unknown' AND clothing_type='unknown'") == 1)
check("обратные слэши заменены", n(f, "SELECT COUNT(*) FROM snapshots WHERE frame_path='frames/a.jpg'") == 1)
r = sl.load_snapshots_file(write("w2.json", doc([snap("a", 0, 2)], camera_id="c2")), conn=f)
check("смена camera_id: предупреждение; отсутствующие срезы не удалены (предупреждение)",
      r["ok"] and any("camera_id" in x for x in r["warnings"]) and any("не удалены" in x for x in r["warnings"])
      and n(f, "SELECT COUNT(*) FROM snapshots") == 3, r)

class BoomShim(Shim):
    def cursor(self):
        c = super().cursor()
        orig = c.execute
        def ex(sql, params=()):
            if sql.startswith("INSERT INTO snapshot_people"):
                raise RuntimeError("boom")
            return orig(sql, params)
        c.execute = ex
        return c

b = base_db(BoomShim)
r = sl.load_snapshots_file(write("boom.json", doc([snap("a", 0, 1, [person(0)])])), conn=b)
check("сбой записи: откат, в базе пусто", not r["ok"] and "boom" in r["errors"][0]
      and n(b, "SELECT COUNT(*) FROM snapshots") == 0 and n(b, "SELECT COUNT(*) FROM videos") == 0, r)

d = base_db()
sl.load_snapshots_file(DB_DIR / "seed_snapshots.json", conn=d)
d.raw.execute("DELETE FROM videos WHERE video_id='test_video_01'")
check("удаление видео каскадно удаляет срезы и людей",
      n(d, "SELECT COUNT(*) FROM snapshots") == 0 and n(d, "SELECT COUNT(*) FROM snapshot_people") == 0)
for sql, name in (("INSERT INTO snapshots VALUES ('v','s',-1,NULL,0)", "t < 0"),
                  ("INSERT INTO snapshots VALUES ('v','s',0,NULL,-1)", "people_count < 0"),
                  ("INSERT INTO snapshots VALUES ('нет_видео','s',0,NULL,0)", "срез без видео"),
                  ("INSERT INTO snapshot_people (video_id,snapshot_id,person_idx) VALUES ('v','нет',0)", "человек без среза")):
    try:
        d.raw.execute("INSERT INTO videos VALUES ('v','c','meva') ON CONFLICT DO NOTHING")
        d.raw.execute(sql)
        check(f"схема отклоняет: {name}", False)
    except sqlite3.IntegrityError:
        check(f"схема отклоняет: {name}", True)

def run_cli(argv, env=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.dict(os.environ, env or {}):
        os.environ.pop("DATABASE_URL", None)
        try:
            code = sl.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()

check("CLI без аргументов -> код 2", run_cli([])[0] == 2)
code, out, err = run_cli([str(DB_DIR / "seed_snapshots.json")])
check("CLI без DATABASE_URL: понятная ошибка, код 1, без traceback", code == 1 and "Ошибка" in err and "Traceback" not in err, err)
code, out, err = run_cli(["--init-schema"])
check("CLI --init-schema без DATABASE_URL: код 1", code == 1 and "Ошибка" in err)
code, out, err = run_cli(["/нет/файла.json"])
check("CLI: файл не найден -> код 1 и отчёт", code == 1 and "файл не найден" in out)

# ---------- функции агента ----------
print("== функции агента по срезам ==")
demo = make_demo_db()
ts = lambda r: [m["t"] for m in r["matches"]]
r = st_.search_by_clothing(color="red", clothing_type="sweater", conn=demo)
check("красная кофта: найдено на 60 и 90 с", r["ok"] and r["found"] and ts(r) == [60.0, 90.0] and r["people_matched"] == 2, r)
check("в совпадении есть кадр и человек", r["matches"][0]["frame_path"].endswith("sn_0002.jpg")
      and r["matches"][0]["people"][0]["clothing_color"] == "red")
check("запрос по-русски («красная кофта») даёт то же самое",
      ts(st_.search_by_clothing(color="красная", clothing_type="кофта", conn=demo)) == [60.0, 90.0])
check("падежи («красную кофту»)", ts(st_.search_by_clothing(color="красную", clothing_type="кофту", conn=demo)) == [60.0, 90.0])
check("зелёная куртка: 150 и 180 с", ts(st_.search_by_clothing(color="зелёная", clothing_type="куртка", conn=demo)) == [150.0, 180.0])
check("окно t_from=160: только 180 с", ts(st_.search_by_clothing(color="green", t_from=160, conn=demo)) == [180.0])
check("окно t_to=100: зелёного нет", not st_.search_by_clothing(color="green", t_to=100, conn=demo)["found"])
check("только цвет: синий на 90, 120, 180", ts(st_.search_by_clothing(color="blue", conn=demo)) == [90.0, 120.0, 180.0])
check("только тип: футболка", ts(st_.search_by_clothing(clothing_type="футболка", conn=demo)) == [90.0, 120.0, 180.0])
sw = ts(st_.search_by_clothing(clothing_type="sweater", conn=demo))
check("'sweater' строгий: hoodie не входит (60, 90, 180, 240)", sw == [60.0, 90.0, 180.0, 240.0], sw)
hd = ts(st_.search_by_clothing(clothing_type="худи", conn=demo))
check("«худи» и «толстовка» = hoodie: 180, 210", hd == [180.0, 210.0] and ts(st_.search_by_clothing(clothing_type="толстовка", conn=demo)) == [180.0, 210.0], hd)
kf = st_.search_by_clothing(clothing_type="кофта", conn=demo)
check("«кофта» = sweater И hoodie: 60, 90, 180, 210, 240", ts(kf) == [60.0, 90.0, 180.0, 210.0, 240.0] and kf["query"]["clothing_type"] == "hoodie/sweater", kf["query"])
check("«свитер» = только sweater", ts(st_.search_by_clothing(clothing_type="свитер", conn=demo)) == [60.0, 90.0, 180.0, 240.0])
check("'grey' = 'gray'", ts(st_.search_by_clothing(color="grey", conn=demo)) == [180.0, 240.0])
check("красная футболка: нет, и честно сказано про неопределённого человека",
      (lambda x: not x["found"] and x["uncertain_people"] == 1 and "not certain" in x["note"])(
          st_.search_by_clothing(color="red", clothing_type="t-shirt", conn=demo)))
r = st_.search_by_clothing(color="unknown", conn=demo)
check("'unknown' как цвет запроса = нет фильтра -> ошибка, а не ложное «нет»", r["ok"] is False and "error" in r, r)
r = st_.search_by_clothing(color="red", video_id="", t_from="", t_to="", clothing_type="", conn=demo)
check("пустые строки вместо отсутствующих аргументов (привычка слабых моделей) работают", r["ok"] and r["found"], r)
for args, name in (({}, "нет color и type"), ({"color": 5}, "color не строка"), ({"color": "red", "t_from": 50, "t_to": 10}, "t_from > t_to"),
                   ({"color": "red", "t_from": -5}, "t_from < 0"), ({"color": "red", "t_from": True}, "t_from bool"),
                   ({"color": "red", "t_to": "abc"}, "t_to не число"), ({"color": "red", "t_from": float("nan")}, "nan")):
    r = st_.search_by_clothing(conn=demo, **args)
    check(f"ошибка ввода: {name}", r["ok"] is False and "error" in r, r)
r = st_.search_by_clothing(color="red", video_id="нет_такого", conn=demo)
check("выдуманный video_id при единственном видео: берётся это видео, с пометкой (не ложное «нет»)",
      r["ok"] and r["found"] and r["matches"][0]["video_id"] == "test_video_01" and "not found" in r["video_id_note"]
      and "test_video_01" in r["video_id_note"], r)
r = st_.search_by_clothing(color="red", video_id="test_video_01", conn=demo)
check("настоящий video_id: пометки нет", r["found"] and "video_id_note" not in r)
for gen in ("any", "all", "clothes", "одежда", "Any"):
    r = st_.search_by_clothing(color="red", clothing_type=gen, conn=demo)
    check(f"clothing_type='{gen}' = без фильтра по типу", r["ok"] and r["found"] and r["query"]["clothing_type"] is None, r)
r = st_.search_by_clothing(color="any", clothing_type="any", conn=demo)
check("и цвет, и тип 'any' -> ошибка ввода", r["ok"] is False and "error" in r, r)
check("голубой = blue, бордовый = red, тёмно-синий = blue",
      st_.normalize_color("голубой") == "blue" and st_.normalize_color("бордовая") == "red"
      and st_.normalize_color("тёмно-синий") == "blue" and st_.normalize_color("dark blue") == "blue")
check("справка про примеры из описаний: 'test_video_01' не встречается в описаниях функций",
      "test_video_01" not in json.dumps(tool_schemas.SPECS, ensure_ascii=False) and "test_cam01" not in json.dumps(tool_schemas.SPECS, ensure_ascii=False))
check("подсказка запрещает выдумывать video_id", "Never invent video_id" in runner.SYSTEM_PROMPT)

x = base_db()
sl.load_snapshots_file(write("m.json", doc([
    snap("a", 0, 2, [person(0, "red", "t-shirt"), person(1, "blue", "sweater")]),
    snap("b", 30, 2),                                     # люди есть, но не описаны
    snap("c", 60, 1, [person(0, "dark blue", "jacket")]),
    snap("d", 90, 0, [])])), conn=x)
check("цвет и тип относятся к одному человеку (красный свитер не находится)",
      not st_.search_by_clothing(color="red", clothing_type="sweater", conn=x)["found"])
check("'blue' находит 'dark blue'", ts(st_.search_by_clothing(color="blue", conn=x)) == [0.0, 60.0])
check("'dark blue' = 'blue' (как в словаре VLM)", ts(st_.search_by_clothing(color="dark blue", conn=x)) == [0.0, 60.0])
r = st_.search_by_clothing(color="green", conn=x)
check("неописанный срез считается и попадает в пометку", r["snapshots_not_described"] == 1 and "not certain" in r["note"], r)
check("срез с 0 людей не считается неописанным", st_.get_snapshot_summary(conn=x)["snapshots"][3]["described"] is True
      and st_.get_snapshot_summary(conn=x)["snapshots"][1]["described"] is False)

r = st_.count_people_at(t=90, conn=demo)
check("count_people_at: точное попадание", r["found"] and r["people_count"] == 3 and r["t"] == 90.0 and r["distance_sec"] == 0.0, r)
r = st_.count_people_at(t=100, conn=demo)
check("ближайший срез и расстояние до него", r["t"] == 90.0 and r["distance_sec"] == 10.0 and r["requested_t"] == 100.0)
check("на равном расстоянии берётся более ранний срез", st_.count_people_at(t=105, conn=demo)["t"] == 90.0)
r = st_.count_people_at(t=1000, conn=demo)
check("далеко за концом видео: ответ с большим distance_sec (допуск не зашит)", r["found"] and r["distance_sec"] == 730.0)
check("t в виде строки '95' принимается", st_.count_people_at(t="95", conn=demo)["t"] == 90.0)
for args, name in (({}, "нет t"), ({"t": ""}, "пустой t"), ({"t": -1}, "t < 0"), ({"t": "скоро"}, "t не число"), ({"t": True}, "t bool")):
    r = st_.count_people_at(conn=demo, **args)
    check(f"count_people_at ошибка: {name}", r["ok"] is False and "error" in r, r)
r = st_.count_people_at(t=10, video_id="нет_такого", conn=demo)
check("count_people_at: выдуманный video_id при единственном видео -> это видео и пометка",
      r["ok"] and r["found"] and r["video_id"] == "test_video_01" and "video_id_note" in r, r)

two = base_db()
sl.load_snapshots_file(write("v1.json", doc([snap("a", 0, 1, [person(0)])])), conn=two)
sl.load_snapshots_file(write("v2.json", doc([snap("a", 0, 7, [])], video_id="v2")), conn=two)
r = st_.count_people_at(t=0, conn=two)
check("два видео и нет video_id -> ошибка со списком видео", r["ok"] is False and "v1" in r["error"] and "v2" in r["error"], r)
check("с video_id работает", st_.count_people_at(t=0, video_id="v2", conn=two)["people_count"] == 7)
check("summary: два видео и нет video_id -> ошибка", st_.get_snapshot_summary(conn=two)["ok"] is False)
check("list_snapshot_videos", st_.list_snapshot_videos(two) == ["v1", "v2"])
check("поиск без video_id идёт по всем видео", len(st_.search_by_clothing(color="red", conn=two)["matches"]) == 1)

s = st_.get_snapshot_summary(conn=demo)
check("summary: 10 срезов, максимум 4 в момент 180 с, среднее 1.6",
      s["snapshots_count"] == 10 and s["max_people"] == 4 and s["max_people_t"] == [180.0] and s["avg_people"] == 1.6, s)
check("summary: срезы по порядку времени", [x["t"] for x in s["snapshots"]] == sorted(x["t"] for x in s["snapshots"]))
tie = base_db()
sl.load_snapshots_file(write("tie.json", doc([snap("a", 0, 3, []), snap("b", 30, 1, []), snap("c", 60, 3, [])])), conn=tie)
check("максимум в нескольких моментах: перечислены все", st_.get_snapshot_summary(conn=tie)["max_people_t"] == [0.0, 60.0])
check("summary: выдуманный video_id при единственном видео -> это видео и пометка",
      (lambda r: r["ok"] and r["snapshots_count"] == 10 and "video_id_note" in r)(st_.get_snapshot_summary(video_id="нет", conn=demo)))
r = st_.search_by_clothing(color="red", video_id="vid001", conn=two)
check("несколько видео и неизвестный video_id -> ошибка со списком доступных видео",
      r["ok"] is False and "unknown video_id" in r["error"] and "v1" in r["error"] and "v2" in r["error"], r)
check("count_people_at и summary: то же при нескольких видео",
      st_.count_people_at(t=0, video_id="vid001", conn=two)["ok"] is False
      and st_.get_snapshot_summary(video_id="vid001", conn=two)["ok"] is False)

print("== защита «Да/Нет» от противоречия с результатом функции ==")
from agent import runner as rn_
Step = rn_.ToolStep
yes_res = {"ok": True, "found": True, "matches": [{"t": 270.0}, {"t": 300.0}]}
no_res = {"ok": True, "found": False, "matches": [], "uncertain_people": 0, "snapshots_not_described": 0}
unc_res = {"ok": True, "found": False, "matches": [], "uncertain_people": 3, "snapshots_not_described": 1}
fx = rn_.fix_clothing_answer
check("модель написала «Нет», а found=true -> ответ из результата функции",
      fx("Нет, не нашёл.", [Step("search_by_clothing", {}, yes_res)], "Был ли человек в красной кофте?")
      == "Да. Найдено на срезах: 270 с, 300 с.")
check("английский вопрос -> английский ответ",
      fx("No", [Step("search_by_clothing", {}, yes_res)], "Was there a red sweater?") == "Yes. Found in snapshots at: 270 s, 300 s.")
check("модель написала «Да», а found=false -> «Нет»", fx("Да", [Step("search_by_clothing", {}, no_res)], "красная куртка?") == "Нет, не нашёл.")
unc_ans = fx("Да", [Step("search_by_clothing", {}, unc_res)], "красная куртка?")
check("found=false и есть неопределённые -> оговорка про неточность", "не точно" in unc_ans and "3 чел." in unc_ans, unc_ans)
check("ответ модели согласован с результатом -> не меняется",
      fx("Да, в 270 с и 300 с (в кадре двое).", [Step("search_by_clothing", {}, yes_res)], "q") == "Да, в 270 с и 300 с (в кадре двое)."
      and fx("Нет.", [Step("search_by_clothing", {}, no_res)], "q") == "Нет.")
check("ответ не да/нет (например, число) -> не меняется", fx("Двое.", [Step("search_by_clothing", {}, yes_res)], "q") == "Двое.")
check("больше одного вызова -> ответ модели не трогаем",
      fx("Нет", [Step("search_by_clothing", {}, yes_res), Step("search_by_clothing", {}, no_res)], "q") == "Нет")
check("другая функция или ok:false -> не трогаем",
      fx("Нет", [Step("count_people_at", {}, yes_res)], "q") == "Нет"
      and fx("Нет", [Step("search_by_clothing", {}, {"ok": False, "error": "x"})], "q") == "Нет")
check("промпт: про ok:false и язык ответа в конце",
      "never answer \"not found\"" in rn_.SYSTEM_PROMPT and rn_.SYSTEM_PROMPT.rstrip().endswith("unless the question is in Chinese."))

class _Seq:
    """Слабая модель: вызывает функцию с выдуманным video_id и после результата отвечает «Нет»."""
    def __init__(self):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "search_by_clothing", "arguments": json.dumps({"color": "red", "clothing_type": "кофта", "video_id": "vid001"})}}]}
        return {"role": "assistant", "content": "Нет, не нашёл."}


r = rn_.run_agent("Был ли человек в красной кофте?", _Seq(), demo)
check("сквозной сценарий: выдуманный video_id + «Нет» модели -> «Да» по данным, без ошибки",
      not r.error and r.steps[0].result.get("found") and r.answer.startswith("Да") and "60 с" in r.answer, (r.error, r.answer))

class _Loop:
    """Модель, которая бесконечно повторяет один и тот же вызов и ни разу не отвечает текстом."""
    def __init__(self, name, args):
        self.name, self.args, self.n = name, args, 0

    def chat(self, messages, tools=None):
        self.n += 1
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{self.n}", "type": "function", "function": {
            "name": self.name, "arguments": json.dumps(self.args)}}]}


lp = _Loop("search_by_clothing", {"color": "red"})
r = rn_.run_agent("Когда был кто-то в красной или синей одежде?", lp, demo)
check("зацикливание на search_by_clothing: ответ собран из результата функции, без ошибки",
      not r.error and r.answer.startswith("Да. Найдено на срезах: 60 с") and len(r.steps) == 2 and lp.n == 2, (r.error, r.answer, len(r.steps)))
check("повторный вызов помечен для модели", "repeated_call" in r.steps[1].result and "repeated_call" not in r.steps[0].result)
r = rn_.run_agent("How many people at 90 s?", _Loop("count_people_at", {"t": 90}), demo)
check("зацикливание на count_people_at: ответ из результата на языке вопроса",
      not r.error and "3 people" in r.answer and "90 s" in r.answer, (r.error, r.answer))
r = rn_.run_agent("покажи входы", _Loop("count_events", {"event_type": "enter"}), demo)
check("зацикливание на других функциях: ответа из результата нет, остаётся ошибка (запасной ответ только для срезов)",
      r.answer is None and "5 шагов" in r.error, (r.answer, r.error))
check("fallback_answer: разные функции или ошибка -> None",
      rn_.fallback_answer([Step("search_by_clothing", {}, yes_res), Step("count_people_at", {}, yes_res)], "q") is None
      and rn_.fallback_answer([Step("search_by_clothing", {}, {"ok": False, "error": "x"})], "q") is None
      and rn_.fallback_answer([], "q") is None)
full = tool_schemas.call_tool("search_by_clothing", {"color": "red"}, demo)
slim = runner.slim_for_model(full)
m0 = slim["matches"][0]
check("slim: у людей только цвет и тип, без описаний; без snapshot_id и video_id в срезах; время и people_count на месте",
      set(m0["people"][0]) == {"clothing_color", "clothing_type"} and "snapshot_id" not in m0 and "video_id" not in m0
      and m0["t"] == 60.0 and m0["people_count"] == 2, m0)
check("slim: результат для модели короче полного", len(json.dumps(slim)) < len(json.dumps(full)) * 0.7)
sn = runner.slim_for_model(tool_schemas.call_tool("count_people_at", {"t": 90}, demo))
check("slim: count_people_at — люди без описаний", all(set(p) <= {"clothing_color", "clothing_type"} for p in sn["people"]) and sn["people_count"] == 3)

class _Compound:
    """Составной вопрос: поиск, потом один и тот же count_people_at по кругу; текстом отвечает только когда функций нет."""
    def __init__(self):
        self.n, self.tool_free_calls = 0, 0

    def chat(self, messages, tools=None):
        self.n += 1
        if not tools:
            self.tool_free_calls += 1
            return {"role": "assistant", "content": "Красное было на срезах 60-300 с; на 90 с людей: 3."}
        if self.n == 1:
            name, args = "search_by_clothing", {"color": "red"}
        else:
            name, args = "count_people_at", {"t": 90}
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{self.n}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(args)}}]}


cp = _Compound()
r = rn_.run_agent("Когда была красная одежда и сколько людей на 90 с?", cp, demo)
check("составной вопрос + зацикливание: останавливаемся на повторе и просим ответ без функций",
      not r.error and r.answer.startswith("Красное было") and cp.tool_free_calls == 1 and len(r.steps) == 3
      and cp.n == 4, (r.error, r.answer, len(r.steps), cp.n))


class _Stubborn(_Compound):
    def chat(self, messages, tools=None):
        self.n += 1
        return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{self.n}", "type": "function", "function": {
            "name": "count_events", "arguments": json.dumps({"event_type": "enter", "zone_id": f"z{self.n}"})}}]}


r = rn_.run_agent("q", _Stubborn(), demo)
check("модель не отвечает даже без функций -> ошибка, а не зависание", r.answer is None and "шагов" in (r.error or ""), (r.answer, r.error))

class _Script:
    """Модель по сценарию: список ответов (каждый — dict сообщения)."""
    def __init__(self, replies):
        self.replies, self.n = list(replies), 0

    def chat(self, messages, tools=None):
        self.n += 1
        return self.replies.pop(0) if self.replies else {"role": "assistant", "content": ""}


def _tc(name, args, i=1):
    return {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}]}


check("is_yes_no_question: русские и английские формулировки",
      all(rn_.is_yes_no_question(q) for q in ("Был ли кто-то в красной одежде?", "Была ли куртка?", "Были ли люди в синем?",
                                              "Есть ли человек в красном?", "Was there a person in red?", "Did anyone wear red?"))
      and not any(rn_.is_yes_no_question(q) for q in ("Когда был кто-то в красном?", "Сколько людей было на 90 с?", "")))
sc = _Script([_tc("search_by_clothing", {"color": "red"})])
r = rn_.run_agent("Был ли кто-то в красной одежде?", sc, demo)
check("вопрос да/нет: после первого поиска ответ строится кодом, лишних обращений к модели нет",
      not r.error and r.answer.startswith("Да. Найдено на срезах: 60 с") and sc.n == 1 and len(r.steps) == 1, (r.error, r.answer, sc.n))
sc = _Script([_tc("count_people_at", {"t": 90}), _tc("search_by_clothing", {"color": "red"}, 2), {"role": "assistant", "content": ""}])
r = rn_.run_agent("Был ли кто-то в красной одежде?", sc, demo)
check("пустой ответ модели после нескольких вызовов (да/нет): ответ по первому успешному поиску",
      not r.error and r.answer.startswith("Да. Найдено на срезах"), (r.error, r.answer))
sc = _Script([_tc("count_people_at", {"t": 90}), _tc("search_by_clothing", {"color": "red"}, 2), {"role": "assistant", "content": ""},
              {"role": "assistant", "content": "Красное было на 60–300 с."}])
r = rn_.run_agent("Когда был кто-то в красном и сколько людей на 90 с?", sc, demo)
check("пустой ответ, не да/нет: ещё один запрос без функций, его текст берётся", not r.error and r.answer.startswith("Красное было"), (r.error, r.answer))
sc = _Script([_tc("count_people_at", {"t": 90}), _tc("search_by_clothing", {"color": "red"}, 2),
              {"role": "assistant", "content": "", "reasoning": "думаю..."}, {"role": "assistant", "content": "", "reasoning": "ещё думаю"}])
r = rn_.run_agent("Когда был кто-то в красном и сколько людей на 90 с?", sc, demo)
check("пустой ответ из-за размышлений: ошибка подсказывает поля ответа и LLM_MAX_TOKENS",
      r.answer is None and "поля ответа модели: reasoning" in r.error and "LLM_MAX_TOKENS" in r.error, r.error)
from agent import config as cfg2, llm as llm2
with contextlib.suppress(KeyError):
    os.environ.pop("LLM_MAX_TOKENS")
check("лимит токенов ответа по умолчанию 1536 (не 512), клиент его использует",
      cfg2.llm_max_tokens() == 1536 and llm2.OpenAICompatClient(base_url="http://x", model="m").max_tokens == 1536)
os.environ["LLM_MAX_TOKENS"] = "2048"
check("LLM_MAX_TOKENS переопределяет лимит; явный max_tokens главнее",
      cfg2.llm_max_tokens() == 2048 and llm2.OpenAICompatClient(base_url="http://x", model="m", max_tokens=100).max_tokens == 100)
os.environ["LLM_MAX_TOKENS"] = "мусор"
check("неверное значение LLM_MAX_TOKENS -> по умолчанию", cfg2.llm_max_tokens() == 1536)
os.environ.pop("LLM_MAX_TOKENS")

print("== схемы, call_tool, промпт, slim ==")
for name in ("search_by_clothing", "count_people_at", "get_snapshot_summary"):
    spec = next(x for x in tool_schemas.SPECS if x["function"]["name"] == name)["function"]
    props = set(spec["parameters"]["properties"])
    params = set(inspect.signature(tool_schemas.TOOL_FUNCS[name]).parameters) - {"conn"}
    check(f"{name}: аргументы схемы = аргументы функции", props == params, (props, params))
    check(f"{name}: required входит в properties", set(spec["parameters"]["required"]) <= props)
r = tool_schemas.call_tool("search_by_clothing", {"color": "red"}, demo)
check("call_tool: search_by_clothing", r["found"])
check("call_tool: count_people_at", tool_schemas.call_tool("count_people_at", {"t": 90}, demo)["people_count"] == 3)
check("call_tool: get_snapshot_summary без аргументов", tool_schemas.call_tool("get_snapshot_summary", {}, demo)["ok"])
check("call_tool: лишний аргумент отклонён", tool_schemas.call_tool("search_by_clothing", {"color": "red", "view": "all"}, demo)["ok"] is False)
check("call_tool: у count_people_at нет аргумента time", tool_schemas.call_tool("count_people_at", {"time": 5}, demo)["ok"] is False)

P = runner.SYSTEM_PROMPT
check("промпт знает три новые функции", all(x in P for x in ("search_by_clothing", "count_people_at", "get_snapshot_summary")))
check("промпт: срезы не связаны с id человека", "NOT linked" in P and "person N" in P)
check("промпт: одежда конкретного человека по id -> нельзя определить", "clothing of a specific" in P)
check("промпт: прежние правила на месте", all(x in P for x in ("count_events", "Нет, не нашёл", "ответить нельзя: в базе нет этих данных")))
check("вопрос q28 про куртку человека 7 остался вне базы",
      "Какого цвета куртка у человека 7?" in (ROOT / "eval" / "questions.csv").read_text(encoding="utf-8"))
slim = runner.slim_for_model(tool_schemas.call_tool("search_by_clothing", {"color": "red"}, demo))
check("slim: кадры из matches убраны, время и люди остались",
      "frame_path" not in slim["matches"][0] and slim["matches"][0]["t"] == 60.0 and slim["matches"][0]["people"])
slim = runner.slim_for_model(tool_schemas.call_tool("count_people_at", {"t": 90}, demo))
check("slim: frame_path верхнего уровня убран, people_count на месте", "frame_path" not in slim and slim["people_count"] == 3)
slim = runner.slim_for_model(tool_schemas.call_tool("get_snapshot_summary", {}, demo))
check("slim: summary без путей к кадрам, с максимумом", all("frame_path" not in x for x in slim["snapshots"]) and slim["max_people"] == 4)


class Scripted:
    def __init__(self, *m): self.m, self.seen = list(m), []
    def chat(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages))); return self.m.pop(0)

call = {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {
    "name": "search_by_clothing", "arguments": '{"color": "red", "clothing_type": "sweater"}'}}]}
c = Scripted(call, {"role": "assistant", "content": "Да, на 60 с и 90 с"})
res = runner.run_agent("Когда был человек в красной кофте или свитере?", c, demo)
sent = json.loads(c.seen[1][3]["content"])
check("агент: вызов, ответ; модели ушли срезы без путей к кадрам, интерфейсу остался полный результат",
      res.answer.startswith("Да") and "frame_path" not in sent["matches"][0]
      and res.steps[0].result["matches"][0]["frame_path"].endswith(".jpg"))
check("во всех вызовах модели видны девять функций", len(runner.tool_schemas.to_openai_tools()) == 9)

# ---------- T25: шаблонные ответы, язык, видео по умолчанию ----------
print("== T25: ответы кодом, защита языка, видео по умолчанию ==")
cp_ = lambda name, args, q: rn_.shortcut_answer([Step(name, args, tool_schemas.call_tool(name, args, demo))], q)
a = cp_("count_people_at", {"t": 90}, "Сколько людей было на 90 секунде?")
check("шаблон: сколько людей на срезе (ru)", a == "На срезе 90 с было 3 человека.", a)
a = cp_("count_people_at", {"t": 100}, "Сколько людей было на 100 с?")
check("шаблон: срез не на запрошенном времени, это сказано", "90 с" in a and "запрошено 100 с" in a, a)
a = cp_("get_snapshot_summary", {}, "Когда было больше всего людей?")
check("шаблон: максимум людей и когда", "Максимум: 4 человека" in a and "180 с" in a and "1.6" in a, a)
a = cp_("get_snapshot_summary", {}, "Какое среднее число людей?")
check("шаблон: среднее", a.startswith("В среднем 1.6"), a)
check("шаблон: вопрос про одежду по сводке отдаётся модели", cp_("get_snapshot_summary", {}, "Во что были одеты люди?") is None)
a = cp_("search_by_clothing", {"color": "red"}, "Сколько людей было в красном?")
check("шаблон: «сколько людей в красном» считает код", a.startswith("Найдено ") and "человек" in a, a)
a = cp_("search_by_clothing", {"color": "red"}, "Когда был кто-то в красной одежде?")
check("шаблон: «когда» -> времена срезов", a.startswith("Да. Найдено на срезах: 60 с"), a)
a = cp_("count_events", {"event_type": "enter"}, "Сколько раз кто-то входил?")
check("шаблон: count_events с фильтром", False if a is None else a == "Найдено событий: 4 (event_type=enter).", a)
check("count_events не шаблонится, если вопрос не про число или составной",
      cp_("count_events", {"event_type": "enter"}, "Опиши входы") is None
      and cp_("count_events", {"event_type": "enter"}, "Сколько было входов и выходов?") is None)
check("составной вопрос не отвечается шаблоном после первого вызова",
      cp_("count_people_at", {"t": 90}, "Когда была красная одежда и сколько людей на 90 с?") is None
      and cp_("search_by_clothing", {"color": "red"}, "Был ли кто-то в красной или синей одежде?") is None)
check("шаблон на английском",
      cp_("count_people_at", {"t": 90}, "How many people at 90 s?") == "At the snapshot at 90 s there were 3 people.")
check("ошибка функции: шаблона нет, решает модель",
      rn_.template_answer(Step("count_people_at", {}, {"ok": False, "error": "x"}), "Сколько людей?") is None)

sc = _Script([_tc("count_people_at", {"t": 90})])
r = rn_.run_agent("Сколько людей было на 90 с?", sc, demo)
check("агент: один вызов модели, ответ собран кодом", r.answer == "На срезе 90 с было 3 человека." and sc.n == 1 and not r.error, (r.answer, sc.n))

print("-- иероглифы --")
check("has_foreign_script: китайский ответ на русский вопрос", rn_.has_foreign_script("找到了红色夹克", "Был ли кто-то в красном?"))
check("has_foreign_script: нормальный ответ и китайский вопрос не считаются нарушением",
      not rn_.has_foreign_script("Да, был", "Был ли?") and not rn_.has_foreign_script("是的", "有人穿红色吗？"))
sc = _Script([_tc("search_by_clothing", {"color": "red"}), {"role": "assistant", "content": "是的，找到了"}])
r = rn_.run_agent("Опиши красную одежду, когда была или нет?", sc, demo)
check("иероглифы: повтор не помог -> ответ собран кодом, иероглифов нет",
      not r.error and not rn_.has_foreign_script(r.answer, "Когда?") and r.answer.startswith("Да. Найдено"), (r.error, r.answer))
sc = _Script([_tc("search_by_clothing", {"color": "red"}), {"role": "assistant", "content": "是的，找到了"},
              {"role": "assistant", "content": "Красное было на 60 с."}])
r = rn_.run_agent("Опиши красную одежду, когда была или нет?", sc, demo)
check("иероглифы: один повтор со строгой инструкцией, его ответ берётся", r.answer == "Красное было на 60 с." and sc.n == 3, (r.answer, sc.n))
check("системный промпт: правило языка в самом начале", rn_.SYSTEM_PROMPT.startswith("LANGUAGE RULE"))

print("-- кофта и видео по умолчанию --")
check("prompt: кофта передаётся как «кофта», не как jacket", 'clothing_type="кофта"' in rn_.SYSTEM_PROMPT)
check("type_set: кофта = sweater+hoodie, худи = hoodie, свитер = sweater, куртка = jacket",
      st_.type_set("кофта") == {"sweater", "hoodie"} and st_.type_set("Худи") == {"hoodie"}
      and st_.type_set("свитер") == {"sweater"} and st_.type_set("куртку") == {"jacket"})
check("hoodie и sweater не смешиваются автоматически", not st_.type_matches("sweater", "hoodie") and st_.type_matches({"sweater", "hoodie"}, "hoodie"))
check("без DEFAULT_VIDEO_ID несколько видео -> ошибка", st_.get_snapshot_summary(conn=two)["ok"] is False)
with mock.patch.dict(os.environ, {"DEFAULT_VIDEO_ID": "v2"}):
    r = st_.get_snapshot_summary(conn=two)
    check("DEFAULT_VIDEO_ID: берётся это видео, в ответе пометка", r["ok"] and r["video_id"] == "v2" and "default video" in r["video_id_note"], r)
    r = st_.search_by_clothing(color="red", video_id="vid001", conn=two)
    check("DEFAULT_VIDEO_ID: выдуманный video_id при нескольких видео -> видео по умолчанию", r["ok"] and "default video" in r["video_id_note"], r)
with mock.patch.dict(os.environ, {"DEFAULT_VIDEO_ID": "нет-такого"}):
    check("DEFAULT_VIDEO_ID, которого нет в архиве, игнорируется (честная ошибка)", st_.get_snapshot_summary(conn=two)["ok"] is False)

print("-- проверка аргументов поиска по вопросу --")
fc = st_.find_clothing_terms
check("find_clothing_terms: цвет и тип по-русски, падежи", fc("Был ли кто-то в красной футболке?") == (["red"], ["футболке"]))
check("find_clothing_terms: английский", fc("Was there anyone in a red t-shirt?") == (["red"], ["t-shirt"]))
check("find_clothing_terms: без одежды ничего не находит", fc("Сколько людей было на 270 секунде?") == ([], []))
check("find_clothing_terms: два цвета видны", len(fc("Был ли кто-то в красной или синей куртке?")[0]) == 2)
fixed = lambda a, q: rn_.repair_clothing_args(a, q)
a, note = fixed({"color": "red", "t_from": 0}, "Был ли кто-то в красной футболке?")
check("repair: модель потеряла тип -> тип из вопроса добавлен, остальное сохранено", a == {"color": "red", "clothing_type": "футболке"} and note, (a, note))
a, note = fixed({"color": "blue", "clothing_type": "jacket", "video_id": ""}, "Был ли кто-то в синей кофте?")
check("repair: «кофта» превращена моделью в jacket -> исправлено на кофту", a["clothing_type"] == "кофте" and note["from"]["clothing_type"] == "jacket", (a, note))
a, note = fixed({"color": "green", "clothing_type": "jacket"}, "Была ли на ком-то зелёная одежда?")
check("repair: вопрос про «одежду», модель добавила тип -> тип убран", "clothing_type" not in a and note, (a, note))
a, note = fixed({"color": "yellow"}, "Был ли кто-то в красном?")
check("repair: модель исказила цвет -> цвет из вопроса", a["color"] == "red" and note, (a, note))
a, note = fixed({"color": "red", "clothing_type": "sweater"}, "Был ли кто-то в красной кофте?")
check("repair: «кофта» шире, чем sweater (ещё и hoodie) -> заменено на «кофте»",
      a == {"color": "red", "clothing_type": "кофте"} and note, (a, note))
a, note = fixed({"color": "red", "clothing_type": "кофта"}, "Был ли кто-то в красной кофте?")
check("repair: модель уже передала «кофта» -> правки нет", note is None, (a, note))
a, note = fixed({"color": "red"}, "Был ли кто-то в красной или синей куртке?")
check("repair: несколько цветов в вопросе -> модель не трогаем", a == {"color": "red"} and note is None)
check("repair: тип, которого нет в словаре, не стираем", fixed({"color": "red", "clothing_type": "jacket"}, "Был ли кто-то в красном бомбере?")[1] is None)

sc = _Script([_tc("search_by_clothing", {"color": "red", "t_from": 0, "t_to": 1200})])
r = rn_.run_agent("Был ли кто-то в красной футболке?", sc, demo)
check("агент: модель потеряла «футболку» -> код исправил вызов, ответ «Нет», в шаге видна поправка",
      r.answer.startswith("Нет") and r.steps[0].arguments["clothing_type"] == "футболке"
      and r.steps[0].result["arguments_corrected"]["from"]["clothing_type"] is None, (r.answer, r.steps[0].arguments))

print("-- короткий промпт для вопросов про срезы --")
isq = rn_.is_snapshot_question
check("маршрут: одежда, число людей на кадре -> срезы",
      all(isq(q) for q in ("Был ли кто-то в красной одежде?", "Сколько людей было на 270 секунде?", "Когда было больше всего людей?",
                           "Was there anyone in a red sweater?", "How many people at 90 s?", "Какое среднее число людей?")))
check("маршрут: события, треки, неясное -> полный промпт",
      not any(isq(q) for q in ("Сколько раз кто-то входил?", "Перечисли все входы.", "Did anyone pick up an object?",
                               "Был ли оставленный предмет?", "Сколько людей вошли в красной одежде?", "Что делал человек 7?", "Какого цвета куртка у человека 7?", "Привет", "")))
check("короткий промпт заметно короче полного (в 4+ раза)", len(rn_.SNAPSHOT_PROMPT) * 4 < len(rn_.SYSTEM_PROMPT),
      (len(rn_.SNAPSHOT_PROMPT), len(rn_.SYSTEM_PROMPT)))
check("короткий промпт: правило языка в начале, кофта, только 3 функции",
      rn_.SNAPSHOT_PROMPT.startswith("LANGUAGE RULE") and 'clothing_type="кофта"' in rn_.SNAPSHOT_PROMPT
      and rn_.SNAPSHOT_TOOLS == ("search_by_clothing", "count_people_at", "get_snapshot_summary"))


class _Spy:
    """Запоминает, что получила модель: системный промпт и список функций."""
    def __init__(self, replies):
        self.replies, self.system, self.tools = list(replies), [], []

    def chat(self, messages, tools=None):
        self.system.append(messages[0]["content"])
        self.tools.append([t["function"]["name"] for t in tools] if tools else None)
        return self.replies.pop(0)


sp = _Spy([_tc("search_by_clothing", {"color": "red"})])
r = rn_.run_agent("Был ли кто-то в красной одежде?", sp, demo)
check("вопрос про одежду: модель получила короткий промпт и 3 функции",
      sp.system[0] is rn_.SNAPSHOT_PROMPT and sp.tools[0] == list(rn_.SNAPSHOT_TOOLS) and r.prompt_mode == "snapshots"
      and r.answer.startswith("Да"), (sp.tools, r.prompt_mode))
sp = _Spy([{"role": "assistant", "content": "Здравствуйте"}])
r = rn_.run_agent("Привет", sp, demo)
check("не вопрос про срезы: промпт этапа 1 и 6 функций событий, текстовый ответ принимается",
      sp.system[0] is rn_.EVENTS_PROMPT and sp.tools[0] == list(rn_.EVENT_ROUTE_TOOLS) and r.prompt_mode == "events"
      and r.answer == "Здравствуйте" and len(sp.system) == 1, (sp.tools, r.prompt_mode, r.answer))
check("промпт событий: правило языка в начале, тексты этапа 1 на месте, короче полного",
      rn_.EVENTS_PROMPT.startswith("LANGUAGE RULE") and "VLM INTERPRETATION RULES" in rn_.EVENTS_PROMPT
      and len(rn_.EVENTS_PROMPT) < len(rn_.SYSTEM_PROMPT) and "search_by_clothing" not in rn_.EVENTS_PROMPT)
sp = _Spy([{"role": "assistant", "content": "Не знаю"}, _tc("search_by_clothing", {"color": "red"})])
r = rn_.run_agent("Была ли красная одежда?", sp, demo)
check("короткий путь не вызвал функций -> повтор полным промптом",
      sp.system[0] is rn_.SNAPSHOT_PROMPT and sp.system[1] is rn_.SYSTEM_PROMPT and r.prompt_mode == "full"
      and r.answer.startswith("Да"), (sp.system and [x[:12] for x in sp.system], r.answer, r.error))
sp = _Spy([{"role": "assistant", "content": "ok"}])
rn_.run_agent("Был ли кто-то в красном?", sp, demo, system_prompt="custom")
check("свой system_prompt не заменяется коротким", sp.system[0] == "custom")

print("-- окно времени, которого нет в вопросе --")
a, note = fixed({"color": "red", "clothing_type": "jacket", "t_from": 0, "t_to": 0}, "Был ли кто-то в красной куртке?")
check("repair: окно [0; 0] убирается", a == {"color": "red", "clothing_type": "jacket"} and note["from"]["t_to"] == 0, (a, note))
a, note = fixed({"color": "red", "t_from": 0, "t_to": 1200}, "Когда был кто-то в красной одежде?")
check("repair: выдуманное окно [0; 1200], в вопросе времени нет -> убирается", "t_from" not in a and "t_to" not in a and note, (a, note))
a, note = fixed({"color": "red", "t_from": 60, "t_to": 120}, "Был ли кто-то в красном между 60 и 120 с?")
check("repair: время названо в вопросе -> окно модели не трогаем", a == {"color": "red", "t_from": 60, "t_to": 120} and note is None, (a, note))
a, note = fixed({"color": "red", "t_from": 0, "t_to": 0}, "Был ли кто-то в красном на 0 секунде?")
check("repair: нулевое окно убирается даже при названном времени", "t_to" not in a, a)
sc = _Script([_tc("search_by_clothing", {"color": "red", "clothing_type": "sweater", "t_from": 0, "t_to": 0})])
r = rn_.run_agent("Был ли кто-то в красной кофте?", sc, demo)
check("агент: модель передала окно [0; 0] -> ответ «Да», как без окна", r.answer.startswith("Да") and "t_to" not in r.steps[0].arguments, (r.answer, r.steps[0].arguments))

print("-- события: выдуманные аргументы, вопросы вне базы, функции срезов на вопрос про события --")
ra = lambda a, q: rn_.repair_event_args(a, q, demo)
a, note = ra({"event_type": "enter", "t_from": 0, "t_to": 3600, "video_id": "video_001"}, "Сколько раз кто-то входил?")
check("события: выдуманные окно времени и video_id убираются", a == {"event_type": "enter"} and note, (a, note))
a, note = ra({"event_type": "enter", "t_from": 0, "t_to": 100}, "Сколько раз кто-то входил в первые 100 секунд?")
check("события: время названо в вопросе -> окно остаётся", a == {"event_type": "enter", "t_from": 0, "t_to": 100} and note is None, (a, note))
a, note = ra({"track_id": 7, "video_id": "test_video_01"}, "Когда человек 7 вошёл?")
check("события: настоящий video_id остаётся", a == {"track_id": 7, "video_id": "test_video_01"} and note is None, (a, note))
a, note = ra({"track_id": 7, "video_id": ""}, "Когда человек 7 вошёл?")
check("события: пустой video_id убирается (цифра в «человек 7» это не время)", a == {"track_id": 7}, (a, note))
oos = lambda q: rn_.out_of_scope_answer(q, demo)
check("вне базы: одежда конкретного человека", oos("Какого цвета куртка у человека 7?").startswith("Ответить нельзя")
      and oos("What was person 7 wearing?").startswith("Cannot answer"))
check("вне базы: «почему» и неизвестная камера", oos("Почему человек 8 оставил предмет?") and oos("Сколько входов было на камере cam02?")
      and "cam02" in oos("Сколько входов было на камере cam02?"))
check("не вне базы: обычные вопросы и известная камера",
      all(oos(q) is None for q in ("Был ли кто-то в красной куртке?", "Сколько раз кто-то входил?", "Когда человек 7 вошёл?",
                                   "Что делал человек 7 после того, как взял предмет?", "Сколько входов было на камере cam01?")))
sc = _Script([_tc("count_events", {})])
r = rn_.run_agent("Какого цвета куртка у человека 7?", sc, demo)
check("агент: вопрос вне базы отвечает код, модель не вызывается", r.answer.startswith("Ответить нельзя") and sc.n == 0 and r.prompt_mode == "code", (r.answer, sc.n))
sc = _Script([_tc("search_by_clothing", {"t_from": 90, "t_to": 90}), {"role": "assistant", "content": "Нет"}])
r = rn_.run_agent("Был ли оставленный предмет до 90-й секунды?", sc, demo)
check("событийный вопрос + функция срезов: шаблонного «Да» по срезам нет, ответ модели не заменяется",
      not r.answer.startswith("Да. Найдено на срезах") and r.answer == "Нет", (r.answer, r.error))
check("fallback по срезам не работает для вопроса про события",
      rn_.fallback_answer([rn_.ToolStep("search_by_clothing", {}, {"ok": True, "found": True, "matches": [{"t": 60}], "count": 1})],
                          "Был ли оставленный предмет до 90-й секунды?") is None)

with mock.patch.dict(os.environ, {"LLM_TIMEOUT": "300"}):
    check("LLM_TIMEOUT меняет таймаут клиента", llm2.OpenAICompatClient(model="m").timeout == 300.0)
with mock.patch.dict(os.environ, {"LLM_TIMEOUT": "abc"}):
    check("LLM_TIMEOUT с мусором -> 120 с", llm2.OpenAICompatClient(model="m").timeout == 120.0)
check("таймаут, переданный явно, сильнее переменной", llm2.OpenAICompatClient(model="m", timeout=5).timeout == 5)

# ---------- помощники интерфейса ----------
print("== помощники интерфейса архива ==")
check("fmt_clock", [ah.fmt_clock(x) for x in (0, 9, 30, 90, 125.5, 3600, -3)] == ["0:00", "0:09", "0:30", "1:30", "2:05", "60:00", "0:00"])
check("color_hex: известные цвета", ah.color_hex("red") == ah.COLOR_HEX["red"] and ah.color_hex("grey") == ah.COLOR_HEX["gray"])
check("color_hex: 'dark blue' -> по последнему слову", ah.color_hex("dark blue") == ah.COLOR_HEX["blue"])
check("color_hex: unknown/пусто/None/мусор -> серый",
      all(ah.color_hex(v) == ah.UNKNOWN_HEX for v in ("unknown", "", None, "red;background:url(http://x)", "<b>")))
evil = {"clothing_color": "red\"><script>alert(1)</script>", "clothing_type": "<img src=x onerror=alert(1)>"}
h = ah.chip_html(evil)
check("chip_html экранирует текст от VLM, в style только цвет из белого списка",
      "<script" not in h and "<img" not in h and "&lt;" in h and 'style="background:#' in h and "url(" not in h, h)
check("person_label: unknown -> понятная подпись", ah.person_label({"clothing_color": "unknown", "clothing_type": "unknown"}) == "could not determine"
      and ah.person_label({"clothing_color": "red", "clothing_type": "unknown"}) == "red")
check("people_html без людей", "no people described" in ah.people_html([]))
sn = st_.get_snapshot_summary(conn=demo)["snapshots"]
cols, typs = ah.filter_options(sn)
check("filter_options: без 'unknown', отсортированы", "unknown" not in cols + typs and cols == sorted(cols) and "red" in cols and "jacket" in typs)
check("match_snapshots без фильтров: matched=False у всех, ничего не потеряно", len(ah.match_snapshots(sn)) == 10
      and not any(m["matched"] for m in ah.match_snapshots(sn)))
check("match_snapshots: красный", [m["t"] for m in ah.match_snapshots(sn, ["red"]) if m["matched"]] == [60.0, 90.0])
check("match_snapshots: цвет и тип у одного человека", [m["t"] for m in ah.match_snapshots(sn, ["blue"], ["sweater"]) if m["matched"]] == [])
check("match_snapshots: несколько цветов (любой)", [m["t"] for m in ah.match_snapshots(sn, ["red", "green"]) if m["matched"]] == [60.0, 90.0, 150.0, 180.0])
steps = [type("S", (), {"result": st_.search_by_clothing(color="red", conn=demo)})(),
         type("S", (), {"result": st_.count_people_at(t=60, conn=demo)})(),
         type("S", (), {"result": {"ok": False, "error": "x"}})(), type("S", (), {"result": "текст"})()]
cs = ah.collect_snapshots(steps)
check("collect_snapshots: из matches и count_people_at, без повторов (sn_0002 один раз)",
      [c["snapshot_id"] for c in cs] == ["sn_0002", "sn_0003"] and cs[0]["frame_path"], cs)
check("collect_snapshots не берёт весь архив из summary",
      ah.collect_snapshots([type("S", (), {"result": st_.get_snapshot_summary(conn=demo)})()]) == [])


print("== вопросы «да/нет» по событиям кодом, язык ответа ==")
EQ = rn_.is_event_exists_question
check("exists-вопросы распознаются",
      all(EQ(q) for q in ["Выходил ли человек 12?", "Брал ли что-нибудь человек 12?", "Подходил ли человек 9 к столу?",
                          "Был ли оставленный предмет?", "Были ли события с человеком 99?", "Did anyone pick up an object?",
                          "Was there an exit after 160 seconds?", "Did anyone enter between 20 and 55 seconds?"]))
check("не exists: счёт, время, составные, срезы",
      not any(EQ(q) for q in ["Сколько раз кто-то входил?", "Когда человек 8 вошёл?", "Когда был последний выход?",
                              "Был ли кто-то в красной кофте?", "Выходил ли человек 12 или 9?", "Did anyone enter and exit?",
                              "Did anyone enter, then exit?", "What happened after person 8 left?", "How many exits were there?",
                              "Что делал человек 7 после того, как взял предмет?", "List all events of person 9."]))
mk_ev = lambda name, count, **a: Step(name, a, {"ok": True, "count": count, "events": [{}] * count} if name == "query_events" else {"ok": True, "count": count})
check("ru: нашлось -> «Да. Найдено событий: N (фильтры).»",
      rn_.template_answer(mk_ev("query_events", 1, event_type="pick_up"), "Брал ли кто-то предмет?") == "Да. Найдено событий: 1 (event_type=pick_up).")
check("ru: не нашлось -> «Нет. Событий не найдено (фильтры).»",
      rn_.template_answer(mk_ev("query_events", 0, event_type="exit", track_id=12), "Выходил ли человек 12?") == "Нет. Событий не найдено (event_type=exit, track_id=12).")
check("en: ответ по-английски, count_events тоже",
      rn_.template_answer(mk_ev("count_events", 0, event_type="enter", t_from=20, t_to=55), "Did anyone enter between 20 and 55 seconds?")
      == "No. No events found (event_type=enter, t=20–55)." and
      rn_.template_answer(mk_ev("count_events", 2, event_type="enter"), "Did anyone enter?") == "Yes. Events found: 2 (event_type=enter).")
check("ошибка функции: шаблона нет, отвечает модель",
      rn_.template_answer(Step("query_events", {}, {"ok": False, "error": "x"}), "Выходил ли человек 12?") is None)
check("вопрос про счёт по-прежнему даёт «Найдено событий»",
      rn_.template_answer(mk_ev("count_events", 4, event_type="enter"), "Сколько раз кто-то входил?") == "Найдено событий: 4 (event_type=enter).")
check("шаблон не ломает грейдер: «Да.»/«Нет.» в начале",
      rn_._YES.match("Да. Найдено событий: 1.") and rn_._NO.match("Нет. Событий не найдено.") and rn_._YES.match("Yes. Events found: 1.") and rn_._NO.match("No. No events found."))
check("язык: русский ответ на английский вопрос плохой, английский нет, русский вопрос не трогается",
      rn_.has_wrong_language("Нет, никто не входил.", "Did anyone enter?") and not rn_.has_wrong_language("No one entered.", "Did anyone enter?")
      and not rn_.has_wrong_language("Нет.", "Выходил ли человек 12?") and not rn_.has_wrong_language("Yes, door.", "Was there a door?"))
class _LangClient:
    def __init__(self, replies): self.replies, self.n = list(replies), 0
    def chat(self, messages, tools):
        self.n += 1
        return {"content": self.replies.pop(0)}
lc = _LangClient(["No one entered between 20 and 55 s."])
check("guard_language: русский ответ на английский вопрос -> одна повторная попытка, берётся английский",
      rn_.guard_language("Нет, никто не входил.", lc, [], [], "Did anyone enter between 20 and 55 seconds?") == "No one entered between 20 and 55 s." and lc.n == 1)
lc = _LangClient(["Снова по-русски."])
check("guard_language: повтор тоже не на том языке -> сообщение, а не русский текст",
      "wrong language" in rn_.guard_language("Нет.", lc, [], [], "Did anyone enter?") and lc.n == 1)
lc = _LangClient([])
check("guard_language: верный язык -> без повторов", rn_.guard_language("No.", lc, [], [], "Did anyone enter?") == "No." and lc.n == 0)


RA = rn_.repair_event_args
check("«до 90 секунды»: t_from=90 -> t_to=90",
      RA({"event_type": "abandoned_object", "t_from": 90}, "Был ли оставленный предмет до 90-й секунды?")[0] == {"event_type": "abandoned_object", "t_to": 90})
check("«before 60 seconds»: t_from=60 -> t_to=60; «after 160 seconds»: t_to=160 -> t_from=160",
      RA({"t_from": "60"}, "Did anyone put something down before 60 seconds?")[0] == {"t_to": "60"}
      and RA({"t_to": 160}, "Was there an exit after 160 seconds?")[0] == {"t_from": 160})
check("верные границы не трогаются",
      RA({"t_to": 90}, "Был ли предмет до 90 секунды?") == ({"t_to": 90}, None)
      and RA({"t_from": 160}, "Was there an exit after 160 seconds?") == ({"t_from": 160}, None)
      and RA({"t_from": 20, "t_to": 55}, "Did anyone enter between 20 and 55 seconds?") == ({"t_from": 20, "t_to": 55}, None))
# Страница архива (на заглушке streamlit) проверяется в eval/test_design.py

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)