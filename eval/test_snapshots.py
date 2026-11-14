"""Тесты этапа 2: схема срезов, загрузчик snapshots.json, функции агента по срезам, помощники и страница архива.
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
check("'sweater' включает hoodie (группа синонимов): 60, 90, 180, 210, 240", sw == [60.0, 90.0, 180.0, 210.0, 240.0], sw)
check("'grey' = 'gray'", ts(st_.search_by_clothing(color="grey", conn=demo)) == [180.0, 240.0])
check("красная футболка: нет, и честно сказано про неопределённого человека",
      (lambda x: not x["found"] and x["uncertain_people"] == 1 and "not certain" in x["note"])(
          st_.search_by_clothing(color="red", clothing_type="t-shirt", conn=demo)))
r = st_.search_by_clothing(color="unknown", conn=demo)
check("'unknown' никогда не совпадает", not r["found"])
r = st_.search_by_clothing(color="red", video_id="", t_from="", t_to="", clothing_type="", conn=demo)
check("пустые строки вместо отсутствующих аргументов (привычка слабых моделей) работают", r["ok"] and r["found"], r)
for args, name in (({}, "нет color и type"), ({"color": 5}, "color не строка"), ({"color": "red", "t_from": 50, "t_to": 10}, "t_from > t_to"),
                   ({"color": "red", "t_from": -5}, "t_from < 0"), ({"color": "red", "t_from": True}, "t_from bool"),
                   ({"color": "red", "t_to": "abc"}, "t_to не число"), ({"color": "red", "t_from": float("nan")}, "nan")):
    r = st_.search_by_clothing(conn=demo, **args)
    check(f"ошибка ввода: {name}", r["ok"] is False and "error" in r, r)
r = st_.search_by_clothing(color="red", video_id="нет_такого", conn=demo)
check("несуществующее видео: found=false, срезов 0, пометка", r["ok"] and not r["found"] and r["snapshots_checked"] == 0 and "note" in r)

x = base_db()
sl.load_snapshots_file(write("m.json", doc([
    snap("a", 0, 2, [person(0, "red", "t-shirt"), person(1, "blue", "sweater")]),
    snap("b", 30, 2),                                     # люди есть, но не описаны
    snap("c", 60, 1, [person(0, "dark blue", "jacket")]),
    snap("d", 90, 0, [])])), conn=x)
check("цвет и тип относятся к одному человеку (красный свитер не находится)",
      not st_.search_by_clothing(color="red", clothing_type="sweater", conn=x)["found"])
check("'blue' находит 'dark blue'", ts(st_.search_by_clothing(color="blue", conn=x)) == [0.0, 60.0])
check("'dark blue' не находит просто 'blue'", ts(st_.search_by_clothing(color="dark blue", conn=x)) == [60.0])
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
check("count_people_at: видео без срезов -> found=false", r["ok"] and not r["found"] and "note" in r)

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
check("summary: видео без срезов", st_.get_snapshot_summary(video_id="нет", conn=demo)["snapshots_count"] == 0)

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
res = runner.run_agent("Был ли человек в красной кофте?", c, demo)
sent = json.loads(c.seen[1][3]["content"])
check("агент: вызов, ответ; модели ушли срезы без путей к кадрам, интерфейсу остался полный результат",
      res.answer.startswith("Да") and "frame_path" not in sent["matches"][0]
      and res.steps[0].result["matches"][0]["frame_path"].endswith(".jpg"))
check("во всех вызовах модели видны девять функций", len(runner.tool_schemas.to_openai_tools()) == 9)

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
check("person_label: unknown -> понятная подпись", ah.person_label({"clothing_color": "unknown", "clothing_type": "unknown"}) == "не удалось определить"
      and ah.person_label({"clothing_color": "red", "clothing_type": "unknown"}) == "red")
check("people_html без людей", "не описано" in ah.people_html([]))
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

# ---------- страница архива на ЗАГЛУШКЕ streamlit ----------
print("== agent/pages/1_Archive.py на ЗАГЛУШКЕ streamlit (не настоящий Streamlit) ==")
PAGE = str(ROOT / "agent" / "pages" / "1_Archive.py")


class Stop(Exception):
    pass


def run_page(db_mode=None, root=str(ROOT), colors=(), types=(), only_hits=True, env=None):
    st = mock.MagicMock()
    st.cache_resource = lambda f: f
    st.stop.side_effect = Stop
    st.selectbox.side_effect = lambda label, options, **kw: (db_mode if (db_mode and label == "База данных") else options[0])
    st.text_input.side_effect = lambda label, value="", **kw: root

    allcols = []

    def columns(spec):
        k = len(spec) if isinstance(spec, (list, tuple)) else spec
        out = [mock.MagicMock() for _ in range(k)]
        allcols.extend(out)
        for m in out:
            m.multiselect.side_effect = lambda label, opts, **kw: list(colors if label.startswith("Цвет") else types)
            m.checkbox.return_value = only_hits
        return out
    st.columns.side_effect = columns
    stopped = False
    with mock.patch.dict(sys.modules, {"streamlit": st}), mock.patch.dict(os.environ, env or {}):
        os.environ.pop("DATABASE_URL", None)
        try:
            runpy.run_path(PAGE, run_name="__main__")
        except Stop:
            stopped = True
    st.metric_calls = [c for m in allcols for c in m.metric.call_args_list]
    return st, stopped


st, stopped = run_page()
check("страница: без падений, четыре показателя", not stopped and len(st.metric_calls) == 4, st.metric_calls)
metrics = {c.args[0]: c.args[1] for c in st.metric_calls}
check("страница: максимум 4 человека, момент 3:00, среднее 1.6, срезов 10",
      metrics == {"Срезов": 10, "Максимум людей": 4, "Когда больше всего": "3:00", "В среднем людей": 1.6}, metrics)
check("страница: график построен", st.bar_chart.called)
check("страница: показаны все 10 кадров", st.image.call_count == 10, st.image.call_count)
check("страница: заголовок «Архив событий»", st.title.call_args.args[0] == "Архив событий")
st, _ = run_page(colors=["red"])
check("страница: фильтр «красный» оставляет 2 среза", st.image.call_count == 2, st.image.call_count)
st, _ = run_page(colors=["red"], only_hits=False)
check("страница: без «только совпадения» видны все 10, совпавшие помечены",
      st.image.call_count == 10 and any("совпадение" in str(c) for c in st.markdown.call_args_list))
st, _ = run_page(colors=["purple"])
check("страница: фильтр без совпадений -> пояснение и ни одного кадра",
      st.image.call_count == 0 and any("Совпадений нет" in str(c) for c in st.write.call_args_list))
with tempfile.TemporaryDirectory() as empty:
    st, _ = run_page(root=empty)
    check("страница: кадров нет в папке -> подписи вместо падения", st.image.call_count == 0
          and sum("Кадр не найден" in str(c) for c in st.caption.call_args_list) == 10)
st, stopped = run_page(db_mode="PostgreSQL (DATABASE_URL)")
check("страница: PostgreSQL без DATABASE_URL -> сообщение и остановка, без traceback",
      stopped and st.error.called and "DATABASE_URL" in st.error.call_args.args[0], st.error.call_args)
check("страница: HTML описаний не выводится через unsafe_allow_html без экранирования",
      all("<script" not in str(c) for c in run_page()[0].markdown.call_args_list))

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)
