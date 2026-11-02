"""Тесты T11 на SQLite и заглушках модели. Настоящая LLM и настоящий Streamlit здесь НЕ запускались:
  * цикл агента, HTTP-клиент и оценка проверяются на заглушках (eval/fake_llm.py);
  * скрипт страницы agent/app.py проверяется на ЗАГЛУШКЕ модуля streamlit (ловит ошибки в нашем коде,
    но не проверяет совместимость с настоящим Streamlit).
Запуск из корня проекта: python -m eval.build && python -m eval.test_agent"""
import csv, inspect, json, os, runpy, sys, tempfile, types
from pathlib import Path
from unittest import mock

from agent import runner, tool_schemas, tools, ui_helpers
from agent.demo_db import make_demo_db
from agent.llm import LLMError, OpenAICompatClient
from eval import eval_agent, fake_llm, grader, testutil

FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


ROWS = eval_agent.load_questions()
conn = make_demo_db()

# ---------- схемы инструментов ----------
print("== схемы инструментов ==")
specs = tool_schemas.to_openai_tools()
names = [s["function"]["name"] for s in specs]
check("ровно шесть обычных функций",
      names == ["query_events", "count_events", "get_track_timeline", "get_event_details",
                "list_zones_cameras", "search_descriptions"], names)
check("debug-функции модели не показываются", not any(n.startswith("debug") for n in names))
FORBIDDEN = {"conn", "vlm_variant", "view", "embedder", "model", "min_score", "include_rejected"}
for s in specs:
    f = s["function"]
    props = f["parameters"]["properties"]
    params = set(inspect.signature(tool_schemas.TOOL_FUNCS[f["name"]]).parameters)
    check(f"{f['name']}: аргументы схемы есть в сигнатуре функции", set(props) <= params, set(props) - params)
    check(f"{f['name']}: нет служебных параметров", not (set(props) & FORBIDDEN), set(props) & FORBIDDEN)
    check(f"{f['name']}: required входит в properties", set(f["parameters"]["required"]) <= set(props))
    check(f"{f['name']}: есть описание и всё сериализуется в JSON", bool(f["description"]) and bool(json.dumps(s)))
check("enum event_type совпадает с tools.EVENT_TYPES",
      specs[0]["function"]["parameters"]["properties"]["event_type"]["enum"] == list(tools.EVENT_TYPES))
check("expected_args во всех вопросах допустимы схемой",
      all(set(json.loads(r["expected_args"])) <= tool_schemas.ALLOWED_ARGS[r["expected_tool"]]
          for r in ROWS if r["expected_tool"] != "none"))

# ---------- call_tool ----------
print("== call_tool ==")
check("неизвестная функция -> ошибка", tool_schemas.call_tool("drop_table", {}, conn)["ok"] is False)
check("debug-функция вызвать нельзя", tool_schemas.call_tool("debug_query_events", {}, conn)["ok"] is False)
for bad in ({"vlm_variant": "B"}, {"view": "all"}, {"conn": 1}, {"include_rejected": True}):
    r = tool_schemas.call_tool("query_events", bad, conn)
    check(f"служебный аргумент {list(bad)[0]} отклонён", r["ok"] is False and "допустимы" in r["error"], r)
check("arguments не словарь -> ошибка", tool_schemas.call_tool("count_events", [1], conn)["ok"] is False)
check("корректный вызов работает",
      tool_schemas.call_tool("count_events", {"event_type": "enter"}, conn)["count"] == 4)
check("arguments=None считается пустым", tool_schemas.call_tool("count_events", None, conn)["ok"] is True)
check("плохое значение -> ошибка из функции, не исключение",
      tool_schemas.call_tool("query_events", {"event_type": "fly"}, conn)["ok"] is False)


class BoomConn:
    def cursor(self): raise RuntimeError("password=secret host=db.internal")

r = tool_schemas.call_tool("count_events", {}, BoomConn())
check("внутренняя ошибка не выдаёт подробностей (пароль, хост)",
      r["ok"] is False and "RuntimeError" in r["error"] and "secret" not in r["error"] and "db.internal" not in r["error"], r)
check("search_descriptions получает embedder",
      tool_schemas.call_tool("search_descriptions", {"query": "desk", "mode": "semantic"}, conn,
                             embedder=__import__("database.embeddings", fromlist=["x"]).HashingEmbedder())["ok"] is True)

# ---------- цикл агента ----------
print("== цикл агента (скриптовая модель) ==")


class Scripted:
    def __init__(self, *msgs):
        self.msgs, self.seen = list(msgs), []

    def chat(self, messages, tools=None):
        self.seen.append(json.loads(json.dumps(messages)))
        m = self.msgs.pop(0)
        if isinstance(m, Exception):
            raise m
        return m


def call(name, args, cid="c1"):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}}]}


ans = lambda t: {"role": "assistant", "content": t}

r = runner.run_agent("Привет", Scripted(ans("Здравствуйте")), conn)
check("ответ без функций", r.answer == "Здравствуйте" and r.steps == [] and r.error is None)

c = Scripted(call("count_events", '{"event_type": "enter"}'), ans("4"))
r = runner.run_agent("Сколько входов?", c, conn)
check("функция -> ответ", r.answer == "4" and len(r.steps) == 1 and r.steps[0].result["count"] == 4, r)
hist = c.seen[1]
check("история: system, user, assistant(tool_calls), tool",
      [m["role"] for m in hist] == ["system", "user", "assistant", "tool"], [m["role"] for m in hist])
check("tool_call_id сохраняется", hist[3]["tool_call_id"] == "c1")
check("системный промпт: функции, «не нашёл», «ответить нельзя»",
      all(x in hist[0]["content"] for x in ("count_events", "не нашёл", "ответить нельзя: в базе нет этих данных")))

c = Scripted(call("query_events", {"event_type": "pick_up"}), ans("Да"))
r = runner.run_agent("q", c, conn)
tool_msg = json.loads(c.seen[1][3]["content"])
check("аргументы-словарь (не строка) принимаются", r.answer == "Да" and r.steps[0].arguments == {"event_type": "pick_up"})
check("модель видит событие без служебных полей",
      tool_msg["events"] and not ({"clip_path", "needs_vlm", "trigger_score", "camera_id", "source"} & set(tool_msg["events"][0])), tool_msg)
check("а интерфейс хранит полный результат (с clip_path)", "clip_path" in r.steps[0].result["events"][0])

m = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "a", "type": "function", "function": {"name": "count_events", "arguments": '{"event_type": "enter"}'}},
    {"id": "b", "type": "function", "function": {"name": "count_events", "arguments": '{"event_type": "exit"}'}}]}
c = Scripted(m, ans("4 и 3"))
r = runner.run_agent("q", c, conn)
check("несколько вызовов в одном сообщении: оба выполнены, tool-сообщения с их id",
      len(r.steps) == 2 and [x["tool_call_id"] for x in c.seen[1][3:]] == ["a", "b"]
      and [s.result["count"] for s in r.steps] == [4, 3])

c = Scripted(call("count_events", "{не json"), ans("не получилось"))
r = runner.run_agent("q", c, conn)
check("битый JSON аргументов: ошибка возвращена модели, цикл продолжается",
      r.answer == "не получилось" and json.loads(c.seen[1][3]["content"])["ok"] is False and r.error is None)

c = Scripted(call("rm_rf", "{}"), ans("ок"))
r = runner.run_agent("q", c, conn)
check("неизвестная функция от модели: ошибка модели, цикл продолжается",
      r.answer == "ок" and "неизвестная функция" in json.loads(c.seen[1][3]["content"])["error"])

c = Scripted(*[call("count_events", "{}", f"c{i}") for i in range(10)])
r = runner.run_agent("q", c, conn, max_steps=3)
check("лимит шагов: ошибка, не зависание", r.error and "3 шагов" in r.error and r.answer is None and len(r.steps) == 3, r.error)

r = runner.run_agent("q", Scripted(LLMError("модель недоступна")), conn)
check("ошибка модели -> result.error, без исключения", r.error == "модель недоступна" and r.answer is None)
r = runner.run_agent("q", Scripted(ans("   ")), conn)
check("пустой ответ модели -> ошибка", r.error and "пуст" in r.error)
r = runner.run_agent("q", Scripted(ans("<think>рассуждаю...</think>Да")), conn)
check("блок <think> вырезается из ответа", r.answer == "Да", r.answer)
check("strip_think: незакрытое начало", runner.strip_think("рассуждение</think>Ответ") == "Ответ")

# ---------- HTTP-клиент ----------
print("== HTTP-клиент (локальный фейковый сервер) ==")
with fake_llm.FakeOpenAIServer(fake_llm.WrongToolClient()) as srv:
    cl = OpenAICompatClient(base_url=srv.url, model="test-model", api_key="k123")
    msg = cl.chat([{"role": "user", "content": "Привет"}], tool_schemas.to_openai_tools())
    body, hdr = srv.requests[0], srv.headers[0]
    check("в запросе: модель, сообщения, tools, tool_choice, temperature 0",
          body["model"] == "test-model" and body["messages"][0]["content"] == "Привет"
          and len(body["tools"]) == 6 and body["tool_choice"] == "auto" and body["temperature"] == 0.0, body.keys())
    check("кириллица передаётся без искажений", body["messages"][0]["content"] == "Привет")
    check("заголовок Authorization", hdr.get("Authorization") == "Bearer k123")
    check("tool_calls разобраны", msg["tool_calls"][0]["function"]["name"] == "count_events")
    check("клиент без tools не шлёт tool_choice",
          "tools" not in (cl.chat([{"role": "user", "content": "x"}]) and srv.requests[1]))

    class Broken500:
        def chat(self, m, t=None): raise RuntimeError("boom")
    srv.client = Broken500()
    try:
        cl.chat([{"role": "user", "content": "x"}])
        check("HTTP 500 -> LLMError", False)
    except LLMError as e:
        check("HTTP 500 -> LLMError с кодом", "500" in str(e), e)

    srv.client = types.SimpleNamespace(chat=lambda m, t=None: "не словарь")
    try:
        cl.chat([{"role": "user", "content": "x"}])
        check("неверная форма ответа -> LLMError", False)
    except LLMError as e:
        check("неверная форма ответа -> LLMError", "формат" in str(e), e)

import socket
s = socket.socket(); s.bind(("127.0.0.1", 0)); free_port = s.getsockname()[1]; s.close()
try:
    OpenAICompatClient(base_url=f"http://127.0.0.1:{free_port}/v1", model="m", timeout=3).chat([{"role": "user", "content": "x"}])
    check("недоступный сервер -> LLMError", False)
except LLMError as e:
    check("недоступный сервер -> LLMError с понятным текстом", "не удалось подключиться" in str(e), e)
try:
    OpenAICompatClient(base_url="http://127.0.0.1:1/v1", model="").chat([])
    check("без модели -> LLMError", False)
except LLMError as e:
    check("без модели -> LLMError", "модель" in str(e))
r = runner.run_agent("q", OpenAICompatClient(base_url=f"http://127.0.0.1:{free_port}/v1", model="m", timeout=3), conn)
check("недоступная модель через run_agent -> result.error", r.error and "подключиться" in r.error)

# ---------- грейдер ----------
print("== грейдер ==")
R = lambda t, e: {"type": t, "expected_answer": e}
cases = [
    ("exists", "да", "Да, был.", True), ("exists", "да", "Yes, one pick-up.", True),
    ("exists", "да", "Нет, не нашёл", False), ("exists", "нет", "Нет, не нашёл", True),
    ("exists", "нет", "No.", True), ("exists", "нет", "Не найдено таких событий", True),
    ("exists", "нет", "Да", False), ("exists", "да", "Возможно", False),
    ("count", "4", "4", True), ("count", "4", "Было 4 входа (12.4 с, 60.1 с)", True),
    ("count", "4", "5", False), ("count", "4", "Входов нет", False),
    ("time", "12.4 с", "12.4 с", True), ("time", "60.1 с", "Человек 8 вошёл на 60.1 с", True),
    ("time", "12.4 с", "12.5 с", False), ("time", "не нашёл", "Не нашёл", True),
    ("time", "не нашёл", "12.4 с", False),
    ("list", "человек 7 (12.4 с); человек 8 (60.1 с)", "человек 7 (12.4 с); человек 8 (60.1 с)", True),
    ("list", "человек 7 (12.4 с); человек 8 (60.1 с)", "человек 7 (12.4 с)", False),
    ("list", "человек 7 (12.4 с); человек 8 (60.1 с)", "7: 12.4 с, 8: 60.1 с, 9: 150.2 с", False),
    ("list", "enter (150.2 с); exit (155.0 с)", "enter at 150.2 s, exit at 155 s", True),
    ("sequence", "leave_zone (35.0 с), затем exit (41.2 с)", "Потом ушёл со стола в 35.0 с и вышел в 41.2 с", True),
    ("sequence", "approach_zone (15.0 с)", "", False),
    ("out_of_scope", "ответить нельзя: в базе нет этих данных", "ответить нельзя: в базе нет этих данных", True),
    ("out_of_scope", "ответить нельзя", "Cannot answer: the journal has no such data", True),
    ("out_of_scope", "ответить нельзя", "Куртка синяя", False),
]
for t, e, a, want in cases:
    check(f"[{t}] ожидалось {e[:25]!r}, ответ {a[:30]!r} -> {want}", grader.grade(R(t, e), a)["correct"] is want)
check("допуск по времени: 12.5 при допуске 0.2 верно", grader.grade(R("time", "12.4 с"), "12.5 с", 0.2)["correct"])
check("допуск по времени: 12.5 при допуске 0.0 неверно", not grader.grade(R("time", "12.4 с"), "12.5 с", 0.0)["correct"])
g = grader.grade(R("list", "a (1.0 с); b (2.0 с)"), "a (1.0 с)")
check("список: precision 1.0, recall 0.5", g["precision"] == 1.0 and g["recall"] == 0.5, g)
check("args: порядок, регистр и значения по умолчанию не важны",
      grader.args_match({"event_type": "Enter", "order": "asc", "limit": 50, "t_from": 0, "zone_id": None},
                        {"event_type": "enter"}))
check("args: 7 и 7.0 равны", grader.args_match({"track_id": 7.0}, {"track_id": 7}))
check("args: лишний фильтр -> не совпало", not grader.args_match({"event_type": "enter", "zone_id": "door"}, {"event_type": "enter"}))
check("args: другое значение -> не совпало", not grader.args_match({"track_id": 8}, {"track_id": 7}))
check("args: limit 1 важен", not grader.args_match({}, {"order": "asc", "limit": 1}))

# ---------- оценка на 30 вопросах ----------
print("== оценка на 30 вопросах (проверка проводки) ==")
oracle_results = eval_agent.evaluate(fake_llm.OracleClient(ROWS), ROWS, conn)
s = eval_agent.summarize(oracle_results)
check("оракул: 30 вопросов, функция 30/30, аргументы 30/30", (s["questions"], s["tool_ok"], s["args_ok"]) == (30, 30, 30), s)
check("оракул: ответ 30/30 (ответы построены из реальных результатов функций)", s["answer_ok"] == 30,
      [(r["id"], r["answer"], r["grade_detail"]) for r in oracle_results if not r["answer_ok"]])
check("оракул: ошибок модели нет", s["model_errors"] == 0)
check("по типам: все шесть типов", sorted(s["by_type"]) == ["count", "exists", "list", "out_of_scope", "sequence", "time"])

n_count = sum(1 for r in ROWS if r["expected_tool"] == "count_events")
s2 = eval_agent.summarize(eval_agent.evaluate(fake_llm.WrongToolClient(), ROWS, conn))
check("модель с неверной функцией: верна только там, где ждали count_events",
      s2["tool_ok"] == n_count and s2["args_ok"] == 0 and s2["answer_ok"] < 10, s2)
s3 = eval_agent.summarize(eval_agent.evaluate(fake_llm.NoToolClient(), ROWS, conn))
n_none = sum(1 for r in ROWS if r["expected_tool"] == "none")
check("модель без функций: «функция верна» только для вопросов вне базы",
      s3["tool_ok"] == n_none and s3["answer_ok"] < 12, s3)
with fake_llm.FakeOpenAIServer(fake_llm.OracleClient(ROWS)) as srv:
    http_results = eval_agent.evaluate(OpenAICompatClient(base_url=srv.url, model="oracle"), ROWS, conn)
s4 = eval_agent.summarize(http_results)
check("через HTTP: оракул тоже 30/30/30", (s4["tool_ok"], s4["args_ok"], s4["answer_ok"]) == (30, 30, 30), s4)
dead = eval_agent.summarize(eval_agent.evaluate(
    OpenAICompatClient(base_url=f"http://127.0.0.1:{free_port}/v1", model="m", timeout=3), ROWS[:3], conn))
check("недоступная модель: 3 ошибки связи, оценка не падает", dead["model_errors"] == 3 and dead["answer_ok"] == 0)
with tempfile.TemporaryDirectory() as d:
    path = eval_agent.save_csv(oracle_results, d, "my model/v1")
    back = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    check("CSV с результатами: 30 строк, имя файла безопасное", len(back) == 30 and "/" not in path.name and back[0]["id"] == "q01", path.name)
txt = eval_agent.format_summary(s, 0.0)
check("текст отчёта содержит три цифры и пометку про допуск",
      "30/30" in txt and "допуск по времени" in txt and "не подтверждён" in txt)
check("CLI без модели -> код 2", eval_agent.main(["--db", "demo"]) == 2 if not os.environ.get("LLM_MODEL") else True)

# ---------- помощники интерфейса ----------
print("== помощники интерфейса ==")
steps = [types.SimpleNamespace(result={"events": [{"event_id": "e1", "video_id": "v", "clip_path": "c.mp4"}, {"event_id": "e2", "video_id": "v"}]}),
         types.SimpleNamespace(result={"event": {"event_id": "e1", "video_id": "v"}}),
         types.SimpleNamespace(result={"results": [{"event_id": "e3", "video_id": "v", "score": 0.9}]}),
         types.SimpleNamespace(result={"ok": False, "error": "x"}), types.SimpleNamespace(result="строка")]
ev = ui_helpers.collect_events(steps)
check("события собраны из events/event/results без повторов, по порядку", [e["event_id"] for e in ev] == ["e1", "e2", "e3"])
check("format_time", ui_helpers.format_time({"t_start": 12.4, "t_end": 14.0}) == "12.4–14 с"
      and ui_helpers.format_time({"t_start": 5.0, "t_end": 5.0}) == "5 с" and ui_helpers.format_time({}) == "")
check("event_row содержит время и описание", ui_helpers.event_row({"event_id": "e1", "t_start": 1.0, "t_end": 2.0, "vlm_description": "d"})["время"] == "1–2 с")

with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "root"
    (root / "clips").mkdir(parents=True)

    (root / "clips" / "a.mp4").write_bytes(b"x")

    outside = Path(d) / "secret.mp4"
    outside.write_bytes(b"x")

    # Windows may not allow symlink creation without special privileges.
    link = root / "link.mp4"
    symlink_created = False

    try:
        link.symlink_to(outside)
        symlink_created = True
    except (OSError, NotImplementedError):
        pass

    sp = lambda p: ui_helpers.safe_clip_path(p, root)

    check(
        "clip inside folder",
        sp("clips/a.mp4") == (root / "clips" / "a.mp4").resolve()
    )
    check(
        "path traversal rejected",
        sp("../secret.mp4") is None
    )
    check(
        "external absolute path rejected",
        sp(str(outside)) is None
    )

    if symlink_created:
        check(
            "external symlink rejected",
            sp("link.mp4") is None
        )
    else:
        print("SKIP: symlink test (not supported by current Windows permissions)")

    check(
        "nonexistent file -> None",
        sp("clips/nope.mp4") is None
    )
    check(
        "empty value and None -> None",
        sp("") is None and sp(None) is None
    )
    check(
        "directory instead of file -> None",
        sp("clips") is None
    )


# ---------- скрипт страницы на заглушке streamlit ----------
print("== agent/app.py на ЗАГЛУШКЕ streamlit (не настоящий Streamlit) ==")
APP = str(Path(__file__).resolve().parents[1] / "agent" / "app.py")


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


def run_app(question, base_url, clips_root, session, click=False):
    st = mock.MagicMock()
    st.session_state = session
    st.cache_resource = lambda f: f
    st.text_input.side_effect = lambda label, value="", **kw: question if label.startswith("Вопрос") else (
        clips_root if label.startswith("Папка") else value)
    st.selectbox.side_effect = lambda label, options, **kw: options[0]
    st.form_submit_button.return_value = True
    st.button.return_value = click
    env = {"LLM_BASE_URL": base_url, "LLM_MODEL": "oracle"}
    with mock.patch.dict(sys.modules, {"streamlit": st}), mock.patch.dict(os.environ, env):
        runpy.run_path(APP, run_name="__main__")
    return st


with fake_llm.FakeOpenAIServer(fake_llm.OracleClient(ROWS)) as srv, tempfile.TemporaryDirectory() as d:
    (Path(d) / "clips").mkdir()
    (Path(d) / "clips" / "test_video_01_e0013.mp4").write_bytes(b"video")
    sess = SS()
    st = run_app("Сколько раз кто-то входил?", srv.url, d, sess)
    check("страница: ответ показан", any(c.args and c.args[0] == "4" for c in st.write.call_args_list), st.write.call_args_list)
    check("страница: ошибок нет, таблицы событий для подсчёта нет", not st.error.called and not st.dataframe.called)
    check("страница: вызовы функций показаны в expander", st.expander.called and st.code.called)

    sess = SS()
    st = run_app("Когда был оставлен предмет?", srv.url, d, sess)
    table = st.dataframe.call_args.args[0]
    check("страница: таблица событий со временем", table and table[0]["время"] == "95–120 с" and table[0]["event"] == "e0013", table)
    check("страница: кнопка просмотра клипа создана", st.button.called and "e0013" in st.button.call_args.args[0])
    check("страница: клип не показан до нажатия", not st.video.called)
    st = run_app("Когда был оставлен предмет?", srv.url, d, sess, click=True)
    check("страница: после нажатия показан клип из безопасной папки",
          st.video.called and st.video.call_args.args[0].endswith("test_video_01_e0013.mp4"), st.video.call_args)
    with tempfile.TemporaryDirectory() as empty:
        sess2 = SS()
        run_app("Когда был оставлен предмет?", srv.url, empty, sess2)
        st = run_app("Когда был оставлен предмет?", srv.url, empty, sess2, click=True)
        check("страница: клипа нет -> предупреждение, а не падение", st.warning.called and not st.video.called)
st = run_app("Сколько раз кто-то входил?", "http://127.0.0.1:%d/v1" % free_port, d, SS())
check("страница: модель недоступна -> сообщение об ошибке, скрипт не падает", st.error.called and "подключиться" in st.error.call_args.args[0])
check("страница: пустой вопрос ничего не запускает", not run_app("   ", "http://127.0.0.1:1/v1", d, SS()).error.called)

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)
