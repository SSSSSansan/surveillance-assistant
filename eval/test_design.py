"""Тесты оформления и страниц Streamlit: хедер и палитры (agent/ui_theme.py), карточки архива, страница архива и чат.
Страницы запускаются на ЗАГЛУШКЕ модуля streamlit: так ловятся ошибки нашего кода, но НЕ проверяется, как вёрстка
выглядит в браузере и совместима ли она с вашей версией Streamlit. Это нужно посмотреть глазами.
Запуск из корня проекта: python -m eval.test_design"""
import base64, io, json, os, re, runpy, sqlite3, sys, tempfile
from pathlib import Path
from unittest import mock

from agent import archive_helpers as ah
from agent import snapshot_tools as st_
from agent import ui_theme
from agent.demo_db import DB_DIR, Shim, make_demo_db
from agent.llm import LLMError
from database import snapshots_loader as sl
from eval import eval_agent, fake_llm

ROOT = Path(__file__).resolve().parents[1]
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


# ---------- палитры, стили, хедер ----------
print("== оформление: палитры, стили, хедер ==")
check("в обеих палитрах одни и те же ключи", set(ui_theme.PALETTES["light"]) == set(ui_theme.PALETTES["dark"]))
with mock.patch.dict(os.environ, {}):
    os.environ.pop("UI_THEME", None)
    check("по умолчанию светлая палитра", ui_theme.palette() is ui_theme.PALETTES["light"])
with mock.patch.dict(os.environ, {"UI_THEME": "dark"}):
    check("UI_THEME=dark включает тёмную", ui_theme.palette() is ui_theme.PALETTES["dark"])
with mock.patch.dict(os.environ, {"UI_THEME": "нет_такой"}):
    check("неизвестное имя палитры -> светлая", ui_theme.palette() is ui_theme.PALETTES["light"])
for name, pal in ui_theme.PALETTES.items():
    css = ui_theme.build_css(pal)
    check(f"CSS ({name}): скобки сбалансированы", css.count("{") == css.count("}"))
    check(f"CSS ({name}): подставлены все цвета палитры", all(v in css for v in pal.values()))
    check(f"CSS ({name}): нет незаполненных {{...}}", not re.search(r"\{[a-z_]+\}", css.replace("{{", "")))
css = ui_theme.build_css()
check("CSS: скрыта боковая панель и стандартное меню страниц", "stSidebarNav" in css and "display: none" in css)
check("CSS: есть стили карточек, плашки навигации и чата", all(x in css for x in (".card", ".nav-pill", "stChatMessage", ".page-title", ".bot-head", ".orb")))
check("палитра по умолчанию: цвета из референса (фон #F5F8FC, основной #3978E8, акцент #38BDF8)",
      (lambda p: p["bg"] == "#F5F8FC" and p["primary"] == "#3978E8" and p["accent"] == "#38BDF8")(ui_theme.PALETTES["light"]))

def make_st(**over):
    st = mock.MagicMock()
    st.session_state = SS()
    mocks, buttons = [], []

    def button(label, key=None, on_click=None, args=(), **kw):
        buttons.append({"label": label, "key": key, "on_click": on_click, "args": args})
        return False

    def columns(spec, **kw):
        k = len(spec) if isinstance(spec, (list, tuple)) else spec
        out = []
        for _ in range(k):
            m = mock.MagicMock()
            m.button.side_effect = button
            m.selectbox.side_effect = lambda label, options, **kw: st._select(label, options, kw)
            m.text_input.side_effect = lambda label, **kw: st._text(label, kw)
            slot = mock.MagicMock()
            m.empty.return_value = slot
            mocks.extend([m, slot])
            out.append(m)
        return out
    st.columns.side_effect = columns
    st.button.side_effect = button
    st._select = lambda label, options, kw: (st.choices.get(label) or st.choices.get(kw.get("key"))
                                              or (settings_db if kw.get("key") == "cfg_db" else options[0]))
    st._text = lambda label, kw: st.texts.get(label, st.texts.get(kw.get("key"), kw.get("value", "")))
    st.choices, st.texts = {}, {}
    st.selectbox.side_effect = lambda label, options, **kw: st._select(label, options, kw)
    st.text_input.side_effect = lambda label, **kw: st._text(label, kw)
    st.mocks, st.buttons = mocks, buttons
    return st


settings_db = ui_theme.DEMO


def all_markdown(st):
    out = []
    for m in [st] + st.mocks:
        out += [str(c.args[0]) for c in m.markdown.call_args_list if c.args]
    return out


st = make_st()
ui_theme.render_header(st, "archive")
md = all_markdown(st)
check("хедер: название приложения", any(ui_theme.APP_NAME in t for t in md))
check("хедер: текущая страница (Архив) это плашка, а не ссылка",
      any('nav-pill">Archive' in t for t in md) and not any(c.args[0] == "pages/1_Archive.py" for m in st.mocks for c in m.page_link.call_args_list))
check("хедер: на Ассистент есть ссылка", any(c.args[0] == "app.py" for m in st.mocks for c in m.page_link.call_args_list))
st = make_st()
ui_theme.render_header(st, "assistant")
check("хедер: для страницы «Ассистент» плашка Ассистент и ссылка на Архив",
      any('nav-pill">Assistant' in t for t in all_markdown(st))
      and any(c.args[0] == "pages/1_Archive.py" for m in st.mocks for c in m.page_link.call_args_list))
check("хедер: число колонок = название + страницы", st.columns.call_args.args[0] and len(st.columns.call_args.args[0]) == 1 + len(ui_theme.NAV))
st = make_st()
st.texts = {"cfg_base_url": "http://x/v1", "cfg_model": "m1", "cfg_clips": "/clips"}
s = ui_theme.setup_page(st, "Event Archive", "archive")
cfg = st.set_page_config.call_args
check("setup_page: широкий макет, боковая панель свёрнута, заголовок вкладки",
      cfg.kwargs["layout"] == "wide" and cfg.kwargs["initial_sidebar_state"] == "collapsed" and "Event Archive" in cfg.kwargs["page_title"], cfg)
check("setup_page: стили подключены", any(t.lstrip().startswith("<style>") for t in all_markdown(st)))
check("setup_page: настройки возвращаются", (s.base_url, s.model, s.clips_root, s.db_mode) == ("http://x/v1", "m1", "/clips", ui_theme.DEMO), s)
check("настройки имеют общие ключи виджетов (значения живут между страницами)",
      {c.kwargs.get("key") for c in st.text_input.call_args_list} == {"cfg_base_url", "cfg_model", "cfg_clips"}
      and st.selectbox.call_args.kwargs["key"] == "cfg_db")
check("демо-база одна на процесс и содержит срезы", ui_theme.demo_conn() is ui_theme.demo_conn()
      and st_.get_snapshot_summary(conn=ui_theme.demo_conn())["snapshots_count"] == 10)
from agent import config as cfg_
check("адрес модели по умолчанию: Ollama http://localhost:11434/v1", cfg_.DEFAULT_BASE_URL == "http://localhost:11434/v1")
src_all = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "agent").rglob("*.py"))
check("нигде не используется use_container_width (устаревший параметр Streamlit)", "use_container_width" not in src_all)

# ---------- помощники карточек ----------
print("== карточки архива ==")
demo = make_demo_db()
sn = st_.get_snapshot_summary(conn=demo)["snapshots"]
by = {s["snapshot_id"]: s for s in sn}
check("snapshot_text: описания людей через «;»", ah.snapshot_text(by["sn_0002"]) == "red sweater, standing near the desk; black jacket, walking")
check("snapshot_text: пустой срез", ah.snapshot_text(by["sn_0000"]) == "No one in the frame.")
check("snapshot_text: люди есть, но не описаны", ah.snapshot_text({"people_count": 2, "people": []}) == "Clothing not described yet (VLM).")
check("snapshot_text: без описания берётся «цвет тип»", ah.snapshot_text({"people_count": 1, "people": [{"clothing_color": "red", "clothing_type": "coat", "description": None}]}) == "red coat")
q = lambda text: [s["snapshot_id"] for s in ah.search_snapshots(sn, text)]
check("поиск: пусто или пробелы -> все 10", len(q("")) == 10 and len(q("   ")) == 10 and len(ah.search_snapshots(sn, None)) == 10)
check("поиск: по слову из описания, без учёта регистра", q("HOODIE") == ["sn_0006", "sn_0007"], q("HOODIE"))
check("поиск: по цвету", q("green") == ["sn_0005", "sn_0006"])
check("поиск: по времени «3:00»", q("3:00") == ["sn_0006"])
check("поиск: по номеру среза", q("sn_0009") == ["sn_0009"])
check("поиск: ничего не найдено", q("zzz") == [])
check("сортировка: новые сначала / старые сначала",
      [s["t"] for s in ah.sort_snapshots(sn, True)][:2] == [270.0, 240.0] and [s["t"] for s in ah.sort_snapshots(sn, False)][:2] == [0.0, 30.0])
check("сортировка не меняет исходный список", [s["t"] for s in sn] == sorted(s["t"] for s in sn))

uri = ah.thumb_data_uri(ROOT / "demo_frames" / "test_video_01" / "sn_0002.jpg")
raw = base64.b64decode(uri.split(",", 1)[1]) if uri else b""
check("миниатюра: data-URI с настоящим JPEG", bool(uri) and uri.startswith("data:image/jpeg;base64,") and raw[:2] == b"\xff\xd8")
if uri:
    from PIL import Image
    check("миниатюра не шире 360 пикселей", Image.open(io.BytesIO(raw)).width <= 360)
check("миниатюра: нет файла / None / пустая строка -> None",
      ah.thumb_data_uri(ROOT / "нет.jpg") is None and ah.thumb_data_uri(None) is None and ah.thumb_data_uri("") is None)
with tempfile.TemporaryDirectory() as d:
    bad = Path(d) / "bad.jpg"
    bad.write_bytes("это не картинка".encode("utf-8"))
    check("миниатюра: битый файл -> None, без падения", ah.thumb_data_uri(bad) is None)

h = ah.card_html(by["sn_0002"], uri, matched=True)
check("карточка: время, номер среза, люди, текст, чипы, кадр",
      all(x in h for x in ("1:00", "sn_0002", "60 s", "people: 2", "red sweater, standing", 'class="chip"', "<img src=")), h[:200])
check("карточка: метка «совпадение» только при совпадении",
      "match" in h and "match" not in ah.card_html(by["sn_0002"], uri, matched=False))
check("карточка без кадра: заглушка с текстом", 'class="noimg"' in ah.card_html(by["sn_0002"], None) and "<img" not in ah.card_html(by["sn_0002"], None))
for bad_uri in ("javascript:alert(1)", "data:text/html;base64,PHNjcmlwdD4=", "http://evil/x.jpg",
                'data:image/jpeg;base64,"><script>alert(1)</script>', "data:image/jpeg;base64,AAAA\" onerror=\"x"):
    hh = ah.card_html(by["sn_0002"], bad_uri)
    check(f"карточка: недопустимый src отклонён ({bad_uri[:26]})", "<img" not in hh and "<script" not in hh and "onerror" not in hh)
evil = {"snapshot_id": "<b>sn</b>", "t": 5.0, "people_count": 1,
        "people": [{"person_idx": 0, "clothing_color": "red", "clothing_type": "<i>x</i>", "description": "<script>alert(1)</script> & \"q\""}]}
h = ah.card_html(evil, None, missing_text="<u>нет</u>")
check("карточка: всё от VLM и из базы экранируется (скрипты, теги, кавычки)",
      all(x not in h for x in ("<script", "<b>", "<i>", "<u>")) and "&lt;script&gt;" in h and "&amp;" in h, h)

# ---------- страница архива на заглушке ----------
print("== agent/pages/1_Archive.py на ЗАГЛУШКЕ streamlit (не настоящий Streamlit) ==")
PAGE = str(ROOT / "agent" / "pages" / "1_Archive.py")


class Stop(Exception):
    pass


def run_archive(choices=None, texts=None, root=str(ROOT), session=None, conn=None, env=None):
    st = make_st()
    st.session_state = session if session is not None else SS()
    st.stop.side_effect = Stop
    st.choices = dict(choices or {})
    st.texts = {"cfg_clips": root, **(texts or {})}
    stopped = False
    patches = [mock.patch.dict(sys.modules, {"streamlit": st}), mock.patch.dict(os.environ, env or {})]
    if conn is not None:
        patches.append(mock.patch.object(ui_theme, "demo_conn", lambda: conn))
    with contextlib_stack(patches):
        os.environ.pop("DATABASE_URL", None)
        try:
            g = runpy.run_path(PAGE, run_name="__main__")
        except Stop:
            stopped, g = True, {}
    st.cards = [t for t in all_markdown(st) if t.startswith('<div class="card')]
    st.count_text = next((t for t in all_markdown(st) if "snapshots</div>" in t), "")
    st.metrics = {c.args[0]: c.args[1] for m in st.mocks for c in m.metric.call_args_list}
    return st, stopped, g


class contextlib_stack:
    def __init__(self, ps): self.ps = ps
    def __enter__(self): [p.start() for p in self.ps]
    def __exit__(self, *a): [p.stop() for p in reversed(self.ps)]


st, stopped, _ = run_archive()
check("архив: 10 карточек, без падений и ошибок", not stopped and len(st.cards) == 10 and not st.error.called, len(st.cards))
check("архив: счётчик «10 из 10 срезов»", "10 of 10 snapshots" in st.count_text, st.count_text)
check("архив: по умолчанию сначала новые (первая 4:30, последняя 0:00)", ">4:30<" in st.cards[0] and ">0:00<" in st.cards[-1])
check("архив: заголовок «Архив событий»", any("Event Archive" in t and "page-title" in t for t in all_markdown(st)))
check("архив: в хедере плашка «Архив»", any('nav-pill">Archive' in t for t in all_markdown(st)))
check("архив: показатели, график", st.metrics == {"Snapshots": 10, "Max people": 4, "Peak time": "3:00", "Avg people": 1.6}
      and st.bar_chart.called, st.metrics)
check("архив: кадры вставлены в карточки миниатюрами", all("<img src=\"data:image/jpeg" in c for c in st.cards))
st, *_ = run_archive({"Sort": "Oldest first"})
check("архив: сортировка «сначала старые»", ">0:00<" in st.cards[0] and ">4:30<" in st.cards[-1])
st, *_ = run_archive({"Clothing color": "red"})
check("архив: фильтр по цвету «red»: 2 карточки с меткой, счётчик «2 из 10»",
      len(st.cards) == 2 and all("match" in c for c in st.cards) and "2 of 10 snapshots" in st.count_text, st.count_text)
st, *_ = run_archive({"Clothing color": "blue", "Clothing type": "sweater"})
check("архив: цвет и тип у одного человека: «blue sweater» нет, понятная подсказка про unknown",
      len(st.cards) == 0 and st.info.called and "Nothing found" in st.info.call_args.args[0] and "unknown" in st.info.call_args.args[0], st.info.call_args)
st, *_ = run_archive({"Clothing color": "purple"})
check("архив: цвет без совпадений: 0 карточек", len(st.cards) == 0 and "0 of 10 snapshots" in st.count_text)
st, *_ = run_archive(texts={"Search descriptions": "GREEN"})
check("архив: поиск по описанию без учёта регистра", len(st.cards) == 2 and "2 of 10 snapshots" in st.count_text, len(st.cards))
st, *_ = run_archive({"Clothing color": "green"}, texts={"Search descriptions": "picking"})
check("архив: фильтр и поиск вместе", len(st.cards) == 1 and ">3:00<" in st.cards[0])
sess = SS(archive_limit=3)
st, *_ = run_archive(session=sess)
more = [b for b in st.buttons if b["key"] == "archive_more"]
check("архив: страница на 3 карточки и кнопка «Show more (7)»", len(st.cards) == 3 and more and "(7)" in more[0]["label"], [b["label"] for b in st.buttons])
if more:
    more[0]["on_click"]()
    check("архив: «Show more» увеличивает лимит на размер страницы", sess["archive_limit"] == 23, sess)
st, *_ = run_archive(session=SS())
check("архив: кнопки «Show more» нет, когда всё помещается", not [b for b in st.buttons if b["key"] == "archive_more"])
with tempfile.TemporaryDirectory() as empty:
    st, stopped, _ = run_archive(root=empty)
    check("архив: кадров нет в папке -> заглушки в карточках, без падения", not stopped and len(st.cards) == 10
          and all('class="noimg"' in c for c in st.cards))
st, stopped, _ = run_archive({"cfg_db": ui_theme.PG})
check("архив: PostgreSQL без DATABASE_URL -> сообщение и остановка",
      stopped and st.error.called and "DATABASE_URL" in st.error.call_args.args[0], st.error.call_args)


def snap_db(snaps, video="v1"):
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.execute("PRAGMA foreign_keys = ON")
    raw.executescript((DB_DIR / "schema.sql").read_text(encoding="utf-8"))
    c = Shim(raw)
    sl.apply_schema(c)
    if snaps:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"video_id": video, "camera_id": "c1", "source": "meva", "snapshots": snaps}, ensure_ascii=False), encoding="utf-8")
            rep = sl.load_snapshots_file(p, conn=c)
            assert rep["ok"], rep
    return c


st, stopped, _ = run_archive(conn=snap_db([]))
check("архив: пустой архив -> подсказка про загрузчик и остановка", stopped and st.info.called and "snapshots_loader" in st.info.call_args.args[0])
hostile = snap_db([{"snapshot_id": "<b>x</b>", "t": 10, "frame_path": "../../etc/passwd", "people_count": 1,
                    "people": [{"person_idx": 0, "clothing_color": "red", "clothing_type": "sweater",
                                "description": "<script>alert(1)</script>"}]}])
st, stopped, _ = run_archive(conn=hostile)
check("архив: враждебные данные из базы не попадают в HTML и путь за пределы папки не открывается",
      not stopped and len(st.cards) == 1 and "<script" not in st.cards[0] and "<b>x" not in st.cards[0]
      and "<img" not in st.cards[0], st.cards)
undescribed = snap_db([{"snapshot_id": "a", "t": 0, "people_count": 2}])
st, *_ = run_archive(conn=undescribed)
check("архив: срез без описаний людей: подпись «ещё не описана»", len(st.cards) == 1 and "not described yet" in st.cards[0])

# ---------- страница «Ассистент» (чат) на заглушке ----------
print("== agent/app.py (чат) на ЗАГЛУШКЕ streamlit (не настоящий Streamlit) ==")
APP = str(ROOT / "agent" / "app.py")
ROWS = eval_agent.load_questions()
free = __import__("socket").socket(); free.bind(("127.0.0.1", 0)); free_port = free.getsockname()[1]; free.close()


def run_chat(session, url, root, prompt=None, env=None):
    st = make_st()
    st.session_state = session
    st.texts = {"cfg_base_url": url, "cfg_model": "oracle", "cfg_clips": root}
    st.roles = []
    st.chat_message.side_effect = lambda role: (st.roles.append(role), mock.MagicMock())[1]
    st.chat_input.return_value = prompt
    with mock.patch.dict(sys.modules, {"streamlit": st}), mock.patch.dict(os.environ, env or {}):
        os.environ.pop("DATABASE_URL", None)
        runpy.run_path(APP, run_name="__main__")
    return st


def btn(st, key):
    return next((b for b in st.buttons if b["key"] == key), None)


class AnswerClient:
    """Заглушка модели: на любой вопрос отвечает «4» (английский текст подсказок не входит в вопросы оракула)."""
    def chat(self, messages, tools=None):
        return {"role": "assistant", "content": "4"}


class ClothingClient:
    """Заглушка модели: ищет красную кофту и отвечает по результату."""
    def chat(self, messages, tools=None):
        if messages[-1]["role"] != "tool":
            return {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {
                "name": "search_by_clothing", "arguments": '{"color": "red", "clothing_type": "sweater"}'}}]}
        return {"role": "assistant", "content": "Да, на 60 с и 90 с"}


with fake_llm.FakeOpenAIServer(fake_llm.OracleClient(ROWS)) as srv, tempfile.TemporaryDirectory() as d:
    (Path(d) / "clips").mkdir()
    (Path(d) / "clips" / "test_video_01_e0013.mp4").write_bytes(b"video")
    sess = SS()
    st = run_chat(sess, srv.url, d)
    check("чат: без вопроса модель не вызывается, история пуста", not srv.requests and sess["chat"] == [])
    check("чат: приветствие от ассистента и поле ввода", st.roles == ["assistant"] and st.chat_input.called
          and "Ask about" in st.chat_input.call_args.args[0])
    check("чат: четыре подсказки-кнопки и «Clear chat»",
          [b["label"] for b in st.buttons if b["key"].startswith("chip_")] == [
              "How many times did someone enter?", "Was there a person in a red sweater?",
              "How many people were there at 90 seconds?", "When was an item left behind?"] and btn(st, "clear_chat"))
    check("чат: хедер с плашкой «Ассистент»", any('nav-pill">Assistant' in t for t in all_markdown(st)))
    check("чат: заголовок, сообщения и поле ввода в одном блоке с рамкой (контейнер chat_panel), поле ввода не прибито к низу окна",
          [c.kwargs.get("key") for c in st.container.call_args_list][:2] == ["chat_panel", "chat_messages"] and st.container.call_count >= 3)
    check("чат: рядом с «Assistant» нет большой иконки", not any("orb big" in t for t in all_markdown(st)))
    ui_text = [str(c.args[0]) for m in [st] + st.mocks for f in (m.markdown, m.caption, m.info, m.error, m.warning, m.chat_input, m.expander)
               for c in f.call_args_list if c.args] + [b["label"] for b in st.buttons]
    check("чат: весь видимый текст интерфейса на английском (нет кириллицы)", ui_text and not re.search(r"[а-яёА-ЯЁ]", " ".join(ui_text)),
          [t for t in ui_text if re.search(r"[а-яёА-ЯЁ]", t)])

    st = run_chat(sess, srv.url, d, prompt="Сколько раз кто-то входил?")
    check("чат: вопрос и ответ попали в историю", [m["role"] for m in sess["chat"]] == ["user", "assistant"]
          and sess["chat"][1]["content"] == "Найдено событий: 4 (event_type=enter).", sess["chat"])
    check("чат: порядок пузырей: приветствие, вопрос, ответ", st.roles == ["assistant", "user", "assistant"], st.roles)
    check("чат: ответ выведен обычным markdown (без unsafe_allow_html)",
          any(c.args and c.args[0] == "Найдено событий: 4 (event_type=enter)." and not c.kwargs.get("unsafe_allow_html") for c in st.markdown.call_args_list))
    check("чат: вызов функций показан в expander", st.expander.called and st.code.called)
    n_req = len(srv.requests)
    st = run_chat(sess, srv.url, d)
    check("чат: при перерисовке история показывается заново, модель не вызывается повторно",
          st.roles == ["assistant", "user", "assistant"] and len(srv.requests) == n_req
          and any(c.args and c.args[0] == "Найдено событий: 4 (event_type=enter)." for c in st.markdown.call_args_list))

    sess2 = SS()
    with fake_llm.FakeOpenAIServer(AnswerClient()) as srv3:
        st = run_chat(sess2, srv3.url, d)
        b = btn(st, "chip_How many times did someone enter?")
        b["on_click"](*b["args"])
        check("чат: нажатие подсказки откладывает вопрос", sess2["pending_q"] == "How many times did someone enter?")
        st = run_chat(sess2, srv3.url, d)
    check("чат: вопрос из подсказки обработан, очередь очищена",
          [m["role"] for m in sess2["chat"]] == ["user", "assistant"] and sess2["chat"][1]["content"] == "4"
          and sess2["chat"][0]["content"] == "How many times did someone enter?" and "pending_q" not in sess2)
    btn(st, "clear_chat")["on_click"]()
    check("чат: «Clear chat» очищает историю и открытые клипы", sess2["chat"] == [] and sess2["open_clips"] == [])
    st = run_chat(SS(), srv.url, d, prompt="   ")
    check("чат: пустой вопрос из пробелов ничего не запускает", not st.error.called and st.roles == ["assistant"])

    sess3 = SS()
    st = run_chat(sess3, srv.url, d, prompt="Когда был оставлен предмет?")
    table = st.dataframe.call_args.args[0]
    check("чат: таблица событий со временем", table and table[0]["time"] == "95–120 s" and table[0]["event"] == "e0013", table)
    check("чат: таблица без индекса", st.dataframe.call_args.kwargs.get("hide_index") is True)
    clipb = next(b for b in st.buttons if b["key"].startswith("clip_"))
    check("чат: кнопка клипа создана, клип до нажатия не показан", "e0013" in clipb["label"] and not st.video.called, clipb["key"])
    clipb["on_click"](*clipb["args"])
    st = run_chat(sess3, srv.url, d)
    check("чат: после нажатия клип показан из безопасной папки",
          st.video.called and st.video.call_args.args[0].endswith("test_video_01_e0013.mp4"), st.video.call_args)
    with tempfile.TemporaryDirectory() as empty:
        sess4 = SS()
        st = run_chat(sess4, srv.url, empty, prompt="Когда был оставлен предмет?")
        k = next(b for b in st.buttons if b["key"].startswith("clip_"))
        k["on_click"](*k["args"])
        st = run_chat(sess4, srv.url, empty)
        check("чат: клипа нет -> предупреждение, а не падение", st.warning.called and not st.video.called)

with fake_llm.FakeOpenAIServer(ClothingClient()) as srv2:
    sess5 = SS()
    st = run_chat(sess5, srv2.url, str(ROOT), prompt="Был ли человек в красной кофте?")
    check("чат: вопрос про одежду -> ответ «Да…» и два кадра из архива",
          sess5["chat"][1]["content"].startswith("Да") and st.image.call_count == 2, st.image.call_count)
    check("чат: кадры показываются с фиксированной шириной, без устаревших параметров",
          all(c.kwargs == {"width": 220} for c in st.image.call_args_list), st.image.call_args_list)
    cap = [str(c.args[0]) for m in [st] + st.mocks for c in m.caption.call_args_list if c.args]
    check("чат: подписи кадров: время и число людей", any("1:00 (60 s), people: 2" in t for t in cap) and any("1:30 (90 s), people: 3" in t for t in cap), cap)
    check("чат: чипы одежды выводятся HTML-ом из белого списка",
          any('class="chip"' in t for t in all_markdown(st)))

st = run_chat(SS(), f"http://127.0.0.1:{free_port}/v1", str(ROOT), prompt="Сколько раз кто-то входил?")
check("чат: модель недоступна -> сообщение об ошибке, скрипт не падает", st.error.called and "connect" in st.error.call_args.args[0])
sess6 = SS()
run_chat(sess6, f"http://127.0.0.1:{free_port}/v1", str(ROOT), prompt="Сколько раз кто-то входил?")
check("чат: ошибка остаётся в истории (видна при перерисовке)", sess6["chat"][1]["result"].error and sess6["chat"][1]["content"] is None)

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)