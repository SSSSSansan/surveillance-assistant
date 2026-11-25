"""Тесты страницы «Отчёт» (T29, пункты 1-5 + экспорт): расчёты agent/report_helpers.py и страница agent/pages/2_Report.py.
Страница запускается на ЗАГЛУШКЕ streamlit: ловятся ошибки нашего кода, но не проверяется вид в браузере и
совместимость с вашей версией Streamlit. Только SQLite. Запуск из корня проекта: py -m eval.test_report"""
import re, runpy, sys
from pathlib import Path
from unittest import mock

from agent import report_helpers as rh
from agent import snapshot_tools as st_
from agent import ui_theme
from agent.demo_db import make_demo_db

ROOT = Path(__file__).resolve().parents[1]
FAILS = 0


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


demo = make_demo_db()
summ = st_.get_snapshot_summary(conn=demo)
vid, snaps = summ["video_id"], summ["snapshots"]
info = rh.video_info(demo, vid)
rep = rh.build_report(vid, snaps, info, rh.load_runs(demo, vid))

print("== расчёты ==")
check("шапка: камера и источник из videos", info == {"camera_id": "test_cam01", "source": "meva"}, info)
check("таблицы runs ещё нет: пустой список, без ошибки", rh.load_runs(demo, vid) == [])
check("плитки: срезов 10, детекций 16, максимум 4 на 180 с, среднее 1.6",
      (rep["snapshots"], rep["detections"], rep["max_people"], rep["max_people_t"], rep["avg_people"]) == (10, 16, 4, [180.0], 1.6), rep)
check("шаг срезов 30 с, охват до 270 с", rep["step"] == 30 and rep["last_t"] == 270.0)
check("unknown по цвету: 1 из 16 людей (6%)", rep["unknown_color"] == 1 and rep["described_people"] == 16 and rh.pct(rep["unknown_color_share"]) == "6%")
check("распределение цветов: сумма = число описанных людей, по убыванию",
      sum(k for _, k in rep["colors"]) == 16 and rep["colors"][0] == ("black", 4) and [k for _, k in rep["colors"]] == sorted((k for _, k in rep["colors"]), reverse=True), rep["colors"])
check("распределение типов содержит unknown и hoodie отдельно от sweater",
      dict(rep["types"]).get("unknown") == 1 and dict(rep["types"]).get("hoodie") == 2 and dict(rep["types"]).get("sweater") == 4)
check("ряд для графика: 10 точек (t, людей)", len(rep["series"]) == 10 and rep["series"][3] == (90.0, 3))
r2 = rh.build_report(vid, snaps, info, [], t_range=(60, 120))
check("окно времени 60-120 с: 3 среза, 2+3+2 = 7 детекций", r2["snapshots"] == 3 and r2["detections"] == 7 and r2["first_t"] == 60.0, (r2["snapshots"], r2["detections"]))
r3 = rh.build_report(vid, snaps, info, [], t_range=(1000, 2000))
check("окно без срезов: нули без ошибок", r3["snapshots"] == 0 and r3["avg_people"] == 0.0 and r3["max_people"] == 0 and r3["series"] == [])
check("шаг для одного среза None", rh.snapshot_step([{"t": 5}]) is None)

mk = lambda t, n, people: {"snapshot_id": f"s{t}", "t": t, "people_count": n, "frame_path": None, "people": people}
pp = lambda c, ty="shirt": {"clothing_color": c, "clothing_type": ty}
rows = rh.snapshot_rows([mk(0, 0, []), mk(30, 3, []), mk(60, 3, [pp("red"), pp("unknown")]), mk(90, 2, [pp("red"), pp("red")])])
check("статусы срезов: empty / not described / partial / described",
      [x["status"] for x in rows] == ["empty", "not described", "partial (2/3)", "described"], [x["status"] for x in rows])
check("топ цветов без unknown, unknown считается отдельно", rows[2]["top_colors"] == "red 1" and rows[2]["unknown"] == 1 and rows[0]["top_colors"] == "-")
check("эталон по кадру пока None (появится после T15/T21)", all(x["gold_people"] is None for x in rows))

print("== runs: прогон виден в шапке, когда таблица появится ==")
raw = make_demo_db()
cur = raw.cursor()
cur.execute("CREATE TABLE runs (run_id TEXT, video_id TEXT, method TEXT, model TEXT, mode TEXT, prompt_variant TEXT, "
            "weights TEXT, started_at TEXT, duration_sec REAL)")
cur.execute("INSERT INTO runs VALUES ('r1', %s, 'vlm', 'qwen3-vl-4b', 'crop', 'A', 'yolo11s.pt', '2026-11-20 10:00', 83.5)", (vid,))
cur.close()
runs = rh.load_runs(raw, vid)
rr = rh.build_report(vid, snaps, info, runs)
fields, note = rh.header_fields(rr)
check("runs: прогон найден, в шапке модель, режим, промпт, веса, время", len(runs) == 1 and dict(fields)["Model"] == "qwen3-vl-4b"
      and dict(fields)["Mode"] == "crop" and dict(fields)["Prompt"] == "A" and dict(fields)["Weights"] == "yolo11s.pt" and note is None, fields)
check("runs: время обработки в плитках", ("Processing time", "83.5 s", "from runs") in rh.tiles(rr), rh.tiles(rr))
check("без runs: честная пометка вместо выдуманной модели", rh.header_fields(rep)[1] == rh.NO_RUN and ("Processing time", "-", "no run data") in rh.tiles(rep))

print("== HTML-фрагменты ==")
svg = rh.chart_svg(rep["series"])
check("график: svg, линия, пик 4 · 3:00 отмечен", svg.startswith("<svg") and "<polyline" in svg and "4 · 3:00" in svg and 'class="peak"' in svg, svg[:200])
check("график: одна точка и пустой ряд не падают", "<svg" in rh.chart_svg([(0, 2)]) and "no snapshots" in rh.chart_svg([]))
check("график: эталон рисуется пунктиром, если передан", 'class="gold"' in rh.chart_svg(rep["series"], gold=[(0, 1), (30, 2)]) and 'class="gold"' not in svg)
check("график: два пика отмечены оба", rh.chart_svg([(0, 3), (30, 1), (60, 3)]).count('class="peak"') == 2)
check("график: в svg только числа и время (нет данных из базы)", not re.search(r"[<>&]\w*script", svg))
bars = rh.bars_html([("<script>x</script>", 3), ("red", 1)])
check("полосы: подписи из базы экранируются", "<script>" not in bars and "&lt;script&gt;" in bars)
check("полосы: цвета только из белого списка", "background:#d64545" in rh.colors_bars(rep) and rh.bars_html([]) .count("no data") == 1)
rr_bad = dict(rep, camera_id="<b>cam</b>", source="a&b")
hh = rh.header_html(rr_bad)
check("шапка: значения экранируются", "<b>cam</b>" not in hh and "&lt;b&gt;cam&lt;/b&gt;" in hh and "a&amp;b" in hh)

print("== экспорт ==")
md = rh.report_markdown(rep)
check("markdown: все разделы", all(x in md for x in ("# Snapshot report: test_video_01", "## Video and run", "## Key figures", "## People per snapshot",
                                                      "## Clothing colors", "## Clothing types", "## Snapshots")), md[:200])
check("markdown: 10 строк таблицы по срезам и числа плиток", md.count("| sn_00") == 10 and "**Max people:** 4 (at 3:00)" in md)
csvtxt = rh.snapshots_csv(rep)
lines = csvtxt.lstrip("﻿").strip().splitlines()
check("csv: BOM для Excel, заголовок и 10 строк", csvtxt.startswith("﻿") and lines[0].startswith("snapshot_id,t,time,people") and len(lines) == 11, lines[:2])
check("csv: значения, похожие на формулы, экранируются", rh._csv_safe("=1+1") == "'=1+1" and rh._csv_safe("@x") == "'@x" and rh._csv_safe("red 1") == "red 1")

print("== страница agent/pages/2_Report.py на ЗАГЛУШКЕ streamlit ==")
PAGE = str(ROOT / "agent" / "pages" / "2_Report.py")


class StopPage(Exception):
    """st.stop() в настоящем Streamlit прерывает скрипт страницы."""


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


def make_st():
    st = mock.MagicMock()
    st.session_state = SS()
    pick = lambda label, options, **kw: options[0]
    text = lambda label, **kw: kw.get("value", "")

    def cols(spec, **kw):
        k = len(spec) if isinstance(spec, (list, tuple)) else spec
        out = []
        for _ in range(k):
            m = mock.MagicMock()
            m.selectbox.side_effect = pick
            m.text_input.side_effect = text
            m.slider.side_effect = lambda label, **kw: kw["value"]
            out.append(m)
        st.cols_made.extend(out)
        return out
    st.cols_made = []
    st.stop.side_effect = StopPage
    st.columns.side_effect = cols
    st.selectbox.side_effect = pick
    st.text_input.side_effect = text
    st.slider.side_effect = lambda label, **kw: kw["value"]
    return st


def run_page(st):
    with mock.patch.dict(sys.modules, {"streamlit": st}):
        try:
            runpy.run_path(PAGE, run_name="__main__")
        except StopPage:
            pass


def everything(st, name):
    out = []
    for m in [st] + st.cols_made:
        out += [c for c in getattr(m, name).call_args_list]
    return out



print("== события (пункт 6) ==")
evs = rh.load_events(demo, vid)
check("события: 19 из демо-базы", len(evs) == 19, len(evs))
ev = rh.build_events(evs)
check("события: типы по убыванию, сумма = 19", ev["total"] == 19 and sum(n for _, n in ev["counts"]) == 19 and ev["counts"][0] == ("enter", 4), ev["counts"])
ev2 = rh.build_events(evs, (0, 30))
check("события: окно 0-30 с оставляет только начавшиеся там", 0 < ev2["total"] < 19 and all(x["t_start"] <= 30 for x in ev2["rows"]), ev2["total"])
check("события: другое видео и окно без событий -> пусто без ошибок",
      rh.load_events(demo, "нет такого") == [] and rh.build_events(evs, (5000, 6000))["total"] == 0)
svg = rh.events_timeline_svg(ev)
check("таймлайн: svg, по строке на тип, 19 точек", svg.startswith("<svg") and svg.count('class="evdot"') == 19 and svg.count("<text") >= 8, svg.count('class="evdot"'))
check("таймлайн пустой: без ошибки", "no events" in rh.events_timeline_svg(rh.build_events([])))
bad = rh.build_events([{"event_id": "e<1>", "event_type": "<b>x</b>", "track_id": None, "zone_id": None, "t_start": 1, "t_end": 1, "trigger_score": None, "needs_vlm": False, "clip_path": None}])
check("таймлайн: тип из базы экранируется", "<b>" not in rh.events_timeline_svg(bad) and "&lt;b&gt;" in rh.events_timeline_svg(bad))
md = rh.report_markdown(rep, ev)
check("markdown: раздел Events с типами и таблицей", "## Events" in md and "| enter | 4 |" in md and "| time | event |" in md)
check("markdown без событий: прежний формат", "## Events" not in rh.report_markdown(rep))
check("markdown: пустые события -> понятная строка", "No events" in rh.report_markdown(rep, rh.build_events([])))
class _Broken:
    def cursor(self): raise RuntimeError("нет таблицы")
    def rollback(self): pass
check("нет таблицы events: пустой список, без ошибки", rh.load_events(_Broken(), vid) == [])

st = make_st()
run_page(st)
md_all = " ".join(str(c.args[0]) for c in everything(st, "markdown") if c.args)
check("страница: открывается на демо-базе без ошибок", not st.error.called and not st.info.called, (st.error.call_args_list, st.info.call_args_list))
check("страница: плашка «Report» в шапке", 'nav-pill">Report' in md_all)
check("страница: шапка, график и полосы выведены", 'class="rhead"' in md_all and "<svg" in md_all and "rbar" in md_all)
check("страница: 6 плиток показателей", sum(1 for c in everything(st, "metric")) == 6, len(everything(st, "metric")))
tbl = st.dataframe.call_args_list[0]
check("страница: таблица по срезам, 10 строк с нужными колонками",
      len(tbl.args[0]) == 10 and {"Time", "People (model)", "People (gold)", "Top colors", "Unknown", "VLM status", "Frame"} <= set(tbl.args[0][0]), list(tbl.args[0][0]))
dl = everything(st, "download_button")
check("страница: две кнопки скачивания: Markdown и CSV", len(dl) == 2 and dl[0].args[1].startswith("# Snapshot report") and "## Events" in dl[0].args[1] and dl[1].args[1].startswith("﻿"), [c.args[0] for c in dl])
check("страница: выбор среза показывает карточку", "card" in md_all)
check("страница: раздел событий: бары, таймлайн, таблица из 19 строк",
      "evdot" in md_all and any(len(c.args[0]) == 19 for c in st.dataframe.call_args_list[1:2]), [len(c.args[0]) for c in st.dataframe.call_args_list])
check("страница: ничего не выдумано: пометка про runs", "no run data" in md_all)

with mock.patch.object(st_, "list_snapshot_videos", return_value=[]):
    st = make_st(); run_page(st)
    check("страница: пустой архив -> сообщение, а не ошибка", st.info.called and st.stop.called)
with mock.patch.object(st_, "list_snapshot_videos", side_effect=RuntimeError("нет базы")):
    st = make_st(); run_page(st)
    check("страница: база недоступна -> понятная ошибка", st.error.called and "нет базы" in str(st.error.call_args))
check("меню: Report между Archive и Assistant", [k for k, _, _ in ui_theme.NAV] == ["archive", "report", "assistant"])

print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)