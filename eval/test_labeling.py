"""Тесты T18: страница agent/pages/3_Labeling.py на ЗАГЛУШКЕ streamlit и SQLite.
Ловятся ошибки нашего кода и правила страницы (чужие значения не видны, пустой выбор не сохраняется).
Настоящий Streamlit и PostgreSQL здесь НЕ запускались.
Запуск из корня проекта: py -m eval.test_labeling"""
import json, runpy, sqlite3, sys, tempfile
from pathlib import Path
from unittest import mock

from agent import tools, ui_theme
from agent.demo_db import Shim
from database import gold_loader as gl

ROOT = Path(__file__).resolve().parents[1]
PAGE = str(ROOT / "agent" / "pages" / "3_Labeling.py")
FAILS = 0
V = "vid1"


def check(name, cond, extra=""):
    global FAILS
    if not cond:
        FAILS += 1
    print(("OK   " if cond else "FAIL ") + name + ((" | " + str(extra)) if not cond and extra != "" else ""))


class StopPage(Exception):
    pass


class Rerun(Exception):
    pass


class SS(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


TMP = Path(tempfile.mkdtemp())
(TMP / "full").mkdir()
(TMP / "frames").mkdir()
for nm in ("sn_0000_p00", "sn_0000_p01"):
    (TMP / "full" / f"{nm}.jpg").write_bytes(b"x")
(TMP / "frames" / "sn_0000.jpg").write_bytes(b"x")


def new_db():
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    c = Shim(raw)
    gl.apply_schema(c)
    crops = [{"crop_id": f"sn_0000_p{i:02d}", "video_id": V, "snapshot_id": "sn_0000", "person_idx": i, "t": 0.0,
              "box_h": 120.0, "kept": True, "reason": None, "frame_path": "frames/sn_0000.jpg",
              "path": "t.jpg", "full_path": f"data/v/crops/full/sn_0000_p{i:02d}.jpg"} for i in range(2)]
    p = TMP / "c.json"; p.write_text(json.dumps(crops)); gl.load_crops_index(str(p), c)
    fr = [{"video_id": V, "snapshot_id": "sn_0000", "t": 0.0, "frame_path": "frames/sn_0000.jpg", "people_count": 99}]
    p = TMP / "f.json"; p.write_text(json.dumps(fr)); gl.load_frames_index(str(p), c)
    return c


def make_st(widgets=None, pressed=(), name="Aigerim", mode=None):
    """widgets: {префикс ключа: значение}; pressed: ключи нажатых кнопок (по префиксу)."""
    widgets = widgets or {}
    st = mock.MagicMock()
    st.session_state = SS()

    def find(key):
        for pre, v in widgets.items():
            if key and key.startswith(pre):
                return v
        return None

    def selectbox(label, options, key=None, **kw):
        v = find(key)
        return v if v is not None else options[0]

    def radio(label, options, key=None, **kw):
        if key == "gold_mode":
            return mode or options[0]
        v = find(key)
        return v if v is not None else options[0]

    def text_input(label, key=None, **kw):
        return name if key == "gold_labeler" else kw.get("value", "")

    def number_input(label, key=None, **kw):
        v = find(key)
        return v if v is not None else kw.get("value", 0)

    def button(label, key=None, **kw):
        return any(key and key.startswith(p) for p in pressed)

    def cols(spec, **kw):
        k = len(spec) if isinstance(spec, (list, tuple)) else spec
        out = []
        for _ in range(k):
            m = mock.MagicMock()
            m.selectbox.side_effect = selectbox; m.text_input.side_effect = text_input
            m.radio.side_effect = radio; m.button.side_effect = button
            out.append(m)
        st.cols_made.extend(out)
        return out
    st.cols_made = []
    st.stop.side_effect = StopPage
    st.rerun.side_effect = Rerun
    st.columns.side_effect = cols
    st.selectbox.side_effect = selectbox; st.radio.side_effect = radio; st.text_input.side_effect = text_input
    st.number_input.side_effect = number_input; st.button.side_effect = button
    return st


def run(st, conn, root=TMP, mode_pg=True):
    s = ui_theme.Settings(base_url="x", model="m", db_mode=ui_theme.PG if mode_pg else ui_theme.DEMO, clips_root=str(root))
    err = None
    with mock.patch.dict(sys.modules, {"streamlit": st}), \
            mock.patch.object(ui_theme, "setup_page", return_value=s), \
            mock.patch.object(tools, "_connect", return_value=conn), \
            mock.patch.object(ui_theme, "demo_conn", return_value=conn):
        try:
            runpy.run_path(PAGE, run_name="__main__")
        except (StopPage, Rerun) as e:
            err = type(e).__name__
    return err


def n(conn, sql):
    cur = conn.cursor(); cur.execute(sql); v = cur.fetchall()[0][0]; cur.close(); return v


def calls(st, name):
    out = []
    for m in [st] + st.cols_made:
        out += list(getattr(m, name).call_args_list)
    return out


print("== страница разметки на заглушке ==")
c = new_db()
st = make_st()
run(st, c)
check("открывается: картинка кропа показана, без ошибок", len(calls(st, "image")) == 1 and not st.error.called
      and "sn_0000_p00" in str(calls(st, "image")[0]), (calls(st, "image"), st.error.call_args_list))
sel = [x.args[0] for x in calls(st, "selectbox")]
check("выбор цвета и типа содержит словарь и «— choose —»", any(len(o) == 15 and o[0].startswith("—") for o in [x.args[1] for x in calls(st, "selectbox")]))

st = make_st(name="")
check("без имени: предупреждение и остановка", run(st, c) == "StopPage" and st.warning.called and not calls(st, "image"))

st = make_st(pressed=("save_",))
err = run(st, c)
check("«Save» без выбора цвета и типа: ошибка, в базу ничего", st.error.called and n(c, "SELECT COUNT(*) FROM labels_gold_people") == 0, st.error.call_args_list)

st = make_st(widgets={"color_": "red", "type_": "sweater", "color2_": "white", "vis_": "спиной"}, pressed=("save_",))
err = run(st, c)
cur = c.cursor(); cur.execute("SELECT color, color_2, type, visible, labeler, box_h, person_idx FROM labels_gold_people"); row = cur.fetchall()
check("«Save»: запись с разметчиком и box_h, затем перезапуск", err == "Rerun" and row == [("red", "white", "sweater", "спиной", "Aigerim", 120.0, 0)], (err, row))

st = make_st()
run(st, c)
check("после сохранения показан следующий кроп", "sn_0000_p01" in str(calls(st, "image")[0]))

st = make_st(pressed=("skip_",))
check("«Skip» перезапускает и ничего не пишет", run(st, c) == "Rerun" and n(c, "SELECT COUNT(*) FROM labels_gold_people") == 1)

st = make_st(pressed=("undo_",))
st.session_state["gold_last"] = (V, "sn_0000", 0)
check("«Undo last» удаляет свою последнюю разметку", run(st, c) == "Rerun" and n(c, "SELECT COUNT(*) FROM labels_gold_people") == 0)

st = make_st(widgets={"vis_": "не видно"}, pressed=("save_",))
run(st, c)
check("«не видно»: цвет и тип можно не выбирать (unknown)", n(c, "SELECT COUNT(*) FROM labels_gold_people WHERE color='unknown' AND type='unknown' AND visible='не видно'") == 1)

# чужие значения не видны
c2 = new_db()
gl.save_people_label(c2, V, "sn_0000", 0, "Arai", "green", "coat", "да")
st = make_st(name="Aigerim")
run(st, c2)
txt = " ".join(str(x) for m in [st] + st.cols_made for x in m.mock_calls)
check("чужая разметка не показывается: имя другого разметчика и его значения нигде не выводятся",
      "Arai" not in txt and "по умолч" not in txt and all("coat" not in str(x) for x in calls(st, "caption") + calls(st, "image")))
check("кроп с одной чужой оценкой идёт первым", "sn_0000_p00" in str(calls(st, "image")[0]))

# нет картинки
st = make_st()
run(st, new_db(), root=TMP / "empty_dir_that_does_not_exist")
check("нет файла картинки: понятная ошибка про папку", any("Folder with clips" in str(x) for x in calls(st, "error")))

# режим кадров
c3 = new_db()
st = make_st(mode="People count (frames)")
run(st, c3)
check("кадры: картинка кадра, число модели нигде не показано", len(calls(st, "image")) == 1 and "99" not in str(calls(st, "image")))
st = make_st(mode="People count (frames)", widgets={"n_": 12}, pressed=("fsave_",))
check("кадры: сохранение числа", run(st, c3) == "Rerun" and n(c3, "SELECT true_people_count FROM labels_gold_frames") == 12)
st = make_st(mode="People count (frames)")
run(st, c3)
check("кадры: после сохранения все посчитаны", st.success.called)

# пустая база и демо
raw = sqlite3.connect(":memory:", check_same_thread=False); empty = Shim(raw); gl.apply_schema(empty)
st = make_st()
check("пустые таблицы: подсказка, а не ошибка", run(st, empty) == "StopPage" and st.info.called and not st.error.called)
st = make_st()
run(st, empty, mode_pg=False)
check("демо-режим: пояснение про PostgreSQL", any("Demo" in str(x) for x in calls(st, "info")))

check("меню содержит Labeling", ("labeling", "Labeling", "pages/3_Labeling.py") in ui_theme.NAV)
print(f"\nИТОГО провалов: {FAILS}")
sys.exit(1 if FAILS else 0)
