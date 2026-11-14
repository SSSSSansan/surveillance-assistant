"""Страница Streamlit «Архив событий»: срезы видео каждые N секунд с кадром, числом людей и одеждой.

Запуск (из корня проекта) тот же, страница появится в боковом меню:
    python -m streamlit run agent/app.py
Чтобы в меню было русское название, переименуйте файл, например в «1_Архив_событий.py».
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import config, snapshot_tools  # noqa: E402
from agent.archive_helpers import (filter_options, fmt_clock, match_snapshots,  # noqa: E402
                                   people_html)
from agent.demo_db import make_demo_db  # noqa: E402
from agent.ui_helpers import safe_clip_path  # noqa: E402

DEMO = "Демо (SQLite, тестовые данные)"
PG = "PostgreSQL (DATABASE_URL)"
COLS = 4

st.set_page_config(page_title="Архив событий", layout="wide")
st.markdown("""<style>
.chip{display:inline-flex;align-items:center;gap:6px;margin:2px 6px 2px 0;padding:2px 10px;
      border-radius:999px;background:rgba(128,128,128,.14);font-size:.85rem}
.dot{width:10px;height:10px;border-radius:50%;border:1px solid rgba(128,128,128,.55);display:inline-block}
.muted{opacity:.6;font-size:.85rem}
.hit{color:#2e9d57;font-weight:600;font-size:.85rem}
</style>""", unsafe_allow_html=True)

st.title("Архив событий")
st.caption("Кадр из видео каждые несколько секунд: сколько людей на нём и во что они одеты. "
           "Данные: детектор (число людей) и VLM (одежда).")

with st.sidebar:
    st.header("Настройки")
    db_mode = st.selectbox("База данных", [DEMO, PG])
    frames_root = st.text_input("Папка с кадрами", value=str(config.clips_root()))


@st.cache_resource
def get_demo_conn():
    return make_demo_db()


conn = get_demo_conn() if db_mode == DEMO else None
try:
    videos = snapshot_tools.list_snapshot_videos(conn)
except Exception as e:  # noqa: BLE001  (нет DATABASE_URL, нет таблиц и т. п.)
    st.error(f"Не удалось прочитать архив срезов: {e}")
    st.stop()
if not videos:
    st.info("В архиве пока нет срезов. Загрузите snapshots.json: python -m database.snapshots_loader snapshots.json")
    st.stop()

video_id = st.selectbox("Видео", videos)
summary = snapshot_tools.get_snapshot_summary(video_id=video_id, conn=conn)
if not summary.get("ok"):
    st.error(summary.get("error", "ошибка"))
    st.stop()
snaps = summary["snapshots"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Срезов", summary["snapshots_count"])
c2.metric("Максимум людей", summary["max_people"])
c3.metric("Когда больше всего", ", ".join(fmt_clock(t) for t in summary["max_people_t"]))
c4.metric("В среднем людей", summary["avg_people"])

counts = [s["people_count"] for s in snaps]
try:
    import pandas as pd
    st.bar_chart(pd.DataFrame({"людей": counts}, index=[fmt_clock(s["t"]) for s in snaps]))
except ImportError:
    st.bar_chart(counts)

color_opts, type_opts = filter_options(snaps)
f1, f2, f3 = st.columns([2, 2, 1])
colors = f1.multiselect("Цвет одежды", color_opts)
types = f2.multiselect("Тип одежды", type_opts)
only_hits = f3.checkbox("Только совпадения", value=True)

shown = match_snapshots(snaps, colors, types)
filtering = bool(colors or types)
if filtering:
    n_hit = sum(1 for s in shown if s["matched"])
    st.write(f"Найдено срезов: **{n_hit}** из {len(shown)}" if n_hit else
             "Совпадений нет среди описанных людей. Если на срезах есть люди без описания, "
             "ответ «нет» недостоверен.")
    if only_hits:
        shown = [s for s in shown if s["matched"]]

for start in range(0, len(shown), COLS):
    row = shown[start:start + COLS]
    for col, sn in zip(st.columns(COLS), row):
        with col:
            with st.container(border=True):
                path = safe_clip_path(sn.get("frame_path"), frames_root)
                if path:
                    st.image(str(path), use_container_width=True)
                else:
                    st.caption(f"Кадр не найден: {sn.get('frame_path') or 'путь не задан'}")
                st.markdown(f"**{fmt_clock(sn['t'])}** · {sn['t']:g} с · людей: **{sn['people_count']}**")
                if filtering and sn["matched"]:
                    st.markdown('<span class="hit">совпадение</span>', unsafe_allow_html=True)
                if sn["people"]:
                    st.markdown(people_html(sn["people"]), unsafe_allow_html=True)
                    with st.expander("Описания"):
                        for p in sn["people"]:
                            st.write(f"{p['person_idx']}: {p.get('description') or '-'}")
                elif sn["people_count"] > 0:
                    st.caption("Одежда ещё не описана (VLM)")
