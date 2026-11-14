"""Страница Streamlit «Архив событий»: срезы видео каждые N секунд списком карточек (кадр, время, люди, одежда).

Запуск из корня проекта (страница появляется в верхнем меню):
    python -m streamlit run agent/app.py
Вёрстка и цвета: agent/ui_theme.py. Фильтры: видео, цвет, тип одежды, поиск по описанию, сортировка.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import snapshot_tools, ui_theme  # noqa: E402
from agent.archive_helpers import (card_html, filter_options, fmt_clock, match_snapshots,  # noqa: E402
                                   search_snapshots, sort_snapshots, thumb_data_uri)
from agent.ui_helpers import safe_clip_path  # noqa: E402

PAGE_SIZE = 20
ALL_COLORS, ALL_TYPES = "Все цвета", "Все типы"
NEWEST, OLDEST = "Сначала новые", "Сначала старые"

settings = ui_theme.setup_page(st, "Архив событий", "archive")
conn = ui_theme.demo_conn() if settings.db_mode == ui_theme.DEMO else None

title_col, count_col = st.columns([4, 2])
title_col.markdown('<div class="page-title"><span class="ico">&#8634;</span>Архив событий</div>', unsafe_allow_html=True)
count_slot = count_col.empty()

try:
    videos = snapshot_tools.list_snapshot_videos(conn)
except Exception as e:  # noqa: BLE001  (нет DATABASE_URL, нет таблиц и т. п.)
    st.error(f"Не удалось прочитать архив срезов: {e}")
    st.stop()
if not videos:
    st.info("В архиве пока нет срезов. Загрузите snapshots.json: python -m database.snapshots_loader snapshots.json")
    st.stop()

c_video, c_color, c_type, c_search, c_sort = st.columns([1.3, 1.2, 1.2, 3, 1.4])
video_id = c_video.selectbox("Видео", videos, label_visibility="collapsed")
summary = snapshot_tools.get_snapshot_summary(video_id=video_id, conn=conn)
if not summary.get("ok"):
    st.error(summary.get("error", "ошибка"))
    st.stop()
snaps = summary["snapshots"]
color_opts, type_opts = filter_options(snaps)
color = c_color.selectbox("Цвет одежды", [ALL_COLORS] + color_opts, label_visibility="collapsed")
clothing_type = c_type.selectbox("Тип одежды", [ALL_TYPES] + type_opts, label_visibility="collapsed")
query = c_search.text_input("Поиск по описанию", placeholder="Поиск по описанию...", label_visibility="collapsed")
order = c_sort.selectbox("Сортировка", [NEWEST, OLDEST], label_visibility="collapsed")

colors = [] if color == ALL_COLORS else [color]
types = [] if clothing_type == ALL_TYPES else [clothing_type]
filtering = bool(colors or types)
shown = match_snapshots(snaps, colors, types)
if filtering:
    shown = [s for s in shown if s["matched"]]
shown = sort_snapshots(search_snapshots(shown, query), newest_first=(order == NEWEST))
count_slot.markdown(f'<div class="page-count">{len(shown)} из {len(snaps)} срезов</div>', unsafe_allow_html=True)

with st.expander("Статистика по видео"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Срезов", summary["snapshots_count"])
    m2.metric("Максимум людей", summary["max_people"])
    m3.metric("Когда больше всего", ", ".join(fmt_clock(t) for t in summary["max_people_t"]))
    m4.metric("В среднем людей", summary["avg_people"])
    counts = [s["people_count"] for s in snaps]
    try:
        import pandas as pd
        st.bar_chart(pd.DataFrame({"людей": counts}, index=[fmt_clock(s["t"]) for s in snaps]))
    except ImportError:
        st.bar_chart(counts)

if not shown:
    hint = ""
    if filtering and any(p.get("clothing_color") == "unknown" for s in snaps for p in s["people"]):
        hint = " Часть людей на срезах не определена (unknown), поэтому «нет» может быть неточным."
    st.info("Ничего не найдено." + hint)

limit = st.session_state.get("archive_limit", PAGE_SIZE)
for s in shown[:limit]:
    path = safe_clip_path(s.get("frame_path"), settings.clips_root)
    st.markdown(card_html(s, thumb_data_uri(path), matched=s["matched"]), unsafe_allow_html=True)


def _more():
    st.session_state["archive_limit"] = st.session_state.get("archive_limit", PAGE_SIZE) + PAGE_SIZE


if len(shown) > limit:
    st.button(f"Показать ещё ({len(shown) - limit})", on_click=_more, key="archive_more")
