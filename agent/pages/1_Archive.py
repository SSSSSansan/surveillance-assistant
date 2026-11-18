"""Streamlit page «Event Archive»: video snapshots every N seconds as a list of cards (frame, time, people, clothing).

Run from the project root (the page appears in the top menu):
    python -m streamlit run agent/app.py
Layout and colours: agent/ui_theme.py. Filters: video, colour, clothing type, description search, sorting.
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
ALL_COLORS, ALL_TYPES = "All colors", "All types"
NEWEST, OLDEST = "Newest first", "Oldest first"

settings = ui_theme.setup_page(st, "Event Archive", "archive")
conn = ui_theme.demo_conn() if settings.db_mode == ui_theme.DEMO else None

title_col, count_col = st.columns([4, 2])
title_col.markdown('<div class="page-title"><span class="ico">&#8634;</span>Event Archive</div>', unsafe_allow_html=True)
count_slot = count_col.empty()

try:
    videos = snapshot_tools.list_snapshot_videos(conn)
except Exception as e:  # noqa: BLE001  (нет DATABASE_URL, нет таблиц и т. п.)
    st.error(f"Could not read the snapshot archive: {e}")
    st.stop()
if not videos:
    st.info("The archive has no snapshots yet. Load snapshots.json: python -m database.snapshots_loader snapshots.json")
    st.stop()

c_video, c_color, c_type, c_search, c_sort = st.columns([1.3, 1.2, 1.2, 3, 1.4])
video_id = c_video.selectbox("Video", videos, label_visibility="collapsed")
summary = snapshot_tools.get_snapshot_summary(video_id=video_id, conn=conn)
if not summary.get("ok"):
    st.error(summary.get("error", "error"))
    st.stop()
snaps = summary["snapshots"]
color_opts, type_opts = filter_options(snaps)
color = c_color.selectbox("Clothing color", [ALL_COLORS] + color_opts, label_visibility="collapsed")
clothing_type = c_type.selectbox("Clothing type", [ALL_TYPES] + type_opts, label_visibility="collapsed")
query = c_search.text_input("Search descriptions", placeholder="Search descriptions...", label_visibility="collapsed")
order = c_sort.selectbox("Sort", [NEWEST, OLDEST], label_visibility="collapsed")

colors = [] if color == ALL_COLORS else [color]
types = [] if clothing_type == ALL_TYPES else [clothing_type]
filtering = bool(colors or types)
shown = match_snapshots(snaps, colors, types)
if filtering:
    shown = [s for s in shown if s["matched"]]
shown = sort_snapshots(search_snapshots(shown, query), newest_first=(order == NEWEST))
count_slot.markdown(f'<div class="page-count">{len(shown)} of {len(snaps)} snapshots</div>', unsafe_allow_html=True)

with st.expander("Video statistics"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Snapshots", summary["snapshots_count"])
    m2.metric("Max people", summary["max_people"])
    m3.metric("Peak time", ", ".join(fmt_clock(t) for t in summary["max_people_t"]))
    m4.metric("Avg people", summary["avg_people"])
    counts = [s["people_count"] for s in snaps]
    try:
        import pandas as pd
        st.bar_chart(pd.DataFrame({"people": counts}, index=[fmt_clock(s["t"]) for s in snaps]))
    except ImportError:
        st.bar_chart(counts)

if not shown:
    hint = ""
    if filtering and any(p.get("clothing_color") == "unknown" for s in snaps for p in s["people"]):
        hint = " Some people in the snapshots are undetermined (unknown), so a \"no\" may be inaccurate."
    st.info("Nothing found." + hint)

limit = st.session_state.get("archive_limit", PAGE_SIZE)
for s in shown[:limit]:
    path = safe_clip_path(s.get("frame_path"), settings.clips_root)
    st.markdown(card_html(s, thumb_data_uri(path), matched=s["matched"]), unsafe_allow_html=True)


def _more():
    st.session_state["archive_limit"] = st.session_state.get("archive_limit", PAGE_SIZE) + PAGE_SIZE


if len(shown) > limit:
    st.button(f"Show more ({len(shown) - limit})", on_click=_more, key="archive_more")