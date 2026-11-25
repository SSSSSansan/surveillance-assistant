"""Streamlit page «Report» (T29, items 1-6): header, key figures, people over time, clothing distributions,
snapshot table, export to Markdown/CSV.

Run from the project root (the page appears in the top menu):
    py -m streamlit run agent/app.py
Calculations: agent/report_helpers.py. Data that does not exist yet (runs table, gold labels: T15/T21) is shown
as «no data», never invented.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import report_helpers as rh  # noqa: E402
from agent import snapshot_tools, ui_theme  # noqa: E402
from agent.archive_helpers import card_html, fmt_clock, thumb_data_uri  # noqa: E402
from agent.ui_helpers import safe_clip_path  # noqa: E402

settings = ui_theme.setup_page(st, "Report", "report")
conn = ui_theme.demo_conn() if settings.db_mode == ui_theme.DEMO else None
st.markdown(rh.REPORT_CSS, unsafe_allow_html=True)
st.markdown('<div class="page-title"><span class="ico">&#9776;</span>Report</div>', unsafe_allow_html=True)

try:
    videos = snapshot_tools.list_snapshot_videos(conn)
except Exception as e:  # noqa: BLE001  (no DATABASE_URL, no tables...)
    st.error(f"Could not read the snapshot archive: {e}")
    st.stop()
if not videos:
    st.info("The archive has no snapshots yet. Load snapshots.json: python -m database.snapshots_loader snapshots.json")
    st.stop()

f_video, f_range = st.columns([2, 3])
video_id = f_video.selectbox("Video", videos)
summary = snapshot_tools.get_snapshot_summary(video_id=video_id, conn=conn)
if not summary.get("ok"):
    st.error(summary.get("error", "error"))
    st.stop()
snaps = summary["snapshots"]
t_all = sorted(s["t"] for s in snaps)
t_range = None
if len(t_all) > 1 and t_all[0] < t_all[-1]:
    t_range = f_range.slider("Time range, s", min_value=float(t_all[0]), max_value=float(t_all[-1]),
                             value=(float(t_all[0]), float(t_all[-1])))

report = rh.build_report(video_id, snaps, rh.video_info(conn, video_id), rh.load_runs(conn, video_id), t_range)
if not report["snapshots"]:
    st.info("No snapshots in this time range.")
    st.stop()

# 1. Header
st.markdown('<div class="rtitle">Video and run</div>', unsafe_allow_html=True)
st.markdown(rh.header_html(report), unsafe_allow_html=True)

# 2. Key figures
st.markdown('<div class="rtitle">Key figures</div>', unsafe_allow_html=True)
tile_list = rh.tiles(report)
for col, (name, value, hint) in zip(st.columns(len(tile_list)), tile_list):
    col.metric(name, value, help=hint or None)

# 3. People over time
st.markdown('<div class="rtitle">People per snapshot</div>', unsafe_allow_html=True)
st.markdown(f'<div class="rpanel">{rh.chart_svg(report["series"])}</div>', unsafe_allow_html=True)
st.caption("The ring marks the peak. The dashed gold-label line appears after manual frame counts are loaded (T15/T21).")

# 4. Distributions
st.markdown('<div class="rtitle">Clothing</div>', unsafe_allow_html=True)
c_colors, c_types = st.columns(2)
c_colors.markdown(f'<div class="rpanel"><b>Colors</b>{rh.colors_bars(report)}</div>', unsafe_allow_html=True)
c_types.markdown(f'<div class="rpanel"><b>Types</b>{rh.types_bars(report)}</div>', unsafe_allow_html=True)
if report["unknown_color"]:
    st.caption(f"{report['unknown_color']} people have an undetermined color and {report['unknown_type']} an undetermined type: "
               "they are counted as 'unknown', so a 'no' answer in the chat may be inaccurate.")

# 5. Snapshot table
st.markdown('<div class="rtitle">Snapshots</div>', unsafe_allow_html=True)
table = []
for x in report["rows"]:
    path = safe_clip_path(x.get("frame_path"), settings.clips_root)
    table.append({"Frame": thumb_data_uri(path, 120), "Time": x["time"], "Snapshot": x["snapshot_id"],
                  "People (model)": x["people"], "People (gold)": "-" if x["gold_people"] is None else x["gold_people"],
                  "Top colors": x["top_colors"], "Unknown": x["unknown"], "VLM status": x["status"]})
try:
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Frame": st.column_config.ImageColumn("Frame")})
except Exception:  # noqa: BLE001  (old Streamlit without ImageColumn: table without thumbnails)
    st.dataframe([{k: v for k, v in row.items() if k != "Frame"} for row in table], hide_index=True)

labels = {f"{x['time']}  ({x['snapshot_id']})": x["snapshot_id"] for x in report["rows"]}
pick = st.selectbox("Open snapshot", list(labels))
chosen = next((s for s in snaps if s["snapshot_id"] == labels[pick]), None)
if chosen:
    st.markdown(card_html(chosen, thumb_data_uri(safe_clip_path(chosen.get("frame_path"), settings.clips_root))),
                unsafe_allow_html=True)

# 6. Events (stage 1)
st.markdown('<div class="rtitle">Events</div>', unsafe_allow_html=True)
events = rh.build_events(rh.load_events(conn, video_id), t_range)
if not events["total"]:
    st.caption("No events for this video in this time range (the events table is filled by the stage-1 pipeline).")
else:
    st.markdown(f'<div class="rpanel"><b>By type</b>{rh.bars_html(events["counts"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rpanel">{rh.events_timeline_svg(events)}</div>', unsafe_allow_html=True)
    st.dataframe([{"Time": x["time"], "Event": x["event_id"], "Type": x["type"], "Track": x["track"], "Zone": x["zone"],
                   "Score": x["score"], "Needs VLM": x["needs_vlm"], "Clip": x["clip_path"] or "-"} for x in events["rows"]],
                 hide_index=True, width="stretch")
    with_clip = [x for x in events["rows"] if safe_clip_path(x["clip_path"], settings.clips_root)]
    if with_clip:
        names = {f"{x['time']}  {x['type']}  ({x['event_id']})": x for x in with_clip}
        chosen_ev = names[st.selectbox("Open event clip", list(names))]
        st.video(str(safe_clip_path(chosen_ev["clip_path"], settings.clips_root)))
    else:
        st.caption("Clip files are not found under CLIPS_ROOT, so there is nothing to play.")

# 9. Export
st.markdown('<div class="rtitle">Export</div>', unsafe_allow_html=True)
e1, e2 = st.columns(2)
e1.download_button("Download report (Markdown)", rh.report_markdown(report, events), file_name=f"report_{video_id}.md",
                   mime="text/markdown")
e2.download_button("Download snapshot table (CSV)", rh.snapshots_csv(report), file_name=f"snapshots_{video_id}.csv",
                   mime="text/csv")

with st.expander("Not in the report yet"):
    st.markdown("- **Quality** (item 7) and **Method comparison** (item 8): need gold labels (T21) and benchmark results (T23).\n"
                "- **Run header**: model, mode and prompt appear when the `runs` table is added (T15).")