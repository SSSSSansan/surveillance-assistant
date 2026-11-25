"""Streamlit page «Labeling» (T18): gold-standard labels for clothing (crops) and people count (frames).

Run from the project root (the page appears in the top menu):
    py -m streamlit run agent/app.py
Needs the tables from database/schema_gold.sql with crops/frames loaded by database/gold_loader.py.
Rules the page enforces:
  * the labeler sees only the picture: model answers and other labelers' values are never shown;
  * one row per (crop, labeler): saving again replaces your own label, never someone else's.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import ui_theme  # noqa: E402
from agent.ui_helpers import safe_clip_path  # noqa: E402
from database import gold_loader as gl  # noqa: E402

CHOOSE = "— choose —"
NONE2 = "— none —"
# Stored values stay Russian (as in eval/gold/README.md); the page shows English captions.
VISIBLE_LABELS = {"да": "yes (front or side)", "спиной": "back turned", "частично": "partly visible",
                  "не видно": "not visible (skip in the benchmark)"}
MODES = ("Clothing (crops)", "People count (frames)")

settings = ui_theme.setup_page(st, "Labeling", "labeling")
st.markdown('<div class="page-title"><span class="ico">&#9998;</span>Labeling</div>', unsafe_allow_html=True)

if settings.db_mode == ui_theme.DEMO:
    conn = ui_theme.demo_conn()
    st.info("Demo database: no crops here. Choose PostgreSQL in «Connection settings» to label real data.")
else:
    from agent import tools
    conn = tools._connect()
try:
    gl.apply_schema(conn)         # no-op when the tables already exist
    videos = gl.gold_videos(conn)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the labeling tables: {e}")
    st.stop()
if not videos:
    st.info("Nothing to label yet. Load the files from Arai: "
            "py -m database.gold_loader crops video1_crops_index.json, then frames video1_frames_index.json")
    st.stop()

c_name, c_video, c_mode = st.columns([1.4, 2.4, 2])
labeler = c_name.text_input("Your name", value="", key="gold_labeler", placeholder="e.g. Aigerim").strip()
video_id = c_video.selectbox("Video", videos, key="gold_video")
mode = c_mode.radio("What to label", MODES, horizontal=True, key="gold_mode")
if not labeler:
    st.warning("Enter your name first: it is saved with every label, and each crop needs two independent labelers.")
    st.stop()

pr = gl.progress(conn, labeler)
if mode == MODES[0]:
    st.progress(min(1.0, pr["people_done"] / max(1, pr["people_total"])),
                text=f"{labeler}: {pr['people_done']} of {pr['people_total']} crops labeled")
else:
    st.progress(min(1.0, pr["frames_done"] / max(1, pr["frames_total"])),
                text=f"{labeler}: {pr['frames_done']} of {pr['frames_total']} frames counted")


def image_path(*candidates):
    """First existing file inside the clips folder; paths from the database are never trusted blindly."""
    root = settings.clips_root
    for c in candidates:
        if not c:
            continue
        for rel in (c, Path(c).name, f"full/{Path(c).name}", f"frames/{Path(c).name}"):
            p = safe_clip_path(str(rel), root)
            if p:
                return p
    return None


skipped = st.session_state.setdefault("gold_skipped", set())

if mode == MODES[0]:
    # ---- clothing on crops
    todo = [c for c in gl.next_crops(conn, labeler, video_id, limit=len(skipped) + 1)
            if c["crop_id"] not in skipped][:1]
    if not todo:
        st.success("All crops of this video are labeled by you." + (" Skipped ones come back after a page reload."
                                                                       if skipped else ""))
    else:
        crop = todo[0]
        left, right = st.columns([1.2, 1])
        img = image_path(crop["full_path"])
        if img:
            left.image(str(img), caption=f"{crop['crop_id']} · box height {crop['box_h']:.0f}px")
        else:
            left.error(f"No picture for {crop['crop_id']} in «{settings.clips_root}». "
                       "Set «Folder with clips and frames» in Connection settings to the folder with the crops.")
        k = crop["crop_id"]
        with right:
            st.caption("Label the clothing of the UPPER body (the whole crop is shown for context).")
            color = st.selectbox("Main color", [CHOOSE] + list(gl.COLORS), key=f"color_{k}")
            color_2 = st.selectbox("Second color (optional)", [NONE2] + [c for c in gl.COLORS if c != "unknown"],
                                   key=f"color2_{k}")
            typ = st.selectbox("Type", [CHOOSE] + list(gl.TYPES), key=f"type_{k}")
            vis_key = st.radio("Is the upper body visible?", list(VISIBLE_LABELS), key=f"vis_{k}",
                               format_func=lambda v: VISIBLE_LABELS[v], index=0)
            b_save, b_skip, b_undo = st.columns(3)
            if b_save.button("Save and next", type="primary", key=f"save_{k}"):
                if vis_key != "не видно" and (color == CHOOSE or typ == CHOOSE):
                    st.error("Choose a color and a type (or mark the crop «not visible»).")
                else:
                    ok, msg = gl.save_people_label(
                        conn, crop["video_id"], crop["snapshot_id"], crop["person_idx"], labeler,
                        "unknown" if color == CHOOSE else color, "unknown" if typ == CHOOSE else typ, vis_key,
                        None if color_2 == NONE2 else color_2, crop["box_h"])
                    if ok:
                        st.session_state["gold_last"] = (crop["video_id"], crop["snapshot_id"], crop["person_idx"])
                        st.rerun()
                    else:
                        st.error(msg)
            if b_skip.button("Skip", key=f"skip_{k}"):
                skipped.add(k)
                st.rerun()
            last = st.session_state.get("gold_last")
            if last and b_undo.button("Undo last", key=f"undo_{k}"):
                gl.delete_people_label(conn, *last, labeler)
                st.session_state["gold_last"] = None
                st.rerun()
else:
    # ---- people count on whole frames
    todo = gl.next_frames(conn, labeler, video_id, limit=1)
    if not todo:
        st.success("All frames of this video are counted by you.")
    else:
        fr = todo[0]
        img = image_path(fr["frame_path"])
        st.caption("Count a person if the head and part of the torso are visible (even if partly hidden). "
                   "People in doorways and at the frame edge count; reflections and photos on walls do not. "
                   "Count yourself, do not rely on any model.")
        if img:
            st.image(str(img), caption=f"{fr['snapshot_id']} · {fr['t']:.0f} s")
        else:
            st.error(f"No picture for {fr['snapshot_id']} in «{settings.clips_root}». "
                     "Set «Folder with clips and frames» in Connection settings to the folder with the frames.")
        k = fr["snapshot_id"]
        n = st.number_input("People in the frame", min_value=0, max_value=200, value=0, step=1, key=f"n_{k}")
        b_save, b_undo = st.columns(2)
        if b_save.button("Save and next", type="primary", key=f"fsave_{k}"):
            ok, msg = gl.save_frame_label(conn, fr["video_id"], fr["snapshot_id"], labeler, int(n))
            if ok:
                st.session_state["gold_last_frame"] = (fr["video_id"], fr["snapshot_id"])
                st.rerun()
            else:
                st.error(msg)
        lf = st.session_state.get("gold_last_frame")
        if lf and b_undo.button("Undo last", key=f"fundo_{k}"):
            gl.delete_frame_label(conn, *lf, labeler)
            st.session_state["gold_last_frame"] = None
            st.rerun()
