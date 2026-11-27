"""«Assistant» page: chat with the agent over the event log and the snapshot archive.

Run from the project root:
    python -m streamlit run agent/app.py
The «Archive» page is in the top menu (agent/pages/1_Archive.py). Layout and colours: agent/ui_theme.py.
Settings (model URL, database, clips folder) are in the collapsed «Connection settings» block under the header.
Every question is handled separately: earlier chat messages are not sent to the model.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:   # streamlit run кладёт в sys.path папку скрипта, а не корень проекта
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import config, ui_theme  # noqa: E402
from agent.archive_helpers import collect_snapshots, fmt_clock, people_html  # noqa: E402
from agent.llm import OpenAICompatClient  # noqa: E402
from agent.runner import run_agent  # noqa: E402
from agent.ui_helpers import collect_events, event_row, safe_clip_path  # noqa: E402

GREETING = ("Hi! I'm your video surveillance assistant. Ask me about events, people, clothing and time: "
            "I answer from the event log and the snapshot archive.")
SUGGESTIONS = ["How many times did someone enter?", "Was there a person in a red sweater?",
               "How many people were there at 90 seconds?", "When was an item left behind?"]
BOT_HEAD = ('<div class="bot-head"><div><div class="bot-name">Assistant</div>'
            '<div class="bot-status">Online · answers from the event log and the snapshot archive</div></div></div>')

settings = ui_theme.setup_page(st, "Assistant", "assistant")
history = st.session_state.setdefault("chat", [])
st.session_state.setdefault("open_clips", [])


def _ask(question):
    st.session_state["pending_q"] = question


def _clear():
    st.session_state["chat"] = []
    st.session_state["open_clips"] = []


def _open_clip(key):
    if key not in st.session_state["open_clips"]:
        st.session_state["open_clips"].append(key)


def render_result(idx, res):
    """Assistant reply: text, events with clip buttons, archive frames, function calls."""
    if res.error:
        st.error(f"Could not get an answer: {res.error}")
    if res.answer:
        st.markdown(res.answer)

    events = collect_events(res.steps)
    if events:
        st.caption("Events")
        st.dataframe([event_row(e) for e in events], hide_index=True)
        for ev in (e for e in events if e.get("clip_path")):
            key = f"clip_{idx}_{ev.get('video_id')}_{ev['event_id']}"
            st.button(f"Show clip: {ev['event_id']} ({event_row(ev)['time']})", key=key,
                      on_click=_open_clip, args=(key,))
            if key in st.session_state["open_clips"]:
                path = safe_clip_path(ev["clip_path"], settings.clips_root)
                if path:
                    st.video(str(path))
                else:
                    st.warning(f"Clip file not found in folder \"{settings.clips_root}\": {ev['clip_path']}")

    snaps = collect_snapshots(res.steps)
    if snaps:
        st.caption("Frames from the archive")
        for start in range(0, min(len(snaps), 8), 4):
            row = snaps[start:start + 4]
            for col, sn in zip(st.columns(4), row):
                with col:
                    path = safe_clip_path(sn.get("frame_path"), settings.clips_root)
                    if path:
                        st.image(str(path), width=220)
                    else:
                        st.caption(f"Frame not found: {sn.get('frame_path') or 'no path'}")
                    st.caption(f"{fmt_clock(sn['t'])} ({sn['t']:g} s), people: {sn.get('people_count')}")
                    if sn.get("people"):
                        st.markdown(people_html(sn["people"]), unsafe_allow_html=True)

    with st.expander("Function calls (for checking)"):
        for step in res.steps:
            st.code(json.dumps({"function": step.name, "arguments": step.arguments, "result": step.result},
                               ensure_ascii=False, indent=2), language="json")
        st.caption(f"Response time: {res.seconds} s")


# One card (CSS class chat_panel) holds everything: header, quick questions, messages and the input field.
# The input is inside a container, so Streamlit keeps it in place (not pinned to the bottom of the window).
def _container(key, **fallback):
    """Container with a CSS key (styled in ui_theme.py). Older Streamlit has no `key`: then a plain container."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container(**fallback)


panel = _container("chat_panel", border=True)
with panel:
    title_col, clear_col = st.columns([5, 1])
    title_col.markdown(BOT_HEAD, unsafe_allow_html=True)
    clear_col.button("Clear chat", key="clear_chat", on_click=_clear)
    for col, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
        col.button(text, key=f"chip_{text}", on_click=_ask, args=(text,))
    messages = _container("chat_messages")   # messages are drawn here, even when the question is handled later in the script
    input_slot = st.container()

pending = st.session_state.pop("pending_q", None)

with messages:
    with st.chat_message("assistant"):
        st.markdown(GREETING)
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(i, msg["result"])

with input_slot:
    prompt = st.chat_input("Ask about events, people or clothing...")
question = ((prompt or "").strip() or pending or "").strip()
if question and settings.db_mode != ui_theme.DEMO and not os.environ.get("DATABASE_URL"):
    # Без базы каждая функция падает, а слабая модель отвечает «нет»: честнее не отвечать вовсе.
    st.error("PostgreSQL is selected in Connection settings, but DATABASE_URL is not set, so the question was not "
             "answered. Set DATABASE_URL and restart the app, or switch to the demo database.")
    st.stop()
if question:
    history.append({"role": "user", "content": question})
    client = OpenAICompatClient(base_url=settings.base_url, model=settings.model)
    conn = ui_theme.demo_conn() if settings.db_mode == ui_theme.DEMO else None
    with messages:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_agent(question, client, conn=conn, embedder=config.embedder_from_env())
            history.append({"role": "assistant", "content": result.answer, "result": result})
            render_result(len(history) - 1, result)