"""Страница «Ассистент»: чат с агентом по журналу событий и архиву срезов.

Запуск из корня проекта:
    python -m streamlit run agent/app.py
Страница «Архив» появляется в верхнем меню (agent/pages/1_Archive.py). Вёрстка и цвета: agent/ui_theme.py.
Настройки (адрес модели, база, папка клипов) лежат в свёрнутом блоке «Настройки подключения» под хедером.
Каждый вопрос обрабатывается отдельно: прежние сообщения чата модели не передаются.
"""
import json
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

GREETING = ("Привет! Я ассистент по журналу событий видеонаблюдения. Спрашивайте про события, людей, "
            "одежду и время: отвечаю по данным журнала и архива срезов.")
SUGGESTIONS = ["Сколько раз кто-то входил?", "Был ли человек в красной кофте?",
               "Сколько людей было на 90 секунде?", "Когда был оставлен предмет?"]

settings = ui_theme.setup_page(st, "Ассистент", "assistant")
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
    """Ответ ассистента: текст, события с кнопками клипов, кадры из архива, вызовы функций."""
    if res.error:
        st.error(f"Не удалось получить ответ: {res.error}")
    if res.answer:
        st.markdown(res.answer)

    events = collect_events(res.steps)
    if events:
        st.caption("События")
        st.dataframe([event_row(e) for e in events], hide_index=True)
        for ev in (e for e in events if e.get("clip_path")):
            key = f"clip_{idx}_{ev.get('video_id')}_{ev['event_id']}"
            st.button(f"Показать клип: {ev['event_id']} ({event_row(ev)['время']})", key=key,
                      on_click=_open_clip, args=(key,))
            if key in st.session_state["open_clips"]:
                path = safe_clip_path(ev["clip_path"], settings.clips_root)
                if path:
                    st.video(str(path))
                else:
                    st.warning(f"Файл клипа не найден в папке «{settings.clips_root}»: {ev['clip_path']}")

    snaps = collect_snapshots(res.steps)
    if snaps:
        st.caption("Кадры из архива")
        for start in range(0, min(len(snaps), 8), 4):
            row = snaps[start:start + 4]
            for col, sn in zip(st.columns(4), row):
                with col:
                    path = safe_clip_path(sn.get("frame_path"), settings.clips_root)
                    if path:
                        st.image(str(path), width=220)
                    else:
                        st.caption(f"Кадр не найден: {sn.get('frame_path') or 'путь не задан'}")
                    st.caption(f"{fmt_clock(sn['t'])} ({sn['t']:g} с), людей: {sn.get('people_count')}")
                    if sn.get("people"):
                        st.markdown(people_html(sn["people"]), unsafe_allow_html=True)

    with st.expander("Вызовы функций (для проверки)"):
        for step in res.steps:
            st.code(json.dumps({"функция": step.name, "аргументы": step.arguments, "результат": step.result},
                               ensure_ascii=False, indent=2), language="json")
        st.caption(f"Время ответа: {res.seconds} с")


title_col, clear_col = st.columns([5, 1])
title_col.markdown('<div class="page-title"><span class="ico">&#9679;</span>Ассистент</div>', unsafe_allow_html=True)
clear_col.button("Очистить чат", key="clear_chat", on_click=_clear)

for col, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
    col.button(text, key=f"chip_{text}", on_click=_ask, args=(text,))

pending = st.session_state.pop("pending_q", None)

with st.chat_message("assistant"):
    st.markdown(GREETING)
for i, msg in enumerate(history):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(i, msg["result"])

prompt = st.chat_input("Спросите о событиях...")
question = ((prompt or "").strip() or pending or "").strip()
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    client = OpenAICompatClient(base_url=settings.base_url, model=settings.model)
    conn = ui_theme.demo_conn() if settings.db_mode == ui_theme.DEMO else None
    with st.chat_message("assistant"):
        with st.spinner("Думаю..."):
            result = run_agent(question, client, conn=conn, embedder=config.embedder_from_env())
        history.append({"role": "assistant", "content": result.answer, "result": result})
        render_result(len(history) - 1, result)
