"""Страница Streamlit: вопрос -> ответ, время события, кнопка просмотра клипа.

Запуск из корня проекта:
    python -m streamlit run agent/app.py

Настройки: переменные окружения (configs/agent.example.env) и боковая панель.
Режим «Демо» использует SQLite в памяти с тестовыми данными, режим PostgreSQL берёт DATABASE_URL.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:   # streamlit run кладёт в sys.path папку скрипта, а не корень проекта
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from agent import config  # noqa: E402
from agent.demo_db import make_demo_db  # noqa: E402
from agent.llm import OpenAICompatClient  # noqa: E402
from agent.runner import run_agent  # noqa: E402
from agent.ui_helpers import collect_events, event_row, safe_clip_path  # noqa: E402

DEMO = "Демо (SQLite, тестовые данные)"
PG = "PostgreSQL (DATABASE_URL)"

st.set_page_config(page_title="Ассистент по видеонаблюдению", layout="wide")
st.title("Ассистент по журналу событий видеонаблюдения")

with st.sidebar:
    st.header("Настройки")
    base_url = st.text_input("LLM: адрес (OpenAI-совместимый API)", value=config.llm_base_url())
    model = st.text_input("LLM: модель", value=config.llm_model())
    db_mode = st.selectbox("База данных", [DEMO, PG])
    clips_root = st.text_input("Папка с клипами", value=str(config.clips_root()))


@st.cache_resource
def get_demo_conn():
    return make_demo_db(embedder=config.embedder_from_env())


with st.form("ask"):
    question = st.text_input("Вопрос (по-русски или по-английски)",
                             placeholder="Сколько раз кто-то входил?")
    submitted = st.form_submit_button("Спросить")

if submitted and question and question.strip():
    client = OpenAICompatClient(base_url=base_url, model=model)
    conn = get_demo_conn() if db_mode == DEMO else None
    with st.spinner("Модель думает..."):
        st.session_state["last"] = run_agent(question.strip(), client, conn=conn,
                                             embedder=config.embedder_from_env())
    st.session_state.pop("clip", None)

result = st.session_state.get("last")
if result is not None:
    if result.error:
        st.error(f"Не удалось получить ответ: {result.error}")
    if result.answer:
        st.subheader("Ответ")
        st.write(result.answer)

    events = collect_events(result.steps)
    if events:
        st.subheader("События")
        st.dataframe([event_row(e) for e in events])
        with_clips = [e for e in events if e.get("clip_path")]
        for ev in with_clips:
            label = f"Показать клип: {ev['event_id']} ({event_row(ev)['время']})"
            if st.button(label, key=f"clip_{ev.get('video_id')}_{ev['event_id']}"):
                st.session_state["clip"] = (ev.get("video_id"), ev["event_id"], ev["clip_path"])
        clip = st.session_state.get("clip")
        if clip:
            path = safe_clip_path(clip[2], clips_root)
            if path:
                st.video(str(path))
            else:
                st.warning(f"Файл клипа не найден в папке «{clips_root}»: {clip[2]}")

    with st.expander("Вызовы функций (для проверки)"):
        for step in result.steps:
            st.code(json.dumps({"функция": step.name, "аргументы": step.arguments,
                                "результат": step.result}, ensure_ascii=False, indent=2),
                    language="json")
        st.caption(f"Время ответа: {result.seconds} с")
