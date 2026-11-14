"""Оформление страниц Streamlit: хедер с навигацией, стили карточек и чата, блок настроек.

Цвета собраны в палитры (PALETTES). Сейчас по умолчанию тёмная нейтральная палитра; фиолетовая палитра
из референса лежит рядом и включается переменной окружения UI_THEME=purple. Чтобы поменять цвета,
правится только словарь палитры.

Модуль не импортирует streamlit при загрузке (Streamlit передаётся в функции), поэтому его можно тестировать
без Streamlit. Вёрстка проверялась только на заглушке Streamlit: как она выглядит в браузере, нужно посмотреть
запуском `python -m streamlit run agent/app.py`.
"""
import os
from dataclasses import dataclass

from agent import config

APP_NAME = "Surveillance Assistant"        # название в левом верхнем углу
DEMO = "Демо (SQLite, тестовые данные)"
PG = "PostgreSQL (DATABASE_URL)"

# (ключ, подпись, путь страницы относительно agent/app.py)
NAV = [("archive", "Архив", "pages/1_Archive.py"),
       ("assistant", "Ассистент", "app.py")]

PALETTES = {
    "dark": {
        "bg": "#0d0e12", "bg2": "#15161d", "text": "#ececf0", "muted": "rgba(236,236,240,.62)",
        "card": "rgba(255,255,255,.045)", "border": "rgba(255,255,255,.10)",
        "accent": "#ff7a45", "accent_soft": "rgba(255,122,69,.16)",
        "pill_active": "rgba(255,255,255,.10)", "chip": "rgba(255,255,255,.09)", "ok": "#3fbf7f",
    },
    "purple": {   # по скриншотам-референсам; включается UI_THEME=purple
        "bg": "#8c27c4", "bg2": "#7a2fe3", "text": "#ffffff", "muted": "rgba(255,255,255,.72)",
        "card": "rgba(255,255,255,.11)", "border": "rgba(255,255,255,.20)",
        "accent": "#ff7a2e", "accent_soft": "rgba(255,122,46,.22)",
        "pill_active": "rgba(255,255,255,.18)", "chip": "rgba(255,255,255,.16)", "ok": "#7dffb3",
    },
}


def palette(name=None):
    name = (name or os.environ.get("UI_THEME") or "dark").strip().lower()
    return PALETTES.get(name, PALETTES["dark"])


def build_css(p=None):
    p = p or palette()
    return f"""
<style>
:root {{
  --bg:{p['bg']}; --bg2:{p['bg2']}; --text:{p['text']}; --muted:{p['muted']}; --card:{p['card']};
  --border:{p['border']}; --accent:{p['accent']}; --accent-soft:{p['accent_soft']};
  --pill-active:{p['pill_active']}; --chip:{p['chip']}; --ok:{p['ok']};
}}
.stApp {{ background: linear-gradient(135deg, var(--bg) 0%, var(--bg2) 100%); color: var(--text); }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1400px; padding-top: 1.4rem; padding-bottom: 6rem; }}

/* ---- хедер ---- */
.app-logo {{ font-size: 1.55rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); padding-top: .2rem; }}
.nav-pill {{ display: inline-block; padding: .42rem 1.05rem; border-radius: 999px; font-weight: 600;
            border: 1px solid var(--border); background: var(--pill-active); color: var(--text); }}
[data-testid="stPageLink-NavLink"] {{ border-radius: 999px; padding: .42rem 1.05rem;
            border: 1px solid transparent; color: var(--muted); }}
[data-testid="stPageLink-NavLink"]:hover {{ background: var(--chip); color: var(--text); }}
.header-line {{ height: 1px; background: var(--border); margin: .6rem 0 1.2rem; }}

/* ---- заголовок страницы ---- */
.page-title {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -.02em; margin: .2rem 0 .8rem; color: var(--text); }}
.page-title .ico {{ color: var(--accent); margin-right: .5rem; }}
.page-count {{ text-align: right; color: var(--muted); padding-top: 1.1rem; }}

/* ---- поля и фильтры ---- */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{
  border-radius: 12px !important; background: var(--card) !important; border-color: var(--border) !important; }}
[data-testid="stTextInput"] input {{ color: var(--text); }}
div.stButton > button {{ border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--text); }}
div.stButton > button:hover {{ border-color: var(--accent); color: var(--text); }}

/* ---- карточки архива ---- */
.card {{ display: flex; gap: 18px; background: var(--card); border: 1px solid var(--border);
        border-left: 3px solid var(--accent); border-radius: 16px; padding: 14px 18px; margin: 0 0 12px; }}
.card.hit {{ box-shadow: 0 0 0 1px var(--accent) inset; }}
.card-img {{ flex: 0 0 200px; }}
.card-img img {{ width: 200px; border-radius: 10px; display: block; }}
.card-img .noimg {{ width: 200px; height: 112px; border-radius: 10px; background: var(--chip); color: var(--muted);
                   display: flex; align-items: center; justify-content: center; font-size: .8rem; text-align: center; padding: 6px; }}
.card-body {{ flex: 1; min-width: 0; }}
.card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 6px; }}
.ts {{ color: var(--accent); font-weight: 700; font-size: 1.15rem; }}
.meta {{ color: var(--muted); font-size: .92rem; }}
.badge {{ background: var(--chip); border: 1px solid var(--border); border-radius: 10px; padding: 1px 10px; font-size: .85rem; }}
.badge.hit {{ background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }}
.card-text {{ line-height: 1.5; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; margin: 6px 6px 0 0; padding: 2px 10px;
        border-radius: 999px; background: var(--chip); font-size: .85rem; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(128,128,128,.55); display: inline-block; }}
.muted {{ color: var(--muted); font-size: .9rem; }}
@media (max-width: 760px) {{ .card {{ flex-direction: column; }} .card-img, .card-img img, .card-img .noimg {{ width: 100%; flex-basis: auto; }} }}

/* ---- чат ---- */
[data-testid="stChatMessage"] {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px;
        padding: 14px 18px; margin-bottom: 10px; }}
[data-testid="stChatInput"] {{ border-radius: 16px; }}
[data-testid="stChatInput"] textarea {{ color: var(--text); }}
</style>
"""


@dataclass
class Settings:
    base_url: str
    model: str
    db_mode: str
    clips_root: str


def render_header(st, active):
    """Название слева, кнопки-страницы справа. Текущая страница рисуется плашкой, остальные это ссылки."""
    cols = st.columns([5] + [1.15] * len(NAV))
    cols[0].markdown(f'<div class="app-logo">{APP_NAME}</div>', unsafe_allow_html=True)
    for col, (key, label, path) in zip(cols[1:], NAV):
        if key == active:
            col.markdown(f'<span class="nav-pill">{label}</span>', unsafe_allow_html=True)
        else:
            col.page_link(path, label=label)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)


def render_settings(st):
    """Блок настроек (свёрнут). Ключи виджетов общие, поэтому значения сохраняются при переходе между страницами."""
    with st.expander("Настройки подключения"):
        base_url = st.text_input("LLM: адрес (OpenAI-совместимый API)", value=config.llm_base_url(), key="cfg_base_url")
        model = st.text_input("LLM: модель", value=config.llm_model(), key="cfg_model")
        db_mode = st.selectbox("База данных", [DEMO, PG], key="cfg_db")
        clips_root = st.text_input("Папка с клипами и кадрами", value=str(config.clips_root()), key="cfg_clips")
    return Settings(base_url=base_url, model=model, db_mode=db_mode, clips_root=clips_root)


def setup_page(st, title, active):
    """Общее начало каждой страницы: настройки окна, стили, хедер, настройки. Возвращает Settings."""
    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(build_css(), unsafe_allow_html=True)
    render_header(st, active)
    return render_settings(st)


_DEMO_CONN = None


def demo_conn():
    """Одна демо-база SQLite на процесс (тестовые данные, не боевая)."""
    global _DEMO_CONN
    if _DEMO_CONN is None:
        from agent.demo_db import make_demo_db
        _DEMO_CONN = make_demo_db(embedder=config.embedder_from_env())
    return _DEMO_CONN
