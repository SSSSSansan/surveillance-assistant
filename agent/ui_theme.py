"""UI theme for the Streamlit pages: glass-style light design, header with navigation, chat bubbles, settings block.

All colours live in PALETTES. The default palette is "light" (soft blue/sky gradients, white glass cards).
A dark variant is available with the environment variable UI_THEME=dark. To change colours, edit only the palette dict.

The module does not import streamlit at load time (Streamlit is passed into the functions), so it can be tested
without Streamlit. The layout was only checked against a stub of Streamlit: how it looks in a browser has to be
checked by running `python -m streamlit run agent/app.py`.
"""
import os
from dataclasses import dataclass

from agent import config

APP_NAME = "Surveillance Assistant"        # name in the top-left corner
DEMO = "Demo (SQLite, test data)"
PG = "PostgreSQL (DATABASE_URL)"

# (key, label, page path relative to agent/app.py)
NAV = [("archive", "Archive", "pages/1_Archive.py"),
       ("assistant", "Assistant", "app.py")]

PALETTES = {
    "light": {   # Background #F5F8FC, Cards #FFFFFF, Primary #3978E8, AI accent #38BDF8, Ink #1E293B
        "bg": "#F5F8FC", "shell": "rgba(255,255,255,.46)", "card": "rgba(255,255,255,.84)",
        "text": "#1E293B", "muted": "#64748B", "border": "rgba(57,120,232,.16)",
        "primary": "#3978E8", "accent": "#38BDF8", "soft": "rgba(57,120,232,.10)", "chip": "#E6EFFD",
        "shadow": "rgba(57,120,232,.16)", "ok": "#22C55E", "grain": ".16",
        "blob1": "rgba(56,189,248,.80)", "blob2": "rgba(57,120,232,.62)", "blob3": "rgba(167,139,250,.62)",
        "blob4": "rgba(244,114,182,.38)", "blob5": "rgba(45,212,191,.38)",
    },
    "dark": {    # UI_THEME=dark
        "bg": "#0F172A", "shell": "rgba(30,41,59,.62)", "card": "rgba(30,41,59,.82)",
        "text": "#E2E8F0", "muted": "#94A3B8", "border": "rgba(148,163,184,.22)",
        "primary": "#5B93F0", "accent": "#38BDF8", "soft": "rgba(91,147,240,.16)", "chip": "rgba(148,163,184,.16)",
        "shadow": "rgba(0,0,0,.35)", "ok": "#4ADE80", "grain": ".10",
        "blob1": "rgba(56,189,248,.28)", "blob2": "rgba(57,120,232,.36)", "blob3": "rgba(167,139,250,.28)",
        "blob4": "rgba(244,114,182,.16)", "blob5": "rgba(45,212,191,.14)",
    },
}


def palette(name=None):
    name = (name or os.environ.get("UI_THEME") or "light").strip().lower()
    return PALETTES.get(name, PALETTES["light"])


_NOISE = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'>"
          "<filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/>"
          "</filter><rect width='100%' height='100%' filter='url(%23n)' opacity='.55'/></svg>")


def build_css(p=None):
    p = p or palette()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {{
  --bg:{p['bg']}; --shell:{p['shell']}; --card:{p['card']}; --text:{p['text']}; --muted:{p['muted']};
  --border:{p['border']}; --primary:{p['primary']}; --accent:{p['accent']}; --soft:{p['soft']};
  --chip:{p['chip']}; --shadow:{p['shadow']}; --ok:{p['ok']};
  --grad: linear-gradient(135deg, {p['primary']} 0%, {p['accent']} 100%);
}}
html, body, .stApp, [data-testid="stAppViewContainer"] {{ font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }}
.stApp {{
  color: var(--text);
  background:
    radial-gradient(50% 45% at 95% 0%, {p['blob1']} 0%, transparent 65%),
    radial-gradient(50% 50% at 0% 100%, {p['blob2']} 0%, transparent 65%),
    radial-gradient(45% 45% at 100% 100%, {p['blob3']} 0%, transparent 65%),
    radial-gradient(40% 40% at 0% 0%, {p['blob4']} 0%, transparent 65%),
    radial-gradient(35% 35% at 50% 100%, {p['blob5']} 0%, transparent 65%),
    var(--bg);
  background-attachment: fixed;
}}
.stApp::before {{ content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
  opacity: {p['grain']}; background-image: url("{_NOISE}"); }}
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}

/* ---- glass shell around the page ---- */
.block-container {{ max-width: 1240px; margin: 1rem auto 1.5rem; padding: 1.4rem 2.2rem 1.8rem;
  background: var(--shell); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
  border: 1px solid rgba(255,255,255,.65); border-radius: 30px; box-shadow: 0 24px 70px var(--shadow); }}

/* ---- header ---- */
.brand {{ display: flex; align-items: center; gap: 12px; padding-top: .1rem; }}
.app-logo {{ font-size: 1.35rem; font-weight: 800; letter-spacing: -.02em; color: var(--text); }}
.orb {{ position: relative; flex: 0 0 auto; width: 44px; height: 44px; border-radius: 50%;
  background: radial-gradient(circle at 30% 28%, #bae6fd 0%, var(--accent) 30%, var(--primary) 72%, #2354c7 100%);
  box-shadow: 0 8px 22px rgba(57,120,232,.45), inset 0 -4px 10px rgba(255,255,255,.25); }}
.orb::after {{ content: ""; position: absolute; inset: 31%; border: 3px solid #fff; border-radius: 34%;
  transform: rotate(45deg); opacity: .95; }}
.orb.big {{ width: 56px; height: 56px; }}
.nav-pill {{ display: inline-block; padding: .46rem 1.15rem; border-radius: 999px; font-weight: 600; color: #fff;
  background: var(--grad); box-shadow: 0 6px 18px var(--shadow); }}
[data-testid="stPageLink-NavLink"] {{ border-radius: 999px; padding: .46rem 1.15rem; border: 1px solid transparent;
  color: var(--muted); font-weight: 600; }}
[data-testid="stPageLink-NavLink"]:hover {{ background: var(--soft); color: var(--primary); }}
.header-line {{ height: 1px; background: var(--border); margin: .8rem 0 1.3rem; }}

/* ---- page title ---- */
.page-title {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -.025em; margin: .1rem 0 .9rem; color: var(--text); }}
.page-title .ico {{ display: inline-block; margin-right: .55rem; background: var(--grad); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent; }}
.page-count {{ text-align: right; color: var(--muted); padding-top: 1.1rem; font-weight: 500; }}

/* ---- inputs and buttons ---- */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {{
  border-radius: 14px !important; background: var(--card) !important; border-color: var(--border) !important; }}
[data-testid="stTextInput"] input {{ color: var(--text); }}
div.stButton > button {{ border-radius: 999px; border: 1px solid var(--border); background: var(--card); color: var(--text);
  font-weight: 500; box-shadow: 0 2px 8px var(--shadow); transition: all .15s ease; }}
div.stButton > button:hover {{ border-color: var(--primary); color: var(--primary); transform: translateY(-1px); }}
[data-testid="stExpander"] {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; }}
[data-testid="stExpander"] details {{ border: 0; }}
[data-testid="stMetric"] {{ background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 10px 16px; }}

/* ---- archive cards ---- */
.card {{ display: flex; gap: 18px; background: var(--card); border: 1px solid var(--border); border-radius: 20px;
  padding: 14px 18px; margin: 0 0 14px; box-shadow: 0 8px 24px var(--shadow); }}
.card.hit {{ border-color: var(--accent); box-shadow: 0 0 0 2px rgba(56,189,248,.45), 0 10px 28px var(--shadow); }}
.card-img {{ flex: 0 0 200px; }}
.card-img img {{ width: 200px; border-radius: 14px; display: block; }}
.card-img .noimg {{ width: 200px; height: 112px; border-radius: 14px; background: var(--chip); color: var(--muted);
  display: flex; align-items: center; justify-content: center; font-size: .8rem; text-align: center; padding: 6px; }}
.card-body {{ flex: 1; min-width: 0; }}
.card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 6px; }}
.ts {{ color: var(--primary); font-weight: 800; font-size: 1.2rem; }}
.meta {{ color: var(--muted); font-size: .9rem; }}
.badge {{ background: var(--chip); border-radius: 999px; padding: 2px 12px; font-size: .82rem; font-weight: 600; color: var(--primary); }}
.badge.hit {{ background: var(--grad); color: #fff; }}
.card-text {{ line-height: 1.55; }}
.chip {{ display: inline-flex; align-items: center; gap: 6px; margin: 6px 6px 0 0; padding: 3px 12px;
  border-radius: 999px; background: var(--chip); font-size: .84rem; font-weight: 500; }}
.dot {{ width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(100,116,139,.45); display: inline-block; }}
.muted {{ color: var(--muted); font-size: .9rem; }}
@media (max-width: 760px) {{ .card {{ flex-direction: column; }} .card-img, .card-img img, .card-img .noimg {{ width: 100%; flex-basis: auto; }} }}

/* ---- assistant header ---- */
.bot-head {{ display: flex; align-items: center; gap: 14px; padding: 4px 0 6px; }}
.bot-name {{ font-size: 1.5rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }}
.bot-status {{ color: var(--muted); font-size: .88rem; margin-top: 3px; }}
.bot-status::before {{ content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 7px; border-radius: 50%;
  background: var(--ok); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }}

/* ---- chat ---- */
/* one soft card for the whole chat (header, quick questions, messages, input): shadow instead of a dark border */
.st-key-chat_panel {{ background: var(--card); border-radius: 26px; padding: 16px 22px 20px; gap: .7rem;
  box-shadow: 0 18px 48px var(--shadow), inset 0 0 0 1px rgba(255,255,255,.75); }}
.st-key-chat_messages {{ min-height: 340px; justify-content: flex-end; gap: .2rem; }}

/* message row: avatar is placed absolutely, so it always sits right next to the bubble */
[data-testid="stChatMessage"] {{ display: block !important; position: relative; background: transparent; border: 0;
  padding: .3rem 0 .3rem 46px; }}
[data-testid^="stChatMessageAvatar"] {{ position: absolute; left: 0; top: .3rem; width: 34px; height: 34px; margin: 0;
  border-radius: 50%; color: #fff; background: var(--grad); box-shadow: 0 6px 16px var(--shadow); }}
[data-testid="stChatMessageContent"] {{ margin: 0; width: fit-content; max-width: min(100%, 880px); padding: 12px 18px;
  background: var(--card); border: 1px solid var(--border); border-radius: 18px 18px 18px 6px;
  box-shadow: 0 6px 20px var(--shadow); }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ padding: .3rem 46px .3rem 0; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {{
  left: auto; right: 0; background: #fff; color: var(--accent); border: 1px solid var(--border); }}
[data-testid="stChatMessageAvatarUser"] *, [data-testid="stChatMessageAvatarUser"] svg {{ color: var(--accent) !important; fill: currentColor; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
  margin-left: auto; max-width: 72%; background: var(--grad); border-color: transparent; border-radius: 18px 18px 6px 18px; color: #fff; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] * {{ color: #fff; }}

/* input field */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb] {{
  border-radius: 18px !important; background: #fff !important; }}
[data-testid="stChatInput"] {{ border: 1px solid var(--border); box-shadow: 0 4px 16px var(--shadow); margin: 0; }}
[data-testid="stChatInput"] textarea {{ color: var(--text); }}
[data-testid="stChatInputSubmitButton"] {{ background: var(--grad); color: #fff; border-radius: 50%; }}
</style>
"""


@dataclass
class Settings:
    base_url: str
    model: str
    db_mode: str
    clips_root: str


def render_header(st, active):
    """Logo and name on the left, page buttons on the right. The current page is a filled pill, others are links."""
    cols = st.columns([5] + [1.15] * len(NAV))
    cols[0].markdown(f'<div class="brand"><span class="orb"></span><span class="app-logo">{APP_NAME}</span></div>',
                     unsafe_allow_html=True)
    for col, (key, label, path) in zip(cols[1:], NAV):
        if key == active:
            col.markdown(f'<span class="nav-pill">{label}</span>', unsafe_allow_html=True)
        else:
            col.page_link(path, label=label)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)


def render_settings(st):
    """Settings block (collapsed). Widget keys are shared, so values persist when switching pages."""
    with st.expander("Connection settings"):
        base_url = st.text_input("LLM: endpoint (OpenAI-compatible API)", value=config.llm_base_url(), key="cfg_base_url")
        model = st.text_input("LLM: model", value=config.llm_model(), key="cfg_model")
        db_mode = st.selectbox("Database", [DEMO, PG], key="cfg_db")
        clips_root = st.text_input("Folder with clips and frames", value=str(config.clips_root()), key="cfg_clips")
    return Settings(base_url=base_url, model=model, db_mode=db_mode, clips_root=clips_root)


def setup_page(st, title, active):
    """Common start of every page: window settings, styles, header, settings. Returns Settings."""
    st.set_page_config(page_title=f"{title} | {APP_NAME}", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(build_css(), unsafe_allow_html=True)
    render_header(st, active)
    return render_settings(st)


_DEMO_CONN = None


def demo_conn():
    """One demo SQLite database per process (test data, not production)."""
    global _DEMO_CONN
    if _DEMO_CONN is None:
        from agent.demo_db import make_demo_db
        _DEMO_CONN = make_demo_db(embedder=config.embedder_from_env())
    return _DEMO_CONN