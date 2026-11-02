"""Настройки агента из переменных окружения (см. configs/agent.example.env)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def llm_base_url():
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)


def llm_model():
    return os.environ.get("LLM_MODEL", "")


def llm_api_key():
    return os.environ.get("LLM_API_KEY", "EMPTY")


def clips_root():
    """Папка, относительно которой читаются clip_path из базы."""
    return Path(os.environ.get("CLIPS_ROOT", str(ROOT))).resolve()


def embedder_from_env():
    """EMBEDDER=hashing включает тестовый эмбеддер (только демо). Иначе None:
    search_descriptions сам загрузит BGE-M3."""
    if os.environ.get("EMBEDDER", "").strip().lower() == "hashing":
        from database.embeddings import HashingEmbedder
        return HashingEmbedder()
    return None
