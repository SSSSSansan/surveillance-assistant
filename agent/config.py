"""Настройки агента из переменных окружения (см. configs/agent.example.env)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def llm_base_url():
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)


def llm_model():
    return os.environ.get("LLM_MODEL", "")


def llm_max_tokens():
    """Лимит токенов ответа модели. Модели с «размышлением» тратят часть на рассуждения, 512 им мало."""
    try:
        return max(64, int(os.environ.get("LLM_MAX_TOKENS", "1536")))
    except ValueError:
        return 1536


def llm_timeout():
    """Сколько секунд ждать ответ модели (LLM_TIMEOUT). Локальной модели на CPU с длинным промптом 120 с бывает мало."""
    try:
        return max(10.0, float(os.environ.get("LLM_TIMEOUT", "120")))
    except ValueError:
        return 120.0


def default_video_id():
    """Видео по умолчанию, когда в архиве несколько видео, а в вопросе видео не названо (env DEFAULT_VIDEO_ID)."""
    return os.environ.get("DEFAULT_VIDEO_ID", "").strip() or None


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