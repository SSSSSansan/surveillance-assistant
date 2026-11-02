"""T10: эмбеддинги описаний VLM для поиска по смыслу.

ВАЖНО: настоящая модель BGE-M3 здесь не запускалась (нет модели и доступа к ней).
HashingEmbedder ниже нужен только для тестов: он проверяет, что поиск, хранение и устаревание
эмбеддингов работают, но НЕ проверяет качество поиска по смыслу.
Качество BGE-M3 против multilingual-e5 (пункт 8 списка «Проверить замером») нужно мерить отдельно.
"""
import hashlib
import json
import math
import re

DEFAULT_MODEL = "BAAI/bge-m3"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class HashingEmbedder:
    """Только для тестов: мешок слов, хешированный в вектор. Не понимает смысл и языки."""
    def __init__(self, dim=64, name="hashing-test"):
        self.dim, self.name = dim, name

    def embed(self, texts):
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            for tok in re.findall(r"[^\W_]+", t.lower()):
                vec[int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class BGEM3Embedder:
    """BGE-M3 через sentence-transformers. Модель грузится при первом вызове embed()."""
    def __init__(self, model_name=DEFAULT_MODEL):
        self.name = model_name
        self._model = None

    def embed(self, texts):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("для режима semantic нужен пакет sentence-transformers "
                                   "(pip install sentence-transformers)") from e
            self._model = SentenceTransformer(self.name)
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()


def get_default_embedder():
    return BGEM3Embedder()


def build_embeddings(conn, embedder, model=None, batch_size=32):
    """Считает эмбеддинги для описаний в vlm_results, которых ещё нет или которые изменились.
    Безопасно запускать повторно."""
    model = model or getattr(embedder, "name", DEFAULT_MODEL)
    cur = conn.cursor()
    try:
        cur.execute("SELECT video_id, event_id, prompt_variant, vlm_description FROM vlm_results "
                    "WHERE vlm_description IS NOT NULL")
        rows = cur.fetchall()
        cur.execute("SELECT video_id, event_id, prompt_variant, text_hash FROM description_embeddings "
                    "WHERE model = %s", [model])
        have = {(r[0], r[1], r[2]): r[3] for r in cur.fetchall()}
        todo = [r for r in rows if have.get((r[0], r[1], r[2])) != text_hash(r[3])]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            vecs = embedder.embed([r[3] for r in chunk])
            if len(vecs) != len(chunk):
                raise RuntimeError("embedder вернул неверное число векторов")
            for r, vec in zip(chunk, vecs):
                cur.execute(
                    "INSERT INTO description_embeddings (video_id, event_id, prompt_variant, model, "
                    "text_hash, embedding_json) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (video_id, event_id, prompt_variant, model) DO UPDATE SET "
                    "text_hash = excluded.text_hash, embedding_json = excluded.embedding_json",
                    [r[0], r[1], r[2], model, text_hash(r[3]), json.dumps([float(x) for x in vec])])
        conn.commit()
        return {"ok": True, "model": model, "embedded": len(todo), "already_current": len(rows) - len(todo)}
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        return {"ok": False, "error": f"эмбеддинги не посчитаны, изменения отменены: {e}"}
    finally:
        cur.close()
