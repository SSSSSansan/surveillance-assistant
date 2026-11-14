"""SQLite-база в памяти с тестовыми данными: для демо страницы, оценки на 30 вопросах и тестов.
Подменяет %s на ?, чтобы функции из agent/tools.py работали без PostgreSQL.
Это НЕ боевая база: в боевом режиме используется PostgreSQL через DATABASE_URL."""
import sqlite3
from pathlib import Path

DB_DIR = Path(__file__).resolve().parents[1] / "database"


class Cur:
    def __init__(self, cur): self.cur = cur
    def execute(self, sql, params=()):
        self.cur.execute(sql.replace("%s", "?"), params)
        return self
    @property
    def description(self): return self.cur.description
    def fetchall(self): return self.cur.fetchall()
    def close(self): self.cur.close()


class Shim:
    """Минимальный аналог соединения psycopg2 поверх sqlite3."""
    def __init__(self, raw): self.raw = raw
    def cursor(self): return Cur(self.raw.cursor())
    def commit(self): self.raw.commit()
    def rollback(self): self.raw.rollback()
    def close(self): pass


def make_demo_db(embedder=None):
    """Схема + seed.sql (+ срезы этапа 2, если файлы есть). Если передан embedder, считает эмбеддинги описаний (для semantic-поиска)."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.execute("PRAGMA foreign_keys = ON")
    for name in ("schema.sql", "seed.sql"):
        raw.executescript((DB_DIR / name).read_text(encoding="utf-8"))
    conn = Shim(raw)
    if (DB_DIR / "schema_snapshots.sql").exists() and (DB_DIR / "seed_snapshots.json").exists():
        from database import snapshots_loader      # этап 2: архив срезов с тестовыми данными
        snapshots_loader.apply_schema(conn)
        rep = snapshots_loader.load_snapshots_file(DB_DIR / "seed_snapshots.json", conn=conn)
        if not rep["ok"]:
            raise RuntimeError(f"тестовые срезы не загрузились: {rep['errors']}")
    if embedder is not None:
        from database import embeddings
        embeddings.build_embeddings(conn, embedder)
    return conn
