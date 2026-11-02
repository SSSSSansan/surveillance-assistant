"""Общие помощники для тестов: SQLite вместо PostgreSQL (подмена %s на ?) и пути к файлам.
Пути считаются от расположения этого файла, поэтому тесты не зависят от текущей папки,
но запускать их нужно из корня проекта через `python -m eval.<модуль>` (см. docs)."""
import sqlite3
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
DB_DIR = ROOT / "database"


def db_path(name):
    """Путь к файлу в database/ (schema.sql, seed.sql, seed_*.json)."""
    return str(DB_DIR / name)


def eval_path(name):
    """Путь к файлу в eval/ (questions.csv)."""
    return str(EVAL_DIR / name)


def read_db(name):
    with open(db_path(name), encoding="utf-8") as f:
        return f.read()


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


def fresh(seed=True):
    """Новая база в памяти со схемой (и, по умолчанию, с seed.sql). Возвращает (raw, shim)."""
    raw = sqlite3.connect(":memory:")
    raw.execute("PRAGMA foreign_keys = ON")
    raw.executescript(read_db("schema.sql"))
    if seed:
        raw.executescript(read_db("seed.sql"))
    return raw, Shim(raw)
