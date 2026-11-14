-- Этап 2: срезы видео каждые N секунд (архив) и описание людей на срезе.
-- Отдельный файл, чтобы не трогать существующий schema.sql. Применять ПОСЛЕ schema.sql
-- (нужна таблица videos). Повторный запуск безопасен (IF NOT EXISTS).
--   psql "$DATABASE_URL" -f database/schema_snapshots.sql
--   или: python -m database.snapshots_loader --init-schema
-- Проверялось только на SQLite; на PostgreSQL не запускалось.
--
-- snapshots.json (Арай: кадры и people_count, Панда: people[...] с одеждой) -> эти две таблицы.
-- person_idx: номер человека внутри ОДНОГО среза. Это НЕ track_id: срезы не связаны трекингом.

CREATE TABLE IF NOT EXISTS snapshots (
    video_id      TEXT NOT NULL REFERENCES videos (video_id) ON DELETE CASCADE,
    snapshot_id   TEXT NOT NULL,                                   -- sn_0000, sn_0001, ...
    t             DOUBLE PRECISION NOT NULL CHECK (t >= 0),        -- секунда среза от начала видео
    frame_path    TEXT,                                            -- путь к кадру для фронта (со слэшами '/')
    people_count  INTEGER NOT NULL CHECK (people_count >= 0),      -- считает детектор (Арай)
    PRIMARY KEY (video_id, snapshot_id)
);

CREATE TABLE IF NOT EXISTS snapshot_people (
    video_id        TEXT NOT NULL,
    snapshot_id     TEXT NOT NULL,
    person_idx      INTEGER NOT NULL CHECK (person_idx >= 0),
    clothing_color  TEXT NOT NULL DEFAULT 'unknown',               -- английский, нижний регистр; unknown = не уверен
    clothing_type   TEXT NOT NULL DEFAULT 'unknown',
    description     TEXT,
    PRIMARY KEY (video_id, snapshot_id, person_idx),
    FOREIGN KEY (video_id, snapshot_id) REFERENCES snapshots (video_id, snapshot_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_snapshots_video_t ON snapshots (video_id, t);
CREATE INDEX IF NOT EXISTS idx_snapshot_people_color ON snapshot_people (clothing_color);
