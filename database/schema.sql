-- Схема журнала событий (T5, версия 2).
-- Проверялась только на SQLite. На PostgreSQL не запускалась.
--
-- Соответствие полей events.json таблицам:
--   верхний уровень файла:  video_id, camera_id, source  -> videos
--   событие (CV, Арай):     event_id, event_type, track_id, zone_id, t_start, t_end,
--                           trigger_score, needs_vlm, clip_path   -> events
--   результат VLM (Панда):  vlm_status, vlm_description, prompt_variant -> vlm_results

-- Допустимые значения source. Список меняется строками, схему трогать не нужно:
--   INSERT INTO sources (source) VALUES ('new_value');
--   DELETE FROM sources WHERE source = 'old_value';   -- если на него не ссылается ни одно видео
-- Значения ниже предложенные, команда их ещё не подтвердила.
CREATE TABLE IF NOT EXISTS sources (
    source TEXT PRIMARY KEY
);
INSERT INTO sources (source) VALUES ('own'), ('meva'), ('warehouse'), ('kbtu')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS videos (
    video_id   TEXT PRIMARY KEY,
    camera_id  TEXT NOT NULL,
    source     TEXT NOT NULL REFERENCES sources (source)
);

-- Зоны задаются на камеру. Нужна для list_zones_cameras.
-- Многоугольники пока не храним: для списка они не нужны. Можно добавить колонку позже.
-- Внешнего ключа events.zone_id -> zones нет: файл зон и events.json приходят отдельно.
CREATE TABLE IF NOT EXISTS zones (
    camera_id    TEXT NOT NULL,
    zone_id      TEXT NOT NULL,
    description  TEXT,
    PRIMARY KEY (camera_id, zone_id)
);

-- Основные события CV. event_id уникален только внутри видео.
CREATE TABLE IF NOT EXISTS events (
    video_id       TEXT NOT NULL REFERENCES videos (video_id),
    event_id       TEXT NOT NULL,
    event_type     TEXT NOT NULL CHECK (event_type IN (
                       'enter','exit','approach_zone','leave_zone',
                       'dwell_in_zone','pick_up','put_down','abandoned_object')),
    track_id       INTEGER,
    zone_id        TEXT,
    t_start        DOUBLE PRECISION NOT NULL CHECK (t_start >= 0),  -- секунды от начала видео
    t_end          DOUBLE PRECISION NOT NULL,
    trigger_score  DOUBLE PRECISION CHECK (trigger_score BETWEEN 0 AND 1),
    needs_vlm      BOOLEAN NOT NULL DEFAULT FALSE,
    clip_path      TEXT,
    PRIMARY KEY (video_id, event_id),
    CHECK (t_end >= t_start)
);

-- Результаты проверки VLM. Способы A и B лежат отдельными строками.
-- PRIMARY KEY гарантирует уникальность (video_id, event_id, prompt_variant).
CREATE TABLE IF NOT EXISTS vlm_results (
    video_id         TEXT NOT NULL,
    event_id         TEXT NOT NULL,
    prompt_variant   TEXT NOT NULL CHECK (prompt_variant IN ('A','B')),
    vlm_status       TEXT NOT NULL DEFAULT 'not_checked'
                         CHECK (vlm_status IN ('not_checked','confirmed','rejected')),
    vlm_description  TEXT,
    PRIMARY KEY (video_id, event_id, prompt_variant),
    FOREIGN KEY (video_id, event_id) REFERENCES events (video_id, event_id) ON DELETE CASCADE
);

-- Эмбеддинги описаний VLM для поиска по смыслу (T10).
-- Хранятся как JSON-текст, поиск считается в Python (перебором). pgvector пока не используется.
-- text_hash нужен, чтобы заметить, что описание изменилось и эмбеддинг устарел.
CREATE TABLE IF NOT EXISTS description_embeddings (
    video_id        TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    prompt_variant  TEXT NOT NULL,
    model           TEXT NOT NULL,
    text_hash       TEXT NOT NULL,
    embedding_json  TEXT NOT NULL,
    PRIMARY KEY (video_id, event_id, prompt_variant, model),
    FOREIGN KEY (video_id, event_id, prompt_variant)
        REFERENCES vlm_results (video_id, event_id, prompt_variant) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_type  ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_events_zone  ON events (zone_id);
CREATE INDEX IF NOT EXISTS idx_events_track ON events (video_id, track_id);
CREATE INDEX IF NOT EXISTS idx_events_time  ON events (video_id, t_start);
