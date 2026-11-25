-- Этап 3 (T15): эталон разметки (gold) и журнал прогонов методов.
-- Отдельный файл, существующие схемы не трогаем. Повторный запуск безопасен (IF NOT EXISTS).
--   python -m database.gold_loader --init-schema
-- Проверялось только на SQLite; на PostgreSQL не запускалось.
-- Внешних ключей на videos/snapshots нет специально: кадры и кропы эталона грузятся из индексов Арай
-- независимо от того, загружены ли срезы Панды (и для видео 2 тоже).

-- Журнал прогонов (runs.jsonl Сании, строка файла = строка таблицы). params = JSON-текст как есть.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    method          TEXT NOT NULL,
    model           TEXT,
    mode            TEXT,
    prompt_variant  TEXT,
    weights         TEXT,
    started_at      TEXT,
    duration_sec    DOUBLE PRECISION,
    params          TEXT
);

-- Список кропов для разметки (crops_index.json Арай). Размечаем только kept = TRUE.
CREATE TABLE IF NOT EXISTS gold_crops (
    video_id     TEXT NOT NULL,
    snapshot_id  TEXT NOT NULL,
    person_idx   INTEGER NOT NULL CHECK (person_idx >= 0),
    crop_id      TEXT NOT NULL,
    t            DOUBLE PRECISION,
    box_h        DOUBLE PRECISION,
    box_w        DOUBLE PRECISION,
    h_bucket     TEXT,
    kept         BOOLEAN NOT NULL,
    reason       TEXT,                  -- у отброшенных: dup_of_<crop_id>
    frame_path   TEXT,
    torso_path   TEXT,
    full_path    TEXT,
    PRIMARY KEY (video_id, snapshot_id, person_idx)
);

-- Кадры для счёта людей (frames_index.json Арай). people_count модели сюда НЕ кладём,
-- чтобы разметчик его не видел.
CREATE TABLE IF NOT EXISTS gold_frames_index (
    video_id     TEXT NOT NULL,
    snapshot_id  TEXT NOT NULL,
    t            DOUBLE PRECISION,
    frame_path   TEXT,
    PRIMARY KEY (video_id, snapshot_id)
);

-- Разметка одежды: одна строка на (кроп, разметчик). Один кроп размечают двое, 50 кропов все трое.
CREATE TABLE IF NOT EXISTS labels_gold_people (
    video_id     TEXT NOT NULL,
    snapshot_id  TEXT NOT NULL,
    person_idx   INTEGER NOT NULL CHECK (person_idx >= 0),
    labeler      TEXT NOT NULL,
    color        TEXT NOT NULL,
    color_2      TEXT,
    type         TEXT NOT NULL,
    visible      TEXT NOT NULL CHECK (visible IN ('да', 'спиной', 'частично', 'не видно')),
    box_h        DOUBLE PRECISION,
    labeled_at   TEXT,
    PRIMARY KEY (video_id, snapshot_id, person_idx, labeler)
);

-- Разметка счёта людей: одна строка на (кадр, разметчик).
CREATE TABLE IF NOT EXISTS labels_gold_frames (
    video_id           TEXT NOT NULL,
    snapshot_id        TEXT NOT NULL,
    labeler            TEXT NOT NULL,
    true_people_count  INTEGER NOT NULL CHECK (true_people_count >= 0),
    labeled_at         TEXT,
    PRIMARY KEY (video_id, snapshot_id, labeler)
);

CREATE INDEX IF NOT EXISTS idx_gold_crops_kept ON gold_crops (video_id, kept);
CREATE INDEX IF NOT EXISTS idx_labels_people_labeler ON labels_gold_people (labeler);
CREATE INDEX IF NOT EXISTS idx_runs_video ON runs (video_id, method);
