# База событий и функции ассистента (T5-T10)

**Статус проверки:** всё проверено только на SQLite (`python -m eval.run_tests`). На PostgreSQL ничего не запускалось.
Модель BGE-M3 тоже не запускалась: семантический поиск проверен только на тестовом эмбеддере (механика, не качество).

## Размещение файлов
```
surveillance-assistant/
├── agent/
│   ├── __init__.py
│   └── tools.py             # единственная копия tools.py; функции ассистента и debug_*
├── database/
│   ├── __init__.py
│   ├── schema.sql           # sources, videos, zones, events, vlm_results, description_embeddings
│   ├── seed.sql             # 19 тестовых событий (test_video_01, test_cam01) + результаты VLM A и B
│   ├── seed_events.json     # CV-файл в формате Арай
│   ├── seed_vlm_A.json      # VLM-файл варианта A (Панда)
│   ├── seed_vlm_B.json      # VLM-файл варианта B (Панда)
│   ├── loader.py            # загрузчик CV и VLM JSON, команда embed
│   └── embeddings.py        # эмбеддинги описаний: BGE-M3 и тестовый HashingEmbedder
├── eval/
│   ├── __init__.py
│   ├── conftest.py          # прячет скрипты-тесты от pytest
│   ├── questions.csv        # 30 вопросов с эталонами
│   ├── build.py             # пересоздаёт seed-файлы и questions.csv
│   ├── testutil.py          # SQLite вместо PostgreSQL, пути к файлам
│   ├── test_tools.py, test_loader.py, test_t10_tools.py
│   └── run_tests.py
└── docs/
    └── database_and_agent_tools.md
```
Каталоги `configs/`, `cv/`, `vlm/` комплект не трогает. Если `agent/__init__.py`, `database/__init__.py` или `eval/__init__.py` в проекте уже есть, нужно оставить существующие (мои пустые).

Установка: ZIP содержит папку `surveillance-assistant/`, распаковывать нужно в **родительскую** папку проекта. Ключ `-n` не перезаписывает уже существующие файлы (обычная распаковка затёрла бы ваши `__init__.py`):
```bash
unzip -n surveillance-assistant-t5-t10.zip -d <папка_на_уровень_выше_surveillance-assistant>
```
Перед распаковкой имеет смысл выполнить `git status`, после неё посмотреть `git status` и `git diff`: должны появиться только новые файлы.

Импорты: `from agent import tools` (в `database/loader.py` и тестах), `from database import embeddings` (в `agent/tools.py` при поиске по смыслу и в `database/loader.py`).

## Команды (всегда из корня `surveillance-assistant`)
Все модули запускаются как `python -m ...`: так находятся пакеты `agent`, `database`, `eval`.

```bash
# все тесты (SQLite): пересобирает данные и запускает три набора
python -m eval.run_tests            # ожидается: ВСЕ ТЕСТЫ ПРОШЛИ (SQLite); 89 + 49 + 66 проверок

# по отдельности
python -m eval.build                # пересоздать database/seed*.{sql,json} и eval/questions.csv
python -m eval.test_tools
python -m eval.test_loader
python -m eval.test_t10_tools
```

PostgreSQL (ещё не проверено):
```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/dbname"
pip install psycopg2-binary
psql "$DATABASE_URL" -f database/schema.sql
python -m database.loader cv  database/seed_events.json
python -m database.loader vlm database/seed_vlm_A.json --variant A
python -m database.loader vlm database/seed_vlm_B.json --variant B
python -c "from agent import tools; print(tools.count_events(event_type='enter'))"   # ожидается count: 4
```
`database/seed.sql` загружать не нужно, если используется загрузчик: результат тот же (проверено тестом).

Эмбеддинги для поиска по смыслу (нужны `pip install sentence-transformers` и модель `BAAI/bge-m3`):
```bash
python -m database.loader embed
```

Использование из кода: `from agent.tools import query_events, count_events, get_track_timeline, get_event_details, list_zones_cameras, search_descriptions`.

## Загрузчик
- **CV-файл Арай** (`cv`) попадает в `videos` и `events`. VLM-поля из этого файла игнорируются (с предупреждением).
- **VLM-файл Панды** (`vlm`) попадает в `vlm_results`. Вариант берётся из `prompt_variant` в файле; `--variant` проверяет, что в файле нет другого варианта. События должны уже быть в `events`, иначе файл отклоняется целиком.
- **Повторная загрузка** делает upsert: дубликатов нет, данные не меняются.
- **CV не удаляет VLM.** События обновляются через `INSERT ... ON CONFLICT DO UPDATE`, не через удаление и вставку (иначе `ON DELETE CASCADE` стёр бы результаты VLM).
- **A и B независимы.** Файл одного варианта обновляет только свои строки.
- **Файл проверяется целиком до записи.** При любой ошибке в базу ничего не пишется, все ошибки возвращаются сразу.
- События, которых нет в новом CV-файле, **не удаляются**: только предупреждение.
- `zone_id` приводится к нижнему регистру (функции ищут по зоне в нижнем регистре), с предупреждением.
- Неизвестный `source` отклоняется. Новое значение добавляют `INSERT INTO sources` после подтверждения командой.
- Загрузчика зон нет: формат файла зон неизвестен.

## Функции ассистента
Все работают только с видимыми событиями, проверяют аргументы и при ошибке возвращают `{"ok": false, "error": ...}`.

| Функция | Что делает |
|---|---|
| `query_events` | поиск по видео, камере, типу, зоне, человеку, окну времени |
| `count_events` | подсчёт с теми же фильтрами |
| `get_track_timeline` | события одного человека по порядку; если `track_id` есть в нескольких видео, требует `video_id` |
| `get_event_details` | детали события, `clip_path`, описание VLM; скрытое событие не находится (`found: false`) |
| `list_zones_cameras` | камеры, их видео и зоны |
| `search_descriptions` | поиск по описаниям VLM: `mode="keyword"` или `mode="semantic"` |

### Поиск по описаниям
- **keyword:** по словам, `score` = доля слов запроса, найденных в описании. Русский запрос на английских описаниях ничего не найдёт: для этого нужен semantic.
- **semantic:** косинусная близость эмбеддингов. Эмбеддинги хранятся в `description_embeddings` как JSON, поиск перебором в Python. **pgvector пока не используется** (на Postgres проверить нельзя; для сотен событий перебора достаточно). Интерфейс при переходе на pgvector не изменится.
- Устаревший или отсутствующий эмбеддинг пропускается, число таких событий возвращается в `skipped_no_embedding`.
- **Порог `min_score` не подобран.** Без него semantic всегда возвращает `top_k` ближайших, а ассистент должен отвечать «не нашёл». Порог нужно измерить на наших вопросах.

## Правило видимости
- `needs_vlm = false` — событие доступно ассистенту.
- `needs_vlm = true` — доступно только при `vlm_status = confirmed`.
- `not_checked` (в том числе когда строки в `vlm_results` нет) и `rejected` в обычных ответах не используются.
- Для отладки: `debug_query_events`, `debug_count_events`, `debug_get_event_details` (`view`: `all`, `unconfirmed`, `not_checked`, `rejected`). Не подключать их к чату как инструменты.

## Как поля events.json попадают в таблицы
| Поле JSON | Таблица |
|---|---|
| `video_id`, `camera_id`, `source` (верхний уровень) | `videos` |
| `event_id`, `event_type`, `track_id`, `zone_id`, `t_start`, `t_end`, `trigger_score`, `needs_vlm`, `clip_path` | `events`, ключ `(video_id, event_id)` |
| `vlm_status`, `vlm_description`, `prompt_variant` | `vlm_results`, ключ `(video_id, event_id, prompt_variant)` |

## Допущения, которые нужно подтвердить
1. **Какой вариант (A или B) определяет «подтверждено» для ассистента — ВРЕМЕННОЕ ДОПУЩЕНИЕ, команда ещё должна подтвердить основной вариант.** Сейчас `DEFAULT_VLM_VARIANT = "A"` в `agent/tools.py`. Эталоны в `eval/questions.csv` и тесты посчитаны для A. Если выбор изменится, нужно поменять константу и пересобрать эталоны (`python -m eval.build`).
2. **Значения `source`:** `own`, `meva`, `warehouse`, `kbtu` (таблица `sources`), команда их ещё не подтвердила. В тестовых данных `source = meva`.
3. **`clip_path`** оставлен без переименования (в стек-документе поле называется `keyframe_path`).
4. **Допуск по времени** в функциях не зашит. В `questions.csv` ответы точные.
5. **`videos` и `zones` добавлены**, потому что нужны для `list_zones_cameras` и чтобы `camera_id` и `source` не дублировались в каждом событии. Многоугольников зон в таблице нет. Таблицы `tracks` нет.
6. `zone_id` в `events` не привязан внешним ключом к `zones`: файл зон и `events.json` приходят отдельно.

## Безопасность seed.sql
- Зоны не удаляются. Тестовые зоны добавляются через `INSERT ... ON CONFLICT DO UPDATE`: существующая зона не затирается, пустое описание заполняется тестовым.
- Тестовые id (`test_video_01`, `test_cam01`) не совпадают с реальными данными. `seed.sql` удаляет и пересоздаёт только данные видео `test_video_01`.
- Повторный запуск `seed.sql` не создаёт дубликатов (проверено тестом).

## Что осталось на T11
Streamlit-страница и подключение LLM с вызовом функций. Схемы инструментов нужны только для шести обычных функций (без `debug_*` и без параметра `vlm_variant`). Подсчёт «правильная функция / правильный ответ» по `eval/questions.csv` использует колонки `expected_tool` и `expected_args`.
