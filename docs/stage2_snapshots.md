# Этап 2 (задачи Айгерим): архив срезов, поиск по одежде, страница «Архив событий»

## Что проверено, а что нет (честно)
- Всё проверено **только на SQLite** и заглушках. **PostgreSQL не запускался**: `schema_snapshots.sql`
  и загрузчик на PostgreSQL нужно прогнать у вас (команды ниже).
- **Настоящий Streamlit не запускался.** Страница `agent/pages/1_Archive.py` проверена на заглушке
  модуля `streamlit` (ловит ошибки нашего кода, но не гарантирует, что всё выглядит и работает так же
  в настоящем Streamlit). Первый запуск на машине со Streamlit обязателен; параметр
  `use_container_width` в `st.image` и `st.container(border=True)` требуют сравнительно новой версии.
- **Настоящая LLM не запускалась.** Как Qwen2.5 3B справится с девятью функциями и новыми правилами
  подсказки, неизвестно: это нужно измерить (см. «Что дальше»).
- Данные в `seed_snapshots.json` и `demo_frames/` выдуманные (картинки-заглушки), это не результат детектора и VLM.

## Что сделано
| Задача из плана | Файл |
|---|---|
| Таблицы `snapshots`, `snapshot_people` | `database/schema_snapshots.sql` (отдельный файл, `schema.sql` не тронут) |
| Загрузчик `snapshots.json` | `database/snapshots_loader.py` |
| `search_by_clothing`, `count_people_at`, `get_snapshot_summary` | `agent/snapshot_tools.py`, регистрация в `agent/tool_schemas.py` |
| Подсказка модели про новые функции | `SYSTEM_PROMPT` в `agent/runner.py` |
| Фронт: раздел «Архив событий» | `agent/pages/1_Archive.py`, помощники `agent/archive_helpers.py`; в чате (`agent/app.py`) под ответом показываются кадры |
| Тестовые данные | `database/seed_snapshots.json`, `demo_frames/`, генератор `database/make_demo_snapshots.py` |
| Тесты | `eval/test_snapshots.py` (141 проверка); `eval/test_agent.py` обновлён под 9 функций (143 проверки) |

## Порядок запуска (из корня `surveillance-assistant`)
```bash
# 1. таблицы (один раз; повторять безопасно)
export DATABASE_URL=postgresql://user:pass@host:5432/db
python -m database.snapshots_loader --init-schema
#    (то же самое: psql "$DATABASE_URL" -f database/schema_snapshots.sql)

# 2. загрузка (сначала файл Арай, потом файл Панды, порядок любой)
python -m database.snapshots_loader snapshots.json
python -m database.snapshots_loader snapshots.json --frames-root /путь/к/папке/с/кадрами   # заодно проверит, что кадры существуют

# 3. страница (раздел «Archive» появится в боковом меню)
python -m streamlit run agent/app.py
```
Тестовые данные для проверки без Арай и Панды: страница в режиме «Демо» уже содержит `seed_snapshots.json`.

Команда из плана `python -m database.loader snapshots snapshots.json` тоже работает (подкоманда `snapshots` в `database/loader.py`).
Тесты этапа 2: `python -m eval.test_snapshots`, они же входят в `python -m eval.run_tests`.

## Как грузится snapshots.json (важно для Арай и Панды)
- Заголовок: `video_id`, `camera_id`, `source` (+ `interval_sec`). Видео создаётся/обновляется так же, как в CV-загрузчике;
  неизвестный `source` отклоняется.
- Файл Арай: срезы с `people_count` и `frame_path`, **без** ключа `people`.
- Файл Панды: те же срезы **с** `people: [{person_idx, clothing_color, clothing_type, description}]`.
- Если в срезе нет ключа `people`, описания людей этого среза **не трогаются** (повторная загрузка файла Арай не
  стирает работу Панды). Если ключ есть, люди среза заменяются целиком; `"people": []` значит «людей нет».
- Файл проверяется целиком: при любой ошибке в базу ничего не пишется. Повторная загрузка без дублей.
- Цвет и тип приводятся к нижнему регистру; пустое значение становится `unknown` (с предупреждением).
- `people_count` (детектор) и число людей в `people` (VLM) могут отличаться: это предупреждение, оба значения сохраняются.
- Срезы, которых нет в новом файле, не удаляются (предупреждение). `frame_path`: `\` заменяется на `/`.

## Функции агента
| Функция | Аргументы | Что возвращает |
|---|---|---|
| `search_by_clothing` | `color`, `clothing_type` (хотя бы один), `video_id`, `t_from`, `t_to` | `found` (да/нет), `matches` (срезы: `t`, `frame_path`, люди), `uncertain_people`, `snapshots_not_described`, `note` |
| `count_people_at` | `t` (обязателен), `video_id` | ближайший срез: `people_count`, `t`, `distance_sec`, одежда |
| `get_snapshot_summary` | `video_id` | все срезы видео, `max_people`, `max_people_t` (все моменты максимума), `avg_people` |

Как это закрывает кейсы из плана: 1 → `count_people_at`; 2 и 5 → `get_snapshot_summary` (максимум, среднее, момент максимума);
3 и 6 → `search_by_clothing`; 4 → `get_snapshot_summary` или `count_people_at` (описания людей на срезе).

Правила ответа, заложенные в функции:
- Человек с цветом или типом `unknown` никогда не совпадает с запросом. Чтобы «нет» было честным, в результате считаются
  `uncertain_people` и `snapshots_not_described` (срез, где люди есть, но VLM ещё не описал); при них в `note` сказано, что «нет» не точно.
- `count_people_at` берёт ближайший срез и **возвращает расстояние до него**; допуск по времени не зашит (команда его не
  подтверждала). Модель в подсказке обязана назвать время среза, если оно отличается от запрошенного.
- Срезы не связаны трекингом: `person_idx` это не `track_id`. Вопрос «какого цвета куртка у человека 7» остаётся
  «ответить нельзя» (вопрос q28 в `questions.csv` не меняется; в подсказке это правило 4 и 10).
- Цвет и тип должны относиться к **одному** человеку: красная футболка у одного и синяя кофта у другого не дают «красную кофту».
- Русские запросы переводятся в английские (красная → red, кофта → sweater и т. д.). Тип `sweater` объединяет
  sweater/hoodie/cardigan/sweatshirt/pullover/jumper; `jacket` = jacket/windbreaker/parka/bomber/puffer; `coat` = coat/overcoat/raincoat;
  `t-shirt` и `shirt` разные. Цвет `blue` совпадает и с `dark blue` (слова запроса входят в цвет), но `dark blue` не совпадает с `blue`.
  Таблицы лежат вверху `agent/snapshot_tools.py` (`TYPE_GROUPS`, `_RU_COLORS`, `_RU_TYPES`) и правятся в одном месте.
- Срезы не подчиняются правилу `needs_vlm`/`confirmed` из событий: это отдельный слой.

## Что изменилось в существующих файлах (точечные правки поверх репозитория, коммит 187c291)
Ничего не заменялось целиком. Твои доработки этапа 1 сохранены: расширенная подсказка (правила про VLM, аргументы, ответы),
починка `pick_up_or_put_candidate` и удаление угаданного `camera_id` в `call_tool`/`run_agent`, `_resolve_vlm_variant`, фильтр
необработанных событий в `load_vlm_file`, тип `pickup_or_put_candidate` в схеме.
- `agent/tool_schemas.py`: импорт `snapshot_tools`, три новые спецификации и три записи в `TOOL_FUNCS`.
- `agent/runner.py`: в `SYSTEM_PROMPT` добавлен блок «SNAPSHOT ARCHIVE» (правила 7–9: `search_by_clothing`, `count_people_at`,
  `get_snapshot_summary`) и заменена одна строка ANSWER RULES «Do not infer clothing colors...»: теперь одежда известна только из
  срезов, а одежда конкретного `track_id` определить нельзя. В `slim_for_model` модели не отдаются пути к кадрам.
- `agent/demo_db.py`: демо-база дополнительно загружает таблицы срезов и `seed_snapshots.json`.
- `agent/app.py`: под ответом показываются кадры из архива.
- `database/loader.py`: подкоманда `snapshots`.
- `eval/test_agent.py`: ожидание «9 функций»; `eval/run_tests.py`, `eval/conftest.py`: добавлен `test_snapshots`.

## Открытые вопросы к команде (нужны решения, я их не принимала)
1. **Словарь `clothing_color` и `clothing_type` для Панды.** Это главный риск: поиск работает по точным значениям,
   поэтому VLM нужно просить выбирать **только** из фиксированного списка (цвета: red, green, blue, black, white, gray, yellow,
   orange, brown, pink, purple, beige, unknown; типы: sweater, t-shirt, shirt, jacket, coat, dress, unknown). Иначе «красная кофта»
   не найдётся из-за «crimson pullover». Группы синонимов в `snapshot_tools.py` смягчают, но не заменяют это.
2. Политика `unknown`: сейчас человек остаётся в списке с `unknown` (как предложено в плане) и не совпадает ни с каким запросом.
3. Допуск по времени для `count_people_at` (сейчас нет; возвращается `distance_sec`).
4. Связь `person_idx` с `track_id`: не делалась (обсуждается в плане); q28 остаётся «ответить нельзя».
5. Одно и то же MEVA-видео или новое; кадр или клип (в плане решено: кадр).
6. Дополнительные вопросы в `eval/questions.csv` на срезы (бонусные «красная кофта?», «зелёная куртка?») я не добавляла:
   без них вы не измерите качество новой функции на модели. Могу подготовить отдельный `questions_stage2.csv`.
