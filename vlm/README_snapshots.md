# VLM: одежда на срезах (этап 2)

Файлы в `vlm/`:
- `describe_snapshot.py` — запросы к vLLM (один кадр целиком или один кроп человека), промпты, словари, разбор JSON.
- `batch_describe_snapshots.py` — прогон всех срезов, пишет НОВЫЙ json. Режимы `crops` (по умолчанию) и `frame`.
- `annotate_snapshots.py` — рисует результат на копиях кадров для проверки глазами.
- `make_snapshots_stub.py` — запасной: нарезать кадры самим (без боксов и счёта людей, годится только для режима `frame`).
- `imgutil.py` — кроп и рисование; работает с OpenCV ИЛИ Pillow (что есть, то и берёт).
- `snapshots_sample.json` — пример формата. **Синтетика** (значения выдуманы, `video_id = sample_video_synthetic`, кадров нет).

## Почему режим `crops`
Целый кадр с 12–15 людьми Qwen3-VL 4B описывает плохо: путает число людей, почти всем ставит `black`, пишет шаблонные
описания. Поэтому на каждую рамку из `boxes` (их делает YOLO у Арай) идёт отдельный запрос с кропом человека
(рамка + 12% отступа, внутри кадра; крошечные кропы увеличиваются до 336 px по длинной стороне — **НЕ ПРОВЕРЕНО, что
это помогает**). `person_idx` = индекс рамки в `boxes`, у каждого человека есть поле `box`. `people_count` остаётся от детектора.

## Запуск (из `~/snapshots` на сервере, venv включён; vLLM на 127.0.0.1:8765)
```bash
# один кроп/кадр (проверка)
python3 vlm/describe_snapshot.py frames/sn_0000.jpg --box 318,377,518,694
# только срез sn_0000
python3 vlm/batch_describe_snapshots.py snapshots.json . out_sn0000.json --only sn_0000
# все срезы (4 запроса параллельно)
python3 vlm/batch_describe_snapshots.py snapshots.json . snapshots_vlm.json --workers 4
# рисунки для проверки
python3 vlm/annotate_snapshots.py snapshots_vlm.json . --out annotated
```
Режим целого кадра: `--mode frame`. Мелкие рамки (< `--min-box-height 60` / `--min-box-width 30` px) не отправляются:
человек получает `unknown`. Пороги подобраны на глаз, не проверены.

## Статусы
- `vlm_status`: `ok` — всё описано; `partial` — у части людей кроп не удался (они `unknown`, номера в `vlm_crop_errors`);
  `error` — не описалось ничего: ключа `people` в срезе нет, и загрузчик не трогает старых людей этого среза в базе.
- `vlm_crops_skipped` — индексы слишком мелких рамок. `vlm_people_count` — сколько людей выдал VLM-слой.
- Ошибки и сырые ответы модели печатаются в stderr (строки `WARN` / `ERROR`).

## Словарь (закрытый, у Айгерим поиск идёт по точным значениям)
Цвета: `red, orange, yellow, green, blue, purple, pink, brown, black, white, gray, beige, multicolor, unknown`.
Типы: `sweater, hoodie, t-shirt, shirt, jacket, coat, dress, pants_only, vest, uniform, other, unknown`.
Цвет и тип описывают ВЕРХНЮЮ одежду (самый внешний слой на торсе), не штаны, не кепку, не рюкзак.

### Пограничные цвета: что куда (правило и в промпте модели, и в коде как страховка; в логе строка `mapped`)
| Что видно | Пишем | Заметка |
|---|---|---|
| тёмно-синий, navy, светло-голубой, sky blue | `blue` | точный оттенок идёт в `description` ("dark navy puffer jacket") |
| слишком тёмный, не отличить navy от чёрного | `black` | |
| бордовый, burgundy, maroon | `red` | |
| хаки песочный, tan, sand, cream | `beige` | |
| оливковый, army green, тёмно-зелёный | `green` | |
| серебристый, тёмно-серый, charcoal | `gray` | |
Если нужна точность («тёмно-синяя» ≠ «чёрная») — словарь надо расширять вместе с Айгерим (`navy`, `maroon`, `olive`).
Значение вне словаря: цвет → `unknown`, тип → `other` (с `WARN`). Типы: sweatshirt/pullover/cardigan → `sweater`,
puffer/windbreaker/parka/blazer → `jacket`, overcoat/raincoat → `coat`, polo/blouse → `shirt`.
Человек, повёрнутый спиной: описывается по тому, что видно на спине (константа `BACK_TURNED_IS_UNKNOWN` в
`describe_snapshot.py` переключает на строгое `unknown`).

## Рисунки (`annotate_snapshots.py`)
Рамка + подпись `idx color type`: зелёный — известно, оранжевый — `unknown`, серый — рамка слишком мелкая, красный — кроп упал.
Сверху: срез, время, счёт детектора и VLM, статус. Исходные кадры не меняются. В режиме `frame` боксов нет:
список людей печатается текстом в углу.

## Загрузка в базу
**Порядок важен.** Файл Арай (`snapshots.json`) содержит `"people": []`, и загрузчик понимает это как «людей нет»: если
загрузить его ПОСЛЕ файла с одеждой, люди среза стираются (проверено: 159 -> 0). Грузить сначала файл Арай, потом `snapshots_vlm.json`
(или убрать `"people": []` из файла Арай / поправить загрузчик). Заглушка `make_snapshots_stub.py` ключ `people` не пишет.
`python -m database.snapshots_loader snapshots_vlm.json` (формат совместим; лишние ключи `box`, `vlm_*` загрузчик игнорирует).
