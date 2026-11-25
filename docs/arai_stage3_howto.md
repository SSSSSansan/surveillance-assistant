# Задачи Арай, этап 3: как запускать (под репозиторий surveillance-assistant)

Код написан поверх того, что уже есть в репозитории (коммит `191cc5e`): ваш `cv/sample_snapshots.py`,
`cv/meva_fetch.py`, `cv/tracks_io.py`, формат `snapshots.json`, словарь и кропы Панды, схема `snapshots` /
`snapshot_people`. Ничего существующего не заменено.

## Что добавлено

| Файл | Задача | Что делает |
|---|---|---|
| `cv/rank_meva.py` | T20, шаг 1 | качает 5 кандидатов (камеры G421/G420), гонит ваш `sample_snapshots.py`, выбирает клип с наибольшим числом людей |
| `cv/crop_people.py` | T14, T20 | `snapshots.json` -> кропы торса и целиком, удаление дублей рамок, индексы кропов и счёта |
| `cv/color_hist.py` | T17 | M1: цвет по пикселям (баланс белого, без кожи, k-means, правила HSV) |
| `vlm/describers/siglip.py` | T24 | M2: SigLIP zero-shot |
| `vlm/describers/par_model.py` | T24 | M3: PAR на Market-1501 (обучение и инференс) |
| `vlm/describers/base.py` | п. 4.1 плана | контракт ответа, словарь = словарь Панды |
| `eval/run_method.py` | T17, T24 | прогон метода по кропам -> `results/*.jsonl` |
| `eval/score_methods.py` | T33 | метрики по одежде и счёт людей; принимает и `snapshots_vlm.json` как строку «VLM» |
| `eval/export_snapshots_json.py` | T34 | результат метода -> формат `snapshots_vlm.json` для вашего `snapshots_loader` |
| `cv/track_vote.py` | T32 | голосование по `track_id` из `tracks.json` этапа 1 |
| `eval/test_cv_methods.py` | | 29 проверок, в т.ч. на реальных 159 рамках этапа 2 |

Две строки в существующих файлах (патч `tests_hook.patch`): `eval/run_tests.py` (добавить
`"eval.test_cv_methods"` в `STEPS`) и `eval/conftest.py` (добавить `"test_cv_methods.py"` в `collect_ignore`).

## Что я выяснил про ваши данные (проверено на `vlm/snapshots_vlm.json`)

- **Люди крупные:** из 159 рамок 121 выше 150 px, 36 в диапазоне 80-150, меньше 80 px только 2 (медиана 263 px).
  Бакеты высоты из плана (`<80`, `80-150`, `>150`) почти пустые в первом; для отчёта это значит, что вывода
  «хуже на мелких людях» на этом видео сделать нельзя. Второе видео должно быть по возможности тем же масштаба.
- **Дубли рамок:** их ровно 3 (`sn_0000 #4`, `sn_0007 #5`, `sn_0009 #17`), у обеих рамок пары одинаковая одежда.
  Находятся правилом IoU > 0.6. Правило «рамка внутри рамки» я выключил: оно удаляло живых людей (`sn_0003`:
  серая футболка и бежевая рубашка лежат внутри одной крупной рамки).
- **Следствие для эталона:** в базе у вас 159 людей, а реальных людей, вероятно, 156. Эталон (T21) разметчики должны
  размечать по `crops_index.json` (там 156 кропов с `kept=true`), иначе дубли попадут в метрики.
- **Словарь:** 13 цветов + `unknown` (нет `light_blue`, `navy`, `maroon`), 11 типов + `unknown`. Тест следит, чтобы
  `vlm/describers/base.py` не разошёлся с `vlm/describe_snapshot.py`.

## Порядок запуска (из корня репозитория; на Windows `py -m` вместо `python -m`)

```bash
pip install -r cv/requirements.txt scikit-learn scipy        # ultralytics, opencv, numpy уже есть
python -m eval.test_cv_methods                               # 29 OK, ничего не требует
```

### 1. T20, шаг 1: выбрать второе видео
```bash
python cv/rank_meva.py --list
python cv/rank_meva.py --out data/candidates --gpu 1         # --gpu: номер карты на общем сервере
```
Скачает до ~0.6 ГБ (проверь место), напечатает таблицу и `ВЫБОР: <ключ>`. Моя ставка до запуска -
`g421_0307_1720` (та же камера G421, другой день), но решают цифры. Если `boto3` не установлен, скрипт
качает по обычной ссылке. Видео этапа 2 (2018-03-15 15:00) в кандидатах нет.

### 2. T14 + T20: кропы (для обоих видео)
Видео 1 (этап 2; нужны `snapshots.json` и папка `frames/` с сервера Панды, в git кадров нет):
```bash
python -m cv.crop_people --snapshots <папка>/snapshots_vlm.json --frames-root <папка> --out data/video1/crops
```
Видео 2 (после шага 1; `snapshots.json` уже создал `rank_meva.py` в `data/candidates/<ключ>/`):
```bash
python -m cv.crop_people --snapshots data/candidates/<ключ>/snapshots.json --out data/video2/crops
```
В конце печатается число рамок и кропов, причины отсева и распределение по высоте. Для проверки без кадров:
`--no-images`. Для T20 нужно ~150 кропов на второе видео.

### 3. T17 и T24: методы
```bash
python -m eval.run_method --method hist   --index data/video1/crops/crops_index.json data/video2/crops/crops_index.json
python -m eval.run_method --method hist   --index data/video1/crops/crops_index.json data/video2/crops/crops_index.json --wb-alpha 0 --variant B
pip install transformers pillow sentencepiece protobuf
python -m eval.run_method --method siglip --index data/video1/crops/crops_index.json data/video2/crops/crops_index.json --device cuda:1
```
Второй запуск (`--variant B`, без баланса белого) нужен для честной ablation. SigLIP маленький, но сервер общий:
предупреди Панду (в плане GPU-прогоны запускает он). M3 (режется первым):
```bash
python -m vlm.describers.par_model train --market-root <путь>/Market-1501-v15.09.15 --attr-mat <путь>/market_attribute.mat --out models/par_market.pt --epochs 15 --device cuda:1
python -m eval.run_method --method par --index data/video1/crops/crops_index.json data/video2/crops/crops_index.json --par-ckpt models/par_market.pt
```

### 4. T33 и сравнение (когда готов эталон T21)
Формат CSV описан в шапке `eval/score_methods.py` (`snapshot_id` строкой `sn_0003`; колонки `clothing_color` /
`clothing_type` тоже принимаются).
```bash
python -m eval.score_methods count  --gold gold_frames.csv --frames-index data/video1/crops/frames_index.json data/video2/crops/frames_index.json
python -m eval.score_methods people --gold gold_people.csv --pred results/hist__crop__A.jsonl results/siglip__crop__A.jsonl vlm/snapshots_vlm.json --out-md results/clothing.md --out-dir results/confusion
```
`vlm/snapshots_vlm.json` во втором списке даёт в таблице строку «VLM (Qwen3-VL 4B)» для сравнения с M1/M2/M3.
Счёт сравнивается в двух вариантах: сырой `people_count` (как сейчас в базе) и после удаления дублей.

### 5. T34: победитель в базу
```bash
python -m eval.export_snapshots_json --base vlm/snapshots_vlm.json --pred results/<победитель>.jsonl --out snapshots_best.json
python -m database.snapshots_loader snapshots_best.json
```
Загрузчик заменяет людей среза целиком, а колонки `method` в базе нет (T15 не сделана), поэтому файл
ПЕРЕЗАПИШЕТ описания Панды. Делай это только когда победитель выбран; старый файл остаётся в git.
`description` берётся у VLM и может противоречить новому цвету; ключ `--fix-description` заменяет его на
«цвет тип».

### 6. T32 (после T33 и T34)
```bash
python -m cv.track_vote --tracks data/out/tracks.json --index data/video1/crops/crops_index.json --pred results/hist__crop__A.jsonl --out results/hist+vote__crop__vote.jsonl
```
`tracks.json` берётся от этапа 1 (то же видео `2018-03-15…G421`). Скрипт печатает, сколько людей встретились на
двух и более срезах и у скольких цвет менялся.

## Что проверено, а что нет

| Проверено (тесты зелёные, 29 + весь ваш набор) | НЕ проверено (в моей песочнице нет доступа) |
|---|---|
| словарь = файл Панды; дубли и порядок `person_idx` на реальных 159 рамках; M1 на цветовых образцах; кроппер, `run_method`, метрики, T33, экспорт (проходит ваш `_validate`), логика голосования | скачивание с AWS S3, `rank_meva.py`, YOLO на видео, SigLIP (HuggingFace), обучение и инференс PAR (формат `market_attribute.mat` помню по памяти), эффект M1 на реальных кропах |

Ограничения методов для отчёта: у M3 только 8 цветов верха, нет `orange`, `pink`, `brown`, `beige`, `multicolor` и нет
типа одежды. M1 тип не определяет. Эффект T32 на 11 срезах раз в 30 секунд будет небольшим.
