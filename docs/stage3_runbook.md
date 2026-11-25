# Этап 3: runbook для сервера (по одной команде)

> Статус: **НЕ ПРОВЕРЕНО на сервере.** Код проверен только на фейковом сервере (`pytest eval/test_stage3_vlm.py`).
> Любой флаг vLLM может не подойти к версии на сервере: читай ошибку, не угадывай.

## Правила общего сервера (важнее всего)
1. Сервер **общий и рабочий**: свободно около 11 ГБ памяти GPU, диск общий.
2. Модели **строго по одной**: V1 → V2 → V3 → V4. Две сразу не поднимаем.
3. Перед каждой загрузкой: `bash check_gpu.sh` (там `nvidia-smi` и `df -h ~`).
4. vLLM на порту **8765**, **GPU 1** (свободно ~11,6 из 23 ГБ), `--gpu-memory-utilization 0.48` по умолчанию (0.48 × 23 ≈ 11,0 ГБ при 11,38 ГиБ свободных). Не хватает памяти — сначала уменьши `MAX_LEN`; поднимать долю нельзя.

> **Измерено на сервере:** Qwen3-VL-4B в bf16 требует около **11,5 ГиБ на старт** (веса 8,44 ГиБ + ~3 ГиБ прогрев кодировщика картинок),
> а на общей карте свободно **11,38 ГиБ**. В bf16 модель не стартует, поэтому **бенчмарк идёт на 4-битных AWQ-весах для всех моделей**.
> Подмена репозитория и имени у любого serve-скрипта: `VLM_REPO=<репозиторий AWQ> VLM_SERVED=qwen3-vl-4b bash vlm/server/serve_v1_qwen3vl4b.sh`.
> `cleanup.sh` в этом случае вызывай с именем **подставленного** репозитория (по нему называются папки кэша).
> Значения по умолчанию, проверенные на сервере: `--enforce-eager`, `--max-num-seqs 8`, `--max-num-batched-tokens` = `max-model-len`, и `VLLM_USE_FLASHINFER_SAMPLER=0` (JIT-сборка семплера FlashInfer падает: ninja + системный nvcc 12.0 против CUDA 13 в venv).
> **Папка вне `~/work`:** FlashInfer создаёт `~/.cache/flashinfer`. `cleanup.sh` (и с именем модели, и с `--all`) показывает её размер и удаляет вместе с остальным после подтверждения. Папка общая для моделей и при необходимости соберётся заново.
5. После замера: останови **свой** vLLM и удали веса и кэши **только этой** модели (`cleanup.sh` делает ровно это и спрашивает `yes`).
6. Чужие процессы и чужие файлы не трогать. Удаляем только то, что скачали сами.
7. Токены (Hugging Face) вводишь только ты сама, никому и никуда их не отправляй.

## 0. Что скопировать на сервер (репо не клонируем; бенчмарки считаем на Mac)
На сервер идут ТОЛЬКО запросы к модели. Нужны файлы из `vlm/` (из `eval/` не нужно ничего), в `~/work/vlm`:
`prompts.py describe_snapshot.py imgutil.py run_predictions.py check_fewshot.py fewshot.json`,
`describers/{__init__,base,vlm_api}.py`, `server/*.sh`. Данные: `~/work/data/snapshots.json`, `~/work/data/frames/`.
Пакеты: stdlib + opencv-python-headless + pillow (проверено на изолированной копии, `python -I`; на самом сервере НЕ ПРОВЕРЕНО).
```bash
ssh <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР> "mkdir -p ~/work/vlm/describers ~/work/vlm/server ~/work/runs"
```
```bash
cd vlm && scp prompts.py describe_snapshot.py imgutil.py run_predictions.py check_fewshot.py fewshot.json <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/vlm/
```
```bash
scp describers/__init__.py describers/base.py describers/vlm_api.py <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/vlm/describers/
```
```bash
scp server/*.sh <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/vlm/server/
```
Все остальные команды на сервере: `source ~/work/venv/bin/activate && cd ~/work`.

## Цикл для каждой модели (V1 → V2 → V3 → V4)
Все serve-скрипты сами ставят GPU 1, кладут кэши в `~/work` (`HF_HOME=~/work/hf_cache`, `XDG_CACHE_HOME=~/work/cache`,
подпапки `triton/`, `vllm/`, `inductor/` на каждую модель), проверяют, что порт 8765 свободен, и берут
`--gpu-memory-utilization 0.48` (меняется параметром), `--enforce-eager`, `--max-num-seqs 8`.

**1. Проверка места.**
```bash
bash vlm/server/check_gpu.sh
```
**2. Показать команду, ничего не запуская.**
```bash
bash vlm/server/serve_v1_qwen3vl4b.sh --print
```
**3. Поднять в фоне** (для других моделей имя файла другое; память можно поменять: `--gpu-memory-utilization 0.45`; AWQ-веса: `VLM_REPO=… VLM_SERVED=…` перед командой).
```bash
nohup bash vlm/server/serve_v1_qwen3vl4b.sh > ~/work/runs/serve_v1.log 2>&1 &
```
**4. Дождаться готовности и записать точное имя и квантизацию.**
```bash
tail -n 30 ~/work/runs/serve_v1.log
```
```bash
curl -s http://127.0.0.1:8765/v1/models | python3 -m json.tool
```
```bash
grep -i -E "quantization|dtype|model=" ~/work/runs/serve_v1.log | head
```
**5. Во втором окне: замер памяти** (GPU 1; в конце Ctrl-C).
```bash
bash vlm/server/measure.sh ~/work/runs/vram_v1.csv 2
```
**6. Одежда: R1 × P-A по всем кропам из snapshots.json** (одна команда; `--model` = имя из serve-скрипта: `qwen3-vl-4b`, `qwen3-vl-8b-awq`, `qwen3.5-4b`, `gemma-3-4b`; для V3 добавь `--disable-thinking`).
```bash
python3 vlm/run_predictions.py crops --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames --model qwen3-vl-4b --variant A --out ~/work/runs/pred_clothing_qwen3-vl-4b_R1_A.jsonl
```
**7. Счёт: R2 по всем кадрам.**
```bash
python3 vlm/run_predictions.py count --mode R2 --snapshots ~/work/data/snapshots.json --frames-dir ~/work/data/frames --model qwen3-vl-4b --out ~/work/runs/count_qwen3-vl-4b_R2.jsonl
```
**8. Стоп замера (Ctrl-C), запиши два пика. Убрать за собой** (остановит мой vLLM, удалит кэши только этой модели, спросит `yes`).
```bash
bash vlm/server/cleanup.sh Qwen/Qwen3-VL-4B-Instruct
```
Репозитории: `Qwen/Qwen3-VL-4B-Instruct`, `cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit`, `Qwen/Qwen3.5-4B`, `google/gemma-3-4b-it`.
**9. Проверить, что память освободилась, и брать следующую модель.**
```bash
bash vlm/server/check_gpu.sh
```

## Подсчёт метрик на Mac (после `scp` результатов)
```bash
scp -r <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/runs ./eval/results_from_server
```
```bash
python3 -m eval.bench_clothing --gold eval/gold/gold_people.csv --predictions eval/results_from_server/pred_clothing_qwen3-vl-4b_R1_A.jsonl --method-name vlm:qwen3-vl-4b
```
```bash
python3 -m eval.bench_count --gold eval/gold/gold_frames.csv --snapshots snapshots.json --counts R2=eval/results_from_server/count_qwen3-vl-4b_R2.jsonl
```
`runs.jsonl` с сервера (`~/work/runs/runs.jsonl`) и есть запись о прогонах; локальный подсчёт новых строк в него не добавляет.

### Сводка по прогонам без эталона (`eval/summarize_runs.py`)
Быстрый обзор всех файлов из папки: сколько ответов пришло, сколько обрезано, сколько секунд, насколько стабильно. Точность здесь не считается, её дают `bench_*` с эталоном.
```bash
python3 -m eval.summarize_runs --dir server_runs
```
- Печатает две таблицы markdown (R1 и счёт R2/R3/R4) и сохраняет CSV в `server_runs/summary_runs.csv` (другой путь задаётся через `--out`).
- Одна строка на модель × режим × вариант × метку. Метка берётся из хвоста имени файла, например `count_qwen3.5-4b-awq_R3_rep1.1.jsonl` даёт метку `rep1.1`. Файл `*_repeat2` своей строки не даёт, по нему считается повторяемость основного файла.
- Доли ok, truncated и «ошибка» в сумме дают 1. Секунды усредняются по каждому запросу (среднее и p95). Запросы шли параллельно (`--workers`) на общей карте.
- Для R1 считаются доля `black`, доля `unknown` и повторяемость: доля совпавших цвета и типа между проходами, только по людям, которые были ok в обоих.
- Для R2/R3/R4 считаются среднее, СКО, минимум и максимум `people_count` по ok-строкам и число `suspicious`.
- Старые файлы (до появления `raw_response`, `finish_reason` и `suspicious`) читаются без ошибок и помечены `old`. В них `suspicious` пустой, то есть неизвестен (это не 0). Обрезанные ответы в них записаны как `error`, поэтому truncated = 0 там ничего не значит.

## Прогоны по кропам Арай (два видео, режим по индексу)
С этого места VLM получает **готовые кропы Арай**, а не режет кадры сама. Сравнение с её методами (M1–M3) идёт на одних и тех же людях.
Схема индекса проверена по `arai_data/video*/video*_crops_index.json` (это выход `cv/crop_people.py`): список записей `crop_id, video_id, snapshot_id, person_idx, box, box_h, box_w, h_bucket, frame_path, kept, reason, path, full_path`.
- Шлём только `kept=true`. Отсеянные записи (`reason`: `dup_of_pNN`, `too_small`, `bad_box`) не отправляются.
- Кроп — готовый файл `<папка индекса>/<crop_id>.jpg` (так лежит у Арай) или `<папка>/full/<crop_id>.jpg`. Заново ничего не режется; мелкий кроп только увеличивается до 336 px, как раньше.
- `snapshot_id` и `crop_id` повторяются в обоих видео (`sn_0000_p00` есть в обоих). Поэтому ключ везде `(video_id, snapshot_id, person_idx)`, а папки задаются **по одной на индекс, в том же порядке**. Одна папка на два индекса отклоняется с ошибкой.
- video_id: видео 1 — `2018-03-15.15-00-00.15-05-00.school.G421.r13` (156 кропов из 159 рамок), видео 2 — `2018-03-07.17-35-00.17-40-00.school.G421.r13` (147 из 152). Оба проверены по индексам.

**Что скопировать на сервер** (дополнительно к разделу 0; `arai_data/` не в git):
```bash
ssh <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР> "mkdir -p ~/work/data/arai"
```
```bash
scp -r arai_data/video1 arai_data/video2 <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/data/arai/
```
Кадры видео 1 уже лежат в `~/work/data/frames`. Кадров видео 2 пока нет. Когда появятся, положи их в `~/work/data/frames_video2/` (имена `sn_XXXX.jpg`).

**Переменные** (одни на все модели; меняется только `M`, у V3 добавь `--disable-thinking` к каждой команде):
```bash
M=qwen3-vl-4b-awq; A=$HOME/work/data/arai; CI="--crops-index $A/video1/video1_crops_index.json $A/video2/video2_crops_index.json"; FI="--frames-index $A/video1/video1_frames_index.json $A/video2/video2_frames_index.json --frames-dir $HOME/work/data/frames $HOME/work/data/frames_video2"
```
Имена моделей на сервере: `qwen3-vl-4b-awq` (V1), `qwen3-vl-8b-awq` (V2), `qwen3.5-4b-awq` (V3), `gemma3-4b-w4a16` (V4).

**Одежда R1 по обоим видео** (вариант A, второй проход для повторяемости):
```bash
python3 vlm/run_predictions.py crops $CI --model $M --variant A --repeat 2 --workers 4 --out ~/work/runs/arai/pred_clothing_${M}_R1_A.jsonl
```
Пишутся два файла:
- наш `~/work/runs/arai/pred_clothing_<M>_R1_A.jsonl`;
- плоский файл Арай `~/work/runs/arai/flat/pred_clothing_<M>_R1_A.jsonl` (плюс `_repeat2` к обоим).

Варианты B и D: те же команды с `--variant B` / `--variant D` и своим именем `--out`.

**Счёт R2 / R3 / R4.** Видео без папки кадров пропускается с сообщением `ПРОПУСК <video_id> …`. Сейчас это видео 2, его строк в файле не будет. Если кадров нет ни у одного видео, команда остановится.
```bash
python3 vlm/run_predictions.py count --mode R2 $FI --model $M --workers 2 --out ~/work/runs/arai/count_${M}_R2.jsonl
```
```bash
python3 vlm/run_predictions.py count --mode R3 $FI --model $M --workers 2 --out ~/work/runs/arai/count_${M}_R3.jsonl
```
```bash
python3 vlm/run_predictions.py count --mode R4 $FI $CI --model $M --workers 2 --out ~/work/runs/arai/count_${M}_R4.jsonl
```
- R4 кладёт в промпт только рамки `kept=true` (без дублей). В строке поле `box_person_idx` показывает, какие `person_idx` стоят за номерами 0, 1, 2… в промпте.
- `suspicious` считается по `kept`-рамкам (в R2/R3 без `--crops-index` берётся `n_kept` из `frames_index.json`).
- `runs.jsonl` получает по строке на каждое видео.
- Повторный прогон видео 2, когда появятся кадры: те же команды, другой `--out`.

**Подсчёт на Mac:**
```bash
scp -r <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/work/runs/arai ./server_runs_arai
```
```bash
python3 -m eval.summarize_runs --dir server_runs_arai
```
```bash
python3 -m eval.score_methods people --gold eval/gold/gold_people.csv --pred "server_runs_arai/flat/pred_clothing_*_R1_?.jsonl" results/hist__crop__A.jsonl results/siglip__crop__A.jsonl --out-md results/clothing_all.md
```
- `summarize_runs` даёт отдельную строку на каждое видео.
- В плоском файле строка не ok (truncated / ошибка / кроп не найден) записана как `color = type = "unknown"`. Поэтому `score_methods` засчитывает её **промахом**, а не пропускает, и она попадает в долю unknown.
- `bench_count` читает файл счёта только по одному `video_id` (он берётся из `--snapshots`). Без `video_id` файл с двумя видео отклоняется.

### Чем считать итоговую таблицу (сравнение `bench_clothing` / `bench_count` / `score_methods`)
**Рекомендация.** Одежду всех методов (M1–M3 и VLM) считать одним скриптом: `eval.score_methods people` по плоским файлам. Тогда у всех одна агрегация эталона и одна формула. VLM-строки сверять с `eval.bench_clothing --predictions <наш jsonl>`: только он даёт повторяемость, тип по бакетам высоты и матрицу ошибок по типу. Счёт людей:
- VLM (R2–R4) против эталона — `eval.bench_count`, по одному видео за запуск;
- сырой YOLO против YOLO без дублей — `eval.score_methods count`.

Перед итоговой таблицей правило по нескольким разметчикам должно быть одно в обоих скриптах (см. п. 2). Ничего не удалено, оба набора скриптов работают.

Дублируется: accuracy, macro-F1, top-2, доля unknown / black, точность по бакетам высоты, матрица ошибок, MAE / точные / недосчёт / пересчёт / смещение.
Есть только у нас: повторяемость (`--repeat 2`), kappa (`eval/agreement.py`, между разметчиками), доля неответов (`n_errors` / `n_failed`), тип по бакетам, исключение few-shot примеров, нет зависимости от sklearn.
Есть только у Арай: сравнение «сырой YOLO против YOLO без дублей», готовое чтение `snapshots_vlm.json`, чтение эталона по маске (несколько CSV).

Расхождения в определениях (числа двух скриптов на одних данных могут отличаться):
1. **Нет предсказания.** `bench_*` считает такого человека ошибкой (`n_errors`). `score_methods` его молча пропускает. Поэтому в плоском формате не-ok строки пишутся как unknown.
2. **Несколько разметчиков.** Одежда: оба берут большинство, при ничьей объект пропускается (в `score_methods` целиком при ничьей по цвету, в `bench_clothing` отдельно по цвету и типу). Счёт: `bench_count` берёт большинство, а кадр с ничьей выкидывает; `score_methods count` берёт нижнюю медиану.
3. **macro-F1.** У нас классы = эталон ∪ предсказания, у Арай только классы эталона (`labels=sorted(set(y))`). Лишний предсказанный класс у Арай не штрафуется, у нас штрафуется.
4. **Доли unknown / black.** У нас считаются среди ответивших, у Арай среди всех пар (с нашими unknown-заглушками туда входят и неответы).
5. **Бакеты высоты.** У нас берётся `box_h` из эталона, и рамка ровно 150 px идёт в «80-150». У Арай `box_h` берётся из предсказания и 150 px идёт в «>150» (как `h_bucket` в индексе).
6. **Тип.** `score_methods` не считает тип, если метод выдаёт только unknown (M1). `bench_clothing` считает всегда.
7. **Счёт.** У нас точные / недосчёт / пересчёт — доли, неответы исключаются и показываются отдельно (`n_failed`). У Арай недосчёт / пересчёт — штуки, неответов нет (YOLO всегда отвечает).
8. **Секунды.** У нас время одного запроса при параллельных запросах (`--workers`). У Арай время пакета, делённое на число кропов. Скорость VLM и CV по этим числам напрямую сравнивать нельзя.
9. **Невидимые люди.** `score_methods` по умолчанию выкидывает `visible` = «не видно» / `not_visible`. `bench_clothing` делает это только с `--exclude-visible`.

## После четырёх моделей: победитель (поднимаем только его, по одной команде)
- Режимы R3 и R4 (счёт + одежда по кадру): `python3 vlm/run_predictions.py count --mode R3` и `--mode R4` на сервере, затем `--counts R3=... --counts R4=...` на Mac.
- Промпты B и D: `python3 vlm/run_predictions.py crops ... --variant B`, затем `--variant D`.
- Промпт C: сначала заполни `vlm/fewshot.json` тремя размеченными кропами **не из тестовой выборки**, перезапусти сервер с `LIMIT_IMAGES=4` (3 примера + кроп; НЕ ПРОВЕРЕНО, что сервер примет столько картинок), затем `--variant C`. Проверка примеров: `python3 vlm/check_fewshot.py`; на Mac добавь `--exclude-fewshot vlm/fewshot.json`, чтобы примеры не попали в оценку.
- Повторяемость: `--repeat 2` в `run_predictions.py crops` (второй проход в `*_repeat2.jsonl`, на Mac `--predictions-repeat`) у лучшей связки (считает долю расхождений с самой собой).

## В самом конце проекта: убрать всё
Сначала забери `~/work/runs` к себе. Затем одна команда (покажет размеры и потребует слово `delete-work`):
```bash
bash ~/work/vlm/server/cleanup.sh --all
```
