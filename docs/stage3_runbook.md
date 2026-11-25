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
