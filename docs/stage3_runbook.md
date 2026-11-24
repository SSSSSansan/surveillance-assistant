# Этап 3: runbook для сервера (по одной команде)

> Статус: **НЕ ПРОВЕРЕНО на сервере.** Код проверен только на фейковом сервере (`pytest eval/test_stage3_vlm.py`).
> Любой флаг vLLM может не подойти к версии на сервере: читай ошибку, не угадывай.

## Правила общего сервера (важнее всего)
1. Сервер **общий и рабочий**: свободно около 11 ГБ памяти GPU, диск общий.
2. Модели **строго по одной**: V1 → V2 → V3 → V4. Две сразу не поднимаем.
3. Перед каждой загрузкой: `bash check_gpu.sh` (там `nvidia-smi` и `df -h ~`).
4. vLLM на порту **8765**, `--gpu-memory-utilization 0.45`. Не хватает памяти — **уменьшай**, не увеличивай.
5. После замера: останови **свой** vLLM и удали веса **только этой** модели (`cleanup.sh` делает ровно это и спрашивает `yes`).
6. Чужие процессы и чужие файлы не трогать. Удаляем только то, что скачали сами.
7. Токены (Hugging Face) вводишь только ты сама, никому и никуда их не отправляй.

## 0. Что скопировать на сервер (репо на сервер НЕ клонируем)
Нужно: папка `vlm/` целиком (внутри `describers/`, `server/`, `prompts.py`, `fewshot.json`), из `eval/` только
`__init__.py metrics.py bench_common.py bench_clothing.py bench_count.py agreement.py`, папка `eval/gold/` с настоящим эталоном,
`snapshots.json` и кадры `frames/`. Пакеты: только стандартная библиотека + OpenCV **или** Pillow (они уже нужны с этапа 2).
Версию Python на сервере НЕ ПРОВЕРЯЛИ (код рассчитан на 3.8+).
```bash
scp -r vlm <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/snapshots/
```
```bash
ssh <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР> "mkdir -p ~/snapshots/eval/gold ~/snapshots/eval/results"
```
```bash
scp eval/__init__.py eval/metrics.py eval/bench_common.py eval/bench_clothing.py eval/bench_count.py eval/agreement.py <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/snapshots/eval/
```
Дальше все команды выполняются на сервере из `~/snapshots` с включённым venv.

## Цикл для каждой модели (V1 → V2 → V3 → V4)

**1. Проверить, что есть место.** Покажет свободную память GPU, диск и кэш. Если свободно меньше нужного — стоп, пиши Сании.
```bash
bash vlm/server/check_gpu.sh
```
**2. Посмотреть команду запуска, ничего не запуская.** Подставь нужный файл: `serve_v1_qwen3vl4b.sh`, `serve_v2_qwen3vl8b_awq.sh`, `serve_v3_qwen35_4b.sh`, `serve_v4_gemma3_4b.sh`.
```bash
bash vlm/server/serve_v1_qwen3vl4b.sh --print
```
**3. Поднять модель в фоне** (лог пишем в файл; первая загрузка скачивает веса).
```bash
nohup bash vlm/server/serve_v1_qwen3vl4b.sh > serve_v1.log 2>&1 &
```
**4. Дождаться готовности** (в логе появится строка про Uvicorn / "Application startup complete"). Ошибку запиши в отчёт как причину провала.
```bash
tail -n 30 serve_v1.log
```
**5. Записать точное имя модели и квантизацию.** `/v1/models` покажет имя и длину контекста; про квантизацию ищи слово `quantization` в логе (НЕ ПРОВЕРЕНО, что оно там есть).
```bash
curl -s http://127.0.0.1:8765/v1/models | python3 -m json.tool
```
```bash
grep -i -E "quantization|dtype|model=" serve_v1.log | head
```
**6. Во втором окне запустить замер памяти** (пока идёт бенчмарк; в конце нажми Ctrl-C, он напечатает пик).
```bash
bash vlm/server/measure.sh vram_v1.csv 2
```
**7. Проверка на 5 кропах** (дёшево; убедись, что ответы разбираются).
```bash
python3 -m eval.bench_clothing --gold eval/gold/gold_people.csv --snapshots snapshots.json --frames-dir . --model qwen3-vl-4b --variant A --limit 5
```
**8. Полный прогон по одежде (R1 × P-A).** Для V3 добавь `--disable-thinking`. Имя `--model` = `--served-model-name` из serve-скрипта (`qwen3-vl-4b`, `qwen3-vl-8b-awq`, `qwen3.5-4b`, `gemma-3-4b`).
```bash
python3 -m eval.bench_clothing --gold eval/gold/gold_people.csv --snapshots snapshots.json --frames-dir . --model qwen3-vl-4b --variant A --workers 2
```
**9. Прогон по счёту людей (R2).** `yolo` берётся из `snapshots.json` и считается один раз; для остальных моделей оставляй только `R2`.
```bash
python3 -m eval.bench_count --gold eval/gold/gold_frames.csv --snapshots snapshots.json --frames-dir . --model qwen3-vl-4b --methods yolo,R2
```
**10. Остановить замер** (Ctrl-C во втором окне), записать два числа: пик «моей» и пик общей памяти. Файл `vram_*.csv` оставить.

**11. Убрать за собой.** Покажет, что остановит и удалит, и спросит `yes`. Имя репозитория — то, что скачивали.
```bash
bash vlm/server/cleanup.sh Qwen/Qwen3-VL-4B-Instruct
```
Репозитории: `Qwen/Qwen3-VL-4B-Instruct`, `cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit`, `Qwen/Qwen3.5-4B`, `google/gemma-3-4b-it`.
**12. Проверить, что место освободилось, и только тогда брать следующую модель.**
```bash
bash vlm/server/check_gpu.sh
```

## После четырёх моделей: победитель (поднимаем только его, по одной команде)
- Режимы R3 и R4 (счёт + одежда по кадру): `python3 -m eval.bench_count ... --methods R3,R4`.
- Промпты B и D: `python3 -m eval.bench_clothing ... --variant B`, затем `--variant D`.
- Промпт C: сначала заполни `vlm/fewshot.json` тремя размеченными кропами **не из тестовой выборки**, перезапусти сервер с `LIMIT_IMAGES=4` (3 примера + кроп; НЕ ПРОВЕРЕНО, что сервер примет столько картинок), затем `--variant C`.
- Повторяемость: `--repeat 2` у лучшей связки (считает долю расхождений с самой собой).

## Что смотреть после
Все результаты лежат в `eval/results/`: `clothing_summary.csv`, `count_summary.csv`, отчёты `.md`, `runs.jsonl` (материал для таблицы `runs` Айгерим).
Забрать на свой компьютер:
```bash
scp -r <ПОЛЬЗОВАТЕЛЬ>@<СЕРВЕР>:~/snapshots/eval/results ./eval/results_from_server
```
