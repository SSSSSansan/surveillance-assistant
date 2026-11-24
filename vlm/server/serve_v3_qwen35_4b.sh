#!/usr/bin/env bash
# Qwen/Qwen3.5-4B
# Использование:  bash serve_v3_qwen35_4b.sh [--print]     (--print только показывает команду, ничего не запускает)
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Модель весит: около 9 ГБ в bf16: впритык к ~11 ГБ свободных (НЕ ПРОВЕРЕНО)
# V3: может требовать БОЛЕЕ НОВЫЙ vLLM, чем стоит на сервере (НЕ ПРОВЕРЕНО). Не обновляй общий venv: если не стартует, запиши причину.
# Thinking лучше выключить: в клиенте флаг --disable-thinking (chat_template_kwargs.enable_thinking=false, НЕ ПРОВЕРЕНО),
# либо на сервере EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}' (флаг есть не во всех версиях).
# Если памяти не хватает: MAX_LEN_OVERRIDE=2048 или квантованная версия (имя НЕ ПРОВЕРЕНО).
source "$(dirname "$0")/_serve_common.sh"
MAX_LEN="${MAX_LEN_OVERRIDE:-4096}"
run_or_print "${1:-}" vllm serve Qwen/Qwen3.5-4B --served-model-name qwen3.5-4b \
  --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt image="$LIMIT_IMAGES"  $EXTRA_ARGS
