#!/usr/bin/env bash
# google/gemma-3-4b-it
# Использование:  bash serve_v4_gemma3_4b.sh [--print]     (--print только показывает команду, ничего не запускает)
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Модель весит: около 9 ГБ в bf16 (НЕ ПРОВЕРЕНО)
# V4: репозиторий закрытый (gated): нужно принять лицензию на сайте Hugging Face и войти САМОЙ: huggingface-cli login
# (токен вводишь только ты, не передавай его никому). Поддержку Gemma 3 в этой версии vLLM НЕ ПРОВЕРЕНО.
# Запасной путь по плану: transformers (отдельный скрипт, пока не написан).
source "$(dirname "$0")/_serve_common.sh"
MAX_LEN="${MAX_LEN_OVERRIDE:-8192}"
run_or_print "${1:-}" vllm serve google/gemma-3-4b-it --served-model-name gemma-3-4b \
  --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt image="$LIMIT_IMAGES"  $EXTRA_ARGS
