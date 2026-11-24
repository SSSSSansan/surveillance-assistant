#!/usr/bin/env bash
# Qwen/Qwen3-VL-4B-Instruct
# Использование:  bash serve_v1_qwen3vl4b.sh [--print]     (--print только показывает команду, ничего не запускает)
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Модель весит: около 9 ГБ в bf16 (НЕ ПРОВЕРЕНО)
# V1: baseline этапа 2.
source "$(dirname "$0")/_serve_common.sh"
MAX_LEN="${MAX_LEN_OVERRIDE:-8192}"
run_or_print "${1:-}" vllm serve Qwen/Qwen3-VL-4B-Instruct --served-model-name qwen3-vl-4b \
  --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt image="$LIMIT_IMAGES"  $EXTRA_ARGS
