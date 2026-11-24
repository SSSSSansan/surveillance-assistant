#!/usr/bin/env bash
# cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit
# Использование:  bash serve_v2_qwen3vl8b_awq.sh [--print]     (--print только показывает команду, ничего не запускает)
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Модель весит: около 6-7 ГБ в AWQ 4-bit (НЕ ПРОВЕРЕНО)
# V2: кандидат на AWQ-чекпоинт, НЕ ПРОВЕРЕНО что он существует и грузится в этой версии vLLM.
# Квантизацию vLLM обычно определяет по config.json сам; если нет, добавь EXTRA_ARGS='--quantization awq' (или compressed-tensors).
source "$(dirname "$0")/_serve_common.sh"
MAX_LEN="${MAX_LEN_OVERRIDE:-8192}"
run_or_print "${1:-}" vllm serve cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit --served-model-name qwen3-vl-8b-awq \
  --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt image="$LIMIT_IMAGES"  $EXTRA_ARGS
