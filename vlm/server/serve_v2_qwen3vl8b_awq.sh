#!/usr/bin/env bash
# cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit  (имя для клиента: qwen3-vl-8b-awq)
# Использование: bash serve_v2_qwen3vl8b_awq.sh [--print] [--gpu-memory-utilization 0.48]
#   --print  только показать команду и кэши, ничего не запускать.
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Вес: около 6-7 ГБ в AWQ 4-bit (НЕ ПРОВЕРЕНО)
# V2: кандидат на AWQ-чекпоинт, НЕ ПРОВЕРЕНО что он существует и грузится в vllm 0.34.0.
# Квантизацию vLLM обычно определяет по config.json сам; если нет: EXTRA_ARGS='--quantization awq' (или compressed-tensors).
source "$(dirname "$0")/_serve_common.sh"
serve_main cpatonn/Qwen3-VL-8B-Instruct-AWQ-4bit qwen3-vl-8b-awq 8192 "$@"
