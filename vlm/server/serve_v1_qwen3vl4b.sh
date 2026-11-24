#!/usr/bin/env bash
# Qwen/Qwen3-VL-4B-Instruct  (имя для клиента: qwen3-vl-4b)
# Использование: bash serve_v1_qwen3vl4b.sh [--print] [--gpu-memory-utilization 0.40]
#   --print  только показать команду и кэши, ничего не запускать.
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Вес: около 9 ГБ в bf16 (НЕ ПРОВЕРЕНО)
# V1: baseline этапа 2. ВНИМАНИЕ: на этапе 2 стояло 0.45 (10,4 ГБ). При 0.40 (9,2 ГБ) веса bf16 почти занимают весь бюджет:
# если vLLM пишет про нехватку памяти под KV-кэш (НЕ ПРОВЕРЕНО), запусти с --gpu-memory-utilization 0.45 или MAX_LEN=4096.
source "$(dirname "$0")/_serve_common.sh"
serve_main Qwen/Qwen3-VL-4B-Instruct qwen3-vl-4b 8192 "$@"
