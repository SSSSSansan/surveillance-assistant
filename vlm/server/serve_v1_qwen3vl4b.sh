#!/usr/bin/env bash
# Qwen/Qwen3-VL-4B-Instruct  (имя для клиента: qwen3-vl-4b)
# Использование: bash serve_v1_qwen3vl4b.sh [--print] [--gpu-memory-utilization 0.48]
#   --print  только показать команду и кэши, ничего не запускать.
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Вес: около 9 ГБ в bf16 (НЕ ПРОВЕРЕНО)
# V1: baseline этапа 2. ИЗМЕРЕНО на сервере: Qwen3-VL-4B в bf16 требует ~11,5 ГиБ на старт (веса 8,44 ГиБ + ~3 ГиБ прогрев
# кодировщика картинок), а на общей карте свободно 11,38 ГиБ: в bf16 НЕ ВЛЕЗАЕТ. Бенчмарк идёт на 4-битных AWQ-весах:
# VLM_REPO=<репозиторий AWQ-весов> VLM_SERVED=qwen3-vl-4b bash serve_v1_qwen3vl4b.sh   (имя AWQ-репозитория НЕ ПРОВЕРЕНО здесь)
source "$(dirname "$0")/_serve_common.sh"
serve_main Qwen/Qwen3-VL-4B-Instruct qwen3-vl-4b 8192 "$@"
