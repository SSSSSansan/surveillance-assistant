#!/usr/bin/env bash
# Qwen/Qwen3.5-4B  (имя для клиента: qwen3.5-4b)
# Использование: bash serve_v3_qwen35_4b.sh [--print] [--gpu-memory-utilization 0.40]
#   --print  только показать команду и кэши, ничего не запускать.
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Вес: около 9 ГБ в bf16: при 0.40 (9,2 ГБ) впритык (НЕ ПРОВЕРЕНО)
# V3: поддержка Qwen3.5 в vllm 0.34.0 НЕ ПРОВЕРЕНА. Не обновляй общий venv: если не стартует, запиши причину.
# Thinking выключаем в клиенте флагом --disable-thinking (chat_template_kwargs.enable_thinking=false, НЕ ПРОВЕРЕНО),
# либо на сервере EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}' (флаг есть не во всех версиях).
# Не хватает памяти: --gpu-memory-utilization 0.45, MAX_LEN=2048, либо квантованная версия (имя НЕ ПРОВЕРЕНО).
source "$(dirname "$0")/_serve_common.sh"
serve_main Qwen/Qwen3.5-4B qwen3.5-4b 4096 "$@"
