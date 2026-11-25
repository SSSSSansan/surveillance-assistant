#!/usr/bin/env bash
# google/gemma-3-4b-it  (имя для клиента: gemma-3-4b)
# Использование: bash serve_v4_gemma3_4b.sh [--print] [--gpu-memory-utilization 0.48]
#   --print  только показать команду и кэши, ничего не запускать.
# Перед запуском: bash check_gpu.sh. Одна модель за раз. Вес: около 9 ГБ в bf16 (НЕ ПРОВЕРЕНО)
# V4: репозиторий закрытый (gated): прими лицензию на сайте Hugging Face и войди САМА: huggingface-cli login
# (токен вводишь только ты, никому его не передавай; он сохранится в HF_HOME=~/work/hf_cache).
# Поддержка Gemma 3 в vllm 0.34.0 НЕ ПРОВЕРЕНО. Запасной путь по плану: transformers (скрипт пока не написан).
source "$(dirname "$0")/_serve_common.sh"
serve_main google/gemma-3-4b-it gemma-3-4b 8192 "$@"
