#!/usr/bin/env bash
# Только чтение. Ничего не меняет, ничего не запускает кроме проверок.
# Что смотрим: свободная память GPU, свободный диск, размер кэша моделей, свои процессы vLLM.
echo "== GPU (память в МиБ) =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo
echo "== Диск (домашняя папка) =="
df -h ~
echo
echo "== Кэш Hugging Face =="
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || echo "кэша пока нет"
ls -1 "${HF_HOME:-$HOME/.cache/huggingface}/hub" 2>/dev/null | grep '^models--' || true
echo
echo "== Мои процессы vLLM (только пользователь $(id -un)) =="
pgrep -u "$(id -un)" -af "vllm serve" || echo "нет"
