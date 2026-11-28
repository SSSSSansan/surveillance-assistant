#!/usr/bin/env bash
# Только чтение. Ничего не меняет и не запускает, кроме проверок. GPU_INDEX=1 по умолчанию.
GPU_INDEX="${GPU_INDEX:-1}"; WORK="${WORK:-$HOME/work}"; PORT="${PORT:-8765}"
echo "== GPU $GPU_INDEX (память в МиБ) =="
nvidia-smi -i "$GPU_INDEX" --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo; echo "== Диск (домашняя папка) =="; df -h ~
echo; echo "== Размер ~/work и кэшей =="
du -sh "$WORK" "$WORK/hf_cache" "$WORK/cache" 2>/dev/null || echo "пока нет"
ls -1 "$WORK/hf_cache/hub" 2>/dev/null | grep '^models--' || true
echo; echo "== Порт $PORT =="
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then echo "ЗАНЯТ"; else echo "свободен"; fi
echo; echo "== Мои процессы vLLM (только пользователь $(id -un)) =="
pgrep -u "$(id -un)" -af "vllm serve" || echo "нет"
