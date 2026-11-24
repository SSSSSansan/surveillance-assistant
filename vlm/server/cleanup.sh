#!/usr/bin/env bash
# Останавливает ТОЛЬКО мой vLLM и удаляет кэш ТОЛЬКО указанной модели. Сначала показывает, что сделает, и просит "yes".
# Использование: bash cleanup.sh Qwen/Qwen3-VL-4B-Instruct
# Чужие процессы (другой пользователь) не трогаются: pkill с -u моего пользователя. Чужие файлы не трогаются:
# удаляется ровно одна папка models--ORG--NAME внутри моего кэша Hugging Face.
set -u
REPO="${1:-}"
if ! [[ "$REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "Укажи модель как ORG/NAME, например Qwen/Qwen3-VL-4B-Instruct"; exit 2
fi
ME="$(id -un)"
HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
TARGET="$HUB/models--${REPO//\//--}"
echo "Пользователь: $ME"
echo "Остановлю процессы (только мои):"
pgrep -u "$ME" -af "vllm serve" || echo "  (моих vLLM нет)"
echo "Удалю папку:"
if [ -d "$TARGET" ]; then du -sh "$TARGET"; echo "  $TARGET"; else echo "  (такой папки нет: $TARGET)"; fi
printf 'Продолжить? Напиши yes: '
read -r ANSWER
if [ "$ANSWER" != "yes" ]; then echo "Отменено, ничего не удалено."; exit 1; fi
[ "${CLEANUP_NO_KILL:-}" = "1" ] || pkill -u "$ME" -f "vllm serve" || true   # CLEANUP_NO_KILL нужен только тестам
[ -d "$TARGET" ] && rm -rf -- "$TARGET" && echo "Удалено: $TARGET"
echo "Готово. Проверь: bash check_gpu.sh"
