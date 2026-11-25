#!/usr/bin/env bash
# Убирает за собой. Всегда сначала показывает, что сделает, и просит подтверждение.
#   bash cleanup.sh ORG/NAME   останавливает МОЙ vLLM и удаляет ТОЛЬКО кэши этой модели:
#                              ~/work/hf_cache/hub/models--ORG--NAME и подпапки models--ORG--NAME в
#                              ~/work/cache/{vllm,triton,inductor}   (их создают serve_*.sh). Подтверждение: yes
#                              Если после этого в ~/work/hf_cache/hub не осталось ни одной models--*, предлагает
#                              удалить осиротевшие hub/blobs и hub/.locks (показывает размер, отдельное подтверждение yes)
#   bash cleanup.sh --all      в самом конце: останавливает мой vLLM и удаляет ВЕСЬ ~/work (venv, данные, runs, кэши).
#                              Сначала скопируй ~/work/runs на свой компьютер! Подтверждение: delete-work
# Кроме ~/work чистится ещё ~/.cache/flashinfer (его создаёт FlashInfer вне ~/work; общий для моделей, при необходимости соберётся заново).
# Чужое не трогается: pkill только с -u моего пользователя и только "vllm serve ... --port 8765"; пути только внутри $HOME/work.
set -u
cleanup_main() {
  local ME; ME="$(id -un)"
  local WORK="${WORK:-$HOME/work}" PORT="${PORT:-8765}"
  local PAT="vllm serve.*--port $PORT"
  stop_mine() { [ "${CLEANUP_NO_KILL:-}" = "1" ] || pkill -u "$ME" -f "$PAT" || true; }   # CLEANUP_NO_KILL нужен только тестам
  local FI="$HOME/.cache/flashinfer"
  show_fi() { if [ -d "$FI" ] && [ ! -L "$FI" ]; then du -sh "$FI"; else echo "  (нет) $FI"; fi; }
  rm_fi() { if [ -d "$FI" ] && [ ! -L "$FI" ]; then rm -rf -- "$FI" && echo "Удалено: $FI"; fi; }
  show_procs() { echo "Остановлю процессы (только мои):"; pgrep -u "$ME" -af "$PAT" || echo "  (моих vLLM нет)"; }

  if [ "${1:-}" = "--all" ]; then
    if [ "$WORK" != "$HOME/work" ] || [ -L "$WORK" ]; then echo "Отказ: удаляю только настоящую папку $HOME/work (не ссылку, не другой путь)."; return 2; fi
    show_procs
    echo "Удалю ВСЮ папку: $WORK"; du -sh "$WORK" "$WORK"/* 2>/dev/null
    show_fi
    echo "ВНИМАНИЕ: там же venv и результаты (runs/). Результаты сначала скопируй к себе."
    printf 'Для удаления напиши delete-work: '; read -r ANSWER
    if [ "$ANSWER" != "delete-work" ]; then echo "Отменено, ничего не удалено."; return 1; fi
    stop_mine; rm -rf -- "$WORK" && echo "Удалено: $WORK"; rm_fi; return 0
  fi

  # hub/blobs и hub/.locks без единой модели в hub: только настоящие папки (не ссылки) моего пользователя внутри $WORK
  clean_orphans() {
    local HUB="$WORK/hf_cache/hub" o ORPH=()
    [ -d "$HUB" ] && [ ! -L "$HUB" ] && [ ! -L "$WORK/hf_cache" ] && [ ! -L "$WORK" ] || return 0
    compgen -G "$HUB/models--*" >/dev/null && return 0   # ещё есть модели: blobs/.locks могут быть им нужны
    for o in "$HUB/blobs" "$HUB/.locks"; do
      [ -e "$o" ] || continue
      if [ -d "$o" ] && [ ! -L "$o" ] && [ -O "$o" ]; then ORPH+=("$o"); else echo "Не трогаю (ссылка или не моя папка): $o"; fi
    done
    [ "${#ORPH[@]}" -gt 0 ] || return 0
    echo "В $HUB не осталось ни одной модели (models--*). Осиротевшие папки:"; du -sh "${ORPH[@]}"
    printf 'Удалить их? Напиши yes: '; read -r ANSWER || ANSWER=""
    if [ "$ANSWER" != "yes" ]; then echo "Оставлено: ${ORPH[*]}"; return 0; fi
    for o in "${ORPH[@]}"; do rm -rf -- "$o" && echo "Удалено: $o"; done
  }

  local REPO="${1:-}"
  if ! [[ "$REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    echo "Использование: bash cleanup.sh ORG/NAME   (например Qwen/Qwen3-VL-4B-Instruct)   или   bash cleanup.sh --all"; return 2
  fi
  local SLUG="models--${REPO//\//--}"
  local TARGETS=("$WORK/hf_cache/hub/$SLUG" "$WORK/cache/vllm/$SLUG" "$WORK/cache/triton/$SLUG" "$WORK/cache/inductor/$SLUG")
  echo "Пользователь: $ME"; show_procs
  echo "Удалю папки (только этой модели):"
  local t; for t in "${TARGETS[@]}"; do if [ -d "$t" ]; then du -sh "$t"; else echo "  (нет) $t"; fi; done
  show_fi
  printf 'Продолжить? Напиши yes: '; read -r ANSWER
  if [ "$ANSWER" != "yes" ]; then echo "Отменено, ничего не удалено."; return 1; fi
  stop_mine
  for t in "${TARGETS[@]}"; do [ -d "$t" ] && rm -rf -- "$t" && echo "Удалено: $t"; done
  clean_orphans
  rm_fi
  echo "Готово. Проверь: bash check_gpu.sh"; return 0
}
cleanup_main "$@"; exit $?
