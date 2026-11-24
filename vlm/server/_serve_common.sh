# Общие настройки serve_*.sh (подключается через source; сам ничего не запускает).
# Флаги vLLM зависят от версии: НЕ ПРОВЕРЕНО без сервера. При ошибке "unrecognized arguments" смотри `vllm serve --help`.
PORT="${PORT:-8765}"                       # порт из плана
GPU_UTIL="${GPU_UTIL:-0.45}"               # подгонять ВНИЗ, не вверх: сервер общий
MAX_LEN="${MAX_LEN:-8192}"
LIMIT_IMAGES="${LIMIT_IMAGES:-1}"          # для few-shot (вариант C) поставь 4: 3 примера + кроп
EXTRA_ARGS="${EXTRA_ARGS:-}"               # любые дополнительные флаги vLLM

run_or_print() {  # run_or_print "--print" cmd... : с --print только показывает команду
  if [ "${1:-}" = "--print" ]; then shift; echo "$@"; else exec "$@"; fi
}
