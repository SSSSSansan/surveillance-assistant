# Общие настройки serve_*.sh (подключается через source; при подключении ничего не запускает).
# Флаги vLLM зависят от версии: НЕ ПРОВЕРЕНО без сервера. При "unrecognized arguments" смотри `vllm serve --help`.
WORK="${WORK:-$HOME/work}"                          # ВСЁ (веса, кэши, данные, результаты) живёт здесь
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"   # GPU 1 (на GPU 1 свободно ~11,6 из 23 ГБ)
PORT="${PORT:-8765}"
GPU_UTIL="${GPU_UTIL:-0.40}"        # доля ПОЛНОЙ памяти карты: 0.40 x 23 ГБ ~ 9,2 ГБ. Меняется параметром --gpu-memory-utilization
MAX_LEN="${MAX_LEN:-}"              # пусто = значение по умолчанию у модели (см. скрипт)
LIMIT_IMAGES="${LIMIT_IMAGES:-1}"   # для few-shot (вариант C) поставь 4: 3 примера + кроп
EXTRA_ARGS="${EXTRA_ARGS:-}"        # любые дополнительные флаги vLLM

# serve_main REPO SERVED_NAME DEFAULT_MAX_LEN [--print] [--gpu-memory-utilization X]
serve_main() {
  local repo="$1" served="$2" maxlen="${MAX_LEN:-$3}"; shift 3
  local printonly=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --print) printonly=1 ;;
      --gpu-memory-utilization) GPU_UTIL="${2:-}"; shift ;;
      --gpu-memory-utilization=*) GPU_UTIL="${1#*=}" ;;
      -h|--help) echo "Использование: bash $0 [--print] [--gpu-memory-utilization 0.40]"; return 0 ;;
      *) echo "Неизвестный параметр: $1"; return 2 ;;
    esac
    shift
  done
  if ! awk -v x="$GPU_UTIL" 'BEGIN{exit !(x ~ /^[0-9]*\.?[0-9]+$/ && x+0 > 0 && x+0 <= 0.95)}'; then
    echo "gpu-memory-utilization должно быть числом от 0 до 0.95, а не '$GPU_UTIL'"; return 2
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "Порт $PORT уже занят (там что-то слушает). Ничего не запускаю. Свой vLLM останавливает: bash cleanup.sh <модель>"; return 1
  fi
  # у каждой модели свои папки кэша: cleanup.sh удаляет ровно их (имя = как у папки в кэше Hugging Face)
  local slug="models--${repo//\//--}"
  export HF_HOME="$WORK/hf_cache" XDG_CACHE_HOME="$WORK/cache" \
         TRITON_CACHE_DIR="$WORK/cache/triton/$slug" VLLM_CACHE_ROOT="$WORK/cache/vllm/$slug" \
         TORCHINDUCTOR_CACHE_DIR="$WORK/cache/inductor/$slug"
  echo "GPU=$CUDA_VISIBLE_DEVICES порт=$PORT gpu-memory-utilization=$GPU_UTIL max-model-len=$maxlen"
  echo "кэши: HF_HOME=$HF_HOME XDG_CACHE_HOME=$XDG_CACHE_HOME"
  echo "      TRITON_CACHE_DIR=$TRITON_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  # shellcheck disable=SC2086
  local cmd=(vllm serve "$repo" --served-model-name "$served" --port "$PORT" --max-model-len "$maxlen"
             --gpu-memory-utilization "$GPU_UTIL" --limit-mm-per-prompt "image=$LIMIT_IMAGES" $EXTRA_ARGS)
  if [ "$printonly" = 1 ]; then echo "${cmd[@]}"; return 0; fi
  command -v vllm >/dev/null || { echo "vllm не найден: включи venv (source ~/work/venv/bin/activate)"; return 1; }
  mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR"
  exec "${cmd[@]}"
}
