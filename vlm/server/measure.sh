#!/usr/bin/env bash
# Пишет пиковую память GPU во время прогона. Запусти в отдельном окне ПЕРЕД бенчмарком, останови Ctrl-C ПОСЛЕ.
# Использование: bash measure.sh OUT.csv [секунд_между_замерами=2]
# Колонки: время, общая занятая память GPU (МиБ, вместе с чужими процессами!), память моих процессов (МиБ, если видно).
# Общая цифра на общем сервере включает чужие задачи: в отчёт идёт "моя" память, а общая только как справка.
# Определение "моих" процессов через pgrep + nvidia-smi --query-compute-apps: НЕ ПРОВЕРЕНО (в контейнерах pid могут не совпасть).
set -u
OUT="${1:?укажи файл, например vram_qwen3vl4b.csv}"
STEP="${2:-2}"
echo "time,gpu_used_total_mib,gpu_used_mine_mib" > "$OUT"
peak_total=0; peak_mine=0
report() { echo; echo "пик общей: ${peak_total} МиБ; пик моей: ${peak_mine} МиБ; файл: $OUT"; exit 0; }
trap report INT TERM
while true; do
  total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  mine=0
  for pid in $(pgrep -u "$(id -un)" -f "vllm|VLLM" 2>/dev/null); do
    m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v p="$pid" '$1==p{s+=$2} END{print s+0}')
    mine=$((mine + m))
  done
  echo "$(date +%H:%M:%S),$total,$mine" >> "$OUT"
  [ "$total" -gt "$peak_total" ] && peak_total=$total
  [ "$mine" -gt "$peak_mine" ] && peak_mine=$mine
  sleep "$STEP"
done
