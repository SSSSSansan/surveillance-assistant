#!/usr/bin/env bash
source ~/work/venv/bin/activate && cd ~/work/vlm
for i in $(seq 1 90); do curl -s localhost:8765/v1/models >/dev/null && break; sleep 10; done
curl -s localhost:8765/v1/models >/dev/null || { echo "СЕРВЕР НЕ ПОДНЯЛСЯ"; exit 1; }
D=~/work/data/crops2; M=qwen3-vl-8b-awq
C="$D/video1/video1_crops_index.json $D/video2/video2_crops_index.json"
F="$D/video1/video1_frames_index.json $D/video2/video2_frames_index.json"
python3 run_predictions.py count --mode R4 --frames-index $F --frames-dir ~/work/data ~/work/data/video2_frames --crops-index $C --model $M --out ~/work/runs2/count_${M}_R4.jsonl || echo "ОШИБКА R4"
python3 run_predictions.py count --mode R3 --repetition-penalty 1.1 --frames-index $F --frames-dir ~/work/data ~/work/data/video2_frames --crops-index $C --model $M --out ~/work/runs2/count_${M}_R3_rep1.1.jsonl || echo "ОШИБКА R3rep"
echo "ВСЕ ГОТОВО"
