#!/usr/bin/env bash
source ~/work/venv/bin/activate && cd ~/work/vlm
for i in $(seq 1 90); do curl -s localhost:8765/v1/models >/dev/null && break; sleep 10; done
curl -s localhost:8765/v1/models >/dev/null || { echo "СЕРВЕР НЕ ПОДНЯЛСЯ"; exit 1; }
D=~/work/data/crops2; M=qwen3.5-4b-awq; T=--disable-thinking
C="$D/video1/video1_crops_index.json $D/video2/video2_crops_index.json"
F="$D/video1/video1_frames_index.json $D/video2/video2_frames_index.json"
FD="--frames-dir $HOME/work/data $HOME/work/data/video2_frames"
python3 run_predictions.py crops --crops-index $C --model $M $T --variant A --repeat 2 --out ~/work/runs2/pred_clothing_${M}_R1_A.jsonl || echo "ОШИБКА A"
python3 run_predictions.py crops --crops-index $C --model $M $T --variant B --out ~/work/runs2/pred_clothing_${M}_R1_B.jsonl || echo "ОШИБКА B"
python3 run_predictions.py crops --crops-index $C --model $M $T --variant D --out ~/work/runs2/pred_clothing_${M}_R1_D.jsonl || echo "ОШИБКА D"
for MODE in R2 R3 R4; do
  python3 run_predictions.py count --mode $MODE --frames-index $F $FD --crops-index $C --model $M $T --out ~/work/runs2/count_${M}_${MODE}.jsonl || echo "ОШИБКА $MODE"
done
python3 run_predictions.py count --mode R3 --repetition-penalty 1.1 --frames-index $F $FD --crops-index $C --model $M $T --out ~/work/runs2/count_${M}_R3_rep1.1.jsonl || echo "ОШИБКА R3rep"
echo "ВСЕ ГОТОВО"
