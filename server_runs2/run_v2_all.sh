#!/usr/bin/env bash
source ~/work/venv/bin/activate && cd ~/work/vlm
D=~/work/data/crops2
M=qwen3-vl-8b-awq
C="$D/video1/video1_crops_index.json $D/video2/video2_crops_index.json"
F="$D/video1/video1_frames_index.json $D/video2/video2_frames_index.json"
python3 run_predictions.py crops --crops-index $C --model $M --variant A --repeat 2 --out ~/work/runs2/pred_clothing_${M}_R1_A.jsonl || echo "ОШИБКА A"
python3 run_predictions.py crops --crops-index $C --model $M --variant B --out ~/work/runs2/pred_clothing_${M}_R1_B.jsonl || echo "ОШИБКА B"
python3 run_predictions.py crops --crops-index $C --model $M --variant D --out ~/work/runs2/pred_clothing_${M}_R1_D.jsonl || echo "ОШИБКА D"
for MODE in R2 R3 R4; do
  python3 run_predictions.py count --mode $MODE --frames-index $F --frames-dir ~/work/data ~/work/data/video2_frames \
    --crops-index $C --model $M --out ~/work/runs2/count_${M}_${MODE}.jsonl || echo "ОШИБКА $MODE"
done
echo "ВСЕ ГОТОВО"
