#!/usr/bin/env bash
source ~/work/venv/bin/activate && cd ~/work/vlm
D=~/work/data/crops2
M=qwen3-vl-4b-awq
for MODE in R2 R3 R4; do
  python3 run_predictions.py count --mode $MODE \
    --frames-index $D/video1/video1_frames_index.json $D/video2/video2_frames_index.json \
    --frames-dir ~/work/data ~/work/data/video2_frames \
    --crops-index $D/video1/video1_crops_index.json $D/video2/video2_crops_index.json \
    --model $M --out ~/work/runs2/count_${M}_${MODE}.jsonl || echo "ОШИБКА $MODE"
done
echo "ВСЕ ГОТОВО"
