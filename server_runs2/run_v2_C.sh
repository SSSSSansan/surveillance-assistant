#!/usr/bin/env bash
source ~/work/venv/bin/activate && cd ~/work/vlm
D=~/work/data/crops2
M=qwen3-vl-8b-awq
C="$D/video1/video1_crops_index.json $D/video2/video2_crops_index.json"
python3 run_predictions.py crops --crops-index $C --model $M --variant C --fewshot-examples ~/work/data/fewshot_examples.json --limit-images 4 --out ~/work/runs2/pred_clothing_${M}_R1_C.jsonl || echo "ОШИБКА C"
echo "ВСЕ ГОТОВО"
