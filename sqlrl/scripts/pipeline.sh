#!/bin/zsh
# Usage: scripts/pipeline.sh <model-dir> <name> <rl-hours>
# The same stages for any base model, so models can be compared from their own starting points:
# Each assistant turn may generate 384 tokens; the longest reference answer is about 200.
#   1. score the untrained model on every evaluation split, including the test set
#   2. supervised distillation from a fresh adapter on verified solutions (teaches the protocol)
#   3. GRPO with early stopping on the selection set, inside the time budget
#   4. score the selected checkpoint on every split, including the test set, and write a report
# Each stage writes a new run directory; nothing is overwritten.
set -euo pipefail
model=$1; name=$2; hours=$3
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
cd "$(dirname "$0")/.."
out=runs/${name}_results
mkdir -p $out
$PY -u evaluate.py --model $model --sets train,select,check,test --k 16 --max-tokens 384 --out $out/base.jsonl > $out/base.log 2>&1
$PY -u sft.py --name ${name}_sft --model $model --max-nll 100 --epochs 2 --max-tokens 384 > runs/${name}_sft.log 2>&1
$PY -u train.py --name ${name}_rl --model $model --resume runs/${name}_sft/final --steps 2000 --max-hours $hours \
    --prompts 32 --gen-batch 128 --max-tokens 384 --lr 1.5e-5 --lr-end 5e-6 --eval-every 10 --patience 6 \
    --teacher glm-5.3-flash --seed 41 > runs/${name}_rl.log 2>&1 &
pid=$!
caffeinate -i -s -m -w $pid &
zsh scripts/watchdog.sh $pid runs/${name}_rl.mem 33792 3072 &
$PY -u scripts/monitor.py --run ${name}_rl --pid $pid --mem-alert 28 > runs/${name}_rl.monitor.log 2>&1 &
wait $pid
selected=$($PY -c "import json; print(json.load(open('runs/${name}_rl/selected.json'))['path'])")
$PY -u evaluate.py --model $model --adapter $selected --sets train,select,check,test --k 16 --max-tokens 384 --out $out/trained.jsonl > $out/trained.log 2>&1
$PY scripts/report.py base=$out/base.jsonl trained=$out/trained.jsonl --out $out/report.md
echo "pipeline done: $out/report.md"
