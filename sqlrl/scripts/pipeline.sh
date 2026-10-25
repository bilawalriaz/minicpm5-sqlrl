#!/bin/zsh
# Usage: scripts/pipeline.sh <model-dir> <name> <rl-hours>
# The same stages for any released model, so models can be compared from their own starting points:
#   1. score the released model on every evaluation split, including the test set
#   2. supervised distillation from a fresh LoRA adapter (all layers) on verified solutions;
#      this also teaches the tool protocol
#   3. GRPO with early stopping on the selection set, inside the time budget
#   4. a second distillation pass on the solutions plus the RL run's own correct transcripts
#   5. keep whichever of the RL selection and the second distillation scores higher on the
#      selection set, score it on every split, and write a report
# Each assistant turn may generate 384 tokens; the longest reference answer is about 200.
# Each stage writes a new run directory; nothing is overwritten.
set -euo pipefail
model=$1; name=$2; hours=$3
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
cd "$(dirname "$0")/.."
out=runs/${name}_results
solutions=data/sql-v2.2/solutions_v2.jsonl
exclude=(avg_order_value_country)  # engineering/test-set-exclusions.txt
layers=$($PY -c "import json; print(json.load(open('$model/config.json'))['num_hidden_layers'])")
mkdir -p $out
echo "$(date '+%F %T') stage 1: released model evaluation" >> $out/progress.log
$PY -u evaluate.py --model $model --sets train,select,check,test --k 16 --max-tokens 384 --out $out/released.jsonl > $out/released.log 2>&1

echo "$(date '+%F %T') stage 2: distillation from a fresh adapter ($layers layers)" >> $out/progress.log
$PY -u sft.py --name ${name}_sft --model $model --lora-layers $layers --solutions $solutions --max-nll 100 \
    --epochs 2 --max-tokens 384 > runs/${name}_sft.log 2>&1

echo "$(date '+%F %T') stage 3: GRPO for up to $hours hours" >> $out/progress.log
$PY -u train.py --name ${name}_rl --model $model --resume runs/${name}_sft/final --steps 2000 --max-hours $hours \
    --prompts 32 --gen-batch 128 --max-tokens 384 --lr 1.5e-5 --lr-end 5e-6 --eval-every 10 --patience 6 \
    --teacher glm-5.3-flash --seed 41 > runs/${name}_rl.log 2>&1 &
pid=$!
caffeinate -i -s -m -w $pid &
zsh scripts/watchdog.sh $pid runs/${name}_rl.mem 33792 3072 &
$PY -u scripts/monitor.py --run ${name}_rl --pid $pid --mem-alert 28 > runs/${name}_rl.monitor.log 2>&1 &
wait $pid
rl=$($PY -c "import json; print(json.load(open('runs/${name}_rl/selected.json'))['path'])")

echo "$(date '+%F %T') stage 4: distillation with replay from $rl" >> $out/progress.log
$PY -u sft.py --name ${name}_sft2 --model $model --resume $rl --solutions $solutions --replay ${name}_rl \
    --epochs 2 --max-nll 1.5 --max-tokens 384 > runs/${name}_sft2.log 2>&1
final=$($PY -c "
import json
r = json.load(open('runs/${name}_sft2/sft_result.json'))
print('runs/${name}_sft2/final' if r['after']['heldout_select'] > r['before']['heldout_select'] else '$rl')")

echo "$(date '+%F %T') stage 5: final evaluation of $final" >> $out/progress.log
echo $final > $out/final_checkpoint.txt
$PY -u evaluate.py --model $model --adapter $final --sets train,select,check,test --k 16 --max-tokens 384 \
    --out $out/trained.jsonl > $out/trained.log 2>&1
$PY scripts/report.py released=$out/released.jsonl trained=$out/trained.jsonl --exclude $exclude --out $out/report.md
echo "$(date '+%F %T') done: $out/report.md" >> $out/progress.log
