#!/bin/zsh
# Usage: scripts/export_gguf.sh <model-dir> <adapter-dir> <name> <llama.cpp-checkout>
# Fuses a LoRA checkpoint (its adapters.safetensors holds the evaluated, averaged weights),
# makes no-think the template default (training used an empty think block), turns off the
# extra BOS token (the template already adds one), and writes BF16, Q8_0 and Q4_K_M GGUFs
# to ../release/. The llama.cpp checkout must match the installed llama-quantize build.
set -euo pipefail
model=$1; adapter=$2; name=$3; llama=$4
PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
out=$(cd "$(dirname "$0")/../.." && pwd)/release
mkdir -p $out
$PY -m mlx_lm fuse --model $model --adapter-path $adapter --save-path $out/$name
cp $model/chat_template.jinja $out/$name/
$PY - $out/$name <<'PYEOF'
import json, sys
d = sys.argv[1]
p = f"{d}/chat_template.jinja"
s = open(p).read()
old = """    {%- if enable_thinking is defined %}
        {%- if enable_thinking is false %}
            {{- '<think>\\n\\n</think>\\n\\n' }}
        {%- elif enable_thinking is true %}
            {{- '<think>\\n' }}
        {%- endif %}
    {%- endif %}"""
new = """    {#- Fine-tuned in no-think mode: the empty think block is the default. #}
    {%- if enable_thinking is defined and enable_thinking is true %}
        {{- '<think>\\n' }}
    {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}"""
assert old in s
open(p, "w").write(s.replace(old, new))
c = json.load(open(f"{d}/tokenizer_config.json"))
c.update(add_bos_token=False, add_eos_token=False)
json.dump(c, open(f"{d}/tokenizer_config.json", "w"), indent=2)
PYEOF
PYTHONPATH=$llama/gguf-py:$llama $PY $llama/convert_hf_to_gguf.py $out/$name --outtype bf16 --outfile $out/$name-BF16.gguf
llama-quantize $out/$name-BF16.gguf $out/$name-Q8_0.gguf Q8_0
llama-quantize $out/$name-BF16.gguf $out/$name-Q4_K_M.gguf Q4_K_M
ls -la $out/$name-*.gguf
