"""Compare sampler log-probabilities with a full-sequence training forward pass.

    python scripts/logprob_check.py --adapter runs/x/final --out engineering/logprob-check.json

Large differences mean the rollout and the trainer score different token sequences:
wrong KV-cache continuation, chat-boundary tokens, masks or target shifts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

import env
from rollout import rollout, set_memory_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[2] / "models/MiniCPM5-1B"))
    ap.add_argument("--adapter")
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    set_memory_budget(26)
    model, tok = load(args.model)
    if args.adapter:
        model = load_adapters(model, args.adapter)
    model.eval()
    tasks = [t for t in env.sample_tasks(args.tasks, seed=777) for _ in range(args.group)]
    eps, _ = rollout(model, tok, tasks, temp=1.0, batch_size=64)
    def train_logprobs(tokens, mask):
        logits = model.lm_head(model.model(mx.array(tokens[:-1])[None])).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        idx = [i for i in range(len(tokens) - 1) if mask[i + 1]]
        return mx.take_along_axis(lp[mx.array(idx)], mx.array([tokens[i + 1] for i in idx])[:, None], axis=-1)[:, 0].tolist()

    diffs, later, control = [], [], []
    for e in eps:
        d = [a - b for a, b in zip(e.sample_logprobs, train_logprobs(e.tokens, e.mask))]
        assert len(d) == len(e.sample_logprobs) == sum(e.mask)
        diffs += d
        if e.turns > 1:
            # Positive control: drop the first turn's stop token, as a cache that missed it would.
            end = next(i for i in range(e.prompt_len, len(e.tokens)) if e.mask[i] and not e.mask[i + 1])
            first_turn = sum(e.mask[:end + 1])
            later += d[first_turn:]
            broken = train_logprobs(e.tokens[:end] + e.tokens[end + 1:], e.mask[:end] + e.mask[end + 1:])
            control += [a - b for a, b in zip(e.sample_logprobs[first_turn:], broken[first_turn - 1:])]
    abs_d = sorted(abs(v) for v in diffs)
    mean_abs = lambda xs: sum(map(abs, xs)) / max(len(xs), 1)
    report = {
        "episodes": len(eps), "multi_turn_episodes": sum(e.turns > 1 for e in eps), "tokens": len(diffs),
        "mean_abs_diff": mean_abs(diffs), "p99_abs_diff": abs_d[int(0.99 * len(abs_d))], "max_abs_diff": abs_d[-1],
        "mean_diff_sampler_minus_train": sum(diffs) / len(diffs),
        "later_turn_tokens": len(later), "later_turn_mean_abs_diff": mean_abs(later),
        "control_missing_stop_token_mean_abs_diff": mean_abs(control),
        "adapter": args.adapter,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
