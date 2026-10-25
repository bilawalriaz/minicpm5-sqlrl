"""Measure pass@1 and pass@k per question template on held-out tasks.

    python evaluate.py                          # base model
    python evaluate.py --adapter runs/x/step_100   # after training
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

import env
from rollout import rollout, set_memory_budget, transcript

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(__file__).parent.parent / "models/MiniCPM5-1B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--per-template", type=int, default=3)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--hide-schema", action="store_true")
    ap.add_argument("--out", default=None, help="jsonl file for transcripts")
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--mem-gb", type=float, default=26)
    args = ap.parse_args()
    set_memory_budget(args.mem_gb)

    model, tok = load(args.model)
    if args.adapter:
        load_adapters(model, args.adapter)
    model.eval()

    eps, stats = rollout(model, tok, env.eval_tasks(args.per_template, args.k), temp=args.temp, max_turns=args.max_turns,
                         show_schema=not args.hide_schema, batch_size=args.gen_batch, verbose=True)

    by_template = defaultdict(list)
    for e in eps:
        by_template[e.task.template].append(e)

    print(f"\n{'template':22s} tier  pass@1  pass@{args.k}  turns")
    tier_scores = defaultdict(list)
    all_p1, all_pk = [], []
    for name, es in by_template.items():
        groups = defaultdict(list)
        for e in es:
            groups[e.task.question].append(e.outcome == "correct")
        p1 = sum(e.outcome == "correct" for e in es) / len(es)
        pk = sum(max(g) for g in groups.values()) / len(groups)
        turns = sum(e.turns for e in es) / len(es)
        tier_scores[es[0].task.tier].append(p1)
        all_p1.append(p1)
        all_pk.append(pk)
        print(f"{name:22s} {es[0].task.tier}     {p1:5.2f}   {pk:5.2f}    {turns:.1f}")
    print(f"\nmean pass@1 {sum(all_p1) / len(all_p1):.3f}   mean pass@{args.k} {sum(all_pk) / len(all_pk):.3f}")
    for t in sorted(tier_scores):
        print(f"  tier {t}: pass@1 {sum(tier_scores[t]) / len(tier_scores[t]):.3f}")
    print("outcomes:", dict(Counter(e.outcome for e in eps)))
    print(f"peak MLX memory {mx.get_peak_memory() / 1024**3:.1f}GB")
    print(f"{stats['gen_tokens']} tokens in {stats['seconds']:.0f}s "
          f"({stats['gen_tokens'] / stats['seconds']:.0f} tok/s)")

    if args.out:
        with open(args.out, "w") as f:
            for e in eps:
                f.write(json.dumps({"template": e.task.template, "question": e.task.question,
                                    "reward": e.reward, "outcome": e.outcome,
                                    "verifier_version": env.VERIFIER_VERSION, "protocol_version": env.PROTOCOL_VERSION,
                                    "final_sql": e.final_sql, "verifier_reason": e.verifier_reason,
                                    "db_path": str(e.task.db_path), "segments": e.segments,
                                    "answer_rows": e.answer_rows, "gold_rows": e.gold_rows,
                                    "transcript": transcript(tok, e)}) + "\n")


if __name__ == "__main__":
    main()
