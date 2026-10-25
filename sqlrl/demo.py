"""Ask the trained SQL agent a question about the shop database.

    python demo.py --adapter runs/<name>/step_NNNNN "Which customers in Paris spent more than 500 on Books?"

The agent explores with run_sql and answers with submit_sql. The transcript and the complete
result of the submitted query are printed. Uses the evaluation database by default.
"""
import argparse
from pathlib import Path

from mlx_lm import load
from mlx_lm.tuner.utils import load_adapters

import env
from rollout import rollout, set_memory_budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--adapter", help="LoRA checkpoint directory; omit for the base model")
    ap.add_argument("--model", default=str(Path(__file__).parent.parent / "models/MiniCPM5-1B"))
    ap.add_argument("--db", choices=["eval", "train"], default="eval")
    ap.add_argument("--temp", type=float, default=0.0, help="0 decodes greedily")
    args = ap.parse_args()
    set_memory_budget(8)
    model, tok = load(args.model)
    if args.adapter:
        model = load_adapters(model, args.adapter)
    model.eval()
    db = env.EVAL_DB_PATH if args.db == "eval" else env.DB_PATH
    task = env.Task("demo", 0, args.question, "", db)
    (ep,), _ = rollout(model, tok, [task], temp=args.temp, batch_size=1, grade=False)
    for seg in ep.segments:
        print(f"[{seg['role']}] {seg['text']}\n")
    if ep.outcome == "submitted":
        print(f"Answer ({len(ep.answer_rows)} rows):")
        for row in ep.answer_rows[:50]:
            print("  " + " | ".join(str(v) for v in row))
    else:
        print(f"No answer: {ep.outcome} {ep.verifier_reason}")


if __name__ == "__main__":
    main()
