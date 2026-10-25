"""Collect verified teacher solutions for trainable templates (API only, no GPU).

    python scripts/collect_solutions.py --per-template 3 --out data/sql-v2.2/solutions.jsonl

Each line holds a template, question, reference SQL and the teacher's actions, kept only
when the submission matched the reference on every training fixture. Held-out templates
are never included.
"""
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env
import evolve
from teacher import Teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-template", type=int, default=3)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--out", type=Path, default=env.DATA_DIR / "solutions.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", type=Path, help="only templates without solutions in this file; append to --out")
    args = ap.parse_args()
    evolve.load_accepted()
    tasks = []
    have = {json.loads(line)["template"] for line in args.skip_existing.read_text().splitlines()} if args.skip_existing else set()
    for name, (tier, fn) in env.TEMPLATES.items():
        if name in env.HELDOUT_TEMPLATES or name in have:
            continue
        seen = {}
        for seed in range(64):
            q, sql = fn(random.Random(seed))
            if q not in env.eval_question_texts():
                seen.setdefault(q, sql)
        for q, sql in list(seen.items())[:args.per_template]:
            tasks.append(env.Task(name, tier, q, sql))
    teacher = Teacher(args.model, workers=args.workers)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(teacher.solve, tasks))
    kept = 0
    with open(args.out, "a" if args.skip_existing else "w") as f:
        for task, sol in zip(tasks, results):
            if sol["correct"]:
                kept += 1
                f.write(json.dumps({"template": task.template, "tier": task.tier, "question": task.question,
                                    "gold_sql": task.gold_sql, "actions": sol["actions"], "final_sql": sol["final_sql"],
                                    "model": args.model, "verifier_version": env.VERIFIER_VERSION}) + "\n")
    teacher.close()
    print(f"{kept} of {len(tasks)} verified solutions, ${teacher.spent:.3f}, written to {args.out}")


if __name__ == "__main__":
    main()
