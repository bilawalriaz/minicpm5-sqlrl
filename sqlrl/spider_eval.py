"""Cross-schema evaluation: Spider dev questions through the run_sql/submit_sql agent.

    python scripts/prepare_spider.py                        # once
    python spider_eval.py --check-only                      # reference queries only, no model
    python spider_eval.py --name spider_1b_released
    python spider_eval.py --name spider_2b_released --model ../models/MiniCPM5-2B

The agent explores the original Spider database. Every submission is graded on that
database and on all of its Spider test-suite variants, with two rules:

- strict: the training verifier (env.grade). Columns keep their positions; row order
  and duplicates are compared as a multiset.
- test-suite: Spider's test-suite rule with DISTINCT kept (exec_eval.result_eq with
  --keep_distinct). Columns may be permuted, and row order counts when the reference
  has ORDER BY.

Reference queries that fail or exceed the query budget on any database are excluded
and listed in excluded.json. Results go to runs/<name>/.
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from itertools import product
from pathlib import Path

import env

DATA = Path(__file__).parent / "data" / "spider"
_ORDER_BY = re.compile(r"\border\s+by\b", re.I)


def databases(db_id):
    """The original Spider database first, then its test-suite variants."""
    folder = DATA / "database" / db_id
    primary = folder / f"{db_id}.sqlite"
    return primary, tuple(sorted(p for p in folder.glob("*.sqlite") if p != primary))


def schema_text(path):
    with closing(env.connect(path)) as con:
        rows = con.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY rowid").fetchall()
    return "\n".join(sql.strip() + ";" for sql, in rows)


def testsuite_match(pred, gold, order_matters):
    """Spider test-suite equality: the same rows under some column permutation."""
    if pred is None or gold is None:
        return False
    if not pred and not gold:
        return True
    if len(pred) != len(gold) or len(pred[0]) != len(gold[0]):
        return False
    try:
        pred = [tuple(env._norm(v) for v in row) for row in pred]
        gold = [tuple(env._norm(v) for v in row) for row in gold]
    except ValueError:
        return False
    n = len(gold[0])
    pred_cols = [Counter(row[j] for row in pred) for j in range(n)]
    gold_cols = [Counter(row[i] for row in gold) for i in range(n)]
    # Gold column i can only come from a predicted column with the same values.
    options = [[j for j in range(n) if pred_cols[j] == gold_cols[i]] for i in range(n)]
    for perm in product(*options):
        if len(set(perm)) < n:
            continue
        moved = [tuple(row[j] for j in perm) for row in pred]
        if moved == gold if order_matters else Counter(moved) == Counter(gold):
            return True
    return False


def gold_results(cons, gold):
    """Reference rows on every database, or the first error."""
    results = []
    for i, con in enumerate(cons):
        rows, err = env.execute(con, gold, deterministic=True)
        if err:
            return None, f"database {i}: {err}"
        results.append(rows)
    return results, None


def grade_testsuite(cons, sql, gold, expected):
    order_matters = bool(_ORDER_BY.search(gold))
    for con, gold_rows in zip(cons, expected):
        rows, err = env.execute(con, sql, require_table=True, deterministic=True)
        if err or not testsuite_match(rows, gold_rows, order_matters):
            return False
    return True


def load_questions(db_filter=None, limit=None, seed=0):
    dev = json.loads((DATA / "dev.json").read_text())
    for i, r in enumerate(dev):
        r["index"] = i
    if db_filter:
        dev = [r for r in dev if r["db_id"] in db_filter]
    if limit:
        dev = sorted(random.Random(seed).sample(dev, min(limit, len(dev))), key=lambda r: r["index"])
    return dev


def interval(rates, n=2000, seed=0):
    """95% interval over questions, as in scripts/report.py."""
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(rates) for _ in rates) / len(rates) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def summarize(records, k):
    by_q = defaultdict(list)
    for r in records:
        by_q[r["index"]].append(r)
    out = {"questions": len(by_q), "attempts": len(records), "k": k,
           "outcomes": dict(Counter(r["outcome"] for r in records))}
    for rule in ("strict", "testsuite"):
        rates = [sum(a[rule] for a in v) / len(v) for v in by_q.values()]
        lo, hi = interval(rates)
        out[rule] = {"pass@1": sum(rates) / len(rates), "interval": [lo, hi],
                     f"pass@{k}": sum(any(a[rule] for a in v) for v in by_q.values()) / len(by_q)}
    by_db = defaultdict(list)
    for v in by_q.values():
        by_db[v[0]["db_id"]].append(sum(a["testsuite"] for a in v) / len(v))
    out["testsuite_by_db"] = {db: {"questions": len(r), "pass@1": sum(r) / len(r)} for db, r in sorted(by_db.items())}
    return out


def check_references(questions):
    """Run every reference on every database; returns {index: rows per database} and exclusions."""
    expected, excluded = {}, []
    for db_id in sorted({q["db_id"] for q in questions}):
        primary, alts = databases(db_id)
        with ExitStack() as stack:
            cons = [stack.enter_context(closing(c)) for c in env.verifier_connections(primary, alts)]
            for q in questions:
                if q["db_id"] != db_id:
                    continue
                rows, err = gold_results(cons, q["query"])
                if err:
                    excluded.append({"index": q["index"], "db_id": db_id, "query": q["query"], "reason": err})
                else:
                    expected[q["index"]] = rows
    return expected, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="run directory under runs/; must not exist")
    ap.add_argument("--model", default=str(Path(__file__).parent.parent / "models/MiniCPM5-1B"))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--k", type=int, default=4, help="attempts per question")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=384, help="generated tokens per assistant turn")
    ap.add_argument("--db", nargs="*", help="only these Spider databases")
    ap.add_argument("--limit", type=int, help="sample this many questions")
    ap.add_argument("--gen-batch", type=int, default=64)
    ap.add_argument("--mem-gb", type=float, default=30)
    ap.add_argument("--check-only", action="store_true", help="check reference queries and exit")
    args = ap.parse_args()

    questions = load_questions(args.db, args.limit)
    started = time.time()
    expected, excluded = check_references(questions)
    print(f"{len(expected)} usable references, {len(excluded)} excluded ({time.time() - started:.0f}s)")
    for x in excluded:
        print(f"  excluded {x['db_id']} #{x['index']}: {x['reason']}")
    if args.check_only:
        return
    if not args.name:
        ap.error("--name is required unless --check-only is given")
    out_dir = Path(__file__).parent / "runs" / args.name
    out_dir.mkdir(parents=True)  # a new directory for every run

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters
    from rollout import rollout, set_memory_budget, transcript

    set_memory_budget(args.mem_gb)
    model, tok = load(args.model)
    if args.adapter:
        load_adapters(model, args.adapter)
    model.eval()
    config = {**vars(args), "eval": "spider-dev-testsuite-v1", "verifier_version": env.VERIFIER_VERSION,
              "protocol_version": env.PROTOCOL_VERSION, "questions": len(questions),
              "usable": len(expected), "excluded": len(excluded)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "excluded.json").write_text(json.dumps(excluded, indent=1) + "\n")

    records, gen_tokens, seconds = [], 0, 0.0
    with open(out_dir / "episodes.jsonl", "w") as f:
        for db_id in sorted({q["db_id"] for q in questions}):
            items = [q for q in questions if q["db_id"] == db_id and q["index"] in expected]
            if not items:
                continue
            primary, alts = databases(db_id)
            schema = schema_text(primary)
            tasks = [env.Task(f"spider/{db_id}", 0, q["question"], q["query"], primary, "spider_dev", schema, alts)
                     for q in items for _ in range(args.k)]
            eps, stats = rollout(model, tok, tasks, temp=args.temp, max_turns=args.max_turns,
                                 max_tokens=args.max_tokens, batch_size=args.gen_batch)
            gen_tokens, seconds = gen_tokens + stats["gen_tokens"], seconds + stats["seconds"]
            with ExitStack() as stack:
                cons = [stack.enter_context(closing(c)) for c in env.verifier_connections(primary, alts)]
                for q, e in zip((q for q in items for _ in range(args.k)), eps):
                    submitted = e.final_sql is not None and e.outcome in ("correct", "wrong")
                    ts = submitted and grade_testsuite(cons, e.final_sql, q["query"], expected[q["index"]])
                    r = {"index": q["index"], "db_id": db_id, "question": q["question"], "gold_sql": q["query"],
                         "outcome": e.outcome, "strict": e.outcome == "correct", "testsuite": ts,
                         "final_sql": e.final_sql, "verifier_reason": e.verifier_reason, "turns": e.turns,
                         "segments": e.segments, "transcript": transcript(tok, e)}
                    records.append(r)
                    f.write(json.dumps(r) + "\n")
            f.flush()
            done = sum(r["testsuite"] for r in records if r["db_id"] == db_id)
            print(f"{db_id:30s} {len(items):4d} q  test-suite {done / len(eps):.2f}  "
                  f"strict {sum(e.outcome == 'correct' for e in eps) / len(eps):.2f}  "
                  f"{stats['gen_tokens']} tokens in {stats['seconds']:.0f}s", flush=True)

    summary = {**summarize(records, args.k), "excluded": len(excluded), "gen_tokens": gen_tokens,
               "seconds": seconds, "peak_mem_gib": mx.get_peak_memory() / 1024**3}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for rule in ("testsuite", "strict"):
        s = summary[rule]
        print(f"{rule:9s} pass@1 {s['pass@1']:.1%} ({s['interval'][0]:.1%}-{s['interval'][1]:.1%})  "
              f"pass@{args.k} {s[f'pass@{args.k}']:.1%}")
    print("outcomes:", summary["outcomes"])


if __name__ == "__main__":
    main()
