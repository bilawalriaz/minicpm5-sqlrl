"""Regrade archived submitted SQL with the original databases plus versioned fixtures.

This compares SQL semantics, not compatibility with the submission protocol.
Usage: python audit_rollouts.py --run curriculum_50b --out engineering/sql-audit-curriculum_50b.json
"""
import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from pathlib import Path

import env

ROOT = Path(__file__).parent
LEGACY = [ROOT / "shop.db", ROOT / "shop_alt1.db", ROOT / "shop_alt2.db"]


def fixture_set(stack, version):
    paths = [p for p in LEGACY if p.exists()]
    data = ROOT / "data" / version
    paths += [data / "train.db"] + [data / f"train_alt{i}.db" for i in (1, 2, 3)]
    return [stack.enter_context(closing(env.connect(p))) for p in paths]


def audit(runs, output, previous="sql-v2.1"):
    counts, by_source, examples, changed = Counter(), defaultdict(Counter), [], Counter()
    cache = {}
    with ExitStack() as stack:
        old_suite, new_suite = fixture_set(stack, previous), fixture_set(stack, env.VERIFIER_VERSION)
        files = [p for run in runs for p in sorted((ROOT / "runs" / run / "samples").glob("step_*.json"))]
        for path in files:
            for record in json.loads(path.read_text()):
                if not record.get("final_sql"):
                    counts["no_recorded_submission"] += 1
                    continue
                key = (record["final_sql"], record["gold_sql"])
                if key not in cache:
                    cache[key] = (env.grade(old_suite, *key)[0], env.grade(new_suite, *key))
                old_ok, (new_ok, reason) = cache[key]
                recorded = record["outcome"] == "correct"
                label = (f"recorded_{'accepted' if recorded else 'rejected'}"
                         f"_{previous}_{'accepted' if old_ok else 'rejected'}"
                         f"_{env.VERIFIER_VERSION}_{'accepted' if new_ok else 'rejected'}")
                counts[label] += 1
                by_source[record.get("source", "student")][label] += 1
                if old_ok and not new_ok:
                    changed[record["template"]] += 1
                    if sum(e["template"] == record["template"] for e in examples) < 3:
                        examples.append({"file": f"{path.parent.parent.name}/{path.stem}", "template": record["template"],
                                         "source": record.get("source", "student"), "question": record["question"],
                                         "sql": key[0], "reference": key[1], "reason": reason})
    result = {"runs": runs, "previous_fixtures": previous, "verifier_version": env.VERIFIER_VERSION,
              "comparison": "SQL semantics on the three original databases plus four training fixtures of each version; "
                            "the submission protocol is not regraded",
              "files": len(files), "counts": dict(counts), "by_source": {k: dict(v) for k, v in by_source.items()},
              f"newly_rejected_by_template": dict(changed), "examples": examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.out)
