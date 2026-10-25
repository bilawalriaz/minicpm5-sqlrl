"""Compare models from evaluate.py transcripts, with bootstrap confidence intervals.

    python scripts/report.py base=runs/release/base.jsonl trained=runs/release/trained.jsonl --out report.md

Each split's pass@1 is the mean over questions of the per-question solve rate. The
95% interval resamples questions (not attempts), because attempts on one question
are correlated.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SPLITS = {"heldout_test": "Never-trained templates (test set)",
          "heldout_check": "Never-trained templates (check set)",
          "heldout_select": "Never-trained templates (selection set)",
          "reserved_params": "Trained templates, unseen parameters and data",
          "shared_wording": "Trained wording, unseen data"}


def load(path):
    by_q = defaultdict(list)
    meta = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        by_q[(r["split"], r["template"], r["question"])].append(r["outcome"] == "correct")
        meta[(r["split"], r["template"], r["question"])] = r
    return by_q


def interval(rates, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(rates) for _ in rates) / len(rates) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label=path.jsonl")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--exclude", nargs="*", default=[], help="templates left out of the split rows (listed in the template rows)")
    args = ap.parse_args()
    models = [(m.split("=", 1)[0], load(m.split("=", 1)[1])) for m in args.models]
    lines = ["| Split | " + " | ".join(label for label, _ in models) + " |",
             "| --- | " + " | ".join("---" for _ in models) + " |"]
    for split, title in SPLITS.items():
        cells = []
        for _, by_q in models:
            rates = [sum(v) / len(v) for (s, t, _), v in by_q.items() if s == split and t not in args.exclude]
            if not rates:
                cells.append("n/a")
                continue
            lo, hi = interval(rates)
            cells.append(f"{sum(rates) / len(rates):.1%} ({lo:.0%}-{hi:.0%}), {len(rates)} q")
        if any(c != "n/a" for c in cells):
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "| Template | Split | " + " | ".join(label for label, _ in models) + " |",
              "| --- | --- | " + " | ".join("---" for _ in models) + " |"]
    keys = sorted({(s, t) for _, by_q in models for (s, t, _) in by_q})
    for split, template in keys:
        cells = []
        for _, by_q in models:
            v = [x for (s, t, _), xs in by_q.items() if (s, t) == (split, template) for x in xs]
            cells.append(f"{sum(v) / len(v):.0%}" if v else "n/a")
        note = " (excluded)" if template in args.exclude else ""
        lines.append(f"| {template}{note} | {split} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
