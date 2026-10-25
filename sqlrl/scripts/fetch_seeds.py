"""Download real question/SQL pairs from Spider's train split as style seeds for evolve.py.

    python scripts/fetch_seeds.py --count 1500

Writes data/sql-v2.2/seeds/spider_train.jsonl. Only the train split is used: Spider dev is
the cross-schema evaluation set (spider_eval.py). Seeds are shown to template proposers as
examples of phrasing and query shape; every template is still written for the shop schema
and validated like any other.
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import env

ROWS_URL = "https://datasets-server.huggingface.co/rows?dataset=xlangai/spider&config=spider&split=train&offset={}&length=100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=env.DATA_DIR / "seeds" / "spider_train.jsonl")
    args = ap.parse_args()
    with urllib.request.urlopen(ROWS_URL.format(0), timeout=60) as r:
        total = json.load(r)["num_rows_total"]
    offsets = sorted(random.Random(0).sample(range(0, total, 100), max(1, args.count // 100)))
    rows = []
    for offset in offsets:
        with urllib.request.urlopen(ROWS_URL.format(offset), timeout=60) as r:
            page = json.load(r)
        rows += [{k: p["row"][k] for k in ("db_id", "question", "query")} for p in page["rows"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows[:args.count]:
            f.write(json.dumps(row) + "\n")
    print(f"{min(len(rows), args.count)} of {total} Spider train pairs written to {args.out}")


if __name__ == "__main__":
    main()
