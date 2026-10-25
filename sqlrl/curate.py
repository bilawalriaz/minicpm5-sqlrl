"""Build a reviewed dataset of current-version submitted SQL traces.

    python curate.py ingest             # every graded trace from runs/, re-checked by the verifier
    python curate.py review --limit 16  # GLM reviews verifier-passed traces, 8 per call
    python curate.py report             # counts, plus every flagged or disputed trace

A trace is safe only when both checks agree: the verifier matched its final query
against the reference on every test database, and the reviewer model approved it.
Use the `safe_traces` view. Rejected traces can be reviewed too (--rejects N): there
the reviewer looks for cases where the verifier was wrong to reject.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import env
from teacher import cost_of, make_client, session_header

HERE = Path(__file__).parent
DB = env.DATA_DIR / "templates.db"
TRACE_DB = env.DATA_DIR / "traces.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY,
    hash TEXT UNIQUE NOT NULL,
    run TEXT NOT NULL,
    step INTEGER NOT NULL,
    source TEXT NOT NULL,              -- student | teacher
    template TEXT NOT NULL,
    tier INTEGER NOT NULL,
    question TEXT NOT NULL,
    gold_sql TEXT NOT NULL,
    final_sql TEXT NOT NULL,           -- the explicit submit_sql query
    segments TEXT NOT NULL,            -- JSON [{role: assistant|tool, text}]
    prose_tokens INTEGER,
    verifier TEXT NOT NULL,            -- pass | fail
    verifier_reason TEXT NOT NULL,
    review TEXT NOT NULL DEFAULT 'pending',
        -- verifier pass: approved | flagged; verifier fail: agreed | disputed
    review_reason TEXT,
    reviewer TEXT,
    reviewed_at REAL,
    added_at REAL NOT NULL
);
CREATE VIEW IF NOT EXISTS safe_traces AS
    SELECT * FROM traces WHERE verifier = 'pass' AND review = 'approved';
"""

PASS_PROMPT = """You audit training data for a small text-to-SQL agent. In each item the agent answered a question \
about a SQLite shop database using SQL tools. Its grade is the complete result of its \
explicit submit_sql call ("final query"). An automatic check already confirmed the final query returns \
the same result as the reference query on versioned databases with different data.

Catch what that check cannot. Approve an item only if all of these hold:
1. The final query answers exactly the question asked: the right filters (status, dates, country, category), \
the right aggregation, only the columns asked for, and any instruction like rounding or "return 0 if none".
2. It is correct in general, not just on this data: same logic as the reference, no shortcut that happens to work.
3. The agent submits its answer through submit_sql. Its complete SQL result is the user answer.
4. Nothing degenerate: no repetition loops, garbage text, or tool output invented by the agent.

Database schema:
{schema}

Reply with JSON only: {{"reviews": [{{"id": <id>, "verdict": "approve" | "flag", "reason": "<one sentence>"}}]}}, \
one entry per item."""

REJECT_PROMPT = """You audit an automatic grader for a small text-to-SQL agent. In each item the agent answered a \
question about a SQLite shop database using a run_sql tool. Its grade is the complete result of its explicit submit_sql \
call ("final query"). The grader REJECTED every item below, because the final query's result differed \
from the reference query's on at least one versioned database.

For each item decide whether the grader was right. Say "disputed" only if the final query answers the question \
as asked (columns, filters, rounding, "return 0 if none") and the reference query or the grading is what is wrong. \
Otherwise say "agreed".

Database schema:
{schema}

Reply with JSON only: {{"reviews": [{{"id": <id>, "verdict": "agreed" | "disputed", "reason": "<one sentence>"}}]}}, \
one entry per item."""


def connect():
    env.ensure_databases()
    con = sqlite3.connect(TRACE_DB)
    con.executescript(SCHEMA)
    return con


def ingest():
    con, cons = connect(), env.verifier_connections()
    added = {"pass": 0, "fail": 0}
    for path in sorted(HERE.glob("runs/*/samples/step_*.json")):
        run, step = path.parent.parent.name, int(path.stem.split("_")[1])
        for r in json.loads(path.read_text()):
            if r.get("protocol_version") != env.PROTOCOL_VERSION or r.get("verifier_version") != env.VERIFIER_VERSION:
                continue
            if not r.get("final_sql"):
                continue  # never ran a query, nothing to grade
            segments = json.dumps(r["segments"])
            key = hashlib.sha1((r["question"] + segments).encode()).hexdigest()
            ok, reason = env.grade(cons, r["final_sql"], r["gold_sql"])
            cur = con.execute(
                "INSERT OR IGNORE INTO traces (hash, run, step, source, template, tier, question, gold_sql, final_sql,"
                " segments, prose_tokens, verifier, verifier_reason, added_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, run, step, r.get("source", "student"), r["template"], r["tier"], r["question"], r["gold_sql"],
                 r["final_sql"], segments, r.get("prose_tokens"), "pass" if ok else "fail", reason, time.time()),
            )
            added["pass" if ok else "fail"] += cur.rowcount
    con.commit()
    print(f"added {added['pass']} passing and {added['fail']} failing traces")


def render(row):
    """One trace as plain text for the reviewer."""
    lines = [f"### Item {row['id']}", f"Question: {row['question']}", f"Reference query: {row['gold_sql']}", "Conversation:"]
    for s in json.loads(row["segments"]):
        if s["role"] == "tool":
            lines.append(f"  [tool result] {s['text'][:600]}")
            continue
        try:
            name, query = env.parse_call(s["text"])
        except ValueError:
            name, query = None, None
        if text := env.prose(s["text"]):
            lines.append(f"  [agent] {text[:600]}")
        if name:
            lines.append(f"  [agent calls {name}] {query}")
    lines.append(f"Final query (graded): {row['final_sql']}")
    lines.append(f"Grader: {row['verifier_reason']}")
    return "\n".join(lines)


def review_batch(client, model, rows, prompt, verdicts):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt.format(schema=env.SCHEMA)},
                  {"role": "user", "content": "\n\n".join(render(r) for r in rows)}],
        reasoning_effort="low", max_tokens=4096, extra_headers=session_header(),
    )
    text = resp.choices[0].message.content or ""
    match = re.search(r"\{.*\}", text, re.S)  # tolerate prose or code fences around the JSON
    reviews = json.loads(match.group(0))["reviews"] if match else []
    ids = {r["id"] for r in rows}
    out = [(rv["id"], rv["verdict"], rv.get("reason", "")) for rv in reviews
           if rv.get("id") in ids and rv.get("verdict") in verdicts]
    return out, cost_of(resp)


def review(model, batch, limit, rejects, workers):
    con = connect()
    con.row_factory = sqlite3.Row
    client = make_client()
    jobs = []
    passing = con.execute("SELECT * FROM traces WHERE verifier='pass' AND review='pending' ORDER BY id LIMIT ?", (limit,)).fetchall()
    jobs += [(passing[i:i + batch], PASS_PROMPT, {"approve": "approved", "flag": "flagged"}) for i in range(0, len(passing), batch)]
    failing = con.execute("SELECT * FROM traces WHERE verifier='fail' AND review='pending' ORDER BY random() LIMIT ?", (rejects,)).fetchall()
    jobs += [(failing[i:i + batch], REJECT_PROMPT, {"agreed": "agreed", "disputed": "disputed"}) for i in range(0, len(failing), batch)]
    if not jobs:
        print("nothing pending")
        return

    t0, spent, done = time.time(), 0.0, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(review_batch, client, model, rows, prompt, verdicts) for rows, prompt, verdicts in jobs]
        for (rows, _, verdicts), fut in zip(jobs, futures):
            results, cost = fut.result()
            spent += cost
            for trace_id, verdict, reason in results:
                con.execute("UPDATE traces SET review=?, review_reason=?, reviewer=?, reviewed_at=? WHERE id=?",
                            (verdicts[verdict], reason, model, time.time(), trace_id))
            done += len(results)
            if len(results) < len(rows):
                print(f"  reviewer skipped {len(rows) - len(results)} of {len(rows)} items; they stay pending")
    con.commit()
    print(f"reviewed {done} traces in {len(jobs)} calls, {time.time() - t0:.0f}s, ${spent:.4f}")


def report():
    con = connect()
    print("verifier  review     traces")
    for v, r, n in con.execute("SELECT verifier, review, COUNT(*) FROM traces GROUP BY 1, 2 ORDER BY 1 DESC, 2"):
        print(f"{v:9s} {r:10s} {n}")
    print(f"\nsafe: {con.execute('SELECT COUNT(*) FROM safe_traces').fetchone()[0]}")
    for label, where in [("Flagged by reviewer", "review='flagged'"), ("Reviewer says the verifier was wrong", "review='disputed'")]:
        rows = con.execute(f"SELECT id, question, final_sql, review_reason FROM traces WHERE {where}").fetchall()
        if rows:
            print(f"\n{label}:")
            for i, q, sql, reason in rows:
                print(f"  #{i} {q[:90]}\n     final: {sql[:140]}\n     why: {reason}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    r = sub.add_parser("review")
    r.add_argument("--model", default="glm-5.3-flash")
    r.add_argument("--batch", type=int, default=8, help="traces per reviewer call")
    r.add_argument("--limit", type=int, default=64, help="verifier-passed traces to review this time")
    r.add_argument("--rejects", type=int, default=0, help="also review this many rejected traces for false rejects")
    r.add_argument("--workers", type=int, default=4)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "review":
        review(args.model, args.batch, args.limit, args.rejects, args.workers)
    else:
        report()
