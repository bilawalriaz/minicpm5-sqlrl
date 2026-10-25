"""Generate new question templates for the skills the student is weakest at.

    python evolve.py --skills anti_join,having --n 2   # propose and validate, no GPU needed
    python evolve.py --list                            # what has been accepted or rejected, and why

Templates are data, never Python code. All parameter combinations (up to 256)
are checked on the verifier suite. A fresh solver checks distinct questions
without seeing reference SQL. Accepted templates are stored under the current
verifier's data directory.
"""

import argparse
import json
import random
import re
import sqlite3
import string
import time
import itertools
import math
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import env
from curate import DB
import httpx

from teacher import BASE_URL, Teacher, _api_key, cost_of, make_client, session_header

TABLE = """
CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    target_skill TEXT NOT NULL,
    skills TEXT NOT NULL,      -- JSON list
    tier INTEGER NOT NULL,
    question TEXT NOT NULL,    -- pattern with {placeholders}
    sql TEXT NOT NULL,         -- pattern with {placeholders}
    params TEXT NOT NULL,      -- JSON {placeholder: [values]}
    status TEXT NOT NULL,      -- accepted | rejected
    reason TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""

PROMPT = """You write training questions for a small text-to-SQL model that is weak at one SQL skill. \
Each question is a template: a natural-language question and the reference SQLite query that answers it, \
with {{placeholders}} for the values that vary.

Database schema:
{schema}

Values that exist in the data (use these exact spellings):
{values}

Target skill: {skill} ({description}).
Existing templates that use it (write different ones: other tables, columns or filters):
{examples}

Write {n} new templates that require the target skill. Rules:
- Every varying value is a {{placeholder}} in both the question and the SQL, with 2 to 12 values in "params" \
taken from the lists above (numbers as numbers). In SQL, quote string placeholders: '{{category}}'.
- The question states exactly which columns to return, any rounding ("rounded to 2 decimal places", and the \
SQL uses ROUND(x, 2)), and "return 0 if none" when an aggregate could be empty (the SQL then uses COALESCE).
- Exactly one correct answer: never ORDER BY ... LIMIT on a value that can tie, such as a count.
- One SQLite SELECT statement. Dates are text 'YYYY-MM-DD'.
- tier: 1 = one table, 2 = a join or grouping, 3 = several of those combined.
{difficulty}
Reply with JSON only:
{{"templates": [{{"name": "snake_case_name", "tier": 2, "skills": ["{skill}", ...], \
"question": "...", "sql": "...", "params": {{"placeholder": [...]}}}}]}}
Allowed skills: {skills}"""


def connect():
    env.ensure_databases()
    con = sqlite3.connect(DB)
    con.execute(TABLE)
    return con


def value_lists():
    con = env.connect()
    names = [r[0] for r in env.execute(con, "SELECT name FROM products ORDER BY name")[0]]
    customers = [r[0] for r in env.execute(con, "SELECT name FROM customers ORDER BY random() LIMIT 15")[0]]
    con.close()
    return "\n".join([
        f"countries and their cities: {json.dumps(env.CITIES)}",
        f"product categories: {env.CATEGORIES}",
        f"order statuses: {env.STATUSES}",
        f"product names: {names}",
        f"some customer names: {customers}",
        "order dates: mostly 2023 to 2025; signup dates: mostly 2022 to 2024 (other fixtures extend the ranges)",
        "prices: 3 to 300; quantities per order line: 1 to 5; orders per customer: about 0 to 14",
    ])


def examples(skill):
    rng = random.Random(0)
    lines = []
    for name, skills in env.SKILLS.items():
        if skill in skills and name not in env.HELDOUT_TEMPLATES:
            q, sql = env.TEMPLATES[name][1](rng)
            lines.append(f"- {name}: {q}\n  SQL: {sql}")
    # Generated templates accumulate; a sample keeps the prompt short and varied.
    return "\n".join(rng.sample(lines, min(8, len(lines)))) or "(none yet)"


STEPS = """You write training questions for a small text-to-SQL model. It keeps failing this question type:

Question: {question}
Reference SQL: {sql}

Database schema:
{schema}

Values that exist in the data (use these exact spellings):
{values}

Write {n} simpler templates that each teach one step this question needs (for example one join, one \
filter, one aggregate, or one subquery level), so the model can learn the steps before the whole. Use \
other filters or columns than the failing question; do not restate it. Rules:
- Every varying value is a {{placeholder}} in both the question and the SQL, with 2 to 12 values in "params" \
taken from the lists above (numbers as numbers). In SQL, quote string placeholders: '{{category}}'.
- The question states exactly which columns to return, any rounding (and the SQL uses ROUND(x, 2)), and \
"return 0 if none" when an aggregate could be empty (the SQL then uses COALESCE).
- Exactly one correct answer: never ORDER BY ... LIMIT on a value that can tie, such as a count.
- One SQLite SELECT statement. Dates are text 'YYYY-MM-DD'. tier: 1 = one table, 2 = a join or grouping.

Reply with JSON only:
{{"templates": [{{"name": "snake_case_name", "tier": 1, "skills": [...], \
"question": "...", "sql": "...", "params": {{"placeholder": [...]}}}}]}}
Allowed skills: {skills}"""

REVIEW = """You review a training question for a small text-to-SQL model. A wrong or ambiguous question \
teaches the model a wrong habit, so reject anything doubtful.

Database schema:
{schema}

Question: {question}
Reference SQL: {sql}
First rows of the reference result ({count} rows): {rows}

Approve only if all hold:
1. The question has exactly one reasonable reading, and the SQL computes that reading: the right filters, \
joins, grouping and aggregation, with nothing extra.
2. The question names the columns to return and the SQL returns exactly those, in that order.
3. Any rounding, "return 0 if none", or tie-breaking the SQL needs is stated in the question.
4. The wording is natural English a user might write.

Reply with JSON only: {{"verdict": "approve" | "reject", "reason": "<one sentence>"}}"""

COMBO = """
- Each template must use all of these skills together: {combo}."""

# Burst prompts ask for harder compositions when training groups stop carrying signal.
HARD = """- Make each question harder than the existing templates: tier 3, combining the target skill with at \
least two other listed skills, and needing a subquery or a derived table. Keep exactly one correct answer."""


def complete(client, model, prompt, max_tokens=8192):
    """Returns (text, dollars). Claude models use the gateway's Messages endpoint."""
    if model.startswith("claude-"):
        resp = httpx.post(f"{BASE_URL}/messages", timeout=180, json={
            "model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
            headers={"x-api-key": _api_key(), "anthropic-version": "2023-06-01",
                     "User-Agent": "sqlrl-teacher/0.1", **session_header()})
        resp.raise_for_status()
        body = resp.json()
        if body.get("stop_reason") == "max_tokens":
            raise ValueError(f"{model} reply hit the {max_tokens}-token limit")
        text = "".join(part.get("text", "") for part in body["content"])
        return text, (body["usage"].get("opencode") or {}).get("list_cost", 0.0)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low", max_tokens=max_tokens, extra_headers=session_header(),
    )
    return resp.choices[0].message.content or "", cost_of(resp)


def review(client, model, spec, question, sql, con):
    """Returns (approved, reason, dollars) from a reviewer model that sees the reference SQL."""
    rows = env.execute(con, sql)[0]
    text, cost = complete(client, model, REVIEW.format(schema=env.SCHEMA, question=question, sql=sql,
                                                       count=len(rows), rows=rows[:5]), max_tokens=1024)
    start = text.find("{")
    verdict = json.JSONDecoder().raw_decode(text, start)[0] if start >= 0 else {}
    return verdict.get("verdict") == "approve", str(verdict.get("reason", "no verdict"))[:200], cost


def propose(client, model, skill, n, hard=False, failing=None, combo=None):
    """failing: (name, question, sql) of a template the student keeps failing; the proposals
    are then stepping stones toward it, targeting its first skill."""
    if failing:
        prompt = STEPS.format(question=failing[1], sql=failing[2], schema=env.SCHEMA, values=value_lists(), n=n,
                              skills=sorted(env.SKILL_DESCRIPTIONS))
    else:
        prompt = PROMPT.format(
            schema=env.SCHEMA, values=value_lists(), skill=skill, description=env.SKILL_DESCRIPTIONS[skill],
            examples=examples(skill), n=n, skills=sorted(env.SKILL_DESCRIPTIONS),
            difficulty=(HARD if hard else "") + (COMBO.format(combo=", ".join(combo)) if combo else ""),
        )
    text, cost = complete(client, model, prompt)
    start = text.find("{")
    # raw_decode reads the first JSON object and ignores trailing prose or code fences.
    specs = json.JSONDecoder().raw_decode(text, start)[0].get("templates", []) if start >= 0 else []
    for spec in specs:
        if isinstance(spec, dict):
            spec["target_skill"], spec["model"] = skill, model
            if failing and isinstance(spec.get("skills"), list) and skill not in spec["skills"]:
                spec["target_skill"] = spec["skills"][0] if spec["skills"] else skill
    return [spec for spec in specs if isinstance(spec, dict)], cost


def check_structure(spec):
    """Returns a reason the spec is malformed, or None."""
    if not all(isinstance(spec.get(k), str) and spec[k] for k in ("name", "question", "sql")):
        return "missing name, question or sql"
    if spec.get("tier") not in (1, 2, 3):
        return "tier must be 1, 2 or 3"
    skills = spec.get("skills")
    if not isinstance(skills, list) or not skills or not set(skills) <= set(env.SKILL_DESCRIPTIONS):
        return f"unknown skills {skills}"
    if spec["target_skill"] not in skills:
        return "does not use the target skill"
    params = spec.get("params")
    if not isinstance(params, dict) or not params:
        return "no params"
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f is not None}
    try:
        q_fields, sql_fields = fields(spec["question"]), fields(spec["sql"])
    except ValueError:
        return "malformed placeholders"
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", field) for field in set(params)):
        return "parameter names must be plain identifiers"
    if q_fields != set(params) or sql_fields != set(params):
        return f"placeholders differ: question {sorted(q_fields)}, sql {sorted(sql_fields)}, params {sorted(params)}"
    for k, values in params.items():
        if not isinstance(values, list) or not 2 <= len(values) <= 12:
            return f"param {k} needs 2 to 12 values"
        if any(isinstance(v, bool) or not isinstance(v, (str, int, float)) or (isinstance(v, float) and not math.isfinite(v)) or (isinstance(v, str) and re.search(r"['\";{}]", v)) for v in values):
            return f"param {k} has a value that is not plain text or a number"
    if spec["name"] in env.TEMPLATES or not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", spec["name"]):
        return "name is taken or not snake_case"
    return None


def signature(rows):
    return frozenset(env.Counter(tuple(env._norm(v) for v in row) for row in rows).items())


def eval_signatures(cons):
    """Results of every evaluation question on the training fixtures. A generated question
    with identical results everywhere asks the same thing in other words."""
    out = set()
    for name in env.BUILTIN:
        for q, sql in (env._questions(name) if name in env.HELDOUT_TEMPLATES else env.reserved_questions(name)):
            out.add(tuple(signature(env.execute(con, sql)[0]) for con in cons))
    return out


def _top_level(sql, start, stops):
    """End of the clause starting at `start`: a stop keyword at parenthesis depth 0,
    the closing parenthesis of an enclosing subquery, or the end of the statement."""
    depth, i = 0, start
    while i < len(sql):
        c = sql[i]
        if c == "'":
            i = sql.index("'", i + 1)
        elif c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return i
            depth -= 1
        elif depth == 0 and (m := re.match(r"\s+(" + stops + r")\b", sql[i:], re.I)):
            return i
        i += 1
    return len(sql.rstrip().rstrip(";"))


def ablations(sql):
    """The reference without HAVING and without each top-level condition of the outer WHERE."""
    out = {}
    if m := re.search(r"\s+HAVING\s", sql, re.I):
        end = _top_level(sql, m.end(), r"ORDER\s+BY|LIMIT|UNION|EXCEPT|INTERSECT")
        out["HAVING"] = sql[:m.start()] + sql[end:]
    if m := re.search(r"\s+WHERE\s", sql, re.I):
        end = _top_level(sql, m.end(), r"GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|UNION|EXCEPT|INTERSECT")
        parts, start = [], m.end()
        while (cut := _top_level(sql[:end], start, r"AND")) < end:
            parts.append(sql[start:cut].strip())
            start = cut + re.match(r"\s+AND\s+", sql[cut:], re.I).end()
        parts.append(sql[start:end].strip())
        for i, part in enumerate(parts):
            rest = parts[:i] + parts[i + 1:]
            out[f"WHERE condition {part!r}"] = sql[:m.start()] + (" WHERE " + " AND ".join(rest) if rest else "") + sql[end:]
    return out


def answer_values(rows):
    return {env._norm(v) for row in rows for v in row if v is not None}


def heldout_values(cons):
    """Values in each held-out question's answer, per training fixture."""
    return [(q, [answer_values(env.execute(con, sql)[0]) for con in cons])
            for name in env.HELDOUT_TEMPLATES for q, sql in env._questions(name)]


def _distinctive(v):
    """Values unlikely to recur by chance: non-integer numbers and dates. Counts, ids and
    category names appear in many unrelated answers."""
    return (isinstance(v, float) and v != int(v)) or (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v))


def contained(question_values, heldout):
    """A held-out question whose answer contains this one's values on every fixture (the
    question asks for part of a held-out answer), or whose distinctive answer values all
    appear in this one's answer (the question's answer includes the held-out answer)."""
    for q, held in heldout:
        if all(a and a <= b for a, b in zip(question_values, held)) and any(a - {0, 1} for a in question_values):
            return q
        if all(b and b <= a for a, b in zip(question_values, held)) and all(
                any(_distinctive(v) for v in b) for b in held):
            return q
    return None


def normalise(sql):
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def validate(spec, teacher, known_sql, reviewer=None):
    """Returns (accepted, reason). reviewer: optional (client, model) that must approve a
    sample question after the data checks and the solver pass."""
    if reason := check_structure(spec):
        return False, reason
    if normalise(spec["sql"]) in known_sql:
        return False, "same SQL as an accepted template"
    cons = env.verifier_connections()
    try:
        keys = list(spec["params"])
        combinations = math.prod(len(spec["params"][k]) for k in keys)
        if combinations > 256:
            return False, "more than 256 parameter combinations; narrow the template"
        questions, effective = {}, set()
        held_out, held_values = eval_signatures(cons), heldout_values(cons)
        for values_tuple in itertools.product(*(spec["params"][k] for k in keys)):
            values = dict(zip(keys, values_tuple))
            q, sql = spec["question"].format(**values), spec["sql"].format(**values)
            results = []
            for i, con in enumerate(cons):
                rows, err = env.execute(con, sql, require_table=True, deterministic=True)
                if err:
                    return False, f"reference SQL fails on database {i}: {err}"
                if env.tie_dependent(con, sql):
                    return False, f"reference result depends on tie order on database {i}: {q}"
                results.append(signature(rows))
            if not any(v is not None for row in env.execute(cons[0], sql)[0] for v in row):
                return False, f"empty or NULL answer on the primary database: {q}"
            if len(set(results)) == 1:
                return False, f"same answer on every database, so a hard-coded value would pass: {q}"
            if tuple(results) in held_out:
                return False, f"same results as an evaluation question: {q}"
            if overlap := contained([answer_values(env.execute(con, sql)[0]) for con in cons], held_values):
                return False, f"answers overlap the held-out question {overlap!r}: {q}"
            for label, variant in ablations(spec["sql"]).items():
                rows = [env.execute(con, variant.format(**values))[0] for con in cons]
                if any(r is None or signature(r) != res for r, res in zip(rows, results)):
                    effective.add(label)
            questions.setdefault(q, sql)
        if vacuous := sorted(set(ablations(spec["sql"])) - effective):
            return False, f"removing {vacuous[0]} never changes the answer"
        # Validate different parameter choices rather than repeated random draws.
        selected = list(questions.items())[:3]
        solved = sum(teacher.solve(env.Task(spec["name"], spec["tier"], q, sql))["correct"]
                     for q, sql in selected)
        needed = min(2, len(selected))
        if solved < needed:
            return False, f"fresh solver got {solved}/{len(selected)} distinct questions right"
        note = ""
        if reviewer:
            approved, why, cost = review(*reviewer, spec, *selected[0], cons[0])
            teacher.spent += cost
            if not approved:
                return False, f"reviewer {reviewer[1]} rejected: {why}"
            note = f"; {reviewer[1]} approved"
        return True, f"all {combinations} combinations execute; solver got {solved}/{len(selected)}{note}"
    finally:
        for con in cons:
            con.close()


def evolve(skills, n, model="glm-5.3-flash", workers=4, solver="glm-5.3-flash", hard=False, failing=None,
           reviewer=None, combos=None):
    """Propose n templates per skill, validate them, store every verdict. Returns accepted specs.
    model may be a comma-separated list; skills rotate through it. The solver that checks
    proposals should differ from the proposers, so acceptance is not one model agreeing with itself.
    failing: optional list of (name, question, sql), one per skill, for stepping-stone proposals.
    reviewer: model that must approve each accepted template. combos: optional list of skill
    tuples, one per skill, that each proposal must combine."""
    models = model.split(",")
    client, teacher = make_client(), Teacher(solver, workers=1)
    con = connect()
    known_sql = {normalise(sql) for (sql,) in con.execute("SELECT sql FROM templates WHERE status='accepted'")}

    def safe_propose(i, skill):
        try:
            return propose(client, models[i % len(models)], skill, n, hard, failing[i] if failing else None,
                           combos[i] if combos else None)
        except Exception as exc:  # one model's API failure should not discard the others' proposals
            print(f"  evolve: {models[i % len(models)]} failed for {skill}: {type(exc).__name__}: {str(exc)[:120]}")
            return [], 0.0

    with ThreadPoolExecutor(workers) as pool:
        proposals = list(pool.map(safe_propose, range(len(skills)), skills))
        specs = [spec for batch, _ in proposals for spec in batch]
        verdicts = list(pool.map(lambda spec: validate(spec, teacher, known_sql,
                                                       (client, reviewer) if reviewer else None), specs))
    accepted = []
    for spec, (ok, reason) in zip(specs, verdicts):
        if ok and (normalise(spec["sql"]) in known_sql or spec["name"] in {x["name"] for x in accepted}):
            ok, reason = False, "duplicate accepted proposal"
        if ok:
            accepted.append(spec)
            known_sql.add(normalise(spec["sql"]))
        con.execute(
            "INSERT OR IGNORE INTO templates (name, target_skill, skills, tier, question, sql, params, status, reason,"
            " model, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (str(spec.get("name")), spec["target_skill"], json.dumps(spec.get("skills")), spec.get("tier") or 0,
             str(spec.get("question")), str(spec.get("sql")), json.dumps(spec.get("params")),
             "accepted" if ok else "rejected", reason, spec.get("model", model), time.time()),
        )
    con.commit()
    con.close()
    spent = sum(cost for _, cost in proposals) + teacher.spent
    teacher.close()
    client.close()
    return accepted, {"proposed": len(specs), "accepted": len(accepted), "spent": spent}


def revalidate():
    """Apply the current data checks to accepted templates; earlier solver results stand."""
    con = connect()
    solved = SimpleNamespace(solve=lambda task: {"correct": True})
    for name, target, skills, tier, question, sql, params in con.execute(
            "SELECT name, target_skill, skills, tier, question, sql, params FROM templates WHERE status='accepted'").fetchall():
        # A placeholder name: the stored name is already registered and may be at the length limit.
        spec = {"name": "revalidation_candidate", "target_skill": target, "skills": json.loads(skills), "tier": tier,
                "question": question, "sql": sql, "params": json.loads(params)}
        ok, reason = validate(spec, solved, set())
        if not ok:
            con.execute("UPDATE templates SET status='rejected', reason=? WHERE name=?", (f"revalidation: {reason}", name))
            print(f"rejected {name}: {reason}")
    con.commit()
    con.close()


def uncovered_combinations(k, seed=None):
    """Skill pairs and triples that no trainable template combines yet, pairs first."""
    have = [set(env.SKILLS[t]) for t in env.TEMPLATES if t not in env.HELDOUT_TEMPLATES]
    skills = sorted(env.SKILL_DESCRIPTIONS)
    out = []
    for size in (2, 3):
        combos = [c for c in itertools.combinations(skills, size) if not any(set(c) <= h for h in have)]
        random.Random(seed).shuffle(combos)
        out += combos
    return out[:k]


def load_accepted():
    """Register every accepted template with env. Returns their names."""
    con = connect()
    rows = con.execute("SELECT name, tier, skills, question, sql, params FROM templates WHERE status='accepted'").fetchall()
    con.close()
    names = []
    for name, tier, skills, question, sql, params in rows:
        env.add_template({"name": name, "tier": tier, "skills": json.loads(skills), "question": question,
                          "sql": sql, "params": json.loads(params)})
        names.append(name)
    return names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", help="comma-separated, e.g. anti_join,having, or all")
    ap.add_argument("--rounds", type=int, default=1, help="repeat proposal and validation")
    ap.add_argument("--n", type=int, default=2, help="templates to propose per skill")
    ap.add_argument("--model", default="glm-5.3-flash", help="proposer, or a comma-separated list to rotate")
    ap.add_argument("--solver", default="glm-5.3-flash", help="model that must solve proposals without the reference")
    ap.add_argument("--hard", action="store_true", help="ask for tier-3 compositions with subqueries")
    ap.add_argument("--reviewer", help="model that must approve each template, e.g. claude-haiku-5-5")
    ap.add_argument("--coverage", type=int, default=0,
                    help="instead of --skills, propose for this many skill pairs and triples not yet combined")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--revalidate", action="store_true", help="recheck accepted templates against the current data checks")
    args = ap.parse_args()
    if args.revalidate:
        revalidate()
    elif args.list:
        for status, name, skill, reason, q in connect().execute(
                "SELECT status, name, target_skill, reason, question FROM templates ORDER BY id"):
            print(f"{status:8s} {name} [{skill}] {reason}\n         {q}")
    else:
        for _ in range(args.rounds):
            load_accepted()  # so names and examples include earlier generations
            combos = uncovered_combinations(args.coverage) if args.coverage else None
            skills = ([c[0] for c in combos] if combos else
                      sorted(env.SKILL_DESCRIPTIONS) if args.skills == "all" else args.skills.split(","))
            t0 = time.time()
            accepted, stats = evolve(skills, args.n, args.model, solver=args.solver, hard=args.hard,
                                     reviewer=args.reviewer, combos=combos)
            print(f"{stats['accepted']} of {stats['proposed']} accepted in {time.time() - t0:.0f}s, ${stats['spent']:.4f}")
            for spec in accepted:
                print(f"  + {spec['name']} {spec['skills']}: {spec['question']}")
