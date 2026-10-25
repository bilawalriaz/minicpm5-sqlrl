"""SQLite agent environment: a synthetic shop database, templated questions with
gold SQL, a read-only tool executor, and a verified SQL submission.

Every question is generated from a template, so the gold answer is known by
construction and the verifier compares the submitted SQL with the reference on multiple
versioned databases.
"""

import random
import re
import sqlite3
from collections import Counter
from contextlib import closing
from functools import lru_cache
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path

VERIFIER_VERSION = "sql-v2.3"
PROTOCOL_VERSION = "submit-sql-v2"
# sql-v2.3 changes only the instruction budget, so it keeps the sql-v2.2 fixtures and template store.
DATA_VERSION = "sql-v2.2"
DATA_DIR = Path(__file__).parent / "data" / DATA_VERSION
DB_PATH = DATA_DIR / "train.db"
ALT_DB_PATHS = [DATA_DIR / f"train_alt{i}.db" for i in (1, 2, 3)]
EVAL_DB_PATH = DATA_DIR / "eval.db"
EVAL_ALT_DB_PATHS = [DATA_DIR / f"eval_alt{i}.db" for i in (1, 2, 3)]
MAX_RESULT_ROWS = 10_000
_db_lock = threading.Lock()

CITIES = {
    "UK": ["London", "Manchester", "Leeds", "Glasgow"],
    "France": ["Paris", "Lyon", "Marseille"],
    "Germany": ["Berlin", "Munich", "Hamburg"],
    "USA": ["New York", "Chicago", "Austin", "Seattle"],
    "Japan": ["Tokyo", "Osaka"],
}
CATEGORIES = ["Books", "Electronics", "Garden", "Toys", "Kitchen", "Sports"]
STATUSES = ["shipped", "pending", "cancelled", "returned"]
FIRST = "Ava Ben Chloe Dan Ella Finn Grace Hugo Isla Jack Kai Lena Max Nora Omar Priya Quinn Rosa Sam Tara Uma Victor Wren Yusuf Zara".split()
LAST = "Ahmed Brown Chen Dubois Evans Fischer Garcia Hughes Ito Jones Khan Lopez Muller Novak Okafor Patel Rossi Smith Tanaka Weber".split()
ADJ = "Red Blue Smart Tiny Mega Eco Turbo Classic Pro Lite".split()
NOUN = {
    "Books": ["Novel", "Atlas", "Cookbook", "Comic"],
    "Electronics": ["Headphones", "Charger", "Speaker", "Keyboard"],
    "Garden": ["Trowel", "Hose", "Planter", "Seeds"],
    "Toys": ["Robot", "Puzzle", "Kite", "Blocks"],
    "Kitchen": ["Kettle", "Pan", "Blender", "Knife"],
    "Sports": ["Ball", "Racket", "Yoga Mat", "Helmet"],
}

SCHEMA = """CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, country TEXT, signup_date TEXT);  -- dates are 'YYYY-MM-DD'
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), order_date TEXT, status TEXT);  -- status in ('shipped','pending','cancelled','returned')
CREATE TABLE order_items (order_id INTEGER REFERENCES orders(id), product_id INTEGER REFERENCES products(id), quantity INTEGER, unit_price REAL);"""


def build_db(path=DB_PATH, seed=0, *, sparse=False, complete_catalog=False, orderable_parity=None,
             adversarial=False):
    """Build one fixture. orderable_parity limits orders to products at even (0) or odd (1)
    positions of the complete catalog, so every product name has zero sales on some fixture.
    adversarial adds cases that separate plausible wrong SQL from the reference."""
    rng = random.Random(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    signup_years = [2021, 2022, 2023, 2024, 2025] if adversarial else [2022, 2023, 2024]
    order_years = [2022, 2023, 2024, 2025, 2026] if adversarial else [2023, 2024, 2025]

    names = set()
    while len(names) < 200:
        names.add(f"{rng.choice(FIRST)} {rng.choice(LAST)}")
    for i, name in enumerate(sorted(names), 1):
        if seed % 2 and i <= 3:
            name = "Alex Smith"  # Names are not unique identifiers.
        country = rng.choice(list(CITIES))
        date = f"{rng.choice(signup_years)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        con.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (i, name, rng.choice(CITIES[country]), country, date))
    if adversarial:
        # Two different customers share a name in every city: COUNT(DISTINCT name) is not a customer count.
        for city in [c for cs in CITIES.values() for c in cs]:
            ids = [r[0] for r in con.execute("SELECT id FROM customers WHERE city=? ORDER BY id", (city,))]
            if len(ids) >= 2:
                con.execute("UPDATE customers SET name=(SELECT name FROM customers WHERE id=?) WHERE id=?", (ids[0], ids[1]))

    pid = 0
    for cat in CATEGORIES:
        for noun in NOUN[cat]:
            for adj in (ADJ if complete_catalog else rng.sample(ADJ, 2)):
                pid += 1
                category = rng.choice(CATEGORIES) if seed % 4 == 2 else cat
                price = round(rng.uniform(3, 300), 2)
                position = (pid - 1) % (len(NOUN[cat]) * (len(ADJ) if complete_catalog else 2))
                if adversarial and position < 4:
                    # Exact thresholds of cheap_products, with a repeated price per category.
                    price = (50.0, 50.0, 100.0, 150.0)[position]
                con.execute("INSERT INTO products VALUES (?,?,?,?)", (pid, f"{adj} {noun}", category, price))

    if orderable_parity is not None:
        orderable = [p for p in range(1, pid + 1) if (p - 1) % 2 == orderable_parity]
    elif complete_catalog:
        orderable = list(range(1, pid + 1))
    else:
        # Leave some products unordered so "never ordered" questions have answers.
        orderable = rng.sample(range(1, pid + 1), pid - 6)
    price_of = dict(con.execute("SELECT id, price FROM products"))

    def add_order(customer, date, status, lines):
        oid = con.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM orders").fetchone()[0]
        con.execute("INSERT INTO orders VALUES (?,?,?,?)", (oid, customer, date, status))
        for prod, qty, unit in lines:
            con.execute("INSERT INTO order_items VALUES (?,?,?,?)", (oid, prod, qty, unit))
        return oid

    for _ in range(80 if sparse else 1200):
        date = f"{rng.choice(order_years)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        status = rng.choices(STATUSES, weights=[70, 15, 10, 5])[0]
        add_order(rng.randint(1, 200), date, status,
                  [(p, rng.randint(1, 5), round(price_of[p] * rng.uniform(0.65, 1.35), 2))
                   for p in rng.sample(orderable, rng.randint(1, 4))])
    if complete_catalog and orderable_parity is None:
        # Every product sells, with repeated quantities, so COUNT(*) and SUM(DISTINCT quantity)
        # differ from SUM(quantity) for every built-in product name.
        for prod in orderable:
            for _ in range(2):
                con.execute("INSERT INTO order_items VALUES (?,?,?,?)",
                            (rng.randint(1, 80 if sparse else 1200), prod, 2, round(price_of[prod] * 0.8, 2)))
    if adversarial:
        _add_adversarial_orders(con, rng, orderable, price_of, add_order)
    con.commit()
    con.close()


def _add_adversarial_orders(con, rng, orderable, price_of, add_order):
    city, country = "London", "UK"
    new_id = lambda: con.execute("SELECT MAX(id) + 1 FROM customers").fetchone()[0]
    expensive = sorted(orderable, key=price_of.get)[-3:]
    cheap = sorted(orderable, key=price_of.get)[:3]
    # Same-name heavy buyers: both exceed every order threshold and both are top spenders,
    # so SELECT DISTINCT name or GROUP BY name changes the answer.
    for _ in range(2):
        cid = new_id()
        con.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (cid, "Nora Khan", city, country, "2023-03-03"))
        for k in range(14):
            add_order(cid, f"{2023 + k % 3}-{k % 12 + 1:02d}-15", "shipped",
                      [(p, 5, round(price_of[p], 2)) for p in expensive])
    # Same-name light buyers: seven orders each, fourteen together.
    for _ in range(2):
        cid = new_id()
        con.execute("INSERT INTO customers VALUES (?,?,?,?,?)", (cid, "Sam Weber", "Paris", "France", "2024-05-05"))
        for k in range(7):
            add_order(cid, f"2024-{k + 1:02d}-10", "shipped", [(cheap[0], 1, round(price_of[cheap[0]], 2))])

    def ranked(year, measure, shipped=True):
        where = "o.status = 'shipped' AND " if shipped else ""
        return [r[0] for r in con.execute(
            f"SELECT p.category FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
            f"WHERE {where}o.order_date LIKE '{year}-%' GROUP BY p.category ORDER BY {measure} DESC")]

    revenue = "SUM(oi.quantity * oi.unit_price)"
    for year in (2023, 2024, 2025):
        cats = ranked(year, revenue)
        totals = dict(con.execute(
            "SELECT p.category, SUM(oi.quantity * oi.unit_price) FROM order_items oi JOIN orders o ON o.id = oi.order_id "
            "JOIN products p ON p.id = oi.product_id WHERE o.status = 'shipped' AND o.order_date LIKE ? GROUP BY p.category",
            (f"{year}-%",)))
        # A large cancelled sale moves the runner-up ahead only when cancelled orders count.
        runner = cats[1]
        gap = totals[cats[0]] - totals[runner] + 2000
        items = [p for p in orderable if con.execute("SELECT category FROM products WHERE id=?", (p,)).fetchone()[0] == runner]
        prod = max(items, key=price_of.get)
        while gap > 0:
            add_order(rng.randint(1, 200), f"{year}-06-15", "cancelled", [(prod, 5, price_of[prod])])
            gap -= 5 * price_of[prod]
        # Many cheap shipped units in the weakest category win on quantity and line count, not revenue.
        weakest = cats[-1]
        items = [p for p in orderable if con.execute("SELECT category FROM products WHERE id=?", (p,)).fetchone()[0] == weakest]
        prod = min(items, key=price_of.get)
        while ranked(year, "SUM(oi.quantity)")[0] != weakest or ranked(year, "COUNT(*)")[0] != weakest:
            add_order(rng.randint(1, 200), f"{year}-07-15", "shipped", [(prod, 5, price_of[prod])])
        assert ranked(year, revenue)[0] == cats[0] != ranked(year, revenue, shipped=False)[0]
        assert ranked(year, revenue)[0] != weakest, "cheap units changed the revenue leader"


# ---------------------------------------------------------------------------
# Question templates. Each returns (question, gold_sql). Tier is a rough
# difficulty label used for per-tier reporting and curriculum.
# ---------------------------------------------------------------------------

def _city(r):
    return r.choice([c for cs in CITIES.values() for c in cs])


TEMPLATES = {
    # tier 1: one table
    "count_city": (1, lambda r: (c := _city(r)) and (
        f"How many customers live in {c}?",
        f"SELECT COUNT(*) FROM customers WHERE city = '{c}'")),
    "count_status": (1, lambda r: (s := r.choice(STATUSES)) and (
        f"How many orders have the status '{s}'?",
        f"SELECT COUNT(*) FROM orders WHERE status = '{s}'")),
    "cheap_products": (1, lambda r: (c := r.choice(CATEGORIES), p := r.choice([50, 100, 150])) and (
        f"List the names of all {c} products that cost less than {p}. Return only the product names.",
        f"SELECT name FROM products WHERE category = '{c}' AND price < {p}")),
    "priciest_in_cat": (1, lambda r: (c := r.choice(CATEGORIES)) and (
        f"What is the name of the most expensive product in the {c} category?",
        f"SELECT name FROM products WHERE category = '{c}' ORDER BY price DESC LIMIT 1")),
    "signups_year": (1, lambda r: (y := r.choice([2022, 2023, 2024]), k := r.choice(list(CITIES))) and (
        f"How many customers from {k} signed up in {y}?",
        f"SELECT COUNT(*) FROM customers WHERE country = '{k}' AND signup_date LIKE '{y}-%'")),
    # tier 2: joins and aggregates
    "orders_by_country": (2, lambda r: (k := r.choice(list(CITIES))) and (
        f"How many orders were placed by customers from {k}?",
        f"SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.country = '{k}'")),
    "revenue_month": (2, lambda r: (y := r.choice([2023, 2024, 2025]), m := r.randint(1, 12)) and (
        f"What was the total revenue (sum of quantity * unit_price) from shipped orders placed in {y}-{m:02d}? Round to 2 decimal places.",
        f"SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM order_items oi JOIN orders o ON oi.order_id = o.id WHERE o.status = 'shipped' AND o.order_date LIKE '{y}-{m:02d}-%'")),
    "avg_price_by_cat": (2, lambda r: (
        "For each product category, what is the average product price? Return the category and the average rounded to 2 decimal places.",
        "SELECT category, ROUND(AVG(price), 2) FROM products GROUP BY category")),
    "distinct_buyers_cat": (2, lambda r: (c := r.choice(CATEGORIES)) and (
        f"How many distinct customers have ordered at least one {c} product?",
        f"SELECT COUNT(DISTINCT o.customer_id) FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE p.category = '{c}'")),
    "units_of_product": (2, lambda r: (n := f"{r.choice(ADJ)} {r.choice(NOUN[r.choice(CATEGORIES)])}") and (
        f"How many units of the product named '{n}' have been ordered in total, across all orders? Return 0 if none (or if no such product exists).",
        f"SELECT COALESCE(SUM(oi.quantity), 0) FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.name = '{n}'")),
    # tier 3: grouping with filters, anti-joins, top-k
    "frequent_customers": (3, lambda r: (n := r.choice([9, 10, 11, 12])) and (
        f"Which customers have placed more than {n} orders? Return only their names.",
        f"SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.id HAVING COUNT(*) > {n}")),
    "never_ordered": (3, lambda r: (
        "Which products have never been ordered? Return only the product names.",
        "SELECT name FROM products WHERE id NOT IN (SELECT product_id FROM order_items)")),
    "top_spenders": (3, lambda r: (k := r.choice([3, 5])) and (
        f"Who are the top {k} customers by total amount spent (sum of quantity * unit_price) on orders that were not cancelled? Return only their names.",
        f"SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id JOIN order_items oi ON oi.order_id = o.id WHERE o.status != 'cancelled' GROUP BY c.id ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT {k}")),
    "best_category_year": (3, lambda r: (y := r.choice([2023, 2024, 2025])) and (
        f"Which product category had the highest revenue (sum of quantity * unit_price) from shipped orders in {y}? Return only the category.",
        f"SELECT p.category FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id WHERE o.status = 'shipped' AND o.order_date LIKE '{y}-%' GROUP BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1")),
    "churned": (3, lambda r: (y := r.choice([2023, 2024])) and (
        f"How many customers placed at least one order in {y} but no orders in {y + 1}?",
        f"SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM orders WHERE order_date LIKE '{y}-%' EXCEPT SELECT customer_id FROM orders WHERE order_date LIKE '{y + 1}-%')")),
    # Held-out evaluation templates (never trained). Each combines skills that trained templates teach.
    "lines_per_order_status": (2, lambda r: (s := r.choice(STATUSES)) and (
        f"On average, how many order lines (rows in order_items) does an order with status '{s}' have? Round to 2 decimal places.",
        f"SELECT ROUND(AVG(n), 2) FROM (SELECT COUNT(*) AS n FROM order_items oi JOIN orders o ON o.id = oi.order_id WHERE o.status = '{s}' GROUP BY o.id)")),
    "repeat_buyers_cat": (3, lambda r: (c := r.choice(CATEGORIES)) and (
        f"How many customers have placed at least two different orders that each contain a {c} product?",
        f"SELECT COUNT(*) FROM (SELECT o.customer_id FROM orders o JOIN order_items oi ON oi.order_id = o.id JOIN products p ON p.id = oi.product_id WHERE p.category = '{c}' GROUP BY o.customer_id HAVING COUNT(DISTINCT o.id) >= 2)")),
    "country_revenue_year": (3, lambda r: (k := r.choice(list(CITIES)), y := r.choice([2023, 2024, 2025])) and (
        f"What was the total revenue (sum of quantity * unit_price) from shipped orders placed in {y} by customers from {k}? Round to 2 decimal places.",
        f"SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) FROM order_items oi JOIN orders o ON o.id = oi.order_id JOIN customers c ON c.id = o.customer_id WHERE o.status = 'shipped' AND o.order_date LIKE '{y}-%' AND c.country = '{k}'")),
    "idle_customers_country": (3, lambda r: (k := r.choice(list(CITIES))) and (
        f"How many customers from {k} have never placed an order?",
        f"SELECT COUNT(*) FROM customers c WHERE c.country = '{k}' AND NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id)")),
    "status_counts_year": (2, lambda r: (y := r.choice([2023, 2024, 2025])) and (
        f"For each order status, how many orders were placed in {y}? Return the status and the number of orders.",
        f"SELECT status, COUNT(*) FROM orders WHERE order_date LIKE '{y}-%' GROUP BY status")),
    "large_orders": (2, lambda r: (n := r.choice([8, 10, 12])) and (
        f"How many orders contain more than {n} units in total (sum of quantity over the order's lines)?",
        f"SELECT COUNT(*) FROM (SELECT order_id FROM order_items GROUP BY order_id HAVING SUM(quantity) > {n})")),
    "first_order_city": (2, lambda r: (c := _city(r)) and (
        f"On what date was the earliest order placed by a customer living in {c}? Return the date.",
        f"SELECT MIN(o.order_date) FROM orders o JOIN customers c ON c.id = o.customer_id WHERE c.city = '{c}'")),
}


# What each template exercises. The curriculum tracks solve rates per skill, and
# new templates are generated for the weakest ones.
SKILLS = {
    "count_city": ("filter", "aggregate"),
    "count_status": ("filter", "aggregate"),
    "cheap_products": ("filter",),
    "priciest_in_cat": ("filter", "top_k"),
    "signups_year": ("filter", "date_filter", "aggregate"),
    "orders_by_country": ("join", "filter", "aggregate"),
    "revenue_month": ("join", "date_filter", "aggregate", "rounding"),
    "avg_price_by_cat": ("group_by", "rounding"),
    "distinct_buyers_cat": ("multi_join", "distinct", "aggregate"),
    "units_of_product": ("join", "null_handling", "aggregate"),
    "frequent_customers": ("join", "group_by", "having"),
    "never_ordered": ("anti_join",),
    "top_spenders": ("multi_join", "group_by", "top_k"),
    "best_category_year": ("multi_join", "group_by", "top_k", "date_filter"),
    "churned": ("set_ops", "date_filter"),
    "lines_per_order_status": ("join", "filter", "group_by", "aggregate", "rounding"),
    "repeat_buyers_cat": ("multi_join", "group_by", "having", "distinct"),
    "country_revenue_year": ("multi_join", "date_filter", "aggregate", "rounding"),
    "idle_customers_country": ("anti_join", "filter", "aggregate"),
    "status_counts_year": ("group_by", "date_filter", "aggregate"),
    "large_orders": ("group_by", "having", "aggregate"),
    "first_order_city": ("join", "filter", "aggregate"),
}
SKILL_DESCRIPTIONS = {
    "filter": "WHERE conditions on one table",
    "aggregate": "COUNT / SUM / AVG",
    "top_k": "ORDER BY ... LIMIT to pick the highest or lowest",
    "date_filter": "filtering 'YYYY-MM-DD' text dates by year, month or range",
    "join": "joining two tables on a foreign key",
    "multi_join": "joining three or more tables",
    "group_by": "GROUP BY with an aggregate per group",
    "having": "HAVING to filter groups",
    "rounding": "ROUND to a requested number of decimals",
    "distinct": "COUNT(DISTINCT ...) or DISTINCT rows",
    "null_handling": "COALESCE / IFNULL when there may be no rows",
    "anti_join": "NOT IN / NOT EXISTS / LEFT JOIN ... IS NULL to find missing matches",
    "set_ops": "EXCEPT / INTERSECT / UNION",
}
# The held-out eval always uses these, so scores stay comparable as templates are added.
BUILTIN = tuple(TEMPLATES)


def add_template(spec):
    """Register a generated template: plain data, never code. spec has name, tier, skills,
    question and sql patterns with {placeholders}, and params {placeholder: [values]}."""
    def fn(rng):
        values = {k: rng.choice(v) for k, v in spec["params"].items()}
        return spec["question"].format(**values), spec["sql"].format(**values)
    TEMPLATES[spec["name"]] = (spec["tier"], fn)
    SKILLS[spec["name"]] = tuple(spec["skills"])


@dataclass
class Task:
    template: str
    tier: int
    question: str
    gold_sql: str
    db_path: Path = DB_PATH
    split: str = "train"  # evaluation: heldout_template, reserved_params or shared_wording


EVAL_SEED = 12345  # training samples tasks with other seeds
EVAL_VERSION = "eval-v3"
# Never trained; their skills appear in other templates. The selection set drives checkpoint
# selection and early stopping. The check set is scored only at the start and on the
# selected and final checkpoints, so its score is not biased by selection.
HELDOUT_SELECT = ("signups_year", "orders_by_country", "best_category_year", "lines_per_order_status", "repeat_buyers_cat")
HELDOUT_CHECK = ("country_revenue_year", "idle_customers_country", "status_counts_year", "large_orders", "first_order_city")
HELDOUT_TEMPLATES = HELDOUT_SELECT + HELDOUT_CHECK


def _questions(name):
    fn = TEMPLATES[name][1]
    candidates = {}
    for seed in range(256):
        q, sql = fn(random.Random(EVAL_SEED + seed))
        candidates.setdefault(q, sql)
    return sorted(candidates.items())


@lru_cache(maxsize=None)
def reserved_questions(name):
    """Reserve parameters explicitly; constant templates use unseen evaluation data."""
    items = _questions(name)
    if name in HELDOUT_TEMPLATES:
        return tuple(random.Random(EVAL_SEED).sample(items, len(items)))
    if len(items) == 1:
        return tuple(items)
    return tuple(items[:min(2, max(1, len(items) // 3))])


@lru_cache(maxsize=None)
def eval_question_texts():
    return frozenset(q for name in BUILTIN for q, _ in
                     (_questions(name) if name in HELDOUT_TEMPLATES else reserved_questions(name)))


def eval_tasks(per_template, k, sets=("train", "select"), heldout_questions=4, heldout_k=None):
    """sets: "train" (reserved parameters and shared wording of trained templates; per_template
    questions, k attempts), "select" and "check" (held-out templates; heldout_questions
    questions, heldout_k attempts, default k)."""
    if per_template < 1 or k < 1 or heldout_questions < 1:
        raise ValueError("Evaluation counts must be positive")
    ensure_databases()
    groups = {"select": HELDOUT_SELECT, "check": HELDOUT_CHECK}
    tasks = []
    for name in BUILTIN:
        tier, _ = TEMPLATES[name]
        held = next((g for g, names in groups.items() if name in names), None)
        if held:
            if held not in sets:
                continue
            split, items, n = f"heldout_{held}", reserved_questions(name)[:heldout_questions], heldout_k or k
        elif "train" in sets:
            items, n = reserved_questions(name)[:per_template], k
            split = "shared_wording" if len(_questions(name)) == 1 else "reserved_params"
        else:
            continue
        for q, sql in items:
            tasks.extend([Task(name, tier, q, sql, EVAL_DB_PATH, split)] * n)
    return tasks


def sample_tasks(n, seed, tiers=(1, 2, 3), weights=None):
    rng = random.Random(seed)
    names = [k for k, (t, _) in TEMPLATES.items() if t in tiers and k not in HELDOUT_TEMPLATES]
    if not names:
        raise ValueError("No templates match the selected tiers")
    w = [weights[k] for k in names] if weights else None
    constant = {name for name in BUILTIN if len(_questions(name)) == 1}
    shared = {q for name in constant for q, _ in _questions(name)}
    tasks = []
    for _ in range(max(100, n * 100)):
        if len(tasks) == n:
            return tasks
        name = rng.choices(names, weights=w)[0]
        tier, fn = TEMPLATES[name]
        q, sql = fn(rng)
        # Constant templates share wording but use unseen evaluation data.
        if q in eval_question_texts() and q not in shared:
            continue
        tasks.append(Task(name, tier, q, sql))
    raise RuntimeError("Could not sample tasks outside reserved evaluation parameters")


def ensure_databases():
    """Create versioned fixtures without replacing databases used by old runs."""
    with _db_lock:
        paths = [DB_PATH, *ALT_DB_PATHS, EVAL_DB_PATH, *EVAL_ALT_DB_PATHS]
        for i, path in enumerate(paths):
            if not path.exists():
                tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
                role = i % 4  # 0 primary, 1 adversarial, 2 sales witness, 3 sparse
                build_db(tmp, seed=100 + i, sparse=role == 3, complete_catalog=role != 0,
                         orderable_parity={1: 1, 3: 0}.get(role), adversarial=role == 1)
                tmp.replace(path)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}


def _authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def connect(path=DB_PATH):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.set_authorizer(_authorizer)
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
    return con


def verifier_connections(db_path=DB_PATH):
    ensure_databases()
    alternates = EVAL_ALT_DB_PATHS if Path(db_path) == EVAL_DB_PATH else ALT_DB_PATHS
    return [connect(db_path)] + [connect(p) for p in alternates]


def grade(cons, sql, gold_sql):
    """Compare complete results on every fixture; invalid references are errors."""
    if not cons:
        raise ValueError("Verifier needs at least one database")
    for i, con in enumerate(cons):
        expected, gold_err = execute(con, gold_sql, deterministic=True)
        if gold_err:
            raise ValueError(f"Reference query failed on database {i}: {gold_err}")
        rows, err = execute(con, sql, require_table=True, deterministic=True)
        if err:
            return False, f"error on database {i}: {err}"
        if not results_match(rows, expected):
            return False, "wrong result" if i == 0 else f"wrong on alternate database {i}"
    return True, "matches on all databases"


# A correct correlated subquery over unindexed order_items needs about 18M steps; references
# need under 0.1M. The budget bounds cross joins without rejecting slow correct SQL.
MAX_STEPS = 50_000_000


def execute(con, sql, max_steps=MAX_STEPS, *, max_rows=MAX_RESULT_ROWS, require_table=False, deterministic=False):
    """Run one bounded read-only statement; overflow returns an explicit error."""
    budget, reads = [0], set()

    def authorize(action, table, column, *_):
        if deterministic and action == sqlite3.SQLITE_FUNCTION and column.lower() in {
            "random", "randomblob", "current_timestamp", "current_date", "current_time",
            "changes", "total_changes", "last_insert_rowid", "load_extension",
        }:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ and table not in ("sqlite_master", "sqlite_schema"):
            reads.add(table)
        return _authorizer(action)

    def progress():
        budget[0] += 1
        return int(budget[0] > max_steps // 1000)

    if deterministic and _CLOCK.search(sql):
        return None, "Date and time functions may not read the current time"
    con.set_authorizer(authorize)  # Also invalidates cached authorization decisions.
    con.set_progress_handler(progress, 1000)
    cur = None
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            return None, f"Result exceeds {max_rows} rows; narrow the query"
        if require_table and not reads:
            return None, "Submitted SQL must read a data table"
        return rows, None
    except sqlite3.Error as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        if cur is not None:
            cur.close()
        con.set_progress_handler(None, 1000)
        con.set_authorizer(_authorizer)


# Date functions read the clock through these modifiers; the authorizer only sees function names.
_CLOCK = re.compile(r"'\s*(now|localtime|utc)\s*'", re.I)


def tie_dependent(con, sql, tries=12):
    """True if a reference's result depends on how ORDER BY ... LIMIT breaks ties.
    Random secondary sort keys are added to every ORDER BY that feeds a LIMIT."""
    shuffled = re.sub(r"(ORDER\s+BY\s[^()]*?)(\s+LIMIT\s)", r"\1, random()\2", sql, flags=re.I)
    if shuffled == sql:
        return False
    first, err = execute(con, sql)
    if err:
        raise ValueError(f"Reference query failed: {err}")
    return any(not results_match(execute(con, shuffled)[0], first) for _ in range(tries))


def format_rows(rows, error, max_rows=10, max_cell=60):
    if error:
        return f"Error: {error}"
    if not rows:
        return "(0 rows)"
    lines = [" | ".join(str(v)[:max_cell] for v in row) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows)} rows total)")
    return "\n".join(lines)


def _norm(v):
    if isinstance(v, (int, float)):
        if not math.isfinite(v):
            raise ValueError("Nonfinite SQL result")
        # Keep integers exact, including values above IEEE float precision.
        return v if isinstance(v, int) else round(v, 6)
    return v


def results_match(a, b):
    """Compare row multisets, preserving column positions, types and duplicates."""
    if a is None or b is None:
        return False
    try:
        key = lambda row: tuple(_norm(v) for v in row)
        return Counter(map(key, a)) == Counter(map(key, b))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Agent protocol: explore with run_sql, then finish with submit_sql. The complete
# result of the submitted query is the user answer. Preview rows and prose are
# never substituted for that result.
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A single read-only SQLite query."}
        }, "required": ["query"], "additionalProperties": False}
    }} for name, description in (
        ("run_sql", "Explore the database. Shows up to 10 preview rows; the query may return more."),
        ("submit_sql", "Submit your final SQL answer and finish. Its complete result is returned to the user. "
         "Return exactly the requested columns, in the requested order. No prose answer is needed."),
    )
]


def system_prompt(show_schema=True):
    p = ("You are a SQL agent with access to a SQLite database of an online shop. "
         "Use run_sql to explore if needed. To answer, call submit_sql with the final query. "
         "submit_sql ends the task and returns its complete result to the user. "
         "Use exactly one function call per turn. A plain-text reply does not submit an answer. "
         "The final query must return exactly the columns requested, in that order, "
         "and work on databases with different data. Do not hard-code result values.")
    if show_schema:
        p += "\n\nDatabase schema:\n" + SCHEMA
    else:
        p += "\n\nDiscover the schema with run_sql (e.g. SELECT sql FROM sqlite_master)."
    return p


def is_literal(sql):
    """A final query that reads no table, e.g. SELECT 11: the agent hard-coded an
    answer it found with an earlier query. Track it separately from correctness."""
    return not re.search(r"\bFROM\b", sql, re.I)


_CALL = re.compile(r'<function name="([^"]+)">(.*?)</function>', re.S)
_PARAM = re.compile(r'<param name="([^"]+)">(.*?)</param>', re.S)


def prose(text):
    """The part of an assistant turn that is not a function call: apologies, narration, the final reply."""
    return _CALL.sub("", text).strip()


def parse_call(text):
    """Read one complete call. Reject malformed, multiple or ambiguous calls."""
    matches = list(_CALL.finditer(text))
    if not matches:
        if "<function" in text or "<param" in text:
            raise ValueError("Malformed function call")
        return None, None
    if len(matches) != 1 or "<function" in _CALL.sub("", text):
        raise ValueError("Use exactly one function call per turn")
    match = matches[0]
    params = _PARAM.findall(match.group(2))
    if len(params) != 1 or params[0][0] != "query" or _PARAM.sub("", match.group(2)).strip():
        raise ValueError("Provide exactly one query parameter")
    query = params[0][1].strip()
    if query.startswith("<![CDATA[") and query.endswith("]]>"):
        query = query[9:-3]
    if not query.strip():
        raise ValueError("The query parameter is empty")
    return match.group(1), query.strip()


if __name__ == "__main__":
    ensure_databases()
    print(f"{VERIFIER_VERSION}: databases ready in {DATA_DIR}")
