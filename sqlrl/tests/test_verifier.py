import random
import sqlite3
from contextlib import ExitStack, closing

import pytest

import env


@pytest.fixture
def cons():
    with ExitStack() as stack:
        yield [stack.enter_context(closing(c)) for c in env.verifier_connections()]


def test_compare_preserves_columns_duplicates_and_whitespace():
    assert env.results_match([('Books', 1), ('Toys', 2)], [('Toys', 2.0), ('Books', 1.0)])
    assert not env.results_match([('Books', 1), (2, 'Toys')], [('Books', 1), ('Toys', 2)])
    assert not env.results_match([('Books',)], [('Books',), ('Books',)])
    assert not env.results_match([(' Books',)], [('Books',)])
    assert not env.results_match([(2**53,)], [(2**53 + 1,)])
    assert not env.results_match([(float('inf'),)], [(float('inf'),)])
    assert not env.results_match([(148.7925,)], [(148.79,)])
    assert env.results_match([], [])
    assert not env.results_match([(None,)], [])


def test_full_result_not_first_thousand(cons):
    assert env.grade(cons, 'SELECT id FROM orders WHERE id <= 1000 ORDER BY id',
                     'SELECT id FROM orders ORDER BY id')[0] is False
    rows, err = env.execute(cons[0], 'SELECT id FROM orders ORDER BY id')
    assert err is None and len(rows) == 1200
    rows, err = env.execute(cons[0], 'SELECT id FROM orders', max_rows=100)
    assert rows is None and 'exceeds 100 rows' in err


def test_historical_price_shortcut_rejected(cons):
    tail = "FROM order_items oi JOIN orders o ON o.id=oi.order_id WHERE o.status='shipped'"
    gold = 'SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) ' + tail
    wrong = "SELECT ROUND(SUM(oi.quantity * p.price), 2) FROM order_items oi JOIN orders o ON o.id=oi.order_id JOIN products p ON p.id=oi.product_id WHERE o.status='shipped'"
    assert not env.grade(cons, wrong, gold)[0]
    assert env.grade(cons, gold, gold)[0]


def test_constants_and_invalid_references(cons):
    assert not env.grade(cons, 'SELECT 1200', 'SELECT COUNT(*) FROM orders')[0]
    assert not env.grade(cons, 'SELECT 1200 FROM products LIMIT 1', 'SELECT COUNT(*) FROM orders')[0]
    with pytest.raises(ValueError, match='Reference query failed'):
        env.grade(cons, 'SELECT id FROM products', 'SELECT missing FROM products')
    with pytest.raises(ValueError, match='at least one'):
        env.grade([], 'SELECT 1', 'SELECT 1')


def test_read_only_and_budget(cons):
    for query in ['DELETE FROM orders', "ATTACH ':memory:' AS other", 'SELECT 1; SELECT 2']:
        assert env.execute(cons[0], query)[1]
    query = 'SELECT SUM(a.id*b.id*c.id) FROM orders a CROSS JOIN orders b CROSS JOIN orders c'
    assert 'interrupted' in env.execute(cons[0], query, max_steps=1000)[1]
    # Progress handlers and authorization are reset after a failure.
    assert env.execute(cons[0], 'SELECT COUNT(*) FROM orders') == ([(1200,)], None)


def test_correct_alternatives_empty_answers_and_duplicate_names(cons):
    gold = 'SELECT name FROM products WHERE id NOT IN (SELECT product_id FROM order_items)'
    alternate = 'SELECT p.name FROM products p WHERE NOT EXISTS (SELECT 1 FROM order_items oi WHERE oi.product_id=p.id)'
    assert env.grade(cons, alternate, gold)[0]
    gold = 'SELECT name FROM products WHERE price < 0'
    assert env.grade(cons, 'SELECT name FROM products WHERE price <= -1', gold)[0]
    gold = 'SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON c.id=o.customer_id GROUP BY c.id'
    wrong = 'SELECT c.name, COUNT(*) FROM customers c JOIN orders o ON c.id=o.customer_id GROUP BY c.name'
    assert not env.grade(cons, wrong, gold)[0]


def test_eval_split_and_reference_coverage(cons):
    tasks = env.eval_tasks(2, 4)
    assert all(t.db_path == env.EVAL_DB_PATH for t in tasks)
    unique = {(t.template, t.question) for t in tasks}
    assert len(tasks) == 4 * len(unique)
    trained = env.sample_tasks(1000, 200)
    eval_questions = {t.question for t in tasks}
    constants = {name for name in env.BUILTIN if len({env.TEMPLATES[name][1](random.Random(i))[0] for i in range(256)}) == 1}
    assert all(t.question not in eval_questions or t.template in constants for t in trained)
    for name, (_, fn) in env.TEMPLATES.items():
        for seed in range(6):
            _, gold = fn(random.Random(seed))
            assert env.grade(cons, gold, gold)[0], (name, gold)


def test_strict_tool_calls():
    call = '<function name="submit_sql"><param name="query"><![CDATA[SELECT name FROM products WHERE price < 10]]></param></function>'
    assert env.parse_call(call) == ('submit_sql', 'SELECT name FROM products WHERE price < 10')
    assert env.parse_call('Done.') == (None, None)
    for bad in [call + call, '<function name="run_sql">',
                '<function name="run_sql"><param name="query">SELECT 1</param><param name="query">SELECT 2</param></function>',
                '<function name="run_sql"><param name="query"></param></function>']:
        with pytest.raises(ValueError):
            env.parse_call(bad)


def test_generated_product_names_have_an_ordered_witness(cons):
    witness = cons[2]
    names = [f'{adj} {noun}' for adj in env.ADJ for nouns in env.NOUN.values() for noun in nouns]
    for name in names:
        rows, err = env.execute(witness, 'SELECT COUNT(*), SUM(oi.quantity) FROM order_items oi JOIN products p ON p.id=oi.product_id WHERE p.name=' + repr(name))
        assert err is None and rows[0][1] > rows[0][0] > 0


def test_nondeterministic_submissions_and_references(cons):
    assert not env.grade(cons, 'SELECT random() FROM products LIMIT 1', 'SELECT id FROM products LIMIT 1')[0]
    with pytest.raises(ValueError, match='Reference query failed'):
        env.grade(cons, 'SELECT id FROM products', 'SELECT random() FROM products')
