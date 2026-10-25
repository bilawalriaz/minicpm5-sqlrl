from types import SimpleNamespace

import evolve


def spec():
    return {'name': 'city_counter_test', 'tier': 1, 'skills': ['filter', 'aggregate'],
            'target_skill': 'filter', 'params': {'city': ['London', 'Paris', 'Tokyo', 'Seattle']},
            'question': 'How many customers live in {city}?',
            'sql': "SELECT COUNT(*) FROM customers WHERE city='{city}'"}


def test_template_validates_distinct_questions_and_all_parameters():
    seen = []
    teacher = SimpleNamespace(solve=lambda task: seen.append(task.question) or {'correct': True})
    good = spec()
    ok, reason = evolve.validate(good, teacher, set())
    assert ok and 'all 4 combinations' in reason
    assert len(seen) == len(set(seen)) == 3
    # A broken reference is rejected before the teacher supplies training traces.
    good['sql'] = "SELECT missing FROM customers WHERE city='{city}'"
    ok, reason = evolve.validate(good, teacher, set())
    assert not ok and 'reference SQL fails' in reason


def test_malformed_template_and_parameter_explosion():
    bad = spec()
    bad['question'] = '{city'
    assert evolve.check_structure(bad) == 'malformed placeholders'
    bad = spec()
    bad['params']['city'] = [float('inf'), 'Paris']
    assert 'not plain' in evolve.check_structure(bad)


def test_template_validation_rejects_free_reward_and_eval_duplicates():
    teacher = SimpleNamespace(solve=lambda task: {'correct': True})
    cases = {
        "SELECT COUNT(*) FROM customers WHERE city='{city}' AND id < 0": 'same answer on every database',
        "SELECT name FROM customers WHERE city='{city}' AND id < 0": 'empty or NULL answer',
        "SELECT name FROM customers WHERE city='{city}' ORDER BY country LIMIT 1": 'tie order',
    }
    for sql, reason in cases.items():
        bad = spec()
        bad['sql'] = sql
        ok, why = evolve.validate(bad, teacher, set())
        assert not ok and reason in why, why
    # The same question as a held-out evaluation template, in other words.
    dup = spec()
    dup.update(question='Count orders from customers living in {country}.', params={'country': ['UK', 'France']},
               sql="SELECT COUNT(o.id) FROM customers c, orders o WHERE o.customer_id = c.id AND c.country = '{country}'")
    ok, why = evolve.validate(dup, teacher, set())
    assert not ok and 'evaluation question' in why, why


def test_clause_that_never_changes_the_answer_is_rejected():
    teacher = SimpleNamespace(solve=lambda task: {'correct': True})
    vacuous = spec()
    vacuous.update(question='Which categories have revenue above {amount}? Return the category and the revenue.',
                   params={'amount': [1, 2]}, skills=['having', 'group_by'], target_skill='having',
                   sql='SELECT p.category, SUM(oi.quantity * oi.unit_price) FROM products p JOIN order_items oi '
                       'ON oi.product_id = p.id GROUP BY p.category HAVING SUM(oi.quantity * oi.unit_price) > {amount}')
    ok, why = evolve.validate(vacuous, teacher, set())
    assert not ok and 'removing HAVING' in why, why


def test_builtin_references_have_no_vacuous_clause():
    import random
    from contextlib import ExitStack, closing
    import env
    with ExitStack() as stack:
        cons = [stack.enter_context(closing(c)) for c in env.verifier_connections()]
        for name in env.BUILTIN:
            pairs = sorted({env.TEMPLATES[name][1](random.Random(s)) for s in range(200)})
            effective = set()
            for _, sql in pairs:
                base = [evolve.signature(env.execute(c, sql)[0]) for c in cons]
                for i, variant in enumerate(evolve.ablations(sql).values()):
                    rows = [env.execute(c, variant)[0] for c in cons]
                    if any(r is None or evolve.signature(r) != b for r, b in zip(rows, base)):
                        effective.add(i)
            assert effective == set(range(len(evolve.ablations(pairs[0][1])))), name
